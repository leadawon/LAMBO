"""Search Agent — per-document agentic <think>/<search>/<info> loop.

Given a document's anchor map, the agent repeatedly picks ONE anchor to open,
receives its raw text as an <info> block, has the Extractor pull atomic facts
from it, and then decides whether enough local evidence has been collected or
whether another anchor should be opened.  No level / type / topology hints are
exposed to the LLM; the only inputs are the user query, the document title,
the anchor map, and the running evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend import QwenLocalClient
from .common import (
    compact_text,
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    read_json,
    write_json,
)
from .anchor_relations import provenance_strength_from_anchor, relation_context


def _format_doc_map(doc_map: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in doc_map:
        entities = ", ".join(entry.get("key_entities") or [])
        role_hint = ", ".join(entry.get("anchor_role_candidates") or [])
        target_hint = ", ".join(entry.get("anchor_targets") or [])
        relation_summary = entry.get("relation_summary", {}) or {}
        related = ", ".join(relation_summary.get("top_related_anchor_ids") or [])
        summary = compact_text(entry.get("summary", ""), limit=320)
        lines.append(
            f"- {entry['anchor_id']} | {entry.get('anchor_title','')}"
            + (f"\n    key_entities: {entities}" if entities else "")
            + (f"\n    roles: {role_hint}" if role_hint else "")
            + (f"\n    targets: {target_hint}" if target_hint else "")
            + (f"\n    related: {related}" if related else "")
            + (f"\n    summary: {summary}" if summary else "")
        )
    return "\n".join(lines) if lines else "(empty)"


def _format_evidence(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "(none yet)"
    rows: List[str] = []
    for i, item in enumerate(items, start=1):
        src = item.get("source_anchor_id", "")
        fact = compact_text(item.get("fact", ""), limit=220)
        ents = ", ".join(item.get("entities") or [])
        provenance = normalize_ws(item.get("provenance_strength", ""))
        rows.append(
            f"{i}. [{src}] {fact}"
            + (f"  (entities: {ents})" if ents else "")
            + (f"  (provenance: {provenance})" if provenance else "")
        )
    return "\n".join(rows)


def _relation_frontier_ids(
    anchors_by_id: Dict[str, Dict[str, Any]],
    opened_ids: List[str],
) -> List[str]:
    opened = set(opened_ids)
    frontier: List[str] = []
    preferred_types = {
        "supports",
        "disambiguates",
        "prerequisite_of",
        "same_company_metric",
        "same_case",
        "same_target_paper",
        "likely_same_entity",
        "same_section",
        "nearby_window",
    }
    for source_id in opened_ids[-4:]:
        anchor = anchors_by_id.get(source_id)
        if not anchor:
            continue
        for relation in anchor.get("anchor_relations", []) or []:
            relation_type = normalize_ws(relation.get("relation_type", ""))
            target_id = normalize_ws(relation.get("target_anchor_id", ""))
            if target_id and target_id not in opened and relation_type in preferred_types and target_id not in frontier:
                frontier.append(target_id)
    return frontier


class ExtractAgent:
    """Extracts atomic facts from a single opened anchor."""

    def __init__(self, llm: QwenLocalClient, prompt_dir: Path) -> None:
        self.llm = llm
        self.system_prompt = (prompt_dir / "extract_system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (prompt_dir / "extract_user.txt").read_text(encoding="utf-8").strip()

    def run(
        self,
        *,
        query: str,
        instruction: str,
        doc_title: str,
        anchor: Dict[str, Any],
        need: str,
        evidence_so_far: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        user_prompt = self.user_template.format(
            query=query or "(empty)",
            instruction=instruction or "(none)",
            doc_title=doc_title,
            anchor_id=anchor["anchor_id"],
            anchor_title=anchor.get("anchor_title", ""),
            need=need or "(none)",
            evidence_so_far=_format_evidence(evidence_so_far),
            anchor_text=anchor.get("text", ""),
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1200,
            metadata={"module": "extract_agent", "anchor_id": anchor["anchor_id"]},
        )
        think_text = normalize_ws(extract_tag_content(raw_text, "think") or "")
        info_text = extract_tag_content(raw_text, "info") or ""
        payload = extract_json_payload(info_text) or {}
        raw_items = payload.get("items", []) if isinstance(payload, dict) else []
        items: List[Dict[str, Any]] = []
        if isinstance(raw_items, list):
            for it in raw_items:
                if not isinstance(it, dict):
                    continue
                fact = normalize_ws(it.get("fact", ""))
                if not fact:
                    continue
                items.append(
                    {
                        "fact": fact,
                        "entities": [normalize_ws(e) for e in (it.get("entities") or []) if normalize_ws(e)],
                        "evidence_span": normalize_ws(it.get("evidence_span", "")),
                        "relevance": normalize_ws(it.get("relevance", "")),
                        "source_doc_id": anchor["doc_id"],
                        "source_anchor_id": anchor["anchor_id"],
                    }
                )
                role_hint = ",".join(anchor.get("anchor_role_candidates", [])[:4])
                if role_hint:
                    items[-1]["source_anchor_role"] = role_hint
                source_relations = relation_context(anchor, max_items=6)
                if source_relations:
                    items[-1]["source_relation_context"] = source_relations
                strength, disambiguation_needed = provenance_strength_from_anchor(anchor)
                items[-1]["provenance_strength"] = strength
                items[-1]["disambiguation_needed"] = disambiguation_needed
        return {
            "think": think_text,
            "items": items,
            "raw_text": raw_text,
        }


class SearchAgent:
    def __init__(
        self,
        llm: QwenLocalClient,
        extract_agent: ExtractAgent,
        prompt_dir: Optional[Path] = None,
        max_rounds: int = 5,
    ) -> None:
        self.llm = llm
        self.extract_agent = extract_agent
        self.max_rounds = max_rounds
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.system_prompt = (self.prompt_dir / "search_system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (self.prompt_dir / "search_user.txt").read_text(encoding="utf-8").strip()

    # ------------------------------------------------------------------
    def _plan_step(
        self,
        *,
        query: str,
        instruction: str,
        doc_title: str,
        doc_map: List[Dict[str, Any]],
        opened_ids: List[str],
        evidence: List[Dict[str, Any]],
        round_index: int,
    ) -> Dict[str, Any]:
        user_prompt = self.user_template.format(
            query=query or "(empty)",
            instruction=instruction or "(none)",
            doc_title=doc_title,
            doc_map=_format_doc_map(doc_map),
            opened_anchors=", ".join(opened_ids) if opened_ids else "(none)",
            evidence_so_far=_format_evidence(evidence),
            round_index=round_index,
            max_rounds=self.max_rounds,
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=700,
            metadata={"module": "search_agent", "doc_title": doc_title, "round": round_index},
        )
        think_text = normalize_ws(extract_tag_content(raw_text, "think") or "")
        search_text = extract_tag_content(raw_text, "search") or ""
        payload = extract_json_payload(search_text) or {}
        action = normalize_ws(str(payload.get("action", ""))).lower() if isinstance(payload, dict) else ""
        anchor_id = normalize_ws(str(payload.get("anchor_id", ""))) if isinstance(payload, dict) else ""
        need = normalize_ws(str(payload.get("need", ""))) if isinstance(payload, dict) else ""
        reason = normalize_ws(str(payload.get("reason", ""))) if isinstance(payload, dict) else ""
        return {
            "think": think_text,
            "action": action or ("open" if anchor_id else "stop"),
            "anchor_id": anchor_id,
            "need": need,
            "reason": reason,
            "raw_text": raw_text,
        }

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        query: str,
        instruction: str,
        doc_payload: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / f"{doc_payload['doc_id']}_search.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        anchors_by_id: Dict[str, Dict[str, Any]] = {a["anchor_id"]: a for a in doc_payload.get("anchors", [])}
        doc_map = doc_payload.get("doc_map") or []

        opened_ids: List[str] = []
        evidence: List[Dict[str, Any]] = []
        trace_lines: List[str] = []
        rounds: List[Dict[str, Any]] = []

        for round_index in range(1, self.max_rounds + 1):
            remaining_ids = [aid for aid in anchors_by_id.keys() if aid not in opened_ids]
            if not remaining_ids:
                break
            relation_frontier_ids = _relation_frontier_ids(anchors_by_id, opened_ids)
            ordered_remaining_ids = sorted(
                remaining_ids,
                key=lambda aid: (aid in relation_frontier_ids, -remaining_ids.index(aid)),
                reverse=True,
            )
            order_index = {anchor_id: idx for idx, anchor_id in enumerate(ordered_remaining_ids)}
            round_doc_map = sorted(
                [m for m in doc_map if m["anchor_id"] in order_index],
                key=lambda m: order_index[m["anchor_id"]],
            )

            plan = self._plan_step(
                query=query,
                instruction=instruction,
                doc_title=doc_payload["doc_title"],
                doc_map=round_doc_map,
                opened_ids=opened_ids,
                evidence=evidence,
                round_index=round_index,
            )
            plan["relation_frontier_ids"] = relation_frontier_ids

            action = plan["action"]
            anchor_id = plan["anchor_id"]
            if action == "stop" and evidence:
                trace_lines.append(f"<think>{plan['think']}</think>")
                trace_lines.append(f"<search>{json.dumps({'action':'stop','reason':plan['reason']}, ensure_ascii=False)}</search>")
                rounds.append(
                    {
                        "round_index": round_index,
                        "plan": plan,
                        "anchor_id": "",
                        "info": None,
                        "extract": None,
                    }
                )
                break

            if anchor_id not in anchors_by_id or anchor_id in opened_ids:
                # graceful fallback: first unopened anchor
                anchor_id = ordered_remaining_ids[0]
                plan["anchor_id"] = anchor_id
                plan["need"] = plan.get("need") or "fallback: inspect next unopened anchor"

            anchor = anchors_by_id[anchor_id]
            opened_ids.append(anchor_id)

            info_payload = {
                "anchor_id": anchor_id,
                "anchor_title": anchor.get("anchor_title", ""),
                "text": anchor.get("text", ""),
            }
            trace_lines.append(f"<think>{plan['think']}</think>")
            trace_lines.append(
                f"<search>{json.dumps({'action':'open','anchor_id':anchor_id,'need':plan['need'],'relation_frontier_ids':relation_frontier_ids}, ensure_ascii=False)}</search>"
            )
            trace_lines.append(f"<info>{json.dumps({'anchor_id':anchor_id,'anchor_title':info_payload['anchor_title']}, ensure_ascii=False)}</info>")

            extraction = self.extract_agent.run(
                query=query,
                instruction=instruction,
                doc_title=doc_payload["doc_title"],
                anchor=anchor,
                need=plan["need"],
                evidence_so_far=evidence,
            )
            evidence.extend(extraction["items"])
            trace_lines.append(f"<think>{extraction['think']}</think>")
            trace_lines.append(
                "<extracted>"
                + json.dumps({"items": extraction["items"]}, ensure_ascii=False)
                + "</extracted>"
            )

            rounds.append(
                {
                    "round_index": round_index,
                    "plan": plan,
                    "anchor_id": anchor_id,
                    "info": {"anchor_title": info_payload["anchor_title"], "char_count": len(info_payload["text"])},
                    "extract": {"think": extraction["think"], "item_count": len(extraction["items"])},
                }
            )

        payload_out = {
            "doc_id": doc_payload["doc_id"],
            "doc_title": doc_payload["doc_title"],
            "opened_anchor_ids": opened_ids,
            "items": evidence,
            "rounds": rounds,
            "tagged_trace": "\n".join(trace_lines),
        }
        write_json(cache_path, payload_out)
        return payload_out
