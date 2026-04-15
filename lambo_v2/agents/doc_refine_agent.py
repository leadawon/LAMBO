"""DocRefineAgent — per-document evidence extraction via a single think→search→info loop.

For each document, the agent runs one LLM call per round:
  1. LLM sees the question, instruction, document map (anchor summaries in document
     order), and the full accumulated trace from prior rounds.
  2. LLM emits <think>...</think> then <search>{"action":"open","anchor_id":"..."} or
     {"action":"stop","reason":"..."}</search>.
  3. If open: the harness injects <info anchor_id="...">raw text</info> into the
     accumulated trace, and the next round begins.
  4. If stop: the LLM also emits <answer>...</answer> containing verbatim gold
     evidence extracted from the opened anchors. Empty <answer></answer> is valid
     when the document is genuinely irrelevant.

The LLM maintains its own reflection loop — it sees every prior <think>, <search>,
and <info> and decides each round whether to open another anchor or stop. There is
no separate Planner/Extractor split, and no query plan is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend import QwenLocalClient, GeminiClient, OpenAIClient
from ..common import (
    compact_text,
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    read_json,
    write_json,
)


class DocRefineAgent:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
        max_rounds: int = 6,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        pdir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "doc_refine"
        self.system_prompt = (pdir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (pdir / "user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _format_doc_map(anchors: List[Dict[str, Any]]) -> str:
        """Format all anchor summaries for the LLM in original document order."""
        lines: List[str] = []
        for anchor in anchors:
            lines.append(
                f"[{anchor['anchor_id']}] {anchor.get('anchor_title', '')} "
                f"| {compact_text(anchor.get('summary', ''), limit=200)}"
            )
        return "\n".join(lines) if lines else "(no anchors available)"

    def _call_llm(
        self,
        *,
        question: str,
        instruction: str,
        doc_id: str,
        doc_title: str,
        doc_map_text: str,
        accumulated_trace: str,
    ) -> str:
        """Single LLM call per round: emit <think> and <search>, optionally <answer>."""
        trace_section = ""
        if accumulated_trace.strip():
            trace_section = f"Previous rounds:\n{accumulated_trace}"
        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title=doc_title,
            doc_id=doc_id,
            doc_map=doc_map_text,
            accumulated_trace=trace_section,
        )
        return self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1500,
            metadata={"module": "doc_refine", "phase": "think_search", "doc_id": doc_id},
        )

    def run(
        self,
        *,
        question: str,
        instruction: str,
        doc_payload: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        doc_id = doc_payload["doc_id"]
        doc_title = doc_payload["doc_title"]
        cache_path = sample_dir / f"{doc_id}_refine.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        all_anchors = doc_payload.get("anchors", [])
        doc_map_text = self._format_doc_map(all_anchors)

        anchor_by_id = {a["anchor_id"]: a for a in all_anchors}
        opened_anchors: List[str] = []
        accumulated_trace = ""

        # Defaults — overwritten when LLM stops with <answer>
        scan_result = "no_evidence"
        evidence_text = ""

        for round_idx in range(1, self.max_rounds + 1):
            raw = self._call_llm(
                question=question,
                instruction=instruction,
                doc_id=doc_id,
                doc_title=doc_title,
                doc_map_text=doc_map_text,
                accumulated_trace=accumulated_trace,
            )

            think_text = extract_tag_content(raw, "think") or ""
            search_text = extract_tag_content(raw, "search") or ""
            answer_text = extract_tag_content(raw, "answer") or ""
            search_payload = extract_json_payload(search_text)

            accumulated_trace += f"\n--- Round {round_idx} ---\n"
            accumulated_trace += f"<think>{think_text}</think>\n"
            accumulated_trace += f"<search>{search_text}</search>\n"

            action = "stop"
            anchor_id = ""
            if isinstance(search_payload, dict):
                action = str(search_payload.get("action", "stop")).lower()
                anchor_id = normalize_ws(str(search_payload.get("anchor_id", "")))

            if action == "stop":
                # Zero-open safety rail: the LLM must see at least one anchor's raw
                # text before it can decide whether the document is irrelevant. Anchor
                # summaries alone are not enough — the LLM would be guessing from the
                # map. On round 1 with no anchors opened, force-open the first anchor.
                if not opened_anchors and all_anchors:
                    anchor_id = all_anchors[0]["anchor_id"]
                    accumulated_trace += (
                        "<note>Stop overridden: must open at least one anchor before "
                        "judging irrelevance. Forcing open of first anchor.</note>\n"
                    )
                    action = "open"
                else:
                    if answer_text:
                        accumulated_trace += f"<answer>{answer_text}</answer>\n"
                        evidence_text = answer_text.strip()
                        scan_result = "evidence_found" if evidence_text else "no_evidence"
                    break

            if not anchor_id:
                if not opened_anchors and all_anchors:
                    anchor_id = all_anchors[0]["anchor_id"]
                else:
                    break

            # Validate anchor_id; fall back to the first unopened anchor when invalid/duplicate
            if anchor_id not in anchor_by_id or anchor_id in opened_anchors:
                fallback_id = ""
                for anc in all_anchors:
                    if anc["anchor_id"] not in opened_anchors:
                        fallback_id = anc["anchor_id"]
                        break
                if not fallback_id:
                    break
                anchor_id = fallback_id

            opened_anchors.append(anchor_id)
            anchor = anchor_by_id[anchor_id]
            anchor_text = anchor.get("text", "")

            # Inject raw anchor text into trace — the LLM sees this on the next round
            accumulated_trace += f"<info anchor_id=\"{anchor_id}\">\n{anchor_text}\n</info>\n"

        evidence_sheet = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "scan_result": scan_result,
            "evidence": evidence_text,
            "opened_anchors": opened_anchors,
            "rounds_used": len(opened_anchors),
            "trace": accumulated_trace,
        }
        write_json(cache_path, evidence_sheet)
        return evidence_sheet
