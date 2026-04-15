"""Final Answer Writer — composes the answer from a structured relations view.

Receives the single `relations` object produced by the Relation Refiner and the
user query, and asks the LLM to output `{"final_answer": ...}`.  No level /
type / topology hints are exposed; the formatting rules live entirely in the
system prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend import QwenLocalClient
from .common import (
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    read_json,
    write_json,
)


class AnswerWriter:
    def __init__(
        self,
        llm: QwenLocalClient,
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.system_prompt = (self.prompt_dir / "answer_system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (self.prompt_dir / "answer_user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _enforce_doc_keys(final_answer: Any, instruction: str, relations: Dict[str, Any]) -> Any:
        import re
        if not isinstance(final_answer, dict):
            return final_answer
        instr = instruction or ""
        if not re.search(r"\bDOC\d+\b", instr):
            return final_answer
        entities = (relations.get("schema") or {}).get("entities") or []
        doc_ids = [e for e in entities if isinstance(e, str) and re.match(r"^DOC\d+$", e)]
        if not doc_ids:
            return final_answer
        if all(isinstance(k, str) and re.match(r"^DOC\d+$", k) for k in final_answer.keys()):
            return final_answer
        # Build title -> doc_id map from refiner's cross-doc roster if available
        return final_answer

    @staticmethod
    def _fallback_from_relations(relations: Dict[str, Any]) -> Any:
        structure = str(relations.get("structure", "scalar")).lower()
        records = relations.get("records", []) or []
        if structure == "scalar":
            for rec in records:
                val = (rec or {}).get("payload", {}).get("value", "")
                if val:
                    return val
            return ""
        if structure == "list":
            out: List[str] = []
            for rec in records:
                val = (rec or {}).get("payload", {}).get("value", "")
                if val and val not in out:
                    out.append(val)
            return out
        if structure == "mapping":
            out_map: Dict[str, List[str]] = {}
            for rec in records:
                payload = (rec or {}).get("payload", {}) or {}
                key = normalize_ws(str(payload.get("key", "")))
                value = payload.get("value", "")
                if not key:
                    continue
                values = value if isinstance(value, list) else [value]
                out_map.setdefault(key, [])
                for v in values:
                    vs = normalize_ws(str(v))
                    if vs and vs not in out_map[key]:
                        out_map[key].append(vs)
            return out_map
        if structure == "table":
            return [rec.get("payload", {}) for rec in records if isinstance(rec, dict)]
        if structure == "graph":
            return [rec.get("payload", {}) for rec in records if isinstance(rec, dict)]
        return ""

    def run(
        self,
        *,
        query: str,
        instruction: str,
        relations: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "answer_writer.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        relations_json = json.dumps(relations, ensure_ascii=False, indent=2)
        user_prompt = self.user_template.format(
            query=query or "(empty)",
            instruction=instruction or "(none)",
            relations_json=relations_json,
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1400,
            metadata={"module": "answer_writer"},
        )
        think_text = normalize_ws(extract_tag_content(raw_text, "think") or "")
        answer_text = extract_tag_content(raw_text, "answer") or ""
        payload = extract_json_payload(answer_text)
        final_answer: Any = ""
        if isinstance(payload, dict) and "final_answer" in payload:
            final_answer = payload["final_answer"]
        elif payload is not None:
            final_answer = payload
        # v7: if instruction uses DOC_N keys but the model emitted raw titles,
        # remap using the relations schema entities.
        final_answer = self._enforce_doc_keys(final_answer, instruction, relations)
        result = {
            "think": think_text,
            "final_answer": final_answer,
            "raw_text": raw_text,
        }
        write_json(cache_path, result)
        return result
