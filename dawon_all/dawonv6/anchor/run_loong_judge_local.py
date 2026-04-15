from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from dawonv6.anchor.backend import QwenLocalClient
from dawonv6.anchor.common import ensure_dir, read_jsonl, write_json, write_jsonl


DAWON_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = DAWON_ROOT / "logs" / "lambo_agentic_set1_10"


JUDGE_PROMPT_TEMPLATE = """[Question]
{question}

[Gold Answer]
{gold_answer}

[The Start of Assistant's Predicted Answer]
{prediction}
[The End of Assistant's Predicted Answer]

[System]
We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above according to the gold answer. Please use the following listed aspects and their descriptions as evaluation criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the gold answer; The numerical value and order need to be accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer should contain all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. Please note that the gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's answer and the gold answer fully meet the above criteria, its overall rating should be the full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Loong judge prompts with local Qwen.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_RUN_DIR / "lambo_predictions.jsonl",
    )
    parser.add_argument(
        "--evaluate-output",
        type=Path,
        default=DEFAULT_RUN_DIR / "judge" / "loong_judge_eval_local_qwen32b.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_RUN_DIR / "reports" / "loong_judge_local_qwen32b.json",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def build_question_text(row: Dict[str, Any]) -> str:
    prompt_template = str(row.get("prompt_template", ""))
    question = str(row.get("question", ""))
    instruction = str(row.get("instruction", ""))
    if row.get("type") != "paper":
        prompt_template = prompt_template.replace("{docs}", "")
    return prompt_template.replace("{question}", question).replace("{instruction}", instruction)


def build_prompt(row: Dict[str, Any]) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=build_question_text(row),
        gold_answer=row.get("answer", ""),
        prediction=row.get("generate_response", ""),
    )


def extract_score(text: str) -> float | None:
    match = re.search(r"\[\[([0-9]*\.?[0-9]+)\]\]", str(text or ""))
    if match:
        return float(match.group(1))
    match = re.search(r"\[([0-9]*\.?[0-9]+)\]", str(text or ""))
    if match:
        return float(match.group(1))
    return None


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    ensure_dir(args.evaluate_output.parent)
    ensure_dir(args.summary_output.parent)

    client = QwenLocalClient(max_output_tokens=args.max_output_tokens)

    eval_rows: List[Dict[str, Any]] = []
    scores: List[float] = []

    for index, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        response = client.generate_text(
            system_prompt="You are a strict evaluation judge. Follow the output format exactly.",
            user_prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            metadata={
                "stage": "loong_judge_local",
                "sample_id": row.get("sample_id", ""),
                "selected_index": row.get("selected_index", ""),
                "judge_item_index": index,
            },
        )

        result = dict(row)
        result["eval_response"] = response
        eval_rows.append(result)

        score = extract_score(response)
        if score is not None:
            scores.append(score)
        print(
            json.dumps(
                {
                    "index": index,
                    "selected_index": row.get("selected_index"),
                    "sample_id": row.get("sample_id"),
                    "score": score,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_jsonl(args.evaluate_output, eval_rows)

    summary = {
        "judge_model": "Qwen2.5-32B-Instruct(local)",
        "prompt_family": "Loong judge prompt",
        "sample_count": len(rows),
        "scoring_success_rate": (len(scores) / len(rows)) if rows else 0.0,
        "avg_score": mean(scores) if scores else 0.0,
        "perfect_rate_calculation": f"{sum(1 for value in scores if value == 100)}/{len(scores)}" if scores else "0/0",
        "perfect_rate": (sum(1 for value in scores if value == 100) / len(scores)) if scores else 0.0,
        "per_sample_scores": [
            {
                "selected_index": row.get("selected_index"),
                "sample_id": row.get("sample_id"),
                "score": extract_score(row.get("eval_response", "")),
            }
            for row in eval_rows
        ],
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
