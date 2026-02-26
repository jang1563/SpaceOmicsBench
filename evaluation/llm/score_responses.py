#!/usr/bin/env python3
"""
SpaceOmicsBench v2 - Response Scoring (Claude-as-Judge)
========================================================

Scores LLM responses using Claude as an expert judge with per-question rubrics
and ground truth context.

Usage:
    python score_responses.py results/eval_claude-sonnet-4-20250514_20250225_120000.json
    python score_responses.py results/eval_gpt-4o_20250225_130000.json --output scored_gpt4o.json
    python score_responses.py results/eval_*.json  # score multiple files
"""

import json
import argparse
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
QUESTION_BANK_FILE = SCRIPT_DIR / "question_bank.json"
ANNOTATION_SCHEMA_FILE = SCRIPT_DIR / "annotation_schema.json"
GROUND_TRUTH_FILE = SCRIPT_DIR / "data_context" / "ground_truth.md"


def _load_dimension_weights() -> Dict[str, float]:
    """Load dimension weights from annotation_schema.json (single source of truth)."""
    with open(ANNOTATION_SCHEMA_FILE) as f:
        schema = json.load(f)
    return {dim: cfg["weight"] for dim, cfg in schema["rating_dimensions"].items()}


DIMENSION_WEIGHTS = _load_dimension_weights()


def load_question_bank() -> Dict[str, Dict]:
    """Load question bank indexed by question ID."""
    with open(QUESTION_BANK_FILE) as f:
        bank = json.load(f)
    return {q["id"]: q for q in bank["questions"]}


def load_ground_truth() -> str:
    """Load the ground truth markdown context."""
    if GROUND_TRUTH_FILE.exists():
        return GROUND_TRUTH_FILE.read_text().strip()
    return ""


def build_scoring_prompt(question_data: Dict, question_bank_entry: Dict, ground_truth: str) -> str:
    """Build the scoring prompt with per-question rubric and ground truth."""

    qid = question_data.get("question_id", "?")
    question_text = question_data.get("question", "")
    response_text = question_data.get("response", "")
    difficulty = question_data.get("difficulty", "unknown")

    # Per-question rubric and expected reasoning
    rubric = question_bank_entry.get("rubric", {})
    expected = question_bank_entry.get("expected_reasoning", [])
    key_facts = question_bank_entry.get("ground_truth_key_facts", [])

    rubric_text = "\n".join(f"  - {dim}: {desc}" for dim, desc in rubric.items())
    expected_text = "\n".join(f"  - {r}" for r in expected)
    facts_text = "\n".join(f"  - {f}" for f in key_facts)

    return f"""You are an expert evaluator for the SpaceOmicsBench benchmark on spaceflight biomedical AI.

## Ground Truth Reference
{ground_truth}

## Question-Specific Evaluation Criteria

**Question ID:** {qid}
**Difficulty:** {difficulty}
**Question:** {question_text}

**Expected Reasoning Points:**
{expected_text}

**Key Facts (must be accurate):**
{facts_text}

**Per-Dimension Rubric:**
{rubric_text}

---

## Response to Evaluate

{response_text}

---

## Scoring Instructions

Score the response on these 5 dimensions (1-5 scale, 5 = excellent):

1. **factual_accuracy** (weight {DIMENSION_WEIGHTS['factual_accuracy']}): Are stated facts correct? Does it cite accurate data values from the context?
2. **reasoning_quality** (weight {DIMENSION_WEIGHTS['reasoning_quality']}): Is the scientific reasoning sound and well-structured?
3. **completeness** (weight {DIMENSION_WEIGHTS['completeness']}): Does it cover the expected reasoning points and key factors?
4. **uncertainty_calibration** (weight {DIMENSION_WEIGHTS['uncertainty_calibration']}): Does it appropriately acknowledge limitations, especially given small sample sizes (N=4, N=1)?
5. **domain_integration** (weight {DIMENSION_WEIGHTS['domain_integration']}): Does it connect findings across omics layers, missions, and space physiology?

Respond ONLY with a JSON object (no markdown fences):
{{
  "factual_accuracy": <1-5>,
  "reasoning_quality": <1-5>,
  "completeness": <1-5>,
  "uncertainty_calibration": <1-5>,
  "domain_integration": <1-5>,
  "weighted_score": <float>,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "missed_points": ["..."],
  "flags": {{
    "hallucination": <bool>,
    "factual_error": <bool>,
    "harmful_recommendation": <bool>,
    "exceeds_data_scope": <bool>,
    "novel_insight": <bool>
  }},
  "justification": "Brief 2-3 sentence justification of the scores."
}}"""


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse the judge's JSON response, handling markdown fences if present."""
    # Try to extract JSON from markdown code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1).strip())

    # Fall back to balanced brace matching (avoids greedy regex issues)
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = None
                    continue
    return json.loads(text.strip())


def score_single(client, question_data: Dict, qb_entry: Dict,
                 ground_truth: str, judge_model: str) -> Dict[str, Any]:
    """Score a single response using Claude as judge."""
    prompt = build_scoring_prompt(question_data, qb_entry, ground_truth)
    raw = ""

    try:
        resp = client.messages.create(
            model=judge_model,
            max_tokens=1000,
            temperature=0,
            system="You are an expert scientific benchmark evaluator. Score the response accurately and return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        if not resp.content:
            return {"success": False, "error": "Empty response from judge model"}

        raw = resp.content[0].text
        scores = parse_judge_response(raw)

        # Always recompute weighted_score server-side (don't trust judge arithmetic)
        weights = {
            "factual_accuracy": 0.25,
            "reasoning_quality": 0.25,
            "completeness": 0.20,
            "uncertainty_calibration": 0.15,
            "domain_integration": 0.15,
        }
        for dim in weights:
            if dim in scores:
                scores[dim] = max(1, min(5, scores[dim]))
        scores["weighted_score"] = round(sum(scores.get(d, 3) * w for d, w in weights.items()), 2)

        scores["success"] = True
        scores["judge_tokens"] = {
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
        }
        return scores

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}", "raw": raw}
    except Exception as e:
        return {"success": False, "error": str(e), "raw": raw}


def score_all(input_file: str, output_file: Optional[str] = None,
              judge_model: str = "claude-sonnet-4-20250514"):
    """Score all responses in an evaluation results file."""
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic")
        return None

    print("=" * 70)
    print("SpaceOmicsBench v2 - Response Scoring (Claude-as-Judge)")
    print("=" * 70)

    # Load inputs
    with open(input_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    qb = load_question_bank()
    gt = load_ground_truth()
    client = anthropic.Anthropic()

    print(f"\nInput:       {input_file}")
    print(f"Judge model: {judge_model}")
    print(f"Responses:   {len(results)}")
    print("-" * 70)

    scored = []
    total_judge_input = 0
    total_judge_output = 0

    for i, r in enumerate(results):
        qid = r.get("question_id", "?")
        if not r.get("success") or not r.get("response"):
            print(f"[{i+1}] {qid} - Skipping (no response)")
            scored.append({**r, "scores": {"success": False, "error": "no response"}})
            continue

        print(f"[{i+1}/{len(results)}] Scoring {qid} ({r.get('difficulty', '?')})...", end=" ", flush=True)

        qb_entry = qb.get(qid, {})
        scores = score_single(client, r, qb_entry, gt, judge_model)

        if scores.get("success"):
            ws = scores.get("weighted_score", "?")
            print(f"score={ws}")
            jt = scores.get("judge_tokens", {})
            total_judge_input += jt.get("input", 0)
            total_judge_output += jt.get("output", 0)
        else:
            print(f"ERROR: {scores.get('error', 'Unknown')}")

        scored.append({**r, "scores": scores})
        time.sleep(0.5)  # Rate limit

    # Aggregates
    ok = [s["scores"] for s in scored if s["scores"].get("success")]
    summary = {}
    if ok:
        for dim in DIMENSION_WEIGHTS:
            vals = [s[dim] for s in ok if dim in s]
            summary[f"avg_{dim}"] = round(sum(vals) / len(vals), 3) if vals else None
        ws_vals = [s["weighted_score"] for s in ok if "weighted_score" in s]
        summary["avg_weighted_score"] = round(sum(ws_vals) / len(ws_vals), 3) if ws_vals else None
        summary["n_scored"] = len(ok)

        # By difficulty
        for diff in ["easy", "medium", "hard", "expert"]:
            diff_scores = [s["scores"]["weighted_score"] for s in scored
                           if s.get("difficulty") == diff and s["scores"].get("success")
                           and "weighted_score" in s["scores"]]
            if diff_scores:
                summary[f"avg_{diff}"] = round(sum(diff_scores) / len(diff_scores), 3)

        # By modality
        for mod in set(s.get("modality") for s in scored):
            if not mod:
                continue
            mod_scores = [s["scores"]["weighted_score"] for s in scored
                          if s.get("modality") == mod and s["scores"].get("success")
                          and "weighted_score" in s["scores"]]
            if mod_scores:
                summary[f"avg_{mod}"] = round(sum(mod_scores) / len(mod_scores), 3)

        # Flag counts
        for flag in ["hallucination", "factual_error", "harmful_recommendation",
                      "exceeds_data_scope", "novel_insight"]:
            summary[f"flag_{flag}"] = sum(
                1 for s in ok if s.get("flags", {}).get(flag, False))

    # Output
    if not output_file:
        stem = Path(input_file).stem
        output_file = str(Path(input_file).parent / f"scored_{stem}.json")

    output = {
        "metadata": {
            **data.get("metadata", {}),
            "scoring_timestamp": datetime.now().isoformat(),
            "judge_model": judge_model,
            "judge_tokens_input": total_judge_input,
            "judge_tokens_output": total_judge_output,
        },
        "summary": summary,
        "results": scored,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("SCORING SUMMARY")
    print("=" * 70)
    print(f"Scored: {summary.get('n_scored', 0)}/{len(results)}")
    if summary:
        print(f"\nDimension Averages (1-5):")
        for dim in DIMENSION_WEIGHTS:
            v = summary.get(f"avg_{dim}")
            print(f"  {dim:<28s} {v:.2f}" if v else f"  {dim:<28s} N/A")
        print(f"  {'─' * 36}")
        print(f"  {'WEIGHTED SCORE':<28s} {summary.get('avg_weighted_score', 0):.2f}/5.00")

        print(f"\nBy Difficulty:")
        for diff in ["easy", "medium", "hard", "expert"]:
            v = summary.get(f"avg_{diff}")
            if v:
                print(f"  {diff:<12s} {v:.2f}")

        print(f"\nFlags:")
        for flag_name in ["hallucination", "factual_error", "novel_insight", "harmful_recommendation", "exceeds_data_scope"]:
            print(f"  {flag_name}: {summary.get(f'flag_{flag_name}', 0)}")

    print(f"\nJudge tokens: {total_judge_input:,} in / {total_judge_output:,} out")
    print(f"Saved to: {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Score SpaceOmicsBench LLM responses")
    parser.add_argument("input_files", nargs="+", help="Evaluation result JSON file(s)")
    parser.add_argument("--output", default=None, help="Output file (single input only)")
    parser.add_argument("--judge-model", default="claude-sonnet-4-20250514",
                        help="Claude model for judging (default: claude-sonnet-4-20250514)")

    args = parser.parse_args()

    for fpath in args.input_files:
        out = args.output if len(args.input_files) == 1 else None
        score_all(fpath, out, args.judge_model)
        if len(args.input_files) > 1:
            print("\n")


if __name__ == "__main__":
    main()
