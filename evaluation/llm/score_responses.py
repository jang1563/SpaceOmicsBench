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

1. **factual_accuracy** (weight {DIMENSION_WEIGHTS['factual_accuracy']}): Are stated facts correct? Does it cite accurate data values from the context? Flag factual_error if: wrong numerical values, wrong task descriptions (classes, metric, N), wrong model rankings, or wrong methodology (e.g., wrong sequencing technology).
2. **reasoning_quality** (weight {DIMENSION_WEIGHTS['reasoning_quality']}): Is the scientific reasoning sound and well-structured?
3. **completeness** (weight {DIMENSION_WEIGHTS['completeness']}): Does it cover the expected reasoning points and key factors?
4. **uncertainty_calibration** (weight {DIMENSION_WEIGHTS['uncertainty_calibration']}): Does it appropriately acknowledge limitations? CRITICAL: This benchmark uses N=4 crew and N=1 twin. Responses must acknowledge small sample size limitations to score above 3. Score 4+ requires explicit discussion of statistical power constraints.
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


def _load_dimension_descriptions() -> Dict[str, Dict]:
    """Load full dimension definitions from annotation_schema.json."""
    with open(ANNOTATION_SCHEMA_FILE) as f:
        schema = json.load(f)
    return schema["rating_dimensions"]


DIMENSION_DEFS = _load_dimension_descriptions()


# Per-dimension scoring instructions (dimension-specific anchoring)
_PERDIM_INSTRUCTIONS = {
    "factual_accuracy": (
        "Are stated facts correct? Does it cite accurate data values from the context? "
        "Flag factual_error if: wrong numerical values, wrong task descriptions "
        "(classes, metric, N), wrong model rankings, or wrong methodology."
    ),
    "reasoning_quality": (
        "Is the scientific reasoning sound and well-structured? "
        "Evaluate logical coherence, valid inference chains, and argument quality."
    ),
    "completeness": (
        "Does it cover the expected reasoning points and key factors? "
        "Check against the expected reasoning points and key facts listed above."
    ),
    "uncertainty_calibration": (
        "Does it appropriately acknowledge limitations? "
        "CRITICAL: This benchmark uses N=4 crew and N=1 twin. "
        "Responses MUST acknowledge small sample size limitations to score above 3. "
        "Score 4+ requires explicit discussion of statistical power constraints."
    ),
    "domain_integration": (
        "Does it connect findings across omics layers, missions, and space physiology? "
        "Evaluate depth of domain expertise and cross-modality reasoning."
    ),
}


def build_single_dimension_prompt(
    question_data: Dict, question_bank_entry: Dict,
    ground_truth: str, dimension: str
) -> str:
    """Build a prompt for scoring a single dimension independently."""
    qid = question_data.get("question_id", "?")
    question_text = question_data.get("question", "")
    response_text = question_data.get("response", "")
    difficulty = question_data.get("difficulty", "unknown")

    rubric = question_bank_entry.get("rubric", {})
    expected = question_bank_entry.get("expected_reasoning", [])
    key_facts = question_bank_entry.get("ground_truth_key_facts", [])

    expected_text = "\n".join(f"  - {r}" for r in expected)
    facts_text = "\n".join(f"  - {f}" for f in key_facts)

    # Dimension-specific rubric from question bank (if available)
    dim_rubric = rubric.get(dimension, "")
    dim_rubric_line = f"\n**Question-Specific Guidance:** {dim_rubric}" if dim_rubric else ""

    # Scale anchors from annotation schema
    dim_def = DIMENSION_DEFS.get(dimension, {})
    scale = dim_def.get("scale", {})
    scale_text = "\n".join(f"  {k}: {v}" for k, v in sorted(scale.items()))

    instruction = _PERDIM_INSTRUCTIONS.get(dimension, dim_def.get("description", ""))

    return f"""You are an expert evaluator for the SpaceOmicsBench benchmark on spaceflight biomedical AI.
You are scoring ONLY the **{dimension}** dimension.

## Ground Truth Reference
{ground_truth}

## Question-Specific Context

**Question ID:** {qid}
**Difficulty:** {difficulty}
**Question:** {question_text}

**Expected Reasoning Points:**
{expected_text}

**Key Facts (must be accurate):**
{facts_text}
{dim_rubric_line}

---

## Response to Evaluate

{response_text}

---

## Scoring: {dimension}

{instruction}

**Scale:**
{scale_text}

Score this response on **{dimension}** ONLY (1-5 scale).

Respond ONLY with a JSON object (no markdown fences):
{{
  "score": <1-5>,
  "justification": "1-2 sentence justification for the {dimension} score."
}}"""


def score_single_perdim(
    client, question_data: Dict, qb_entry: Dict,
    ground_truth: str, judge_model: str,
    judge_backend: str = "anthropic"
) -> Dict[str, Any]:
    """Score a single response with independent per-dimension LLM calls (5 calls)."""
    scores = {}
    total_in = 0
    total_out = 0

    for dim in DIMENSION_WEIGHTS:
        prompt = build_single_dimension_prompt(question_data, qb_entry, ground_truth, dim)
        try:
            if judge_backend == "openai":
                raw, tok_in, tok_out = _call_openai(client, prompt, judge_model)
            else:
                raw, tok_in, tok_out = _call_anthropic(client, prompt, judge_model)

            total_in += tok_in
            total_out += tok_out

            parsed = parse_judge_response(raw)
            score_val = parsed.get("score", 3)
            scores[dim] = max(1, min(5, score_val))
            scores[f"{dim}_justification"] = parsed.get("justification", "")

            time.sleep(0.3)  # Small delay between dimension calls

        except Exception as e:
            scores[dim] = 3  # Neutral default on failure
            scores[f"{dim}_error"] = str(e)

    # Compute weighted score
    scores["weighted_score"] = round(
        sum(scores.get(d, 3) * w for d, w in DIMENSION_WEIGHTS.items()), 2
    )
    scores["success"] = True
    scores["scoring_mode"] = "per_dimension"
    scores["judge_tokens"] = {"input": total_in, "output": total_out}

    # Flags — run a separate quick call for flags only
    try:
        flag_prompt = _build_flag_prompt(question_data, qb_entry, ground_truth)
        if judge_backend == "openai":
            raw, tok_in, tok_out = _call_openai(client, flag_prompt, judge_model)
        else:
            raw, tok_in, tok_out = _call_anthropic(client, flag_prompt, judge_model)
        total_in += tok_in
        total_out += tok_out
        flag_result = parse_judge_response(raw)
        scores["flags"] = flag_result.get("flags", {})
        scores["strengths"] = flag_result.get("strengths", [])
        scores["weaknesses"] = flag_result.get("weaknesses", [])
        scores["missed_points"] = flag_result.get("missed_points", [])
        scores["judge_tokens"] = {"input": total_in, "output": total_out}
    except Exception:
        scores["flags"] = {}

    return scores


def _build_flag_prompt(question_data: Dict, question_bank_entry: Dict, ground_truth: str) -> str:
    """Build a prompt for flags/strengths/weaknesses only (used in per-dimension mode)."""
    qid = question_data.get("question_id", "?")
    question_text = question_data.get("question", "")
    response_text = question_data.get("response", "")
    key_facts = question_bank_entry.get("ground_truth_key_facts", [])
    facts_text = "\n".join(f"  - {f}" for f in key_facts)

    return f"""You are an expert evaluator for SpaceOmicsBench.

## Ground Truth
{ground_truth}

## Question: {qid}
{question_text}

## Key Facts
{facts_text}

## Response
{response_text}

---

Identify flags, strengths, weaknesses, and missed points. Do NOT score dimensions.

Respond ONLY with JSON:
{{
  "flags": {{
    "hallucination": <bool>,
    "factual_error": <bool>,
    "harmful_recommendation": <bool>,
    "exceeds_data_scope": <bool>,
    "novel_insight": <bool>
  }},
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "missed_points": ["..."]
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


def _call_anthropic(client, prompt: str, judge_model: str) -> tuple:
    """Call Anthropic API and return (raw_text, input_tokens, output_tokens)."""
    resp = client.messages.create(
        model=judge_model,
        max_tokens=1000,
        temperature=0,
        system="You are an expert scientific benchmark evaluator. Score the response accurately and return only valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    if not resp.content:
        raise ValueError("Empty response from judge model")
    return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_openai(client, prompt: str, judge_model: str,
                 max_retries: int = 5) -> tuple:
    """Call OpenAI API with retry on rate limits. Returns (raw_text, input_tokens, output_tokens)."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                max_tokens=1000,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are an expert scientific benchmark evaluator. Score the response accurately and return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            if not resp.choices:
                raise ValueError("Empty response from judge model")
            return (resp.choices[0].message.content,
                    resp.usage.prompt_tokens, resp.usage.completion_tokens)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 2 ** attempt + 3  # 4, 5, 7, 11, 19 seconds
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")


def score_single(client, question_data: Dict, qb_entry: Dict,
                 ground_truth: str, judge_model: str,
                 judge_backend: str = "anthropic") -> Dict[str, Any]:
    """Score a single response using an LLM judge."""
    prompt = build_scoring_prompt(question_data, qb_entry, ground_truth)
    raw = ""

    try:
        if judge_backend == "openai":
            raw, tok_in, tok_out = _call_openai(client, prompt, judge_model)
        else:
            raw, tok_in, tok_out = _call_anthropic(client, prompt, judge_model)

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
        scores["judge_tokens"] = {"input": tok_in, "output": tok_out}
        return scores

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}", "raw": raw}
    except Exception as e:
        return {"success": False, "error": str(e), "raw": raw}


def score_all(input_file: str, output_file: Optional[str] = None,
              judge_model: str = "claude-sonnet-4-20250514",
              judge_backend: str = "anthropic",
              per_dimension: bool = False,
              sample_n: Optional[int] = None):
    """Score all responses in an evaluation results file."""
    if judge_backend == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("Error: pip install openai")
            return None
        client = OpenAI()
        judge_label = f"GPT-as-Judge ({judge_model})"
    else:
        try:
            import anthropic
        except ImportError:
            print("Error: pip install anthropic")
            return None
        client = anthropic.Anthropic()
        judge_label = f"Claude-as-Judge ({judge_model})"

    print("=" * 70)
    print(f"SpaceOmicsBench v2 - Response Scoring ({judge_label})")
    print("=" * 70)

    # Load inputs
    with open(input_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    qb = load_question_bank()
    gt = load_ground_truth()

    # Sample if requested
    if sample_n and sample_n < len(results):
        import random
        random.seed(42)
        results = random.sample(results, sample_n)

    scoring_mode = "per_dimension" if per_dimension else "combined"
    print(f"\nInput:       {input_file}")
    print(f"Judge model: {judge_model}")
    print(f"Scoring:     {scoring_mode}")
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
        if per_dimension:
            scores = score_single_perdim(client, r, qb_entry, gt, judge_model, judge_backend)
        else:
            scores = score_single(client, r, qb_entry, gt, judge_model, judge_backend)

        if scores.get("success"):
            ws = scores.get("weighted_score", "?")
            print(f"score={ws}")
            jt = scores.get("judge_tokens", {})
            total_judge_input += jt.get("input", 0)
            total_judge_output += jt.get("output", 0)
        else:
            print(f"ERROR: {scores.get('error', 'Unknown')}")

        scored.append({**r, "scores": scores})
        time.sleep(6 if judge_backend == "openai" else 0.5)  # OpenAI Tier 1: 30K TPM

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
        suffix = "_perdim" if per_dimension else ""
        output_file = str(Path(input_file).parent / f"scored_{stem}{suffix}.json")

    output = {
        "metadata": {
            **data.get("metadata", {}),
            "scoring_timestamp": datetime.now().isoformat(),
            "judge_model": judge_model,
            "scoring_mode": scoring_mode,
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
                        help="Judge model (default: claude-sonnet-4-20250514)")
    parser.add_argument("--judge-backend", default="anthropic",
                        choices=["anthropic", "openai"],
                        help="Judge API backend (default: anthropic)")
    parser.add_argument("--per-dimension", action="store_true",
                        help="Score each dimension independently (5 API calls per question)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Score only N randomly sampled questions")

    args = parser.parse_args()

    for fpath in args.input_files:
        out = args.output if len(args.input_files) == 1 else None
        score_all(fpath, out, args.judge_model, args.judge_backend,
                  args.per_dimension, args.sample)
        if len(args.input_files) > 1:
            print("\n")


if __name__ == "__main__":
    main()
