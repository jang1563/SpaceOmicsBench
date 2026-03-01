# Results Directory

## v2.1 Scored Results (Official Leaderboard)

The `v2.1/` subdirectory contains the official scored evaluation files for all 9 models.
These are the canonical results behind the leaderboard at `llm_leaderboard.html`.

**v2.1 corrections applied:**
- Q27/Q28/Q64 `ground_truth_key_facts` corrected (exact spatial DE counts)
- Judge `max_tokens` increased 1000 → 2048 (fixes Haiku 4.5 Q10 JSON truncation)
- Gemini 2.5 Flash re-evaluated with `max_tokens=8192` (resolves thinking-token truncation)

| File | Model | Overall Score |
|------|-------|:---:|
| `scored_eval_claude-sonnet-4-6_judged-by-sonnet46.json` | Claude Sonnet 4.6 | 4.62 |
| `scored_eval_haiku-4-5_judged-by-sonnet46.json` | Claude Haiku 4.5 | 4.41 |
| `scored_eval_deepseek-v3_judged-by-sonnet46.json` | DeepSeek-V3 | 4.34 |
| `scored_eval_claude-sonnet-4_judged-by-sonnet46.json` | Claude Sonnet 4 | 4.03 |
| `scored_eval_gemini-2.5-flash_judged-by-sonnet46.json` | Gemini 2.5 Flash | 4.00 |
| `scored_eval_gpt-4o-mini_judged-by-sonnet46.json` | GPT-4o Mini | 3.32 |
| `scored_eval_llama33-70b-groq_judged-by-sonnet46.json` | Llama-3.3-70B (Groq) | 3.31 |
| `scored_eval_llama33-70b-together_judged-by-sonnet46.json` | Llama-3.3-70B (Together) | 3.31 |
| `scored_eval_gpt-4o_judged-by-sonnet46.json` | GPT-4o | 3.30 |

All scores on 1–5 scale. Judge: Claude Sonnet 4.6.

## Running Your Own Evaluation

```bash
# Step 1: Run the model
python evaluation/llm/run_llm_evaluation.py --model your-model --full --output-dir results/raw

# Step 2: Score with judge
python evaluation/llm/score_responses.py results/raw/eval_your-model_*.json \
    --judge-model claude-sonnet-4-6 --judge-backend anthropic
```

For baseline reference scores, see `baselines/baseline_results.json`.
