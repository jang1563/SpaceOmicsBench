#!/usr/bin/env python3
"""
SpaceOmicsBench v2.1 - Response Scoring Script
===============================================

Scores LLM responses using Claude as a judge.

Usage:
    python score_responses.py results/evaluation_results_XXXX.json
    python score_responses.py quick_test_results_XXXX.json
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("❌ Please install anthropic: pip install anthropic")
    exit(1)


SCORING_PROMPT = """You are an expert evaluator for the SpaceOmicsBench benchmark on spaceflight biomedical AI.

Score the following response on these 5 dimensions (1-5 scale, 5=excellent):

1. **Factual Accuracy** (1-5): Are the stated facts correct? Does it accurately cite the data?
2. **Reasoning Quality** (1-5): Is the scientific reasoning sound and well-structured?
3. **Completeness** (1-5): Does it address all aspects of the question?
4. **Uncertainty Calibration** (1-5): Does it appropriately acknowledge limitations?
5. **Domain Integration** (1-5): Does it connect findings across omics layers appropriately?

Also identify:
- **Key Strengths**: What did the response do well?
- **Key Weaknesses**: What could be improved?
- **Flags**: Any hallucinations, harmful recommendations, or factual errors?

Respond in JSON format:
```json
{
  "factual_accuracy": X,
  "reasoning_quality": X,
  "completeness": X,
  "uncertainty_calibration": X,
  "domain_integration": X,
  "overall_score": X.X,
  "key_strengths": ["...", "..."],
  "key_weaknesses": ["...", "..."],
  "flags": {
    "hallucination": false,
    "harmful": false,
    "factual_error": false
  },
  "brief_justification": "..."
}
```"""


GROUND_TRUTH_CONTEXT = """
## Ground Truth Data for Scoring

**Cross-Mission Statistics:**
- Transcriptomics correlation: r = -0.02 (p = 0.11)
- Metabolomics correlation: r = +0.40 (p < 0.001)
- Transcriptomics concordance: 46.4%
- Metabolomics concordance: 67.0%
- Erythroid gene concordance: 75%

**Top Findings:**
- EPB42: PD -4.17, I4 -1.21 (erythrocyte membrane)
- ANK1: PD -1.88, I4 -0.80 (RBC cytoskeleton)
- HBB: PD -0.64, I4 -2.22 (hemoglobin)
- 9 DEGs at R+1, 0 DEGs at R+39 (transient response)
- NLR: +52% increase in Polaris Dawn
- Corticosterone: +0.60 (PD), +0.54 (I4) log2FC
- Caffeine: -2.12 (PD), -1.93 (I4) log2FC
"""


def score_response(client, question: str, response: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Score a single response using Claude as judge"""
    
    eval_prompt = f"""{GROUND_TRUTH_CONTEXT}

---

**Question:** {question}

**Response to Evaluate:**
{response}

---

{SCORING_PROMPT}"""
    
    try:
        result = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        # Parse JSON from response
        response_text = result.content[0].text
        
        # Extract JSON block
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text
            
        scores = json.loads(json_str.strip())
        scores['success'] = True
        return scores
        
    except json.JSONDecodeError as e:
        return {'success': False, 'error': f'JSON parse error: {e}', 'raw': response_text}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def score_all_responses(input_file: str, output_file: str = None, api_key: str = None):
    """Score all responses in an evaluation results file"""
    
    print("="*70)
    print("SpaceOmicsBench v2.1 - Response Scoring")
    print("="*70)
    
    # Load results
    with open(input_file) as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"\n📄 Input: {input_file}")
    print(f"📝 Responses to score: {len(results)}")
    
    # Initialize client
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    
    # Score each response
    scored_results = []
    
    for i, r in enumerate(results):
        if not r.get('success', False) or not r.get('response'):
            print(f"[{i+1}] {r.get('question_id', '?')} - Skipping (no response)")
            continue
            
        print(f"[{i+1}/{len(results)}] Scoring {r.get('question_id', '?')}...", end=" ", flush=True)
        
        scores = score_response(
            client,
            question=r.get('question', ''),
            response=r.get('response', '')
        )
        
        if scores.get('success'):
            print(f"✓ Overall: {scores.get('overall_score', 'N/A')}")
        else:
            print(f"✗ Error: {scores.get('error', 'Unknown')}")
        
        scored_results.append({
            **r,
            'scores': scores
        })
        
        time.sleep(0.5)  # Rate limiting
    
    # Calculate aggregates
    successful_scores = [r['scores'] for r in scored_results if r['scores'].get('success')]
    
    if successful_scores:
        avg_scores = {
            'factual_accuracy': sum(s['factual_accuracy'] for s in successful_scores) / len(successful_scores),
            'reasoning_quality': sum(s['reasoning_quality'] for s in successful_scores) / len(successful_scores),
            'completeness': sum(s['completeness'] for s in successful_scores) / len(successful_scores),
            'uncertainty_calibration': sum(s['uncertainty_calibration'] for s in successful_scores) / len(successful_scores),
            'domain_integration': sum(s['domain_integration'] for s in successful_scores) / len(successful_scores),
            'overall': sum(s['overall_score'] for s in successful_scores) / len(successful_scores)
        }
    else:
        avg_scores = {}
    
    # Save results
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'scored_results_{timestamp}.json'
    
    output = {
        'metadata': {
            **data.get('metadata', {}),
            'scoring_timestamp': datetime.now().isoformat(),
            'n_scored': len(successful_scores)
        },
        'summary': avg_scores,
        'results': scored_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SCORING SUMMARY")
    print("="*70)
    print(f"Scored: {len(successful_scores)}/{len(results)}")
    
    if avg_scores:
        print(f"\nAverage Scores:")
        print(f"  Factual Accuracy:      {avg_scores['factual_accuracy']:.2f}/5")
        print(f"  Reasoning Quality:     {avg_scores['reasoning_quality']:.2f}/5")
        print(f"  Completeness:          {avg_scores['completeness']:.2f}/5")
        print(f"  Uncertainty Calib.:    {avg_scores['uncertainty_calibration']:.2f}/5")
        print(f"  Domain Integration:    {avg_scores['domain_integration']:.2f}/5")
        print(f"  ─────────────────────────────")
        print(f"  OVERALL:               {avg_scores['overall']:.2f}/5")
    
    print(f"\n💾 Saved to: {output_file}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Score SpaceOmicsBench responses')
    parser.add_argument('input_file', type=str, help='Path to evaluation results JSON')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--api-key', type=str, help='Anthropic API key')
    
    args = parser.parse_args()
    score_all_responses(args.input_file, args.output, args.api_key)


if __name__ == '__main__':
    main()
