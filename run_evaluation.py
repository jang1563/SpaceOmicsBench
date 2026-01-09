#!/usr/bin/env python3
"""
SpaceOmicsBench v2.1 - LLM Evaluation Script
=============================================

Usage:
    python run_evaluation.py --sample 10          # Test with 10 questions
    python run_evaluation.py --full               # Run all 115 questions
    python run_evaluation.py --modality metabolomics  # Specific modality
    python run_evaluation.py --retry results/evaluation_results_XXX.json  # Retry failed questions

Requirements:
    pip install anthropic pandas tqdm

API Key:
    export ANTHROPIC_API_KEY="your-key-here"
"""

import anthropic
import json
import os
import argparse
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data context to provide to the model
DATA_CONTEXT = """
## SpaceOmicsBench v2.1 - Multi-Omics Spaceflight Data Context

### Dataset Overview
You are answering questions about real spaceflight biomedical data from commercial missions.

**Missions:**
- Inspiration4 (I4): 3-day mission, 4 crew members
- Polaris Dawn (PD): 5-day mission, 4 crew members, included EVA

**Data Modalities:**
1. Clinical (CBC/CMP): 34 blood biomarkers
2. Transcriptomics (cfRNA-seq): 5,346 genes
3. Metabolomics: 199 metabolites

### Key Statistics

**Cross-Mission Correlation:**
| Modality | Pearson r | Direction Concordance |
|----------|-----------|----------------------|
| Transcriptomics | -0.02 (near zero) | 46.4% |
| Metabolomics | +0.40 (significant) | 67.0% |

**Key Finding:** Metabolomics shows 20x higher cross-mission correlation than transcriptomics.

**Top Conserved Transcriptomic Changes (Erythroid - 75% concordance):**
- EPB42: PD -4.17, I4 -1.21 (erythrocyte membrane protein)
- ANK1: PD -1.88, I4 -0.80 (RBC cytoskeleton)
- HBB/HBA: Hemoglobin genes consistently DOWN

**Top Conserved Metabolomic Changes:**
- UP: Corticosterone (+0.60, +0.54), 1-Methylnicotinamide (+1.09, +0.91)
- DOWN: Caffeine (-2.12, -1.93), Quinic Acid (-3.74, -2.51)

**Clinical Findings:**
- NLR (Neutrophil-to-Lymphocyte Ratio): +52% increase in Polaris Dawn
- Lymphocytes: -40% decrease
- Liver enzymes (ALT, AST): Elevated post-flight

**Temporal Patterns:**
- Transcriptomics: 9 DEGs at R+1, 0 DEGs at R+39 (complete normalization)
- Most changes are transient, normalizing within 5-6 weeks
"""

SYSTEM_PROMPT = """You are an expert in space medicine, bioinformatics, and multi-omics analysis. 
You are answering questions about spaceflight biomedical data from the SpaceOmicsBench benchmark.

When answering:
1. Be specific and cite relevant statistics when available
2. Explain biological mechanisms when relevant
3. Acknowledge limitations and uncertainties
4. Connect findings across different omics layers when appropriate
5. Consider both statistical and biological significance

Keep responses focused and informative (300-500 words typical)."""


def load_questions(benchmark_dir: str) -> dict:
    """Load all questions from SpaceOmicsBench"""
    questions = {
        'clinical': [],
        'transcriptomics': [],
        'metabolomics': []
    }
    
    # Load clinical questions
    clinical_path = Path(benchmark_dir) / 'tasks' / 'question_bank.json'
    if clinical_path.exists():
        with open(clinical_path) as f:
            data = json.load(f)
            for q in data.get('questions', []):
                q['modality'] = 'clinical'
                questions['clinical'].append(q)
    
    # Load transcriptomics questions
    trans_path = Path(benchmark_dir) / 'tasks' / 'transcriptomics_questions_v2.json'
    if trans_path.exists():
        with open(trans_path) as f:
            data = json.load(f)
            for q in data.get('questions', []):
                q['modality'] = 'transcriptomics'
                questions['transcriptomics'].append(q)
    
    # Load metabolomics questions
    met_path = Path(benchmark_dir) / 'tasks' / 'metabolomics_questions.json'
    if met_path.exists():
        with open(met_path) as f:
            data = json.load(f)
            for q in data.get('questions', []):
                q['modality'] = 'metabolomics'
                questions['metabolomics'].append(q)
    
    return questions


def query_claude(client, question: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Send a question to Claude API and get response"""
    
    full_prompt = f"{DATA_CONTEXT}\n\n---\n\n**Question:** {question}\n\n**Your Answer:**"
    
    start_time = time.time()
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'response': response.content[0].text,
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens,
            'response_time_sec': round(elapsed, 2),
            'model': model
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'response_time_sec': time.time() - start_time
        }


def run_evaluation(
    benchmark_dir: str,
    output_dir: str = "results",
    model: str = "claude-sonnet-4-20250514",
    sample_size: int = None,
    modality: str = None,
    api_key: str = None
):
    """Run full LLM evaluation"""
    
    # Initialize client
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load questions
    print("📚 Loading questions...")
    all_questions = load_questions(benchmark_dir)
    
    # Filter by modality if specified
    if modality:
        questions_to_run = all_questions.get(modality, [])
    else:
        questions_to_run = []
        for mod_questions in all_questions.values():
            questions_to_run.extend(mod_questions)
    
    print(f"   Total questions available: {len(questions_to_run)}")
    
    # Sample if requested
    if sample_size and sample_size < len(questions_to_run):
        import random
        random.seed(42)
        questions_to_run = random.sample(questions_to_run, sample_size)
        print(f"   Sampling {sample_size} questions")
    
    # Run evaluation
    print(f"\n🚀 Running evaluation with {model}...")
    print(f"   Questions to evaluate: {len(questions_to_run)}")
    print("-" * 60)
    
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, q in enumerate(questions_to_run):
        q_id = q.get('id', f'Q{i+1}')
        q_text = q.get('question', '')
        q_diff = q.get('difficulty', 'unknown')
        q_mod = q.get('modality', 'unknown')
        
        print(f"[{i+1}/{len(questions_to_run)}] {q_id} ({q_mod}, {q_diff})...", end=" ", flush=True)
        
        result = query_claude(client, q_text, model)
        
        if result['success']:
            print(f"✓ ({result['response_time_sec']}s, {result['output_tokens']} tokens)")
            total_input_tokens += result['input_tokens']
            total_output_tokens += result['output_tokens']
        else:
            print(f"✗ Error: {result.get('error', 'Unknown')}")
        
        results.append({
            'question_id': q_id,
            'modality': q_mod,
            'difficulty': q_diff,
            'question': q_text,
            **result
        })
        
        # Rate limiting - wait between requests
        time.sleep(0.5)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = Path(output_dir) / f'evaluation_results_{timestamp}.json'
    
    output = {
        'metadata': {
            'model': model,
            'timestamp': timestamp,
            'n_questions': len(results),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'modality_filter': modality,
            'sample_size': sample_size
        },
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Print summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n📊 EVALUATION SUMMARY")
    print(f"   Successful: {successful}/{len(results)}")
    print(f"   Total input tokens: {total_input_tokens:,}")
    print(f"   Total output tokens: {total_output_tokens:,}")
    
    # Estimate cost (Sonnet pricing)
    input_cost = total_input_tokens * 0.003 / 1000
    output_cost = total_output_tokens * 0.015 / 1000
    print(f"   Estimated cost (Sonnet): ${input_cost + output_cost:.2f}")
    
    return output


def retry_failed(
    input_file: str,
    model: str = None,
    api_key: str = None
):
    """Retry only failed questions from a previous evaluation run"""

    # Load existing results
    with open(input_file) as f:
        data = json.load(f)

    results = data['results']
    metadata = data['metadata']

    # Use original model if not specified
    if not model:
        model = metadata.get('model', 'claude-sonnet-4-20250514')

    # Find failed questions
    failed_indices = []
    for i, r in enumerate(results):
        if not r.get('success', False) or not r.get('response'):
            failed_indices.append(i)

    if not failed_indices:
        print("✅ No failed questions found. All questions completed successfully!")
        return data

    print(f"📋 Found {len(failed_indices)} failed questions to retry")
    print(f"🔄 Using model: {model}")
    print("-" * 60)

    # Initialize client
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # Retry failed questions
    retried = 0
    for idx in failed_indices:
        r = results[idx]
        q_id = r.get('question_id', f'Q{idx+1}')
        q_text = r.get('question', '')

        print(f"[{retried+1}/{len(failed_indices)}] Retrying {q_id}...", end=" ", flush=True)

        result = query_claude(client, q_text, model)

        if result['success']:
            print(f"✓ ({result['response_time_sec']}s, {result['output_tokens']} tokens)")
            # Update the result in place
            results[idx].update(result)
            metadata['total_input_tokens'] = metadata.get('total_input_tokens', 0) + result['input_tokens']
            metadata['total_output_tokens'] = metadata.get('total_output_tokens', 0) + result['output_tokens']
            retried += 1
        else:
            print(f"✗ Error: {result.get('error', 'Unknown')}")

        time.sleep(0.5)

    # Save updated results back to the same file
    data['metadata']['retry_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    data['metadata']['retried_count'] = retried

    with open(input_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Updated {input_file}")
    print(f"   Successfully retried: {retried}/{len(failed_indices)}")

    return data


def main():
    parser = argparse.ArgumentParser(description='SpaceOmicsBench LLM Evaluation')
    parser.add_argument('--benchmark-dir', type=str, default='.',
                        help='Path to benchmark directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514',
                        choices=['claude-sonnet-4-20250514', 'claude-opus-4-20250514',
                                 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
                        help='Model to use')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of questions to sample (default: all)')
    parser.add_argument('--full', action='store_true',
                        help='Run full evaluation (all questions)')
    parser.add_argument('--modality', type=str, default=None,
                        choices=['clinical', 'transcriptomics', 'metabolomics'],
                        help='Filter to specific modality')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--retry', type=str, default=None,
                        help='Path to previous results file to retry failed questions')

    args = parser.parse_args()

    # If retrying, use retry function
    if args.retry:
        retry_failed(
            input_file=args.retry,
            model=args.model if args.model != 'claude-sonnet-4-20250514' else None,
            api_key=args.api_key
        )
        return

    # Determine sample size
    sample_size = None if args.full else (args.sample or 10)

    run_evaluation(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        model=args.model,
        sample_size=sample_size,
        modality=args.modality,
        api_key=args.api_key
    )


if __name__ == '__main__':
    main()
