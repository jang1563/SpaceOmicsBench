#!/usr/bin/env python3
"""
SpaceOmicsBench v2.1 - Multi-Model LLM Evaluation Script
=========================================================

Supports Claude, OpenAI, and HuggingFace (local) models including fine-tuned LoRA models.

Usage:
    # Claude (API)
    export ANTHROPIC_API_KEY="your-key"
    python run_model_evaluation.py --model claude-sonnet-4-20250514 --sample 10

    # OpenAI (API)
    export OPENAI_API_KEY="your-key"
    python run_model_evaluation.py --model gpt-4 --sample 10

    # HuggingFace (local) - requires GPU
    python run_model_evaluation.py --model mistralai/Mistral-7B-v0.3 --sample 10

    # Fine-tuned LoRA model - requires GPU
    python run_model_evaluation.py \
        --model mistralai/Mistral-7B-v0.3 \
        --adapter-path ./ecosystem_improved_model \
        --sample 10

Requirements:
    pip install anthropic openai transformers torch peft bitsandbytes accelerate pandas tqdm
"""

import json
import os
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

# ============================================================================
# MODEL BACKENDS
# ============================================================================

class BaseModelBackend:
    """Abstract base class for model backends."""

    def __init__(self, model_name: str, max_tokens: int = 1500, temperature: float = 0.3):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        raise NotImplementedError


class ClaudeBackend(BaseModelBackend):
    """Anthropic Claude API backend."""

    def __init__(self, model_name: str, max_tokens: int = 1500, temperature: float = 0.3):
        super().__init__(model_name, max_tokens, temperature)
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        start_time = time.time()
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)
            elapsed = time.time() - start_time

            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            return {
                'success': True,
                'response': response.content[0].text,
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'response_time_sec': round(elapsed, 2),
                'model': self.model_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time_sec': time.time() - start_time
            }


class OpenAIBackend(BaseModelBackend):
    """OpenAI API backend."""

    def __init__(self, model_name: str, max_tokens: int = 1500, temperature: float = 0.3):
        super().__init__(model_name, max_tokens, temperature)
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        start_time = time.time()
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages
            )
            elapsed = time.time() - start_time

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            return {
                'success': True,
                'response': response.choices[0].message.content,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'response_time_sec': round(elapsed, 2),
                'model': self.model_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time_sec': time.time() - start_time
            }


class HuggingFaceBackend(BaseModelBackend):
    """HuggingFace local model backend with LoRA support."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1500,
        temperature: float = 0.3,
        adapter_path: Optional[str] = None,
        use_4bit: bool = True,
    ):
        super().__init__(model_name, max_tokens, temperature)
        self.adapter_path = adapter_path
        self._load_model(model_name, adapter_path, use_4bit)

    def _load_model(self, model_name: str, adapter_path: Optional[str], use_4bit: bool):
        """Load the model and tokenizer."""
        try:
            import torch
            self.torch = torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch accelerate"
            )

        print(f"Loading HuggingFace model: {model_name}")

        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            device_map = "auto"
        else:
            print("No GPU detected, using CPU (will be slow)")
            device_map = "cpu"
            use_4bit = False

        # Load tokenizer
        tokenizer_path = adapter_path if adapter_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, quantization_config=bnb_config,
                    device_map=device_map, trust_remote_code=True,
                )
            except ImportError:
                print("bitsandbytes not available, loading without quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map=device_map,
                    torch_dtype=torch.float16, trust_remote_code=True,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )

        # Load LoRA adapters
        if adapter_path:
            try:
                from peft import PeftModel
                print(f"Loading LoRA adapters from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except ImportError:
                raise ImportError("Please install peft: pip install peft")

        self.model.eval()
        print("Model loaded successfully!")

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        start_time = time.time()

        try:
            # Format prompt
            if system:
                formatted = f"### System:\n{system}\n\n### Instruction:\n{prompt}\n\n### Response:\n"
            else:
                formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

            inputs = self.tokenizer(formatted, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_tokens = inputs["input_ids"].shape[1]

            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else 0.1,
                    top_p=0.9,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()

            output_tokens = outputs.shape[1] - input_tokens
            elapsed = time.time() - start_time

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            return {
                'success': True,
                'response': response,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'response_time_sec': round(elapsed, 2),
                'model': self.model_name + (f" (+{self.adapter_path})" if self.adapter_path else "")
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time_sec': time.time() - start_time
            }


def get_model_backend(
    model_name: str,
    adapter_path: Optional[str] = None,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    use_4bit: bool = True,
) -> BaseModelBackend:
    """Factory function to create appropriate model backend."""
    if "claude" in model_name.lower():
        return ClaudeBackend(model_name, max_tokens, temperature)
    elif "gpt" in model_name.lower():
        return OpenAIBackend(model_name, max_tokens, temperature)
    else:
        return HuggingFaceBackend(model_name, max_tokens, temperature, adapter_path, use_4bit)


# ============================================================================
# CONFIGURATION
# ============================================================================

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
2. Distinguish between what the data shows vs. your interpretation
3. Express uncertainty appropriately when data is insufficient
4. Consider multi-omics integration where relevant
5. Avoid hallucinating statistics not in the provided context"""


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_questions(benchmark_dir: str) -> Dict[str, List[Dict]]:
    """Load questions from JSON files."""
    questions = {'clinical': [], 'transcriptomics': [], 'metabolomics': []}
    benchmark_path = Path(benchmark_dir)

    # Clinical questions
    clinical_file = benchmark_path / 'clinical_questions.json'
    if clinical_file.exists():
        with open(clinical_file) as f:
            data = json.load(f)
            for q in data.get('questions', []):
                q['modality'] = 'clinical'
                questions['clinical'].append(q)

    # Transcriptomics questions (v2)
    trans_file = benchmark_path / 'transcriptomics_questions_v2.json'
    if trans_file.exists():
        with open(trans_file) as f:
            data = json.load(f)
            for q in data.get('questions', []):
                q['modality'] = 'transcriptomics'
                questions['transcriptomics'].append(q)

    # Metabolomics questions
    metab_file = benchmark_path / 'metabolomics_questions.json'
    if metab_file.exists():
        with open(metab_file) as f:
            data = json.load(f)
            for q in data.get('questions', []):
                q['modality'] = 'metabolomics'
                questions['metabolomics'].append(q)

    return questions


def run_evaluation(
    benchmark_dir: str,
    output_dir: str = "results",
    model_name: str = "claude-sonnet-4-20250514",
    adapter_path: Optional[str] = None,
    sample_size: int = None,
    modality: str = None,
    use_4bit: bool = True,
):
    """Run full LLM evaluation."""

    # Initialize model backend
    print(f"Initializing model: {model_name}")
    if adapter_path:
        print(f"With LoRA adapter: {adapter_path}")
    backend = get_model_backend(model_name, adapter_path, use_4bit=use_4bit)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Load questions
    print("Loading questions...")
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
    print(f"\nRunning evaluation...")
    print(f"   Questions to evaluate: {len(questions_to_run)}")
    print("-" * 60)

    results = []

    for i, q in enumerate(questions_to_run):
        q_id = q.get('id', f'Q{i+1}')
        q_text = q.get('question', '')
        q_diff = q.get('difficulty', 'unknown')
        q_mod = q.get('modality', 'unknown')

        print(f"[{i+1}/{len(questions_to_run)}] {q_id} ({q_mod}, {q_diff})...", end=" ", flush=True)

        full_prompt = f"{DATA_CONTEXT}\n\n---\n\n**Question:** {q_text}\n\n**Your Answer:**"
        result = backend.generate(full_prompt, SYSTEM_PROMPT)

        if result['success']:
            print(f"OK ({result['response_time_sec']}s)")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")

        results.append({
            'question_id': q_id,
            'modality': q_mod,
            'difficulty': q_diff,
            'question': q_text,
            **result
        })

        time.sleep(0.1)  # Small delay

    # Save results
    model_suffix = model_name.replace("/", "_")
    if adapter_path:
        model_suffix += "_finetuned"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(output_dir) / f'evaluation_results_{model_suffix}_{timestamp}.json'

    output_data = {
        'metadata': {
            'model': model_name,
            'adapter_path': adapter_path,
            'timestamp': timestamp,
            'total_questions': len(results),
            'successful': sum(1 for r in results if r.get('success', False)),
            'failed': sum(1 for r in results if not r.get('success', True)),
            'total_input_tokens': backend.total_input_tokens,
            'total_output_tokens': backend.total_output_tokens,
        },
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Model: {model_name}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print(f"Questions: {output_data['metadata']['successful']}/{output_data['metadata']['total_questions']} successful")
    print(f"Tokens: {backend.total_input_tokens} input, {backend.total_output_tokens} output")
    print(f"Results saved to: {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="SpaceOmicsBench Multi-Model Evaluation")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Model name (claude-*, gpt-*, or HuggingFace path)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapters (for fine-tuned models)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample")
    parser.add_argument("--full", action="store_true",
                        help="Run all 115 questions")
    parser.add_argument("--modality", type=str, choices=["clinical", "transcriptomics", "metabolomics"],
                        help="Specific modality to test")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--benchmark-dir", type=str, default="../benchmark_questions_v2.1",
                        help="Directory containing benchmark questions")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (HuggingFace only)")

    args = parser.parse_args()

    sample_size = None if args.full else (args.sample or 10)

    run_evaluation(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        adapter_path=args.adapter_path,
        sample_size=sample_size,
        modality=args.modality,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
