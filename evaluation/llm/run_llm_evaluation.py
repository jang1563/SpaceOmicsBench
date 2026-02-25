#!/usr/bin/env python3
"""
SpaceOmicsBench v2 - LLM Evaluation Script
============================================

Runs LLM models on the SpaceOmicsBench question bank with dynamic data context loading.
Supports Claude, OpenAI, and HuggingFace (local) model backends.

Usage:
    # Claude (API)
    export ANTHROPIC_API_KEY="your-key"
    python run_llm_evaluation.py --model claude-sonnet-4-20250514 --sample 5

    # OpenAI (API)
    export OPENAI_API_KEY="your-key"
    python run_llm_evaluation.py --model gpt-4o --sample 5

    # HuggingFace (local, requires GPU)
    python run_llm_evaluation.py --model mistralai/Mistral-7B-v0.3 --sample 5

    # Full evaluation (all 60 questions)
    python run_llm_evaluation.py --model claude-sonnet-4-20250514 --full

    # Filter by modality or difficulty
    python run_llm_evaluation.py --model claude-sonnet-4-20250514 --modality cross_mission
    python run_llm_evaluation.py --model claude-sonnet-4-20250514 --difficulty expert
"""

import json
import os
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CONTEXT_DIR = SCRIPT_DIR / "data_context"
QUESTION_BANK_FILE = SCRIPT_DIR / "question_bank.json"

SYSTEM_PROMPT = (
    "You are an expert in space medicine, bioinformatics, and multi-omics data analysis. "
    "You are answering questions about spaceflight biomedical data from the SpaceOmicsBench benchmark, "
    "which includes data from the Inspiration4 (I4) 3-day mission and the NASA Twins Study (340-day ISS mission).\n\n"
    "When answering:\n"
    "1. Be specific and cite relevant statistics when available from the provided context\n"
    "2. Distinguish between what the data shows versus your interpretation\n"
    "3. Express uncertainty appropriately, especially given the extremely small sample sizes (N=4 for I4, N=1 for Twins)\n"
    "4. Consider multi-omics integration and cross-mission comparisons where relevant\n"
    "5. Do not fabricate statistics, gene names, or citations not present in the provided context"
)


# ============================================================================
# MODEL BACKENDS
# ============================================================================

class BaseModelBackend:
    """Abstract base class for model backends."""

    def __init__(self, model_name: str, max_tokens: int = 2000, temperature: float = 0.3):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        raise NotImplementedError


class ClaudeBackend(BaseModelBackend):
    """Anthropic Claude API backend."""

    def __init__(self, model_name: str, max_tokens: int = 2000, temperature: float = 0.3):
        super().__init__(model_name, max_tokens, temperature)
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        import anthropic
        start = time.time()
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            resp = self.client.messages.create(**kwargs)
            elapsed = time.time() - start
            self.total_input_tokens += resp.usage.input_tokens
            self.total_output_tokens += resp.usage.output_tokens
            return {
                "success": True,
                "response": resp.content[0].text,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "response_time_sec": round(elapsed, 2),
                "model": self.model_name,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "response_time_sec": round(time.time() - start, 2)}


class OpenAIBackend(BaseModelBackend):
    """OpenAI API backend."""

    def __init__(self, model_name: str, max_tokens: int = 2000, temperature: float = 0.3):
        super().__init__(model_name, max_tokens, temperature)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        start = time.time()
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
            )
            elapsed = time.time() - start
            inp = resp.usage.prompt_tokens if resp.usage else 0
            out = resp.usage.completion_tokens if resp.usage else 0
            self.total_input_tokens += inp
            self.total_output_tokens += out
            return {
                "success": True,
                "response": resp.choices[0].message.content,
                "input_tokens": inp,
                "output_tokens": out,
                "response_time_sec": round(elapsed, 2),
                "model": self.model_name,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "response_time_sec": round(time.time() - start, 2)}


class HuggingFaceBackend(BaseModelBackend):
    """HuggingFace local model backend with optional LoRA adapter support."""

    def __init__(self, model_name: str, max_tokens: int = 2000, temperature: float = 0.3,
                 adapter_path: Optional[str] = None, use_4bit: bool = True):
        super().__init__(model_name, max_tokens, temperature)
        self.adapter_path = adapter_path
        self._load_model(model_name, adapter_path, use_4bit)

    def _load_model(self, model_name, adapter_path, use_4bit):
        try:
            import torch
            self.torch = torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers and torch: pip install transformers torch accelerate")

        print(f"Loading HuggingFace model: {model_name}")
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            use_4bit = False
            print("No GPU detected, using CPU (will be slow)")

        tok_path = adapter_path if adapter_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, quantization_config=bnb, device_map=device_map, trust_remote_code=True)
            except ImportError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map=device_map, torch_dtype=torch.float16, trust_remote_code=True)
        else:
            dt = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device_map, torch_dtype=dt, trust_remote_code=True)

        if adapter_path:
            from peft import PeftModel
            print(f"Loading LoRA adapters from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print("Model loaded successfully!")

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        start = time.time()
        try:
            fmt = f"### System:\n{system}\n\n### Instruction:\n{prompt}\n\n### Response:\n" if system \
                else f"### Instruction:\n{prompt}\n\n### Response:\n"
            inputs = self.tokenizer(fmt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            n_in = inputs["input_ids"].shape[1]

            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=self.max_tokens,
                    temperature=max(self.temperature, 0.1),
                    top_p=0.9, do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id)

            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            if "### Response:" in text:
                text = text.split("### Response:")[-1].strip()

            n_out = out.shape[1] - n_in
            elapsed = time.time() - start
            self.total_input_tokens += n_in
            self.total_output_tokens += n_out
            return {
                "success": True, "response": text,
                "input_tokens": n_in, "output_tokens": n_out,
                "response_time_sec": round(elapsed, 2),
                "model": self.model_name + (f" (+{self.adapter_path})" if self.adapter_path else ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "response_time_sec": round(time.time() - start, 2)}


def get_backend(model_name: str, adapter_path: Optional[str] = None,
                max_tokens: int = 2000, temperature: float = 0.3,
                use_4bit: bool = True) -> BaseModelBackend:
    """Factory: pick Claude / OpenAI / HuggingFace backend by model name."""
    name = model_name.lower()
    if "claude" in name or "anthropic" in name:
        return ClaudeBackend(model_name, max_tokens, temperature)
    elif "gpt" in name or "o1" in name or "o3" in name:
        return OpenAIBackend(model_name, max_tokens, temperature)
    else:
        return HuggingFaceBackend(model_name, max_tokens, temperature, adapter_path, use_4bit)


# ============================================================================
# DATA CONTEXT LOADING
# ============================================================================

def load_data_context(context_files: List[str]) -> str:
    """Load and concatenate markdown data context files."""
    parts = []
    for fname in context_files:
        fpath = DATA_CONTEXT_DIR / fname
        if fpath.exists():
            parts.append(fpath.read_text().strip())
        else:
            parts.append(f"[Context file not found: {fname}]")
    return "\n\n---\n\n".join(parts)


def load_question_bank(
    modality: Optional[str] = None,
    difficulty: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> List[Dict]:
    """Load questions from question_bank.json with optional filters."""
    with open(QUESTION_BANK_FILE) as f:
        bank = json.load(f)

    questions = bank["questions"]

    if modality:
        questions = [q for q in questions if q["modality"] == modality]
    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]

    if sample_size and sample_size < len(questions):
        import random
        random.seed(42)
        questions = random.sample(questions, sample_size)

    return questions


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation(
    model_name: str = "claude-sonnet-4-20250514",
    adapter_path: Optional[str] = None,
    output_dir: str = "results",
    sample_size: Optional[int] = None,
    modality: Optional[str] = None,
    difficulty: Optional[str] = None,
    use_4bit: bool = True,
):
    """Run LLM evaluation on the SpaceOmicsBench question bank."""

    print("=" * 70)
    print("SpaceOmicsBench v2 - LLM Evaluation")
    print("=" * 70)

    # Init backend
    print(f"\nModel: {model_name}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    backend = get_backend(model_name, adapter_path, use_4bit=use_4bit)

    # Load questions
    questions = load_question_bank(modality, difficulty, sample_size)
    print(f"Questions to evaluate: {len(questions)}")
    if modality:
        print(f"  Modality filter: {modality}")
    if difficulty:
        print(f"  Difficulty filter: {difficulty}")
    print("-" * 70)

    # Evaluate
    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        qmod = q["modality"]
        qdiff = q["difficulty"]
        qtext = q["question"]

        print(f"[{i+1}/{len(questions)}] {qid} ({qmod}/{qdiff})...", end=" ", flush=True)

        # Build prompt with dynamic context
        context = load_data_context(q.get("data_context_files", ["overview.md"]))
        prompt = (
            f"## Data Context\n\n{context}\n\n"
            f"---\n\n"
            f"**Question ({qdiff}):** {qtext}\n\n"
            f"**Your Answer:**"
        )

        result = backend.generate(prompt, SYSTEM_PROMPT)

        if result["success"]:
            print(f"OK ({result['response_time_sec']}s, "
                  f"{result.get('output_tokens', '?')} tokens)")
        else:
            print(f"FAILED: {result.get('error', 'Unknown')}")

        results.append({
            "question_id": qid,
            "modality": qmod,
            "difficulty": qdiff,
            "category": q.get("category", ""),
            "question": qtext,
            "data_context_files": q.get("data_context_files", []),
            **result,
        })

        time.sleep(0.1)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_suffix = model_name.replace("/", "_").replace(":", "_")
    if adapter_path:
        model_suffix += "_finetuned"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = Path(output_dir) / f"eval_{model_suffix}_{ts}.json"

    output = {
        "metadata": {
            "benchmark": "SpaceOmicsBench_v2",
            "model": model_name,
            "adapter_path": adapter_path,
            "timestamp": ts,
            "total_questions": len(results),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success", True)),
            "total_input_tokens": backend.total_input_tokens,
            "total_output_tokens": backend.total_output_tokens,
            "filters": {"modality": modality, "difficulty": difficulty, "sample_size": sample_size},
        },
        "results": results,
    }

    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Model:     {model_name}")
    print(f"Questions: {output['metadata']['successful']}/{output['metadata']['total_questions']} successful")
    print(f"Tokens:    {backend.total_input_tokens:,} in / {backend.total_output_tokens:,} out")
    print(f"Output:    {outfile}")

    return str(outfile)


def main():
    parser = argparse.ArgumentParser(description="SpaceOmicsBench v2 LLM Evaluation")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Model name (claude-*, gpt-*, or HuggingFace path)")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to LoRA adapters (HuggingFace fine-tuned models)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample (default: 10)")
    parser.add_argument("--full", action="store_true",
                        help="Run all 60 questions (overrides --sample)")
    parser.add_argument("--modality", type=str, default=None,
                        choices=["clinical", "transcriptomics", "proteomics", "metabolomics",
                                 "spatial", "microbiome", "cross_mission", "multi_omics", "methods"],
                        help="Filter by modality")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard", "expert"],
                        help="Filter by difficulty level")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (HuggingFace only)")

    args = parser.parse_args()

    sample_size = None if args.full else (args.sample or 10)

    run_evaluation(
        model_name=args.model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        sample_size=sample_size,
        modality=args.modality,
        difficulty=args.difficulty,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
