#!/usr/bin/env python3
"""
SpaceOmicsBench v2 - LLM Evaluation Script
============================================

Runs LLM models on the SpaceOmicsBench question bank with dynamic data context loading.
Supports Claude, OpenAI, OpenAI-compatible (Groq, Together, DeepSeek, Ollama, …),
and HuggingFace local model backends.

Usage:
    # Claude (Anthropic API)
    export ANTHROPIC_API_KEY="your-key"
    python run_llm_evaluation.py --model claude-sonnet-4-6 --sample 5

    # OpenAI (API)
    export OPENAI_API_KEY="your-key"
    python run_llm_evaluation.py --model gpt-4o --sample 5

    # Open-source via Groq (Llama 3.3 70B — free tier)
    export GROQ_API_KEY="your-key"
    python run_llm_evaluation.py --model llama-3.3-70b-versatile \
        --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY

    # Open-source via Together.ai (Qwen 2.5 72B)
    export TOGETHER_API_KEY="your-key"
    python run_llm_evaluation.py --model Qwen/Qwen2.5-72B-Instruct-Turbo \
        --base-url https://api.together.xyz/v1 --api-key-env TOGETHER_API_KEY

    # DeepSeek R1 via DeepSeek API
    export DEEPSEEK_API_KEY="your-key"
    python run_llm_evaluation.py --model deepseek-reasoner \
        --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY

    # Ollama (fully local, no auth)
    python run_llm_evaluation.py --model llama3.3:70b \
        --base-url http://localhost:11434/v1 --api-key-env OLLAMA_API_KEY

    # HuggingFace local (Apple Silicon MPS supported)
    python run_llm_evaluation.py --model meta-llama/Llama-3.3-70B-Instruct --sample 5

    # Full evaluation (all 100 questions)
    python run_llm_evaluation.py --model claude-sonnet-4-6 --full

    # Filter by modality or difficulty
    python run_llm_evaluation.py --model claude-sonnet-4-6 --modality cross_mission
    python run_llm_evaluation.py --model claude-sonnet-4-6 --difficulty expert
"""

import json
import os
import argparse
import time
import time as time_module
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any


def retry_api_call(func, max_retries=3, base_delay=2):
    """Retry API calls with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  Retry {attempt+1}/{max_retries} after {delay}s: {e}")
            time_module.sleep(delay)


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
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            resp = retry_api_call(lambda: self.client.messages.create(**kwargs))
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
            resp = retry_api_call(lambda: self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
            ))
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


class OpenAICompatibleBackend(BaseModelBackend):
    """Backend for any OpenAI-compatible API endpoint.

    Supports: Groq, Together.ai, DeepSeek, Mistral API, OpenRouter, Ollama, etc.

    Examples:
        Groq       : base_url="https://api.groq.com/openai/v1",   api_key_env="GROQ_API_KEY"
        Together   : base_url="https://api.together.xyz/v1",       api_key_env="TOGETHER_API_KEY"
        DeepSeek   : base_url="https://api.deepseek.com/v1",       api_key_env="DEEPSEEK_API_KEY"
        Mistral    : base_url="https://api.mistral.ai/v1",         api_key_env="MISTRAL_API_KEY"
        OpenRouter : base_url="https://openrouter.ai/api/v1",      api_key_env="OPENROUTER_API_KEY"
        Ollama     : base_url="http://localhost:11434/v1",          api_key_env=None (uses "ollama")
    """

    def __init__(self, model_name: str, max_tokens: int = 2000, temperature: float = 0.3,
                 base_url: Optional[str] = None, api_key_env: Optional[str] = None):
        super().__init__(model_name, max_tokens, temperature)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        # Ollama and some local servers accept any non-empty key
        env_var = api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(env_var, "ollama")
        if api_key_env and not os.environ.get(api_key_env):
            raise ValueError(f"API key not found. Set the {api_key_env} environment variable.")
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.base_url = base_url or "(default)"

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        start = time.time()
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = retry_api_call(lambda: self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
            ))
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
                "base_url": self.base_url,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "response_time_sec": round(time.time() - start, 2)}


class HuggingFaceBackend(BaseModelBackend):
    """HuggingFace local model backend with optional LoRA adapter support."""

    def __init__(self, model_name: str, max_tokens: int = 2000, temperature: float = 0.3,
                 adapter_path: Optional[str] = None, use_4bit: bool = True,
                 trust_remote_code: bool = False):
        super().__init__(model_name, max_tokens, temperature)
        self.adapter_path = adapter_path
        self._load_model(model_name, adapter_path, use_4bit, trust_remote_code)

    def _load_model(self, model_name, adapter_path, use_4bit, trust_remote_code):
        try:
            import torch
            self.torch = torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers and torch: pip install transformers torch accelerate")

        print(f"Loading HuggingFace model: {model_name}")
        if torch.cuda.is_available():
            device_map = "auto"
            print("CUDA GPU detected.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
            use_4bit = False  # BitsAndBytes not supported on MPS
            print("Apple Silicon MPS detected — using float16, 4-bit quantization disabled.")
        else:
            device_map = "cpu"
            use_4bit = False
            print("No GPU detected, using CPU (will be slow).")

        tok_path = adapter_path if adapter_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, quantization_config=bnb, device_map=device_map, trust_remote_code=trust_remote_code)
            except ImportError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map=device_map, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
        else:
            # MPS prefers float16; CPU falls back to float32
            dt = torch.float32 if device_map == "cpu" else torch.float16
            load_kwargs: Dict[str, Any] = {
                "torch_dtype": dt, "trust_remote_code": trust_remote_code
            }
            if device_map == "mps":
                load_kwargs["low_cpu_mem_usage"] = True
            else:
                load_kwargs["device_map"] = device_map
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            if device_map == "mps":
                self.model = self.model.to("mps")

        if adapter_path:
            from peft import PeftModel
            print(f"Loading LoRA adapters from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print("Model loaded successfully!")

    def generate(self, prompt: str, system: str = None) -> Dict[str, Any]:
        start = time.time()
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback for models without chat template
                formatted = f"### System:\n{system}\n\n### Instruction:\n{prompt}\n\n### Response:\n" if system \
                    else f"### Instruction:\n{prompt}\n\n### Response:\n"

            inputs = self.tokenizer(formatted, return_tensors="pt")
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
                use_4bit: bool = True,
                backend_override: Optional[str] = None,
                trust_remote_code: bool = False,
                base_url: Optional[str] = None,
                api_key_env: Optional[str] = None) -> BaseModelBackend:
    """Factory: select backend by model name or explicit override.

    Backends:
        claude      — Anthropic API (claude-*)
        openai      — OpenAI API (gpt-*, o1/o3/o4)
        compatible  — Any OpenAI-compatible endpoint (Groq, Together, DeepSeek, Ollama, …)
        huggingface — Local inference via HuggingFace transformers (CUDA / Apple MPS / CPU)
    """
    if backend_override:
        override = backend_override.lower()
        if override == "claude":
            return ClaudeBackend(model_name, max_tokens, temperature)
        elif override == "openai":
            return OpenAIBackend(model_name, max_tokens, temperature)
        elif override == "compatible":
            return OpenAICompatibleBackend(model_name, max_tokens, temperature, base_url, api_key_env)
        elif override == "huggingface":
            return HuggingFaceBackend(model_name, max_tokens, temperature, adapter_path, use_4bit,
                                      trust_remote_code=trust_remote_code)
        else:
            raise ValueError(f"Unknown backend: {backend_override}. Use: claude, openai, compatible, huggingface")

    # If a base_url is provided, always use the compatible backend
    if base_url:
        return OpenAICompatibleBackend(model_name, max_tokens, temperature, base_url, api_key_env)

    # Auto-detect from model name prefix/keywords
    name = model_name.lower()
    if any(k in name for k in ["claude", "anthropic", "sonnet", "opus", "haiku"]):
        return ClaudeBackend(model_name, max_tokens, temperature)
    elif any(k in name for k in ["gpt", "o1", "o3", "o4", "chatgpt"]):
        return OpenAIBackend(model_name, max_tokens, temperature)
    else:
        # Open-source models default to HuggingFace local
        # (use --base-url to route to an API instead)
        return HuggingFaceBackend(model_name, max_tokens, temperature, adapter_path, use_4bit,
                                  trust_remote_code=trust_remote_code)


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
    model_name: str = "claude-sonnet-4-6",
    adapter_path: Optional[str] = None,
    output_dir: str = "results",
    sample_size: Optional[int] = None,
    modality: Optional[str] = None,
    difficulty: Optional[str] = None,
    use_4bit: bool = True,
    backend_override: Optional[str] = None,
    log_prompts: bool = False,
    trust_remote_code: bool = False,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
):
    """Run LLM evaluation on the SpaceOmicsBench question bank."""

    print("=" * 70)
    print("SpaceOmicsBench v2 - LLM Evaluation")
    print("=" * 70)

    # Init backend
    print(f"\nModel: {model_name}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    if base_url:
        print(f"Endpoint: {base_url}")
    backend = get_backend(model_name, adapter_path, use_4bit=use_4bit,
                          backend_override=backend_override,
                          trust_remote_code=trust_remote_code,
                          base_url=base_url, api_key_env=api_key_env)

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
            f"**Question:** {qtext}\n\n"
            f"**Your Answer:**"
        )

        if log_prompts:
            prompt_log_dir = Path(output_dir) / "prompt_logs"
            prompt_log_dir.mkdir(parents=True, exist_ok=True)
            (prompt_log_dir / f"{qid}.txt").write_text(
                f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n\n=== PROMPT ===\n{prompt}")

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
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
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
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of questions to sample (default: 10)")
    parser.add_argument("--full", action="store_true",
                        help="Run all 100 questions (overrides --sample)")
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
                        help="Disable 4-bit quantization (HuggingFace/CUDA only; auto-disabled on MPS)")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["claude", "openai", "compatible", "huggingface"],
                        help="Force specific backend (overrides auto-detection from model name). "
                             "Use 'compatible' for Groq/Together/DeepSeek/Ollama/OpenRouter.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for OpenAI-compatible endpoints "
                             "(e.g. https://api.groq.com/openai/v1). "
                             "Automatically selects the 'compatible' backend.")
    parser.add_argument("--api-key-env", type=str, default=None,
                        help="Environment variable name containing the API key for --base-url "
                             "(e.g. GROQ_API_KEY, TOGETHER_API_KEY). "
                             "Defaults to OPENAI_API_KEY if not set.")
    parser.add_argument("--log-prompts", action="store_true",
                        help="Save full prompts to output_dir/prompt_logs/")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow remote code execution for HuggingFace models")

    args = parser.parse_args()

    sample_size = None if args.full else args.sample

    run_evaluation(
        model_name=args.model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        sample_size=sample_size,
        modality=args.modality,
        difficulty=args.difficulty,
        use_4bit=not args.no_4bit,
        backend_override=args.backend,
        log_prompts=args.log_prompts,
        trust_remote_code=args.trust_remote_code,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )


if __name__ == "__main__":
    main()
