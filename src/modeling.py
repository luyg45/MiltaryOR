"""
建模层模块 —— ATF-8B 自动形式化 + Lean 4 验证反馈循环。

重构说明：
  - 验证逻辑已抽取到 lean4_verifier.py
  - 工具函数已抽取到 lean4_utils.py
  - 本文件仅保留建模层业务逻辑
"""

import os
import time
import json
import signal
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import importlib.util

# 使用国内镜像加速 Hugging Face 模型下载
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# 启用 PyTorch 可扩展内存段，减少显存碎片化导致的 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lean4_verifier import HAS_VERIFIER, verify_batch, verify_single
from lean4_utils import (
    extract_lean_code,
    strip_lean_header,
    format_errors_for_modeling,
    postprocess_modeling_output,
    enhance_error_feedback_for_modeling,
)

from config import CFG

# 中断标志：SIGINT 处理器设置此标志，主循环在安全点检查并退出
_interrupt_requested = False


def _sigint_handler(signum, frame):
    """SIGINT 信号处理器：设置中断标志，不抛异常，确保当前操作完成后再退出。"""
    global _interrupt_requested
    _interrupt_requested = True
    print("\n⚠️  收到中断信号，将在当前题目完成后优雅退出...")


# ────────────────────────────────────────────────────────────
# 常量 & 配置（从 config.py 统一读取）
# ────────────────────────────────────────────────────────────

MODEL_DIR = CFG.MODELING_MODEL_DIR
DEFAULT_DATASET = Path(CFG.DEFAULT_DATASET)
DEFAULT_SUMMARY_FILE = Path(CFG.DEFAULT_SUMMARY_FILE)
SOLVE_SCRIPT_PATH = Path(__file__).resolve().parent / "solver.py"

SYSTEM_PROMPT = '''
You are an expert in mathematics and Lean 4. Your task is to convert natural language problems into valid Lean 4 formal statements (Compatible with Lean 4 v4.9).

Your code must begin with:

import Mathlib
import Aesop

open Real Finset BigOperators


CRITICAL Lean 4 Coding Rules (MUST follow to avoid compilation errors):

1. ALWAYS include `open Real Finset BigOperators` after imports.
2. Allocation vectors MUST be typed as `Fin n → ℕ` (NOT plain `ℕ` or `List`).
   Example: `(x : Fin 4 → ℕ)`
3. When summing over Fin n, use `∑ i : Fin n, f i` (requires `open Finset BigOperators`).
4. For probability expressions like `(1 - p) ^ k`:
   - Ensure `p` is `ℝ` and `k` is `ℕ` (NOT `Fin n`).
   - If `x : Fin n → ℕ`, use `(x i : ℕ)` directly — it is already `ℕ`, no cast needed.
   - NEVER write `(x i : Fin n)` in arithmetic — `Fin n` is NOT a number type.
5. For HPow/HMul: always ensure operand types are explicit.
   - `(1 - p) ^ (x i)` is valid when `p : ℝ` and `x i : ℕ`.
   - If you need real exponentiation, cast: `(↑(x i) : ℝ)` or use `Nat.cast`.
   - ⚠️ CRITICAL: NEVER cast ℕ exponent to ℝ! This triggers `Real.rpow` which is NOT computable.
     ✗ `(0.9 : ℝ) ^ (x₁ : ℝ)` — WRONG, uses rpow, breaks norm_num
     ✓ `(0.9 : ℝ) ^ x₁` where `x₁ : ℕ` — CORRECT, uses computable HPow
     ✗ `(0.9 : ℝ) ^ (↑x₁ : ℝ)` — WRONG, same rpow problem
     ✓ `(0.9 : ℝ) ^ (x₁ : ℕ)` — CORRECT, keeps ℕ exponent
6. Threat values and probabilities should be defined as local `let` bindings with explicit types:
   - `let threat : Fin n → ℝ := ![v₁, v₂, ...]`
   - `let prob : Fin n → ℝ := ![p₁, p₂, ...]`
7. Use `theorem my_favorite_theorem` as the theorem name.
8. The theorem body should use `sorry` as placeholder proof.
9. Do NOT use semicolons (`;`) between tactics or statements — use newlines.
10. Do NOT use deprecated syntax like `#check` or `#eval` in theorem files.

Optimality proof pattern (for proving a candidate solution is optimal):
When the problem asks to prove (x₁*, x₂*, ...) is optimal under constraints:
- Use plain ℕ variables (x₁ x₂ x₃ : ℕ), NOT Fin n → ℕ
- The benefit function F should use ℕ exponents: `(0.9 : ℝ) ^ x₁` (NOT `^ (x₁ : ℝ)`)
- Structure: `∀ x₁ x₂ x₃ : ℕ, x₁ + x₂ + x₃ = n → F_opt ≥ F x₁ x₂ x₃`
- The `xi ≥ 0` condition is redundant for ℕ — omit it
- Example:
  ```lean4
  theorem my_favorite_theorem :
    let F : ℕ → ℕ → ℕ → ℝ := fun x₁ x₂ x₃ =>
      3 * (1 - (0.9 : ℝ) ^ x₁) + 2 * (1 - (0.95 : ℝ) ^ x₂) + 1 * (1 - (0.8 : ℝ) ^ x₃)
    ∀ x₁ x₂ x₃ : ℕ, x₁ + x₂ + x₃ = 10 →
      F 5 2 3 ≥ F x₁ x₂ x₃ := by sorry
  ```

CRITICAL modeling correctness rules (avoid false statements):

11. Inequality direction: When problem data shows two values are EQUAL, use `≥` or `≤` (not strict `>` or `<`).
    Example: If x₃ = 2 and x₄ = 2, write `x₃ ≥ x₄` NOT `x₃ > x₄`.
12. Universal quantification: Do NOT use `∀ k1 k2 : ℕ, ...` for multi-stage allocation theorems
    when the problem specifies CONCRETE allocation values. Universal statements are often FALSE
    (e.g., k=0 makes probability formulas trivially fail).
    Instead, use concrete `let` bindings:
    ```lean4
    let k1 : ℕ := 3
    let k2 : ℕ := 2
    ```
13. Residual threat formula: For multi-stage interception, residual = v * (1-p₁)^k₁ * (1-p₂)^k₂.
    Do NOT accidentally swap multiplication order or omit stages.
14. Feasibility proofs: When proving an allocation is feasible (satisfies constraints),
    verify that ALL inequalities in the theorem are actually TRUE for the given data.
    A common mistake is asserting strict inequality where equality holds.

15. When summing/producting over Finset.range:
    ✗ `∑ k in Finset.range n, f k` — WRONG (Lean 3 syntax, causes 'unexpected token in')
    ✓ `∑ k ∈ Finset.range n, f k` — CORRECT (use ∈ not 'in')
    ✓ `∑ k : Fin n, f k` — ALSO CORRECT (preferred when indexing 0..n-1)
    Same rule applies to `∏` (Finset.prod).

16. For inner products in EuclideanSpace:
    ✗ `⟪a, b⟫_ℝ` — WRONG (subscript _ℝ causes parse error)
    ✓ `@inner ℝ _ _ a b` — CORRECT (explicit application)
    ✓ `inner (𝕜 := ℝ) a b` — CORRECT (named type argument)
    For simple 2D geometry, prefer coordinate formulation with `(ℝ × ℝ)` and
    dot product as `a.1*b.1 + a.2*b.2` to avoid EuclideanSpace complexity.

17. Finset nonemptiness for sup'/inf':
    ✗ `Finset.filter_nonempty` — DOES NOT EXIST in Mathlib
    ✓ `Finset.univ_nonempty` — for proving `Finset.univ.Nonempty` when `[Nonempty α] [Fintype α]`
    ✓ Prove manually: `⟨x, Finset.mem_filter.mpr ⟨Finset.mem_univ _, proof⟩⟩`
    ✓ For `Finset.univ.sup'`/`inf'`, the nonemptiness proof is `Finset.univ_nonempty`.

18. Fin n arithmetic pitfalls:
    ✗ `q.2 + 1` when `q.2 : Fin n` and `n` is a variable — `OfNat (Fin n) 1` fails to synthesize
    ✓ Work with `.val` (natural number): `q.2.val + 1`
    ✓ Or add hypothesis `(hn : n ≥ 2)` and construct explicitly: `⟨1, by omega⟩ : Fin n`
    ✓ For Fin arithmetic, prefer `Fin.val` extraction + ℕ operations.

19. DecidablePred / DecidableRel:
    When using `Finset.filter` with a custom predicate (e.g., `fun x => R x y`),
    Lean needs a `Decidable` instance. Solutions:
    ✗ Bare `Finset.filter (fun x => custom_prop x)` — fails without decidability
    ✓ Add `[DecidablePred P]` or `[DecidableRel R]` to theorem parameters
    ✓ Or prefix with `open Classical in` before the theorem

20. Finset vs Set type mismatch:
    ✗ `(⋃ i, finset_expr i) = some_finset` — ⋃ returns Set, not Finset
    ✓ Use `Finset.biUnion` for Finset operations
    ✓ Or convert: `(some_finset : Finset α).toSet` when comparing with Set
    ✓ For partition proofs, use `Finset.disjiUnion` or state membership properties directly.

21. ℕ+ (PNat) vs ℕ coercion:
    ✗ Assigning `ℕ+` value where `ℕ` is expected without cast
    ✓ Use `(x : ℕ)` or `x.val` to convert ℕ+ to ℕ
    ✓ Or define the function with `ℕ` from the start if ℕ+ is unnecessary.

Verification workflow:

- Analyze the problem and create initial Lean 4 statement
- Call syntax_check to verify compilation
- If syntax check passes, call consistency_check
- If any check fails, analyze errors, modify code and restart verification
- Repeat until BOTH checks pass
'''.strip()

MODELING_FIX_PROMPT_TEMPLATE = '''
Your previous Lean 4 formalization has compilation errors. Please fix the code based on the error messages below.

Original problem:
{problem_text}

Your previous Lean 4 code:
```lean4
{lean_code}
```

Compilation errors:
{error_feedback}

Common Fix Patterns (apply the relevant ones):

[Fin/ℕ type mismatch]
- If error mentions "Fin" vs "ℕ" mismatch: `x : Fin n → ℕ` means `x i` is already `ℕ`.
  Do NOT cast `x i` to `Fin`. Use `(x i)` directly in `^` exponent.
- `(1 - p) ^ (x i)` is correct when `p : ℝ`, `x i : ℕ`.

[HMul/HPow/failed to synthesize]
- Add `open Real Finset BigOperators` after imports.
- Ensure all numeric literals used with ℝ operations have explicit type:
  e.g., `(0.42 : ℝ)` not just `0.42`.
- For `threat * (1 - (1 - p)^k)`, ensure threat is `ℝ` and k is `ℕ`.

[rpow / Real.rpow errors]
- If error mentions `rpow` or `Real.rpow`, you have cast the exponent to ℝ by mistake.
- Fix: change `^ (x₁ : ℝ)` to `^ x₁` where `x₁ : ℕ`.
- The exponent in `(base : ℝ) ^ exp` must stay as `ℕ`, NOT be cast to `ℝ`.

[Syntax errors: unexpected token/semicolons]
- Remove all `;` between statements. Use newlines instead.
- Each `let` binding must be followed by a newline, not `;`.
- Ensure `sorry` is on its own line or after `:= by`.

[Inhabited / failed to synthesize Inhabited]
- If error mentions "failed to synthesize Inhabited <Type>", it means you used `list[i]!` (List.get!)
  on a custom inductive type without an Inhabited instance.
- Fix: add `instance : Inhabited <Type> := \u27e8<default_constructor>\u27e9` after the inductive definition.
- Example: `instance : Inhabited State := \u27e8State.Search\u27e9`

[Statement is FALSE / unprovable]
- If the theorem cannot be proven because the statement itself is false:
  - Check inequality direction: use `≥` instead of `>` when values might be equal.
  - Check universal quantification: replace `∀ k : ℕ, ...` with concrete values when the problem specifies them.
  - Verify numeric values match the problem exactly.

[unexpected token 'in' with ∑/∏ over Finset.range]
- `∑ k in Finset.range n, f k` is Lean 3 syntax and INVALID in Lean 4.
- Fix: Replace `in` with `∈`: `∑ k ∈ Finset.range n, f k`
- Or rewrite as: `∑ k : Fin n, f ↑k` (preferred for simple ranges)

[Unknown constant 'Finset.filter_nonempty']
- This API does not exist in current Mathlib.
- For `Finset.univ.sup'`/`inf'`, use `Finset.univ_nonempty` as the nonemptiness proof.
- For filtered sets, prove nonemptiness manually: `⟨elem, mem_proof⟩`.

[DecidablePred / failed to synthesize DecidablePred]
- Lean cannot auto-derive decidability for custom propositions in Finset.filter.
- Fix: Add `[DecidableRel R]` or `[DecidablePred P]` to the theorem parameters.
- Or: Add `open Classical in` before the theorem declaration.

[OfNat (Fin n) literal — numeral cannot be used]
- When `n` is a variable, `Fin n` has no `OfNat` instance for numerals.
- Fix: Use `.val` to work with ℕ: replace `p.2 + 1` with `p.2.val + 1`
- Or construct explicitly: `(⟨1, by omega⟩ : Fin n)` (requires proof that n ≥ 2).

[Type mismatch: ℕ+ vs ℕ]
- `ℕ+` (PNat) is not the same as `ℕ`. Cast with `(x : ℕ)` or use `x.val`.
- If the function expects `ℕ`, define inputs as `ℕ` with positivity hypotheses.

[Type mismatch: Finset vs Set (⋃ / Set operations)]
- `⋃` operates on `Set`, not `Finset`. They are different types.
- Fix: Use `Finset.biUnion` for Finset union, or convert via `.toSet`.
- For partition equality, state as: `∀ x, x ∈ T ↔ ∃ i, x ∈ P i`.

[Inner product ⟪...⟫_ℝ parse error]
- The subscript `_ℝ` after `⟫` causes a parse error.
- Fix: Use `@inner ℝ _ _ a b` or `inner (𝕜 := ℝ) a b`.
- Better: For 2D geometry, use coordinate pairs `(ℝ × ℝ)` with explicit dot product.

[Missing cases in pattern match]
- If error says "Missing cases: (Nat.succ ...)" — your match is not exhaustive.
- Fix: Add a wildcard case `| _ => ...` or enumerate all missing constructors.

Instructions:
- Fix ALL errors listed above
- Keep `import Mathlib` and `import Aesop` at the top
- MUST include `open Real Finset BigOperators` after imports
- Allocation vectors: `Fin n → ℕ` for integer allocations, `Fin n → ℝ` for real values
- Use `theorem my_favorite_theorem` as the theorem name
- Output ONLY the complete fixed Lean 4 code, no explanations
'''.strip()


# ────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────

def get_formal_statement_prompt(informal_problem: str) -> str:
    prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
    prompt += informal_problem
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Batch modeling -> verify -> solve pipeline.")
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_DATASET), help="Input JSON/JSONL dataset.")
    parser.add_argument("--batch-size", type=int, default=CFG.MODELING_BATCH_SIZE, help="Batch size for parallel generation.")
    parser.add_argument("--max-new-tokens", type=int, default=CFG.MODELING_MAX_NEW_TOKENS, help="Max tokens per sample generation.")
    parser.add_argument("--temperature", type=float, default=CFG.MODELING_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only process first N records.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip first N records (0-based).")
    parser.add_argument("--record-id", type=str, default=None, help="Only process record with this ID.")
    parser.add_argument("--server-timeout", type=int, default=600, help="Lean verify timeout seconds.")
    parser.add_argument("--summary-file", type=str, default=str(DEFAULT_SUMMARY_FILE), help="Batch summary JSON output.")
    parser.add_argument("--solver-max-retries", type=int, default=CFG.SOLVER_MAX_RETRIES, help="Max retries for solve stage.")
    parser.add_argument("--modeling-max-retries", type=int, default=CFG.MODELING_MAX_RETRIES, help="Max retries for modeling verification.")
    parser.add_argument("--sampling-budget", type=int, default=CFG.SOLVER_SAMPLING_BUDGET, help="采样预算：每道题独立生成多少个候选证明（Pass@k 中的 k，默认1=不多次采样）")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset only, skip model/solve.")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有断点文件，从头开始实验。")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────

def load_records(input_file: Path):
    if not input_file.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_file}")

    suffix = input_file.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with input_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL 第 {line_no} 行格式错误: {e}") from e
        return records

    if suffix == ".json":
        data = json.loads(input_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("records"), list):
                return data["records"]
            return [data]
        raise ValueError("不支持的 JSON 结构，期望 list/dict/records。")

    raise ValueError(f"不支持的输入后缀: {suffix}，仅支持 .json/.jsonl")


def extract_problem_text(record):
    for key in ("problem_text", "informal_problem", "question", "prompt", "lean_code"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), key
    for path_key in ("lean_file_mirror", "lean_file_formalized", "lean_file"):
        path_value = record.get(path_key)
        if isinstance(path_value, str) and path_value.strip():
            candidate = Path(path_value).expanduser()
            if candidate.exists():
                return candidate.read_text(encoding="utf-8").strip(), path_key
    raise ValueError("记录缺少可用题面字段。")


def batch_iter(items: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ────────────────────────────────────────────────────────────
# 断点续存
# ────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path) -> List[Dict[str, Any]]:
    """加载断点续存文件，返回已完成的 details 列表。"""
    if not checkpoint_path.exists():
        return []
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        details = data.get("details", [])
        print(f"📂 加载断点文件: {checkpoint_path} （已完成 {len(details)} 题）")
        return details
    except Exception as e:
        print(f"⚠️  断点文件加载失败: {e}，从头开始")
        return []


def save_checkpoint(checkpoint_path: Path, details: List[Dict[str, Any]]):
    """原子写入断点续存文件（先写临时文件再重命名，防止写入中断导致损坏）。"""
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps({"details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(str(tmp_path), str(checkpoint_path))


# ────────────────────────────────────────────────────────────
# 求解器桥接
# ────────────────────────────────────────────────────────────

def load_solver_bridge() -> Tuple[Any, Any]:
    if not SOLVE_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"找不到求解脚本: {SOLVE_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("solve_bridge", str(SOLVE_SCRIPT_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载求解脚本: {SOLVE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve_lean_code = getattr(module, "solve_lean_code")
    get_solver_llm = getattr(module, "get_solver_llm")
    return solve_lean_code, get_solver_llm


# ────────────────────────────────────────────────────────────
# 建模层反馈循环
# ────────────────────────────────────────────────────────────

def _run_modeling_feedback_loop(
    item: Dict[str, Any],
    lean_code: str,
    passed: bool,
    info: dict,
    model,
    tokenizer,
    args,
) -> Tuple[str, bool, dict, int, int, int]:
    """
    建模层反馈修复循环。

    返回 (final_code, final_passed, final_info, attempts, 
          total_fix_input_tokens, total_fix_output_tokens)
    """
    modeling_attempts = 1
    current_lean_code = lean_code
    current_passed = passed
    current_info = info
    fix_in_tokens = 0
    fix_out_tokens = 0

    while not current_passed and modeling_attempts < args.modeling_max_retries:
        modeling_attempts += 1
        error_feedback = format_errors_for_modeling(current_info)
        # 智能追加针对性修复提示
        error_feedback = enhance_error_feedback_for_modeling(error_feedback)
        print(
            f"  [{item['id']}] 建模验证失败，第 {modeling_attempts}/{args.modeling_max_retries} 轮修复..."
        )

        fix_content = MODELING_FIX_PROMPT_TEMPLATE.format(
            problem_text=item["problem_text"],
            lean_code=current_lean_code,
            error_feedback=error_feedback,
        )
        fix_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fix_content},
        ]
        fix_prompt = tokenizer.apply_chat_template(
            fix_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        fix_inputs = tokenizer(
            [fix_prompt], return_tensors="pt", padding=True, truncation=True,
            max_length=getattr(CFG, "MODELING_MAX_SEQ_LEN", 4096),
        )
        fix_prompt_len = int(fix_inputs["input_ids"].shape[1])
        fix_in_tokens += fix_prompt_len
        if torch.cuda.is_available():
            fix_inputs = {k: v.cuda() for k, v in fix_inputs.items()}
            torch.cuda.empty_cache()

        with torch.no_grad():
            fix_outputs = model.generate(
                **fix_inputs,
                temperature=args.temperature,
                top_p=0.95,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        fix_out_tokens += int(fix_outputs.shape[1]) - fix_prompt_len
        fix_response = tokenizer.decode(
            fix_outputs[0][fix_prompt_len:], skip_special_tokens=True
        )
        new_lean_code = extract_lean_code(fix_response)

        # 释放修复轮次的 GPU 张量
        del fix_outputs, fix_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not new_lean_code:
            print(f"  [{item['id']}] 修复输出为空，停止重试")
            break

        # 对修复输出也执行后处理清洗
        new_lean_code = postprocess_modeling_output(new_lean_code)
        current_lean_code = new_lean_code

        # 重新验证
        if HAS_VERIFIER:
            try:
                re_passed, re_info = verify_single(
                    current_lean_code, server_timeout=args.server_timeout
                )
                if re_info is not None:
                    current_passed = re_passed
                    current_info = re_info
                else:
                    break
            except Exception as ve:
                print(f"  [{item['id']}] 重新验证异常: {ve}")
                break
        else:
            break

    return current_lean_code, current_passed, current_info, modeling_attempts, fix_in_tokens, fix_out_tokens


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def run_batch_pipeline(args) -> Dict[str, Any]:
    input_file = Path(args.input_file).expanduser().resolve()
    records = load_records(input_file)
    # 按 ID 筛选
    if args.record_id:
        records = [r for r in records if str(r.get("id", "")) == args.record_id]
        if not records:
            raise ValueError(f"未找到 ID 为 '{args.record_id}' 的记录")
    # 按索引切片
    if args.start_index > 0:
        records = records[args.start_index:]
    if args.max_samples is not None:
        records = records[:args.max_samples]

    print(f"数据文件: {input_file}")
    print(f"待处理记录总数: {len(records)}")
    if args.dry_run:
        return {"total": len(records), "modeled_ok": 0, "verified_ok": 0, "solved_ok": 0, "details": []}

    # ── 断点续存 ──
    checkpoint_path = Path(args.summary_file).expanduser().with_name(
        Path(args.summary_file).stem + "_checkpoint.json"
    )
    if not args.no_resume:
        details = load_checkpoint(checkpoint_path)
    else:
        details = []
        if checkpoint_path.exists():
            print("🔄 --no-resume 模式，忽略已有断点文件")
    completed_ids = {d["id"] for d in details}
    remaining = [r for r in records if str(r.get("id", "")) not in completed_ids]
    if completed_ids:
        print(f"⏭  跳过已完成: {len(completed_ids)} 题，剩余: {len(remaining)} 题")
    else:
        remaining = records
    if not remaining and completed_ids:
        print("✅ 所有题目已在断点文件中，无需继续处理")
        n = len(details)
        return {
            "total": len(records),
            "modeled_ok": sum(1 for d in details if d.get("modeling_status") == "success"),
            "verified_ok": sum(1 for d in details if d.get("verify_status") == "passed"),
            "solved_ok": sum(1 for d in details if d.get("solve_status") == "success"),
            "failed": len(records) - sum(1 for d in details if d.get("solve_status") == "success"),
            "details": details, "resumed": True,
        }

    if not torch.cuda.is_available():
        print("⚠️  警告：未检测到 CUDA GPU，模型将在 CPU 上运行。")

    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("正在加载模型（建模层）...")
    if torch.cuda.is_available():
        _modeling_gpus = getattr(CFG, "MODELING_GPU_IDS", [0, 1])
        _mem_limit = getattr(CFG, "MODELING_MAX_MEM_PER_GPU", 10)
        _num_gpus = torch.cuda.device_count()
        # 建模层专用卡，其他卡给求解层，完全不重叠
        _max_mem = {}
        for _i in range(_num_gpus):
            if _i in _modeling_gpus:
                _max_mem[_i] = f"{_mem_limit}GiB"
            else:
                _max_mem[_i] = "0GiB"
        print(f"  建模层 GPU: {_modeling_gpus}  每卡限 {_mem_limit}GiB  (求解层用: {[i for i in range(_num_gpus) if i not in _modeling_gpus]})")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.float16,
            device_map="auto",
            max_memory=_max_mem,
            trust_remote_code=True,
            local_files_only=True,
        )
        # 加载后验证实际 GPU 分布
        if hasattr(model, "hf_device_map"):
            _used_devices = set(str(v) for v in model.hf_device_map.values())
            print(f"  ✅ 建模层实际分布: {_used_devices}")
            for _i in range(_num_gpus):
                if _i not in _modeling_gpus:
                    if any(f"cuda:{_i}" in d for d in _used_devices):
                        print(f"  ⚠️  警告: 建模层意外占用 GPU {_i}!")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
        )

    solve_lean_code, get_solver_llm = load_solver_bridge()
    solver_llm = get_solver_llm()

    # ── 中断处理：SIGINT 设置标志而非抛异常，确保当前操作完成后再退出 ──
    global _interrupt_requested
    _interrupt_requested = False
    _original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    # 从断点恢复已有计数器
    modeled_ok = sum(1 for d in details if d.get("modeling_status") == "success")
    verified_ok = sum(1 for d in details if d.get("verify_status") == "passed")
    solved_ok = sum(1 for d in details if d.get("solve_status") == "success")
    total_elapsed = sum(d.get("elapsed_seconds", 0) for d in details)
    total_modeling_in = sum(d.get("modeling_tokens", {}).get("input", 0) for d in details)
    total_modeling_out = sum(d.get("modeling_tokens", {}).get("output", 0) for d in details)
    total_modeling_fix_in = sum(d.get("modeling_tokens", {}).get("fix_input", 0) for d in details)
    total_modeling_fix_out = sum(d.get("modeling_tokens", {}).get("fix_output", 0) for d in details)
    total_solve_prompt = sum(d.get("solve_tokens", {}).get("prompt", 0) for d in details)
    total_solve_completion = sum(d.get("solve_tokens", {}).get("completion", 0) for d in details)
    total_solve_elapsed = 0.0  # 求解层单独耗时（断点中未存储，仅统计新处理部分）
    total_modeling_elapsed = sum(d.get("modeling_elapsed_seconds", 0) for d in details)  # 建模层单独耗时
    records_with_time = len(details)  # 断点中已有的记录数
    # ── Pass@k 统计计数器 ──
    pass_at_k_count = 0  # Pass@budget 成功的题目数（进入求解且至少1个采样通过）
    first_pass_distribution = []  # 记录每道题首次成功的采样索引，用于计算 Pass@1, @4, @8 等

    try:
      for batch_idx, raw_batch in enumerate(batch_iter(remaining, args.batch_size), 1):
        prepared_items = []
        for local_idx, record in enumerate(raw_batch):
            item_id = str(record.get("id", f"batch{batch_idx}_item{local_idx}"))
            try:
                problem_text, _ = extract_problem_text(record)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": get_formal_statement_prompt(problem_text)},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                task_type = record.get("task_type", "")
                prepared_items.append({"id": item_id, "problem_text": problem_text, "prompt_text": prompt_text, "task_type": task_type})
            except Exception as e:
                details.append({
                    "id": item_id, "modeling_status": "failed",
                    "verify_status": "skipped", "solve_status": "skipped",
                    "error": f"题面提取失败: {e}",
                })

        if not prepared_items or _interrupt_requested:
            continue

        prompt_texts = [x["prompt_text"] for x in prepared_items]
        _max_seq = getattr(CFG, "MODELING_MAX_SEQ_LEN", 4096)
        batch_inputs = tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=_max_seq,
        )
        prompt_len = int(batch_inputs["input_ids"].shape[1])
        seq_lengths = batch_inputs["attention_mask"].sum(dim=1).tolist()
        print(
            f"[Batch {batch_idx}] size={len(prepared_items)}, "
            f"padding_side={tokenizer.padding_side}, max_prompt_tokens={prompt_len}, "
            f"min_len={min(seq_lengths)}, max_len={max(seq_lengths)}"
        )

        if torch.cuda.is_available():
            batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}
            torch.cuda.empty_cache()

        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                temperature=args.temperature,
                top_p=0.95,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        responses = []
        for i in range(outputs.shape[0]):
            text = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
            responses.append(text)

        # 在释放 outputs 之前捕获输出 token 统计
        shared_output_tokens = max(int(outputs.shape[1]) - prompt_len, 0)

        # 释放 generate 产生的 KV cache / 输出张量，减少显存碎片
        del outputs, batch_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        lean_codes = [extract_lean_code(resp) for resp in responses]
        # 建模层后处理清洗（去重import、修复Lean3语法等）
        lean_codes = [postprocess_modeling_output(c) if c else c for c in lean_codes]
        for code in lean_codes:
            if code:
                modeled_ok += 1

        # ----- 逐样本建模 token 统计（含 padding 下仍近似准确）-----
        per_sample_input_tokens = {}
        for idx, seq_len in enumerate(seq_lengths):
            per_sample_input_tokens[idx] = int(seq_len)

        # ===== 批量验证 =====
        try:
            verify_results = verify_batch(lean_codes, server_timeout=args.server_timeout)
        except Exception as e:
            verify_results = [{
                "pass": False,
                "info": {"system_errors": f"验证异常: {e}", "errors": [], "warnings": [], "sorries": []},
            } for _ in lean_codes]

        # ===== 逐条处理 =====
        for local_idx, (item, lean_code, verify_result) in enumerate(zip(prepared_items, lean_codes, verify_results)):
            item_start = time.time()
            item_info = verify_result.get("info", {})
            passed = bool(verify_result.get("pass", False))

            # 逐样本建模 token 基线
            m_in = per_sample_input_tokens.get(local_idx, 0)
            m_out = shared_output_tokens

            if not lean_code:
                details.append({
                    "id": item["id"], "modeling_status": "failed",
                    "verify_status": "skipped", "solve_status": "skipped",
                    "error": "建模输出为空",
                })
                continue

            # 反馈循环
            feedback_start = time.time()
            current_lean_code, current_passed, current_info, modeling_attempts, fix_in, fix_out = (
                _run_modeling_feedback_loop(item, lean_code, passed, item_info, model, tokenizer, args)
            )
            modeling_elapsed = time.time() - feedback_start

            if not current_passed:
                item_elapsed = time.time() - item_start
                total_elapsed += item_elapsed
                total_modeling_elapsed += modeling_elapsed
                total_modeling_in += m_in
                total_modeling_out += m_out
                total_modeling_fix_in += fix_in
                total_modeling_fix_out += fix_out
                records_with_time += 1
                details.append({
                    "id": item["id"], "modeling_status": "success",
                    "verify_status": "failed", "solve_status": "skipped",
                    "modeling_attempts": modeling_attempts,
                    "modeling_output": current_lean_code,
                    "error": current_info.get("system_errors") or "Lean4 校验未通过（含反馈重试）",
                    "last_errors": format_errors_for_modeling(current_info)[:500],
                    "elapsed_seconds": round(item_elapsed, 2),
                    "modeling_elapsed_seconds": round(modeling_elapsed, 2),
                    "modeling_tokens": {"input": m_in, "output": m_out,
                                         "fix_input": fix_in, "fix_output": fix_out},
                    "solve_tokens": {"prompt": 0, "completion": 0},
                })
                # ── 增量断点保存 ──
                save_checkpoint(checkpoint_path, details)
                if _interrupt_requested:
                    break
                continue

            verified_ok += 1
            print(f"  [{item['id']}] ✅ 建模验证通过（第 {modeling_attempts} 轮）")
            solve_result = solve_lean_code(
                formal_statement=current_lean_code,
                llm=solver_llm,
                max_retries=args.solver_max_retries,
                print_logs=False,
                task_type=item.get("task_type", ""),
                sampling_budget=args.sampling_budget,
            )
            item_elapsed = time.time() - item_start
            solve_tokens = solve_result.get("tokens", {})
            solve_elapsed = solve_result.get("solve_elapsed", 0)

            total_elapsed += item_elapsed
            total_modeling_elapsed += modeling_elapsed
            total_solve_elapsed += solve_elapsed
            total_modeling_in += m_in
            total_modeling_out += m_out
            total_modeling_fix_in += fix_in
            total_modeling_fix_out += fix_out
            total_solve_prompt += solve_tokens.get("prompt", 0)
            total_solve_completion += solve_tokens.get("completion", 0)
            records_with_time += 1

            if solve_result.get("success"):
                solved_ok += 1
            detail_entry = {
                "id": item["id"], "modeling_status": "success",
                "verify_status": "passed",
                "solve_status": "success" if solve_result.get("success") else "failed",
                "modeling_attempts": modeling_attempts,
                "modeling_output": current_lean_code,
                "solve_attempts": solve_result.get("attempts", 0),
                "solve_output": strip_lean_header(solve_result.get("best_code", "")),
                "solve_error": solve_result.get("error", ""),
                "elapsed_seconds": round(item_elapsed, 2),
                "modeling_elapsed_seconds": round(modeling_elapsed, 2),
                "solve_elapsed_seconds": round(solve_elapsed, 2),
                "modeling_tokens": {"input": m_in, "output": m_out,
                                     "fix_input": fix_in, "fix_output": fix_out},
                "solve_tokens": {"prompt": solve_tokens.get("prompt", 0),
                                  "completion": solve_tokens.get("completion", 0)},
            }
            # 记录求解方向（正向证明 / 反例证明 / 双向失败）
            if solve_result.get("direction"):
                detail_entry["solve_direction"] = solve_result["direction"]
            # 记录采样预算统计
            sampling_info = solve_result.get("sampling", {})
            if sampling_info:
                detail_entry["sampling"] = sampling_info
                if sampling_info.get("pass_at_k"):
                    pass_at_k_count += 1
                    if sampling_info.get("first_pass_at") is not None:
                        first_pass_distribution.append(sampling_info["first_pass_at"])
            details.append(detail_entry)

            # ── 增量断点保存 ──
            save_checkpoint(checkpoint_path, details)
            if _interrupt_requested:
                break

      # for 循环正常结束或被中断
    finally:
      signal.signal(signal.SIGINT, _original_sigint)
      if _interrupt_requested:
        print(f"\n⚠️  实验中断！已保存 {len(details)}/{len(records)} 题到断点文件")
        print(f"   断点文件: {checkpoint_path}")
        print(f"   下次运行将自动从上次位置继续")

    n = records_with_time if records_with_time > 0 else 1
    # 每题平均总 token 消耗（建模+求解全部 token 之和）
    avg_total_tokens = round(
        (total_modeling_in + total_modeling_out + total_modeling_fix_in +
         total_modeling_fix_out + total_solve_prompt + total_solve_completion) / n, 1
    )
    summary = {
        "total": len(records),
        "modeled_ok": modeled_ok,
        "verified_ok": verified_ok,
        "solved_ok": solved_ok,
        "failed": len(records) - solved_ok,
        "avg_elapsed_seconds": round(total_elapsed / n, 2),
        "avg_modeling_elapsed_seconds": round(total_modeling_elapsed / n, 2),
        "avg_solve_elapsed_seconds": round(total_solve_elapsed / n, 2),
        "avg_total_tokens": avg_total_tokens,
        "avg_modeling_tokens": {
            "input": round(total_modeling_in / n, 1),
            "output": round(total_modeling_out / n, 1),
            "fix_input": round(total_modeling_fix_in / n, 1),
            "fix_output": round(total_modeling_fix_out / n, 1),
        },
        "avg_solve_tokens": {
            "prompt": round(total_solve_prompt / n, 1),
            "completion": round(total_solve_completion / n, 1),
        },
        "details": details,
    }

    # ── Pass@k 采样预算统计 ──
    budget = args.sampling_budget
    if budget > 1:
        # 计算各 k 值的 Pass@k
        pass_at = {}
        entered_solve = sum(1 for d in details if d.get("verify_status") == "passed")
        for k in [1, 4, 8, 16, 32, 64, 128]:
            if k > budget:
                break
            # Pass@k = 首次成功索引 < k 的题目数
            cnt = sum(1 for idx in first_pass_distribution if idx < k)
            pass_at[f"pass@{k}"] = cnt
        summary["sampling"] = {
            "budget": budget,
            "entered_solve": entered_solve,
            "pass_at_k": pass_at_k_count,
            "pass_at_k_rate": f"{pass_at_k_count / max(entered_solve, 1) * 100:.1f}%",
            "pass_at": pass_at,
        }

    summary_file = Path(args.summary_file).expanduser()
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"批处理汇总已写入: {summary_file.resolve()}")
    print(
        f"统计 -> total={summary['total']}, modeled_ok={modeled_ok}, "
        f"verified_ok={verified_ok}, solved_ok={solved_ok}, failed={summary['failed']}"
    )
    print(f"耗时 -> 平均端到端 {summary['avg_elapsed_seconds']}s/题 | 建模 {summary['avg_modeling_elapsed_seconds']}s/题 | 求解 {summary['avg_solve_elapsed_seconds']}s/题")
    print(
        f"Token -> 平均总消耗 {summary['avg_total_tokens']}tokens/题 | "
        f"建模 平均入={summary['avg_modeling_tokens']['input']} "
        f"出={summary['avg_modeling_tokens']['output']} "
        f"修复入={summary['avg_modeling_tokens']['fix_input']} "
        f"修复出={summary['avg_modeling_tokens']['fix_output']} | "
        f"求解 平均prompt={summary['avg_solve_tokens']['prompt']} "
        f"completion={summary['avg_solve_tokens']['completion']}"
    )
    # 打印采样预算统计
    if budget > 1 and "sampling" in summary:
        s = summary["sampling"]
        print(f"采样 -> budget={budget}, Pass@{budget}={s['pass_at_k']}/{s['entered_solve']} ({s['pass_at_k_rate']})")
        for k_name, k_cnt in s["pass_at"].items():
            rate = f"{k_cnt / max(s['entered_solve'], 1) * 100:.1f}%"
            print(f"         {k_name} = {k_cnt}/{s['entered_solve']} ({rate})")
    return summary


if __name__ == "__main__":
    print("该文件已作为建模模块使用，不再作为主程序入口。")
    print("请运行: pipeline.py")
