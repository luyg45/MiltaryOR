"""
求解层模块 —— Goedel-Prover-V2-8B (GGUF) 推理 + Lean 4 验证反馈循环。

重构说明：
  - 验证逻辑已抽取到 lean4_verifier.py
  - 工具函数已抽取到 lean4_utils.py
  - 本文件仅保留求解层业务逻辑
  - 支持条件性双向求解（正向证明 + 反向证伪）
"""

import os
import re
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Windows 终端下强制 UTF-8，避免中文输出乱码。
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 注意：llama_cpp / transformers 均改为延迟导入（按 backend 决定加载哪个）
# 详见 _SolverLLM 包装类。

from config import CFG
from lean4_verifier import HAS_VERIFIER, verify_single
from lean4_utils import (
    extract_lean_code,
    extract_header,
    assemble_lean_code,
    postprocess_solver_output,
    format_errors_for_solver,
    print_verification_result,
)


# ────────────────────────────────────────────────────────────
# 常量 & 配置（从 config.py 统一读取）
# ────────────────────────────────────────────────────────────

MODEL_PATH = CFG.SOLVER_MODEL_PATH
DEFAULT_INPUT_FILE = CFG.DEFAULT_INPUT_FILE
MAX_RETRIES = CFG.SOLVER_MAX_RETRIES
_SOLVER_LLM: Optional["_SolverLLM"] = None

PROMPT_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a brief proof plan (at most 5 lines) outlining the main strategy.

CRITICAL OUTPUT RULES:
- You MUST output valid Lean 4 code inside a ```lean4 ``` code block.
- Do NOT output natural language explanations, analysis paragraphs, or mathematical derivations.
- If you cannot fully prove the theorem, still output Lean 4 code with `sorry` for unfinished sub-goals.
- Every line of your code output must be valid Lean 4 syntax.

Important tactic guidelines:
- For ℝ arithmetic equalities/inequalities with concrete numbers: use `norm_num` first, then `linarith` if needed.
- For ℕ arithmetic: prefer `omega`.
- NEVER use `tauto` for ℝ inequalities — use `linarith` or `nlinarith` instead.
- NEVER chain `<;>` after a tactic that might fully close all goals (like `omega`, `norm_num`, `linarith`).
  If a tactic closes all goals, do NOT append any `<;> ...` after it.
- For destructuring hypotheses: use `obtain ⟨h1, h2⟩ := h` or `rcases h with ⟨h1, h2⟩`.
- Keep proof structure flat (≤4 levels of nested `have`). Prefer `calc` for chain reasoning.
- For negation goals (¬P, ¬(A ∧ B)): use `push_neg` to normalize, then `linarith` or `by_cases`.
- For completeness proofs (A ∨ B ∨ C covering all cases): use `by_cases` or `rcases` with decidable conditions.
- For fraction literals in ℝ context: always annotate type, e.g. `(95:ℝ)/100` instead of `95/100`.
- For StrictMono → Injective: use `exact StrictMono.injective h` directly.
- For non-linear ℝ inequalities where linarith fails: use `nlinarith` with helper squares, e.g.
  `nlinarith [sq_nonneg (x - y), mul_pos hx hy]`.
""".strip()

FIX_PROMPT_TEMPLATE = """
Your previous Lean 4 code has compilation errors. Please fix the code based on the errors below.

Previous code:
```lean4
{}```

Errors:
{}

Key fix strategies (apply as relevant):
- If "no goals to be solved": remove ALL trailing `<;> ...` chains after the tactic that closed the goal.
  This includes multi-level chains like `<;> a <;> b <;> c` — delete everything from the first `<;>` onward.
- If "tauto failed" on ℝ inequalities: replace `tauto` with `linarith` or `nlinarith`.
- If "linarith failed" on negation goals (¬P or ¬(A ∧ B)): insert `push_neg` before `linarith` to normalize negations into inequalities, then use `linarith` or `by_cases`.
- If "linarith failed" on non-linear goals: use `nlinarith` with helper hints like
  `nlinarith [sq_nonneg (x - y), mul_pos hx hy, pow_pos h 3]`.
- If "unexpected identifier" or deep nesting issues: simplify the proof structure, use fewer nested `have` statements.
- If typeclass/Preorder issues: avoid `lt_iff_lt` or abstract order lemmas; use direct computation with `norm_num` or `linarith`.
- If "Unknown constant 'Finset.card_le_of_subset'": use `Finset.card_le_card` instead.
- If "Unknown constant 'StrictMono.lt_iff_lt'": for StrictMono → Injective, use `exact StrictMono.injective h` directly.
- If variable scope error after `norm_num at *`: avoid `norm_num at *` on let-bindings; use `omega` or `interval_cases` instead.
- If `cases`/`.right`/`.left` fails on And: use `obtain ⟨h1, h2⟩ := ...` or `rcases` pattern matching.
- For ℕ arithmetic goals: prefer `omega`. For ℝ arithmetic: prefer `norm_num` then `linarith`.
- NEVER chain `<;>` after a tactic that might fully close all goals (omega, norm_num, linarith, simp, trivial, decide).
- For fraction literals like `95/100` in ℝ context: use `(95:ℝ)/100` to avoid Nat division.

CRITICAL: Output ONLY valid Lean 4 code inside a ```lean4 ``` block. No natural language.
""".strip()


# ────────────────────────────────────────────────────────────
# 模型管理
# ────────────────────────────────────────────────────────────

def _resolve_backend(model_path: str, declared: str) -> str:
    """根据声明值与路径自动判定后端类型。"""
    declared = (declared or "auto").lower()
    if declared in ("llama_cpp", "transformers"):
        return declared
    # auto 模式：以 .gguf 结尾走 llama_cpp，否则走 transformers
    return "llama_cpp" if model_path.lower().endswith(".gguf") else "transformers"


class _SolverLLM:
    """
    求解层模型统一包装：屏蔽 llama_cpp / transformers 两种后端的差异。
    对外暴露 .create_chat_completion() / .reset()，与 llama_cpp.Llama 同形。
    """

    def __init__(self, model_path: str, backend: str, gpu_id: int = -1):
        self.backend = backend
        self.model_path = model_path

        if backend == "llama_cpp":
            from llama_cpp import Llama  # 延迟导入
            print(f"[solver] 后端: llama_cpp  GGUF: {model_path}")
            self._llm = Llama(
                model_path=model_path,
                n_ctx=CFG.SOLVER_N_CTX,
                n_gpu_layers=CFG.SOLVER_N_GPU_LAYERS,
                verbose=False,
            )
            self._tokenizer = None
            self._torch = None
        elif backend == "transformers":
            import torch  # 延迟导入
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if torch.cuda.is_available():
                # 求解层专用 GPU，与建模层完全不重叠
                _solver_gpus = getattr(CFG, "SOLVER_GPU_IDS", [2, 3])
                _mem_limit = getattr(CFG, "SOLVER_MAX_MEM_PER_GPU", 10)
                _num_gpus = torch.cuda.device_count()
                _max_mem = {}
                for _i in range(_num_gpus):
                    if _i in _solver_gpus:
                        _max_mem[_i] = f"{_mem_limit}GiB"
                    else:
                        _max_mem[_i] = "0GiB"  # 不碰建模层的卡
                print(f"[solver] 后端: transformers  HF dir: {model_path}")
                print(f"[solver] 求解层 GPU: {_solver_gpus}  每卡限 {_mem_limit}GiB")
                _device_map = "auto"
            else:
                _device_map = "cpu"
                _max_mem = None

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            _load_kwargs = dict(
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=_device_map,
                trust_remote_code=True,
                local_files_only=True,
            )
            if _max_mem is not None:
                _load_kwargs["max_memory"] = _max_mem
            self._llm = AutoModelForCausalLM.from_pretrained(model_path, **_load_kwargs)
            self._llm.eval()
            self._torch = torch
            # 加载后验证实际 GPU 分布
            if hasattr(self._llm, "hf_device_map"):
                _used = set(str(v) for v in self._llm.hf_device_map.values())
                print(f"[solver] ✅ 求解层实际分布: {_used}")
                for _i in range(_num_gpus):
                    if _i not in _solver_gpus and any(f"cuda:{_i}" in d for d in _used):
                        print(f"[solver] ⚠️  警告: 求解层意外占用 GPU {_i}!")
        else:
            raise ValueError(f"未知 SOLVER_BACKEND: {backend}（应为 llama_cpp / transformers / auto）")

    # —— 统一推理接口（与 llama_cpp.Llama.create_chat_completion 同形）——
    def create_chat_completion(
        self,
        messages,
        max_tokens: int,
        temperature: float,
        repeat_penalty: float = 1.0,
    ):
        if self.backend == "llama_cpp":
            return self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
            )

        # —— transformers 路径 ——
        torch = self._torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _max_seq = getattr(CFG, "SOLVER_MAX_SEQ_LEN", 4096)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=_max_seq
        ).to(self._llm.device)
        prompt_len = int(inputs.input_ids.shape[1])

        do_sample = temperature is not None and temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            repetition_penalty=repeat_penalty,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = 0.95

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            out = self._llm.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0][prompt_len:]
        text = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": int(gen_tokens.shape[0]),
            },
        }

    # —— 清空 KV cache（题间隔离）——
    def reset(self):
        if self.backend == "llama_cpp":
            self._llm.reset()
        # transformers 每次 generate 都是新一次前向，无需显式 reset

    # 兼容老代码 fallback：llm._ctx.kv_cache_clear()
    @property
    def _ctx(self):
        if self.backend == "llama_cpp":
            return self._llm._ctx
        class _Dummy:
            def kv_cache_clear(self):
                pass
        return _Dummy()


class _SolverLLMProxy:
    """
    求解层模型子进程代理：在独立 GPU 上运行 llama_cpp 推理。
    通过 stdin/stdout JSON 管道与 solver_worker.py 子进程通信。
    对外接口与 _SolverLLM 完全一致（create_chat_completion / reset）。
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.backend = "llama_cpp_subprocess"
        worker_script = str(Path(__file__).resolve().parent / "solver_worker.py")
        env = os.environ.copy()
        env["SOLVER_WORKER_GPU_ID"] = str(gpu_id)
        print(f"[solver] 启动子进程后端: GPU {gpu_id}, worker={worker_script}")
        self._proc = subprocess.Popen(
            [sys.executable, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # 后台线程转发 worker 的 stderr 到主进程 stderr（方便看日志）
        self._stderr_thread = threading.Thread(
            target=self._forward_stderr, daemon=True
        )
        self._stderr_thread.start()

        # 等待 worker 就绪
        ready_resp = self._read_response(timeout=300)  # 模型加载可能较慢
        if ready_resp is None or ready_resp.get("status") != "ready":
            raise RuntimeError(
                f"solver_worker 启动失败 (GPU {gpu_id}): {ready_resp}"
            )
        print(f"[solver] 子进程就绪 (GPU {gpu_id})")

    def _forward_stderr(self):
        """转发子进程 stderr 到主进程（不阻塞主流程）。"""
        try:
            for line in self._proc.stderr:
                sys.stderr.write(line)
                sys.stderr.flush()
        except (ValueError, OSError):
            pass  # 进程已退出

    def _send_request(self, req: dict):
        """向子进程发送 JSON 请求。"""
        if self._proc.poll() is not None:
            raise RuntimeError("solver_worker 子进程已意外退出")
        line = json.dumps(req, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _read_response(self, timeout: float = 600) -> Optional[dict]:
        """从子进程读取一行 JSON 响应。"""
        # 使用简单阻塞读取（推理可能耗时较长）
        line = self._proc.stdout.readline()
        if not line:
            return None
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return {"status": "error", "error": f"worker 响应解析失败: {line[:200]}"}

    # —— 统一推理接口（与 _SolverLLM 同形）——
    def create_chat_completion(
        self,
        messages,
        max_tokens: int,
        temperature: float,
        repeat_penalty: float = 1.0,
    ):
        self._send_request({
            "cmd": "generate",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
        })
        resp = self._read_response(timeout=600)
        if resp is None:
            raise RuntimeError("solver_worker 无响应（可能已崩溃）")
        if resp.get("status") != "ok":
            raise RuntimeError(f"solver_worker 推理失败: {resp.get('error', '未知错误')}")
        return resp["result"]

    def reset(self):
        self._send_request({"cmd": "reset"})
        resp = self._read_response(timeout=30)
        if resp and resp.get("status") != "ok":
            print(f"[solver] reset 警告: {resp.get('error')}")

    # 兼容老代码 fallback
    @property
    def _ctx(self):
        class _Dummy:
            def kv_cache_clear(self_inner):
                self.reset()
        return _Dummy()

    def shutdown(self):
        """优雅关闭子进程。"""
        try:
            self._send_request({"cmd": "shutdown"})
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()

    def __del__(self):
        try:
            if self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass


def get_solver_llm(model_path: str = MODEL_PATH) -> "_SolverLLM":
    """懒加载并缓存求解模型。
    
    - llama_cpp 后端 + SOLVER_GPU_IDS 配置 → 子进程模式（CUDA_VISIBLE_DEVICES 隔离）
    - transformers 后端 → 同进程 device_map + max_memory 固定到专用 GPU
    """
    global _SOLVER_LLM
    if _SOLVER_LLM is None:
        backend = _resolve_backend(model_path, getattr(CFG, "SOLVER_BACKEND", "auto"))
        solver_gpus = getattr(CFG, "SOLVER_GPU_IDS", [])

        # llama_cpp + 指定 GPU → 子进程模式
        if backend == "llama_cpp" and solver_gpus:
            gpu_id = solver_gpus[0]  # 子进程模式用第一张卡
            print(f"正在加载求解模型: {model_path}  (backend=llama_cpp, GPU={gpu_id}, 子进程模式)")
            _SOLVER_LLM = _SolverLLMProxy(gpu_id=gpu_id)
            print("模型加载完成（子进程）")
        else:
            # transformers 后端通过 device_map + max_memory 固定到专用 GPU
            print(f"正在加载求解模型: {model_path}  (backend={backend}, GPUs={solver_gpus})")
            _SOLVER_LLM = _SolverLLM(model_path=model_path, backend=backend)
            print("模型加载完成")
    return _SOLVER_LLM


# ────────────────────────────────────────────────────────────
# 不等式反转工具
# ────────────────────────────────────────────────────────────

def _reverse_inequality(formal_statement: str) -> Optional[str]:
    """
    反转 Lean4 定理中的不等式方向。
    例如 ≥ → <，≤ → >，> → ≤，< → ≥
    返回 None 表示无法反转（无法识别单一不等式符号）。
    """
    # 定位 := by sorry 之前的定理声明部分
    match = re.search(r':=\s*by\s+sorry', formal_statement)
    if not match:
        return None

    stmt_part = formal_statement[:match.start()]
    proof_part = formal_statement[match.start():]

    # 查找所有不等式符号（仅处理恰好一个的情况）
    # 先排除 => 和 -> 箭头符号，避免误匹配
    clean_stmt = re.sub(r'=>|->|→', '', stmt_part)
    ineq_symbols = re.findall(r'[≥≤><]', clean_stmt)
    if len(ineq_symbols) != 1:
        return None

    # 反转不等式
    reversed_stmt = stmt_part
    if '≥' in reversed_stmt:
        reversed_stmt = reversed_stmt.replace('≥', '<', 1)
    elif '≤' in reversed_stmt:
        reversed_stmt = reversed_stmt.replace('≤', '>', 1)
    elif '>' in reversed_stmt:
        reversed_stmt = reversed_stmt.replace('>', '≤', 1)
    elif '<' in reversed_stmt:
        reversed_stmt = reversed_stmt.replace('<', '≥', 1)
    else:
        return None

    # 修改定理名以区分
    reversed_stmt = reversed_stmt.replace(
        'my_favorite_theorem', 'my_favorite_theorem_negation', 1
    )

    return reversed_stmt + proof_part


# ────────────────────────────────────────────────────────────
# 单方向求解（内部函数）
# ────────────────────────────────────────────────────────────

def _solve_one_direction(
    formal_statement: str,
    llm: "_SolverLLM",
    max_retries: int,
    max_tokens: int,
    temperature: float,
    print_logs: bool,
) -> Dict[str, Any]:
    """
    单方向求解：对给定的 formal_statement 执行推理+验证循环。
    """
    solve_start = time.time()
    lean_header = extract_header(formal_statement)
    if print_logs:
        if lean_header:
            print(f"📋 已提取文件头部 ({len(lean_header)} 字符): import/open/set_option")
        else:
            print("⚠️  未检测到 import/open 头部，验证时代码可能缺少类型类实例")

    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(formal_statement)},
    ]

    best_code = ""
    best_passed = False
    last_error = ""
    attempts = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_solve_elapsed = 0.0

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        current_temp = temperature if attempt == 1 else min(temperature + CFG.SOLVER_TEMP_INCREMENT * (attempt - 1), CFG.SOLVER_TEMP_CAP)
        if print_logs:
            print(f"\n{'=' * 60}")
            print(f"  第 {attempt}/{max_retries} 轮推理" +
                  (" (初始生成)" if attempt == 1 else " (修复错误)") +
                  f" [temp={current_temp:.2f}]")
            print(f"{'=' * 60}")

        start = time.time()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=current_temp,
            repeat_penalty=CFG.SOLVER_REPEAT_PENALTY,
        )
        elapsed = time.time() - start

        raw_output = response["choices"][0]["message"]["content"]
        # 采集 llama_cpp 返回的 token 用量
        usage = response.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        if print_logs:
            print(f"生成完成，耗时 {elapsed:.1f} 秒")
            print("-" * 40)
            print(raw_output[:600] + ("..." if len(raw_output) > 600 else ""))
            print("-" * 40)

        lean_code = extract_lean_code(raw_output)

        # 检测自然语言输出：如果提取的代码不含任何 theorem/lemma/def，
        # 说明模型退化为自然语言，需要强制重试
        if lean_code and not re.search(r'\b(theorem|lemma|def|example)\b', lean_code):
            if print_logs:
                print("⚠️  检测到输出为自然语言（无 theorem/lemma），强制重试")
            lean_code = ""  # 标记为无效，触发重试逻辑

        if print_logs:
            print(f"\n📦 已提取 Lean 4 代码（共 {len(lean_code)} 字符）")

        lean_code_full = assemble_lean_code(lean_code, lean_header)
        # 通用后处理：修复常见生成缺陷（多余<;>链、tauto误用、死循环嵌套等）
        lean_code_full = postprocess_solver_output(lean_code_full)
        if lean_code_full != lean_code and print_logs:
            print(f"🔧 已自动拼接头部（+{len(lean_header)} 字符），"
                  f"完整代码 {len(lean_code_full)} 字符")

        best_code = lean_code_full

        # 如果提取为空（自然语言检测或真正无输出），直接进入下一轮重试
        if not lean_code:
            if attempt < max_retries:
                if print_logs:
                    print(f"\n🔄 代码提取失败，进入第 {attempt + 1} 轮重试...")
                # 清理历史，重新开始
                if len(messages) > 1:
                    messages = messages[:-2]
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous output did not contain valid Lean 4 code. "
                        "You MUST output a complete Lean 4 proof inside a ```lean4 ``` code block. "
                        "Do NOT output natural language explanations. "
                        f"Here is the theorem to prove again:\n\n```lean4\n{formal_statement}```"
                    ),
                })
                continue
            last_error = "模型未输出有效 Lean 4 代码"
            break

        if not HAS_VERIFIER:
            last_error = "验证模块不可用，求解结果未经过自动验证"
            break

        if print_logs:
            print("📡 正在连接 Lean 4 验证服务器...")
        passed, info = verify_single(lean_code_full)
        if print_logs:
            print_verification_result(passed, info)

        if passed:
            best_passed = True
            break

        if info:
            error_feedback = format_errors_for_solver(info)
        else:
            error_feedback = "验证服务器未返回详细错误信息"

        if attempt < max_retries:
            if print_logs:
                print(f"\n🔄 将错误反馈给模型，进入第 {attempt + 1} 轮修复...")
            # 只保留初始 prompt + 最新一轮交互，避免 context 随重试累积膨胀
            if len(messages) > 1:
                messages = messages[:-2]
            messages.append({"role": "assistant", "content": raw_output})

            # 高轮次策略切换：第5轮起建议使用sorry封装子引理
            fix_content = FIX_PROMPT_TEMPLATE.format(lean_code_full, error_feedback)
            if attempt >= 4:
                fix_content += (
                    "\n\nIMPORTANT: Previous attempts all failed. Simplify your approach:\n"
                    "- Use `sorry` for complex sub-lemmas and focus on the MAIN proof structure.\n"
                    "- Reduce nested `have` to at most 2 levels.\n"
                    "- For numeric computations, use `norm_num` or `native_decide` directly.\n"
                    "- If the proof requires >20 lines, you are overcomplicating it."
                )

            messages.append({"role": "user", "content": fix_content})
            continue

        last_error = f"自动验证未通过: {error_feedback}"

    total_solve_elapsed = round(time.time() - solve_start, 2)
    return {
        "success": best_passed,
        "best_code": best_code,
        "best_passed": best_passed,
        "error": last_error,
        "attempts": attempts,
        "verifier_available": HAS_VERIFIER,
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
        },
        "solve_elapsed": total_solve_elapsed,
    }


# ────────────────────────────────────────────────────────────
# 核心求解入口（支持双向求解）
# ────────────────────────────────────────────────────────────

def solve_lean_code(
    formal_statement: str,
    llm: Optional["_SolverLLM"] = None,
    max_retries: int = MAX_RETRIES,
    max_tokens: int = CFG.SOLVER_MAX_TOKENS,
    temperature: float = CFG.SOLVER_TEMPERATURE,
    print_logs: bool = True,
    task_type: str = "",
    sampling_budget: int = CFG.SOLVER_SAMPLING_BUDGET,
) -> Dict[str, Any]:
    """
    求解入口。支持条件性双向求解 + 多次独立采样（Pass@k）：
    - 默认执行正向证明
    - 当 task_type 含 'counterexample' 且正向失败时，自动反转不等式尝试证伪
    - sampling_budget > 1 时，对每道题独立生成多个候选证明，统计 Pass@k
    """
    if not formal_statement or not formal_statement.strip():
        return {
            "success": False,
            "best_code": "",
            "best_passed": False,
            "error": "formal_statement 为空",
            "attempts": 0,
            "direction": "",
            "verifier_available": HAS_VERIFIER,
            "tokens": {"prompt": 0, "completion": 0},
            "solve_elapsed": 0,
            "sampling": {"budget": 0, "first_pass_at": None, "total_pass": 0},
        }

    if llm is None:
        llm = get_solver_llm()

    # ===== 多次独立采样（Pass@k）=====
    budget = max(1, sampling_budget)
    samples = []          # 每次采样的结果
    first_pass_idx = None  # 第一次成功的采样索引（0-based）
    total_pass_count = 0   # 总共成功了几次
    total_attempts = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_elapsed = 0.0
    best_result = None     # 最佳结果（优先选成功的）

    for sample_idx in range(budget):
        if print_logs and budget > 1:
            print(f"\n{'#' * 60}")
            print(f"  采样 {sample_idx + 1}/{budget} (Pass@{sample_idx + 1})")
            print(f"{'#' * 60}")

        # 清空 KV cache，确保每次采样独立
        try:
            llm.reset()
        except AttributeError:
            llm._ctx.kv_cache_clear()

        # 每次采样用略微不同的温度（在基础温度上小幅浮动，增加多样性）
        sample_temp = temperature + 0.05 * sample_idx
        sample_temp = min(sample_temp, CFG.SOLVER_TEMP_CAP)

        # ===== 正向求解 =====
        if print_logs:
            print("\n🔹 [正向求解] 尝试证明原命题...")
        forward_result = _solve_one_direction(
            formal_statement, llm, max_retries, max_tokens, sample_temp, print_logs
        )

        if forward_result["success"]:
            forward_result["direction"] = "forward"
            forward_result["solve_elapsed"] = forward_result.get("solve_elapsed", 0)
            sample_result = forward_result
        else:
            # ===== 条件性反向求解 =====
            enable_reverse = "counterexample" in task_type.lower()
            if not enable_reverse:
                forward_result["direction"] = "forward"
                forward_result["solve_elapsed"] = forward_result.get("solve_elapsed", 0)
                sample_result = forward_result
            else:
                reversed_statement = _reverse_inequality(formal_statement)
                if reversed_statement is None:
                    if print_logs:
                        print("\n⚠️  无法自动反转不等式（未识别到单一不等式符号），跳过反向求解")
                    forward_result["direction"] = "forward"
                    forward_result["solve_elapsed"] = forward_result.get("solve_elapsed", 0)
                    sample_result = forward_result
                else:
                    if print_logs:
                        print("\n🔸 [反向求解] 正向证明失败，尝试证明反方向（证伪原命题）...")
                        print(f"   反转后定理片段: ...{reversed_statement[-120:]}")

                    reverse_result = _solve_one_direction(
                        reversed_statement, llm, max_retries, max_tokens, sample_temp, print_logs
                    )

                    # 汇总正向+反向 token 和耗时
                    fwd_tok = forward_result.get("tokens", {"prompt": 0, "completion": 0})
                    rev_tok = reverse_result.get("tokens", {"prompt": 0, "completion": 0})
                    fwd_time = forward_result.get("solve_elapsed", 0)
                    rev_time = reverse_result.get("solve_elapsed", 0)

                    if reverse_result["success"]:
                        reverse_result["direction"] = "counterexample"
                        reverse_result["attempts"] = forward_result["attempts"] + reverse_result["attempts"]
                        reverse_result["tokens"] = {
                            "prompt": fwd_tok["prompt"] + rev_tok["prompt"],
                            "completion": fwd_tok["completion"] + rev_tok["completion"],
                        }
                        reverse_result["solve_elapsed"] = round(fwd_time + rev_time, 2)
                        sample_result = reverse_result
                    else:
                        # 双向均失败
                        sample_result = {
                            "success": False,
                            "best_code": forward_result["best_code"],
                            "best_passed": False,
                            "error": f"双向求解均失败。正向: {forward_result['error']}; 反向: {reverse_result['error']}",
                            "attempts": forward_result["attempts"] + reverse_result["attempts"],
                            "direction": "both_failed",
                            "verifier_available": HAS_VERIFIER,
                            "tokens": {
                                "prompt": fwd_tok["prompt"] + rev_tok["prompt"],
                                "completion": fwd_tok["completion"] + rev_tok["completion"],
                            },
                            "solve_elapsed": round(fwd_time + rev_time, 2),
                        }

        # 记录本次采样
        samples.append(sample_result)
        total_attempts += sample_result.get("attempts", 0)
        tok = sample_result.get("tokens", {"prompt": 0, "completion": 0})
        total_prompt_tokens += tok.get("prompt", 0)
        total_completion_tokens += tok.get("completion", 0)
        total_elapsed += sample_result.get("solve_elapsed", 0)

        if sample_result.get("success"):
            total_pass_count += 1
            if first_pass_idx is None:
                first_pass_idx = sample_idx
            if best_result is None:
                best_result = sample_result  # 记录第一个成功的结果

        # 如果已经成功了，可以选择提前终止（节省算力）
        # 但为了完整的 Pass@k 统计，默认继续采样
        # 如果想提前终止，取消下面的注释：
        # if sample_result.get("success"):
        #     break

    # ===== 汇总结果 =====
    if best_result is None:
        # 所有采样都失败了，返回最后一个结果
        best_result = samples[-1] if samples else {
            "success": False, "best_code": "", "best_passed": False,
            "error": "所有采样均失败", "attempts": 0, "direction": "",
            "verifier_available": HAS_VERIFIER,
            "tokens": {"prompt": 0, "completion": 0}, "solve_elapsed": 0,
        }

    # 添加采样统计信息
    best_result["sampling"] = {
        "budget": budget,
        "first_pass_at": first_pass_idx,   # None 表示全部失败
        "total_pass": total_pass_count,
        "pass_at_k": total_pass_count > 0,  # Pass@budget 是否成功
    }
    # 汇总所有采样的 token 和耗时
    best_result["attempts"] = total_attempts
    best_result["tokens"] = {
        "prompt": total_prompt_tokens,
        "completion": total_completion_tokens,
    }
    best_result["solve_elapsed"] = round(total_elapsed, 2)

    if print_logs and budget > 1:
        print(f"\n{'=' * 60}")
        print(f"  采样预算汇总: budget={budget}, 成功次数={total_pass_count}, "
              f"首次成功@{first_pass_idx + 1 if first_pass_idx is not None else 'N/A'}")
        print(f"  Pass@{budget} = {'✅ 通过' if total_pass_count > 0 else '❌ 未通过'}")
        print(f"{'=' * 60}")

    return best_result


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

def solve_from_file(input_file: str = DEFAULT_INPUT_FILE, print_logs: bool = True) -> Dict[str, Any]:
    """兼容原有 CLI 文件入口。"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            formal_statement = f.read().strip()
        if not formal_statement:
            raise ValueError(f"文件 {input_file} 为空")
        if print_logs:
            print(f"已读取 {input_file}，共 {len(formal_statement)} 字符")
            print("-" * 40)
            print(formal_statement)
            print("-" * 40)
    except FileNotFoundError:
        return {
            "success": False,
            "best_code": "",
            "best_passed": False,
            "error": f"找不到输入文件 {input_file}",
            "attempts": 0,
            "verifier_available": HAS_VERIFIER,
        }

    return solve_lean_code(formal_statement=formal_statement, print_logs=print_logs)


def main() -> int:
    result = solve_from_file(DEFAULT_INPUT_FILE, print_logs=True)
    print("\n" + "=" * 60)
    print("  最终结果")
    print("=" * 60)
    print(result.get("best_code", ""))
    print("=" * 60)
    if result["success"]:
        print("\n✅ 验证通过！代码可直接用于 Lean 4。")
        return 0
    print(f"\n⚠️  未通过自动验证，请手动检查代码。原因: {result.get('error', 'unknown')}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
