"""
Lean 4 工具函数 —— 代码提取、头部处理、错误格式化。

职责：
  1. extract_lean_code()   — 从 LLM 输出中提取 Lean 4 代码块
  2. extract_header()      — 提取 import/open 头部
  3. strip_lean_header()   — 去除 import/open 头部，保留定理主体
  4. assemble_lean_code()  — 将头部拼接到模型输出前
  5. postprocess_solver_output() — 求解输出后处理（通用缺陷修复）
  6. format_errors_for_modeling() — 建模层错误格式化
  7. format_errors_for_solver()  — 求解层错误格式化（含修复建议）
"""

import re
from typing import Optional


# ────────────────────────────────────────────────────────────
# 代码提取
# ────────────────────────────────────────────────────────────

def extract_lean_code(response: str) -> str:
    """
    从模型输出中提取 Lean 4 代码块。
    策略：
      1. 如果有 ```lean4 ``` 代码块 → 提取所有代码块内容
      2. 如果有 ``` ``` 代码块 → 同上
      3. 检测混合格式（裸代码 + Markdown 标记 + 代码块）→ 清除 Markdown 标记后合并
      4. 无代码块 → 直接返回原始文本
      5. 最终对提取结果执行去重（防止模型重复输出同一定理）
    """
    # 模式 1：```lean4 ```
    pattern = r'```lean4?\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # 检查代码块前是否还有 Lean 代码（inductive/def/theorem 等）
        before_first_fence = response[:response.index('```')].strip()
        if before_first_fence and _looks_like_lean(before_first_fence):
            # 混合格式：代码块外有 Lean 代码，合并
            clean_before = _strip_markdown_lines(before_first_fence)
            code = clean_before + '\n\n' + '\n\n'.join(m.strip() for m in matches)
        else:
            code = '\n\n'.join(m.strip() for m in matches)
        return _deduplicate_theorems(code)

    # 模式 2：``` ```
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        before_first_fence = response[:response.index('```')].strip()
        if before_first_fence and _looks_like_lean(before_first_fence):
            clean_before = _strip_markdown_lines(before_first_fence)
            code = clean_before + '\n\n' + '\n\n'.join(m.strip() for m in matches)
        else:
            code = '\n\n'.join(m.strip() for m in matches)
        return _deduplicate_theorems(code)

    # 模式 3：无代码块，清除可能的 Markdown 标记后返回
    code = _strip_markdown_lines(response.strip())
    return _deduplicate_theorems(code)


def _looks_like_lean(text: str) -> bool:
    """判断文本是否看起来像 Lean 代码（含关键字）。"""
    lean_keywords = (
        'theorem', 'lemma', 'def', 'inductive', 'structure',
        'import', 'open', 'namespace', 'instance', 'class',
    )
    return any(kw in text for kw in lean_keywords)


def _deduplicate_theorems(code: str) -> str:
    """
    检测并移除重复的定理声明。

    当模型陷入重复输出循环时，会产生多个同名 theorem 声明。
    此函数只保留第一个完整的定理（包括其前置 def/inductive）。
    """
    # 查找所有 theorem/lemma 声明的起始位置
    theorem_starts = [m.start() for m in re.finditer(
        r'^(?:theorem|lemma)\s+\w+', code, re.MULTILINE
    )]

    if len(theorem_starts) <= 1:
        return code

    # 检测是否为同名重复（取第一个和第二个定理名比较）
    first_name_match = re.match(r'(?:theorem|lemma)\s+(\w+)', code[theorem_starts[0]:])
    second_name_match = re.match(r'(?:theorem|lemma)\s+(\w+)', code[theorem_starts[1]:])

    if first_name_match and second_name_match:
        name1 = first_name_match.group(1)
        name2 = second_name_match.group(1)
        if name1 == name2:
            # 同名重复：只保留到第二个定理声明之前
            return code[:theorem_starts[1]].rstrip()

    # 不同名定理（如正向+反向）：保留全部
    return code


def _strip_markdown_lines(text: str) -> str:
    """移除文本中的 Markdown 格式行（标题、代码围栏、水平线等）。"""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 跳过 Markdown 标题行
        if stripped.startswith('#'):
            continue
        # 跳过代码围栏行
        if stripped.startswith('```'):
            continue
        # 跳过纯 Markdown 水平线
        if re.match(r'^[-=*]{3,}$', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()


# ────────────────────────────────────────────────────────────
# 建模层代码后处理（extract后自动清洗）
# ────────────────────────────────────────────────────────────

def postprocess_modeling_output(code: str) -> str:
    """
    对建模层 LLM 生成的 Lean4 形式化代码执行通用后处理清洗。

    修复项（均为高频通用规则，不损失通用性）：
      1. 去除重复 import 行（保留首次出现）
      2. 修复 Lean3 风格的 ∑/∏ k in → ∑/∏ k ∈
      3. 修复内积 subscript ⟪...⟫_ℝ → @inner ℝ _ _
      4. 修复绝对值内嵌类型标注 |(expr : Type)| → |expr|
      5. 去除重复的空行（连续超过2个空行合并为2个）
    """
    if not code or not code.strip():
        return code

    # ── 规则1: 去重 import（保留首次出现的顺序） ──
    lines = code.split('\n')
    seen_imports = set()
    deduped_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import '):
            if stripped in seen_imports:
                continue  # 跳过重复 import
            seen_imports.add(stripped)
        deduped_lines.append(line)
    code = '\n'.join(deduped_lines)

    # ── 规则2: 修复 Lean3 风格 ∑/∏ k in Finset.xxx → ∑/∏ k ∈ Finset.xxx ──
    # 匹配: ∑ <var> in Finset. 或 ∏ <var> in Finset.
    code = re.sub(
        r'([∑∏])\s+(\w+)\s+in\s+(Finset\.[a-zA-Z]+)',
        r'\1 \2 ∈ \3',
        code
    )

    # ── 规则3: 修复内积 subscript ⟪a, b⟫_ℝ → @inner ℝ _ _ a b ──
    # 模式: ⟪expr1, expr2⟫_ℝ  (可能有空格)
    def _fix_inner_product(m):
        expr1 = m.group(1).strip()
        expr2 = m.group(2).strip()
        return f'@inner ℝ _ _ ({expr1}) ({expr2})'

    code = re.sub(
        r'⟪([^,⟫]+),\s*([^⟫]+)⟫_ℝ',
        _fix_inner_product,
        code
    )

    # ── 规则4: 修复绝对值内嵌类型标注 |(x i : ℝ) - 3| → |(x i) - 3| ──
    # 匹配: |(expr : Type) op ...| 中的 Type annotation
    # 仅处理绝对值上下文中的错误类型标注
    code = re.sub(
        r'\|\(([^)]+)\s*:\s*ℝ\)\s*([+\-*/])',
        r'|(\1) \2',
        code
    )

    # ── 规则5: 合并过多连续空行 ──
    code = re.sub(r'\n{4,}', '\n\n\n', code)

    return code


def enhance_error_feedback_for_modeling(error_feedback: str) -> str:
    """
    基于错误反馈内容，智能追加针对性修复提示。

    在 format_errors_for_modeling 输出的基础上，检测特定错误模式，
    追加明确的修复指令，帮助 8B 模型更高效地修复代码。
    """
    hints = []

    # Finset.range 'in' 语法错误
    if "unexpected token 'in'" in error_feedback or 'unexpected token' in error_feedback and 'in' in error_feedback:
        if 'Finset.range' in error_feedback or '∑' in error_feedback or '∏' in error_feedback:
            hints.append(
                "HINT: Replace `∑ k in Finset.range n, ...` with `∑ k ∈ Finset.range n, ...` "
                "(use Unicode ∈ symbol, NOT the keyword 'in')."
            )

    # filter_nonempty 不存在
    if 'filter_nonempty' in error_feedback or 'Unknown constant' in error_feedback and 'nonempty' in error_feedback.lower():
        hints.append(
            "HINT: `Finset.filter_nonempty` does not exist. "
            "Use `Finset.univ_nonempty` for full Finset, or prove nonemptiness manually."
        )

    # DecidablePred
    if 'DecidablePred' in error_feedback or 'DecidableRel' in error_feedback:
        hints.append(
            "HINT: Add `[DecidableRel can_detect]` or `[DecidablePred P]` to theorem parameters. "
            "Or add `open Classical in` before the theorem."
        )

    # OfNat (Fin n)
    if 'OfNat' in error_feedback and 'Fin' in error_feedback:
        hints.append(
            "HINT: `Fin n` has no numeral instance when `n` is variable. "
            "Use `.val` to extract ℕ, or construct via `⟨1, by omega⟩`."
        )

    # ℕ+ vs ℕ type mismatch
    if 'ℕ+' in error_feedback or ('PNat' in error_feedback and 'mismatch' in error_feedback.lower()):
        hints.append(
            "HINT: Cast ℕ+ to ℕ using `(x : ℕ)` or `x.val`. "
            "Or define the parameter as `ℕ` with `(h : x > 0)` hypothesis."
        )

    # Finset vs Set
    if ('⋃' in error_feedback or 'Set' in error_feedback) and 'Finset' in error_feedback and 'mismatch' in error_feedback.lower():
        hints.append(
            "HINT: `⋃` returns `Set`, not `Finset`. Use `Finset.biUnion` for Finset operations, "
            "or convert with `.toSet`."
        )

    # 内积 parse error
    if '⟪' in error_feedback or '⟫' in error_feedback or 'expected token' in error_feedback.lower():
        if 'inner' in error_feedback.lower() or '⟪' in error_feedback:
            hints.append(
                "HINT: Inner product `⟪a, b⟫_ℝ` syntax fails. "
                "Use `@inner ℝ _ _ a b` instead."
            )

    # Missing cases
    if 'Missing cases' in error_feedback or 'missing cases' in error_feedback:
        hints.append(
            "HINT: Your pattern match is not exhaustive. "
            "Add `| _ => sorry` as a wildcard case, or enumerate all missing constructors."
        )

    # omega could not prove
    if 'omega could not prove' in error_feedback:
        hints.append(
            "HINT: `omega` only works on linear arithmetic over ℕ/ℤ. "
            "If the goal involves multiplication or non-linear terms, use `nlinarith` or `norm_num` instead. "
            "If the goal requires variable bounds, add explicit hypotheses."
        )

    if hints:
        return error_feedback + '\n\n' + '\n'.join(hints)
    return error_feedback


# ────────────────────────────────────────────────────────────
# 头部处理
# ────────────────────────────────────────────────────────────

def extract_header(file_content: str) -> str:
    """
    从 Lean 4 源文件中提取头部（主定理 `my_favorite_theorem` 之前的所有内容）。

    用途：8B 模型通常只输出定理体，不包含 open Real 或前置 def 等声明。
          提取头部后在验证前自动拼接，确保代码可独立编译。

    策略：
      - 优先查找 `theorem my_favorite_theorem` 或 `lemma my_favorite_theorem`
      - 如找不到，退而查找第一个 `theorem`/`lemma`（不含 `def`）
      - 这样 def/inductive 等辅助定义会被保留在 header 中，随主定理一起送入验证
    """
    # 优先：查找主定理声明
    match = re.search(r'\b(?:theorem|lemma)\s+my_favorite_theorem\b', file_content)
    if match:
        return file_content[:match.start()].strip()

    # 退而：第一个 theorem/lemma（但不包含 def，因为 def 可能是辅助定义）
    match = re.search(r'\b(?:theorem|lemma)\b', file_content)
    if match:
        return file_content[:match.start()].strip()
    return ""


def strip_lean_header(code: str) -> str:
    """去掉 Lean4 代码开头的 import/open/set_option 行，只保留定理主体。"""
    lines = code.split('\n')
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(('import ', 'open ', 'set_option ', '#', '--')):
            start = i
            break
    # 跳过定理前的空行
    while start < len(lines) and not lines[start].strip():
        start += 1
    return '\n'.join(lines[start:]).strip()


def assemble_lean_code(model_code: str, header: str) -> str:
    """
    始终将 input.lean 的头部拼接到模型输出前。

    修复说明：
      - 老代码见 import Mathlib 就跳过拼接，但 8B 模型常输出
        import Mathlib 却不带 open Real，导致 HAdd/HPow 等错误。
      - 新策略：除非模型输出第一行完全等于头部第一行，否则强制拼接。
      - 重复 import 在 Lean 4 中无害。
    """
    code = model_code.strip()
    if not code or not header:
        return code

    # 仅当模型输出第一行与头部第一行完全一致时才跳过（避免完全重复）
    code_first = code.split('\n')[0].strip()
    header_first = header.split('\n')[0].strip()
    if code_first == header_first:
        return code

    # 强制拼接头部
    return header + "\n\n" + code


# ────────────────────────────────────────────────────────────
# 错误格式化
# ────────────────────────────────────────────────────────────

def postprocess_solver_output(code: str) -> str:
    """
    对 Solver 生成的 Lean4 证明代码执行通用后处理，修复常见生成缺陷。

    修复项（均为通用规则，不针对特定题目）：
      1. 移除已关闭目标后的多余 `<;>` 尾链（防止 "no goals" 错误）
      2. 截断无限递归嵌套 `have` 语句（防止死循环/栈溢出）
      3. 移除意外重复的 `by sorry` 残留
      4. 修复 `tauto` 误用于实数不等式（替换为 linarith）
      5. 修复裸分数字面量的类型问题
      6. 修复关键字与标识符粘连（如 ifvar → if var）
    """
    if not code or not code.strip():
        return code

    # ── 规则6: 修复关键字与标识符粘连（token拼接缺少空格） ──
    # 例如: ifallocation → if allocation, thenx → then x, elsex → else x
    code = re.sub(r'\b(if|then|else|let|fun|match|do|return|where)([a-zA-Z_])', r'\1 \2', code)

    lines = code.split('\n')
    result_lines = []
    have_nesting_depth = 0
    max_have_nesting = 15  # 超过此深度视为死循环，截断

    for i, line in enumerate(lines):
        stripped = line.strip()

        # ── 规则1: 移除行尾多余 <;> 链 ──
        # 当一个能独立关闭所有目标的 tactic 后面还跟着 <;> 链时，
        # 后续 tactic 会报 "no goals to be solved"。
        # 策略：强闭合tactic后的所有<;>链全部移除（包括多层）。
        strong_closers = r'omega|norm_num|linarith|nlinarith|ring|simp|trivial|done|interval_cases|decide|native_decide|tauto|aesop'
        if re.search(rf'\b({strong_closers})\b.*<;>', stripped):
            # 保留第一个强闭合 tactic 及其直接参数，删除其后所有 <;> 链
            cleaned = re.sub(
                rf'(\b(?:{strong_closers})\b(?:\s+[^<;>\n]*?)??)\s*<;>.*$',
                r'\1', stripped
            )
            line = line[:len(line) - len(line.lstrip())] + cleaned

        # ── 规则2: 截断过深的 have 嵌套 ──
        if stripped.startswith('have ') or stripped.startswith('have\t'):
            have_nesting_depth += 1
        elif stripped and not stripped.startswith('--'):
            # 非 have 行且非空/注释，重置深度计数
            if not stripped.startswith('·') and not stripped.startswith('|'):
                have_nesting_depth = max(0, have_nesting_depth - 1)

        if have_nesting_depth > max_have_nesting:
            # 截断: 用 sorry 结束当前证明分支
            result_lines.append(line[:len(line) - len(line.lstrip())] + '  sorry')
            # 跳过后续所有深层嵌套行
            break

        result_lines.append(line)

    code = '\n'.join(result_lines)

    # ── 规则3: 移除孤立的 `by sorry` 残留（整行只有 `by sorry`） ──
    # 不影响 `:= by sorry`（那是建模层占位符）
    code = re.sub(r'^\s*by sorry\s*$', '', code, flags=re.MULTILINE)

    # ── 规则4: 将误用的 tauto 替换为 linarith（当上下文含 ℝ 不等式） ──
    # 仅在 proof body 中（:= by 之后），如果文件含 Real/ℝ 相关内容
    if 'ℝ' in code or 'Real' in code:
        # 将独立的 `tauto` tactic 替换为 `linarith`（保守: 仅行首或 by 后）
        code = re.sub(r'(\bby\s+)tauto\b', r'\1linarith', code)
        code = re.sub(r'^\s*tauto\s*$', lambda m: m.group(0).replace('tauto', 'linarith'), code, flags=re.MULTILINE)

    # ── 规则5: 修复裸分数字面量的类型问题 ──
    # Lean4 中 `95/100` 是自然数除法(=0)，不等于 `0.95`(ℝ)
    # 当上下文涉及 ℝ 类型时，将 `数字/数字` 转为 `(数字:ℝ)/数字`
    if 'ℝ' in code or 'Real' in code:
        # 匹配独立的 分子/分母 模式（排除已有类型注解的情况）
        # 例: 95/100 → (95:ℝ)/100，但不改 (95:ℝ)/100 或 Fin 3/2
        def _fix_fraction(m):
            prefix = m.group(1)  # 前导字符
            num = m.group(2)
            den = m.group(3)
            # 排除：已有类型注解、在 Fin/Nat 上下文中
            if prefix in (':', 'n', 't'):  # :ℝ)/, Fin, Nat 等
                return m.group(0)
            return f'{prefix}({num}:ℝ)/{den}'
        # 匹配模式：非字母数字字符 + 数字 + / + 数字（排除已有注解）
        code = re.sub(
            r'([^a-zA-Z0-9:])(\d{1,4})/(\d{1,4})(?!\s*[a-zA-Z])',
            _fix_fraction, code
        )

    return code


def format_errors_for_modeling(info: dict) -> str:
    """将 Lean 验证错误格式化为建模模型可理解的反馈（简洁英文风格）。"""
    parts = []

    if info.get('errors'):
        for err in info['errors']:
            pos = err.get('pos', {})
            line = pos.get('line', '?')
            col = pos.get('column', '?')
            data = err.get('data', str(err))
            short = data[:300] + '...' if len(data) > 300 else data
            parts.append(f"Line {line}:{col} → {short}")

    if info.get('warnings'):
        for w in info['warnings'][:3]:
            pos = w.get('pos', {})
            parts.append(f"Warning at line {pos.get('line', '?')}: {w.get('data', str(w))[:200]}")

    if info.get('sorries'):
        parts.append(
            f"{len(info['sorries'])} sorry placeholder(s) remain — "
            f"replace with `sorry` is acceptable for statement_only."
        )

    if info.get('system_errors'):
        parts.append(f"System error: {info['system_errors']}")

    return '\n'.join(parts) if parts else "Unknown verification error"


def format_errors_for_solver(info: dict) -> str:
    """
    将 Lean 的 raw error 翻译为求解模型能理解的修复指令（含具体修复建议）。
    """
    parts = []

    synthesize_errors = []
    unknown_tactics = []
    unsolved_goals = []
    timeout_errors = []
    no_goals_errors = []
    tauto_errors = []
    already_declared = []
    unexpected_token_errors = []
    scope_errors = []
    other_errors = []

    if info.get('errors'):
        for err in info['errors']:
            data = err.get('data', str(err))
            pos = err.get('pos', {})
            line = pos.get('line', '?')
            if 'failed to synthesize' in data.lower():
                synthesize_errors.append((line, data))
            elif 'unknown tactic' in data.lower():
                m = re.search(r"unknown tactic '?(\w+)'?", data)
                tactic = m.group(1) if m else 'unknown'
                unknown_tactics.append((line, tactic))
            elif 'unsolved goals' in data.lower():
                unsolved_goals.append((line, data))
            elif 'deterministic timeout' in data.lower() or 'maximum recursion depth' in data.lower():
                timeout_errors.append((line, data))
            elif 'no goals to be solved' in data.lower() or 'no goals' in data.lower():
                no_goals_errors.append((line, data))
            elif "tactic 'tauto' failed" in data.lower() or "tactic `tauto` failed" in data.lower():
                tauto_errors.append((line, data))
            elif 'has already been declared' in data.lower():
                already_declared.append((line, data))
            elif 'unexpected' in data.lower() and ('token' in data.lower() or 'identifier' in data.lower()):
                unexpected_token_errors.append((line, data))
            elif 'unknown identifier' in data.lower() or 'unknown variable' in data.lower():
                scope_errors.append((line, data))
            else:
                other_errors.append((line, data))

    # typeclass 缺失 -> 加 open Real
    if synthesize_errors:
        parts.append(
            "缺少 'open' 声明：代码使用了 ℝ (Real) 运算 "
            "(+, -, *, ^, 数字)，但没有包含 'open Real'。"
            "修复方法：在定理声明之前添加 'open Real'。"
        )
        affected = sorted(set(str(l) for l, _ in synthesize_errors))
        parts.append(f"  受影响的行: {', '.join(affected[:8])}")

    # 未知 tactic -> 给替代方案
    if unknown_tactics:
        suggestions = {
            'norm_num1': "使用 'norm_num' 代替 'norm_num1'",
            'arith': "使用 'linarith' 或 'nlinarith' 代替",
            'omega': "使用 'linarith' 或 'omega'",
            'calc?': "使用 'calc' 块，每步用 by ... 证明",
        }
        for line, tactic in unknown_tactics:
            hint = suggestions.get(
                tactic,
                f"未知 tactic '{tactic}'。"
                f"常用有效 tactics: nlinarith, linarith, ring, field_simp, "
                f"norm_num, rw, simp, apply, exact, intro, have, calc, refine, "
                f"fin_cases, native_decide",
            )
            parts.append(f"修复第 {line} 行: {hint}")

    # unsolved goals
    if unsolved_goals:
        parts.append(
            "未解决的目标：证明不完整。"
            "请确保所有目标都已关闭（没有遗留 'sorry'）。"
        )

    # timeout -> 建议简化证明
    if timeout_errors:
        parts.append(
            "❗ 证明超时(deterministic timeout)：证明计算量过大。"
            "修复方法（按优先级）："
        )
        parts.append(
            "  1. 尝试用 `native_decide` 替代复杂的分支证明"
            "  2. 删除 `interval_cases`（会组合爆炸）"
            "  3. 减少嵌套的 `<;>` 链式调用"
            "  4. 把复杂的 norm_num/simp 调用拆成多个 `have` 步骤"
        )
        affected = sorted(set(str(l) for l, _ in timeout_errors))
        parts.append(f"  超时位置: 第 {', '.join(affected[:5])} 行")

    # tauto failed -> 指引用 linarith
    if tauto_errors:
        parts.append(
            "❌ `tauto` 对实数(ℝ)不等式无效。`tauto` 只能处理纯命题逻辑(∧/∨/¬/→)。"
            "修复方法：将所有 `tauto` 替换为 `linarith` 或 `omega`（自然数时）。"
            "对于含有 ≥/≤/>/<  的 ℝ 类型目标，使用 `linarith` 即可。"
        )
        affected = sorted(set(str(l) for l, _ in tauto_errors))
        parts.append(f"  受影响的行: {', '.join(affected[:5])}")

    # no goals -> 删除多余 tactic
    if no_goals_errors:
        parts.append(
            "❌ 'No goals to be solved'：之前的 tactic 已关闭所有目标，"
            "但后面的 `<;> ...` 链式调用仍在执行。"
            "修复方法："
        )
        parts.append(
            "  1. 删除报错行及之后的所有 `<;> ...` 后缀"
            "  2. 或将 `<;>` 链拆成独立的 `· ...` 分支（focused tactic），"
            "     让每个分支只作用于对应目标"
        )

    # already declared -> 只输出一份
    if already_declared:
        parts.append(
            "❌ 定理名重复声明：输出中包含多个同名 theorem。"
            "修复方法：只输出一个完整的 theorem 定义，不要重复。"
            "请删除所有重复的 theorem 声明，只保留一份完整证明。"
        )

    # unexpected token/identifier -> 简化证明结构
    if unexpected_token_errors:
        parts.append(
            "❌ 语法错误(unexpected token/identifier)：证明结构过于复杂或嵌套过深。"
            "修复方法："
        )
        parts.append(
            "  1. 减少嵌套的 `have` 层数（最多 3-4 层）"
            "  2. 使用 `obtain ⟨h1, h2⟩ := ...` 代替 `cases ... with`"
            "  3. 避免在 `calc` 块内嵌套 `have`"
            "  4. 确保每个 `have` 的证明体用 `:= by ...` 而非 `by` 后换行缩进"
        )
        affected = sorted(set(str(l) for l, _ in unexpected_token_errors))
        parts.append(f"  出错位置: 第 {', '.join(affected[:5])} 行")

    # scope/unknown variable -> 变量作用域问题
    if scope_errors:
        parts.append(
            "❌ 变量作用域错误：引用了不存在或已被消除的变量。"
            "修复方法："
        )
        parts.append(
            "  1. 不要对 let-binding 变量使用 `norm_num at *`（会消除绑定）"
            "  2. 使用 `omega` 或 `interval_cases` 处理有界自然数"
            "  3. 用 `rcases` 或 `obtain` 在使用前解构复合假设"
        )
        affected = sorted(set(str(l) for l, _ in scope_errors))
        parts.append(f"  出错位置: 第 {', '.join(affected[:5])} 行")

    # other
    if other_errors:
        parts.append("其他错误:")
        for line, data in other_errors[:4]:
            short = data[:180] + '...' if len(data) > 180 else data
            parts.append(f"  第 {line} 行: {short}")

    if info.get('warnings'):
        parts.append(f"警告 ({len(info['warnings'])} 条):")
        for w in info['warnings'][:2]:
            parts.append(f"  {w.get('data', str(w))[:120]}")

    if info.get('sorries'):
        parts.append(
            f"未完成：还有 {len(info['sorries'])} 处 'sorry'。"
            f"请全部替换为实际证明代码。"
        )

    return '\n'.join(parts)


def print_verification_result(passed: bool, info: Optional[dict]) -> None:
    """友好的验证结果打印（供 CLI / 调试使用）。"""
    if info is None:
        return

    print(f"\n{'✅ 语法验证通过' if passed else '❌ 语法验证未通过'}")

    if info.get('errors'):
        print(f"\n🔴 语法错误 ({len(info['errors'])} 个):")
        for err in info['errors']:
            pos = err.get('pos', {})
            line = pos.get('line', '?')
            col = pos.get('column', '?')
            data = err.get('data', str(err))
            print(f"   行 {line}:{col} → {data}")

    if info.get('warnings'):
        print(f"\n🟡 警告 ({len(info['warnings'])} 个):")
        for w in info['warnings']:
            pos = w.get('pos', {})
            print(f"   行 {pos.get('line', '?')} → {w.get('data', str(w))}")

    if info.get('sorries'):
        print(f"\n🔵 未完成的 sorry ({len(info['sorries'])} 处):")
        for s in info['sorries']:
            pos = s.get('pos', {})
            goal = s.get('goal', '')
            short = (goal[:100] + '...') if len(goal) > 100 else goal
            print(f"   行 {pos.get('line', '?')} → goal: {short}")

    if info.get('system_errors'):
        print(f"\n💥 系统错误: {info['system_errors']}")

    if passed and not info.get('sorries') and not info.get('warnings'):
        print("\n🎉 完美！无语法错误、无 sorry 占位、无警告。")
