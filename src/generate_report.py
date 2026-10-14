# -*- coding: utf-8 -*-
"""分析 反馈+采样4.json 并生成 ADFP_100_评估分析报告.docx"""

import json
import re
from collections import defaultdict, Counter
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

# ── 1. 加载数据 ──
with open(r"C:\Users\Lenovo\Desktop\反馈+采样4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]
total = data["total"]
modeled_ok = data["modeled_ok"]
verified_ok = data["verified_ok"]
solved_ok = data["solved_ok"]
failed = data["failed"]
avg_elapsed = data["avg_elapsed_seconds"]
avg_modeling_elapsed = data["avg_modeling_elapsed_seconds"]
avg_solve_elapsed = data["avg_solve_elapsed_seconds"]
avg_total_tokens = data["avg_total_tokens"]
avg_modeling_tokens = data["avg_modeling_tokens"]
avg_solve_tokens = data["avg_solve_tokens"]
sampling_info = data.get("sampling", {})

# ── 2. 分析 ──

# 2a. 失败链路诊断
verify_failed = []   # 建模成功但验证失败
solve_failed = []     # 验证通过但求解失败
all_success = []      # 全部通过

for d in details:
    vid = d["id"]
    vs = d.get("verify_status", "")
    ss = d.get("solve_status", "")
    if vs == "failed":
        verify_failed.append(d)
    elif vs == "passed" and ss == "failed":
        solve_failed.append(d)
    elif vs == "passed" and ss == "success":
        all_success.append(d)

# 2b. 按 ID 前缀分组
prefix_map = {}
for d in details:
    prefix = d["id"].split("-")[1] if "-" in d["id"] else "OTHER"
    prefix_map.setdefault(prefix, []).append(d)

prefix_labels = {
    "FEAS": "可行性验证",
    "LOGIC": "逻辑推理",
    "OPT": "最优性证明",
    "PATH": "路径规划",
    "GEOM": "几何推理",
    "SET": "集合论",
    "RES": "资源分配",
    "STATE": "状态可达性",
    "COMB": "组合数学",
    "SEQ": "序列/递推",
    "INEQ": "不等式证明",
    "GAME": "博弈论",
    "FUNC": "函数方程",
    "NT": "数论",
    "INV": "不变量",
    "HALLU": "幻觉陷阱",
}

# 2c. 采样有效性
first_pass_count = 0
multi_pass_count = 0
for d in all_success:
    sp = d.get("sampling", {})
    fpa = sp.get("first_pass_at")
    if fpa is not None and fpa == 0:
        first_pass_count += 1
    elif fpa is not None and fpa > 0:
        multi_pass_count += 1

# 2d. 求解方向分布
direction_counter = Counter()
for d in details:
    sd = d.get("solve_direction", "")
    if sd:
        direction_counter[sd] += 1

# 2e. 建模反馈
modeling_first_pass = sum(1 for d in details if d.get("modeling_attempts", 0) == 1)
modeling_multi = sum(1 for d in details if d.get("modeling_attempts", 0) > 1)

# 2f. 失败题错误归因
def classify_error(d):
    err = d.get("solve_error", "") or d.get("error", "") or ""
    last_err = d.get("last_errors", "") or ""
    combined = err + " " + last_err
    if "语法错误" in combined or "unexpected token" in combined:
        return "嵌套过深/语法错误"
    if "sorry" in combined and "未完成" in combined:
        return "残留sorry占位符"
    if "模型未输出" in combined:
        return "模型未输出有效代码"
    if "omega could not prove" in combined:
        return "omega策略不足"
    if "failed to synthesize" in combined:
        return "类型类合成失败"
    if "Type mismatch" in combined or "type mismatch" in combined:
        return "类型不匹配"
    if "Unknown constant" in combined:
        return "未知常量引用"
    if "Application type mismatch" in combined:
        return "应用类型不匹配"
    if "Lean4 校验未通过" in combined:
        return "Lean4校验未通过"
    if not combined.strip():
        return "其他/无错误信息"
    return "其他错误"

error_type_counter = Counter()
for d in verify_failed + solve_failed:
    et = classify_error(d)
    error_type_counter[et] += 1

# 2g. 耗时统计
all_times = [d["elapsed_seconds"] for d in details if d.get("elapsed_seconds", 0) > 0]
solve_times = [d["solve_elapsed_seconds"] for d in details if d.get("solve_elapsed_seconds", 0) > 0]
max_time_d = max(details, key=lambda d: d.get("elapsed_seconds", 0))
min_time_d = min((d for d in details if d.get("elapsed_seconds", 0) > 0), key=lambda d: d["elapsed_seconds"])

# 2h. Token统计
total_modeling_input = sum(d.get("modeling_tokens", {}).get("input", 0) for d in details)
total_modeling_output = sum(d.get("modeling_tokens", {}).get("output", 0) for d in details)
total_modeling_fix_input = sum(d.get("modeling_tokens", {}).get("fix_input", 0) for d in details)
total_modeling_fix_output = sum(d.get("modeling_tokens", {}).get("fix_output", 0) for d in details)
total_solve_prompt = sum(d.get("solve_tokens", {}).get("prompt", 0) for d in details)
total_solve_completion = sum(d.get("solve_tokens", {}).get("completion", 0) for d in details)
grand_total_tokens = total_modeling_input + total_modeling_output + total_modeling_fix_input + total_modeling_fix_output + total_solve_prompt + total_solve_completion

modeling_total = total_modeling_input + total_modeling_output + total_modeling_fix_input + total_modeling_fix_output
solve_total = total_solve_prompt + total_solve_completion

# ── 3. 生成 DOCX ──
doc = Document()

# 设置默认字体
style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

def set_cell_font(cell, text, bold=False, size=Pt(9)):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    run.font.size = size
    run.font.name = '宋体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    if bold:
        run.bold = True

def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers), style='Table Grid')
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        set_cell_font(table.rows[0].cells[i], h, bold=True)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            set_cell_font(table.rows[ri + 1].cells[ci], val)
    return table

def add_heading_para(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(14)
    run.font.name = '黑体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    return p

def add_body(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    run.font.name = '宋体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    return p

# ── 标题 ──
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title_p.add_run("ADFP-100 全量评估分析报告（反馈+采样4）")
title_run.bold = True
title_run.font.size = Pt(18)
title_run.font.name = '黑体'
title_run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

# ── 一、执行概况 ──
add_heading_para(doc, "一、执行概况")
add_body(doc,
    f"本次评估在 ADFP-100 数据集上对三层智能体系统（建模层 ATF-8B / 验证层 Lean 4 / 求解层 Goedel-Prover-V2-8B）"
    f"执行端到端测试。采用 4 次采样预算（budget=4）、断点续存模式。核心指标如下：")

add_table(doc,
    ["指标", "数值"],
    [
        ["总题数", str(total)],
        ["建模成功", f"{modeled_ok}/{total} = {modeled_ok/total*100:.1f}%"],
        ["验证通过", f"{verified_ok}/{total} = {verified_ok/total*100:.1f}%"],
        ["求解成功（端到端贯通）", f"{solved_ok}/{total} = {solved_ok/total*100:.1f}%"],
        ["失败题数", str(failed)],
        ["平均总耗时", f"{avg_elapsed:.1f}s（≈{avg_elapsed/3600:.2f}h）"],
        ["平均建模耗时", f"{avg_modeling_elapsed:.2f}s"],
        ["平均求解耗时", f"{avg_solve_elapsed:.2f}s"],
        ["平均总Token", f"{avg_total_tokens:.1f}"],
    ])

# ── 二、失败链路诊断 ──
add_heading_para(doc, "二、失败链路诊断")
add_body(doc, "对失败题按其挂在哪一关进行链路归类，定位系统瓶颈：")

vf_count = len(verify_failed)
sf_count = len(solve_failed)
add_table(doc,
    ["失败类型", "数量", "占比"],
    [
        ["验证层失败（建模未通过Lean4校验）", str(vf_count), f"{vf_count/failed*100:.1f}%"],
        ["求解层失败（验证通过但证明未完成）", str(sf_count), f"{sf_count/failed*100:.1f}%"],
        ["合计", str(failed), "100%"],
    ])

add_body(doc,
    f"系统主要瓶颈在求解层（{sf_count}/{failed} = {sf_count/failed*100:.1f}% 的失败发生于求解层），"
    f"建模与验证层组合通过率 {verified_ok/total*100:.1f}%，证明 ATF-8B 建模能力与 Lean 4 验证机制已较为稳定。")

# ── 三、按 ID 前缀（领域）分组成功率 ──
add_heading_para(doc, "三、按 ID 前缀（领域）分组成功率")
add_body(doc, "按题目编号前缀（即领域类）汇总，便于看哪些防空反导子领域是强项：")

prefix_rows = []
for prefix in sorted(prefix_map.keys(), key=lambda p: (-sum(1 for d in prefix_map[p] if d.get("solve_status")=="success")/len(prefix_map[p]), prefix)):
    items = prefix_map[prefix]
    t_count = len(items)
    s_count = sum(1 for d in items if d.get("solve_status") == "success")
    v_count = sum(1 for d in items if d.get("verify_status") == "passed")
    label = prefix_labels.get(prefix, prefix)
    rate = s_count / t_count * 100 if t_count > 0 else 0
    prefix_rows.append([f"{prefix}（{label}）", str(t_count), str(s_count), f"{rate:.1f}%", str(v_count), f"{v_count/t_count*100:.1f}%"])

add_table(doc, ["领域前缀", "题数", "求解成功", "求解率", "验证通过", "验证率"], prefix_rows)

# ── 四、反馈循环与多轮采样机制有效性 ──
add_heading_para(doc, "四、反馈循环与多轮采样机制有效性")
add_body(doc, "本节直接对应消融实验设计，量化反馈机制的边际收益：")

add_body(doc, "建模反馈循环：")
add_body(doc,
    f"建模一次过 {modeling_first_pass}/{total} = {modeling_first_pass/total*100:.1f}%，"
    f"反馈循环救回 {modeling_multi} 题，使建模通过率从 {modeling_first_pass/total*100:.1f}% 升至 {modeled_ok/total*100:.1f}%。")

add_body(doc, "求解多轮采样（成功题中）：")
solved_total = len(all_success)
add_body(doc,
    f"求解一次过仅 {first_pass_count}/{solved_total} = {first_pass_count/solved_total*100:.1f}%，"
    f"渐进温度采样多轮重试救回 {multi_pass_count} 题，是端到端贯通率的关键放大器。")

# Pass@k 统计
if sampling_info:
    add_body(doc, f"全局采样统计（budget={sampling_info.get('budget',4)}）：")
    add_body(doc,
        f"进入求解层 {sampling_info.get('entered_solve', '?')} 题，"
        f"Pass@1 = {sampling_info.get('pass_at', {}).get('pass@1', '?')}，"
        f"Pass@4 = {sampling_info.get('pass_at', {}).get('pass@4', '?')}，"
        f"Pass@k 通过率 = {sampling_info.get('pass_at_k_rate', '?')}。")

# ── 五、双向求解方向分布 ──
add_heading_para(doc, "五、双向求解方向分布")
add_body(doc, "针对最优性命题，系统支持 forward（直接证）/ counterexample（反证）双向求解：")

dir_rows = []
for d_name, d_count in direction_counter.items():
    dir_rows.append([d_name, str(d_count), f"{d_count/len(details)*100:.1f}%"])
add_table(doc, ["求解方向", "题数", "占比"], dir_rows)

# ── 六、端到端性能指标 ──
add_heading_para(doc, "六、端到端性能指标")

add_table(doc,
    ["指标", "数值"],
    [
        ["端到端求解成功率", f"{solved_ok}/{total} = {solved_ok/total*100:.1f}%"],
        ["建模层通过率", f"{modeled_ok}/{total} = {modeled_ok/total*100:.1f}%"],
        ["验证层通过率", f"{verified_ok}/{total} = {verified_ok/total*100:.1f}%"],
        ["平均总耗时（每题）", f"{avg_elapsed:.1f}s"],
        ["平均建模耗时（每题）", f"{avg_modeling_elapsed:.2f}s"],
        ["平均求解耗时（每题）", f"{avg_solve_elapsed:.2f}s"],
        ["最长耗时题", f"{max_time_d['id']}（{max_time_d['elapsed_seconds']:.1f}s ≈ {max_time_d['elapsed_seconds']/3600:.2f}h）"],
        ["最短耗时题", f"{min_time_d['id']}（{min_time_d['elapsed_seconds']:.1f}s）"],
    ])

# ── 七、Token 消耗结构 ──
add_heading_para(doc, "七、Token 消耗结构")

solve_pct = solve_total / grand_total_tokens * 100 if grand_total_tokens > 0 else 0
modeling_pct = modeling_total / grand_total_tokens * 100 if grand_total_tokens > 0 else 0

add_table(doc,
    ["阶段", "Input/Prompt", "Output/Completion", "Fix Input", "Fix Output", "小计", "占比"],
    [
        ["建模层", str(total_modeling_input), str(total_modeling_output),
         str(total_modeling_fix_input), str(total_modeling_fix_output),
         str(modeling_total), f"{modeling_pct:.1f}%"],
        ["求解层", str(total_solve_prompt), str(total_solve_completion),
         "-", "-", str(solve_total), f"{solve_pct:.1f}%"],
        ["合计", "-", "-", "-", "-", str(grand_total_tokens), "100%"],
    ])

add_body(doc,
    f"求解层占总 token 的 {solve_pct:.1f}%，是计算开销的绝对大头；"
    f"建模反馈环节仅占 {total_modeling_fix_input/(grand_total_tokens)*100:.1f}%（fix_input），"
    f"以极小代价换取建模成功率提升。")

# ── 八、失败题错误类型统计 ──
add_heading_para(doc, "八、失败题错误类型统计")
add_body(doc, f"对全部 {failed} 道失败题按 Lean 4 错误日志特征进行规则归类，统计错误类型分布：")

err_rows = []
for et, cnt in error_type_counter.most_common():
    err_rows.append([et, str(cnt), f"{cnt/failed*100:.1f}%"])
add_table(doc, ["错误类型", "数量", "占比"], err_rows)

# ── 九、失败题逐题错误归因 ──
add_heading_para(doc, "九、失败题逐题错误归因")
add_body(doc, f"对 {failed} 道未通过题逐题标注错误类型与关键错误片段，便于针对性优化：")

fail_rows = []
for d in verify_failed + solve_failed:
    vid = d["id"]
    vs = d.get("verify_status", "")
    ss = d.get("solve_status", "")
    stage = "验证层" if vs == "failed" else "求解层"
    et = classify_error(d)
    err_text = (d.get("last_errors", "") or d.get("solve_error", "") or d.get("error", "") or "")[:80]
    err_text = err_text.replace("\n", " ").strip()
    fail_rows.append([vid, stage, et, err_text])

add_table(doc, ["题目ID", "失败阶段", "错误类型", "关键错误片段（截取）"], fail_rows)

# ── 十、核心结论与改进方向 ──
add_heading_para(doc, "十、核心结论与改进方向")

# (1) 核心成果
add_body(doc, "（1）核心成果")
add_body(doc,
    f"端到端求解贯通率达 {solved_ok/total*100:.1f}%（{solved_ok}/{total}）。"
    f"建模层 {modeled_ok/total*100:.1f}% 成功率证明 ATF-8B 在防空反导工程化场景下建模能力稳定。")

# (2) 亮点
add_body(doc, "（2）亮点分析")

# 找出100%成功的领域
perfect_prefixes = []
for prefix, items in prefix_map.items():
    s = sum(1 for d in items if d.get("solve_status") == "success")
    if s == len(items) and len(items) >= 3:
        label = prefix_labels.get(prefix, prefix)
        perfect_prefixes.append(f"{prefix}（{label}）{s}/{s} 全胜")

if perfect_prefixes:
    add_body(doc, "① 全胜领域：" + "；".join(perfect_prefixes) + "。")

# 幻觉陷阱
hallu_items = prefix_map.get("HALLU", [])
hallu_ok = sum(1 for d in hallu_items if d.get("solve_status") == "success")
if hallu_items:
    add_body(doc, f"② 幻觉陷阱题（HALLU）{hallu_ok}/{len(hallu_items)} {'全胜' if hallu_ok==len(hallu_items) else '通过'}，"
        f"系统能稳定识别射程/概率/单位等物理常识陷阱。")

add_body(doc, f"③ 采样机制效果显著：Pass@1={sampling_info.get('pass_at',{}).get('pass@1','?')} → "
    f"Pass@4={sampling_info.get('pass_at',{}).get('pass@4','?')}，"
    f"多轮采样将求解成功率从首次尝试大幅提升。")

# (3) 短板与改进
add_body(doc, "（3）短板与改进方向")

# 找出最差的领域
worst_prefixes = []
for prefix, items in prefix_map.items():
    s = sum(1 for d in items if d.get("solve_status") == "success")
    rate = s / len(items) * 100 if len(items) > 0 else 0
    if rate < 60 and len(items) >= 2:
        label = prefix_labels.get(prefix, prefix)
        worst_prefixes.append((prefix, label, s, len(items), rate))

worst_prefixes.sort(key=lambda x: x[4])
if worst_prefixes:
    for wp in worst_prefixes[:3]:
        add_body(doc,
            f"① {wp[0]}（{wp[1]}）求解率仅 {wp[2]}/{wp[3]} = {wp[4]:.1f}%，"
            f"主要错误为「{classify_error(next((d for d in prefix_map[wp[0]] if d.get('solve_status')!='success'), {}))}」，"
            f"建议在求解层 Prompt 中增加针对性提示。")

# 验证层失败
if verify_failed:
    add_body(doc,
        f"② {len(verify_failed)} 道验证层失败题集中在「{error_type_counter.most_common(1)[0][0]}」，"
        f"建议在建模层 Prompt 中显式强约束类型选择（ℕ/ℤ/ℝ 不混用）。")

# ── 保存 ──
output_path = r"C:\Users\Lenovo\Desktop\ADFP_100_评估分析报告_采样4.docx"
doc.save(output_path)
print(f"报告已生成: {output_path}")
