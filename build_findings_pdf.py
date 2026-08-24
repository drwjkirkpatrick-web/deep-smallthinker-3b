#!/usr/bin/env python3
"""
Build the comprehensive deep-smallthinker-3b findings PDF report.

Covers ALL 75 experiments:
  - 10-prompt baseline benchmark (temp 1.0, 32K context)
  - 20-run temperature sweep (0.1-0.9, 4 creative prompts)
  - 45-run variable sweep (top_p, top_k, repeat_penalty; 3 prompts each)
  - 10-prompt tuned retest (best settings per category, 32K context)
  - Tuned settings matrix + per-category recommendations
"""
import json
import os
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BASE = os.path.expanduser("~/projects/deep-smallthinker-3b/results")
OUTPUT = os.path.expanduser("~/projects/deep-smallthinker-3b/Deep_SmallThinker_3B_Findings_Report.pdf")

# ---- Load all data ----
with open(f"{BASE}/benchmark_final.json") as f:
    BENCH = json.load(f)
with open(f"{BASE}/benchmark_results.json") as f:
    BENCH_RAW = json.load(f)
with open(f"{BASE}/temp_sweep_results.json") as f:
    TEMP_SWEEP = json.load(f)
with open(f"{BASE}/temp_sweep_quality.json") as f:
    TEMP_QUALITY = json.load(f)
with open(f"{BASE}/variable_sweep_results.json") as f:
    VAR_SWEEP = json.load(f)
with open(f"{BASE}/variable_sweep_quality.json") as f:
    VAR_QUALITY = json.load(f)
with open(f"{BASE}/quality_scores.json") as f:
    QUALITY_SCORES = json.load(f)
with open(f"{BASE}/tuned_retest_results.json") as f:
    TUNED_RETEST = json.load(f)
with open(f"{BASE}/tuned_retest_quality.json") as f:
    TUNED_QUALITY = json.load(f)

# ---- Colors (house style) ----
HEADER_BG = colors.HexColor("#1a237e")
HEADER_FG = colors.white
BEST_BG = colors.HexColor("#e8f5e9")   # green
GOOD_BG = colors.HexColor("#fff8e1")   # yellow
BAD_BG = colors.HexColor("#ffebee")    # red
ALT_ROW = colors.HexColor("#f5f5f5")
AVG_BG = colors.HexColor("#e3f2fd")

# ---- Styles ----
styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=4, textColor=HEADER_BG)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey, spaceAfter=10)
section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6, textColor=HEADER_BG)
sub_section = ParagraphStyle("SubSection", parent=styles["Heading3"], fontSize=10, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#333333"))
cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.5, leading=9, alignment=TA_CENTER)
cell_left = ParagraphStyle("CellLeft", parent=cell_style, alignment=TA_LEFT)
cell_bold = ParagraphStyle("CellBold", parent=cell_style, fontName="Helvetica-Bold")
cell_bold_left = ParagraphStyle("CellBoldLeft", parent=cell_left, fontName="Helvetica-Bold")
header_cell = ParagraphStyle("HeaderCell", parent=styles["Normal"], fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=colors.white, fontName="Helvetica-Bold")
header_left = ParagraphStyle("HeaderLeft", parent=header_cell, alignment=TA_LEFT)
note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8.5, leading=11, spaceAfter=4)
small_note = ParagraphStyle("SmallNote", parent=styles["Normal"], fontSize=7, leading=9, textColor=colors.grey)
finding_style = ParagraphStyle("Finding", parent=styles["Normal"], fontSize=9.5, leading=13, backColor=colors.HexColor("#fff3e0"), borderPadding=6, spaceBefore=4, spaceAfter=8)


def P(text, style=cell_style):
    return Paragraph(str(text), style)


def color_for_think(pct):
    """Color the think% cell: >80% = red (loop), <40% = green (clean), else yellow."""
    if pct >= 80:
        return BAD_BG
    elif pct <= 40:
        return BEST_BG
    return GOOD_BG


def color_for_wall(wall, best_wall=None):
    """Color wall time: green if fastest, red if 2x slower than best."""
    if best_wall and wall <= best_wall * 1.1:
        return BEST_BG
    if best_wall and wall >= best_wall * 2:
        return BAD_BG
    return None


# =====================================================================
# PAGE 1: Title + Executive Summary
# =====================================================================

elements = []
elements.append(Paragraph("Deep-SmallThinker-3B: Complete Findings Report", title_style))
elements.append(Paragraph(
    "SmallThinker-3B-Preview (Q4_K_M, 2.1 GB) on NVIDIA Jetson Orin Nano 8GB | llama.cpp CUDA | "
    "65 experiments total: 10-prompt benchmark + 20-run temperature sweep + 45-run variable sweep | "
    "32K context fixed throughout | GUI off",
    subtitle_style
))

elements.append(Paragraph("Executive Summary", section_style))
elements.append(Paragraph(
    "This report documents 75 experiments to fine-tune SmallThinker-3B for maximum quality on an 8GB Jetson. "
    "Three rounds of testing progressively narrowed the optimal settings from default values to empirically-tuned "
    "per-category parameters, then a final 10-prompt retest validated the tuned settings. The central discovery: "
    "SmallThinker's meta-reasoning loop on creative tasks is <b>chaotic, not monotonic</b> -- there is no single "
    "'safe' temperature, and the loop is quasi-random per (prompt, temperature) pair. The solution is a keyword-based "
    "auto-adjuster that classifies each prompt and applies the best settings for its category. <b>The result: average "
    "quality rose from 6.4/10 to 9.8/10 -- all 10 prompts improved, none regressed.</b>",
    finding_style
))

# Summary metrics table
summary_data = [
    [P("Metric", header_left), P("Value", header_cell), P("Detail", header_left)],
    [P("Total experiments", cell_left), P("75", cell_bold), P("10 benchmark + 20 temp sweep + 45 variable sweep + 10 tuned retest", cell_left)],
    [P("Generation speed", cell_left), P("~20 t/s", cell_bold), P("Consistent across all 75 runs (memory-bandwidth bound)", cell_left)],
    [P("Baseline quality (temp 1.0)", cell_left), P("6.4/10", cell_bold), P("Code/math: 8/10; creative/prose: 3-4/10 (loop)", cell_left)],
    [P("Tuned quality (retest)", cell_left), P("9.8/10", cell_bold), P("10/10 prompts improved, 0 regressed (avg +3.4)", cell_left)],
    [P("Best single speedup", cell_left), P("2.6x", cell_bold), P("top_k=80 vs 40 for reasoning (37s vs 98s)", cell_left)],
    [P("Most sensitive variable", cell_left), P("rep", cell_bold), P("Proof completeness varies 2/5 to 5/5 across repeat_penalty", cell_left)],
    [P("Most robust category", cell_left), P("code", cell_bold), P("14/15 variable runs scored 7/7 code elements", cell_left)],
    [P("Biggest quality jump", cell_left), P("+7", cell_bold), P("Creative writing: 3/10 to 10/10 (temp 0.9, rep 1.2)", cell_left)],
]
st = Table(summary_data, colWidths=[45*mm, 22*mm, 115*mm], repeatRows=1)
st.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
elements.append(st)

# =====================================================================
# PAGE 2: Baseline Benchmark (10 prompts)
# =====================================================================

elements.append(PageBreak())
elements.append(Paragraph("1. Baseline Benchmark — 10 Prompts at Temp 1.0", section_style))
elements.append(Paragraph(
    "The same 10-prompt suite from prior Jetson LLM benchmarks (5 general + 5 coding), run at temp 1.0 with "
    "32K context and 8192 thinking tokens. This established the baseline: deep thinking shifts quality, it doesn't raise it.",
    small_note
))

# Benchmark table
bench_header = [
    P("Prompt", header_left), P("Suite", header_cell), P("Gen\ntok/s", header_cell),
    P("Wall\nsec", header_cell), P("Think\nchars", header_cell), P("Answer\nchars", header_cell),
    P("Think\n%", header_cell), P("Qual\n1-10", header_cell),
]
bench_rows = [bench_header]
for r in BENCH["results"]:
    if "error" in r:
        bench_rows.append([P(r["id"], cell_left), P("ERROR"), P("-"), P("-"), P("-"), P("-"), P("-"), P("-")])
        continue
    bench_rows.append([
        P(r["id"], cell_left), P(r["suite"], cell_left),
        P(f"{r['gen_tps']:.1f}"), P(str(r["wall_time_s"])),
        P(str(r["thinking_chars"])), P(str(r["answer_chars"])),
        P(f"{r['think_ratio_pct']:.0f}%"), P(str(r["quality"])),
    ])
# Average row
gen_vals = [r['gen_tps'] for r in BENCH['results'] if 'gen_tps' in r]
bench_rows.append([
    P("AVERAGE", cell_bold_left), P(""), P(f"{sum(gen_vals)/len(gen_vals):.1f}"),
    P(""), P(""), P(""), P(""), P(f"{BENCH['average_quality']:.1f}"),
])

bt = Table(bench_rows, colWidths=[20*mm, 16*mm, 14*mm, 13*mm, 16*mm, 16*mm, 13*mm, 13*mm], repeatRows=1)
bt_cmds = [
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]
for i in range(1, len(bench_rows) - 1):
    r = BENCH["results"][i - 1]
    if "think_ratio_pct" in r:
        bt_cmds.append(("BACKGROUND", (6, i), (6, i), color_for_think(r["think_ratio_pct"])))
    q = r.get("quality")
    if q and q >= 8:
        bt_cmds.append(("BACKGROUND", (7, i), (7, i), BEST_BG))
    elif q and q <= 4:
        bt_cmds.append(("BACKGROUND", (7, i), (7, i), BAD_BG))
    elif q:
        bt_cmds.append(("BACKGROUND", (7, i), (7, i), GOOD_BG))
    if i % 2 == 0:
        bt_cmds.append(("BACKGROUND", (0, i), (5, i), ALT_ROW))
last = len(bench_rows) - 1
bt_cmds.append(("BACKGROUND", (0, last), (-1, last), AVG_BG))
bt_cmds.append(("FONTNAME", (0, last), (-1, last), "Helvetica-Bold"))
bt.setStyle(TableStyle(bt_cmds))
elements.append(bt)

elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "Quality splits cleanly: verifiable tasks (code, math, HTML, Python, C) score 7-8/10 with clean answers. "
    "Open-ended creative-form tasks (iambic, creative, basic) collapse into meta-reasoning loops at >80% think ratio. "
    "Prose is a special case -- it produces complete medical explanations but lacks a 'Final Answer' delimiter, "
    "so the naive splitter flags it as 100% loop when it's actually complete output.",
    note_style
))

# =====================================================================
# PAGE 3: Temperature Sweep (20 runs)
# =====================================================================

elements.append(PageBreak())
elements.append(Paragraph("2. Temperature Sweep — 20 Runs (0.1 to 0.9)", section_style))
elements.append(Paragraph(
    f"Fixed: ctx={TEMP_SWEEP['fixed_settings']['ctx']}, n_tokens={TEMP_SWEEP['fixed_settings']['n_tokens']}, "
    f"top_p={TEMP_SWEEP['fixed_settings']['top_p']}, top_k={TEMP_SWEEP['fixed_settings']['top_k']}, "
    f"rep={TEMP_SWEEP['fixed_settings']['repeat_penalty']}. Only temperature varied. "
    f"4 open-ended prompts (the ones that collapsed at temp 1.0) x 5 temperatures.",
    small_note
))

# Build temp sweep quality matrix
tq = TEMP_QUALITY["quality_scores"]
temp_header = [
    P("Prompt", header_left), P("0.1", header_cell), P("0.3", header_cell),
    P("0.5", header_cell), P("0.7", header_cell), P("0.9", header_cell),
    P("Best", header_cell),
]
temp_rows = [temp_header]
for pid in ["iambic", "creative", "basic", "prose"]:
    scores = tq[pid]
    row = [P(pid, cell_left)]
    for t in ["0.1", "0.3", "0.5", "0.7", "0.9"]:
        s = scores[t]
        cell_text = f"{s['score']}"
        row.append(P(cell_text, cell_bold))
    row.append(P(str(scores["best"]), cell_bold))
    temp_rows.append(row)

tt = Table(temp_rows, colWidths=[25*mm, 20*mm, 20*mm, 20*mm, 20*mm, 20*mm, 20*mm], repeatRows=1)
tt_cmds = [
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]
# Color the quality cells
for row_idx, pid in enumerate(["iambic", "creative", "basic", "prose"], start=1):
    scores = tq[pid]
    for col_idx, t in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9"], start=1):
        s = scores[t]["score"]
        if s >= 7:
            tt_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), BEST_BG))
        elif s <= 3:
            tt_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), BAD_BG))
        else:
            tt_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), GOOD_BG))
# Highlight best column (best may be a range string, not a single value)
for row_idx, pid in enumerate(["iambic", "creative", "basic", "prose"], start=1):
    scores = tq[pid]
    best_val = str(scores["best"])
    # Try to match a single temp value
    for col_idx, t in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9"], start=1):
        if t == best_val:
            tt_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), BEST_BG))
for i in range(1, len(temp_rows)):
    if i % 2 == 0:
        tt_cmds.append(("BACKGROUND", (0, i), (0, i), ALT_ROW))
tt.setStyle(TableStyle(tt_cmds))
elements.append(tt)

elements.append(Spacer(1, 3*mm))

# Temperature sweep raw metrics
elements.append(Paragraph("Raw Metrics (wall time, think %, answer chars)", sub_section))
raw_header = [
    P("Prompt", header_left), P("Temp", header_cell), P("Gen\ntok/s", header_cell),
    P("Wall\nsec", header_cell), P("Think\nchars", header_cell), P("Answer\nchars", header_cell),
    P("Think\n%", header_cell),
]
raw_rows = [raw_header]
for run in TEMP_SWEEP["runs"]:
    raw_rows.append([
        P(run["id"], cell_left), P(str(run["temp"])),
        P(f"{run['gen_tps']:.1f}"), P(f"{run['wall_time_s']:.1f}"),
        P(str(run["thinking_chars"])), P(str(run["answer_chars"])),
        P(f"{run['think_ratio_pct']:.0f}%"),
    ])

rt = Table(raw_rows, colWidths=[22*mm, 14*mm, 14*mm, 14*mm, 18*mm, 18*mm, 14*mm], repeatRows=1)
rt_cmds = [
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("FONTSIZE", (0, 1), (-1, -1), 6.5),
]
for i in range(1, len(raw_rows)):
    run = TEMP_SWEEP["runs"][i - 1]
    rt_cmds.append(("BACKGROUND", (6, i), (6, i), color_for_think(run["think_ratio_pct"])))
    if i % 2 == 0:
        rt_cmds.append(("BACKGROUND", (0, i), (5, i), ALT_ROW))
rt.setStyle(TableStyle(rt_cmds))
elements.append(rt)

elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "<b>Key finding:</b> The meta-reasoning loop is <b>chaotic, not monotonic</b>. There is no smooth "
    "'lower temp = less looping' curve. The best temperature is prompt-dependent: iambic=0.3 (only temp "
    "that produced a poem), creative=0.9 (only temp that committed to prose), basic=0.7 (closest to "
    "correct TRS-80 BASIC), prose=robust everywhere (0.1-0.9 all scored 8/10 -- the '100% loop' reading "
    "was a false positive from the naive splitter). The single worst run was basic@0.3: 426s, 32K chars "
    "of pure loop, zero output.",
    note_style
))

# =====================================================================
# PAGE 4: Variable Sweep — top_p
# =====================================================================

elements.append(PageBreak())
elements.append(Paragraph("3. Variable Sweep — top_p (15 runs)", section_style))
vs_tp = VAR_SWEEP["sweeps"]["top_p"]
elements.append(Paragraph(
    f"{vs_tp['description']} Values tested: {vs_tp['values']}. "
    "Fixed: top_k=40, repeat_penalty=1.1, ctx=32768, n_tokens=4096. "
    "Temperature per category best from temp sweep.",
    small_note
))

# top_p table
tp_header = [
    P("Prompt", header_left), P("top_p", header_cell), P("Gen\ntok/s", header_cell),
    P("Wall\nsec", header_cell), P("Think\nchars", header_cell), P("Answer\nchars", header_cell),
    P("Think\n%", header_cell), P("Notes", header_left),
]
tp_quality = VAR_QUALITY["findings"]["top_p"]
notes_map = {
    "reasoning": {"0.8": "4/5 proof, 57s (fast)", "0.9": "2x slower (126s)", "0.95": "3/5 proof, 140ch ans",
                  "0.98": "3/5 proof", "1.0": "3/5 proof, 45s (fast)"},
    "creative": {"0.8": "5/5 scene, 2549ch", "0.9": "5/5 scene, 33s (fast)", "0.95": "5/5 scene, 5679ch (most)",
                 "0.98": "5/5 scene, 35s", "1.0": "5/5 scene, 5149ch"},
    "coding": {"0.8": "7/7 code, 99s", "0.9": "7/7 code, 58s", "0.95": "7/7 code, 93s",
               "0.98": "7/7 code, 131s (slow)", "1.0": "7/7 code, 45s (fast)"},
}
tp_rows = [tp_header]
for run in vs_tp["runs"]:
    val_key = str(run["top_p"])
    note = notes_map.get(run["prompt_id"], {}).get(val_key, "")
    tp_rows.append([
        P(run["prompt_id"], cell_left), P(str(run["top_p"])),
        P(f"{run['gen_tps']:.1f}"), P(f"{run['wall_time']:.1f}"),
        P(str(run["think_chars"])), P(str(run["ans_chars"])),
        P(f"{run['think_pct']:.0f}%"), P(note, cell_left),
    ])

tpt = Table(tp_rows, colWidths=[20*mm, 14*mm, 14*mm, 14*mm, 16*mm, 16*mm, 13*mm, 40*mm], repeatRows=1)
tpt_cmds = [
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("FONTSIZE", (0, 1), (-1, -1), 7),
]
# Find fastest wall per prompt for highlighting
for pid in ["reasoning", "creative", "coding"]:
    runs_for_pid = [r for r in vs_tp["runs"] if r["prompt_id"] == pid]
    fastest = min(r["wall_time"] for r in runs_for_pid)
    for i, run in enumerate(vs_tp["runs"]):
        if run["prompt_id"] == pid and abs(run["wall_time"] - fastest) < 0.5:
            row_idx = i + 1
            tpt_cmds.append(("BACKGROUND", (3, row_idx), (3, row_idx), BEST_BG))
for i in range(1, len(tp_rows)):
    run = vs_tp["runs"][i - 1]
    tpt_cmds.append(("BACKGROUND", (6, i), (6, i), color_for_think(run["think_pct"])))
    if i % 2 == 0:
        tpt_cmds.append(("BACKGROUND", (0, i), (2, i), ALT_ROW))
tpt.setStyle(TableStyle(tpt_cmds))
elements.append(tpt)

elements.append(Spacer(1, 3*mm))
elements.append(Paragraph(
    f"<b>Finding:</b> {tp_quality['best_reasoning']['note']} "
    f"Creative quality is robust (all 5/5 scene elements across all values). "
    f"Coding is equally robust (7/7 code elements everywhere). The differentiator is speed: "
    f"top_p=0.90 is 2x slower for reasoning (126s) but fastest for creative (33s). "
    f"Recommendation: {tp_quality['recommendation']}",
    note_style
))

# =====================================================================
# PAGE 5: Variable Sweep — top_k
# =====================================================================

elements.append(PageBreak())
elements.append(Paragraph("4. Variable Sweep — top_k (15 runs)", section_style))
vs_tk = VAR_SWEEP["sweeps"]["top_k"]
elements.append(Paragraph(
    f"{vs_tk['description']} Values tested: {vs_tk['values']}. "
    "Fixed: top_p=0.95, repeat_penalty=1.1, ctx=32768, n_tokens=4096.",
    small_note
))

tk_notes = {
    "reasoning": {"0": "3/5 proof, 18.4 t/s (SLOW)", "20": "4/5 proof, 54s", "40": "4/5 proof, 98s (SLOWEST)",
                  "60": "5/5 proof, 476ch ans", "80": "4/5 proof, 37s (FASTEST)"},
    "creative": {"0": "3/5 scene, 18.3 t/s", "20": "3/5 scene, 41s", "40": "3/5 scene, 45s",
                 "60": "5/5 scene, 4217ch (most)", "80": "5/5 scene, 33s (fastest)"},
    "coding": {"0": "7/7 code, 229s (SLOWEST)", "20": "7/7 code, 211s", "40": "7/7 code, 127s",
               "60": "7/7 code, 209s", "80": "7/7 code, 182s"},
}

tk_header = [
    P("Prompt", header_left), P("top_k", header_cell), P("Gen\ntok/s", header_cell),
    P("Wall\nsec", header_cell), P("Think\nchars", header_cell), P("Answer\nchars", header_cell),
    P("Think\n%", header_cell), P("Notes", header_left),
]
tk_rows = [tk_header]
for run in vs_tk["runs"]:
    val_key = str(run["top_k"])
    note = tk_notes.get(run["prompt_id"], {}).get(val_key, "")
    tk_rows.append([
        P(run["prompt_id"], cell_left), P(str(run["top_k"])),
        P(f"{run['gen_tps']:.1f}"), P(f"{run['wall_time']:.1f}"),
        P(str(run["think_chars"])), P(str(run["ans_chars"])),
        P(f"{run['think_pct']:.0f}%"), P(note, cell_left),
    ])

tkt = Table(tk_rows, colWidths=[20*mm, 14*mm, 14*mm, 14*mm, 16*mm, 16*mm, 13*mm, 40*mm], repeatRows=1)
tkt_cmds = [
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("FONTSIZE", (0, 1), (-1, -1), 7),
]
for pid in ["reasoning", "creative", "coding"]:
    runs_for_pid = [r for r in vs_tk["runs"] if r["prompt_id"] == pid]
    fastest = min(r["wall_time"] for r in runs_for_pid)
    for i, run in enumerate(vs_tk["runs"]):
        if run["prompt_id"] == pid and abs(run["wall_time"] - fastest) < 0.5:
            row_idx = i + 1
            tkt_cmds.append(("BACKGROUND", (3, row_idx), (3, row_idx), BEST_BG))
for i in range(1, len(tk_rows)):
    run = vs_tk["runs"][i - 1]
    tkt_cmds.append(("BACKGROUND", (6, i), (6, i), color_for_think(run["think_pct"])))
    # Highlight top_k=0 slower gen speed
    if run.get("top_k") == 0 and run["gen_tps"] < 19.5:
        tkt_cmds.append(("BACKGROUND", (2, i), (2, i), BAD_BG))
    if i % 2 == 0:
        tkt_cmds.append(("BACKGROUND", (0, i), (2, i), ALT_ROW))
tkt.setStyle(TableStyle(tkt_cmds))
elements.append(tkt)

tk_quality = VAR_QUALITY["findings"]["top_k"]
elements.append(Spacer(1, 3*mm))
elements.append(Paragraph(
    f"<b>Finding:</b> {tk_quality['best_reasoning']['note']} "
    f"The common default top_k=40 is the <b>worst</b> value for reasoning (98s, slowest). "
    f"top_k=0 disables sampling and slows generation to ~18.3 t/s (vs 20.1 for any non-zero value). "
    f"Creative quality improves at top_k>=60 (skips thinking, produces more content, faster). "
    f"Coding is robust everywhere (7/7 code elements in all 15 runs). "
    f"Recommendation: {tk_quality['recommendation']}",
    note_style
))

# =====================================================================
# PAGE 6: Variable Sweep — repeat_penalty
# =====================================================================

elements.append(PageBreak())
elements.append(Paragraph("5. Variable Sweep — repeat_penalty (15 runs)", section_style))
vs_rp = VAR_SWEEP["sweeps"]["repeat_penalty"]
elements.append(Paragraph(
    f"{vs_rp['description']} Values tested: {vs_rp['values']}. "
    "Fixed: top_p=0.95, top_k=40, ctx=32768, n_tokens=4096.",
    small_note
))

rp_notes = {
    "reasoning": {"1.0": "2/5 proof (INCOMPLETE)", "1.05": "2/5 proof (INCOMPLETE)", "1.1": "5/5 proof (COMPLETE!)",
                  "1.15": "3/5 proof", "1.2": "3/5 proof, 161s (SLOW)"},
    "creative": {"1.0": "5/5 scene, 6499ch, 61s", "1.05": "5/5 scene, 15337ch (BLOATED), 160s",
                 "1.1": "5/5 scene, 5030ch, 47s", "1.15": "1/5 scene (LOOP), 4075ch think",
                 "1.2": "5/5 scene, 3633ch, 33s (FASTEST)"},
    "coding": {"1.0": "7/7 code, 10659ch, 121s", "1.05": "7/7 code, 9286ch, 112s",
               "1.1": "7/7 code, 11621ch, 130s", "1.15": "7/7 code, 8734ch, 94s",
               "1.2": "7/7 code, 5131ch, 49s (FASTEST)"},
}

rp_header = [
    P("Prompt", header_left), P("rep", header_cell), P("Gen\ntok/s", header_cell),
    P("Wall\nsec", header_cell), P("Think\nchars", header_cell), P("Answer\nchars", header_cell),
    P("Think\n%", header_cell), P("Notes", header_left),
]
rp_rows = [rp_header]
for run in vs_rp["runs"]:
    val_key = str(run["repeat_penalty"])
    note = rp_notes.get(run["prompt_id"], {}).get(val_key, "")
    rp_rows.append([
        P(run["prompt_id"], cell_left), P(str(run["repeat_penalty"])),
        P(f"{run['gen_tps']:.1f}"), P(f"{run['wall_time']:.1f}"),
        P(str(run["think_chars"])), P(str(run["ans_chars"])),
        P(f"{run['think_pct']:.0f}%"), P(note, cell_left),
    ])

rpt = Table(rp_rows, colWidths=[20*mm, 14*mm, 14*mm, 14*mm, 16*mm, 16*mm, 13*mm, 40*mm], repeatRows=1)
rpt_cmds = [
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("FONTSIZE", (0, 1), (-1, -1), 7),
]
for pid in ["reasoning", "creative", "coding"]:
    runs_for_pid = [r for r in vs_rp["runs"] if r["prompt_id"] == pid]
    fastest = min(r["wall_time"] for r in runs_for_pid)
    for i, run in enumerate(vs_rp["runs"]):
        if run["prompt_id"] == pid and abs(run["wall_time"] - fastest) < 0.5:
            row_idx = i + 1
            rpt_cmds.append(("BACKGROUND", (3, row_idx), (3, row_idx), BEST_BG))
for i in range(1, len(rp_rows)):
    run = vs_rp["runs"][i - 1]
    rpt_cmds.append(("BACKGROUND", (6, i), (6, i), color_for_think(run["think_pct"])))
    # Highlight INCOMPLETE proofs
    if run["prompt_id"] == "reasoning" and run["repeat_penalty"] in [1.0, 1.05]:
        rpt_cmds.append(("BACKGROUND", (7, i), (7, i), BAD_BG))
    # Highlight COMPLETE proof
    if run["prompt_id"] == "reasoning" and run["repeat_penalty"] == 1.1:
        rpt_cmds.append(("BACKGROUND", (7, i), (7, i), BEST_BG))
    if i % 2 == 0:
        rpt_cmds.append(("BACKGROUND", (0, i), (2, i), ALT_ROW))
rpt.setStyle(TableStyle(rpt_cmds))
elements.append(rpt)

rp_quality = VAR_QUALITY["findings"]["repeat_penalty"]
elements.append(Spacer(1, 3*mm))
elements.append(Paragraph(
    f"<b>Finding:</b> {rp_quality['best_reasoning']['note']} "
    f"repeat_penalty=1.10 is the <b>only</b> value that produces a complete mathematical proof (5/5 elements). "
    f"Values 1.0 and 1.05 produce incomplete proofs (missing even/odd argument and p^2=2q^2 step). "
    f"For creative and coding, repeat_penalty=1.20 is universally faster (33-49s vs 47-130s) with identical quality. "
    f"This is the most impactful variable: reasoning proof completeness varies from 2/5 to 5/5.",
    note_style
))

# =====================================================================
# PAGE 8: Tuned Retest — 10 Prompts with Best Settings
# =====================================================================

elements.append(PageBreak())
elements.append(Paragraph("7. Tuned Retest — 10 Prompts with Best Settings", section_style))
elements.append(Paragraph(
    "The original 10 benchmark prompts were re-run with the tuned per-category settings from all 65 prior experiments. "
    "Each prompt was classified by auto_temp.py, then run with its category's best temperature, top_p, top_k, and "
    "repeat_penalty. Context remained 32K; thinking tokens were 8192. This is the validation round.",
    small_note
))

# Tuned retest comparison table
rt_header = [
    P("Prompt", header_left), P("Style", header_cell), P("Settings", header_left),
    P("Wall\nsec", header_cell), P("Total\nchars", header_cell),
    P("Orig\n1-10", header_cell), P("Tuned\n1-10", header_cell), P("Delta", header_cell),
]
rt_rows = [rt_header]
rt_scores = TUNED_QUALITY["scores"]
for run in TUNED_RETEST["runs"]:
    pid = run["id"]
    if "error" in run:
        rt_rows.append([P(pid, cell_left), P("ERROR"), P("-"), P("-"), P("-"), P("-"), P("-"), P("-")])
        continue
    s = rt_scores[pid]
    stg = s["settings"]
    settings_str = f"t={stg['temp']} p={stg['top_p']} k={stg['top_k']} r={stg['repeat_penalty']}"
    rt_rows.append([
        P(pid, cell_left), P(s["style"], cell_left), P(settings_str, cell_left),
        P(f"{s['wall_time']:.1f}"), P(str(s["total_chars"])),
        P(str(s["orig_score"]), cell_bold), P(str(s["tuned_score"]), cell_bold),
        P(f"+{s['delta']}" if s["delta"] >= 0 else str(s["delta"]), cell_bold),
    ])

# Average row
rt_rows.append([
    P("AVERAGE", cell_bold_left), P(""), P(""), P(""), P(""),
    P(str(TUNED_QUALITY["average_original"]), cell_bold), P(str(TUNED_QUALITY["average_tuned"]), cell_bold),
    P(f"+{TUNED_QUALITY['delta']}", cell_bold),
])

rtt = Table(rt_rows, colWidths=[20*mm, 16*mm, 40*mm, 16*mm, 16*mm, 16*mm, 16*mm, 16*mm], repeatRows=1)
rtt_cmds = [
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("FONTSIZE", (0, 1), (-1, -1), 7),
]
# Color quality cells: green for 9-10, yellow for 5-8, red for <5
for i in range(1, len(rt_rows) - 1):
    pid = TUNED_RETEST["runs"][i - 1]["id"]
    if "error" in TUNED_RETEST["runs"][i - 1]:
        continue
    s = rt_scores[pid]
    # Original score color
    oq = s["orig_score"]
    if oq >= 8:
        rtt_cmds.append(("BACKGROUND", (5, i), (5, i), BEST_BG))
    elif oq <= 4:
        rtt_cmds.append(("BACKGROUND", (5, i), (5, i), BAD_BG))
    else:
        rtt_cmds.append(("BACKGROUND", (5, i), (5, i), GOOD_BG))
    # Tuned score color (all should be 9-10 = green)
    tq = s["tuned_score"]
    if tq >= 9:
        rtt_cmds.append(("BACKGROUND", (6, i), (6, i), BEST_BG))
    elif tq >= 7:
        rtt_cmds.append(("BACKGROUND", (6, i), (6, i), GOOD_BG))
    else:
        rtt_cmds.append(("BACKGROUND", (6, i), (6, i), BAD_BG))
    # Delta color (all positive = green)
    if s["delta"] >= 4:
        rtt_cmds.append(("BACKGROUND", (7, i), (7, i), BEST_BG))
    elif s["delta"] > 0:
        rtt_cmds.append(("BACKGROUND", (7, i), (7, i), GOOD_BG))
    if i % 2 == 0:
        rtt_cmds.append(("BACKGROUND", (0, i), (2, i), ALT_ROW))
# Average row
last = len(rt_rows) - 1
rtt_cmds.append(("BACKGROUND", (0, last), (-1, last), AVG_BG))
rtt_cmds.append(("FONTNAME", (0, last), (-1, last), "Helvetica-Bold"))
rtt.setStyle(TableStyle(rtt_cmds))
elements.append(rtt)

elements.append(Spacer(1, 4*mm))

# Big win callout
big_wins = sorted(rt_scores.items(), key=lambda x: x[1]["delta"], reverse=True)[:3]
win_text = "<b>Biggest improvements:</b> " + ", ".join(
    f"{pid} (+{s['delta']}: {s['orig_score']} to {s['tuned_score']})"
    for pid, s in big_wins
)
elements.append(Paragraph(win_text, finding_style))

elements.append(Paragraph(
    "<b>Result: all 10 prompts improved, zero regressions.</b> The average quality score rose from 6.4/10 to 9.8/10 "
    "(+53%). The biggest jumps were on the prompts that looped hardest at temp 1.0: creative writing (+7, from 3/10 "
    "to 10/10), iambic pentameter (+6, from 4/10 to 10/10), and TRS-80 BASIC (+5, from 4/10 to 9/10). "
    "The code-generation prompts (Python, C, HTML, Julia) that already scored 7-8/10 still gained 1-3 points from "
    "the tuned sampling parameters -- primarily from top_k=80 producing more complete code with less thinking overhead. "
    "The math proof gained a boxed final answer it lacked at default settings. "
    "Generation speed remained ~20 t/s across all 10 retest runs.",
    note_style
))

elements.append(Spacer(1, 3*mm))
elements.append(Paragraph(
    "<b>Caveat: speed did not improve.</b> The tuned settings prioritize quality, not speed. Several runs took longer "
    "than the original benchmark (e.g. code: 189s vs 107s, html: 237s vs 99s) because the model produced substantially "
    "more complete output (14K vs 8K chars for code, 18K vs 8K for HTML). The variable sweep speedups (top_k=80 = 2.6x) "
    "apply to the reasoning category specifically; the retest used 8192 tokens and 32K context for all categories, "
    "which is the quality-first configuration. For speed-sensitive use, use think-creative.sh with 4K context.",
    small_note
))

# =====================================================================
# PAGE 9: Tuned Settings Matrix + Recommendations
# =====================================================================

elements.append(PageBreak())
elements.append(Paragraph("8. Tuned Settings Matrix — Final Recommendations", section_style))
elements.append(Paragraph(
    "The auto-adjuster (auto_temp.py) classifies each prompt into one of five categories and applies the "
    "empirically-best settings from all 65 experiments. Context is always 32K for reasoning, 4K for creative.",
    small_note
))

# Tuned settings table
tuned_header = [
    P("Category", header_left), P("Temp", header_cell), P("top_p", header_cell),
    P("top_k", header_cell), P("rep", header_cell), P("Ctx", header_cell),
    P("Tokens", header_cell), P("Key result", header_left),
]
tuned_rows = [tuned_header]
tuned_data = [
    ("reasoning", "1.0", "0.80", "80", "1.10", "32K", "16K", "5/5 proof elements, 37s (was 98s at default)"),
    ("fiction", "0.9", "0.90", "80", "1.20", "4K", "1.2K", "5/5 scene elements, 33s (was 160s at rep=1.05)"),
    ("poetry", "0.3", "0.90", "80", "1.20", "4K", "1.2K", "Only temp that emits a poem (3/10 quality -- weak)"),
    ("prose", "0.5", "0.90", "80", "1.20", "4K", "2K", "8/10 quality, robust at every temperature tested"),
    ("code", "0.7", "1.00", "40", "1.20", "4K", "2K", "7/7 code elements, 49s (was 130s at rep=1.1)"),
]
for row in tuned_data:
    tuned_rows.append([P(c, cell_left if i in [0, 7] else cell_bold) for i, c in enumerate(row)])

tuned_t = Table(tuned_rows, colWidths=[20*mm, 14*mm, 14*mm, 14*mm, 14*mm, 14*mm, 14*mm, 50*mm], repeatRows=1)
tuned_t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("FONTSIZE", (0, 1), (-1, -1), 7.5),
    # Row colors
    ("BACKGROUND", (0, 1), (-1, 1), BEST_BG),  # reasoning (best tuned)
    ("BACKGROUND", (0, 2), (-1, 2), GOOD_BG),  # fiction
    ("BACKGROUND", (0, 3), (-1, 3), BAD_BG),   # poetry (weak)
    ("BACKGROUND", (0, 4), (-1, 4), BEST_BG),  # prose
    ("BACKGROUND", (0, 5), (-1, 5), GOOD_BG),  # code
]))
elements.append(tuned_t)

elements.append(Spacer(1, 6*mm))

# Before/after comparison
elements.append(Paragraph("Before & After: Default vs Tuned", sub_section))
ba_header = [
    P("Setting", header_left), P("Old default", header_cell), P("New (tuned)", header_cell),
    P("Impact", header_left),
]
ba_rows = [ba_header]
ba_data = [
    ("top_p", "0.95", "0.80 (reasoning) / 0.90 (creative) / 1.00 (code)", "Avoids 2x slowdown at 0.90 for reasoning; fastest per category"),
    ("top_k", "40", "80 (reasoning/creative) / 40 (code)", "2.6x faster reasoning (37s vs 98s); code keeps 40 to avoid over-generation"),
    ("repeat_penalty", "1.10", "1.10 (reasoning) / 1.20 (creative/code)", "Only 1.10 gives complete proofs; 1.20 is 2.6x faster for creative with same quality"),
    ("temperature", "1.0 (fixed)", "0.3-1.0 (per category)", "Fiction needs 0.9, poetry needs 0.3, prose 0.5, code 0.7, reasoning 1.0"),
]
for row in ba_data:
    ba_rows.append([P(row[0], cell_bold_left), P(row[1], cell_left), P(row[2], cell_left), P(row[3], cell_left)])

bat = Table(ba_rows, colWidths=[25*mm, 25*mm, 55*mm, 65*mm], repeatRows=1)
bat.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("FONTSIZE", (0, 1), (-1, -1), 7.5),
]))
elements.append(bat)

elements.append(Spacer(1, 6*mm))
elements.append(Paragraph("Methodology Notes", sub_section))
elements.append(Paragraph(
    "All 65 experiments used the same llama.cpp build, same Q4_K_M model, and 32K context (except creative tasks "
    "in think-creative.sh which use 4K to cap loop spiraling). Generation speed was consistent at ~20 t/s across "
    "all 65 runs (memory-bandwidth bound, not compute bound). Token counts were estimated from character counts "
    "divided by 4 (this llama.cpp build does not print n_eval in a parseable format). "
    "Quality scoring used automated keyword detection for proof elements (contradiction, even/odd, p^2=2q^2, "
    "boxed, gcd), scene elements (dialogue, setting, sensory, character, atmosphere), and code elements (def, "
    "merge, recursion, split, type_hints, docstring, edge_cases). "
    "The temperature sweep quality scores were manually verified by reading the full output text; the prose "
    "'100% loop' reading was corrected to 8/10 after manual review showed complete medical explanations lacking "
    "only the 'Final Answer' delimiter.",
    small_note
))

# ---- Build the document ----
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=landscape(A4),
    leftMargin=12*mm, rightMargin=12*mm,
    topMargin=12*mm, bottomMargin=12*mm,
)
doc.build(elements)
print(f"PDF written: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT)/1024:.0f} KB")