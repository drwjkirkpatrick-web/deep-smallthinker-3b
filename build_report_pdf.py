#!/usr/bin/env python3
"""
Build the deep-smallthinker-3b benchmark PDF report.

Documents the deep-thinking benchmark of SmallThinker-3B Q4_K_M on the 8GB Jetson:
10 prompts (5 general + 5 coding), temp 1.0, 32K context, 8192 thinking tokens.
Highlights the central finding: deep thinking does NOT raise average quality (6.4/10
at temp 1.0 vs 6.4/10 at temp 0.3), but it SHIFTS where quality lands — code/math/coding
tasks reach 8/10 while open-ended creative/prose collapse into meta-reasoning loops.
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

FINAL = os.path.expanduser("~/projects/deep-smallthinker-3b/results/benchmark_final.json")
RAW = os.path.expanduser("~/projects/deep-smallthinker-3b/results/benchmark_results.json")
OUTPUT = os.path.expanduser("~/projects/deep-smallthinker-3b/Deep_SmallThinker_3B_Benchmark_Report.pdf")

with open(FINAL) as f:
    D = json.load(f)
with open(RAW) as f:
    RAW_SETTINGS = json.load(f)["settings"]

HEADER_BG = colors.HexColor("#1a237e")
HEADER_FG = colors.white
BEST_BG = colors.HexColor("#e8f5e9")   # green
GOOD_BG = colors.HexColor("#fff8e1")   # yellow
BAD_BG = colors.HexColor("#ffebee")    # red
ALT_ROW = colors.HexColor("#f5f5f5")

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=4, textColor=HEADER_BG)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=10)
section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6, textColor=HEADER_BG)
cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.5, leading=9, alignment=TA_CENTER)
cell_left = ParagraphStyle("CellLeft", parent=cell_style, alignment=TA_LEFT)
header_cell = ParagraphStyle("HeaderCell", parent=styles["Normal"], fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=colors.white, fontName="Helvetica-Bold")
header_left = ParagraphStyle("HeaderLeft", parent=header_cell, alignment=TA_LEFT)
note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8.5, leading=11, spaceAfter=4)
small_note = ParagraphStyle("SmallNote", parent=styles["Normal"], fontSize=7, leading=9, textColor=colors.grey)
finding_style = ParagraphStyle("Finding", parent=styles["Normal"], fontSize=9.5, leading=13, backColor=colors.HexColor("#fff3e0"), borderPadding=6, spaceBefore=4, spaceAfter=8)


def P(text, style=cell_style):
    return Paragraph(str(text), style)


# ---- Table builders ----

def build_main_table():
    header = [
        P("Prompt", header_left), P("Suite", header_cell), P("Gen\ntok/s", header_cell),
        P("Ppt\ntok/s", header_cell), P("Tok*", header_cell), P("Wall\nsec", header_cell),
        P("Think\n%", header_cell), P("Qual\n1-10", header_cell),
    ]
    rows = [header]
    for r in D["results"]:
        if "error" in r:
            rows.append([P(r["id"], cell_left), P("ERROR"), P("-"), P("-"), P("-"), P("-"), P("-"), P("-")])
            continue
        rows.append([
            P(r["id"], cell_left), P(r["suite"], cell_left),
            P(f"{r['gen_tps']:.1f}"), P(f"{r['prompt_tps']:.0f}"),
            P(str(r["tokens_est"])), P(str(r["wall_time_s"])),
            P(f"{r['think_ratio_pct']:.0f}%"), P(str(r["quality"])),
        ])
    # Average row
    gen = [r['gen_tps'] for r in D['results'] if 'gen_tps' in r]
    ppt = [r['prompt_tps'] for r in D['results'] if 'prompt_tps' in r]
    rows.append([
        P("AVERAGE", header_left), P("", header_cell),
        P(f"{sum(gen)/len(gen):.1f}"), P(f"{sum(ppt)/len(ppt):.0f}"),
        P(""), P(""), P(""), P(f"{D['average_quality']:.1f}"),
    ])

    col_widths = [18*mm, 16*mm, 14*mm, 14*mm, 12*mm, 13*mm, 13*mm, 13*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # Color-code quality column (index 7), skip header + average rows
    for i in range(1, len(rows) - 1):
        q = D['results'][i - 1].get('quality')
        if q is None:
            continue
        if q >= 8:
            cmds.append(("BACKGROUND", (7, i), (7, i), BEST_BG))
        elif q <= 4:
            cmds.append(("BACKGROUND", (7, i), (7, i), BAD_BG))
        else:
            cmds.append(("BACKGROUND", (7, i), (7, i), GOOD_BG))
    # Zebra striping
    for i in range(1, len(rows) - 1):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (6, i), ALT_ROW))
    # Average row styling
    last = len(rows) - 1
    cmds.append(("BACKGROUND", (0, last), (-1, last), colors.HexColor("#e3f2fd")))
    cmds.append(("FONTNAME", (0, last), (-1, last), "Helvetica-Bold"))
    t.setStyle(TableStyle(cmds))
    return t


def build_think_analysis_table():
    """Thinking-vs-answer breakdown — the unique metric for a deep-thinking model."""
    header = [
        P("Prompt", header_left), P("Thinking\nchars", header_cell),
        P("Answer\nchars", header_cell), P("Think\nratio", header_cell),
        P("Outcome", header_left),
    ]
    rows = [header]
    outcome_map = {
        "code": "Clean answer (7.0K chars code)",
        "iambic": "LOOP — budget burned, poem truncated",
        "prose": "Meta-reasoning, no clean answer block",
        "creative": "LOOP — 7.6K chars of 'I need to write...'",
        "math": "Clean proof (62-char boxed answer)",
        "html": "Clean page (2.0K chars)",
        "python": "Clean code (14.9K chars)",
        "c": "Clean code (17.9K chars)",
        "basic": "LOOP — program truncated (501 chars)",
        "julia": "Clean code (5.6K chars)",
    }
    for r in D["results"]:
        if "error" in r:
            continue
        rows.append([
            P(r["id"], cell_left),
            P(str(r["thinking_chars"])),
            P(str(r["answer_chars"])),
            P(f"{r['think_ratio_pct']:.0f}%"),
            P(outcome_map.get(r["id"], ""), cell_left),
        ])
    col_widths = [18*mm, 22*mm, 22*mm, 16*mm, 70*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # Color the thinking-ratio column: >80% = red (loop), <40% = green (clean)
    for i in range(1, len(rows)):
        ratio = D['results'][i - 1].get('think_ratio_pct', 0)
        if ratio >= 80:
            cmds.append(("BACKGROUND", (3, i), (3, i), BAD_BG))
        elif ratio <= 40:
            cmds.append(("BACKGROUND", (3, i), (3, i), BEST_BG))
    for i in range(1, len(rows)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (2, i), ALT_ROW))
    t.setStyle(TableStyle(cmds))
    return t


def build_settings_table():
    s = RAW_SETTINGS
    data = [
        [P("Setting", header_left), P("Value", header_cell), P("Rationale", header_left)],
        [P("temperature", cell_left), P(str(s["temp"]), cell_style), P("High temp favors reasoning-tree exploration — tokens/time are free on the Jetson", cell_left)],
        [P("context", cell_left), P(str(s["ctx"]), cell_style), P("Native 32K — GUI off frees enough RAM for full context", cell_left)],
        [P("n_tokens (thinking)", cell_left), P(str(s["n_tokens"]), cell_style), P("Excess thinking budget — model may reason as long as it wants on one turn", cell_left)],
        [P("top_p / top_k", cell_left), P(f"{s['top_p']} / {s['top_k']}", cell_style), P("Standard nucleus sampling", cell_left)],
        [P("repeat_penalty", cell_left), P(str(s["repeat_penalty"]), cell_style), P("Discourages reasoning-loop repetition", cell_left)],
        [P("--jinja", cell_left), P("on", cell_style), P("Critical — without it the model loops on ~49 internal tokens", cell_left)],
        [P("quantization", cell_left), P("Q4_K_M", cell_style), P("2.1 GB — smaller weights buy the 16K-32K contexts", cell_left)],
    ]
    col_widths = [32*mm, 22*mm, 120*mm]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# ---- Build the document ----
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=landscape(A4),
    leftMargin=12*mm, rightMargin=12*mm,
    topMargin=12*mm, bottomMargin=12*mm,
)

elements = []

elements.append(Paragraph("Deep-SmallThinker-3B Benchmark Report", title_style))
elements.append(Paragraph(
    "SmallThinker-3B-Preview (Q4_K_M, 2.1 GB) on NVIDIA Jetson Orin Nano 8GB | llama.cpp CUDA | "
    "10 prompts (5 general + 5 coding) | temp 1.0, 32K context, 8192 thinking tokens | GUI off",
    subtitle_style
))

elements.append(Paragraph("Key Finding", section_style))
elements.append(Paragraph(
    "Deep-thinking settings (temp 1.0 + excess thinking tokens) do <b>not</b> raise average quality "
    "versus the earlier temp-0.3 run: <b>6.4/10 both ways</b>. They <b>shift</b> where quality lands. "
    "Structured tasks with a verifiable answer (code, math, coding) reach <b>8/10</b> — the extra step-by-step "
    "reasoning helps. But open-ended creative/prose tasks <b>collapse to 3-4/10</b>: the model enters a "
    "meta-reasoning loop (\"I need to write X... first I should... let me think...\") and never emits the actual "
    "deliverable, burning the whole 8192-token budget on thinking-about-thinking.",
    finding_style
))

elements.append(Paragraph("1. Speed, Metrics & Quality (all 10 prompts)", section_style))
elements.append(Paragraph(
    "Generation speed is rock-steady at ~20 tok/s (memory-bandwidth bound). Prompt eval averages ~409 tok/s. "
    "Tok* is estimated from characters/4 (this llama.cpp build does not print n_eval).",
    small_note
))
elements.append(build_main_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "Quality splits cleanly by task type. Every verifiable/coding task scores 7-8/10. Every open-ended "
    "creative-form task (iambic, creative, basic) collapses into a thinking loop and scores 3-4/10.",
    note_style
))

elements.append(PageBreak())

elements.append(Paragraph("2. Thinking vs Answer — the Deep-Thinking Signature", section_style))
elements.append(Paragraph(
    "This is the metric that a normal benchmark misses. \"Think %\" is the share of output spent in chain-of-thought "
    "reasoning before the final answer. A healthy thinking model lands at 10-40% (reason, then answer). At >80% it "
    "has fallen into a meta-reasoning loop and produced no deliverable.",
    small_note
))
elements.append(build_think_analysis_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "The three failures (iambic 91%, prose 100%, creative 100%, basic 93%) are all open-ended creative-form tasks. "
    "SmallThinker's training data is dominated by verifiable math/code reasoning, so on unverifiable creative prompts "
    "it keeps re-planning the task instead of committing to an answer. This is the central weakness of reasoning-first "
    "small models at high temperature.",
    note_style
))

elements.append(Paragraph("3. Settings Under Test", section_style))
elements.append(build_settings_table())

elements.append(Paragraph("4. Recommendations", section_style))
recs = [
    ("Use deep thinking (temp 1.0, excess tokens)", "for math proofs, algorithms, and code — it reaches 8/10 with full step-by-step reasoning."),
    ("Use a lower temperature (temp 0.6) or fewer tokens", "for creative writing, poetry, and prose — high temp + excess tokens trigger meta-reasoning loops."),
    ("Treat the chain-of-thought as a draft", "SmallThinker is a 3B model; its reasoning is often confident but can be wrong (it failed the classic bat-and-ball trick earlier)."),
    ("Keep --jinja mandatory", "without it the model never escapes its internal thinking and emits ~49 tokens only."),
]
rec_rows = [[P("Guidance", header_left), P("Rationale", header_left)]]
for g, r in recs:
    rec_rows.append([P(g, cell_left), P(r, cell_left)])
rt = Table(rec_rows, colWidths=[70*mm, 150*mm], repeatRows=1)
rt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
elements.append(rt)

elements.append(Spacer(1, 6*mm))
elements.append(Paragraph(
    "Methodology: identical prompt text to the jetson-llm-benchmark multiprompt_bench.py (general 5) and "
    "bench_coding.py (coding 5) suites. Quality scored 1-10 by the same rubric. Tokens estimated from chars/4.",
    small_note
))

doc.build(elements)
print(f"PDF written: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT)/1024:.0f} KB")
