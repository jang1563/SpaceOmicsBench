#!/usr/bin/env python3
"""Generate public README/Hugging Face visual assets for SpaceOmicsBench."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "spaceomicsbench_summary.png"
WIDTH = 1600
HEIGHT = 920

FONT_REGULAR_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]
FONT_BOLD_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]

COLORS = {
    "bg": "#f6f8fb",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#5d6879",
    "line": "#d8e0ea",
    "navy": "#17324d",
    "teal": "#0f9f9a",
    "blue": "#2f6fed",
    "violet": "#7c3aed",
    "amber": "#f59e0b",
    "green": "#22a06b",
    "gray": "#8896a8",
}

MODEL_LABELS = {
    "rf": "RF",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "logreg": "LogReg",
    "mlp": "MLP",
}

LLM_SCORES = [
    ("Claude Sonnet 4.6", 4.62),
    ("Claude Haiku 4.5", 4.41),
    ("DeepSeek-V3", 4.34),
    ("Claude Sonnet 4", 4.03),
    ("Gemini 2.5 Flash", 4.00),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_REGULAR_CANDIDATES
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=face)
    return box[2] - box[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    for word in text.split():
        trial = " ".join(current + [word])
        if current and text_width(draw, trial, face) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    face: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, face, max_width):
        draw.text((x, y), line, font=face, fill=fill)
        y += face.size + line_gap
    return y


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, radius: int = 26) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 1)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int = 28) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 4, y1 + 6, x2 + 4, y2 + 6), radius=radius, fill="#e6ebf2")
    rounded(draw, box, COLORS["panel"], COLORS["line"], radius)


def metric_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, value: str, caption: str, accent: str) -> None:
    card(draw, box, radius=22)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1 + 24, y1 + 24, x1 + 72, y1 + 34), radius=5, fill=accent)
    draw.text((x1 + 24, y1 + 50), value, font=font(43, True), fill=COLORS["ink"])
    draw.text((x1 + 24, y1 + 105), label.upper(), font=font(17, True), fill=accent)
    multiline(draw, (x1 + 24, y1 + 132), caption, font(16), COLORS["muted"], x2 - x1 - 48, 3)


def load_composite() -> list[tuple[str, float]]:
    data = json.loads((ROOT / "baselines" / "baseline_results.json").read_text())
    composite = data["_composite"]
    wanted = ["rf", "xgboost", "lightgbm", "logreg", "mlp"]
    return [(MODEL_LABELS[key], composite[key]["composite"]) for key in wanted]


def count_questions() -> int:
    payload = json.loads((ROOT / "evaluation" / "llm" / "question_bank.json").read_text())
    if isinstance(payload, list):
        return len(payload)
    return len(payload.get("questions", []))


def draw_title(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, subtitle: str) -> None:
    draw.text((x, y), title, font=font(31, True), fill=COLORS["ink"])
    draw.text((x, y + 42), subtitle, font=font(19), fill=COLORS["muted"])


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    rows: list[tuple[str, float]],
    max_value: float,
    xlabel: str,
    palette: list[str],
) -> None:
    card(draw, box, radius=28)
    x1, y1, x2, y2 = box
    draw_title(draw, x1 + 34, y1 + 28, title, subtitle)

    label_x = x1 + 34
    bar_x = x1 + 250
    bar_y = y1 + 122
    bar_h = 28
    row_gap = 48
    bar_w = x2 - bar_x - 132

    axis_y = y2 - 54
    draw.line((bar_x, axis_y, bar_x + bar_w, axis_y), fill=COLORS["line"], width=2)
    draw.text((bar_x, axis_y + 16), "0", font=font(15), fill=COLORS["muted"])
    draw.text((bar_x + bar_w - 38, axis_y + 16), f"{max_value:g}", font=font(15), fill=COLORS["muted"])
    draw.text((bar_x + 90, axis_y + 16), xlabel, font=font(15), fill=COLORS["muted"])

    for i, (label, value) in enumerate(rows):
        y = bar_y + i * row_gap
        color = palette[i % len(palette)]
        draw.text((label_x, y - 4), label, font=font(21, i == 0), fill=COLORS["ink"])
        draw.rounded_rectangle((bar_x, y, bar_x + bar_w, y + bar_h), radius=13, fill="#edf2f7")
        value_w = max(4, int(bar_w * (value / max_value)))
        draw.rounded_rectangle((bar_x, y, bar_x + value_w, y + bar_h), radius=13, fill=color)
        draw.text((bar_x + value_w + 14, y - 1), f"{value:.3f}" if max_value < 1 else f"{value:.2f}", font=font(20, True), fill=COLORS["ink"])


def main() -> None:
    composite = load_composite()
    questions = count_questions()
    tasks = len(list((ROOT / "tasks").glob("*.json")))
    processed = len(list((ROOT / "data" / "processed").glob("*.csv")))
    llm_runs = len(list((ROOT / "results" / "v2.1").glob("*.json")))

    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"])
    draw = ImageDraw.Draw(image)

    header = (56, 44, 1544, 292)
    card(draw, header, radius=34)
    draw.text((96, 84), "SpaceOmicsBench", font=font(58, True), fill=COLORS["ink"])
    draw.text((99, 154), "Multi-omics AI benchmark for spaceflight biomedical data", font=font(28), fill=COLORS["muted"])
    multiline(
        draw,
        (99, 203),
        "Public v2.1 package: processed OSDR / published-table benchmark data, task specs, splits, baselines, and LLM evaluation artifacts.",
        font(20),
        COLORS["muted"],
        660,
        5,
    )
    draw.rounded_rectangle((99, 248, 302, 276), radius=14, fill="#e8f7f5")
    draw.text((119, 253), "HF + GitHub ready", font=font(16, True), fill=COLORS["teal"])

    metric_card(draw, (825, 84, 1048, 252), "ML tasks", f"{tasks}", "19 main + 2 supplementary", COLORS["teal"])
    metric_card(draw, (1078, 84, 1301, 252), "Modalities", "9", "omics, clinical, cross-mission", COLORS["violet"])
    metric_card(draw, (1331, 84, 1504, 252), "LLM QA", f"{questions}", f"{llm_runs} scored runs", COLORS["amber"])

    draw_bar_panel(
        draw,
        (56, 335, 760, 690),
        "ML baseline composite",
        "Category-balanced normalized score",
        composite,
        0.30,
        "normalized composite",
        [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["violet"], COLORS["gray"]],
    )
    draw_bar_panel(
        draw,
        (840, 335, 1544, 690),
        "LLM spaceflight QA",
        "Top scored v2.1 runs across 100 questions",
        LLM_SCORES,
        5.0,
        "Claude-as-judge score",
        [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["violet"], COLORS["amber"]],
    )

    footer = (56, 732, 1544, 876)
    card(draw, footer, radius=28)
    draw.text((96, 770), "Open-track package boundary", font=font(28, True), fill=COLORS["ink"])
    multiline(
        draw,
        (96, 815),
        f"{processed} processed CSV tables | task JSON | split JSON | LLM question bank | scored v2.1 results | canonical baselines",
        font(19),
        COLORS["muted"],
        850,
        5,
    )
    multiline(
        draw,
        (940, 770),
        "Sequence-level or restricted human files are excluded. Controlled-access analyses should go back to OSDR/DAR or the original source.",
        font(19),
        COLORS["muted"],
        520,
        5,
    )
    draw.rounded_rectangle((940, 836, 1468, 864), radius=14, fill="#eef2ff")
    draw.text((962, 841), "HF viewer disabled for mixed CSV/JSON artifacts.", font=font(16, True), fill=COLORS["violet"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
