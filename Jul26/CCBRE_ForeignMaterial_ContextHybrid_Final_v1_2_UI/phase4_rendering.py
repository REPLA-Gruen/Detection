#!/usr/bin/env python3
"""Readable diagnostic rendering shared by test evaluation and visual app.

The original photo is never covered by long diagnostic text. A separate panel
is added above the image. The image itself contains only compact IDs (G1, P1, etc.) beside boxes.
Material identity is communicated by prediction-box color and explained in the Help legend.
"""
from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

from phase4_inference import sanitize_box


GT_MATCH_COLOR = (30, 160, 70, 255)
GT_MISS_COLOR = (245, 165, 25, 255)
PRED_MATCH_COLOR = (30, 115, 215, 255)
PRED_ERROR_COLOR = (220, 45, 45, 255)
PRED_UNCERTAIN_COLOR = (150, 85, 200, 255)

# Prediction colors are class-specific in the standalone/product rendering.
# GT colors remain status-oriented (matched/missed), while prediction colors
# communicate material identity at a glance.
FO_CLASS_COLORS = {
    'baw-sack': (0, 155, 230, 255),       # blue/cyan
    'papier': (240, 175, 0, 255),         # amber
    'plastiksack': (0, 175, 105, 255),    # green
    'other': (225, 75, 145, 255),         # pink/magenta
}
PANEL_BG = (28, 28, 28, 255)
PANEL_TEXT = (250, 250, 250, 255)
PANEL_MUTED = (205, 205, 205, 255)


def _class_key(name: str | None) -> str:
    if name is None:
        return ''
    return str(name).strip().casefold().replace('_', '-').replace(' ', '-')


def prediction_color(class_name: str | None):
    if class_name is None:
        return PRED_UNCERTAIN_COLOR
    key = _class_key(class_name)
    return FO_CLASS_COLORS.get(key, PRED_MATCH_COLOR)


def short_class_label(class_name: str | None) -> str:
    if class_name is None:
        return 'Uncertain'
    key = _class_key(class_name)
    labels = {
        'baw-sack': 'BAW-Sack',
        'papier': 'Papier',
        'plastiksack': 'Plastiksack',
        'other': 'Other',
    }
    return labels.get(key, str(class_name))


def get_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            (
                r"C:\Windows\Fonts\arialbd.ttf",
                "arialbd.ttf",
                "DejaVuSans-Bold.ttf",
            )
        )
    candidates.extend(
        (
            r"C:\Windows\Fonts\arial.ttf",
            "arial.ttf",
            "DejaVuSans.ttf",
        )
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return max(0, box[2] - box[0])


def text_height(draw: ImageDraw.ImageDraw, font) -> int:
    box = draw.textbbox((0, 0), "Ag", font=font)
    return max(1, box[3] - box[1])


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_tag(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font,
    color,
    image_size: tuple[int, int],
):
    width, height = image_size
    box = draw.textbbox((0, 0), text, font=font)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    pad_x = 4
    pad_y = 2
    left = max(0, min(int(x), max(0, width - tw - 2 * pad_x)))
    top = max(0, min(int(y), max(0, height - th - 2 * pad_y)))
    draw.rectangle(
        (
            left,
            top,
            left + tw + 2 * pad_x,
            top + th + 2 * pad_y,
        ),
        fill=color,
    )
    draw.text(
        (left + pad_x, top + pad_y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
    )


def _compose_panel(
    image: Image.Image,
    title: str,
    subtitle: str,
    sections: list[tuple[str, list[tuple[str, tuple[int, int, int, int] | None]]]],
) -> tuple[Image.Image, int]:
    """Create a top information panel and paste the original image below it."""
    width, height = image.size
    body_font = get_font(max(14, round(min(width, height) / 54)))
    bold_font = get_font(max(14, round(min(width, height) / 50)), bold=True)
    title_font = get_font(max(17, round(min(width, height) / 42)), bold=True)

    # Temporary draw context for measuring/wrapping.
    scratch = Image.new("RGBA", (width, 10), PANEL_BG)
    draw = ImageDraw.Draw(scratch, "RGBA")

    margin_x = max(10, round(width * 0.008))
    margin_y = 8
    available_width = width - 2 * margin_x
    line_gap = max(4, round(text_height(draw, body_font) * 0.28))
    section_gap = max(7, round(text_height(draw, body_font) * 0.45))

    panel_lines: list[dict[str, Any]] = []

    for line in wrap_text(draw, title, title_font, available_width):
        panel_lines.append(
            {
                "text": line,
                "font": title_font,
                "color": PANEL_TEXT,
                "swatch": None,
                "gap_after": 2,
            }
        )

    if subtitle:
        for line in wrap_text(draw, subtitle, body_font, available_width):
            panel_lines.append(
                {
                    "text": line,
                    "font": body_font,
                    "color": PANEL_MUTED,
                    "swatch": None,
                    "gap_after": 2,
                }
            )

    for section_name, entries in sections:
        if not entries:
            continue
        panel_lines.append(
            {
                "text": section_name,
                "font": bold_font,
                "color": PANEL_TEXT,
                "swatch": None,
                "gap_after": 2,
                "gap_before": section_gap,
            }
        )
        for entry_text, swatch in entries:
            # Reserve width for a small colored swatch.
            reserve = 22 if swatch is not None else 0
            wrapped = wrap_text(
                draw,
                entry_text,
                body_font,
                max(80, available_width - reserve),
            )
            for wrap_index, line in enumerate(wrapped):
                panel_lines.append(
                    {
                        "text": line,
                        "font": body_font,
                        "color": PANEL_TEXT,
                        "swatch": swatch if wrap_index == 0 else None,
                        "indent": reserve,
                        "gap_after": 1,
                    }
                )

    panel_height = margin_y
    for item in panel_lines:
        panel_height += int(item.get("gap_before", 0))
        panel_height += text_height(draw, item["font"])
        panel_height += line_gap + int(item.get("gap_after", 0))
    panel_height += margin_y

    canvas = Image.new(
        "RGB",
        (width, panel_height + height),
        PANEL_BG[:3],
    )
    canvas.paste(image, (0, panel_height))

    panel_draw = ImageDraw.Draw(canvas, "RGBA")
    y = margin_y
    for item in panel_lines:
        y += int(item.get("gap_before", 0))
        swatch = item.get("swatch")
        indent = int(item.get("indent", 0))
        if swatch is not None:
            swatch_size = max(10, text_height(panel_draw, item["font"]) - 2)
            swatch_y = y + 1
            panel_draw.rectangle(
                (
                    margin_x,
                    swatch_y,
                    margin_x + swatch_size,
                    swatch_y + swatch_size,
                ),
                fill=swatch,
            )
        panel_draw.text(
            (margin_x + indent, y),
            item["text"],
            font=item["font"],
            fill=item["color"],
        )
        y += text_height(panel_draw, item["font"])
        y += line_gap + int(item.get("gap_after", 0))

    # Separator between panel and photo.
    panel_draw.line(
        (0, panel_height - 1, width, panel_height - 1),
        fill=(120, 120, 120, 255),
        width=1,
    )
    return canvas, panel_height


def render_unlabeled(image: Image.Image, result: dict[str, Any]) -> Image.Image:
    predictions = []
    for index, detection in enumerate(result["detections"], 1):
        name = detection["class_name"] or "ForeignMaterial (class uncertain)"
        color = prediction_color(detection["class_name"])
        predictions.append(
            (
                f"P{index}: {name} | "
                f"Det={detection['detector_confidence']:.2f} | "
                f"Ver={detection['verifier_fo_probability']:.2f} | "
                f"Cls={detection['exact_confidence']:.2f}",
                color,
            )
        )

    title = (
        f"FOREIGN MATERIAL DETECTED | Verified FO={len(result['detections'])}"
        if result["detections"]
        else "NO FOREIGN MATERIAL DETECTED"
    )
    subtitle = (
        "Unlabeled inference. Det = detector confidence; "
        "Ver = FO verifier probability; Cls = exact-class confidence."
    )
    canvas, panel_height = _compose_panel(
        image,
        title,
        subtitle,
        [("Predictions", predictions)],
    )

    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = image.size
    line_width = max(2, round(min(width, height) / 300))
    tag_font = get_font(max(14, round(min(width, height) / 54)), bold=True)

    for index, detection in enumerate(result["detections"], 1):
        clean = sanitize_box(detection["bbox"], image.size)
        if clean is None:
            continue
        x1, y1, x2, y2 = clean
        y1 += panel_height
        y2 += panel_height
        color = prediction_color(detection["class_name"])
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width + 1,
        )
        draw_tag(
            draw,
            x1,
            max(panel_height, y1 - 24),
            f"P{index}",
            tag_font,
            color,
            canvas.size,
        )
    return canvas


def render_with_gt(
    image: Image.Image,
    result: dict[str, Any],
    gt: list[dict[str, Any]],
    exact_predictions: list[dict[str, Any]],
    same_matches: list[dict[str, Any]],
    names: dict[int, str],
    category: str,
    same_fp: int,
    same_fn: int,
) -> Image.Image:
    matched_gt = {
        int(match["gt_index"])
        for match in same_matches
    }
    matched_exact_pred = {
        int(match["prediction_index"])
        for match in same_matches
    }
    known_verified_indices = [
        index
        for index, detection in enumerate(result["detections"])
        if detection["class_name"] is not None
    ]
    exact_to_verified = {
        exact_index: verified_index
        for exact_index, verified_index in enumerate(known_verified_indices)
    }
    matched_verified = {
        exact_to_verified[index]
        for index in matched_exact_pred
        if index in exact_to_verified
    }

    gt_entries = []
    for index, gt_item in enumerate(gt, 1):
        matched = (index - 1) in matched_gt
        color = GT_MATCH_COLOR if matched else GT_MISS_COLOR
        gt_entries.append(
            (
                f"G{index}: {names[int(gt_item['class_id'])]} | "
                f"{'MATCH' if matched else 'MISS'}",
                color,
            )
        )

    prediction_entries = []
    for index, detection in enumerate(result["detections"], 1):
        verified_index = index - 1
        if detection["class_name"] is None:
            color = PRED_UNCERTAIN_COLOR
            name = "ForeignMaterial (class uncertain)"
            status = "UNCERTAIN"
        elif verified_index in matched_verified:
            color = prediction_color(detection["class_name"])
            name = detection["class_name"]
            status = "MATCH"
        else:
            color = prediction_color(detection["class_name"])
            name = detection["class_name"]
            status = "WRONG / EXTRA"

        prediction_entries.append(
            (
                f"P{index}: {name} | {status} | "
                f"Det={detection['detector_confidence']:.2f} | "
                f"Ver={detection['verifier_fo_probability']:.2f} | "
                f"Cls={detection['exact_confidence']:.2f}",
                color,
            )
        )

    title = (
        f"{category} | GT={len(gt)} | "
        f"VerifiedFO={len(result['detections'])} | "
        f"Exact={len(exact_predictions)} | "
        f"TP={len(same_matches)} | FP={same_fp} | FN={same_fn}"
    )
    subtitle = (
        "GT: green = matched, orange = missed. "
        "PRED box color = material class; purple = class uncertain; "
        "a red inner outline marks wrong/extra predictions in GT comparison. "
        "Det = detector confidence; Ver = FO verifier probability; "
        "Cls = exact-class confidence."
    )
    canvas, panel_height = _compose_panel(
        image,
        title,
        subtitle,
        [
            ("Ground truth", gt_entries),
            ("Predictions", prediction_entries),
        ],
    )

    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = image.size
    line_width = max(2, round(min(width, height) / 300))
    tag_font = get_font(max(14, round(min(width, height) / 54)), bold=True)

    for index, gt_item in enumerate(gt, 1):
        clean = sanitize_box(gt_item["bbox"], image.size)
        if clean is None:
            continue
        x1, y1, x2, y2 = clean
        y1 += panel_height
        y2 += panel_height
        matched = (index - 1) in matched_gt
        color = GT_MATCH_COLOR if matched else GT_MISS_COLOR
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width + 1,
        )
        draw_tag(
            draw,
            x1,
            max(panel_height, y1 - 24),
            f"G{index}",
            tag_font,
            color,
            canvas.size,
        )

    for index, detection in enumerate(result["detections"], 1):
        clean = sanitize_box(detection["bbox"], image.size)
        if clean is None:
            continue
        x1, y1, x2, y2 = clean
        y1 += panel_height
        y2 += panel_height

        verified_index = index - 1
        color = prediction_color(detection["class_name"])

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width + 1,
        )
        # Preserve audit visibility: exact-class wrong/extra predictions retain
        # their class color but receive a thin red inner outline.
        if detection["class_name"] is not None and verified_index not in matched_verified:
            inset = max(2, line_width + 1)
            if (x2 - x1) > 2 * inset and (y2 - y1) > 2 * inset:
                draw.rectangle(
                    (x1 + inset, y1 + inset, x2 - inset, y2 - inset),
                    outline=PRED_ERROR_COLOR,
                    width=max(1, line_width),
                )
        draw_tag(
            draw,
            x1,
            min(canvas.height - 24, y1 + 3),
            f"P{index}",
            tag_font,
            color,
            canvas.size,
        )

    return canvas
