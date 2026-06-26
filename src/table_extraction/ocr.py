from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def preprocess_cell_for_ocr(
    image: Image.Image,
    *,
    scale: int = 3,
    padding: int = 8,
) -> Image.Image:
    """Prepare one cell crop for OCR while preserving readable text."""
    grayscale = ImageOps.grayscale(image)
    if scale > 1:
        grayscale = grayscale.resize(
            (grayscale.width * scale, grayscale.height * scale),
            Image.Resampling.LANCZOS,
        )

    padded = ImageOps.expand(grayscale, border=padding, fill=255)
    arr = cv2.GaussianBlur(np.array(padded), (3, 3), 0)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def ocr_cell_image(
    image: Image.Image,
    *,
    tesseract_cmd: str | None = None,
    lang: str = "eng",
    psm: int = 6,
) -> str:
    """OCR one PIL image using the local Tesseract CLI."""
    tesseract_cmd = _resolve_tesseract_cmd(tesseract_cmd)
    prepared = preprocess_cell_for_ocr(image)

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        prepared.save(tmp.name)
        command = [
            tesseract_cmd,
            tmp.name,
            "stdout",
            "-l",
            lang,
            "--psm",
            str(psm),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Tesseract OCR failed: {result.stderr.strip()}")

    return result.stdout.strip()


def ocr_cell_crops(
    cell_crops: list[dict[str, Any]],
    *,
    rows: Iterable[int] | None = None,
    cols: Iterable[int] | None = None,
    max_cells: int = 16,
    tesseract_cmd: str | None = None,
    lang: str = "eng",
    psm: int = 6,
) -> pd.DataFrame:
    """OCR a selected sample of cell crops and return raw text for review."""
    row_filter = set(rows) if rows is not None else None
    col_filter = set(cols) if cols is not None else None

    selected = []
    for crop in sorted(cell_crops, key=lambda item: (item["row"], item["col"])):
        if row_filter is not None and crop["row"] not in row_filter:
            continue
        if col_filter is not None and crop["col"] not in col_filter:
            continue
        selected.append(crop)
        if len(selected) >= max_cells:
            break

    records = []
    for crop in selected:
        text = ocr_cell_image(
            crop["image"],
            tesseract_cmd=tesseract_cmd,
            lang=lang,
            psm=psm,
        )
        records.append(
            {
                "row": crop["row"],
                "col": crop["col"],
                "text": text,
                "path": str(crop["path"]) if crop.get("path") else None,
            }
        )

    return pd.DataFrame(records)


def show_ocr_sample_cells(
    cell_crops: list[dict[str, Any]],
    ocr_df: pd.DataFrame,
    *,
    columns: int = 4,
) -> None:
    """Show OCR sample cells with row/column labels."""
    if ocr_df.empty:
        print("No OCR samples to display.")
        return

    by_position = {(crop["row"], crop["col"]): crop for crop in cell_crops}
    rows = ocr_df.to_dict(orient="records")
    columns = max(1, min(columns, len(rows)))
    fig_rows = (len(rows) + columns - 1) // columns
    fig, axes = plt.subplots(fig_rows, columns, figsize=(4 * columns, 2.4 * fig_rows))
    axes_array = [axes] if len(rows) == 1 else axes.ravel()

    for ax, record in zip(axes_array, rows):
        crop = by_position.get((record["row"], record["col"]))
        if crop is not None:
            ax.imshow(crop["image"])
        ax.set_title(f"r{record['row']} c{record['col']}", fontsize=9)
        ax.axis("off")

    for ax in axes_array[len(rows):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_ocr_sample_cells_with_text(
    cell_crops: list[dict[str, Any]],
    ocr_df: pd.DataFrame,
    *,
    columns: int | None = None,
    max_text_chars: int = 100,
    max_rows_per_figure: int = 8,
    annotation_fontsize: int = 13,
    image_transform: Callable[[Image.Image], Image.Image] | None = None,
) -> None:
    """Show OCR sample cells annotated with extracted text and confidence."""
    if ocr_df.empty:
        print("No OCR samples to display.")
        return

    by_position = {(crop["row"], crop["col"]): crop for crop in cell_crops}
    records = ocr_df.to_dict(orient="records")
    by_record_position = {
        (int(record["row"]), int(record["col"])): record for record in records
    }
    if columns is None:
        max_col = max(
            [int(crop["col"]) for crop in cell_crops]
            + [int(record["col"]) for record in records]
        )
        columns = max_col + 1
    columns = max(1, int(columns))
    max_rows_per_figure = max(1, int(max_rows_per_figure))

    display_rows = sorted({int(record["row"]) for record in records})
    for start in range(0, len(display_rows), max_rows_per_figure):
        row_chunk = display_rows[start : start + max_rows_per_figure]
        fig_rows = len(row_chunk)
        fig_width = max(8.0, 1.65 * columns)
        fig_height = max(2.0, 1.45 * fig_rows)
        fig, axes = plt.subplots(
            fig_rows,
            columns,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )

        for row_index, row in enumerate(row_chunk):
            for col in range(columns):
                ax = axes[row_index][col]
                crop = by_position.get((row, col))
                if crop is not None:
                    display_image = crop["image"]
                    if image_transform is not None:
                        display_image = image_transform(display_image)
                    ax.imshow(display_image)

                record = by_record_position.get((row, col))
                ax.set_title(f"r{row} c{col}", fontsize=8)
                if record is not None:
                    annotation = _format_ocr_annotation(
                        record,
                        max_text_chars=max_text_chars,
                    )
                    ax.text(
                        0.02,
                        0.02,
                        annotation,
                        transform=ax.transAxes,
                        fontsize=annotation_fontsize,
                        color="red",
                        va="bottom",
                        ha="left",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "#333333",
                            "alpha": 0.9,
                            "pad": 2,
                        },
                        clip_on=True,
                    )
                ax.axis("off")

        plt.tight_layout(pad=0.35)
        plt.show()


def _format_ocr_annotation(record: dict[str, Any], *, max_text_chars: int) -> str:
    text = str(record.get("text") or "").replace("\n", " ")
    text = " ".join(text.split())
    if len(text) > max_text_chars:
        text = text[: max_text_chars - 1] + "..."
    confidence = _format_confidence(record.get("scores"))
    annotation = f"{text or '[no text]'} ({confidence})"
    return textwrap.fill(annotation, width=18)


def _format_confidence(scores: Any) -> str:
    if scores is None:
        return "-"
    if isinstance(scores, str):
        return scores
    try:
        values = [float(score) for score in scores]
    except TypeError:
        return "-"
    if not values:
        return "-"
    return f"{sum(values) / len(values):.2f}"


def _resolve_tesseract_cmd(tesseract_cmd: str | None) -> str:
    candidates = [tesseract_cmd, "tesseract", "/opt/homebrew/bin/tesseract"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) if "/" not in candidate else candidate
        if resolved and Path(resolved).exists():
            return resolved
    raise RuntimeError(
        "Tesseract was not found. Install it with `brew install tesseract` "
        "or pass `tesseract_cmd='/path/to/tesseract'`."
    )
