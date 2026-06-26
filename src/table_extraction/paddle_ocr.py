from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

from .grid_detection import remove_light_watermark
from .ocr import show_ocr_sample_cells


BOREHOLEAI_ROOT = Path(
    "/Users/qinli/Documents/Digitalisiation/BoreholeAI/Agentic 06 - Geotech Document Intelligence"
)
BOREHOLEAI_PYTHON = BOREHOLEAI_ROOT / ".venv" / "bin" / "python"


def paddle_ocr_cell_crops(
    cell_crops: list[dict[str, Any]],
    *,
    rows: Iterable[int] | None = None,
    cols: Iterable[int] | None = None,
    max_cells: int = 16,
    model_name: str = "paddle_ocr_model_4096",
    confidence_threshold: float = 0.0,
    separator: str = " ",
    remove_watermark_gray_threshold: int | None = None,
    boreholeai_root: Path = BOREHOLEAI_ROOT,
    boreholeai_python: Path = BOREHOLEAI_PYTHON,
) -> pd.DataFrame:
    """OCR selected cell crops with BoreholeAI's PaddleOCR environment."""
    selected = _select_cell_crops(cell_crops, rows=rows, cols=cols, max_cells=max_cells)
    if not selected:
        return pd.DataFrame(columns=["row", "col", "text", "scores", "path"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        payload = []
        for crop in selected:
            source_path = crop.get("path")
            ocr_path = source_path
            if remove_watermark_gray_threshold is not None:
                ocr_path = Path(tmp_dir) / f"r{crop['row']:03d}_c{crop['col']:03d}_ocr.png"
                ocr_image = remove_light_watermark(
                    crop["image"],
                    gray_threshold=remove_watermark_gray_threshold,
                )
                ocr_image.save(ocr_path)
            elif ocr_path is None:
                ocr_path = Path(tmp_dir) / f"r{crop['row']:03d}_c{crop['col']:03d}.png"
                crop["image"].save(ocr_path)
            payload.append(
                {
                    "row": crop["row"],
                    "col": crop["col"],
                    "path": str(ocr_path),
                    "source_path": str(source_path) if source_path else str(ocr_path),
                }
            )

        input_path = Path(tmp_dir) / "paddle_ocr_input.json"
        input_path.write_text(json.dumps(payload))

        records = _run_boreholeai_paddle_ocr(
            input_path=input_path,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            separator=separator,
            boreholeai_root=Path(boreholeai_root),
            boreholeai_python=Path(boreholeai_python),
        )

    return pd.DataFrame(records, columns=["row", "col", "text", "scores", "path"])


def _select_cell_crops(
    cell_crops: list[dict[str, Any]],
    *,
    rows: Iterable[int] | None,
    cols: Iterable[int] | None,
    max_cells: int,
) -> list[dict[str, Any]]:
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
    return selected


def _run_boreholeai_paddle_ocr(
    *,
    input_path: Path,
    model_name: str,
    confidence_threshold: float,
    separator: str,
    boreholeai_root: Path,
    boreholeai_python: Path,
) -> list[dict[str, Any]]:
    if not boreholeai_python.exists():
        raise RuntimeError(f"BoreholeAI Python not found: {boreholeai_python}")

    script = r"""
import json
import sys
from pathlib import Path
from PIL import Image

from src.core.vision_utils import PADDLE_OCR_MODELS, get_ocr_text_from_image

input_path = Path(sys.argv[1])
model_name = sys.argv[2]
confidence_threshold = float(sys.argv[3])
separator = sys.argv[4]

model = PADDLE_OCR_MODELS[model_name]
payload = json.loads(input_path.read_text())
records = []

for item in payload:
    image = Image.open(item["path"]).convert("RGB")
    text, scores = get_ocr_text_from_image(
        image,
        ocr_model=model,
        separator=separator,
        no_text_value="",
        display=False,
        return_scores=True,
    )
    if confidence_threshold > 0 and scores:
        kept = [score for score in scores if score >= confidence_threshold]
        if not kept:
            text = ""
    records.append({
        "row": item["row"],
        "col": item["col"],
        "text": text,
        "scores": [float(score) for score in scores],
        "path": item.get("source_path", item["path"]),
    })

print("PADDLE_OCR_JSON_START")
print(json.dumps(records))
print("PADDLE_OCR_JSON_END")
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(boreholeai_root)
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    env["FLAGS_log_level"] = "3"
    env["FLAGS_use_mkldnn"] = "0"
    env["DNNL_MAX_CPU_ISA"] = "AVX2"
    env["ONEDNN_MAX_CPU_ISA"] = "AVX2"

    result = subprocess.run(
        [
            str(boreholeai_python),
            "-c",
            script,
            str(input_path),
            model_name,
            str(confidence_threshold),
            separator,
        ],
        cwd=str(boreholeai_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    stdout = result.stdout.strip()
    start_marker = "PADDLE_OCR_JSON_START"
    end_marker = "PADDLE_OCR_JSON_END"
    if start_marker not in stdout or end_marker not in stdout:
        raise RuntimeError(f"Could not parse PaddleOCR output: {stdout}")
    json_text = stdout.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    return json.loads(json_text)


__all__ = ["paddle_ocr_cell_crops", "show_ocr_sample_cells"]
