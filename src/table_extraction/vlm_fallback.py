from __future__ import annotations

import json
import os
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from PIL import Image

from .table_assembly import cell_ocr_to_dataframe


VLM_MODEL_CONFIGS = {
    "qwen_vl_ocr_20251120": {
        "provider": "qwen",
        "model": "qwen-vl-ocr-2025-11-20",
    },
    "qwen_vl_ocr": {
        "provider": "qwen",
        "model": "qwen-vl-ocr",
    },
    "qwen3_vl_plus": {
        "provider": "qwen",
        "model": "qwen3-vl-plus",
    },
    "qwen_3_vl": {
        "provider": "qwen",
        "model": "qwen3-vl-plus",
    },
    "qwen3_vl_32b": {
        "provider": "openrouter",
        "model": "qwen/qwen3-vl-32b-instruct",
    },
    "qwen3_vl_8b": {
        "provider": "openrouter",
        "model": "qwen/qwen3-vl-8b-instruct",
    },
    "openai_4o_mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
    },
    "openai_4o": {
        "provider": "openai",
        "model": "gpt-4o",
    },
}

DPI = 300
VLM_DPI = 130
VLM_IMAGE_DETAIL = "auto"
MIN_VLM_IMAGE_PX = 12
DEFAULT_VLM_MODEL_KEY = "qwen3_vl_32b"
DEFAULT_VLM_EXECUTION_MODE = "parallel"
DEFAULT_VLM_MAX_WORKERS = 4
_LLM_CACHE: dict[str, Any] | None = None


def add_mean_confidence(ocr_df: pd.DataFrame) -> pd.DataFrame:
    """Add a mean_confidence column from PaddleOCR score lists."""
    output = ocr_df.copy()
    output["mean_confidence"] = output["scores"].apply(_mean_score)
    return output


def select_low_confidence_ocr(
    ocr_df: pd.DataFrame,
    *,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Return OCR records whose mean confidence is below threshold."""
    scored = add_mean_confidence(ocr_df)
    return scored.loc[scored["mean_confidence"].fillna(0) < threshold].reset_index(drop=True)


def correct_low_confidence_cells_with_vlm(
    *,
    cell_crops: list[dict[str, Any]],
    ocr_df: pd.DataFrame,
    grid_data: dict[str, Any],
    threshold: float = 0.95,
    model_key: str = DEFAULT_VLM_MODEL_KEY,
    max_cells: int | None = None,
    execution_mode: str = DEFAULT_VLM_EXECUTION_MODE,
    max_workers: int = DEFAULT_VLM_MAX_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reread low-confidence cell crops with a standalone VLM call.

    Returns:
        corrected_ocr_df: original OCR plus corrected_text/final_text columns
        vlm_review_df: only cells sent to VLM
        corrected_table_df: one-page table using final_text
    """
    scored_ocr_df = add_mean_confidence(ocr_df)
    low_conf_df = select_low_confidence_ocr(scored_ocr_df, threshold=threshold)
    if max_cells is not None:
        low_conf_df = low_conf_df.head(max_cells).copy()

    execution_mode = _normalize_execution_mode(execution_mode)
    crop_by_position = {(crop["row"], crop["col"]): crop for crop in cell_crops}
    jobs = []
    for order, record in enumerate(low_conf_df.to_dict(orient="records")):
        crop = crop_by_position.get((int(record["row"]), int(record["col"])))
        if crop is None:
            continue
        jobs.append((order, record, crop))

    if jobs:
        _get_vlm_model(model_key)

    if execution_mode == "parallel" and len(jobs) > 1:
        worker_count = _resolve_worker_count(max_workers, len(jobs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_read_low_confidence_cell_with_vlm, order, record, crop, model_key, execution_mode)
                for order, record, crop in jobs
            ]
            vlm_records = [future.result() for future in as_completed(futures)]
        vlm_records.sort(key=lambda item: item["_order"])
    else:
        vlm_records = [
            _read_low_confidence_cell_with_vlm(order, record, crop, model_key, "sequence")
            for order, record, crop in jobs
        ]

    for item in vlm_records:
        item.pop("_order", None)

    vlm_review_df = pd.DataFrame(
        vlm_records,
        columns=[
            "row",
            "col",
            "paddle_text",
            "paddle_confidence",
            "vlm_text",
            "model_key",
            "execution_mode",
            "path",
        ],
    )
    corrected_ocr_df = _merge_vlm_corrections(scored_ocr_df, vlm_review_df)
    corrected_table_df = cell_ocr_to_dataframe(corrected_ocr_df, grid_data, text_column="final_text")
    return corrected_ocr_df, vlm_review_df, corrected_table_df


def print_vlm_fallback_summary(
    ocr_df: pd.DataFrame,
    vlm_review_df: pd.DataFrame,
    *,
    threshold: float = 0.95,
) -> None:
    """Print summary of low-confidence fallback work."""
    scored = add_mean_confidence(ocr_df)
    low_count = int((scored["mean_confidence"].fillna(0) < threshold).sum())
    print(f"Low-confidence threshold: {threshold}")
    print(f"Low-confidence cells: {low_count}")
    print(f"Cells sent to VLM: {len(vlm_review_df)}")


def _merge_vlm_corrections(ocr_df: pd.DataFrame, vlm_review_df: pd.DataFrame) -> pd.DataFrame:
    corrected = ocr_df.copy()
    corrected["corrected_text"] = ""
    corrected["final_text"] = corrected["text"].fillna("").astype(str)
    corrected["fallback_method"] = "paddleocr"

    if vlm_review_df.empty:
        return corrected

    correction_map = {
        (int(row["row"]), int(row["col"])): str(row.get("vlm_text") or "").strip()
        for _, row in vlm_review_df.iterrows()
    }
    for index, row in corrected.iterrows():
        key = (int(row["row"]), int(row["col"]))
        vlm_text = correction_map.get(key)
        if vlm_text is None:
            continue
        corrected.at[index, "corrected_text"] = vlm_text
        corrected.at[index, "final_text"] = vlm_text
        corrected.at[index, "fallback_method"] = "vlm"
    return corrected


def vlm_ocr_cell_image(
    image: Image.Image,
    *,
    paddle_text: str = "",
    model_key: str = DEFAULT_VLM_MODEL_KEY,
) -> str:
    """Read one cell image with the copied BoreholeAI VLM message pattern."""
    model = _get_vlm_model(model_key)
    media_type, image_base64 = encode_image_b64_direct_method_3(image)
    prompt = (
        "Read the text in this single table cell image. "
        "Return only the exact visible text. "
        "Do not add explanations. If the cell is blank, return an empty string. "
        f"PaddleOCR read: {paddle_text!r}"
    )
    message = create_message_with_images(prompt, [(media_type, image_base64)])
    response = model.invoke([message])
    return _extract_langchain_response_text(response).strip()


def _read_low_confidence_cell_with_vlm(
    order: int,
    record: dict[str, Any],
    crop: dict[str, Any],
    model_key: str,
    execution_mode: str,
) -> dict[str, Any]:
    paddle_text = str(record.get("text") or "")
    vlm_text = vlm_ocr_cell_image(
        crop["image"],
        paddle_text=paddle_text,
        model_key=model_key,
    )
    return {
        "_order": order,
        "row": int(record["row"]),
        "col": int(record["col"]),
        "paddle_text": paddle_text,
        "paddle_confidence": record.get("mean_confidence"),
        "vlm_text": vlm_text,
        "model_key": model_key,
        "execution_mode": execution_mode,
        "path": str(crop.get("path") or ""),
    }


def _normalize_execution_mode(execution_mode: str) -> str:
    normalized = str(execution_mode).strip().lower()
    if normalized in {"sequence", "sequential"}:
        return "sequence"
    if normalized == "parallel":
        return "parallel"
    raise ValueError("execution_mode must be either 'sequence' or 'parallel'.")


def _resolve_worker_count(max_workers: int | None, job_count: int) -> int:
    if max_workers is None:
        return min(DEFAULT_VLM_MAX_WORKERS, job_count)
    return max(1, min(int(max_workers), job_count))


def encode_image_b64_direct_method_3(
    image: Image.Image,
    target_dpi: int = VLM_DPI,
) -> tuple[str, str]:
    """
    Copied/adapted from BoreholeAI `src.core.vision_utils`.

    Encodes a PIL image for VLM input with optional DPI downsampling and
    minimum-size padding for providers that reject tiny image crops.
    """
    image = image.convert("RGB")
    if target_dpi != DPI:
        scale_factor = target_dpi / DPI
        new_width = max(1, int(image.width * scale_factor))
        new_height = max(1, int(image.height * scale_factor))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if image.width < MIN_VLM_IMAGE_PX or image.height < MIN_VLM_IMAGE_PX:
        padded_width = max(image.width, MIN_VLM_IMAGE_PX)
        padded_height = max(image.height, MIN_VLM_IMAGE_PX)
        canvas = Image.new("RGB", (padded_width, padded_height), (255, 255, 255))
        paste_x = (padded_width - image.width) // 2
        paste_y = (padded_height - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))
        image = canvas

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "image/png", image_base64


def create_message_with_images(
    prompt: str | list[Any],
    image_data: list[tuple[str, str]],
    image_detail: str = VLM_IMAGE_DETAIL,
):
    """
    Copied/adapted from BoreholeAI `src.core.vision_utils`.

    Creates a LangChain HumanMessage with image_url blocks.
    """
    try:
        from langchain_core.messages import HumanMessage
    except ImportError as exc:
        raise RuntimeError(
            "VLM dependencies are not installed. Run `uv sync` after adding "
            "langchain-core/langchain-openai/langchain-qwq dependencies."
        ) from exc

    media_blocks = []
    for media_type, b64_data in image_data:
        image_url_dict = {
            "url": f"data:{media_type};base64,{b64_data}",
            "detail": image_detail,
        }
        media_blocks.append({"type": "image_url", "image_url": image_url_dict})

    if isinstance(prompt, str):
        content = [{"type": "text", "text": prompt}] + media_blocks
        return HumanMessage(content=content)

    last_msg = prompt[-1]
    if isinstance(last_msg, HumanMessage):
        if isinstance(last_msg.content, str):
            new_content = [{"type": "text", "text": last_msg.content}] + media_blocks
        elif isinstance(last_msg.content, list):
            new_content = last_msg.content + media_blocks
        else:
            new_content = [{"type": "text", "text": str(last_msg.content)}] + media_blocks
        return prompt[:-1] + [HumanMessage(content=new_content)]

    return prompt + [HumanMessage(content=media_blocks)]


def _resolve_model_config(model_key: str) -> dict[str, str]:
    if model_key not in VLM_MODEL_CONFIGS:
        known = ", ".join(sorted(VLM_MODEL_CONFIGS))
        raise ValueError(f"Unknown VLM model_key {model_key!r}. Known keys: {known}")
    return VLM_MODEL_CONFIGS[model_key]


def _get_vlm_model(model_key: str):
    global _LLM_CACHE
    if _LLM_CACHE is None:
        _LLM_CACHE = llm_setup()
    if model_key not in _LLM_CACHE:
        model_config = _resolve_model_config(model_key)
        if model_config["provider"] == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError(
                f"VLM model_key {model_key!r} uses OpenRouter model "
                f"{model_config['model']!r}, but OPENROUTER_API_KEY is not set."
            )
        known = ", ".join(sorted(_LLM_CACHE))
        raise ValueError(f"Unknown VLM model_key {model_key!r}. Known keys: {known}")
    return _LLM_CACHE[model_key]


def llm_setup(callbacks=None) -> dict[str, Any]:
    """
    Copied/adapted from BoreholeAI `src.core.llm.llm_setup`.

    This repo keeps only the VLM model keys needed by table-cell OCR fallback.
    """
    _load_env_files()
    try:
        from langchain_openai import ChatOpenAI
        from langchain_qwq import ChatQwen
    except ImportError as exc:
        raise RuntimeError(
            "VLM dependencies are not installed. Add/sync dependencies: "
            "`langchain-openai`, `langchain-qwq`, and `python-dotenv`."
        ) from exc

    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    llm: dict[str, Any] = {}

    llm["qwen_vl_ocr"] = ChatQwen(
        model="qwen-vl-ocr",
        temperature=0,
        callbacks=callbacks,
    )
    llm["qwen_vl_ocr_20251120"] = ChatQwen(
        model="qwen-vl-ocr-2025-11-20",
        temperature=0,
        callbacks=callbacks,
    )
    llm["qwen3_vl_plus"] = ChatQwen(
        model="qwen3-vl-plus",
        temperature=0,
        callbacks=callbacks,
    )
    llm["qwen_3_vl"] = llm["qwen3_vl_plus"]

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        llm["qwen3_vl_32b"] = ChatOpenAI(
            model="qwen/qwen3-vl-32b-instruct",
            temperature=0,
            max_retries=5,
            base_url=openrouter_base_url,
            api_key=openrouter_key,
            callbacks=callbacks,
        )
        llm["qwen3_vl_8b"] = ChatOpenAI(
            model="qwen/qwen3-vl-8b-instruct",
            temperature=0,
            max_retries=5,
            base_url=openrouter_base_url,
            api_key=openrouter_key,
            callbacks=callbacks,
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        llm["openai_4o_mini"] = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_key,
            callbacks=callbacks,
        )
        llm["openai_4o"] = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_key,
            callbacks=callbacks,
        )

    return llm


def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()
    secrets_path = "/Users/qinli/secrets/.env"
    if os.path.exists(secrets_path):
        load_dotenv(secrets_path)


def _extract_langchain_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _mean_score(scores: Any) -> float | None:
    if scores is None:
        return None
    if isinstance(scores, str):
        try:
            scores = json.loads(scores)
        except json.JSONDecodeError:
            return None
    try:
        values = [float(score) for score in scores]
    except TypeError:
        return None
    if not values:
        return 0.0
    return sum(values) / len(values)
