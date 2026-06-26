from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

import pandas as pd

from .cell_extraction import crop_cells_from_grid
from .paddle_ocr import paddle_ocr_cell_crops
from .workflow import detect_grid_from_table_crop, detect_table_boundary_from_page


DEFAULT_VLM_MODEL_KEY = "qwen3_vl_32b"
DEFAULT_VLM_EXECUTION_MODE = "parallel"
DEFAULT_VLM_MAX_WORKERS = 4


@dataclass
class DocumentTableExtractionResult:
    """Container for document-level table extraction outputs."""

    page_results: list[dict[str, Any]]
    raw_cell_ocr_df: pd.DataFrame
    raw_table_df: pd.DataFrame
    cleaned_table_df: pd.DataFrame
    failed_pages_df: pd.DataFrame
    export_paths: dict[str, Path]


def ocr_one_page_to_dataframe(
    cell_crops: list[dict[str, Any]],
    grid_data: dict[str, Any],
    *,
    model_name: str = "paddle_ocr_model_4096",
    confidence_threshold: float = 0.0,
    remove_watermark_gray_threshold: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OCR every detected cell on one page and return raw OCR plus table-shaped data."""
    ocr_df = paddle_ocr_cell_crops(
        cell_crops,
        rows=range(grid_data["rows"]),
        cols=range(grid_data["cols"]),
        max_cells=grid_data["rows"] * grid_data["cols"],
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        remove_watermark_gray_threshold=remove_watermark_gray_threshold,
    )
    table_df = cell_ocr_to_dataframe(ocr_df, grid_data)
    return ocr_df, table_df


def cell_ocr_to_dataframe(
    ocr_df: pd.DataFrame,
    grid_data: dict[str, Any],
    *,
    text_column: str = "text",
    column_prefix: str = "col",
) -> pd.DataFrame:
    """Map cell-level OCR records into a rectangular row/column DataFrame."""
    rows = int(grid_data["rows"])
    cols = int(grid_data["cols"])
    table = pd.DataFrame(
        "",
        index=range(rows),
        columns=[f"{column_prefix}_{col:02d}" for col in range(cols)],
    )

    if ocr_df.empty:
        return table

    for record in ocr_df.to_dict(orient="records"):
        row = int(record["row"])
        col = int(record["col"])
        if row >= rows or col >= cols:
            continue
        table.iat[row, col] = str(record.get(text_column) or "").strip()

    table.index.name = "row"
    return table


def print_one_page_dataframe_summary(
    table_df: pd.DataFrame,
    ocr_df: pd.DataFrame,
) -> None:
    """Print a compact summary for the one-page table DataFrame."""
    non_empty_cells = int((table_df.astype(str).map(lambda value: bool(value.strip()))).sum().sum())
    print(f"Table shape: {table_df.shape[0]} rows x {table_df.shape[1]} columns")
    print(f"OCR records: {len(ocr_df)}")
    print(f"Non-empty cells: {non_empty_cells}")


def process_pages_to_dataframes(
    page_paths: list[Path],
    *,
    cell_crop_dir: Path,
    horizontal_detector: str = "morphology",
    preserve_blue_rules: bool = True,
    watermark_gray_threshold: int = 200,
    model_name: str = "paddle_ocr_model_8192",
    confidence_threshold: float = 0.0,
    ocr_watermark_gray_threshold: int | None = None,
    vlm_confidence_threshold: float | None = None,
    vlm_model_key: str = DEFAULT_VLM_MODEL_KEY,
    vlm_max_cells_per_page: int | None = None,
    vlm_execution_mode: str = DEFAULT_VLM_EXECUTION_MODE,
    vlm_max_workers: int = DEFAULT_VLM_MAX_WORKERS,
    limit_pages: int | None = None,
    continue_on_error: bool = True,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Run the validated page workflow over multiple pages."""
    selected_paths = list(page_paths[:limit_pages] if limit_pages else page_paths)
    page_results = []
    ocr_frames = []
    table_frames = []

    for page_index, page_path in enumerate(selected_paths):
        page_path = Path(page_path)
        # print(f"[{page_index + 1}/{len(selected_paths)}] Processing {page_path.name}")

        try:
            boundary_result = detect_table_boundary_from_page(
                page_path,
                remove_watermark=True,
                watermark_gray_threshold=watermark_gray_threshold,
                horizontal_detector=horizontal_detector,
                preserve_blue_rules=preserve_blue_rules,
            )
            grid_result = detect_grid_from_table_crop(
                boundary_result,
                horizontal_detector=horizontal_detector,
            )
            page_cell_crop_dir = Path(cell_crop_dir) / page_path.stem
            cell_crops = crop_cells_from_grid(
                grid_result.table_image,
                grid_result.grid_data,
                output_dir=page_cell_crop_dir,
                padding=3,
            )
            page_ocr_df, page_table_df = ocr_one_page_to_dataframe(
                cell_crops,
                grid_result.grid_data,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                remove_watermark_gray_threshold=ocr_watermark_gray_threshold,
            )
            vlm_review_df = pd.DataFrame()
            if vlm_confidence_threshold is not None:
                from .vlm_fallback import correct_low_confidence_cells_with_vlm

                page_ocr_df, vlm_review_df, page_table_df = correct_low_confidence_cells_with_vlm(
                    cell_crops=cell_crops,
                    ocr_df=page_ocr_df,
                    grid_data=grid_result.grid_data,
                    threshold=vlm_confidence_threshold,
                    model_key=vlm_model_key,
                    max_cells=vlm_max_cells_per_page,
                    execution_mode=vlm_execution_mode,
                    max_workers=vlm_max_workers,
                )
        except Exception as exc:
            if not continue_on_error:
                raise
            page_results.append(
                {
                    "page_index": page_index,
                    "source_page": page_path.name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "page_path": page_path,
                }
            )
            continue

        page_ocr_df = page_ocr_df.copy()
        page_ocr_df.insert(0, "source_page", page_path.name)
        page_ocr_df.insert(0, "page_index", page_index)
        if not vlm_review_df.empty:
            vlm_review_df = vlm_review_df.copy()
            vlm_review_df.insert(0, "source_page", page_path.name)
            vlm_review_df.insert(0, "page_index", page_index)

        page_table_out = page_table_df.reset_index().rename(columns={"row": "table_row"})
        page_table_out.insert(0, "source_page", page_path.name)
        page_table_out.insert(0, "page_index", page_index)

        page_results.append(
            {
                "page_index": page_index,
                "source_page": page_path.name,
                "status": "ok",
                "boundary_result": boundary_result,
                "grid_result": grid_result,
                "cell_crops": cell_crops,
                "ocr_df": page_ocr_df,
                "vlm_review_df": vlm_review_df,
                "table_df": page_table_out,
            }
        )
        ocr_frames.append(page_ocr_df)
        table_frames.append(page_table_out)

    all_ocr_df = pd.concat(ocr_frames, ignore_index=True) if ocr_frames else pd.DataFrame()
    all_table_df = pd.concat(table_frames, ignore_index=True) if table_frames else pd.DataFrame()
    return page_results, all_ocr_df, all_table_df


def process_document_tables(
    page_paths: list[Path],
    *,
    cell_crop_dir: Path,
    output_dir: Path | None = None,
    horizontal_detector: str = "morphology",
    preserve_blue_rules: bool = True,
    boundary_watermark_gray_threshold: int = 200,
    ocr_watermark_gray_threshold: int | None = 140,
    model_name: str = "paddle_ocr_model_8192",
    confidence_threshold: float = 0.0,
    infer_repeated_header: bool = True,
    drop_metadata: bool = True,
    export_basename: str = "extracted_table",
    vlm_confidence_threshold: float | None = None,
    vlm_model_key: str = DEFAULT_VLM_MODEL_KEY,
    vlm_max_cells_per_page: int | None = None,
    vlm_execution_mode: str = DEFAULT_VLM_EXECUTION_MODE,
    vlm_max_workers: int = DEFAULT_VLM_MAX_WORKERS,
    limit_pages: int | None = None,
    continue_on_error: bool = True,
) -> DocumentTableExtractionResult:
    """Run the validated extraction method across a whole document."""
    page_results, raw_cell_ocr_df, raw_table_df = process_pages_to_dataframes(
        page_paths,
        cell_crop_dir=cell_crop_dir,
        horizontal_detector=horizontal_detector,
        preserve_blue_rules=preserve_blue_rules,
        watermark_gray_threshold=boundary_watermark_gray_threshold,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        ocr_watermark_gray_threshold=ocr_watermark_gray_threshold,
        vlm_confidence_threshold=vlm_confidence_threshold,
        vlm_model_key=vlm_model_key,
        vlm_max_cells_per_page=vlm_max_cells_per_page,
        vlm_execution_mode=vlm_execution_mode,
        vlm_max_workers=vlm_max_workers,
        limit_pages=limit_pages,
        continue_on_error=continue_on_error,
    )
    failed_pages_df = _failed_pages_to_dataframe(page_results)
    cleaned_table_df = clean_extracted_table(
        raw_table_df,
        infer_repeated_header=infer_repeated_header,
        drop_metadata=drop_metadata,
    )
    export_paths = {}
    if output_dir is not None:
        export_paths = export_extracted_tables(
            cleaned_table_df=cleaned_table_df,
            raw_table_df=raw_table_df,
            raw_ocr_df=raw_cell_ocr_df,
            failed_pages_df=failed_pages_df,
            output_dir=output_dir,
            basename=export_basename,
        )
    return DocumentTableExtractionResult(
        page_results=page_results,
        raw_cell_ocr_df=raw_cell_ocr_df,
        raw_table_df=raw_table_df,
        cleaned_table_df=cleaned_table_df,
        failed_pages_df=failed_pages_df,
        export_paths=export_paths,
    )


def clean_extracted_table(
    table_df: pd.DataFrame,
    *,
    infer_repeated_header: bool = True,
    drop_metadata: bool = True,
) -> pd.DataFrame:
    """Clean OCR table output and optionally infer repeated page headers."""
    if table_df.empty:
        return table_df.copy()

    cleaned = table_df.copy()
    text_columns = [column for column in cleaned.columns if column not in {"page_index", "source_page", "table_row"}]

    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    has_text = cleaned[text_columns].apply(lambda row: any(bool(value) for value in row), axis=1)
    cleaned = cleaned.loc[has_text].reset_index(drop=True)

    header_row_signature = None
    if infer_repeated_header:
        header_row_signature = _infer_repeated_header_signature(cleaned, text_columns)
        if header_row_signature is not None:
            cleaned = _drop_rows_matching_signature(cleaned, text_columns, header_row_signature)
            cleaned = cleaned.reset_index(drop=True)
            header_names = _make_unique_column_names(header_row_signature)
            rename_map = dict(zip(text_columns, header_names))
            cleaned = cleaned.rename(columns=rename_map)
            text_columns = [rename_map[column] for column in text_columns]

    non_empty_columns = [
        column
        for column in text_columns
        if cleaned[column].astype(str).str.strip().ne("").any()
    ]
    keep_columns = [] if drop_metadata else [
        column for column in ["page_index", "source_page", "table_row"] if column in cleaned.columns
    ]
    cleaned = cleaned[keep_columns + non_empty_columns]
    return cleaned


def _infer_repeated_header_signature(
    table_df: pd.DataFrame,
    text_columns: list[str],
) -> tuple[str, ...] | None:
    if "page_index" not in table_df.columns or "table_row" not in table_df.columns:
        return None

    first_rows = (
        table_df.sort_values(["page_index", "table_row"])
        .groupby("page_index", as_index=False)
        .first()
    )
    if len(first_rows) < 2:
        return None

    signatures = [
        tuple(str(row[column]).strip() for column in text_columns)
        for _, row in first_rows.iterrows()
    ]
    signature_counts = pd.Series(signatures).value_counts()
    best_signature = signature_counts.index[0]
    if signature_counts.iloc[0] < 2:
        return None
    if not any(value for value in best_signature):
        return None
    return tuple(best_signature)


def _drop_rows_matching_signature(
    table_df: pd.DataFrame,
    text_columns: list[str],
    signature: tuple[str, ...],
) -> pd.DataFrame:
    row_signatures = table_df[text_columns].apply(
        lambda row: tuple(str(value).strip() for value in row),
        axis=1,
    )
    return table_df.loc[row_signatures != signature].copy()


def _make_unique_column_names(header_values: tuple[str, ...]) -> list[str]:
    names = []
    counts: dict[str, int] = {}
    for index, header_text in enumerate(header_values):
        name = " ".join(str(header_text).split()).strip() or f"col_{index:02d}"
        count = counts.get(name, 0)
        counts[name] = count + 1
        if count:
            name = f"{name}_{count + 1}"
        names.append(name)
    return names


def export_extracted_tables(
    *,
    cleaned_table_df: pd.DataFrame,
    raw_table_df: pd.DataFrame,
    raw_ocr_df: pd.DataFrame,
    failed_pages_df: pd.DataFrame | None = None,
    output_dir: Path,
    basename: str = "extracted_table",
) -> dict[str, Path]:
    """Export cleaned and raw extraction outputs to CSV/XLSX."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{basename}.csv"
    xlsx_path = output_dir / f"{basename}.xlsx"

    cleaned_table_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        cleaned_table_df.to_excel(writer, sheet_name="cleaned_table", index=False)
        raw_table_df.to_excel(writer, sheet_name="raw_table", index=False)
        raw_ocr_df.to_excel(writer, sheet_name="raw_cell_ocr", index=False)
        if failed_pages_df is not None and not failed_pages_df.empty:
            failed_pages_df.to_excel(writer, sheet_name="failed_pages", index=False)

    return {"csv": csv_path, "xlsx": xlsx_path}


def print_multi_page_summary(raw_table_df: pd.DataFrame, raw_ocr_df: pd.DataFrame) -> None:
    """Print compact summary for multi-page extraction."""
    page_count = raw_table_df["page_index"].nunique() if "page_index" in raw_table_df else 0
    print(f"Processed pages: {page_count}")
    print(f"Raw table rows: {len(raw_table_df)}")
    print(f"Raw OCR records: {len(raw_ocr_df)}")


def _failed_pages_to_dataframe(page_results: list[dict[str, Any]]) -> pd.DataFrame:
    records = [
        {
            "page_index": result.get("page_index"),
            "source_page": result.get("source_page"),
            "error_type": result.get("error_type"),
            "error": result.get("error"),
            "page_path": str(result.get("page_path") or ""),
        }
        for result in page_results
        if result.get("status") == "failed"
    ]
    return pd.DataFrame(records, columns=["page_index", "source_page", "error_type", "error", "page_path"])
