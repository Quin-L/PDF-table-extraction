from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from PIL import Image

from .grid_detection import (
    create_grid,
    crop_pil_to_bbox,
    detect_horizontal_vertical_lines,
    drop_lines_near_image_boundary,
    infer_table_bbox,
    process_PIL_image_for_cv2,
    remove_light_watermark,
)


@dataclass
class TableBoundaryResult:
    page_path: Path
    page_image: Image.Image
    cleaned_page_image: Image.Image
    full_binary_image: np.ndarray
    full_h_lines: list[dict]
    full_v_lines: list[dict]
    full_failed_h: list[dict]
    full_failed_v: list[dict]
    full_h_debug: dict
    full_v_debug: dict
    full_line_overlay: np.ndarray
    table_bbox: dict[str, int]
    table_image: Image.Image
    table_binary_image: np.ndarray


@dataclass
class TableGridResult(TableBoundaryResult):
    h_lines: list[dict]
    v_lines: list[dict]
    inner_h_lines: list[dict]
    inner_v_lines: list[dict]
    failed_h: list[dict]
    failed_v: list[dict]
    h_debug: dict
    v_debug: dict
    boundary_dropped_h: list[dict]
    boundary_dropped_v: list[dict]
    grid_data: dict
    grid_overlay: np.ndarray


def detect_table_boundary_from_page(
    page_path: Path,
    *,
    bbox_padding: int = 4,
    remove_watermark: bool = True,
    watermark_gray_threshold: int = 200,
    horizontal_detector: str = "combined",
    preserve_blue_rules: bool = False,
) -> TableBoundaryResult:
    """Detect full-page lines, infer the table bbox, and crop the table."""
    page_path = Path(page_path)
    page_image = Image.open(page_path).convert("RGB")
    cleaned_page_image = (
        remove_light_watermark(page_image, gray_threshold=watermark_gray_threshold)
        if remove_watermark
        else page_image
    )
    full_binary_image = process_PIL_image_for_cv2(
        cleaned_page_image,
        preserve_blue_rules=preserve_blue_rules,
    )

    (
        full_h_lines,
        full_v_lines,
        _,
        _,
        full_failed_h,
        full_failed_v,
        full_h_debug,
        full_v_debug,
    ) = detect_horizontal_vertical_lines(
        full_binary_image,
        horizontal_detector=horizontal_detector,
    )
    _, full_line_overlay = create_grid(
        full_h_lines,
        full_v_lines,
        full_binary_image,
        line_width=2,
        failed_horizontal=full_failed_h,
        failed_vertical=full_failed_v,
    )

    table_bbox = infer_table_bbox(full_h_lines, full_v_lines, page_image.size, padding=bbox_padding)
    table_image = crop_pil_to_bbox(cleaned_page_image, table_bbox)
    table_binary_image = process_PIL_image_for_cv2(
        table_image,
        preserve_blue_rules=preserve_blue_rules,
    )

    return TableBoundaryResult(
        page_path=page_path,
        page_image=page_image,
        cleaned_page_image=cleaned_page_image,
        full_binary_image=full_binary_image,
        full_h_lines=full_h_lines,
        full_v_lines=full_v_lines,
        full_failed_h=full_failed_h,
        full_failed_v=full_failed_v,
        full_h_debug=full_h_debug or {},
        full_v_debug=full_v_debug or {},
        full_line_overlay=full_line_overlay,
        table_bbox=table_bbox,
        table_image=table_image,
        table_binary_image=table_binary_image,
    )


def detect_grid_from_table_crop(
    boundary_result: TableBoundaryResult,
    *,
    boundary_tolerance: int = 8,
    horizontal_detector: str = "combined",
) -> TableGridResult:
    """Detect the internal grid inside an already-cropped table."""
    table_binary_image = boundary_result.table_binary_image
    h_lines, v_lines, _, _, failed_h, failed_v, h_debug, v_debug = detect_horizontal_vertical_lines(
        table_binary_image,
        horizontal_detector=horizontal_detector,
    )
    inner_h_lines, inner_v_lines = drop_lines_near_image_boundary(
        h_lines,
        v_lines,
        table_binary_image.shape,
        tolerance=boundary_tolerance,
    )
    # Capture the accepted lines that the boundary filter discards, so the
    # vertical-line funnel can show separators dropped only at this stage.
    crop_h, crop_w = table_binary_image.shape[:2]
    boundary_dropped_h = [
        {**line, "reason": "image_boundary"}
        for line in h_lines
        if not (boundary_tolerance < line["y_center"] < crop_h - boundary_tolerance)
    ]
    boundary_dropped_v = [
        {**line, "reason": "image_boundary"}
        for line in v_lines
        if not (boundary_tolerance < line["x_center"] < crop_w - boundary_tolerance)
    ]
    grid_data, grid_overlay = create_grid(
        inner_h_lines,
        inner_v_lines,
        table_binary_image,
        line_width=2,
        failed_horizontal=failed_h,
        failed_vertical=failed_v,
    )

    return TableGridResult(
        page_path=boundary_result.page_path,
        page_image=boundary_result.page_image,
        cleaned_page_image=boundary_result.cleaned_page_image,
        full_binary_image=boundary_result.full_binary_image,
        full_h_lines=boundary_result.full_h_lines,
        full_v_lines=boundary_result.full_v_lines,
        full_failed_h=boundary_result.full_failed_h,
        full_failed_v=boundary_result.full_failed_v,
        full_h_debug=boundary_result.full_h_debug,
        full_v_debug=boundary_result.full_v_debug,
        full_line_overlay=boundary_result.full_line_overlay,
        table_bbox=boundary_result.table_bbox,
        table_image=boundary_result.table_image,
        table_binary_image=boundary_result.table_binary_image,
        h_lines=h_lines,
        v_lines=v_lines,
        inner_h_lines=inner_h_lines,
        inner_v_lines=inner_v_lines,
        failed_h=failed_h,
        failed_v=failed_v,
        h_debug=h_debug or {},
        v_debug=v_debug or {},
        boundary_dropped_h=boundary_dropped_h,
        boundary_dropped_v=boundary_dropped_v,
        grid_data=grid_data,
        grid_overlay=grid_overlay,
    )


def detect_table_grid_from_page(
    page_path: Path,
    *,
    bbox_padding: int = 4,
    boundary_tolerance: int = 8,
    remove_watermark: bool = True,
    watermark_gray_threshold: int = 200,
    horizontal_detector: str = "combined",
    preserve_blue_rules: bool = False,
) -> TableGridResult:
    """Convenience wrapper for boundary detection followed by grid detection."""
    boundary_result = detect_table_boundary_from_page(
        page_path,
        bbox_padding=bbox_padding,
        remove_watermark=remove_watermark,
        watermark_gray_threshold=watermark_gray_threshold,
        horizontal_detector=horizontal_detector,
        preserve_blue_rules=preserve_blue_rules,
    )
    return detect_grid_from_table_crop(
        boundary_result,
        boundary_tolerance=boundary_tolerance,
        horizontal_detector=horizontal_detector,
    )


def print_table_boundary_summary(result: TableBoundaryResult) -> None:
    """Print full-page line-detection and table-boundary counts."""
    print(f"Full-page H lines: {len(result.full_h_lines)}")
    print(f"Full-page V lines: {len(result.full_v_lines)}")
    print(f"Rejected H candidates: {len(result.full_failed_h)}")
    print(f"Rejected V candidates: {len(result.full_failed_v)}")
    if result.full_h_debug.get("source_counts"):
        print(f"Full-page H sources: {result.full_h_debug['source_counts']}")
    _print_vertical_funnel_oneline(result.full_v_debug, scope="full_page")
    print(f"Table bbox: {result.table_bbox}")
    print(f"Cropped table size: {result.table_image.size}")


def print_table_grid_summary(result: TableGridResult) -> None:
    """Print cropped-table line-detection and grid counts."""
    print(f"Cropped-table H lines: {len(result.h_lines)} -> internal: {len(result.inner_h_lines)}")
    print(f"Cropped-table V lines: {len(result.v_lines)} -> internal: {len(result.inner_v_lines)}")
    if result.h_debug.get("source_counts"):
        print(f"Cropped-table H sources: {result.h_debug['source_counts']}")
    _print_vertical_funnel_oneline(result.v_debug, scope="cropped_table")
    if result.boundary_dropped_v:
        dropped_x = [int(line["x_center"]) for line in result.boundary_dropped_v]
        print(f"V lines dropped by image-boundary filter (x_center): {dropped_x}")
    print(f"Grid rows: {result.grid_data['rows']}")
    print(f"Grid cols: {result.grid_data['cols']}")


def _print_vertical_funnel_oneline(v_debug: dict, *, scope: str) -> None:
    """Compact one-line vertical funnel for the summary printers."""
    if not v_debug:
        return
    thresholds = v_debug.get("thresholds", {})
    print(
        f"{scope} V funnel: "
        f"contours={v_debug.get('contour_count', 0)} "
        f"-> shape_ok={len(v_debug.get('candidate_lines_pre_merge', []))} "
        f"-> merged={len(v_debug.get('candidate_lines_post_merge', []))} "
        f"-> len_pass={len(v_debug.get('candidate_lines_post_merge', [])) - len(v_debug.get('length_failed', []))} "
        f"-> final={len(v_debug.get('final_lines', []))} "
        f"(min_length_px={thresholds.get('min_length_px')}, "
        f"min_sep_px={thresholds.get('min_separation_px')})"
    )


def print_vertical_line_funnel(
    result: TableBoundaryResult | TableGridResult,
    scope: str = "cropped_table",
) -> None:
    """Print the vertical-line detection funnel stage-by-stage with x positions.

    Use this to see exactly where a separator (e.g. the borehole-ID / item rule)
    drops out of the pipeline. Stages, in order:
        contour -> thickness/aspect -> merge -> length -> proximity
        -> [cropped_table only] image-boundary -> final

    For ``scope="cropped_table"`` the x positions are also printed in full-page
    coordinates (``x_center + table_bbox.xmin``) so they can be compared directly
    against the ``scope="full_page"`` funnel.
    """
    if scope == "full_page":
        v_debug = result.full_v_debug
        boundary_dropped: list[dict] = []
        x_offset = 0
    elif scope == "cropped_table":
        if not isinstance(result, TableGridResult):
            raise ValueError("scope='cropped_table' requires a TableGridResult (from Step 6).")
        v_debug = result.v_debug
        boundary_dropped = result.boundary_dropped_v
        x_offset = int(result.table_bbox.get("xmin", 0))
    else:
        raise ValueError("scope must be 'full_page' or 'cropped_table'")

    if not v_debug:
        print(f"[{scope}] no vertical debug captured.")
        return

    thresholds = v_debug.get("thresholds", {})
    print(f"=== Vertical-line funnel: {scope} ===")
    print(f"thresholds: {thresholds}")
    if x_offset:
        print(f"(full-page x = crop x + table_bbox.xmin={x_offset})")

    def _xs(lines: list[dict]) -> list[int]:
        return sorted(int(line["x_center"]) for line in lines)

    def _xs_fullpage(lines: list[dict]) -> list[int]:
        return sorted(int(line["x_center"]) + x_offset for line in lines)

    def _detail(lines: list[dict], *keys: str) -> list[tuple]:
        return [tuple(line.get(key) for key in ("x_center", *keys)) for line in lines]

    pre_merge = v_debug.get("candidate_lines_pre_merge", [])
    post_merge = v_debug.get("candidate_lines_post_merge", [])
    length_failed = v_debug.get("length_failed", [])
    proximity_removed = v_debug.get("proximity_removed", [])
    contour_failed = v_debug.get("contour_failed", [])
    final_lines = v_debug.get("final_lines", [])
    min_length_px = thresholds.get("min_length_px")
    min_sep_px = thresholds.get("min_separation_px")

    print(f"1. contours found:             {v_debug.get('contour_count', 0)}")
    print(f"2. passed thickness/aspect:   {len(pre_merge):>3}  x={_xs(pre_merge)}")
    if contour_failed:
        print(f"   rejected at contour:       {len(contour_failed):>3}  "
              f"(x_center, reason, width, height)={_detail(contour_failed, 'reason', 'width', 'height')}")
    print(f"3. after merge:               {len(post_merge):>3}  x={_xs(post_merge)}")
    print(f"4. passed min_length:         {len(post_merge) - len(length_failed):>3}  "
          f"(min_length_px={min_length_px})")
    if length_failed:
        print(f"   dropped by length:         {len(length_failed):>3}  "
              f"(x_center, height)={_detail(length_failed, 'height')}")
    if proximity_removed:
        print(f"   dropped by proximity:      {len(proximity_removed):>3}  x={_xs(proximity_removed)}  "
              f"(min_sep_px={min_sep_px})")
    print(f"5. extract_vertical_lines out: {len(final_lines):>3}  x={_xs(final_lines)}")
    if x_offset:
        print(f"   (same, full-page coords):      x={_xs_fullpage(final_lines)}")
    if scope == "cropped_table":
        if boundary_dropped:
            print(f"6. dropped by image-boundary:  {len(boundary_dropped):>3}  x={_xs(boundary_dropped)}  "
                  f"(near crop edge)")
        kept = result.inner_v_lines
        print(f"7. internal (final grid):      {len(kept):>3}  x={_xs(kept)}")
        if x_offset:
            print(f"   (same, full-page coords):      x={_xs_fullpage(kept)}")


def full_page_line_audit(result: TableBoundaryResult) -> pd.DataFrame:
    """Return accepted/rejected full-page line candidates with source and reason."""
    return _build_line_audit_table(
        accepted_horizontal=result.full_h_lines,
        accepted_vertical=result.full_v_lines,
        rejected_horizontal=result.full_failed_h,
        rejected_vertical=result.full_failed_v,
        scope="full_page",
    )


def cropped_grid_line_audit(result: TableGridResult) -> pd.DataFrame:
    """Return accepted/rejected cropped-table grid candidates with source and reason."""
    return _build_line_audit_table(
        accepted_horizontal=result.inner_h_lines,
        accepted_vertical=result.inner_v_lines,
        rejected_horizontal=result.failed_h,
        rejected_vertical=result.failed_v,
        scope="cropped_table",
    )


def show_full_page_line_detection(result: TableBoundaryResult) -> None:
    """Display accepted and rejected full-page line candidates."""
    show_line_detection_overlay(
        image=result.cleaned_page_image,
        horizontal_lines=result.full_h_lines,
        vertical_lines=result.full_v_lines,
        failed_horizontal=result.full_failed_h,
        failed_vertical=result.full_failed_v,
        title=f"Full-page detected lines: {result.page_path.name}",
    )


def show_horizontal_detector_comparison(
    result: TableBoundaryResult | TableGridResult,
    scope: str = "full_page",
    *,
    label_sources: bool = False,
) -> None:
    """Display morphology, Hough, and combined horizontal detections side by side."""
    if scope == "full_page":
        image = result.full_binary_image
        debug = result.full_h_debug
        title_prefix = "Full page"
    elif scope == "cropped_table":
        image = result.table_binary_image
        debug = result.h_debug
        title_prefix = "Cropped table"
    else:
        raise ValueError("scope must be 'full_page' or 'cropped_table'")

    morph_lines = (debug.get("morphology") or {}).get("final_lines", [])
    hough_lines = (debug.get("hough") or {}).get("final_lines", [])
    combined_lines = debug.get("final_lines", [])
    if not combined_lines:
        combined_lines = result.full_h_lines if scope == "full_page" else result.h_lines

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
    panels = [
        ("Morphology", morph_lines, "lime"),
        ("Hough", hough_lines, "cyan"),
        ("Combined", combined_lines, "yellow"),
    ]

    for ax, (label, lines, color) in zip(axes, panels):
        ax.imshow(image, cmap="gray")
        _draw_horizontal_lines(ax, lines, color=color, label_sources=label_sources)
        ax.set_title(f"{title_prefix}: {label} ({len(lines)} lines)")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_table_boundary(result: TableBoundaryResult) -> None:
    """Display the inferred table bbox and cropped binary table image."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(result.cleaned_page_image)
    axes[0].add_patch(
        plt.Rectangle(
            (result.table_bbox["xmin"], result.table_bbox["ymin"]),
            result.table_bbox["xmax"] - result.table_bbox["xmin"],
            result.table_bbox["ymax"] - result.table_bbox["ymin"],
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
    axes[0].set_title("Inferred outer table boundary")
    axes[0].axis("off")

    axes[1].imshow(result.table_binary_image, cmap="gray")
    axes[1].set_title("Cropped table binary image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_cropped_table_grid(result: TableGridResult) -> None:
    """Display the internal grid detected inside the cropped table."""
    show_line_detection_overlay(
        image=result.table_image,
        horizontal_lines=result.inner_h_lines,
        vertical_lines=result.inner_v_lines,
        failed_horizontal=result.failed_h,
        failed_vertical=result.failed_v,
        title=(
            f"Internal grid inside cropped table: "
            f"{result.grid_data['rows']} rows x {result.grid_data['cols']} cols"
        ),
    )


def show_cropped_table_grid_review(result: TableGridResult) -> None:
    """Display the cropped table and final grid overlay for notebook review."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(result.table_image)
    axes[0].set_title("Cropped table")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(result.grid_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Final internal grid: {result.grid_data['rows']} rows x {result.grid_data['cols']} cols"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_vertical_line_crop_vs_fullpage(result: TableGridResult) -> None:
    """Overlay Step-4 (full-page) vs Step-6 (cropped) vertical lines on the crop.

    Three layers are drawn on the cropped table image so a missing separator is
    obvious at a glance:

    * green dotted  - full-page vertical lines from Step 4, reprojected into crop
      coordinates (``x_center - table_bbox.xmin``). These are the columns the
      boundary step expected.
    * blue solid    - vertical lines kept in the Step-6 internal grid.
    * red dashed    - Step-6 vertical lines that were rejected, annotated with the
      stage/reason they dropped out (length, proximity, image_boundary, ...).

    A green line with no blue line under it is a separator that Step 6 lost; the
    nearest red line (or its absence) tells you why.
    """
    x_offset = int(result.table_bbox.get("xmin", 0))
    crop_h = result.table_image.size[1]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(result.table_image)

    # Layer 1: full-page expectations, reprojected into crop space.
    reprojected = []
    for line in result.full_v_lines:
        x = int(line["x_center"]) - x_offset
        reprojected.append(x)
        ax.plot([x, x], [0, crop_h], color="#2ecc71", linewidth=2.4, linestyle=":")

    # Layer 2: lines that survived into the final grid.
    kept_x = set()
    for line in result.inner_v_lines:
        x = int(line["x_center"])
        kept_x.add(x)
        ax.plot([x, x], [0, crop_h], color="#00a3ff", linewidth=2.2)

    # Layer 3: rejected lines, labelled with the reason they were dropped.
    rejected = list(result.failed_v) + list(result.boundary_dropped_v)
    for line in rejected:
        x = int(line["x_center"])
        y0 = int(line.get("y", 0))
        y1 = y0 + int(line.get("height", crop_h))
        ax.plot([x, x], [y0, y1], color="#ff4d4d", linewidth=1.6, linestyle="--")
        ax.text(x + 3, max(12, y0), line.get("reason", "?"), color="#ff4d4d", fontsize=8, va="top")

    # Flag full-page columns that have no kept counterpart nearby.
    missing = [x for x in reprojected if all(abs(x - kx) > 8 for kx in kept_x)]
    for x in missing:
        ax.text(x + 3, crop_h - 8, "MISSING in grid", color="#117a3d", fontsize=9, va="bottom", weight="bold")

    handles = [
        mlines.Line2D([], [], color="#2ecc71", linewidth=2, linestyle=":", label=f"Full-page expected ({len(reprojected)})"),
        mlines.Line2D([], [], color="#00a3ff", linewidth=2, label=f"Kept in grid ({len(result.inner_v_lines)})"),
        mlines.Line2D([], [], color="#ff4d4d", linewidth=2, linestyle="--", label=f"Rejected in crop ({len(rejected)})"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_title(
        f"Vertical lines: full-page expectation vs cropped-table result "
        f"({len(missing)} expected column(s) missing from grid)"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_line_detection_overlay(
    *,
    image: Image.Image | np.ndarray,
    horizontal_lines: list[dict],
    vertical_lines: list[dict],
    failed_horizontal: list[dict] | None = None,
    failed_vertical: list[dict] | None = None,
    title: str = "Detected lines",
) -> None:
    """Shared line-overlay plot with consistent colors and legend."""
    failed_horizontal = failed_horizontal or []
    failed_vertical = failed_vertical or []

    fig, ax = plt.subplots(figsize=(16, 10))
    if isinstance(image, Image.Image):
        ax.imshow(image)
    elif image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    _draw_horizontal_lines(ax, failed_horizontal, color="#ff9f1c", linewidth=1.2, linestyle="--")
    _draw_vertical_lines(ax, failed_vertical, color="#ff4d4d", linewidth=1.2, linestyle="--")
    _draw_horizontal_lines(ax, horizontal_lines, color="#d000ff", linewidth=2.4)
    _draw_vertical_lines(ax, vertical_lines, color="#00a3ff", linewidth=2.0)

    handles = [
        mlines.Line2D([], [], color="#d000ff", linewidth=2, label=f"Accepted horizontal ({len(horizontal_lines)})"),
        mlines.Line2D([], [], color="#00a3ff", linewidth=2, label=f"Accepted vertical ({len(vertical_lines)})"),
        mlines.Line2D([], [], color="#ff9f1c", linewidth=2, linestyle="--", label=f"Rejected horizontal ({len(failed_horizontal)})"),
        mlines.Line2D([], [], color="#ff4d4d", linewidth=2, linestyle="--", label=f"Rejected vertical ({len(failed_vertical)})"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def _build_line_audit_table(
    *,
    accepted_horizontal: list[dict],
    accepted_vertical: list[dict],
    rejected_horizontal: list[dict],
    rejected_vertical: list[dict],
    scope: str,
) -> pd.DataFrame:
    rows = []
    for axis, status, lines in [
        ("horizontal", "accepted", accepted_horizontal),
        ("vertical", "accepted", accepted_vertical),
        ("horizontal", "rejected", rejected_horizontal),
        ("vertical", "rejected", rejected_vertical),
    ]:
        for index, line in enumerate(lines):
            rows.append(_line_audit_record(line, scope=scope, axis=axis, status=status, index=index))

    columns = [
        "scope",
        "axis",
        "status",
        "index",
        "source",
        "reason",
        "x",
        "y",
        "x_center",
        "y_center",
        "width",
        "height",
        "coverage",
        "angle_deg",
    ]
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return table
    return table.sort_values(["status", "axis", "y_center", "x_center"], na_position="last").reset_index(drop=True)


def _line_audit_record(line: dict, *, scope: str, axis: str, status: str, index: int) -> dict:
    return {
        "scope": scope,
        "axis": axis,
        "status": status,
        "index": index,
        "source": _line_source(line, axis=axis),
        "reason": line.get("reason", "") if status == "rejected" else "",
        "x": line.get("x"),
        "y": line.get("y"),
        "x_center": line.get("x_center"),
        "y_center": line.get("y_center"),
        "width": line.get("width"),
        "height": line.get("height"),
        "coverage": line.get("coverage", line.get("hough_coverage")),
        "angle_deg": line.get("angle_deg"),
    }


def _line_source(line: dict, *, axis: str) -> str:
    source = line.get("source")
    if source:
        return source
    if axis == "vertical":
        return "morphology"
    return "unknown"


def _draw_horizontal_lines(
    ax,
    lines: list[dict],
    color: str,
    label_sources: bool = False,
    linewidth: float = 2,
    linestyle: str = "-",
) -> None:
    for line in lines:
        y = _horizontal_line_plot_y(line)
        x0 = line.get("x", 0)
        x1 = x0 + line.get("width", 0)
        ax.plot([x0, x1], [y, y], color=color, linewidth=linewidth, linestyle=linestyle)
        source = line.get("source")
        if label_sources and source:
            ax.text(x1 + 4, y, source, color=color, fontsize=7, va="center")


def _horizontal_line_plot_y(line: dict) -> int:
    y = int(line.get("y", line["y_center"]))
    y_center = int(line["y_center"])
    height = max(1, int(line.get("height", 1)))

    # Hough can merge dense header pixels into a candidate whose top extent is
    # the real visible rule while y_center has drifted into the colored band.
    if abs(y_center - y) > max(3, height * 2):
        return y
    return y_center


def _draw_vertical_lines(
    ax,
    lines: list[dict],
    color: str,
    linewidth: float = 2,
    linestyle: str = "-",
) -> None:
    for line in lines:
        x = line["x_center"]
        y0 = line.get("y", 0)
        y1 = y0 + line.get("height", 0)
        ax.plot([x, x], [y0, y1], color=color, linewidth=linewidth, linestyle=linestyle)
