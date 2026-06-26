from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def crop_cells_from_grid(
    table_image: Image.Image,
    grid_data: dict[str, Any],
    *,
    output_dir: Path | None = None,
    padding: int = 3,
    image_format: str = "PNG",
) -> list[dict[str, Any]]:
    """Crop table cells from a PIL image using grid cell coordinates."""
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    crops = []
    for cell in grid_data["cells"]:
        bbox = _padded_bbox(cell, table_image.size, padding=padding)
        cell_image = table_image.crop(bbox)
        crop = {
            "row": cell["row"],
            "col": cell["col"],
            "xmin": bbox[0],
            "ymin": bbox[1],
            "xmax": bbox[2],
            "ymax": bbox[3],
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1],
            "image": cell_image,
            "path": None,
        }

        if output_dir is not None:
            crop_path = output_dir / f"r{cell['row']:03d}_c{cell['col']:03d}.png"
            cell_image.save(crop_path, format=image_format)
            crop["path"] = crop_path

        crops.append(crop)

    return crops


def cell_crops_to_dataframe(cell_crops: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert cell crop metadata to a DataFrame without embedding PIL images."""
    records = []
    for crop in cell_crops:
        records.append(
            {
                "row": crop["row"],
                "col": crop["col"],
                "xmin": crop["xmin"],
                "ymin": crop["ymin"],
                "xmax": crop["xmax"],
                "ymax": crop["ymax"],
                "width": crop["width"],
                "height": crop["height"],
                "path": str(crop["path"]) if crop.get("path") else None,
            }
        )
    return pd.DataFrame(records)


def print_cell_crop_summary(cell_crops: list[dict[str, Any]], grid_data: dict[str, Any]) -> None:
    """Print a compact summary for the notebook."""
    print(f"Cell crops: {len(cell_crops)}")
    print(f"Grid size: {grid_data['rows']} rows x {grid_data['cols']} cols")
    saved_count = sum(1 for crop in cell_crops if crop.get("path"))
    if saved_count:
        first_path = next(crop["path"] for crop in cell_crops if crop.get("path"))
        print(f"Saved crops: {saved_count}")
        print(f"First crop: {first_path}")


def show_cell_crop_preview(
    cell_crops: list[dict[str, Any]],
    *,
    max_rows: int = 4,
    max_cols: int = 6,
) -> None:
    """Show a small top-left sample of cropped cells."""
    preview = [
        crop
        for crop in cell_crops
        if crop["row"] < max_rows and crop["col"] < max_cols
    ]
    if not preview:
        print("No cell crops to preview.")
        return

    n_rows = max(crop["row"] for crop in preview) + 1
    n_cols = max(crop["col"] for crop in preview) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 1.6 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    by_position = {(crop["row"], crop["col"]): crop for crop in preview}
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row][col]
            crop = by_position.get((row, col))
            if crop:
                ax.imshow(crop["image"])
                ax.set_title(f"r{row} c{col}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def _padded_bbox(cell: dict[str, Any], image_size: tuple[int, int], *, padding: int) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    xmin = max(0, int(cell["xmin"]) + padding)
    ymin = max(0, int(cell["ymin"]) + padding)
    xmax = min(image_width, int(cell["xmax"]) - padding)
    ymax = min(image_height, int(cell["ymax"]) - padding)

    if xmax <= xmin:
        xmin = max(0, int(cell["xmin"]))
        xmax = min(image_width, int(cell["xmax"]))
    if ymax <= ymin:
        ymin = max(0, int(cell["ymin"]))
        ymax = min(image_height, int(cell["ymax"]))

    return xmin, ymin, xmax, ymax
