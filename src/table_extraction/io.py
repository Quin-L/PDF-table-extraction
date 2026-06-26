from __future__ import annotations

import subprocess
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
PDF_EXTENSION = ".pdf"


def render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    """Render a PDF to PNG page images and return generated image paths."""
    pdf_path = pdf_path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / pdf_path.stem

    existing = sorted(output_dir.glob(f"{pdf_path.stem}-*.png"))
    if existing:
        return existing

    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(output_dir.glob(f"{pdf_path.stem}-*.png"))


def load_input_pages(input_path: Path, page_image_dir: Path) -> list[Path]:
    """Accept a PDF, an image, or a folder of PDFs/images and return page image paths."""
    input_path = input_path.expanduser().resolve()

    if input_path.is_dir():
        page_paths = []
        input_files = sorted(
            [
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | {PDF_EXTENSION}
            ],
            key=lambda path: path.name.lower(),
        )
        for file_path in input_files:
            if file_path.suffix.lower() == PDF_EXTENSION:
                page_paths.extend(render_pdf_to_images(file_path, page_image_dir / file_path.stem))
            else:
                page_paths.append(file_path)
        return page_paths

    if input_path.is_file() and input_path.suffix.lower() == PDF_EXTENSION:
        return render_pdf_to_images(input_path, page_image_dir / input_path.stem)

    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_path]

    raise ValueError(f"Unsupported input path: {input_path}")
