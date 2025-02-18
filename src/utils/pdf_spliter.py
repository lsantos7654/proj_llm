import argparse
from pathlib import Path
from typing import Tuple

from PyPDF2 import PdfReader, PdfWriter


def split_pdf_vertically(input_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Split a PDF document vertically into left and right halves.

    Args:
        input_path: Path to the input PDF file
        output_dir: Directory where split PDFs will be saved

    Returns:
        Tuple[Path, Path]: Paths to the left and right PDF files

    Raises:
        FileNotFoundError: If input PDF doesn't exist
        ValueError: If PDF is empty or cannot be processed
        OSError: If there are issues writing the output files
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    try:
        # Read the input PDF
        reader = PdfReader(input_path)
        if not reader.pages:
            raise ValueError("PDF is empty")

        page = reader.pages[0]  # Get the first page

        # Get original page dimensions
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)

        # Create two new PDF writers
        left_writer = PdfWriter()
        right_writer = PdfWriter()

        # Create two copies of the page for left and right
        page_left = page
        page_right = page.clone(right_writer)

        # Crop the pages
        page_left.mediabox.upper_right = (width / 2, height)
        page_right.mediabox.upper_left = (width / 2, height)

        # Add pages to respective writers
        left_writer.add_page(page_left)
        right_writer.add_page(page_right)

        # Create output paths with original filename
        left_path = output_dir / f"{input_path.stem}_left.pdf"
        right_path = output_dir / f"{input_path.stem}_right.pdf"

        # Save the split PDFs
        try:
            left_path.parent.mkdir(parents=True, exist_ok=True)
            with open(left_path, "wb") as left_file:
                left_writer.write(left_file)
            with open(right_path, "wb") as right_file:
                right_writer.write(right_file)
        except OSError as e:
            raise OSError(f"Failed to write output files: {e}")

        return left_path, right_path

    except Exception as e:
        raise ValueError(f"Failed to process PDF {input_path}: {str(e)}")


def main() -> int:
    """
    Main function to handle command line arguments and process PDF splitting.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Split PDF vertically into two halves")
    parser.add_argument("input_pdf", help="Input PDF file", type=Path)
    parser.add_argument(
        "--output", default="output", help="Output directory for split PDFs", type=Path
    )

    args = parser.parse_args()

    try:
        # Create output directory if it doesn't exist
        args.output.mkdir(parents=True, exist_ok=True)

        left_path, right_path = split_pdf_vertically(args.input_pdf, args.output)
        print(f"Successfully created:")
        print(f"- Left half: {left_path}")
        print(f"- Right half: {right_path}")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
