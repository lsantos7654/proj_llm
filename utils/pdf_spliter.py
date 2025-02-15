import argparse
from PyPDF2 import PdfReader, PdfWriter
import os


def split_pdf_vertically(input_path, output_dir):
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Read the input PDF
    reader = PdfReader(input_path)
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
    left_path = os.path.join(output_dir, f"{base_name}_left.pdf")
    right_path = os.path.join(output_dir, f"{base_name}_right.pdf")

    # Save the split PDFs
    with open(left_path, "wb") as left_file:
        left_writer.write(left_file)
    with open(right_path, "wb") as right_file:
        right_writer.write(right_file)

    return left_path, right_path


def main():
    parser = argparse.ArgumentParser(description="Split PDF vertically into two halves")
    parser.add_argument("input_pdf", help="Input PDF file")
    parser.add_argument(
        "--output", default="output", help="Output directory for split PDFs"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    left_path, right_path = split_pdf_vertically(args.input_pdf, args.output)
    print(f"Created {left_path} and {right_path}")


if __name__ == "__main__":
    main()
