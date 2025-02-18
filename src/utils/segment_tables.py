import argparse
from pathlib import Path
from typing import List

import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection


def get_table_filename(source_path: Path, page_num: int, table_num: int) -> Path:
    """
    Generate a unique filename for an extracted table.

    Args:
        source_path: Original document path
        page_num: Page number where table was found
        table_num: Index of table on the page

    Returns:
        Path: Filename for the extracted table
    """
    base_name = source_path.stem
    # return Path(f"{base_name}_page_{page_num}_table_{table_num}.png")
    return Path(f"{base_name}_table_{table_num}.png")


def detect_and_extract_tables(
    image: Image.Image,
    output_path: Path,
    source_path: Path,
    page_num: int,
    confidence_threshold: float = 0.7,
) -> List[Path]:
    """
    Detect and extract tables from an image using Table Transformer model.

    Args:
        image: Input PIL Image object containing tables
        output_path: Base path where extracted tables should be saved
        source_path: Path to original source document
        page_num: Page number being processed
        confidence_threshold: Minimum confidence score for table detection (0-1)

    Returns:
        List[Path]: Paths to all extracted table images
    """
    processor = DetrImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection"
    )
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    saved_paths = []
    for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
        xmin, ymin, xmax, ymax = [int(coord) for coord in box.tolist()]
        table_region = image.crop((xmin, ymin, xmax, ymax))

        table_filename = get_table_filename(source_path, page_num, i + 1)
        table_path = output_path.parent / table_filename
        table_region.save(table_path)
        saved_paths.append(table_path)
        print(f"Table {i+1} saved to {table_path} (confidence: {score:.2f})")

    return saved_paths


def extract_table_from_pdf(pdf_path: Path, output_dir: Path) -> List[Path]:
    """
    Process a PDF file and extract tables from each page.

    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory where extracted tables should be saved

    Returns:
        List[Path]: List of paths to all extracted table images
    """
    output_dir.mkdir(exist_ok=True)
    all_table_paths = []

    print("Converting PDF to images...")
    images = convert_from_path(str(pdf_path))

    for page_num, image in enumerate(images, 1):
        print(f"Processing page {page_num}...")
        table_paths = detect_and_extract_tables(
            image=image,
            output_path=output_dir / "temp.png",
            source_path=pdf_path,
            page_num=page_num,
        )
        all_table_paths.extend(table_paths)

    return all_table_paths


def extract_table_from_image(image_path: Path, output_dir: Path) -> List[Path]:
    """
    Process a single image and extract tables from it.

    Args:
        image_path: Path to the input image file
        output_dir: Directory where extracted tables should be saved

    Returns:
        List[Path]: List of paths to all extracted table images
    """
    output_dir.mkdir(exist_ok=True)
    image = Image.open(image_path)
    return detect_and_extract_tables(
        image=image,
        output_path=output_dir / "temp.png",
        source_path=image_path,
        page_num=1,  # Single images are treated as page 1
    )


def main() -> int:
    """
    Main function to handle command line arguments and process input files.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Extract tables from PDF or image using Table Transformer"
    )
    parser.add_argument("input_path", help="Path to input PDF or image")
    parser.add_argument(
        "--output_dir",
        "-o",
        default="extracted_tables",
        help="Directory for output images (default: extracted_tables)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Confidence threshold for table detection (0-1)",
    )

    args = parser.parse_args()

    try:
        input_path = Path(args.input_path)
        output_dir = Path(args.output_dir)

        if input_path.suffix.lower() == ".pdf":
            table_paths = process_pdf(input_path, output_dir)
        else:
            table_paths = process_image(input_path, output_dir)

        print(f"\nExtracted {len(table_paths)} tables:")
        for path in table_paths:
            print(f"- {path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
