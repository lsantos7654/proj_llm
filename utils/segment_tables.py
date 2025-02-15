import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import argparse
import os
from pdf2image import convert_from_path


def detect_and_extract_tables(image, output_path, confidence_threshold=0.7):
    # Load model and processor
    processor = DetrImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection"
    )
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    # Process image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Post-process predictions
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    # Extract and save tables
    for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
        # Get coordinates
        xmin, ymin, xmax, ymax = [int(coord) for coord in box.tolist()]

        # Crop table
        table_region = image.crop((xmin, ymin, xmax, ymax))

        # Generate output filename
        base, ext = os.path.splitext(output_path)
        table_path = f"{base}_table_{i+1}{ext}"

        # Save table
        table_region.save(table_path)
        print(f"Table {i+1} saved to {table_path} (confidence: {score:.2f})")


def process_pdf(pdf_path, output_dir):
    """Process PDF and extract tables from each page"""
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF to images
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path)

    # Process each page
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"page_{i+1}_table.png")
        print(f"Processing page {i+1}...")
        detect_and_extract_tables(image, output_path)


def process_image(image_path, output_dir):
    """Process single image and extract tables"""
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path)
    output_path = os.path.join(output_dir, "extracted_table.png")
    detect_and_extract_tables(image, output_path)


def main():
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
        # Check if input is PDF or image
        if args.input_path.lower().endswith(".pdf"):
            process_pdf(args.input_path, args.output_dir)
        else:
            process_image(args.input_path, args.output_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
