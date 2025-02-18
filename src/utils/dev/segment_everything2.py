import argparse
import os

import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def detect_document_elements(image, output_dir, confidence_threshold=0.7):
    # Load DiT model and processor
    processor = AutoImageProcessor.from_pretrained(
        "microsoft/DiT-document-layout-detection"
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/DiT-document-layout-detection"
    )

    # Process image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Post-process predictions
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    # Define layout element labels
    layout_labels = ["text", "title", "list", "table", "figure"]

    # Extract and save elements
    for i, (box, score, label) in enumerate(
        zip(results["boxes"], results["scores"], results["labels"])
    ):
        if label >= len(layout_labels):
            continue

        # Get coordinates
        xmin, ymin, xmax, ymax = [int(coord) for coord in box.tolist()]

        # Get element type
        element_type = layout_labels[label]

        # Crop element
        element_region = image.crop((xmin, ymin, xmax, ymax))

        # Generate output filename
        element_path = os.path.join(output_dir, f"{element_type}_{i+1}.png")

        # Save element
        element_region.save(element_path)
        print(
            f"{element_type.capitalize()} {i+1} saved to {element_path} (confidence: {score:.2f})"
        )


def process_pdf(pdf_path, output_dir):
    """Process PDF and extract document elements from each page"""
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF to images
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path)

    # Process each page
    for i, image in enumerate(images):
        page_dir = os.path.join(output_dir, f"page_{i+1}")
        os.makedirs(page_dir, exist_ok=True)
        print(f"Processing page {i+1}...")
        detect_document_elements(image, page_dir)


def process_image(image_path, output_dir):
    """Process single image and extract document elements"""
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path)
    detect_document_elements(image, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Extract document elements from PDF or image"
    )
    parser.add_argument("input_path", help="Path to input PDF or image")
    parser.add_argument(
        "--output_dir",
        "-o",
        default="extracted_elements",
        help="Directory for output images (default: extracted_elements)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Confidence threshold for element detection (0-1)",
    )

    args = parser.parse_args()

    try:
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
