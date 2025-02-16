import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import argparse
import os
from pdf2image import convert_from_path


def detect_document_elements(image, output_dir, confidence_threshold=0.7):
    # Load layout detection model (DiT)
    layout_processor = AutoImageProcessor.from_pretrained("microsoft/dit-base")
    layout_model = AutoModelForObjectDetection.from_pretrained("microsoft/dit-base")

    # Load table detection model
    table_processor = DetrImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection"
    )
    table_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    # Process image for layout detection
    layout_inputs = layout_processor(images=image, return_tensors="pt")
    layout_outputs = layout_model(**layout_inputs)

    # Process image for table detection
    table_inputs = table_processor(images=image, return_tensors="pt")
    table_outputs = table_model(**table_inputs)

    # Post-process predictions
    target_sizes = torch.tensor([image.size[::-1]])

    layout_results = layout_processor.post_process_object_detection(
        layout_outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    table_results = table_processor.post_process_object_detection(
        table_outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    # Define labels for layout elements
    layout_labels = ["text", "title", "list", "figure", "table"]

    # Extract and save layout elements
    for i, (box, score, label) in enumerate(
        zip(layout_results["boxes"], layout_results["scores"], layout_results["labels"])
    ):
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

    # Extract and save tables (using specialized table detector)
    for i, (box, score) in enumerate(
        zip(table_results["boxes"], table_results["scores"])
    ):
        xmin, ymin, xmax, ymax = [int(coord) for coord in box.tolist()]
        table_region = image.crop((xmin, ymin, xmax, ymax))
        table_path = os.path.join(output_dir, f"table_detailed_{i+1}.png")
        table_region.save(table_path)
        print(f"Detailed table {i+1} saved to {table_path} (confidence: {score:.2f})")


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
