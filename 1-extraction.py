#!/usr/bin/env python3
import argparse
from pathlib import Path
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
import sys
import json


def convert_pdf(file_path, output_format):
    converter = DocumentConverter()
    path = Path(file_path).absolute()

    try:
        result = converter.convert(path)
        document = result.document

        if output_format == "markdown":
            return document.export_to_markdown()
        elif output_format == "json":
            return document.export_to_dict()
    except Exception as e:
        print(f"Error converting PDF: {e}", file=sys.stderr)
        sys.exit(1)


def convert_html(url, output_format):
    converter = DocumentConverter()
    try:
        result = converter.convert(url)
        document = result.document

        if output_format == "markdown":
            return document.export_to_markdown()
        elif output_format == "json":
            return document.export_to_dict()
    except Exception as e:
        print(f"Error converting HTML: {e}", file=sys.stderr)
        sys.exit(1)


def convert_sitemap(sitemap_url, output_format):
    converter = DocumentConverter()
    try:
        sitemap_urls = get_sitemap_urls(sitemap_url)
        conv_results_iter = converter.convert_all(sitemap_urls)

        docs = []
        for result in conv_results_iter:
            if result.document:
                document = result.document
                if output_format == "markdown":
                    docs.append(document.export_to_markdown())
                elif output_format == "json":
                    docs.append(document.export_to_dict())
        return docs
    except Exception as e:
        print(f"Error processing sitemap: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert documents to markdown or JSON"
    )

    # Create subparsers for different conversion types
    subparsers = parser.add_subparsers(dest="command", help="Conversion type")

    # PDF parser
    pdf_parser = subparsers.add_parser("pdf", help="Convert PDF file")
    pdf_parser.add_argument("file", type=str, help="Path to PDF file")
    pdf_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # HTML parser
    html_parser = subparsers.add_parser("html", help="Convert HTML page")
    html_parser.add_argument("url", type=str, help="URL of the webpage")
    html_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # Sitemap parser
    sitemap_parser = subparsers.add_parser(
        "sitemap", help="Convert all pages in sitemap"
    )
    sitemap_parser.add_argument("url", type=str, help="URL of the sitemap")
    sitemap_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "pdf":
        output = convert_pdf(args.file, args.format)
    elif args.command == "html":
        output = convert_html(args.url, args.format)
    elif args.command == "sitemap":
        output = convert_sitemap(args.url, args.format)

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        print(output)


if __name__ == "__main__":
    main()
