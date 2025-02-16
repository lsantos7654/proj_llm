import argparse
from pathlib import Path
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
import sys
import json


class DocumentConverterCLI:
    def __init__(self):
        self.converter = DocumentConverter()

    def convert_pdf(self, file_path, output_format="markdown"):
        path = Path(file_path).absolute()

        try:
            result = self.converter.convert(path)
            document = result.document

            if output_format == "markdown":
                return document.export_to_markdown()
            elif output_format == "json":
                return document.export_to_dict()
        except Exception as e:
            raise Exception(f"Error converting PDF: {e}")

    def convert_html(self, url, output_format="markdown"):
        try:
            result = self.converter.convert(url)
            document = result.document

            if output_format == "markdown":
                return document.export_to_markdown()
            elif output_format == "json":
                return document.export_to_dict()
        except Exception as e:
            raise Exception(f"Error converting HTML: {e}")

    def convert_sitemap(self, sitemap_url, output_format="markdown"):
        try:
            sitemap_urls = get_sitemap_urls(sitemap_url)
            conv_results_iter = self.converter.convert_all(sitemap_urls)

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
            raise Exception(f"Error processing sitemap: {e}")

    def run_cli(self):
        parser = argparse.ArgumentParser(
            description="Convert documents to markdown or JSON"
        )

        subparsers = parser.add_subparsers(dest="command", help="Conversion type")

        pdf_parser = subparsers.add_parser("pdf", help="Convert PDF file")
        pdf_parser.add_argument("file", type=str, help="Path to PDF file")
        pdf_parser.add_argument(
            "--format",
            choices=["markdown", "json"],
            default="markdown",
            help="Output format (default: markdown)",
        )

        html_parser = subparsers.add_parser("html", help="Convert HTML page")
        html_parser.add_argument("url", type=str, help="URL of the webpage")
        html_parser.add_argument(
            "--format",
            choices=["markdown", "json"],
            default="markdown",
            help="Output format (default: markdown)",
        )

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

        try:
            if args.command == "pdf":
                output = self.convert_pdf(args.file, args.format)
            elif args.command == "html":
                output = self.convert_html(args.url, args.format)
            elif args.command == "sitemap":
                output = self.convert_sitemap(args.url, args.format)

            if args.format == "json":
                print(json.dumps(output, indent=2))
            else:
                print(output)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    converter = DocumentConverterCLI()
    converter.run_cli()


if __name__ == "__main__":
    main()
