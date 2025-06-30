import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse
import re
import shutil
from typing import Dict, List

from docling.document_converter import DocumentConverter

from utils.sitemap import get_sitemap_urls


class MarkdownCleaner:
    """Comprehensive markdown cleanup processor for LLM-friendly content"""
    
    def __init__(self, backup_suffix: str = ".backup"):
        self.backup_suffix = backup_suffix
        self.stats = {
            'files_processed': 0,
            'files_backed_up': 0,
            'html_entities_fixed': 0,
            'navigation_sections_removed': 0,
            'paragraph_symbols_fixed': 0,
            'whitespace_normalized': 0,
            'duplicate_headings_removed': 0,
            'empty_sections_removed': 0
        }
        
        # HTML entity mappings
        self.html_entities = {
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&apos;': "'",
            '&nbsp;': ' ',
            '&#39;': "'",
            '&#8217;': "'",
            '&#8220;': '"',
            '&#8221;': '"',
            '&#8211;': '–',
            '&#8212;': '—',
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '…',
        }
        
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the original file in backup subdirectory"""
        # Create backup directory if it doesn't exist
        backup_dir = file_path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        # Create backup path in the backup directory
        backup_path = backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        self.stats['files_backed_up'] += 1
        return backup_path
        
    def fix_html_entities(self, content: str) -> str:
        """Replace HTML entities with proper characters"""
        original_content = content
        for entity, replacement in self.html_entities.items():
            content = content.replace(entity, replacement)
        
        if content != original_content:
            self.stats['html_entities_fixed'] += 1
            
        return content
        
    def remove_navigation_sections(self, content: str) -> str:
        """Remove redundant navigation sections"""
        original_content = content
        lines = content.split('\n')
        cleaned_lines = []
        skip_mode = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if we're starting a navigation section
            if (line.strip() == '<!-- image -->' or 
                (line.strip() == '- Home' and i < 200)):
                skip_mode = True
                
            # Look for the end of navigation section
            if skip_mode:
                # End navigation when we hit a proper heading or significant content
                if (line.startswith('#') or 
                    (line.strip() and not line.startswith('-') and not line.startswith(' ') and 
                     len(line.strip()) > 50 and not 'Table of contents' in line)):
                    skip_mode = False
                    cleaned_lines.append(line)
                # Skip this line if still in navigation
            else:
                cleaned_lines.append(line)
            
            i += 1
        
        result = '\n'.join(cleaned_lines)
        if result != original_content:
            self.stats['navigation_sections_removed'] += 1
            
        return result
        
    def fix_paragraph_symbols(self, content: str) -> str:
        """Fix standalone paragraph symbols and improve heading structure"""
        original_content = content
        
        # Remove standalone ¶ symbols
        content = re.sub(r'^¶\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s*¶\s*\n', '\n', content)
        
        if content != original_content:
            self.stats['paragraph_symbols_fixed'] += 1
            
        return content
        
    def normalize_whitespace(self, content: str) -> str:
        """Normalize excessive whitespace and blank lines"""
        original_content = content
        
        # Replace multiple consecutive blank lines with at most 2
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace from lines
        lines = content.split('\n')
        lines = [line.rstrip() for line in lines]
        content = '\n'.join(lines)
        
        # Remove excessive whitespace at the beginning and end
        content = content.strip() + '\n'
        
        if content != original_content:
            self.stats['whitespace_normalized'] += 1
            
        return content
        
    def remove_duplicate_headings(self, content: str) -> str:
        """Remove duplicate consecutive headings"""
        original_content = content
        lines = content.split('\n')
        cleaned_lines = []
        previous_heading = None
        
        for line in lines:
            if line.startswith('#'):
                # Extract heading text without the # symbols
                heading_text = line.lstrip('#').strip()
                if heading_text != previous_heading:
                    cleaned_lines.append(line)
                    previous_heading = heading_text
                # Skip duplicate heading
            else:
                cleaned_lines.append(line)
                if line.strip():  # Reset heading tracking on non-empty content
                    previous_heading = None
        
        result = '\n'.join(cleaned_lines)
        if result != original_content:
            self.stats['duplicate_headings_removed'] += 1
            
        return result
        
    def remove_empty_sections(self, content: str) -> str:
        """Remove sections that are essentially empty"""
        original_content = content
        
        # Remove sections with only whitespace between headings
        content = re.sub(r'(#{1,6}\s+[^\n]+\n)\s*\n\s*(#{1,6}\s+[^\n]+)', r'\1\2', content)
        
        if content != original_content:
            self.stats['empty_sections_removed'] += 1
            
        return content
        
    def clean_content(self, content: str) -> str:
        """Apply all cleaning operations to content"""
        content = self.fix_html_entities(content)
        content = self.remove_navigation_sections(content)
        content = self.fix_paragraph_symbols(content)
        content = self.normalize_whitespace(content)
        content = self.remove_duplicate_headings(content)
        content = self.remove_empty_sections(content)
        return content
        
    def clean_directory(self, directory: Path, create_backups: bool = True) -> Dict:
        """Clean all markdown files in a directory"""
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")
        
        markdown_files = list(directory.glob("*.md"))
        
        for file_path in markdown_files:
            try:
                # Create backup if requested
                if create_backups:
                    self.backup_file(file_path)
                
                # Read and clean content
                content = file_path.read_text(encoding='utf-8')
                cleaned_content = self.clean_content(content)
                
                # Write cleaned content back
                file_path.write_text(cleaned_content, encoding='utf-8')
                self.stats['files_processed'] += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return self.stats


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

    def convert_sitemap(self, sitemap_url, output_format="markdown", output_dir=None, clean_output=False):
        try:
            sitemap_urls = get_sitemap_urls(sitemap_url)
            conv_results_iter = self.converter.convert_all(sitemap_urls)

            docs = []
            for i, result in enumerate(conv_results_iter):
                if result.document:
                    document = result.document
                    if output_format == "markdown":
                        content = document.export_to_markdown()
                        
                        # Clean content if requested
                        if clean_output:
                            cleaner = MarkdownCleaner()
                            content = cleaner.clean_content(content)
                        
                        if output_dir:
                            # Create filename from URL
                            url = sitemap_urls[i]
                            parsed = urlparse(url)
                            filename = re.sub(r'[^\w\-_.]', '_', parsed.path.strip('/') or 'index')
                            if not filename.endswith('.md'):
                                filename += '.md'
                            
                            output_path = Path(output_dir) / filename
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            output_path.write_text(content, encoding='utf-8')
                            print(f"Saved: {output_path}")
                        docs.append(content)
                    elif output_format == "json":
                        docs.append(document.export_to_dict())
            return docs
        except Exception as e:
            raise Exception(f"Error processing sitemap: {e}")
    
    def clean_markdown_directory(self, directory, create_backups=True):
        """Clean all markdown files in a directory"""
        try:
            cleaner = MarkdownCleaner()
            stats = cleaner.clean_directory(Path(directory), create_backups)
            
            print(f"Markdown Cleanup Complete!")
            print(f"Files processed: {stats['files_processed']}")
            print(f"Files backed up: {stats['files_backed_up']}")
            print(f"HTML entities fixed: {stats['html_entities_fixed']}")
            print(f"Navigation sections removed: {stats['navigation_sections_removed']}")
            print(f"Paragraph symbols fixed: {stats['paragraph_symbols_fixed']}")
            print(f"Whitespace normalized: {stats['whitespace_normalized']}")
            print(f"Duplicate headings removed: {stats['duplicate_headings_removed']}")
            print(f"Empty sections removed: {stats['empty_sections_removed']}")
            
            return stats
        except Exception as e:
            raise Exception(f"Error cleaning markdown files: {e}")

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
        sitemap_parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory to save individual markdown files"
        )
        sitemap_parser.add_argument(
            "--clean", "-c",
            action="store_true",
            help="Clean output markdown for better LLM parsing"
        )

        # Clean command for processing existing markdown files
        clean_parser = subparsers.add_parser(
            "clean", help="Clean existing markdown files for better LLM parsing"
        )
        clean_parser.add_argument(
            "directory",
            type=str,
            help="Directory containing markdown files to clean"
        )
        clean_parser.add_argument(
            "--no-backup",
            action="store_true",
            help="Don't create backup files (default: create backups)"
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
                output = self.convert_sitemap(
                    args.url, 
                    args.format, 
                    getattr(args, 'output_dir', None),
                    getattr(args, 'clean', False)
                )
            elif args.command == "clean":
                create_backups = not getattr(args, 'no_backup', False)
                stats = self.clean_markdown_directory(args.directory, create_backups)
                return  # Exit early for clean command

            if args.format == "json":
                print(json.dumps(output, indent=2))
            else:
                if args.command == "sitemap" and getattr(args, 'output_dir', None):
                    print(f"Converted {len(output)} pages to {args.output_dir}")
                else:
                    if isinstance(output, list):
                        print("\n\n---\n\n".join(output))
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
