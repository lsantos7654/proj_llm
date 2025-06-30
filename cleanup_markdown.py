#!/usr/bin/env python3
"""
Markdown Cleanup Script for LLM-Friendly Processing

This script systematically cleans up markdown files to make them more suitable for LLM consumption
by removing redundant navigation, fixing HTML entities, normalizing formatting, and improving structure.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime


class MarkdownCleaner:
    """Comprehensive markdown cleanup processor"""
    
    def __init__(self, input_dir: str = "output", backup_suffix: str = ".backup"):
        self.input_dir = Path(input_dir)
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
        
        # Patterns for navigation cleanup
        self.nav_patterns = [
            # Remove redundant navigation lists at the beginning
            r'^(\s*<!-- image -->\s*\n)?\s*- Home\s*\n\s*- Introduction.*?(?=\n#|\n\n[^-\s]|$)',
            # Remove table of contents indicators
            r'\s+Table of contents\s*\n',
            # Remove duplicate section headers
            r'^(\s*- [A-Z][a-zA-Z\s]+ [A-Z][a-zA-Z\s]+)\s*\n',
        ]
        
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the original file"""
        backup_path = file_path.with_suffix(file_path.suffix + self.backup_suffix)
        shutil.copy2(file_path, backup_path)
        self.stats['files_backed_up'] += 1
        return backup_path
        
    def fix_html_entities(self, content: str) -> str:
        """Replace HTML entities with proper characters"""
        original_content = content
        for entity, replacement in self.html_entities.items():
            content = content.replace(entity, replacement)
        
        # Count changes
        if content != original_content:
            self.stats['html_entities_fixed'] += 1
            
        return content
        
    def remove_navigation_sections(self, content: str) -> str:
        """Remove redundant navigation sections"""
        original_content = content
        
        # Remove navigation lists at the beginning
        # Look for patterns starting with "- Home" or "<!-- image -->"
        lines = content.split('\n')
        cleaned_lines = []
        skip_mode = False
        nav_section_found = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if we're starting a navigation section
            if (line.strip() == '<!-- image -->' or 
                (line.strip() == '- Home' and i < 200)):  # Extended check range
                skip_mode = True
                nav_section_found = True
                i += 1
                continue
                
            # If we're in skip mode, look for the end of navigation
            if skip_mode:
                # End navigation when we hit a main heading or substantial content
                if (line.startswith('# ') and not line.strip().endswith('Table of contents')):
                    skip_mode = False
                    cleaned_lines.append(line)
                i += 1
                continue
                
            cleaned_lines.append(line)
            i += 1
            
        if nav_section_found:
            self.stats['navigation_sections_removed'] += 1
            
        return '\n'.join(cleaned_lines)
        
    def fix_paragraph_symbols(self, content: str) -> str:
        """Replace standalone ¶ symbols with proper formatting"""
        original_content = content
        
        # Replace standalone paragraph symbols
        content = re.sub(r'^\s*¶\s*$', '', content, flags=re.MULTILINE)
        
        # Remove ¶ symbols that appear after headings
        content = re.sub(r'(##?\s+[^\n]+)¶', r'\1', content)
        
        if content != original_content:
            self.stats['paragraph_symbols_fixed'] += 1
            
        return content
        
    def normalize_headings(self, content: str) -> str:
        """Ensure proper heading hierarchy and remove duplicates"""
        lines = content.split('\n')
        cleaned_lines = []
        seen_headings = set()
        
        for line in lines:
            # Check for duplicate headings (same text appearing multiple times)
            if line.startswith('#'):
                heading_text = line.strip()
                if heading_text in seen_headings:
                    self.stats['duplicate_headings_removed'] += 1
                    continue
                seen_headings.add(heading_text)
                
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)
        
    def normalize_whitespace(self, content: str) -> str:
        """Normalize excessive whitespace and empty lines"""
        original_content = content
        
        # Replace multiple consecutive empty lines with maximum 2
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace from lines
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Ensure file ends with single newline
        content = content.rstrip() + '\n'
        
        if content != original_content:
            self.stats['whitespace_normalized'] += 1
            
        return content
        
    def remove_empty_sections(self, content: str) -> str:
        """Remove empty sections and redundant content"""
        original_content = content
        
        # Remove sections that are just heading followed by empty content
        content = re.sub(r'^(#+\s+[^\n]*)\n\s*\n(?=#+|\Z)', r'\1\n\n', content, flags=re.MULTILINE)
        
        # Remove module-attribute sections that are just labels
        content = re.sub(r'^module-attribute\s*\n\s*¶?\s*\n', '', content, flags=re.MULTILINE)
        
        if content != original_content:
            self.stats['empty_sections_removed'] += 1
            
        return content
        
    def fix_code_blocks(self, content: str) -> str:
        """Ensure consistent code block formatting"""
        # Fix incomplete code blocks and empty code blocks
        content = re.sub(r'^```\s*$\n(.*?)(?=\n```|\n#|\Z)', r'```\n\1\n```', content, flags=re.MULTILINE | re.DOTALL)
        
        # Remove empty code blocks
        content = re.sub(r'^```\s*\n\s*```\s*$', '', content, flags=re.MULTILINE)
        
        # Fix broken code blocks (multiple consecutive ``` lines)
        content = re.sub(r'(```[^\n]*\n)```\s*\n', r'\1', content)
        
        return content
        
    def fix_list_formatting(self, content: str) -> str:
        """Clean up inconsistent list formatting"""
        # Fix inconsistent bullet points
        content = re.sub(r'^(\s*)[\u2022\u2023\u25E6\u2043\u2219]\s+', r'\1- ', content, flags=re.MULTILINE)
        
        # Fix inconsistent indentation in lists
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Standardize list indentation
            if re.match(r'^\s*-\s+', line):
                # Count leading spaces before the dash
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 0 and leading_spaces % 4 != 0:
                    # Round to nearest multiple of 4
                    new_indent = ((leading_spaces + 2) // 4) * 4
                    line = ' ' * new_indent + line.lstrip()
            
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)
        
    def clean_content(self, content: str) -> str:
        """Apply all cleanup operations to content"""
        # Apply all cleanup operations in order
        content = self.fix_html_entities(content)
        content = self.remove_navigation_sections(content)
        content = self.fix_paragraph_symbols(content)
        content = self.normalize_headings(content)
        content = self.remove_empty_sections(content)
        content = self.fix_code_blocks(content)
        content = self.fix_list_formatting(content)
        content = self.normalize_whitespace(content)
        
        return content
        
    def process_file(self, file_path: Path) -> bool:
        """Process a single markdown file"""
        try:
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            # Skip if file is empty
            if not original_content.strip():
                return False
                
            # Create backup
            backup_path = self.backup_file(file_path)
            
            # Clean content
            cleaned_content = self.clean_content(original_content)
            
            # Only write if content changed
            if cleaned_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                return True
            else:
                # Remove backup if no changes were made
                backup_path.unlink()
                self.stats['files_backed_up'] -= 1
                return False
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
            
    def process_directory(self) -> None:
        """Process all markdown files in the directory"""
        if not self.input_dir.exists():
            print(f"Error: Directory {self.input_dir} does not exist")
            return
            
        md_files = list(self.input_dir.glob("*.md"))
        
        if not md_files:
            print(f"No markdown files found in {self.input_dir}")
            return
            
        print(f"Found {len(md_files)} markdown files to process")
        print(f"Backup files will be created with suffix: {self.backup_suffix}")
        print("=" * 60)
        
        for file_path in md_files:
            print(f"Processing: {file_path.name}...", end=" ")
            if self.process_file(file_path):
                print("✓ Modified")
            else:
                print("- No changes")
            self.stats['files_processed'] += 1
            
    def print_summary(self) -> None:
        """Print summary of processing results"""
        print("\n" + "=" * 60)
        print("CLEANUP SUMMARY")
        print("=" * 60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files backed up: {self.stats['files_backed_up']}")
        print(f"HTML entities fixed: {self.stats['html_entities_fixed']} files")
        print(f"Navigation sections removed: {self.stats['navigation_sections_removed']} files")
        print(f"Paragraph symbols fixed: {self.stats['paragraph_symbols_fixed']} files")
        print(f"Whitespace normalized: {self.stats['whitespace_normalized']} files")
        print(f"Duplicate headings removed: {self.stats['duplicate_headings_removed']} files")
        print(f"Empty sections removed: {self.stats['empty_sections_removed']} files")
        
        if self.stats['files_backed_up'] > 0:
            print(f"\nBackup files created in: {self.input_dir}")
            restore_cmd = f"find {self.input_dir} -name '*{self.backup_suffix}' -exec sh -c 'mv \"$1\" \"${{1%.*}}\"' _ {{}} \\;"
            print(f"To restore originals: {restore_cmd}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up markdown files to make them more LLM-friendly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cleanup_markdown.py                    # Process ./output/ directory
    python cleanup_markdown.py -d docs           # Process ./docs/ directory
    python cleanup_markdown.py -d output -b .orig # Use .orig backup suffix
    python cleanup_markdown.py --dry-run         # Show what would be changed
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        default='output',
        help='Directory containing markdown files to process (default: output)'
    )
    
    parser.add_argument(
        '-b', '--backup-suffix',
        default='.backup',
        help='Suffix for backup files (default: .backup)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making modifications'
    )
    
    args = parser.parse_args()
    
    print("Markdown Cleanup Script")
    print("=" * 60)
    print(f"Target directory: {args.directory}")
    print(f"Backup suffix: {args.backup_suffix}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    cleaner = MarkdownCleaner(args.directory, args.backup_suffix)
    
    if args.dry_run:
        print("DRY RUN: This would process the following files:")
        md_files = list(cleaner.input_dir.glob("*.md"))
        for file_path in md_files:
            print(f"  - {file_path.name}")
        print(f"\nTotal: {len(md_files)} files")
    else:
        cleaner.process_directory()
        cleaner.print_summary()


if __name__ == "__main__":
    main()