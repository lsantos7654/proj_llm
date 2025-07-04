#!/usr/bin/env python3
"""
Rustdoc JSON Extractor - Extract structured documentation from docs.rs JSON API
This provides much higher quality documentation than HTML scraping
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import re
from datetime import datetime


class RustdocJsonExtractor:
    """Extract documentation using docs.rs JSON API"""
    
    def __init__(self, output_dir: str = "output", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
    def log(self, message: str, emoji: str = "ℹ️"):
        """Simple logging with emojis"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {emoji}  {message}")
    
    def extract_from_crate_url(self, url: str) -> bool:
        """Extract documentation from a docs.rs crate URL"""
        try:
            # Parse the URL to extract crate name and version
            crate_info = self._parse_docs_rs_url(url)
            if not crate_info:
                self.log(f"Could not parse docs.rs URL: {url}", "❌")
                return False
            
            crate_name, version = crate_info
            self.log(f"Extracting {crate_name} v{version}", "🦀")
            
            # Get JSON documentation
            json_data = self._fetch_rustdoc_json(crate_name, version)
            if not json_data:
                return False
            
            # Extract documentation to markdown files
            return self._extract_to_markdown(json_data, crate_name, version)
            
        except Exception as e:
            self.log(f"Error extracting from {url}: {e}", "❌")
            return False
    
    def _parse_docs_rs_url(self, url: str) -> Optional[tuple]:
        """Parse docs.rs URL to extract crate name and version"""
        # Example: https://docs.rs/ratatui/latest/ratatui/
        # Example: https://docs.rs/clap/4.0.0/clap/
        
        patterns = [
            r'docs\.rs/([^/]+)/([^/]+)',  # crate/version
            r'docs\.rs/([^/]+)/?$',       # just crate name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                crate_name = match.group(1)
                version = match.group(2) if len(match.groups()) > 1 else "latest"
                return (crate_name, version)
        
        return None
    
    def _fetch_rustdoc_json(self, crate_name: str, version: str = "latest") -> Optional[Dict]:
        """Fetch JSON documentation from docs.rs"""
        json_url = f"https://docs.rs/crate/{crate_name}/{version}/json"
        
        try:
            self.log(f"Fetching JSON from: {json_url}", "🌐")
            response = requests.get(json_url, timeout=30)
            response.raise_for_status()
            
            # Handle different content types
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return response.json()
            elif 'gzip' in content_type or 'compressed' in content_type:
                # Handle compressed JSON
                import gzip
                return json.loads(gzip.decompress(response.content))
            else:
                # Try to parse as JSON anyway
                return response.json()
                
        except requests.exceptions.RequestException as e:
            self.log(f"Failed to fetch JSON: {e}", "❌")
            return None
        except json.JSONDecodeError as e:
            self.log(f"Failed to parse JSON: {e}", "❌")
            return None
    
    def _extract_to_markdown(self, json_data: Dict, crate_name: str, version: str) -> bool:
        """Extract JSON documentation to markdown files"""
        try:
            # Get the root crate
            root_crate = json_data.get('index', {})
            if not root_crate:
                self.log("No index found in JSON data", "⚠️")
                return False
            
            # Extract different types of items
            items_extracted = 0
            
            # Extract modules
            modules = self._extract_modules(json_data, root_crate)
            items_extracted += len(modules)
            
            # Extract functions
            functions = self._extract_functions(json_data, root_crate) 
            items_extracted += len(functions)
            
            # Extract structs
            structs = self._extract_structs(json_data, root_crate)
            items_extracted += len(structs)
            
            # Extract enums
            enums = self._extract_enums(json_data, root_crate)
            items_extracted += len(enums)
            
            # Extract traits
            traits = self._extract_traits(json_data, root_crate)
            items_extracted += len(traits)
            
            # Create overview file
            self._create_overview_file(crate_name, version, {
                'modules': modules,
                'functions': functions, 
                'structs': structs,
                'enums': enums,
                'traits': traits
            })
            
            self.log(f"Extracted {items_extracted} items to markdown", "🎉")
            return True
            
        except Exception as e:
            self.log(f"Error extracting to markdown: {e}", "❌")
            return False
    
    def _extract_modules(self, json_data: Dict, root_crate: Dict) -> List[Dict]:
        """Extract module documentation"""
        modules = []
        index = json_data.get('index', {})
        
        for item_id, item in index.items():
            if item.get('kind') == 'module':
                module_doc = self._extract_item_documentation(item, 'module')
                if module_doc:
                    modules.append(module_doc)
                    self._save_item_to_file(module_doc, 'module')
        
        return modules
    
    def _extract_functions(self, json_data: Dict, root_crate: Dict) -> List[Dict]:
        """Extract function documentation"""
        functions = []
        index = json_data.get('index', {})
        
        for item_id, item in index.items():
            if item.get('kind') == 'function':
                func_doc = self._extract_item_documentation(item, 'function')
                if func_doc:
                    functions.append(func_doc)
                    self._save_item_to_file(func_doc, 'function')
        
        return functions
    
    def _extract_structs(self, json_data: Dict, root_crate: Dict) -> List[Dict]:
        """Extract struct documentation"""
        structs = []
        index = json_data.get('index', {})
        
        for item_id, item in index.items():
            if item.get('kind') == 'struct':
                struct_doc = self._extract_item_documentation(item, 'struct')
                if struct_doc:
                    structs.append(struct_doc)
                    self._save_item_to_file(struct_doc, 'struct')
        
        return structs
    
    def _extract_enums(self, json_data: Dict, root_crate: Dict) -> List[Dict]:
        """Extract enum documentation"""
        enums = []
        index = json_data.get('index', {})
        
        for item_id, item in index.items():
            if item.get('kind') == 'enum':
                enum_doc = self._extract_item_documentation(item, 'enum')
                if enum_doc:
                    enums.append(enum_doc)
                    self._save_item_to_file(enum_doc, 'enum')
        
        return enums
    
    def _extract_traits(self, json_data: Dict, root_crate: Dict) -> List[Dict]:
        """Extract trait documentation"""
        traits = []
        index = json_data.get('index', {})
        
        for item_id, item in index.items():
            if item.get('kind') == 'trait':
                trait_doc = self._extract_item_documentation(item, 'trait')
                if trait_doc:
                    traits.append(trait_doc)
                    self._save_item_to_file(trait_doc, 'trait')
        
        return traits
    
    def _extract_item_documentation(self, item: Dict, item_type: str) -> Optional[Dict]:
        """Extract documentation for a single item"""
        try:
            name = item.get('name', 'unnamed')
            docs = item.get('docs', '')
            
            # Skip items without documentation
            if not docs or len(docs.strip()) < 10:
                return None
            
            doc_info = {
                'name': name,
                'type': item_type,
                'docs': docs,
                'attrs': item.get('attrs', []),
                'visibility': item.get('visibility', {}),
                'source': item.get('source', {}),
                'inner': item.get('inner', {})
            }
            
            return doc_info
            
        except Exception as e:
            self.log(f"Error extracting item documentation: {e}", "⚠️")
            return None
    
    def _save_item_to_file(self, item_doc: Dict, item_type: str):
        """Save item documentation to a markdown file"""
        try:
            name = item_doc['name']
            # Create safe filename
            safe_name = re.sub(r'[^\w\-_.]', '_', name)
            filename = f"{item_type}_{safe_name}.md"
            filepath = self.output_dir / filename
            
            # Generate markdown content
            content = self._generate_markdown_content(item_doc)
            
            # Write to file
            filepath.write_text(content, encoding='utf-8')
            
        except Exception as e:
            self.log(f"Error saving {item_type} {item_doc.get('name')}: {e}", "⚠️")
    
    def _generate_markdown_content(self, item_doc: Dict) -> str:
        """Generate markdown content for an item"""
        name = item_doc['name']
        item_type = item_doc['type']
        docs = item_doc['docs']
        
        content = f"# {item_type.title()}: {name}\n\n"
        
        # Add documentation
        if docs:
            content += f"{docs}\n\n"
        
        # Add source information if available
        source = item_doc.get('source', {})
        if source:
            filename = source.get('filename', '')
            if filename:
                content += f"**Source**: {filename}\n\n"
        
        # Add visibility information
        visibility = item_doc.get('visibility', {})
        if visibility:
            content += f"**Visibility**: {visibility}\n\n"
        
        # Add type-specific information
        inner = item_doc.get('inner', {})
        if inner and item_type == 'function':
            # Add function signature information
            if 'decl' in inner:
                content += "## Function Signature\n\n"
                content += f"```rust\n{inner.get('decl', '')}\n```\n\n"
        
        return content
    
    def _create_overview_file(self, crate_name: str, version: str, extracted_items: Dict):
        """Create an overview file with all extracted items"""
        overview_path = self.output_dir / f"{crate_name}_overview.md"
        
        content = f"# {crate_name} v{version} - Documentation Overview\n\n"
        content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for item_type, items in extracted_items.items():
            if items:
                content += f"## {item_type.title()} ({len(items)} items)\n\n"
                for item in items:
                    name = item['name']
                    content += f"- **{name}**: {item['docs'][:100]}{'...' if len(item['docs']) > 100 else ''}\n"
                content += "\n"
        
        overview_path.write_text(content, encoding='utf-8')
        self.log(f"Created overview: {overview_path.name}", "📋")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract documentation using Rustdoc JSON API")
    parser.add_argument("url", help="docs.rs URL (e.g., https://docs.rs/ratatui/latest/ratatui/)")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    extractor = RustdocJsonExtractor(
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    success = extractor.extract_from_crate_url(args.url)
    
    if success:
        print("✅ Extraction completed successfully!")
    else:
        print("❌ Extraction failed!")
        exit(1)


if __name__ == "__main__":
    main()