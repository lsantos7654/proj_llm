#!/usr/bin/env python3
"""
Smart Document Extractor - Automatically determines the best extraction method
Handles docs.rs, sites with sitemaps, and general websites intelligently
"""

import argparse
import sys
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Tuple, Optional

# Import the existing extraction functionality
from extract import DocumentConverterCLI, MarkdownCleaner


class SmartExtractor:
    """Intelligent extractor that automatically determines the best extraction method"""
    
    def __init__(self, output_dir="output", clean=True, verbose=True):
        self.output_dir = Path(output_dir)
        self.clean = clean
        self.verbose = verbose
        self.converter = DocumentConverterCLI()
        self.output_dir.mkdir(exist_ok=True)
        
    def log(self, message, emoji="ℹ️", level="info"):
        """Pretty logging with emojis"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            colors = {
                "info": "\033[36m",    # Cyan
                "success": "\033[32m", # Green
                "warning": "\033[33m", # Yellow
                "error": "\033[31m",   # Red
                "debug": "\033[90m"    # Gray
            }
            reset = "\033[0m"
            color = colors.get(level, "")
            print(f"{color}[{timestamp}] {emoji}  {message}{reset}")
    
    def extract(self, url: str) -> bool:
        """Main extraction method that figures out the best approach"""
        self.log(f"Starting smart extraction for: {url}", "🚀", "info")
        
        # Parse and validate URL
        parsed = urlparse(url)
        if not parsed.scheme:
            url = f"https://{url}"
            parsed = urlparse(url)
        
        domain = parsed.netloc
        
        # Determine extraction method
        if "docs.rs" in domain:
            self.log("Detected docs.rs - using specialized extractor", "🦀", "info")
            return self._extract_docs_rs(url)
        else:
            self.log("Checking for sitemap...", "🔍", "info")
            sitemap_url = self._find_sitemap(url)
            
            if sitemap_url:
                self.log(f"Found sitemap at: {sitemap_url}", "✅", "success")
                return self._extract_sitemap(sitemap_url)
            else:
                self.log("No sitemap found - using smart crawling", "🕷️", "warning")
                return self._extract_by_crawling(url)
    
    def _find_sitemap(self, url: str) -> Optional[str]:
        """Find sitemap for a given URL"""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Check robots.txt first
        try:
            robots_url = f"{base_url}/robots.txt"
            self.log(f"Checking robots.txt...", "🤖", "debug")
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        self.log(f"Found sitemap in robots.txt", "📍", "success")
                        return sitemap_url
        except:
            pass
        
        # Check common sitemap locations
        common_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap.xml.gz',
            '/sitemap/sitemap.xml',
            '/sitemaps/sitemap.xml',
            '/wp-sitemap.xml'  # WordPress
        ]
        
        for path in common_paths:
            sitemap_url = base_url + path
            try:
                self.log(f"Checking {path}...", "🔎", "debug")
                response = requests.head(sitemap_url, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    return sitemap_url
            except:
                continue
        
        return None
    
    def _extract_sitemap(self, sitemap_url: str) -> bool:
        """Extract all URLs from a sitemap"""
        try:
            self.log("Processing sitemap...", "📑", "info")
            
            # Let the existing converter handle it
            self.converter.convert_sitemap(
                sitemap_url, 
                output_format="markdown",
                output_dir=str(self.output_dir),
                clean_output=self.clean
            )
            
            # Count files created
            md_files = list(self.output_dir.glob("*.md"))
            self.log(f"Successfully extracted {len(md_files)} pages", "🎉", "success")
            
            if self.clean:
                self.log("Content was cleaned during extraction", "🧹", "success")
            
            return True
            
        except Exception as e:
            self.log(f"Error extracting sitemap: {e}", "❌", "error")
            return False
    
    def _extract_docs_rs(self, url: str) -> bool:
        """Extract documentation from docs.rs using high-quality JSON API only"""
        try:
            self.log("Using high-quality Rustdoc JSON API for docs.rs", "🦀", "info")
            
            # Import and use the rustdoc JSON extractor
            from rustdoc_json_extractor import RustdocJsonExtractor
            
            extractor = RustdocJsonExtractor(
                output_dir=str(self.output_dir),
                verbose=self.verbose
            )
            
            result = extractor.extract_from_crate_url(url)
            
            if result:
                self.log("Rustdoc JSON extraction completed successfully", "🎉", "success")
            else:
                self.log("Rustdoc JSON extraction failed - no fallback for docs.rs", "❌", "error")
            
            return result
            
        except ImportError:
            self.log("Rustdoc JSON extractor not available", "❌", "error")
            return False
        except Exception as e:
            self.log(f"Error with Rustdoc JSON extraction: {e}", "❌", "error")
            return False
    
    
    def _extract_by_crawling(self, url: str) -> bool:
        """Extract by intelligently crawling the website"""
        try:
            self.log("Starting intelligent crawl...", "🤖", "info")
            
            # Get all linked pages from the starting URL
            urls = self._smart_crawl(url)
            self.log(f"Found {len(urls)} pages to extract", "📊", "info")
            
            if not urls:
                self.log("No pages found to extract", "⚠️", "warning")
                return False
            
            # Extract URLs directly
            result = self._extract_urls_directly(urls)
            
            return result
            
        except Exception as e:
            self.log(f"Error during crawl extraction: {e}", "❌", "error")
            return False
    
    def _smart_crawl(self, start_url: str, max_pages=100) -> List[str]:
        """Intelligently crawl a website to find documentation pages"""
        parsed_start = urlparse(start_url)
        base_domain = f"{parsed_start.scheme}://{parsed_start.netloc}"
        
        urls = set()
        visited = set()
        to_visit = [start_url]
        
        # Patterns that indicate documentation/content pages
        good_patterns = ['docs', 'guide', 'tutorial', 'api', 'reference', 'manual']
        skip_patterns = ['.pdf', '.zip', '.tar', '.gz', 'download', 'signin', 'login']
        
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            
            visited.add(url)
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                
                # Check if it looks like a content page
                if not any(skip in url.lower() for skip in skip_patterns):
                    urls.add(url)
                
                # Find more links
                soup = BeautifulSoup(response.content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    
                    # Only follow same-domain links
                    if (full_url.startswith(base_domain) and 
                        full_url not in visited and
                        not any(skip in full_url.lower() for skip in skip_patterns)):
                        
                        # Prioritize documentation-like URLs
                        if any(good in full_url.lower() for good in good_patterns):
                            to_visit.insert(0, full_url)  # Add to front
                        else:
                            to_visit.append(full_url)
                
                time.sleep(0.1)  # Be respectful
                
            except Exception as e:
                self.log(f"Error crawling {url}: {e}", "⚠️", "debug")
                continue
        
        return sorted(list(urls))
    
    def _fallback_html_extract(self, url: str) -> str:
        """Fallback HTML extraction using requests + BeautifulSoup when docling fails"""
        try:
            self.log(f"Using fallback extraction for: {url}", "🔧", "debug")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content areas
            main_content = None
            for selector in ['main', '.main-content', '#main', '.content', 'article']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.body or soup
            
            # Convert to markdown-like format
            text = main_content.get_text(separator='\n', strip=True)
            
            # Basic markdown formatting
            lines = text.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    formatted_lines.append(line)
            
            return '\n\n'.join(formatted_lines)
            
        except Exception as e:
            self.log(f"Fallback extraction also failed for {url}: {e}", "❌", "error")
            return ""
    
    def _extract_urls_directly(self, urls: List[str]) -> bool:
        """Extract content directly from a list of URLs"""
        success_count = 0
        
        for i, url in enumerate(urls, 1):
            try:
                self.log(f"Processing [{i}/{len(urls)}]: {url}", "📄", "info")
                
                # Use the HTML converter with error handling
                try:
                    content = self.converter.convert_html(url, output_format="markdown")
                except Exception as docling_error:
                    self.log(f"Docling failed for {url}, trying fallback: {str(docling_error)[:100]}...", "⚠️", "warning")
                    # Fallback to simple requests + BeautifulSoup
                    content = self._fallback_html_extract(url)
                
                if not content or len(content.strip()) < 50:
                    self.log(f"No meaningful content extracted from {url}", "⚠️", "warning")
                    continue
                
                # Clean content if requested
                if self.clean:
                    cleaner = MarkdownCleaner()
                    content = cleaner.clean_content(content)
                
                # Create filename from URL
                parsed = urlparse(url)
                path_parts = parsed.path.strip('/').split('/')
                if path_parts[-1]:
                    filename = path_parts[-1]
                else:
                    filename = path_parts[-2] if len(path_parts) > 1 else "index"
                
                # Clean filename
                filename = re.sub(r'[^\w\-_.]', '_', filename)
                if not filename.endswith('.md'):
                    filename += '.md'
                
                # Write to file
                output_path = self.output_dir / filename
                
                # Handle filename conflicts
                counter = 1
                original_filename = filename
                while output_path.exists():
                    name, ext = original_filename.rsplit('.', 1)
                    filename = f"{name}_{counter}.{ext}"
                    output_path = self.output_dir / filename
                    counter += 1
                
                output_path.write_text(content, encoding='utf-8')
                success_count += 1
                self.log(f"Saved: {filename}", "✅", "debug")
                
            except Exception as e:
                self.log(f"Error processing {url}: {str(e)[:100]}...", "❌", "error")
                continue
        
        if success_count > 0:
            self.log(f"Successfully extracted {success_count}/{len(urls)} pages", "🎉", "success")
            
            if self.clean:
                self.log("Content was cleaned during extraction", "🧹", "success")
            
            return True
        else:
            self.log("Failed to extract any pages", "❌", "error")
            return False
    
    def _create_sitemap_from_urls(self, urls: List[str], output_path: Path):
        """Create a sitemap XML file from a list of URLs"""
        urlset = ET.Element('urlset')
        urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')
        
        for url in urls:
            url_elem = ET.SubElement(urlset, 'url')
            loc = ET.SubElement(url_elem, 'loc')
            loc.text = url
            lastmod = ET.SubElement(url_elem, 'lastmod')
            lastmod.text = datetime.now().strftime('%Y-%m-%d')
        
        tree = ET.ElementTree(urlset)
        tree.write(str(output_path), encoding='utf-8', xml_declaration=True)
        self.log(f"Created temporary sitemap with {len(urls)} URLs", "📝", "debug")


def main():
    parser = argparse.ArgumentParser(
        description="Smart document extractor - automatically handles any website"
    )
    parser.add_argument("url", help="URL to extract (site, sitemap, or docs.rs crate)")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--no-clean", action="store_true", help="Don't clean markdown output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    extractor = SmartExtractor(
        output_dir=args.output_dir,
        clean=not args.no_clean,
        verbose=not args.quiet
    )
    
    success = extractor.extract(args.url)
    
    if success:
        extractor.log("Extraction completed successfully! 🎉", "✅", "success")
        extractor.log(f"Files saved to: {args.output_dir}", "📁", "info")
    else:
        extractor.log("Extraction failed. Please check the URL and try again.", "❌", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()