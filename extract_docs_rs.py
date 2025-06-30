#!/usr/bin/env python3
"""
Extract documentation URLs from docs.rs crates
Specifically designed for Rust documentation sites
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

class DocsRsExtractor:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.visited = set()
        self.urls = set()
        
    def extract_urls(self):
        """Extract all documentation URLs from a docs.rs crate"""
        # Start with the main page and all.html if it exists
        start_urls = [
            self.base_url + '/',
            self.base_url + '/index.html',
            self.base_url + '/all.html'
        ]
        
        for url in start_urls:
            self._crawl_page(url)
            
        return sorted(self.urls)
    
    def _crawl_page(self, url):
        """Crawl a single page and extract links"""
        if url in self.visited:
            return
            
        self.visited.add(url)
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Add current URL if it's a documentation page
            if self._is_doc_page(url):
                self.urls.add(url)
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                # Only process URLs within the same crate documentation
                if (full_url.startswith(self.base_url) and 
                    full_url not in self.visited and
                    self._is_valid_doc_url(full_url)):
                    self._crawl_page(full_url)
                    
        except Exception as e:
            print(f"Error processing {url}: {e}", file=sys.stderr)
    
    def _is_doc_page(self, url):
        """Check if URL is a documentation page"""
        # Skip source code views and other non-doc pages
        skip_patterns = ['/src/', '?search=', '#', '.js', '.css', '.json']
        return not any(pattern in url for pattern in skip_patterns)
    
    def _is_valid_doc_url(self, url):
        """Check if URL should be crawled"""
        # Include HTML pages and directory indexes
        return url.endswith('.html') or url.endswith('/')
    
    def create_sitemap(self, urls, output_file='sitemap.xml'):
        """Create a sitemap XML file from URLs"""
        urlset = ET.Element('urlset')
        urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')
        
        for url in urls:
            url_elem = ET.SubElement(urlset, 'url')
            loc = ET.SubElement(url_elem, 'loc')
            loc.text = url
            lastmod = ET.SubElement(url_elem, 'lastmod')
            lastmod.text = datetime.now().strftime('%Y-%m-%d')
            
        tree = ET.ElementTree(urlset)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"Sitemap saved to {output_file}")
    
    def save_urls_txt(self, urls, output_file='urls.txt'):
        """Save URLs as a simple text file"""
        with open(output_file, 'w') as f:
            for url in urls:
                f.write(url + '\n')
        print(f"URL list saved to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_docs_rs.py <docs.rs-url> [--sitemap] [--txt]")
        print("Example: python extract_docs_rs.py https://docs.rs/ratatui/latest/ratatui")
        sys.exit(1)
    
    base_url = sys.argv[1]
    create_sitemap = '--sitemap' in sys.argv
    create_txt = '--txt' in sys.argv
    
    # Default to creating both if neither specified
    if not create_sitemap and not create_txt:
        create_sitemap = create_txt = True
    
    print(f"Extracting URLs from {base_url}...")
    extractor = DocsRsExtractor(base_url)
    urls = extractor.extract_urls()
    
    print(f"Found {len(urls)} documentation pages")
    
    if create_sitemap:
        extractor.create_sitemap(urls)
        
    if create_txt:
        extractor.save_urls_txt(urls)
    
    # Print first few URLs as preview
    print("\nFirst 10 URLs:")
    for url in urls[:10]:
        print(f"  - {url}")
    
    if len(urls) > 10:
        print(f"  ... and {len(urls) - 10} more")


if __name__ == "__main__":
    main()