# Sitemap Finding and Extraction Guide

## Quick Start

### 1. Find a Sitemap
```bash
# Check common sitemap locations for any domain
make find-sitemap DOMAIN=https://example.com

# For docs.rs specifically
make find-sitemap DOMAIN=https://docs.rs
```

### 2. Extract Documentation

**If sitemap exists:**
```bash
make extract-sitemap-clean URL=https://example.com/sitemap.xml
```

**If no sitemap exists:**
```bash
# Generate URL list for docs.rs crates
python extract_docs_rs.py https://docs.rs/ratatui/latest/ratatui --txt

# Extract from URL list
make extract-urls FILE=urls.txt

# Clean the extracted files
make clean-markdown DIR=output
```

## Detailed Instructions

### Finding Sitemaps

Most websites follow standard conventions. The `check_sitemap.sh` script checks:

1. **robots.txt** - Often contains sitemap location
2. **Common paths**:
   - `/sitemap.xml`
   - `/sitemap_index.xml`
   - `/sitemap.xml.gz`
   - `/sitemap/sitemap.xml`

### Working with docs.rs

docs.rs is special because individual crate documentation doesn't have sitemaps. Use the custom extractor:

```bash
# Extract all documentation URLs from a crate
python extract_docs_rs.py https://docs.rs/ratatui/latest/ratatui

# This creates:
# - sitemap.xml (standard sitemap format)
# - urls.txt (simple list of URLs)

# Then extract all pages
make extract-urls FILE=urls.txt

# Finally, clean the output
make clean-markdown DIR=output
```

### Alternative Methods

#### 1. Use Online Sitemap Generators
- [xml-sitemaps.com](https://www.xml-sitemaps.com/) (free for up to 500 pages)
- Screaming Frog SEO Spider
- HTTrack Website Copier

#### 2. Create Manual URL List
Create a file `urls.txt`:
```
https://docs.rs/ratatui/latest/ratatui/index.html
https://docs.rs/ratatui/latest/ratatui/widgets/index.html
https://docs.rs/ratatui/latest/ratatui/layout/index.html
# Add more URLs...
```

Then extract:
```bash
make extract-urls FILE=urls.txt
```

#### 3. Use Web Scraping
For comprehensive extraction:
```python
# Simple scraper for any website
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_all_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    links = set()
    for link in soup.find_all('a', href=True):
        full_url = urljoin(url, link['href'])
        if full_url.startswith(url):  # Same domain
            links.add(full_url)
    
    return links
```

## Common Sitemap Formats

### Standard XML Sitemap
```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/page1.html</loc>
    <lastmod>2024-01-01</lastmod>
  </url>
  <url>
    <loc>https://example.com/page2.html</loc>
    <lastmod>2024-01-01</lastmod>
  </url>
</urlset>
```

### Sitemap Index (for large sites)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://example.com/sitemap1.xml</loc>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap2.xml</loc>
  </sitemap>
</sitemapindex>
```

## Complete Workflow Example

### For a site with sitemap:
```bash
# 1. Find the sitemap
make find-sitemap DOMAIN=https://example.com

# 2. Extract and clean in one step
make extract-sitemap-clean URL=https://example.com/sitemap.xml
```

### For docs.rs or sites without sitemaps:
```bash
# 1. Generate URL list
python extract_docs_rs.py https://docs.rs/tokio/latest/tokio --txt

# 2. Extract all URLs
make extract-urls FILE=urls.txt

# 3. Clean the markdown files
make clean-markdown DIR=output

# 4. (Optional) Check what was extracted
ls -la output/ | head -20
```

## Tips

1. **Check robots.txt first** - It often contains the sitemap location
2. **Use the all.html page** - For docs.rs, check `/all.html` for a complete index
3. **Be respectful** - Add delays between requests when scraping
4. **Check for API access** - Some sites offer APIs that are better than scraping
5. **Verify extraction** - Always check a few extracted files to ensure quality

## Troubleshooting

### No sitemap found
- Try the online generators
- Check if the site has an API
- Look for "Site Map" or "Index" pages
- Use the custom scrapers provided

### Extraction fails
- Check if the site requires authentication
- Verify the URL format is correct
- Some sites block automated access - check robots.txt
- Try extracting individual pages first

### Large sites
- Break into smaller chunks
- Use sitemap index files
- Process in batches to avoid memory issues

This guide should help you extract documentation from any website, whether it has a sitemap or not!