# How to Find Sitemaps for Any Website

## Common Sitemap Locations

Most websites follow standard conventions for sitemap locations. Try these URLs first:

### 1. **Standard Sitemap Locations**
```
https://domain.com/sitemap.xml
https://domain.com/sitemap_index.xml
https://domain.com/sitemap.xml.gz
https://domain.com/sitemap/sitemap.xml
https://domain.com/sitemaps/sitemap.xml
```

### 2. **Check robots.txt**
The robots.txt file often contains sitemap location:
```
https://domain.com/robots.txt
```

Look for lines like:
```
Sitemap: https://domain.com/sitemap.xml
```

### 3. **Common CMS-Specific Locations**
- WordPress: `/sitemap_index.xml` or `/wp-sitemap.xml`
- Drupal: `/sitemap.xml`
- Joomla: `/index.php?option=com_xmap&view=xml`

## For docs.rs Specifically

Let's check docs.rs:

```bash
# Check robots.txt
curl https://docs.rs/robots.txt

# Try common sitemap locations
curl -I https://docs.rs/sitemap.xml
curl -I https://docs.rs/sitemap_index.xml
```

### Results for docs.rs:
- `https://docs.rs/robots.txt` exists but doesn't list a sitemap
- `https://docs.rs/sitemap.xml` returns 404 (not found)
- docs.rs appears to not have a traditional sitemap

## Alternative Approaches When No Sitemap Exists

### 1. **Use a Sitemap Generator**
Create your own sitemap by crawling the site:

```bash
# Using Python with scrapy
pip install scrapy
scrapy crawl -s USER_AGENT="Mozilla/5.0" -s ROBOTSTXT_OBEY=True

# Using online tools
# - xml-sitemaps.com (free for up to 500 pages)
# - screaming frog SEO spider
```

### 2. **Create a Custom Scraper**
For docs.rs specifically, you could:
- Use their API if available
- Scrape the navigation/index pages
- Follow documentation structure

### 3. **Manual URL Collection**
For documentation sites like docs.rs/ratatui:
- Start with the main index page
- Extract all documentation links
- Build your own URL list

## Specific Solution for docs.rs/ratatui

Since docs.rs doesn't have a sitemap, here's a custom approach:

```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def get_docs_rs_urls(base_url):
    """Extract all documentation URLs from a docs.rs crate"""
    urls = set()
    to_visit = [base_url]
    visited = set()
    
    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
            
        visited.add(url)
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links in the sidebar and content
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                # Only include URLs from the same crate
                if full_url.startswith(base_url) and full_url not in urls:
                    urls.add(full_url)
                    if full_url not in visited:
                        to_visit.append(full_url)
                        
        except Exception as e:
            print(f"Error processing {url}: {e}")
            
    return sorted(urls)

# Usage
base_url = "https://docs.rs/ratatui/latest/ratatui/"
urls = get_docs_rs_urls(base_url)

# Save as a simple text file
with open("ratatui_urls.txt", "w") as f:
    for url in urls:
        f.write(url + "\n")
```

## Quick Bash Solution

For a quick check of common sitemap locations:

```bash
#!/bin/bash
DOMAIN="$1"

echo "Checking for sitemaps on $DOMAIN..."

# Check robots.txt
echo "Checking robots.txt..."
curl -s "$DOMAIN/robots.txt" | grep -i sitemap

# Check common sitemap locations
for path in "sitemap.xml" "sitemap_index.xml" "sitemap.xml.gz" "sitemap/sitemap.xml"; do
    echo "Checking $DOMAIN/$path..."
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$DOMAIN/$path")
    if [ "$STATUS" = "200" ]; then
        echo "✓ Found sitemap at $DOMAIN/$path"
    else
        echo "✗ No sitemap at $DOMAIN/$path (HTTP $STATUS)"
    fi
done
```

## Using with extract.py

Once you have URLs (either from a sitemap or collected manually):

### Option 1: Create a Simple Sitemap
```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://docs.rs/ratatui/latest/ratatui/index.html</loc>
  </url>
  <url>
    <loc>https://docs.rs/ratatui/latest/ratatui/widgets/index.html</loc>
  </url>
  <!-- Add more URLs -->
</urlset>
```

### Option 2: Process URLs Directly
Modify extract.py to accept a text file of URLs:
```bash
# Process each URL individually
while read -r url; do
    python src/extract.py html "$url" --format markdown
done < ratatui_urls.txt
```

## Recommended Approach for docs.rs

Since docs.rs is a documentation hosting service without sitemaps, the best approach is:

1. Use their crate documentation structure
2. Start from the index page and follow navigation links
3. Use a web scraper to collect all documentation URLs
4. Process the collected URLs with extract.py

This gives you more control and ensures you get all the documentation pages you need.