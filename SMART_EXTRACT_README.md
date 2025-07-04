# 🚀 Smart Document Extraction System

A simplified, intelligent document extraction tool that automatically determines the best extraction method for any website.

## ✨ Features

- **🔍 Auto-detects sitemaps** - Automatically finds and uses sitemaps when available
- **🦀 docs.rs support** - Special handling for Rust documentation
- **🕷️ Smart crawling** - Intelligently crawls sites without sitemaps
- **🧹 Auto-cleaning** - Cleans markdown for optimal LLM consumption
- **📊 Progress tracking** - Real-time feedback with emojis and colored output

## 🎯 Quick Start

### One Command Does It All

```bash
# Extract any website - it figures out the best method
make extract URL=https://example.com

# Extract docs.rs crates
make extract URL=https://docs.rs/tokio/latest/tokio

# Extract without cleaning
make extract-no-clean URL=https://example.com
```

That's it! The system will:
1. Detect if it's a docs.rs site
2. Look for sitemaps automatically
3. Use intelligent crawling if needed
4. Clean the output for LLM use
5. Save everything to the `output/` directory

## 📋 How It Works

### 1. **docs.rs Detection**
If the URL contains "docs.rs", it uses a specialized crawler that:
- Starts from the main page and `/all.html`
- Follows documentation structure
- Respects rate limits
- Creates a temporary sitemap

### 2. **Sitemap Discovery**
For other sites, it checks:
- `robots.txt` for sitemap references
- Common sitemap locations:
  - `/sitemap.xml`
  - `/sitemap_index.xml`
  - `/wp-sitemap.xml`
  - And more...

### 3. **Smart Crawling**
If no sitemap is found:
- Crawls from the starting URL
- Prioritizes documentation-like pages
- Respects robots.txt
- Limits crawling depth

### 4. **Automatic Cleaning**
By default, all extracted content is cleaned:
- HTML entities decoded
- Navigation sections removed
- Formatting normalized
- Structure improved for LLMs

## 🛠️ Installation

```bash
# Install dependencies
make install

# Or manually
pip install -r requirements.txt
```

## 📚 Examples

### Extract Documentation Sites

```bash
# Python documentation
make extract URL=https://docs.python.org

# Rust crate documentation
make extract URL=https://docs.rs/serde/latest/serde

# Any documentation site
make extract URL=https://fastapi.tiangolo.com
```

### Extract and Process

```bash
# Extract with automatic cleaning (default)
make extract URL=https://example.com

# Extract without cleaning
make extract-no-clean URL=https://example.com

# Clean existing files
make clean-markdown DIR=output
```

### Other Operations

```bash
# Extract from PDF
make extract-pdf FILE=document.pdf

# Process a list of URLs
make extract-urls FILE=urls.txt

# Find sitemap for a domain
make find-sitemap DOMAIN=https://example.com
```

## 📊 Console Output

The system provides real-time feedback with emojis:

```
[12:34:56] 🚀  Starting smart extraction for: https://docs.rs/tokio
[12:34:56] 🦀  Detected docs.rs - using specialized extractor
[12:34:57] 🕸️  Crawling documentation pages...
[12:35:12] 📊  Found 127 documentation pages
[12:35:12] 📑  Processing sitemap...
[12:35:45] 🧹  Content was cleaned during extraction
[12:35:45] 🎉  Successfully extracted 127 pages
[12:35:45] ✅  Extraction completed successfully! 🎉
[12:35:45] 📁  Files saved to: output
```

## 🔧 Advanced Usage

### Python API

```python
from src.smart_extract import SmartExtractor

# Create extractor
extractor = SmartExtractor(
    output_dir="my_docs",
    clean=True,
    verbose=True
)

# Extract any URL
success = extractor.extract("https://docs.rs/tokio/latest/tokio")
```

### Command Line

```bash
# Direct usage
python src/smart_extract.py https://example.com --output-dir docs

# Quiet mode
python src/smart_extract.py https://example.com --quiet

# No cleaning
python src/smart_extract.py https://example.com --no-clean
```

## 🐛 Troubleshooting

### No pages extracted?
- Check if the site requires authentication
- Verify the URL is correct
- Some sites block automated access

### Extraction is slow?
- Large sites take time to crawl
- docs.rs sites can have many pages
- Consider extracting specific sections

### Missing content?
- Check if content is loaded dynamically (JavaScript)
- Some sites require special handling
- Try extracting individual pages

## 🎯 Best Practices

1. **Start with the root URL** - Let the crawler find all pages
2. **Check robots.txt** - Respect the site's crawling rules
3. **Use specific URLs for docs.rs** - Point to the exact crate version
4. **Monitor output** - Watch the console for any issues
5. **Clean regularly** - Use the cleaning feature for better LLM results

## 📄 License

This tool is designed for legitimate documentation extraction and learning purposes. Always respect website terms of service and robots.txt files.