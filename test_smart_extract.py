#!/usr/bin/env python3
"""
Test the smart extraction system with different types of URLs
"""

import sys
import subprocess

# Test URLs
test_cases = [
    {
        "name": "docs.rs crate",
        "url": "https://docs.rs/serde/latest/serde",
        "expected": "Should detect docs.rs and use specialized extraction"
    },
    {
        "name": "Site with sitemap",
        "url": "https://python.org",
        "expected": "Should find and use sitemap"
    },
    {
        "name": "Direct sitemap URL",
        "url": "https://example.com/sitemap.xml",
        "expected": "Should recognize as sitemap and extract directly"
    }
]

def test_extraction(url, dry_run=True):
    """Test extraction without actually running it"""
    print(f"\n{'='*60}")
    print(f"Testing URL: {url}")
    print(f"{'='*60}")
    
    if dry_run:
        # Just show what would happen
        cmd = f"python src/smart_extract.py '{url}' --output-dir test_output"
        print(f"Would run: {cmd}")
    else:
        # Actually run the extraction
        result = subprocess.run(
            ["python", "src/smart_extract.py", url, "--output-dir", "test_output"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

def main():
    print("Smart Extraction Test Suite")
    print("="*60)
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"URL: {test['url']}")
        print(f"Expected: {test['expected']}")
        
    print("\n" + "="*60)
    print("To run actual tests, use: python test_smart_extract.py --run")
    
    if "--run" in sys.argv:
        print("\nRunning actual extractions...")
        for test in test_cases[:1]:  # Only test first one to avoid heavy downloads
            test_extraction(test['url'], dry_run=False)

if __name__ == "__main__":
    main()