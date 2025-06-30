#!/bin/bash
# Script to check for sitemaps on a given domain

DOMAIN="${1:-https://docs.rs}"

echo "Checking for sitemaps on $DOMAIN..."
echo "========================================"

# Check robots.txt
echo -e "\n1. Checking robots.txt..."
ROBOTS_CONTENT=$(curl -s "$DOMAIN/robots.txt" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✓ Found robots.txt"
    SITEMAP_IN_ROBOTS=$(echo "$ROBOTS_CONTENT" | grep -i sitemap)
    if [ -n "$SITEMAP_IN_ROBOTS" ]; then
        echo "✓ Sitemap reference found in robots.txt:"
        echo "$SITEMAP_IN_ROBOTS"
    else
        echo "✗ No sitemap reference in robots.txt"
    fi
else
    echo "✗ No robots.txt found"
fi

# Check common sitemap locations
echo -e "\n2. Checking common sitemap locations..."
for path in "sitemap.xml" "sitemap_index.xml" "sitemap.xml.gz" "sitemap/sitemap.xml" "sitemaps/sitemap.xml"; do
    URL="$DOMAIN/$path"
    echo -n "   Checking $URL... "
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null)
    if [ "$STATUS" = "200" ]; then
        echo "✓ FOUND (HTTP 200)"
        echo "   → Sitemap available at: $URL"
    else
        echo "✗ Not found (HTTP $STATUS)"
    fi
done

# For docs.rs specifically
if [[ "$DOMAIN" == *"docs.rs"* ]]; then
    echo -e "\n3. docs.rs specific information:"
    echo "   docs.rs doesn't typically provide sitemaps."
    echo "   Consider using their API or scraping the navigation."
    echo ""
    echo "   For the ratatui crate specifically:"
    echo "   - Main page: https://docs.rs/ratatui/latest/ratatui/"
    echo "   - All items: https://docs.rs/ratatui/latest/ratatui/all.html"
    echo ""
    echo "   You can use the 'all.html' page to get a list of all documented items."
fi