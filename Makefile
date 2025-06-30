# Makefile for Document Extraction and Processing
# Provides convenient commands for extracting documentation from websites and PDFs,
# converting to markdown, and cleaning output for LLM consumption.

# Default configuration
PYTHON := python
SRC_DIR := src
OUTPUT_DIR := output
EXTRACT_SCRIPT := $(SRC_DIR)/extract.py
VENV_ACTIVATE := . venv/bin/activate &&

# Colors for pretty output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

.PHONY: help install clean-output extract-sitemap extract-html extract-pdf clean-markdown test-extraction all

# Default target
help: ## Show this help message
	@echo "$(BLUE)Document Extraction and Processing Makefile$(RESET)"
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BLUE)Examples:$(RESET)"
	@echo "  make extract URL=https://example.com                     # Auto-detects extraction method"
	@echo "  make extract URL=https://docs.rs/tokio/latest/tokio     # Handles docs.rs automatically"
	@echo "  make extract-pdf FILE=document.pdf                      # Extract from PDF"
	@echo "  make clean-markdown DIR=output                          # Clean existing files"
	@echo ""
	@echo "$(GREEN)Smart Features:$(RESET)"
	@echo "  🔍 Auto-detects sitemaps"
	@echo "  🦀 Special handling for docs.rs"
	@echo "  🕷️  Intelligent crawling when no sitemap exists"
	@echo "  🧹 Automatic markdown cleaning"

install: ## Install required dependencies
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully!$(RESET)"

clean-output: ## Remove all files from output directory
	@echo "$(YELLOW)Cleaning output directory...$(RESET)"
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		find $(OUTPUT_DIR) -type f -name "*.md" -delete 2>/dev/null || true; \
		find $(OUTPUT_DIR) -type f -name "*.json" -delete 2>/dev/null || true; \
		find $(OUTPUT_DIR) -type f -name "*.backup" -delete 2>/dev/null || true; \
		echo "$(GREEN)Output directory cleaned!$(RESET)"; \
	else \
		echo "$(YELLOW)Output directory doesn't exist.$(RESET)"; \
	fi

create-output-dir: ## Create output directory if it doesn't exist
	@mkdir -p $(OUTPUT_DIR)

# Smart extraction - automatically handles any website
extract: create-output-dir ## Smart extraction from any website (requires URL=<site_url>)
	@if [ -z "$(URL)" ]; then \
		echo "$(RED)Error: URL parameter is required$(RESET)"; \
		echo "Usage: make extract URL=https://example.com"; \
		echo "       make extract URL=https://docs.rs/tokio/latest/tokio"; \
		exit 1; \
	fi
	@echo "$(BLUE)🚀 Starting smart extraction for: $(URL)$(RESET)"
	$(VENV_ACTIVATE) $(PYTHON) src/smart_extract.py "$(URL)" --output-dir $(OUTPUT_DIR)
	@echo "$(GREEN)✨ Smart extraction completed!$(RESET)"

# Extract without cleaning
extract-no-clean: create-output-dir ## Smart extraction without markdown cleaning (requires URL=<site_url>)
	@if [ -z "$(URL)" ]; then \
		echo "$(RED)Error: URL parameter is required$(RESET)"; \
		echo "Usage: make extract-no-clean URL=https://example.com"; \
		exit 1; \
	fi
	@echo "$(BLUE)🚀 Starting extraction (no cleaning): $(URL)$(RESET)"
	$(VENV_ACTIVATE) $(PYTHON) src/smart_extract.py "$(URL)" --output-dir $(OUTPUT_DIR) --no-clean
	@echo "$(GREEN)✨ Extraction completed!$(RESET)"

# Legacy commands for backward compatibility
extract-sitemap: create-output-dir ## [Legacy] Extract from sitemap URL (use 'make extract' instead)
	@echo "$(YELLOW)⚠️  This command is deprecated. Use 'make extract URL=...' instead$(RESET)"
	@if [ -z "$(URL)" ]; then \
		echo "$(RED)Error: URL parameter is required$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(EXTRACT_SCRIPT) sitemap "$(URL)" --output-dir $(OUTPUT_DIR) --format markdown

extract-sitemap-clean: extract-sitemap ## [Legacy] Use 'make extract' instead
	@echo "$(YELLOW)⚠️  This command is deprecated. Use 'make extract URL=...' instead$(RESET)"

# Extract single HTML page
extract-html: ## Extract content from a single HTML page (requires URL=<page_url>)
	@if [ -z "$(URL)" ]; then \
		echo "$(RED)Error: URL parameter is required$(RESET)"; \
		echo "Usage: make extract-html URL=https://example.com/page.html"; \
		exit 1; \
	fi
	@echo "$(BLUE)Extracting HTML page: $(URL)$(RESET)"
	$(PYTHON) $(EXTRACT_SCRIPT) html "$(URL)" --format markdown
	@echo "$(GREEN)HTML extraction completed!$(RESET)"

# Extract PDF file
extract-pdf: ## Extract content from a PDF file (requires FILE=<pdf_path>)
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: FILE parameter is required$(RESET)"; \
		echo "Usage: make extract-pdf FILE=document.pdf"; \
		exit 1; \
	fi
	@if [ ! -f "$(FILE)" ]; then \
		echo "$(RED)Error: File $(FILE) not found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Extracting PDF: $(FILE)$(RESET)"
	$(PYTHON) $(EXTRACT_SCRIPT) pdf "$(FILE)" --format markdown
	@echo "$(GREEN)PDF extraction completed!$(RESET)"

# Clean existing markdown files
clean-markdown: ## Clean markdown files for better LLM parsing (requires DIR=<directory>)
	@if [ -z "$(DIR)" ]; then \
		echo "$(RED)Error: DIR parameter is required$(RESET)"; \
		echo "Usage: make clean-markdown DIR=output"; \
		exit 1; \
	fi
	@if [ ! -d "$(DIR)" ]; then \
		echo "$(RED)Error: Directory $(DIR) not found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Cleaning markdown files in: $(DIR)$(RESET)"
	$(VENV_ACTIVATE) $(PYTHON) $(EXTRACT_SCRIPT) clean "$(DIR)"
	@echo "$(GREEN)Markdown cleaning completed!$(RESET)"

# Clean without backups
clean-markdown-no-backup: ## Clean markdown files without creating backups (requires DIR=<directory>)
	@if [ -z "$(DIR)" ]; then \
		echo "$(RED)Error: DIR parameter is required$(RESET)"; \
		echo "Usage: make clean-markdown-no-backup DIR=output"; \
		exit 1; \
	fi
	@if [ ! -d "$(DIR)" ]; then \
		echo "$(RED)Error: Directory $(DIR) not found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Cleaning markdown files in: $(DIR) (no backups)$(RESET)"
	$(VENV_ACTIVATE) $(PYTHON) $(EXTRACT_SCRIPT) clean "$(DIR)" --no-backup
	@echo "$(GREEN)Markdown cleaning completed!$(RESET)"

# Restore backups
restore-backups: ## Restore all files from backup subdirectory (requires DIR=<directory>)
	@if [ -z "$(DIR)" ]; then \
		echo "$(RED)Error: DIR parameter is required$(RESET)"; \
		echo "Usage: make restore-backups DIR=output"; \
		exit 1; \
	fi
	@if [ ! -d "$(DIR)" ]; then \
		echo "$(RED)Error: Directory $(DIR) not found$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -d "$(DIR)/backup" ]; then \
		echo "$(RED)Error: Backup directory $(DIR)/backup not found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restoring backup files from: $(DIR)/backup$(RESET)"
	@find "$(DIR)/backup" -name "*.md" -exec sh -c 'cp "$$1" "$(DIR)/$$(basename "$$1")"' _ {} \; 2>/dev/null || true
	@echo "$(GREEN)Backup restoration completed!$(RESET)"

# Extract from URL list
extract-urls: create-output-dir ## Extract content from a list of URLs (requires FILE=<url_list.txt>)
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: FILE parameter is required$(RESET)"; \
		echo "Usage: make extract-urls FILE=urls.txt"; \
		exit 1; \
	fi
	@if [ ! -f "$(FILE)" ]; then \
		echo "$(RED)Error: File $(FILE) not found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Extracting URLs from: $(FILE)$(RESET)"
	@count=0; \
	while IFS= read -r url || [ -n "$$url" ]; do \
		if [ -n "$$url" ] && [ "$${url:0:1}" != "#" ]; then \
			count=$$((count + 1)); \
			echo "$(YELLOW)Processing [$$count]: $$url$(RESET)"; \
			filename=$$(echo "$$url" | sed 's|https\?://||g' | sed 's|[^a-zA-Z0-9.-]|_|g'); \
			if [ "$${filename: -1}" = "_" ]; then filename="$${filename}index"; fi; \
			if [[ ! "$$filename" =~ \.md$$ ]]; then filename="$$filename.md"; fi; \
			$(PYTHON) $(EXTRACT_SCRIPT) html "$$url" --format markdown > "$(OUTPUT_DIR)/$$filename" 2>/dev/null || \
				echo "$(RED)Failed to extract: $$url$(RESET)"; \
		fi; \
	done < "$(FILE)"; \
	echo "$(GREEN)Processed $$count URLs!$(RESET)"

# Helper to find sitemaps
find-sitemap: ## Check common sitemap locations for a domain (requires DOMAIN=<domain>)
	@if [ -z "$(DOMAIN)" ]; then \
		echo "$(RED)Error: DOMAIN parameter is required$(RESET)"; \
		echo "Usage: make find-sitemap DOMAIN=https://example.com"; \
		exit 1; \
	fi
	@./check_sitemap.sh "$(DOMAIN)"

# Configuration display
config: ## Show current configuration
	@echo "$(BLUE)Current Configuration:$(RESET)"
	@echo "Python: $(PYTHON)"
	@echo "Source Directory: $(SRC_DIR)"
	@echo "Output Directory: $(OUTPUT_DIR)"
	@echo "Extract Script: $(EXTRACT_SCRIPT)"
	@echo ""
	@echo "$(BLUE)Directory Status:$(RESET)"
	@echo "Source exists: $$([ -d "$(SRC_DIR)" ] && echo "✓" || echo "✗")"
	@echo "Output exists: $$([ -d "$(OUTPUT_DIR)" ] && echo "✓" || echo "✗")"
	@echo "Extract script exists: $$([ -f "$(EXTRACT_SCRIPT)" ] && echo "✓" || echo "✗")"