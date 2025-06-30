# Makefile for Document Extraction and Processing
# Provides convenient commands for extracting documentation from websites and PDFs,
# converting to markdown, and cleaning output for LLM consumption.

# Default configuration
PYTHON := python
SRC_DIR := src
OUTPUT_DIR := output
EXTRACT_SCRIPT := $(SRC_DIR)/extract.py

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
	@echo "  make extract-sitemap URL=https://example.com/sitemap.xml"
	@echo "  make extract-sitemap-clean URL=https://example.com/sitemap.xml"
	@echo "  make extract-html URL=https://example.com/page.html"
	@echo "  make extract-pdf FILE=document.pdf"
	@echo "  make clean-markdown DIR=output"
	@echo "  make restore-backups DIR=output"

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

# Extract documentation from sitemap
extract-sitemap: create-output-dir ## Extract all pages from a sitemap (requires URL=<sitemap_url>)
	@if [ -z "$(URL)" ]; then \
		echo "$(RED)Error: URL parameter is required$(RESET)"; \
		echo "Usage: make extract-sitemap URL=https://example.com/sitemap.xml"; \
		exit 1; \
	fi
	@echo "$(BLUE)Extracting sitemap: $(URL)$(RESET)"
	$(PYTHON) $(EXTRACT_SCRIPT) sitemap "$(URL)" --output-dir $(OUTPUT_DIR) --format markdown
	@echo "$(GREEN)Sitemap extraction completed!$(RESET)"

# Extract and clean documentation from sitemap  
extract-sitemap-clean: create-output-dir ## Extract and clean all pages from a sitemap (requires URL=<sitemap_url>)
	@if [ -z "$(URL)" ]; then \
		echo "$(RED)Error: URL parameter is required$(RESET)"; \
		echo "Usage: make extract-sitemap-clean URL=https://example.com/sitemap.xml"; \
		exit 1; \
	fi
	@echo "$(BLUE)Extracting and cleaning sitemap: $(URL)$(RESET)"
	$(PYTHON) $(EXTRACT_SCRIPT) sitemap "$(URL)" --output-dir $(OUTPUT_DIR) --format markdown --clean
	@echo "$(GREEN)Sitemap extraction and cleaning completed!$(RESET)"

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
	$(PYTHON) $(EXTRACT_SCRIPT) clean "$(DIR)"
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
	$(PYTHON) $(EXTRACT_SCRIPT) clean "$(DIR)" --no-backup
	@echo "$(GREEN)Markdown cleaning completed!$(RESET)"

# Restore backups
restore-backups: ## Restore all .backup files in a directory (requires DIR=<directory>)
	@if [ -z "$(DIR)" ]; then \
		echo "$(RED)Error: DIR parameter is required$(RESET)"; \
		echo "Usage: make restore-backups DIR=output"; \
		exit 1; \
	fi
	@if [ ! -d "$(DIR)" ]; then \
		echo "$(RED)Error: Directory $(DIR) not found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restoring backup files in: $(DIR)$(RESET)"
	@find "$(DIR)" -name "*.backup" -exec sh -c 'mv "$$1" "$${1%.*}"' _ {} \; 2>/dev/null || true
	@echo "$(GREEN)Backup restoration completed!$(RESET)"

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