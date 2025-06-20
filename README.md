# LightRAG Document Intelligence System

A comprehensive document processing and knowledge graph system built on LightRAG, designed to extract, analyze, and query information from various document sources using advanced NLP and graph-based retrieval techniques.

## Overview

This project provides a powerful framework for:
- **Document Processing**: Automatically extract and chunk content from PDFs, websites, and code repositories
- **Knowledge Graph Construction**: Build semantic relationships between entities using LightRAG
- **Intelligent Search**: Query documents using multiple search modes (naive, local, global, hybrid, mix)
- **Interactive UI**: Streamlit-based chat interface with visual knowledge graph exploration

## Key Features

### 📄 Multi-Source Document Processing
- **PDF Processing**: Extract text and tables with optional vertical splitting
- **Web Scraping**: Process individual URLs or entire sitemaps
- **Code Analysis**: Tree-sitter based code parsing for semantic understanding
- **Git Repositories**: Clone and analyze entire codebases

### 🧠 Advanced Knowledge Management
- **Entity Extraction**: Automatically identify and categorize entities
- **Relationship Mapping**: Build connections between concepts and entities
- **Hybrid Chunking**: Intelligent document segmentation preserving context
- **Multi-Modal Search**: Five different search strategies for optimal retrieval

### 💬 Interactive Chat Interface
- **Conversational Q&A**: Natural language queries over your document corpus
- **Visual Knowledge Graph**: Explore entity relationships interactively
- **Context Display**: View retrieved chunks, entities, and relationships
- **Debug Information**: Transparent view into the retrieval process

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for embeddings and LLM)
- Neo4j (optional, for graph database integration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/proj_llm.git
cd proj_llm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

#### Document Processing

Process PDFs:
```bash
python src/embed.py process pdf path/to/document.pdf --extract-tables
```

Process websites:
```bash
python src/embed.py process url https://example.com/sitemap.xml --sitemap-only
```

Process code repositories:
```bash
python src/embed.py process general https://github.com/user/repo.git
```

#### Document Search

Search processed documents:
```bash
python src/embed.py search "What is transformer architecture?" --mode hybrid --limit 5
```

Available search modes:
- `naive`: Basic keyword search
- `local`: Context-aware local search
- `global`: High-level conceptual search
- `hybrid`: Combined local and global search
- `mix`: Knowledge graph + vector retrieval

### Web Interface

Launch the Streamlit chat interface:
```bash
streamlit run src/chat.py
```

Features:
- Interactive Q&A with your documents
- Visual knowledge graph exploration
- Real-time entity and relationship display
- Multiple search mode selection

### Document Extraction

Extract content from various sources:
```bash
# Extract PDF to markdown
python src/extract.py pdf document.pdf --format markdown

# Extract website content
python src/extract.py html https://example.com --format json

# Process entire sitemap
python src/extract.py sitemap https://example.com/sitemap.xml
```

## Architecture

### Core Components

1. **Document Processors** (`src/embed.py`)
   - `PDFProcessor`: Handles PDF files with table extraction
   - `URLProcessor`: Processes web pages and sitemaps
   - `GitProcessor`: Clones and analyzes repositories
   - `TreeSitterChunker`: Language-aware code chunking

2. **Storage Manager** (`LightRAGStorageManager`)
   - Manages document chunking strategies
   - Handles embedding generation
   - Stores in LightRAG knowledge base

3. **Search Engine** (`DocumentSearcher`)
   - Implements multiple search strategies
   - Integrates with LightRAG query system

4. **Chat Interface** (`src/chat.py`)
   - Streamlit-based UI
   - Real-time knowledge graph visualization
   - Multi-tab result display

### Data Flow

```
Input Documents → Processors → Chunking → Embeddings → LightRAG Storage
                                                              ↓
User Query → Search Engine → LightRAG Query → Results → Chat Interface
```

## Configuration

### Processing Options

Configure in `ProcessingOptions` class:
- `extract_tables`: Enable table extraction from PDFs
- `split_vertical`: Split PDFs vertically for two-column layouts
- `embedding_dim`: Embedding dimension (default: 1536)
- `max_token_size`: Maximum tokens per chunk (default: 8192)

### Database Paths

- `db_path`: LightRAG database location (default: `data/default_db`)
- `output_dir`: Processed file output (default: `output/`)

## Neo4j Integration

Convert and import knowledge graphs to Neo4j:
```bash
python src/graph.py --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password your_password
```

## Development

### Project Structure
```
proj_llm/
├── src/
│   ├── chat.py          # Streamlit chat interface
│   ├── embed.py         # Document processing and storage
│   ├── extract.py       # Content extraction utilities
│   ├── graph.py         # Neo4j integration
│   └── utils/           # Helper utilities
├── data/                # Default database location
├── output/              # Processed files output
├── raw_data/            # Sample documents
└── requirements.txt     # Python dependencies
```

### Adding New Processors

1. Inherit from `FileProcessor` base class
2. Implement required methods:
   - `can_process()`: File type detection
   - `process()`: Processing logic
   - `get_source_type()`: Source identifier

3. Register with `ProcessingEngine`

## Future Enhancements

Based on TODO.md:
- [ ] Async processing for better performance
- [ ] Enhanced code parsing with LSP integration
- [ ] Alternative table extraction methods
- [ ] Multi-file semantic analysis for large projects
- [ ] Web crawler improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on [LightRAG](https://github.com/HKUDS/LightRAG) framework
- Uses [Tree-sitter](https://tree-sitter.github.io/) for code parsing
- Powered by OpenAI embeddings and GPT models