# Project Vision: Local AI Documentation Server

## Overview
Transform this project into a backend server that hosts a vector database for AI agents to access documentation locally.

## Architecture
```
Docker Compose Stack:
├── Vector Database (Qdrant/Chroma)
├── Embedding Service (sentence-transformers) 
├── MCP Server (FastAPI)
└── Document Processing Pipeline
```

## Workflow
1. **Start Server** - Docker compose brings up all services
2. **Extract Documentation** - Improved extraction pipeline gets quality docs
3. **Review & Process** - Clean and prepare documents for embedding
4. **Ingest & Embed** - Create vector embeddings and store in database
5. **AI Agent Access** - MCP server provides semantic search for AI agents

## Technical Components

### What to Remove
- Chat functionality
- Complex smart extraction logic (keep simple, working extraction)
- Current poor-quality markdown files

### What to Keep & Improve
- Basic extraction pipeline (but fix the core issues)
- Markdown cleaning functionality
- Makefile structure for operations

### What to Build
- **Document Ingestion API** - Process and embed documents
- **Vector Search Endpoints** - Semantic search for AI agents  
- **MCP Protocol Implementation** - Standard interface for AI agents
- **Docker Compose Setup** - Fully local deployment

## Current Problems to Solve
1. **Extraction Quality** - Current docs.rs extraction is broken
   - Getting metadata instead of actual documentation
   - Need better approach for rustdoc sites
   - Only 31 files with duplicate/poor content

2. **Content Issues** - Current output insufficient for AI agents
   - Missing API documentation, examples, guides
   - Need comprehensive documentation extraction

## Next Steps
1. **Fix Extraction Pipeline** - Get working, high-quality document extraction
2. **Build Vector Database Backend** - Implement embedding and search
3. **Create MCP Server** - Standard interface for AI agent access
4. **Docker Compose Setup** - Local deployment stack

## Success Criteria
- AI agents can get comprehensive, accurate documentation
- Everything runs locally without external dependencies
- High-quality semantic search across documentation
- Extensible to multiple documentation sources