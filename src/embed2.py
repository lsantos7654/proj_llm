from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from enum import Enum, auto
from pathlib import Path
from numpy import result_type
import requests
from xml.etree import ElementTree
import re
import json
from utils.pdf_spliter import split_pdf_vertically
from utils.segment_tables import process_image, process_pdf
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from utils.tokenizer import OpenAITokenizerWrapper
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
from lancedb.embeddings import get_registry
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import sys


load_dotenv()

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Enum for different processing modes"""

    APPEND = auto()
    OVERWRITE = auto()


@dataclass
class ProcessingOptions:
    """Configuration options for file processing"""

    extract_tables: bool = False
    split_vertical: bool = False
    sitemap_only: bool = False
    preview: bool = False
    mode: ProcessingMode = ProcessingMode.APPEND
    db_path: Path = Path("data/lancedb")
    table_name: str = "docling"
    output_dir: Path = Path("output")


# Get the OpenAI embedding function
embedding_func = get_registry().get("openai").create(name="text-embedding-3-large")


class ChunkMetadata(LanceModel):
    """Schema for chunk metadata"""

    chunk_number: Optional[int]
    source_id: Optional[str]
    source_type: Optional[str]
    summary: Optional[str]
    title: Optional[str]


class Chunks(LanceModel):
    """Schema for document chunks with embeddings"""

    text: str = embedding_func.SourceField()
    vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()
    metadata: ChunkMetadata


class ProcessingResult:
    """Class to hold processing results and metadata"""

    def __init__(
        self,
        processed_files: List[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.processed_files = processed_files
        self.metadata = metadata or {}
        self.success = bool(processed_files)
        self.error = None

    @classmethod
    def error(cls, error_message: str) -> "ProcessingResult":
        result = cls([])
        result.success = False
        result.error = error_message
        return result


class FileProcessor(ABC):
    """Base class for file type processors"""

    def __init__(self):
        self.options: Optional[ProcessingOptions] = None

    @abstractmethod
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the given file"""
        pass

    @abstractmethod
    def process(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process the file and return ProcessingResult"""
        pass

    def set_options(self, options: ProcessingOptions) -> None:
        """Set processing options"""
        self.options = options


class LanceDBStorage:
    """Handles storage and retrieval from LanceDB"""

    def __init__(self, db_path: Path):
        self.db = lancedb.connect(str(db_path))
        self.client = OpenAI()
        self._table = None  # Add a cached table reference

    def create_or_get_table(self, table_name: str, mode: ProcessingMode) -> Table:
        """Create a new table or get existing one based on mode"""
        try:
            logger.debug(
                f"Entering create_or_get_table with table_name={table_name}, mode={mode}"
            )
            logger.debug(f"Current database connection: {self.db}")
            logger.info(f"Attempting to create/get table: {table_name}")

            exists = table_name in self.db.table_names()
            logger.debug(f"Table '{table_name}' exists: {exists}")

            if mode == ProcessingMode.APPEND and exists:
                logger.info(f"Opening existing table: {table_name}")
                try:
                    self._table = self.db.open_table(table_name)
                    logger.debug(f"Successfully opened table: {self._table}")
                except Exception as e:
                    logger.error(f"Failed to open existing table: {str(e)}")
                    raise
            else:
                logger.info(f"Creating new table: {table_name}")
                try:
                    self._table = self.db.create_table(
                        name=table_name,
                        schema=Chunks,
                        mode=(
                            "overwrite"
                            if mode == ProcessingMode.OVERWRITE
                            else "create"
                        ),
                    )
                    logger.debug(f"Successfully created table: {self._table}")
                except Exception as e:
                    logger.error(f"Failed to create table: {str(e)}")
                    raise

            return self._table

        except Exception as e:
            logger.error(
                f"Detailed error creating/getting table: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Failed to create/get table: {str(e)}")

    def _generate_title_and_summary(self, text: str, source_id: str) -> Dict[str, str]:
        """Generate title and summary for chunk using OpenAI."""
        logger.debug(
            f"Starting title and summary generation for source_id: {source_id}"
        )
        logger.debug(f"Input text length: {len(text)}")

        system_prompt = """You are an AI that extracts titles and summaries from document chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: Create a concise, descriptive title for this chunk.
        For the summary: Create a brief summary of the main points in this chunk."""

        logger.debug("Using system prompt for OpenAI completion")

        try:
            logger.debug("Making API call to OpenAI")
            logger.debug(
                f"Using first {min(len(text), 1000)} characters of text for context"
            )

            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Source: {source_id}\n\nContent:\n{text[:1000]}...",
                    },
                ],
                response_format={"type": "json_object"},
            )

            logger.debug("Successfully received response from OpenAI")
            result = json.loads(response.choices[0].message.content)
            logger.debug(f"Parsed JSON response: {result}")

            # Log the length of generated title and summary
            logger.debug(f"Generated title length: {len(result.get('title', ''))}")
            logger.debug(f"Generated summary length: {len(result.get('summary', ''))}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            logger.debug(f"Raw response content: {response.choices[0].message.content}")
            return {
                "title": "Error processing title",
                "summary": "Error processing summary",
            }
        except Exception as e:
            logger.error(f"Error generating title and summary: {e}")
            logger.debug(f"Exception type: {type(e).__name__}")
            logger.debug(f"Exception details: {str(e)}")
            return {
                "title": "Error processing title",
                "summary": "Error processing summary",
            }

    def process_chunks(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """Process document chunks into storable format with enhanced metadata"""
        processed_chunks: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            try:
                # Extract metadata from chunk
                filename = (
                    self._extract_filename(chunk) or f"chunk_{i}"
                )  # Ensure non-null
                content = str(chunk.text)

                # Determine source type based on filename
                source_type = "pdf" if filename.endswith(".pdf") else "web"

                generated_data = self._generate_title_and_summary(content, filename)

                chunk_dict = {
                    "text": content,
                    "metadata": {
                        "source_type": source_type,  # Always non-null
                        "source_id": filename,  # Always non-null
                        "chunk_number": i,  # Always non-null
                        "title": generated_data.get("title"),  # Can be null
                        "summary": generated_data.get("summary"),  # Can be null
                    },
                }
                processed_chunks.append(chunk_dict)
                logger.info(f"Processed chunk {i} from {filename}")

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue

        if not processed_chunks:
            raise ValueError("No chunks were successfully processed")

        return processed_chunks

    def _extract_filename(self, chunk: Any) -> Optional[str]:
        """Extract filename from chunk metadata"""
        try:
            return chunk.meta.origin.filename
        except AttributeError:
            return None


class StorageHandler:
    """Coordinates storage of processed files"""

    def __init__(self, options: ProcessingOptions):
        self.options = options
        self.storage = LanceDBStorage(options.db_path)
        self.table = None
        self.initialize_storage()

    def initialize_storage(self) -> None:
        """Initialize storage and create/get table"""
        try:
            logger.debug("Starting storage initialization")
            self.table = self.storage.create_or_get_table(
                self.options.table_name, self.options.mode
            )
            # Verify table was created/opened successfully
            if (
                not hasattr(self.table, "name")
                or self.table.name != self.options.table_name
            ):
                logger.error(f"Table validation failed - table object: {self.table}")
                raise RuntimeError("Invalid table object returned")

            logger.debug(
                f"Storage initialized successfully with table: {self.table.name}"
            )

        except Exception as e:
            logger.error(f"Storage initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize storage: {str(e)}")

    def store_results(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Store processing results in LanceDB"""

        successful_stores = 0
        failed_stores = 0
        total_chunks = 0

        for result in results:
            if not result.success:
                failed_stores += 1
                continue

            try:
                # Process and store each file
                for file_path in result.processed_files:
                    logger.debug(f"Processing file: {file_path}")
                    chunks = self.storage.process_chunks(
                        self._get_chunks_for_file(file_path)
                    )
                    logger.debug(f"Generated {len(chunks)} chunks for file")
                    self.table.add(chunks)
                    total_chunks += len(chunks)
                    successful_stores += 1
                    logger.debug(f"Successfully stored chunks for {file_path}")

            except Exception as e:
                logger.error(f"Error storing result: {str(e)}")
                failed_stores += 1

        return {
            "successful_stores": successful_stores,
            "failed_stores": failed_stores,
            "total_chunks": total_chunks,
        }

    def _get_chunks_for_file(self, file_path: Path) -> List[Any]:
        converter = DocumentConverter()
        chunker = HybridChunker(
            tokenizer=OpenAITokenizerWrapper(),
            max_tokens=8191,
            merge_peers=True,
        )

        try:
            result = converter.convert(file_path)
            chunks = list(chunker.chunk(dl_doc=result.document))
            return chunks
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {str(e)}")
            return []

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage state"""
        if not self.table:
            raise RuntimeError("Storage not initialized")

        return {
            "table_name": self.options.table_name,
            "row_count": self.table.count_rows(),
            "schema": str(Chunks.schema()),
        }


class ProcessingEngine:
    """Main engine for coordinating file processing"""

    def __init__(self):
        self.processors: List[FileProcessor] = []
        self.options: Optional[ProcessingOptions] = None

    def register_processor(self, processor: FileProcessor) -> None:
        """Register a new file processor"""
        self.processors.append(processor)

    def set_options(self, options: ProcessingOptions) -> None:
        """Set processing options and propagate to all processors"""
        self.options = options
        for processor in self.processors:
            processor.set_options(options)

    def get_processor(self, file_path: Union[str, Path]) -> Optional[FileProcessor]:
        """Get appropriate processor for a file"""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None

    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a single file"""
        if not self.options:
            return ProcessingResult.error("No processing options set")

        processor = self.get_processor(file_path)
        if not processor:
            return ProcessingResult.error(f"No processor found for {file_path}")

        try:
            return processor.process(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return ProcessingResult.error(str(e))

    def process_input(self, input_path: Union[str, Path]) -> List[ProcessingResult]:
        """Process input path (file, directory, or URL)"""
        if not self.options:
            return [ProcessingResult.error("No processing options set")]

        results: List[ProcessingResult] = []

        try:
            # Handle URLs separately
            if isinstance(input_path, str) and (
                input_path.startswith("http://") or input_path.startswith("https://")
            ):
                results.append(self.process_file(input_path))
            else:
                # Convert to Path object for file system paths
                input_path = (
                    Path(input_path) if isinstance(input_path, str) else input_path
                )

                if input_path.is_file():
                    results.append(self.process_file(input_path))
                elif input_path.is_dir():
                    for file_path in input_path.rglob("*"):
                        if file_path.is_file():
                            results.append(self.process_file(file_path))
                else:
                    results.append(
                        ProcessingResult.error(f"Invalid input path: {input_path}")
                    )
        except Exception as e:
            results.append(ProcessingResult.error(str(e)))

        return results


class PDFProcessor(FileProcessor):
    """Processor for PDF files"""

    def __init__(self):
        super().__init__()
        self.converter = DocumentConverter()
        self.tokenizer = OpenAITokenizerWrapper()
        self.MAX_TOKENS: int = 8191

    def can_process(self, file_path: Union[str, Path]) -> bool:
        if isinstance(file_path, str):
            return False
        result = file_path.suffix.lower() == ".pdf"
        if result:
            logger.info(f"Processing pdf: {file_path}")
        return result

    def setup_chunker(self) -> HybridChunker:
        return HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.MAX_TOKENS,
            merge_peers=True,
        )

    def process(self, file_path: Path) -> ProcessingResult:
        if not self.options:
            return ProcessingResult.error("No processing options set")

        processed_files: List[Path] = []
        metadata: Dict[str, Any] = {}

        try:
            # Handle PDF splitting if enabled
            if self.options.split_vertical:
                left_path, right_path = split_pdf_vertically(
                    file_path, self.options.output_dir
                )
                processed_files.extend([left_path, right_path])
                metadata["split_pdfs"] = [str(left_path), str(right_path)]

                # Process split PDFs for tables if enabled
                if self.options.extract_tables:
                    tables: List[Path] = []
                    for split_pdf in [left_path, right_path]:
                        pdf_tables = process_pdf(split_pdf, self.options.output_dir)
                        tables.extend(pdf_tables)
                        processed_files.extend(pdf_tables)
                    metadata["split_pdf_tables"] = [str(t) for t in tables]
            else:
                processed_files.append(file_path)

                # Process original PDF for tables if enabled
                if self.options.extract_tables:
                    tables = process_pdf(file_path, self.options.output_dir)
                    processed_files.extend(tables)
                    metadata["tables"] = [str(t) for t in tables]

            # Preview if enabled
            if self.options.preview:
                preview = self.preview_document(file_path)
                metadata["preview"] = preview

            return ProcessingResult(processed_files, metadata)

        except Exception as e:
            return ProcessingResult.error(f"Error processing PDF {file_path}: {str(e)}")

    def preview_document(self, file_path: Path) -> str:
        try:
            result = self.converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            raise Exception(f"Error previewing PDF: {e}")


class URLProcessor(FileProcessor):
    """Processor for URLs and sitemaps"""

    def __init__(self):
        super().__init__()
        self.converter = DocumentConverter()

    def _get_title_and_summary(self, chunk: str) -> Dict[str, str]:
        """Generate title and summary for a chunk using GPT-4"""
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        Title: Create a concise, descriptive title for this content (max 10 words)
        Summary: Create a brief summary of the main points (max 50 words)"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": chunk[:1000],
                    },  # First 1000 chars for context
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error getting title and summary: {e}")
            return {
                "title": "Error processing title",
                "summary": "Error processing summary",
            }

    def can_process(self, file_path: Union[str, Path]) -> bool:
        if isinstance(file_path, Path):
            return False
        result = file_path.startswith(("http://", "https://"))
        if result:
            logger.info(f"Processing URL: {file_path}")
        return result

    def process(self, file_path: Union[str, Path]) -> ProcessingResult:
        if not self.options:
            return ProcessingResult.error("No processing options set")

        url = str(file_path)
        try:
            if url.endswith("sitemap.xml") and self.options.sitemap_only:
                urls = self.process_sitemap(url)
                return ProcessingResult([u for u in urls], {"sitemap_urls": urls})

            # For regular URLs, process the content
            if self.options.preview:
                preview = self.preview_url(url)
                return ProcessingResult([url], {"preview": preview})

            return ProcessingResult([url])

        except Exception as e:
            return ProcessingResult.error(f"Error processing URL {url}: {str(e)}")

    def process_sitemap(self, url: str) -> List[str]:
        try:
            logger.info(f"Processing sitemap: {url}")
            response = requests.get(url)
            response.raise_for_status()

            root = ElementTree.fromstring(response.content)
            namespaces = {
                "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
                "xhtml": "http://www.w3.org/1999/xhtml",
            }

            urls: List[str] = []

            # Handle sitemap index
            sitemaps = root.findall(".//sm:sitemap/sm:loc", namespaces)
            if sitemaps:
                logger.info(f"Found sitemap index with {len(sitemaps)} sitemaps")
                for sitemap in sitemaps:
                    if sitemap.text:
                        logger.info(f"Processing sub-sitemap: {sitemap.text}")
                        urls.extend(self.process_sitemap(sitemap.text))
            else:
                # Handle regular sitemap
                locations = root.findall(".//sm:url/sm:loc", namespaces)
                valid_locations = [loc.text for loc in locations if loc.text]
                logger.info(f"Found {len(valid_locations)} URLs in sitemap")
                for url in valid_locations:
                    logger.info(f"Found URL: {url}")
                urls.extend(valid_locations)

            return urls

        except Exception as e:
            logger.error(f"Error processing sitemap {url}: {e}")
            return []

    def preview_url(self, url: str) -> str:
        try:
            result = self.converter.convert(url)
            return result.document.export_to_markdown()
        except Exception as e:
            raise Exception(f"Error previewing URL: {e}")


class GitProcessor(FileProcessor):
    """Processor for Git repositories"""

    def can_process(self, file_path: Union[str, Path]) -> bool:
        path_str = str(file_path)
        return (
            path_str.endswith(".git")
            or "github.com" in path_str
            or "gitlab.com" in path_str
        )

    def validate_git_url(self, url: str) -> bool:
        git_patterns = [
            r"https?://github\.com/[\w-]+/[\w-]+(?:\.git)?/?$",
            r"git@github\.com:[\w-]+/[\w-]+(?:\.git)?/?$",
        ]
        return any(re.match(pattern, url) for pattern in git_patterns)

    def process(self, file_path: Union[str, Path]) -> ProcessingResult:
        if not self.options:
            return ProcessingResult.error("No processing options set")

        repo_url = str(file_path)
        if not self.validate_git_url(repo_url):
            return ProcessingResult.error(f"Invalid Git repository URL: {repo_url}")

        try:
            from git import Repo, GitCommandError

            # Clone repository
            repo = Repo.clone_from(repo_url, self.options.output_dir)
            if not repo.git_dir:
                return ProcessingResult.error("Repository appears empty")

            # Get all files in the repository
            repo_files = list(Path(self.options.output_dir).rglob("*"))
            repo_files = [f for f in repo_files if f.is_file()]

            return ProcessingResult(
                repo_files, {"repo_url": repo_url, "file_count": len(repo_files)}
            )

        except Exception as e:
            return ProcessingResult.error(f"Error processing Git repository: {str(e)}")


class DocumentProcessor:
    """Main class coordinating document processing workflow"""

    def __init__(self):
        self.engine = ProcessingEngine()
        self.storage_handler: Optional[StorageHandler] = None

        # Register processors
        self.engine.register_processor(PDFProcessor())
        self.engine.register_processor(URLProcessor())
        self.engine.register_processor(GitProcessor())

    def initialize(self, options: ProcessingOptions) -> None:
        """Initialize processor with options"""
        try:
            self.engine.set_options(options)
            self.storage_handler = StorageHandler(options)
            logger.debug("Document processor fully initialized")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize processor: {str(e)}")

    def process(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """Process input and return results"""
        if not self.storage_handler:
            logger.error("Attempted to process before initialization")
            raise RuntimeError("Processor not initialized")

        logger.info(f"Processing input: {input_path}")

        # Process files
        results = self.engine.process_input(input_path)
        logger.debug(f"Got {len(results)} processing results")

        # Summarize processing results
        summary = self._summarize_processing(results)

        # Store results if not in preview mode
        if not self.engine.options.preview:
            logger.debug("Storing results in database")
            storage_results = self.storage_handler.store_results(results)
            summary.update(storage_results)
            logger.debug(f"Updated summary with storage results: {storage_results}")

        return summary

    def _summarize_processing(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Create summary of processing results"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return {
            "total_files": len(results),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "errors": [r.error for r in failed if r.error],
        }


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Document Processing and Storage System"
    )

    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")

    # Common arguments for all modes
    common_args = {
        "-o": {
            "dest": "output_dir",
            "default": "output",
            "help": "Output directory for processed files",
        },
        "-db": {
            "dest": "db_path",
            "default": "data/lancedb",
            "help": "Path to LanceDB database",
        },
        "-t": {
            "dest": "table_name",
            "default": "docling",
            "help": "Table name in database",
        },
        "--preview": {
            "action": "store_true",
            "help": "Preview processing without storing",
        },
        "--mode": {
            "choices": ["append", "overwrite"],
            "default": "append",
            "help": "Database operation mode",
        },
    }

    # PDF mode
    pdf_parser = subparsers.add_parser("pdf", help="Process PDF documents")
    pdf_parser.add_argument("input", help="PDF file or directory")
    pdf_parser.add_argument("--extract-tables", action="store_true")
    pdf_parser.add_argument("--split-vertical", action="store_true")
    for arg, kwargs in common_args.items():
        pdf_parser.add_argument(arg, **kwargs)

    # URL mode
    url_parser = subparsers.add_parser("url", help="Process URLs and sitemaps")
    url_parser.add_argument("input", help="URL or sitemap URL")
    url_parser.add_argument("--sitemap-only", action="store_true")
    for arg, kwargs in common_args.items():
        url_parser.add_argument(arg, **kwargs)

    # General mode
    general_parser = subparsers.add_parser(
        "general", help="Process any supported files"
    )
    general_parser.add_argument("input", help="Input file or directory")
    for arg, kwargs in common_args.items():
        general_parser.add_argument(arg, **kwargs)

    return parser


def main() -> int:
    """Main entry point"""
    try:
        # Parse arguments
        parser = create_parser()
        args = parser.parse_args()

        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Create processing options
        options = ProcessingOptions(
            extract_tables=getattr(args, "extract_tables", False),
            split_vertical=getattr(args, "split_vertical", False),
            sitemap_only=getattr(args, "sitemap_only", False),
            preview=args.preview,
            mode=(
                ProcessingMode.OVERWRITE
                if args.mode == "overwrite"
                else ProcessingMode.APPEND
            ),
            db_path=Path(args.db_path),
            table_name=args.table_name,
            output_dir=Path(args.output_dir),
        )

        # Initialize processor
        processor = DocumentProcessor()
        processor.initialize(options)

        # Process input
        logger.info("Starting processing...")
        results = processor.process(args.input)

        # Print results
        print("\nProcessing Results:")
        print(f"Total files: {results['total_files']}")
        print(f"Successfully processed: {results['successful_files']}")
        print(f"Failed: {results['failed_files']}")

        if "total_chunks" in results:
            print(f"Total chunks stored: {results['total_chunks']}")

        if results["errors"]:
            print("\nErrors encountered:")
            for error in results["errors"]:
                print(f"- {error}")

        return 0 if results["failed_files"] == 0 else 1

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
