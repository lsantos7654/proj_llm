"""Document processing and embedding system for storing and retrieving documents."""

import argparse
import logging
import re
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from xml.etree import ElementTree

import requests
from dotenv import load_dotenv
from git import Repo
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

from utils.pdf_spliter import split_pdf_vertically
from utils.segment_tables import extract_table_from_pdf
from utils.tokenizer import OpenAITokenizerWrapper

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingOptions:
    """Configuration options for file processing."""

    extract_tables: bool = False
    split_vertical: bool = False
    sitemap_only: bool = False
    db_path: Path = Path("data/testdb")
    table_name: str = "docling"
    output_dir: Path = Path("output")
    embedding_dim: int = 1536
    max_token_size: int = 8192


class ProcessingResult:
    """Class to hold processing results and metadata.

    This class encapsulates the results of processing one or more files, including
    success/failure status, processed file paths, associated metadata, and any error
    messages that occurred during processing.
    """

    def __init__(
        self,
        processed_files: List[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a new ProcessingResult instance.

        Args:
            processed_files: List of paths (str or Path) to successfully processed files
            metadata: Optional dictionary containing additional processing metadata
        """
        self.processed_files = processed_files
        self.metadata = metadata or {}
        self.success = bool(processed_files)
        self.error = None

    @classmethod
    def error(cls, error_message: str) -> "ProcessingResult":
        """Create a ProcessingResult instance representing an error.

        Args:
            error_message: Description of the error that occurred

        Returns:
            ProcessingResult: A new instance with success=False
            and the specified error message
        """
        result = cls([])
        result.success = False
        result.error = error_message
        return result


class FileProcessor(ABC):
    """Base class for file type processors."""

    def __init__(self):
        """Initialize processing options with default values."""
        self.options: Optional[ProcessingOptions] = None

    @abstractmethod
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Determine if a given file can be processed by this processor.

        Args:
            file_path (Union[str, Path]): Path to the file to be checked

        Returns:
            bool: True if the file can be processed, False otherwise
        """
        pass

    @abstractmethod
    def get_source_type(self) -> str:
        """Return the source type for this processor."""
        pass

    @abstractmethod
    def process(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process the file and return ProcessingResult."""
        pass

    def set_options(self, options: ProcessingOptions) -> None:
        """Set processing options."""
        self.options = options


class LightRAGStorageManager:
    """Manages document chunking and storage using LightRAG."""

    def __init__(self, options: ProcessingOptions):
        """Initialize storage manager with processing options."""
        self.options = options
        self.rag = LightRAG(
            working_dir=str(options.db_path),
            embedding_func=EmbeddingFunc(
                embedding_dim=options.embedding_dim,
                max_token_size=options.max_token_size,
                func=openai_embed,
            ),
            llm_model_func=gpt_4o_mini_complete,
            addon_params={"insert_batch_size": 20},
        )

    def process_chunks(self, chunks: List[Any]) -> List[str]:
        """Process document chunks into text content for LightRAG insertion."""
        processed_chunks = []

        for chunk in chunks:
            try:
                content = str(chunk.text)
                processed_chunks.append(content)
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        return processed_chunks

    def store_results(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Store processing results using LightRAG."""
        successful_stores = 0
        failed_stores = 0
        total_chunks = 0

        for result in results:
            if not result.success:
                failed_stores += 1
                continue

            try:
                for file_path in result.processed_files:
                    # Get chunks using HybridChunker
                    chunks = self._get_chunks_for_file(file_path)

                    # Process chunks to get text content
                    processed_chunks = self.process_chunks(chunks)

                    # Store chunks using LightRAG's batch insert
                    try:
                        self.rag.insert(processed_chunks)
                        total_chunks += len(processed_chunks)
                        successful_stores += 1
                    except Exception as e:
                        logger.error(f"Error storing chunks: {str(e)}")
                        failed_stores += 1
                        continue

            except Exception as e:
                logger.error(f"Error storing result: {str(e)}")
                failed_stores += 1

        return {
            "successful_stores": successful_stores,
            "failed_stores": failed_stores,
            "total_chunks": total_chunks,
        }

    def _get_chunks_for_file(self, file_path: Union[str, Path]) -> List[Any]:
        """Generate document chunks from a file using HybridChunker."""
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


class ProcessingEngine:
    """Main engine for coordinating file processing.

    This class serves as the central coordinator for processing various types of files
    using registered file processors. It manages the processing workflow, handles
    processor registration, and coordinates the processing of individual files or
    directories.

    Attributes:
        processors (List[FileProcessor]): List of registered file processors
        options (Optional[ProcessingOptions]): Configuration options for processing
    """

    def __init__(self):
        """Initialize a new ProcessingEngine instance.

        Creates an empty list of processors and initializes options as None.
        Options must be set via set_options() before processing can begin.
        """
        self.processors: List[FileProcessor] = []
        self.options: Optional[ProcessingOptions] = None

    def register_processor(self, processor: FileProcessor) -> None:
        """Register a new file processor with the engine.

        Args:
            processor (FileProcessor): The file processor instance to register
        """
        self.processors.append(processor)

    def set_options(self, options: ProcessingOptions) -> None:
        """Set processing options and propagate them to all registered processors.

        Args:
            options (ProcessingOptions): The processing options to set
        """
        self.options = options
        for processor in self.processors:
            processor.set_options(options)

    def get_processor(self, file_path: Union[str, Path]) -> Optional[FileProcessor]:
        """Get the appropriate processor for a given file.

        Iterates through registered processors to find one that can handle the
        specified file type.

        Args:
            file_path (Union[str, Path]): Path to the file needing processing

        Returns:
            Optional[FileProcessor]: The first processor that can handle the file,
                                   or None if no suitable processor is found
        """
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None

    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a single file using the appropriate processor.

        Args:
            file_path (Union[str, Path]): Path to the file to process

        Returns:
            ProcessingResult: Result of the processing operation, including success/failure
                            status and any error messages

        Note:
            Returns an error result if no processing options are set or if no suitable
            processor is found for the file type.
        """
        processor = self.get_processor(file_path)
        if not processor:
            return ProcessingResult.error(f"No processor found for {file_path}")

        try:
            return processor.process(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return ProcessingResult.error(str(e))

    def process_input(self, input_path: Union[str, Path]) -> List[ProcessingResult]:
        """Process an input path which can be a file, directory, or URL.

        Args:
            input_path (Union[str, Path]): Path to process, can be:
                - A URL (string starting with http:// or https://)
                - A file path
                - A directory path (will process all files recursively)

        Returns:
            List[ProcessingResult]: List of processing results for all processed files

        Note:
            - For directories, processes all files recursively
            - Returns a list containing a single error result if no processing options
              are set
            - Handles URLs separately from filesystem paths
        """
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
    """Processor for handling PDF files.

    This class provides functionality to process PDF files,
    including splitting PDFs vertically,
    extracting tables, and converting content to markdown format.
    It inherits from FileProcessor and implements specific
    PDF processing capabilities.

    Attributes:
        converter (DocumentConverter): Converter for transforming
        PDFs to processable documents
        tokenizer (OpenAITokenizerWrapper): Tokenizer for text processing
        MAX_TOKENS (int): Maximum number of tokens allowed per chunk
    """

    def __init__(self):
        """Initialize the PDF processor with document converter and tokenizer.

        Sets up required components for PDF processing including:
        - Document converter for PDF transformation
        - OpenAI tokenizer for text processing
        - Maximum token limit (8191)
        """
        super().__init__()

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if the file can be processed by this PDF processor.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if file is a PDF and can be processed, False otherwise
        """
        if isinstance(file_path, str):
            return False
        result = file_path.suffix.lower() == ".pdf"
        if result:
            logger.info(f"Processing pdf: {file_path}")
        return result

    def get_source_type(self) -> str:
        """Return the source type identifier for this processor.

        Returns:
            str: The string 'pdf' indicating this processes PDF files
        """
        return "pdf"

    def process(self, file_path: Path) -> ProcessingResult:
        """Process a PDF file according to configured options.

        This method handles the main PDF processing workflow, including:
        - Vertical splitting of PDFs if enabled
        - Table extraction if enabled

        Args:
            file_path (Path): Path to the PDF file to process

        Returns:
            ProcessingResult: Object containing processing results and metadata

        Raises:
            ProcessingResult.error: If processing fails or no options are set

        Note:
            The processing behavior is controlled by the options set in self.options:
            - split_vertical: Splits PDF into left and right pages
            - extract_tables: Extracts tables from the PDF
        """
        if not self.options:
            return ProcessingResult.error("No processing options set for pdf")

        processed_files: List[Path] = []
        metadata: Dict[str, Any] = {}
        metadata["file_type"] = "pdf"

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
                        pdf_tables = extract_table_from_pdf(
                            split_pdf, self.options.output_dir
                        )
                        tables.extend(pdf_tables)
                        processed_files.extend(pdf_tables)
                    metadata["split_pdf_tables"] = [str(t) for t in tables]
            else:
                processed_files.append(file_path)

                # Process original PDF for tables if enabled
                if self.options.extract_tables:
                    tables = extract_table_from_pdf(file_path, self.options.output_dir)
                    processed_files.extend(tables)
                    metadata["tables"] = [str(t) for t in tables]

            return ProcessingResult(processed_files, metadata)

        except Exception as e:
            return ProcessingResult.error(f"Error processing PDF {file_path}: {str(e)}")


class URLProcessor(FileProcessor):
    """Processor for URLs and sitemaps."""

    def __init__(self):
        """Initialize URL and sitemap processor."""
        super().__init__()
        self.converter = DocumentConverter()

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the given URL.

        Args:
            file_path: URL to process

        Returns:
            bool: True if URL can be processed, False otherwise
        """
        if isinstance(file_path, Path):
            return False
        result = file_path.startswith(("http://", "https://"))
        if result:
            logger.info(f"Processing URL: {file_path}")
        return result

    def get_source_type(self) -> str:
        """Return the source type identifier for this processor.

        Returns:
            str: The string 'url' indicating this processes PDF files
        """
        return "url"

    def process(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a URL or sitemap.

        Args:
            file_path: URL to process

        Returns:
            ProcessingResult: Results of URL processing
        """
        url = str(file_path)
        try:
            if url.endswith("sitemap.xml") and self.options.sitemap_only:
                urls = self.process_sitemap(url)
                return ProcessingResult([u for u in urls], {"sitemap_urls": urls})

            return ProcessingResult([url])

        except Exception as e:
            return ProcessingResult.error(f"Error processing URL {url}: {str(e)}")

    def process_sitemap(self, url: str) -> List[str]:
        """Process a sitemap XML file and extract all URLs.

        This method handles both regular sitemaps and sitemap indexes.
        For sitemap indexes, it recursively processes
        all referenced sitemaps.

        Args:
            url (str): The URL of the sitemap to process

        Returns:
            List[str]: A list of all URLs found in the sitemap and its sub-sitemaps

        Note:
            - Supports both standard sitemaps and sitemap index files
            - Uses XML namespaces for proper parsing
            - Handles nested sitemaps recursively
        """
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


class GitProcessor(FileProcessor):
    """Processor for Git repositories."""

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the given Git repository.

        Args:
            file_path: Path or URL to Git repository

        Returns:
            bool: True if repository can be processed, False otherwise
        """
        path_str = str(file_path)
        return (
            path_str.endswith(".git")
            or "github.com" in path_str
            or "gitlab.com" in path_str
        )

    def get_source_type(self) -> str:
        """Return the source type identifier for this processor.

        Returns:
            str: The string 'git' indicating this processes PDF files
        """
        return "git"

    def validate_git_url(self, url: str) -> bool:
        """Validate if a URL points to a valid Git repository.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL is valid Git repository, False otherwise
        """
        git_patterns = [
            r"https?://github\.com/[\w-]+/[\w-]+(?:\.git)?/?$",
            r"git@github\.com:[\w-]+/[\w-]+(?:\.git)?/?$",
        ]
        return any(re.match(pattern, url) for pattern in git_patterns)

    def process(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a Git repository.

        Args:
            file_path: Path or URL to Git repository

        Returns:
            ProcessingResult: Results of repository processing
        """
        if not self.options:
            return ProcessingResult.error("No processing options set")

        repo_url = str(file_path)
        if not self.validate_git_url(repo_url):
            return ProcessingResult.error(f"Invalid Git repository URL: {repo_url}")

        try:
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
    def __init__(self):
        self.engine = ProcessingEngine()
        self.storage_manager: Optional[LightRAGStorageManager] = None

        # Register processors
        self.engine.register_processor(PDFProcessor())
        self.engine.register_processor(URLProcessor())
        self.engine.register_processor(GitProcessor())

    def initialize(self, options: ProcessingOptions) -> None:
        try:
            self.engine.set_options(options)
            self.storage_manager = LightRAGStorageManager(options)
            logger.debug("Document processor fully initialized")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize processor: {str(e)}")

    def process(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        if not self.storage_manager:
            logger.error("Attempted to process before initialization")
            raise RuntimeError("Processor not initialized")

        logger.info(f"Processing input: {input_path}")

        # Process input files based off of option set + file_type
        results = self.engine.process_input(input_path)
        logger.debug(f"Got {len(results)} processing results")

        # Summarize processing results
        summary = self._summarize_processing(results)

        logger.debug("Chunking/Embedding/Storing results in database")
        storage_results = self.storage_manager.store_results(results)
        summary.update(storage_results)
        logger.debug(f"Updated summary with storage results: {storage_results}")

        return summary

    def _summarize_processing(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return {
            "total_files": len(results),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "errors": [r.error for r in failed if r.error],
        }


class DocumentSearcher:
    """Handles document search operations using LightRAG."""

    def __init__(self, rag: LightRAG):
        """Initialize searcher with LightRAG instance.

        Args:
            rag: Initialized LightRAG instance
        """
        self.rag = rag

    def search(self, query: str, mode: str = "hybrid", limit: int = 10) -> Any:
        """Search documents using specified LightRAG search mode.

        Args:
            query: The search query
            mode: Search mode to use. One of:
                - "naive": Basic search
                - "local": Context-dependent search
                - "global": Global knowledge search
                - "hybrid": Combined local and global
                - "mix": Knowledge graph + vector retrieval
            limit: Maximum number of results to return

        Returns:
            Search results from LightRAG

        Raises:
            ValueError: If invalid search mode specified
        """
        valid_modes = ["naive", "local", "global", "hybrid", "mix"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid search mode. Must be one of: {valid_modes}")

        try:
            logger.debug(f"Searching with mode '{mode}', query: '{query}'")

            results = self.rag.query(query, param=QueryParam(mode=mode, top_k=limit))

            logger.debug(f"Search completed, got results")
            return results

        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return None

    def get_available_modes(self) -> List[str]:
        """Get list of available search modes.

        Returns:
            List of valid search mode strings
        """
        return ["naive", "local", "global", "hybrid", "mix"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Document Processing and Search System"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search processed documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--mode",
        choices=["naive", "local", "global", "hybrid", "mix"],
        default="hybrid",
        help="Search mode",
    )
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results"
    )

    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_subparsers = process_parser.add_subparsers(
        dest="type", help="Type of content to process"
    )

    # Common arguments for processing modes
    common_args = {
        "-o": {
            "dest": "output_dir",
            "default": "output",
            "help": "Output directory for processed files",
        },
        "-db": {
            "dest": "db_path",
            "default": "lightrag_cache",
            "help": "Path to LightRAG database",
        },
    }

    # PDF mode
    pdf_parser = process_subparsers.add_parser("pdf", help="Process PDF documents")
    pdf_parser.add_argument("input", help="PDF file or directory")
    pdf_parser.add_argument("--extract-tables", action="store_true")
    pdf_parser.add_argument("--split-vertical", action="store_true")
    for arg, kwargs in common_args.items():
        pdf_parser.add_argument(arg, **kwargs)

    # URL mode
    url_parser = process_subparsers.add_parser("url", help="Process URLs and sitemaps")
    url_parser.add_argument("input", help="URL or sitemap URL")
    url_parser.add_argument("--sitemap-only", action="store_true")
    for arg, kwargs in common_args.items():
        url_parser.add_argument(arg, **kwargs)

    # General mode
    general_parser = process_subparsers.add_parser(
        "general", help="Process any supported files"
    )
    general_parser.add_argument("input", help="Input file or directory")
    for arg, kwargs in common_args.items():
        general_parser.add_argument(arg, **kwargs)

    return parser


def main() -> int:
    try:
        # Parse arguments
        parser = create_parser()
        args = parser.parse_args()

        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Handle search command
        if args.command == "search":
            # Initialize LightRAG
            rag = LightRAG(
                working_dir=str(args.db_path),
                embedding_func=openai_embed,
                llm_model_func=gpt_4o_mini_complete,
            )
            searcher = DocumentSearcher(rag)

            # Perform search
            results = searcher.search(
                query=args.query, mode=args.mode, limit=args.limit
            )

            # Print results
            if results:
                print(f"\nSearch Results for: {args.query}")
                print("=" * 80)
                print(results)
                print("=" * 80)
            else:
                print("No results found")

            return 0

        # Handle processing commands
        elif args.command == "process":
            options = ProcessingOptions(
                extract_tables=getattr(args, "extract_tables", False),
                split_vertical=getattr(args, "split_vertical", False),
                sitemap_only=getattr(args, "sitemap_only", False),
                db_path=Path(args.db_path),
                output_dir=Path(args.output_dir),
            )

            processor = DocumentProcessor()
            processor.initialize(options)

            logger.info("Starting processing...")
            results = processor.process(args.input)

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

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
