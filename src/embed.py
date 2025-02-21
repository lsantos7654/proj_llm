"""Document processing and embedding system for storing and retrieving documents."""

import argparse
import json
import logging
import re
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from xml.etree import ElementTree

import requests
import weaviate
from dotenv import load_dotenv
from git import Repo
from openai import OpenAI
from weaviate.util import generate_uuid5

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


class ProcessingMode(Enum):
    """Enum for different processing modes."""

    APPEND = auto()
    OVERWRITE = auto()


@dataclass
class ProcessingOptions:
    """Configuration options for file processing."""

    extract_tables: bool = False
    split_vertical: bool = False
    sitemap_only: bool = False
    mode: ProcessingMode = ProcessingMode.APPEND
    db_path: Path = Path("data/lancedb")
    table_name: str = "docling"
    output_dir: Path = Path("output")


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


class WeaviateSchema:
    def __init__(self, client):
        self.client = client

    def create_schema(self):
        class_obj = {
            "class": "DocumentChunk",
            "vectorizer": "text2vec-openai",
            "vectorIndexConfig": {"distance": "cosine"},
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "vectorizer": "text2vec-openai",
                },
                {
                    "name": "title",
                    "dataType": ["text"],
                    "vectorizer": "text2vec-openai",
                },
                {
                    "name": "summary",
                    "dataType": ["text"],
                    "vectorizer": "text2vec-openai",
                },
                {
                    "name": "sourceId",
                    "dataType": ["string"],
                },
                {
                    "name": "sourceType",
                    "dataType": ["string"],
                },
                {
                    "name": "chunkNumber",
                    "dataType": ["int"],
                },
            ],
        }

        self.client.schema.create_class(class_obj)


class WeaviateStorageManager:
    """Manages document storage and retrieval in Weaviate."""

    def __init__(self, options: ProcessingOptions):
        """Initialize storage manager with processing options."""
        self.options = options
        self.client = weaviate.Client(
            url="http://localhost:8080",  # Update with your Weaviate instance URL
        )
        self.openai_client = OpenAI()
        self.initialize_storage()

    def initialize_storage(self) -> None:
        """Initialize Weaviate storage and create schema if needed."""
        try:
            schema = WeaviateSchema(self.client)

            # Check if schema exists
            existing_schema = self.client.schema.get()
            schema_exists = (
                any(c["class"] == "DocumentChunk" for c in existing_schema["classes"])
                if existing_schema.get("classes")
                else False
            )

            if not schema_exists or self.options.mode == ProcessingMode.OVERWRITE:
                if schema_exists:
                    self.client.schema.delete_class("DocumentChunk")
                schema.create_schema()

            logger.debug("Storage initialized successfully")

        except Exception as e:
            logger.error(f"Storage initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize storage: {str(e)}")

    def _generate_title_and_summary(self, text: str, source_id: str) -> Dict[str, str]:
        """Generate title and summary for a document chunk using OpenAI's GPT model."""
        logger.debug(
            f"Starting title and summary generation for source_id: {source_id}"
        )

        system_prompt = (
            "You are an AI that extracts titles and summaries from document chunks. "
            "Return a JSON object with 'title' and 'summary' keys. "
            "For the title: Create a concise, descriptive title for this chunk. "
            "For the summary: Create a brief summary of the main points in this chunk."
        )

        try:
            response = self.openai_client.chat.completions.create(
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

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            logger.error(f"Error generating title and summary: {e}")
            return {
                "title": "Error processing title",
                "summary": "Error processing summary",
            }

    def process_chunks(
        self, chunks: List[Any], metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process document chunks into storable format."""
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                filename = self._extract_filename(chunk) or f"chunk_{i}"
                content = str(chunk.text)
                source_type = (
                    metadata.get("file_type") if isinstance(metadata, dict) else None
                )
                generated_data = self._generate_title_and_summary(content, filename)

                chunk_dict = {
                    "text": content,
                    "title": generated_data.get("title", ""),
                    "summary": generated_data.get("summary", ""),
                    "sourceId": filename,
                    "sourceType": source_type,
                    "chunkNumber": i,
                }
                processed_chunks.append(chunk_dict)

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue

        return processed_chunks

    def _extract_filename(self, chunk: Any) -> Optional[str]:
        """Extract filename from chunk metadata."""
        try:
            return chunk.meta.origin.filename
        except AttributeError:
            return None

    def store_results(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Store processing results in Weaviate."""
        successful_stores = 0
        failed_stores = 0
        total_chunks = 0

        for result in results:
            if not result.success:
                failed_stores += 1
                continue

            try:
                for file_path in result.processed_files:
                    chunks = self.process_chunks(
                        self._get_chunks_for_file(file_path), result.metadata
                    )

                    # Store chunks in Weaviate
                    for chunk in chunks:
                        try:
                            uuid = generate_uuid5(
                                chunk["sourceId"], chunk["chunkNumber"]
                            )
                            self.client.data_object.create(
                                data_object=chunk, class_name="DocumentChunk", uuid=uuid
                            )
                            total_chunks += 1

                        except Exception as e:
                            logger.error(f"Error storing chunk: {str(e)}")
                            failed_stores += 1
                            continue

                    successful_stores += 1

            except Exception as e:
                logger.error(f"Error storing result: {str(e)}")
                failed_stores += 1

        return {
            "successful_stores": successful_stores,
            "failed_stores": failed_stores,
            "total_chunks": total_chunks,
        }

    def _get_chunks_for_file(self, file_path: Union[str, Path]) -> List[Any]:
        """Generate document chunks from a file."""
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
    """Main class coordinating document processing workflow.

    This class serves as the primary coordinator for document processing operations,
    managing the initialization and execution of various document processors and storage
    handlers. It supports processing of multiple document types including PDFs, URLs,
    and Git repositories.

    Attributes:
        engine (ProcessingEngine):
            The main processing enginethat coordinates file processing
        storage_manager (Optional[StorageManager]):
            Handler for storing processed documents
    """

    def __init__(self):
        """Initialize the DocumentProcessor with default processors.

        Creates a new ProcessingEngine instance
        and registers the default set of processors
        (PDF, URL, and Git).
        The storage handler is initially set to None and must be
        initialized via the initialize() method before processing can begin.
        """
        self.engine = ProcessingEngine()
        self.storage_manager: Optional[WeaviateStorageManager] = None

        # Register processors
        self.engine.register_processor(PDFProcessor())
        self.engine.register_processor(URLProcessor())
        self.engine.register_processor(GitProcessor())

    def initialize(self, options: ProcessingOptions) -> None:
        """Initialize the processor with the specified options.

        Sets up the processing engine and storage handler
        with the provided configuration options.
        This method must be called before any processing can occur.

        Args:
            options (ProcessingOptions):
                Configuration options for processing and storage

        Raises:
            RuntimeError: If initialization fails for any reason
        """
        try:
            self.engine.set_options(options)
            self.storage_manager = WeaviateStorageManager(options)
            logger.debug("Document processor fully initialized")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize processor: {str(e)}")

    def process(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """Process the input path and return processing results.

        Processes a single input path which can be a file, directory, or URL. The method
        coordinates the processing workflow, handles storage,
        and generates a summary of the processing results.

        Args:
            input_path (Union[str, Path]):
                Path to the input to process. Can be a file path,
                directory path, or URL

        Returns:
            Dict[str, Any]: A dictionary containing processing results including:
                - total_files: Total number of files processed
                - successful_files: Number of successfully processed files
                - failed_files: Number of files that failed processing
                - errors: List of error messages from failed processing attempts
                - total_chunks: (If stored) Number of chunks stored in the database

        Raises:
            RuntimeError: If process is called before initialization
        """
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
        """Create a summary of processing results.

        Analyzes a list of ProcessingResult objects and generates a summary dictionary
        containing counts of successful and failed operations,
        along with any error messages.

        Args:
            results (List[ProcessingResult]): List of processing results to summarize

        Returns:
            Dict[str, Any]: A dictionary containing:
                - total_files: Total number of files processed
                - successful_files: Number of successfully processed files
                - failed_files: Number of files that failed processing
                - errors: List of error messages from failed operations
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return {
            "total_files": len(results),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "errors": [r.error for r in failed if r.error],
        }


class DocumentSearcher:
    """Handles document search operations using Weaviate."""

    def __init__(self, client: weaviate.Client):
        self.client = client

    def analyze_search_results(self, results: List[Dict[str, Any]], query: str):
        """Analyze and print detailed information about search results."""
        print("\nSearch Analysis:")
        print(f"Query: '{query}'")

        if not results:
            print("No results to analyze")
            return

        # Get distances and scores, filtering out None values and converting to float
        distances = [r.get("_additional", {}).get("distance") for r in results]
        scores = [r.get("_additional", {}).get("score") for r in results]

        print("\nMetrics distribution:")
        if any(d is not None for d in distances):
            valid_distances = [float(d) for d in distances if d is not None]
            print(f"Min distance: {min(valid_distances):.4f}")
            print(f"Max distance: {max(valid_distances):.4f}")
            print(f"Average distance: {sum(valid_distances)/len(valid_distances):.4f}")

        if any(s is not None for s in scores):
            valid_scores = [float(s) for s in scores if s is not None]
            try:
                print(f"Min score: {min(valid_scores):.4f}")
                print(f"Max score: {max(valid_scores):.4f}")
                print(f"Average score: {sum(valid_scores)/len(valid_scores):.4f}")
            except (ValueError, TypeError) as e:
                print("Raw scores:", scores)
                logger.error(f"Error processing scores: {e}")

        print("\nField contribution analysis:")
        for idx, result in enumerate(results[:3], 1):
            print(f"\nResult #{idx}:")
            distance = result.get("_additional", {}).get("distance")
            score = result.get("_additional", {}).get("score")

            if distance is not None:
                try:
                    print(f"Distance: {float(distance):.4f}")
                except (ValueError, TypeError):
                    print(f"Distance: {distance}")

            if score is not None:
                try:
                    print(f"Score: {float(score):.4f}")
                except (ValueError, TypeError):
                    print(f"Score: {score}")

            # Content analysis
            title = result.get("title", "").lower()
            text = result.get("text", "").lower()
            summary = result.get("summary", "").lower()
            query_lower = query.lower()

            print(f"Title relevance: {'High' if query_lower in title else 'Low'}")
            print(f"Text relevance: {'High' if query_lower in text else 'Low'}")
            print(f"Summary relevance: {'High' if query_lower in summary else 'Low'}")

    def semantic_search(
        self, query: str, vector_field: str = "text", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform pure vector similarity search."""
        try:
            # Log search parameters
            logger.debug(
                f"""
                Search Configuration:
                - Query: {query}
                - Vector Field: {vector_field}
                - Limit: {limit}
                """
            )

            # Build query and capture it before execution
            search_query = (
                self.client.query.get(
                    "DocumentChunk",
                    [
                        "text",
                        "title",
                        "summary",
                        "sourceId",
                        "sourceType",
                        "chunkNumber",
                    ],
                )
                .with_near_text(
                    {
                        "concepts": [query],
                        "properties": [vector_field],
                        "certainty": 0.7,  # Add certainty threshold
                    }
                )
                .with_additional(["distance", "vector"])  # Add vector to response
            )

            # Log the raw GraphQL query
            raw_query = search_query.build()
            logger.debug(f"Generated GraphQL query:\n{raw_query}")

            # Execute search
            result = search_query.with_limit(limit).do()

            if (
                result
                and "data" in result
                and "Get" in result["data"]
                and "DocumentChunk" in result["data"]["Get"]
            ):
                first_result = result["data"]["Get"]["DocumentChunk"][0]
                if (
                    "_additional" in first_result
                    and "vector" in first_result["_additional"]
                ):
                    logger.debug(
                        f"Vector dimensions: {len(first_result['_additional']['vector'])}"
                    )
                logger.debug("Distance calculation method: cosine")

            return result["data"]["Get"]["DocumentChunk"]

        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return []

    def hybrid_search(
        self,
        query: str,
        vector_field: str = "text",
        limit: int = 10,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and text matching."""
        try:
            # Log search parameters
            logger.debug(
                f"""
                Hybrid Search Configuration:
                - Query: {query}
                - Vector Field: {vector_field}
                - Limit: {limit}
                - Alpha: {alpha}
                """
            )

            result = (
                self.client.query.get(
                    "DocumentChunk",
                    [
                        "text",
                        "title",
                        "summary",
                        "sourceId",
                        "sourceType",
                        "chunkNumber",
                    ],
                )
                .with_hybrid(query=query, properties=[vector_field], alpha=alpha)
                .with_additional(["score"])
                .with_limit(limit)
                .do()
            )

            return result["data"]["Get"]["DocumentChunk"]

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Document Processing and Search System"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search processed documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--mode", choices=["hybrid", "semantic"], default="hybrid", help="Search mode"
    )
    search_parser.add_argument(
        "--field",
        choices=["text", "title", "summary"],
        default="text",
        help="Field to search against",
    )
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results"
    )

    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_subparsers = process_parser.add_subparsers(
        dest="mode", help="Processing mode"
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
            "default": "data/weaviate",
            "help": "Path to Weaviate database",
        },
        "-t": {
            "dest": "table_name",
            "default": "DocumentChunk",
            "help": "Class name in Weaviate",
        },
        "--mode": {
            "choices": ["append", "overwrite"],
            "default": "append",
            "help": "Database operation mode",
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
            client = weaviate.Client("http://localhost:8080")
            searcher = DocumentSearcher(client)

            if args.mode == "hybrid":  # Fixed from args.search_mode
                results = searcher.hybrid_search(args.query, args.field, args.limit)
                print("debug")
                print(results[0].keys())
                print(results[0].get("_additional"))
            else:
                results = searcher.semantic_search(args.query, args.field, args.limit)

            # Print search results
            print(f"\nSearch Results for: {args.query}")
            for i, result in enumerate(results, 1):
                print(
                    f"\n{i}. Distance: {result.get('_additional', {}).get('distance', 'N/A')}"
                )
                print(f"Title: {result.get('title', 'No title')}")
                print(f"Summary: {result.get('summary', 'No summary')}")
                print(f"Source: {result.get('sourceId', 'No source')}")
                print(f"Text Preview: {result.get('text', 'No text')[:150]}...")
                print("-" * 80)

            # Add analysis of results
            searcher.analyze_search_results(results, args.query)
            return 0

        # Handle processing commands
        elif args.command == "process":
            options = ProcessingOptions(
                extract_tables=getattr(args, "extract_tables", False),
                split_vertical=getattr(args, "split_vertical", False),
                sitemap_only=getattr(args, "sitemap_only", False),
                mode=(
                    ProcessingMode.OVERWRITE
                    if args.mode == "overwrite"
                    else ProcessingMode.APPEND
                ),
                db_path=Path(args.db_path),
                table_name=args.table_name,
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


if __name__ == "__main__":
    sys.exit(main())
