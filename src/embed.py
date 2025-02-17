from typing import List, Dict, Any, Optional, Set, Union
import argparse
import os
import lancedb
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper
from pathlib import Path
from utils.pdf_spliter import split_pdf_vertically
from utils.segment_tables import process_image, process_pdf
import requests
from xml.etree import ElementTree
import logging
import tempfile
from git import Repo, GitCommandError
import tempfile
import re


class ChunkMetadata(LanceModel):
    """
    Schema for chunk metadata. Fields must be in alphabetical order per Pydantic requirements.
    """

    filename: Optional[str]
    page_numbers: Optional[List[int]]
    title: Optional[str]


# Get the OpenAI embedding function
func = get_registry().get("openai").create(name="text-embedding-3-large")


class Chunks(LanceModel):
    """Schema for document chunks with embeddings."""

    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata


class Extract:
    def __init__(self) -> None:
        load_dotenv()
        self.client = OpenAI()
        self.tokenizer = OpenAITokenizerWrapper()
        self.MAX_TOKENS: int = 8191
        self.converter = DocumentConverter()

    def extract_document(self, source: Union[str, Path]) -> Any:
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            try:
                return self.converter.convert(str(source))
            except requests.RequestException as e:
                raise Exception(f"Failed to fetch URL {source}: {e}")
        else:
            return self.converter.convert(source)

    def preview_document(self, source: Union[str, Path], output_format="markdown"):
        try:
            result = self.converter.convert(source)
            document = result.document

            if output_format == "markdown":
                return document.export_to_markdown()
            elif output_format == "json":
                return document.export_to_dict()
        except Exception as e:
            raise Exception(f"Error converting PDF: {e}")


class LanceDBExtractor(Extract):
    def __init__(self, db_path: Path = Path("data/lancedb")) -> None:
        super().__init__()
        self.db = lancedb.connect(db_path)
        self.embedding_func = (
            get_registry().get("openai").create(name="text-embedding-3-large")
        )

    def setup_chunker(self) -> HybridChunker:
        return HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.MAX_TOKENS,
            merge_peers=True,
        )

    def process_chunks(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        processed_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            try:
                # Process page numbers with error handling
                try:
                    page_numbers: Optional[List[int]] = [
                        page_no
                        for page_no in sorted(
                            set(
                                prov.page_no
                                for item in chunk.meta.doc_items
                                for prov in item.prov
                            )
                        )
                    ] or None
                except (AttributeError, TypeError) as e:
                    print(f"Warning: Failed to process page numbers: {str(e)}")
                    page_numbers = None

                # Process filename and title with error handling
                try:
                    filename: Optional[str] = chunk.meta.origin.filename
                except AttributeError:
                    print("Warning: Failed to get filename from chunk metadata")
                    filename = None

                try:
                    title: Optional[str] = (
                        chunk.meta.headings[0] if chunk.meta.headings else None
                    )
                except (AttributeError, IndexError):
                    print("Warning: Failed to get title from chunk metadata")
                    title = None

                chunk_dict: Dict[str, Any] = {
                    "text": str(chunk.text),
                    "metadata": {
                        "filename": filename,
                        "page_numbers": page_numbers,
                        "title": title,
                    },
                }
                processed_chunks.append(chunk_dict)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue

        if not processed_chunks:
            raise ValueError("No chunks were successfully processed")

        return processed_chunks

    def create_or_get_table(self, table_name: str, mode: str) -> Table:
        try:
            print(f"\nAttempting to create/get table:")
            print(f"  Table name: {table_name}")
            print(f"  Mode: {mode}")

            if mode == "append":
                print("  Checking if table exists...")
                table_names = self.db.table_names()
                print(f"  Available tables: {table_names}")

                if table_name in table_names:
                    print(f"  Found existing table '{table_name}', opening...")
                    table = self.db.open_table(table_name)
                    print("  Successfully opened existing table")
                    return table
            elif mode == "overwrite":
                print("  Checking if table exists...")
                table_names = self.db.table_names()
                print(f"  Overwriting tables: {table_names}")
            else:
                print(f"  Creating new table '{table_name}'...")

            try:
                table = self.db.create_table(
                    name=table_name, schema=Chunks, mode="overwrite"
                )
                print("  Successfully created new table")
                return table
            except Exception as e:
                print(f"  Error creating table: {str(e)}")
                print(f"  Schema type: {type(Chunks)}")
                print(f"  Schema details: {dir(Chunks)}")
                raise

        except Exception as e:
            error_msg = f"Failed to create or get table '{table_name}': {str(e)}"
            print(f"  Fatal error: {error_msg}")
            raise RuntimeError(error_msg)

    def process_source(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        try:
            print(f"Processing source: {source}")
            # Extract document
            result = self.extract_document(source)
            if not result:
                raise ValueError(f"Document extraction failed for source: {source}")

            # Setup and apply chunking
            chunker = self.setup_chunker()
            if not chunker:
                raise ValueError("Failed to initialize chunker")

            try:
                chunks = list(chunker.chunk(dl_doc=result.document))
                if not chunks:
                    print(f"Warning: No chunks generated for source: {source}")
                    return []
            except Exception as e:
                raise ValueError(f"Chunking failed: {str(e)}")

            # Process chunks
            return self.process_chunks(chunks)
        except Exception as e:
            print(f"Error processing source {source}: {str(e)}")
            return []

    def extract_and_store(
        self,
        sources: List[Path],
        table_name: str,
        mode: str,
    ) -> Table:
        if not sources:
            raise ValueError("No sources provided")

        try:
            # Create or get table
            table = self.create_or_get_table(table_name, mode)

            successful_sources = 0
            failed_sources = 0

            # Process each source
            for source in sources:
                try:
                    processed_chunks = self.process_source(source)
                    if processed_chunks:
                        table.add(processed_chunks)
                        successful_sources += 1
                    else:
                        failed_sources += 1
                        print(f"No chunks produced for source: {source}")
                except Exception as e:
                    failed_sources += 1
                    print(f"Failed to process source {source}: {str(e)}")
                    continue

            # Print processing summary
            print(f"\nProcessing Summary:")
            print(f"Successfully processed: {successful_sources} sources")
            print(f"Failed to process: {failed_sources} sources")

            if successful_sources == 0:
                raise ValueError("No sources were successfully processed")

            return table

        except Exception as e:
            raise RuntimeError(f"Failed to extract and store documents: {str(e)}")

    def get_table_info(self, table: Table) -> Dict[str, Any]:
        return {"dataframe": table.to_pandas(), "row_count": table.count_rows()}


def process_sitemap(url: str) -> List[Union[str, Path]]:
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Handle different sitemap namespaces
        namespaces = {
            "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
            "xhtml": "http://www.w3.org/1999/xhtml",
        }

        urls: List[Union[str, Path]] = []

        # Check if this is a sitemap index
        sitemaps = root.findall(".//sm:sitemap/sm:loc", namespaces)
        if sitemaps:
            # This is a sitemap index, process each sitemap
            for sitemap in sitemaps:
                if sitemap.text:  # Check for None
                    urls.extend(process_sitemap(sitemap.text))
        else:
            # This is a regular sitemap, extract URLs
            locations = root.findall(".//sm:url/sm:loc", namespaces)
            # Filter out None values and convert to list of strings
            valid_urls = [loc.text for loc in locations if loc.text is not None]
            urls.extend(valid_urls)

        return urls
    except requests.RequestException as e:
        logging.error(f"Failed to fetch sitemap: {e}")
        return []
    except ElementTree.ParseError as e:
        logging.error(f"Failed to parse sitemap XML: {e}")
        return []


def validate_git_url(url: str) -> bool:
    """Validate if the URL is a valid Git repository URL."""
    # Basic pattern for git URLs
    git_patterns = [
        r"https?://github\.com/[\w-]+/[\w-]+(?:\.git)?/?$",
        r"git@github\.com:[\w-]+/[\w-]+(?:\.git)?/?$",
    ]
    return any(re.match(pattern, url) for pattern in git_patterns)


def process_git_repo(repo_url: str, output_dir: Path) -> List[Union[str, Path]]:
    """Clone and process a Git repository.

    Args:
        repo_url: URL of the Git repository

    Returns:
        List of Path objects for files in the repository
    """
    # Validate URL format first
    if not validate_git_url(repo_url):
        logging.error(f"Invalid Git repository URL format: {repo_url}")
        return []

    try:
        print(f"Attempting to clone {repo_url} to {output_dir}")

        # Try to clone the repository
        repo = Repo.clone_from(repo_url, output_dir)

        # Verify the clone was successful
        if not repo.git_dir:
            raise GitCommandError("git clone", "Repository appears empty")

        # List files in the repository for debugging
        temp_path = Path(output_dir)
        all_files = list(temp_path.rglob("*"))
        print(f"Files found in repository: {len(all_files)}")
        for file in all_files[:5]:  # Show first 5 files
            print(f"Found: {file.relative_to(temp_path)}")

        # Process the directory
        sources = process_input(Path(output_dir), output_dir)

        if not sources:
            print("No supported files found in repository")
        else:
            print(f"Found {len(sources)} supported files")

        return sources

    except GitCommandError as e:
        logging.error(f"Git command failed: {e.command}, {e.status}, {e.stderr}")
        return []
    except Exception as e:
        logging.error(f"Failed to process Git repository {repo_url}: {e}")
        return []


def process_input(
    input_path: Union[str, Path], output_dir: Path
) -> List[Union[str, Path]]:
    """Process input sources and return a list of valid sources.

    Args:
        input_path: Either a string URL or Path object

    Returns:
        List of string URLs and Path objects
    """
    if isinstance(input_path, str) and (
        input_path.endswith(".git")
        or "github.com" in input_path
        or "gitlab.com" in input_path
    ):
        print(f"Processing Git Repo {input_path}")
        return process_git_repo(input_path, output_dir)

    if isinstance(input_path, str) and input_path.startswith(("http://", "https://")):
        if input_path.endswith("sitemap.xml"):
            return process_sitemap(input_path)
        return [str(input_path)]  # Return URL as string

    # Convert string to Path if it's a local path
    input_path = Path(input_path) if isinstance(input_path, str) else input_path

    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        # Return all files, regardless of extension
        return [p for p in input_path.rglob("*") if p.is_file()]
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def main() -> int:
    """
    Main function to process documents and store them in LanceDB.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Extract and store documents in LanceDB"
    )
    parser.add_argument("input", help="Input file, directory, or URL", type=str)
    parser.add_argument(
        "-db",
        "--db-path",
        default="data/lancedb",
        help="Path to LanceDB database",
        type=str,
    )
    parser.add_argument(
        "-t", "--table", default="docling", help="Table name to use or create", type=str
    )
    parser.add_argument(
        "--mode",
        choices=["append", "overwrite"],
        default="append",
        help="Whether to append to existing table or create new one",
        type=str,
    )
    parser.add_argument(
        "--extract-tables",
        action="store_true",
        help="Extract tables from PDFs and images",
    )
    parser.add_argument(
        "--split-pdf", action="store_true", help="Split PDFs vertically into two halves"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="Output directory for extracted tables and split PDFs",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--preview",
        action="store_true",
        help="Preview the document processing results without storing in database",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    processed_sources: List[Union[str, Path]] = []

    try:
        print("\n=== Starting Processing ===")
        print(f"Input path: {args.input}")
        print(f"Output directory: {output_dir}")

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Created/verified output directory")

        # Process input path
        print("\nScanning for files...")
        sources: List[Union[str, Path]] = process_input(args.input, args.output_dir)
        total_files: int = len(sources)

        if not sources:
            print(f"No valid files found in {args.input}")
            return 1

        if args.preview:
            if len(sources) > 1:
                print(f"\nFound {total_files} files to preview:")
                for idx, src in enumerate(sources, 1):
                    print(f"{idx}. {src}")
                print("\nWould you like to proceed with previewing these files? (y/n)")
                response: str = input().lower()
                if response != "y":
                    print("Operation cancelled by user")
                    return 0
            extractor = Extract()
            for source in sources:
                try:
                    print(extractor.preview_document(source))
                except Exception as e:
                    print(f"Error previewing {source}: {e}")
            return 0

        print(f"\nFound {total_files} files to process:")
        for idx, src in enumerate(sources, 1):
            print(f"{idx}. {src}")

        print("\nWould you like to proceed with processing these files? (y/n)")
        response: str = input().lower()
        if response != "y":
            print("Operation cancelled by user")
            return 0

        for idx, source in enumerate(sources, 1):
            if isinstance(source, str) and source.startswith(("http://", "https://")):
                print(f"\n[{idx}/{total_files}] Processing: {source}")
                processed_sources.append(source)
                continue

            source_path = Path(source)
            print(f"\n[{idx}/{total_files}] Processing: {source_path}")

            if args.split_pdf and source_path.suffix.lower() == ".pdf":
                print(f"  - Splitting PDF...")
                try:
                    left_path, right_path = split_pdf_vertically(
                        source_path, output_dir
                    )
                    processed_sources.extend([left_path, right_path])
                    print(
                        f"  ✓ Split complete: Created {left_path.name} and {right_path.name}"
                    )

                    if args.extract_tables:
                        print(f"  - Extracting tables from split PDFs...")
                        print(f"    Processing left half...")
                        left_tables = process_pdf(left_path, output_dir)
                        processed_sources.extend(left_tables)
                        print(f"    Processing right half...")
                        right_tables = process_pdf(right_path, output_dir)
                        processed_sources.extend(right_tables)
                        print("  ✓ Table extraction complete for split PDFs")
                except Exception as e:
                    print(f"  ! Error processing split PDF: {e}")
                    continue
            else:
                processed_sources.append(source_path)
                if args.extract_tables:
                    try:
                        if source_path.suffix.lower() == ".pdf":
                            print(f"  - Extracting tables from PDF...")
                            table_paths = process_pdf(source_path, output_dir)
                            processed_sources.extend(table_paths)
                            print(f"  ✓ Extracted {len(table_paths)} tables")
                        elif source_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                            print(f"  - Extracting tables from image...")
                            table_paths = process_image(source_path, output_dir)
                            processed_sources.extend(table_paths)
                            print(f"  ✓ Extracted {len(table_paths)} tables")
                    except Exception as e:
                        print(f"  ! Error extracting tables: {e}")
                        continue

        print("\nInitializing LanceDB extractor...")
        extractor = LanceDBExtractor(db_path=args.db_path)

        print(f"\nStoring {len(processed_sources)} processed sources in LanceDB...")
        print("Sources to be stored:", [str(p) for p in processed_sources])

        try:
            table = extractor.extract_and_store(
                sources=processed_sources,
                table_name=args.table,
                mode=args.mode,
            )

            # Print results
            table_info = extractor.get_table_info(table)
            print(f"\nDatabase Results:")
            print(f"✓ Successfully processed {table_info['row_count']} rows")
            print(f"✓ Data stored in table '{args.table}'")

            # Print summary of processing
            if args.extract_tables or args.split_pdf:
                print("\n=== Processing Summary ===")
                print(f"Total files processed: {total_files}")
                if args.split_pdf:
                    print(f"PDF splitting results saved in: {output_dir}")
                if args.extract_tables:
                    print(f"Extracted tables saved in: {output_dir}")
                print(f"Processing complete!")

        except Exception as e:
            print(f"Error storing in LanceDB: {str(e)}")
            return 1

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
