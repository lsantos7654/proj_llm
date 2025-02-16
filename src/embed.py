from typing import List, Dict, Any, Optional, Set
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

    def extract_document(self, source: Path) -> Any:
        return self.converter.convert(str(source))

    def preview_document(self, source: Path, output_format="markdown"):
        try:
            result = self.converter.convert(str(source))
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

    def process_source(self, source: Path) -> List[Dict[str, Any]]:
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


def process_input(input_path: Path) -> List[Path]:
    if os.path.isfile(input_path):
        return [Path(input_path)]
    elif os.path.isdir(input_path):
        supported_extensions: Set[str] = {
            ".pdf",
            ".txt",
            ".html",
            ".htm",
            ".jpg",
            ".png",
        }
        files: List[Path] = []
        for root, _, filenames in os.walk(input_path):
            for filename in filenames:
                if Path(filename).suffix.lower() in supported_extensions:
                    files.append(Path(os.path.join(root, filename)))
        return files
    elif isinstance(input_path, str) and input_path.startswith(("http://", "https://")):
        return [Path(input_path)]
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
    processed_sources: List[Path] = []

    try:
        print("\n=== Starting Processing ===")
        print(f"Input path: {args.input}")
        print(f"Output directory: {output_dir}")

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Created/verified output directory")

        # Process input path
        print("\nScanning for files...")
        sources: List[Path] = process_input(args.input)
        total_files: int = len(sources)

        if not sources:
            print(f"No valid files found in {args.input}")
            return 1

        if args.preview == True:
            if len(sources) > 1:
                print(f"\nFound {total_files} files to preview:")
                for idx, src in enumerate(sources, 1):
                    print(f"{idx}. {src}")
                print("\nWould you like to proceed with processing these files? (y/n)")
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
