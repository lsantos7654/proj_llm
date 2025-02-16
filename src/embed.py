from typing import List
import argparse
import os
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper
from pathlib import Path


# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: str | None
    page_numbers: List[int] | None
    title: str | None


# Get the OpenAI embedding function
func = get_registry().get("openai").create(name="text-embedding-3-large")


# Define the main Schema
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata


class Extract:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()
        self.tokenizer = OpenAITokenizerWrapper()
        self.MAX_TOKENS = 8191
        self.converter = DocumentConverter()

    def extract_document(self, source):
        return self.converter.convert(source)


class LanceDBExtractor(Extract):
    def __init__(self, db_path="data/lancedb"):
        super().__init__()
        self.db = lancedb.connect(db_path)
        self.embedding_func = (
            get_registry().get("openai").create(name="text-embedding-3-large")
        )

    def setup_chunker(self):
        return HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.MAX_TOKENS,
            merge_peers=True,
        )

    def process_chunks(self, chunks):
        return [
            {
                "text": chunk.text,
                "metadata": {
                    "filename": chunk.meta.origin.filename,
                    "page_numbers": [
                        page_no
                        for page_no in sorted(
                            set(
                                prov.page_no
                                for item in chunk.meta.doc_items
                                for prov in item.prov
                            )
                        )
                    ]
                    or None,
                    "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                },
            }
            for chunk in chunks
        ]

    def create_or_get_table(self, table_name="docling", mode="overwrite"):
        if mode == "append" and table_name in self.db.table_names():
            return self.db.open_table(table_name)

        table = self.db.create_table(table_name, schema=Chunks, mode="overwrite")
        return table

    def process_source(self, source):
        # Extract document
        result = self.extract_document(source)

        # Setup and apply chunking
        chunker = self.setup_chunker()
        chunks = list(chunker.chunk(dl_doc=result.document))

        # Process chunks
        return self.process_chunks(chunks)

    def extract_and_store(self, sources, table_name="docling", mode="overwrite"):
        # Create or get table
        table = self.create_or_get_table(table_name, mode)

        # Process each source
        for source in sources:
            processed_chunks = self.process_source(source)
            table.add(processed_chunks)

        return table

    def get_table_info(self, table):
        return {"dataframe": table.to_pandas(), "row_count": table.count_rows()}


def process_input(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        supported_extensions = {".pdf", ".txt", ".html", ".htm", ".jpg", ".png"}
        files = []
        for root, _, filenames in os.walk(input_path):
            for filename in filenames:
                if Path(filename).suffix.lower() in supported_extensions:
                    files.append(os.path.join(root, filename))
        return files
    elif input_path.startswith(("http://", "https://")):
        return [input_path]
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and store documents in LanceDB"
    )
    parser.add_argument("input", help="Input file, directory, or URL")
    parser.add_argument(
        "--db-path", default="data/lancedb", help="Path to LanceDB database"
    )
    parser.add_argument(
        "--table", default="docling", help="Table name to use or create"
    )
    parser.add_argument(
        "--mode",
        choices=["append", "overwrite"],
        default="append",
        help="Whether to append to existing table or create new one",
    )

    args = parser.parse_args()

    try:
        # Initialize extractor
        extractor = LanceDBExtractor(db_path=args.db_path)

        # Process input path
        sources = process_input(args.input)

        if not sources:
            print(f"No valid files found in {args.input}")
            return

        # Extract and store
        table = extractor.extract_and_store(
            sources=sources, table_name=args.table, mode=args.mode
        )

        # Print results
        table_info = extractor.get_table_info(table)
        print(f"Successfully processed {table_info['row_count']} rows")
        print(f"Data stored in table '{args.table}'")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
