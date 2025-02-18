import argparse
import lancedb
import numpy as np


def search_table(db_path: str, table_name: str, query: str = "", limit: int = 10):
    """Search LanceDB table using semantic search."""
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    # If no query provided, will return all results up to limit
    results = table.search(query).metric("cosine").limit(limit).to_pandas()

    # Display results with clear formatting
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\n{'='*80}")
        print(f"RESULT {i+1}")
        print(f"{'='*80}")

        # Display all fields
        for column in row.index:
            print(f"\n{column.upper()}:")
            value = row[column]

            if column == "vector":
                print(f"Type: {type(value)}")
                print(f"Shape: {value.shape}")
                print("First 5 values:", value[:5])
                print("Last 5 values:", value[-5:])
            elif column == "metadata":
                for key, val in value.items():
                    print(f"{key}: {val}")
            else:
                print(value)


def main():
    parser = argparse.ArgumentParser(description="Search LanceDB table")
    parser.add_argument(
        "--db-path", default="data/lancedb", help="Path to LanceDB database"
    )
    parser.add_argument("--table", default="docling", help="Table name")
    parser.add_argument(
        "--query", default="", help="Search query (empty for all results)"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of results to show"
    )

    args = parser.parse_args()
    search_table(args.db_path, args.table, args.query, args.limit)


if __name__ == "__main__":
    main()
