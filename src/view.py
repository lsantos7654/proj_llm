import argparse
import lancedb
import pandas as pd
import numpy as np

pd.set_option("display.max_colwidth", None)


class TableViewer:
    def __init__(self, db_path):
        self.db = lancedb.connect(db_path)

    def list_tables(self):
        return self.db.table_names()

    def get_table(self, table_name):
        return self.db.open_table(table_name)

    def show_sample(self, table, n=5):
        return table.to_pandas().head(n)

    def show_stats(self, table):
        df = table.to_pandas()
        return {
            "total_rows": len(df),
            "unique_files": df["metadata"].apply(lambda x: x.get("filename")).nunique(),
            "total_pages": df["metadata"]
            .apply(lambda x: len(x.get("page_numbers", []) or []))
            .sum(),
        }

    def search(self, table, query, limit=5):
        return table.search(query).limit(limit).to_pandas()

    def inspect_rows(self, table, start_row=0, num_rows=1):
        """
        Inspect a range of rows from the table.

        Args:
            table: LanceDB table
            start_row (int): Starting row index
            num_rows (int): Number of rows to inspect
        """
        df = table.to_pandas()
        total_rows = len(df)

        if start_row >= total_rows:
            print(
                f"Error: Start row {start_row} exceeds table size ({total_rows} rows)"
            )
            return

        end_row = min(start_row + num_rows, total_rows)
        rows_to_show = df.iloc[start_row:end_row]

        for idx, row in rows_to_show.iterrows():
            print(f"\n=== Row {idx} ===")
            for column in df.columns:
                value = row[column]
                print(f"\n{column}:")

                if isinstance(value, (np.ndarray, list)):
                    print(f"Type: {type(value)}")
                    print(f"Shape/Length: {len(value)}")
                    print("Values:")
                    if len(value) > 10:
                        head_values = value[:5]
                        tail_values = value[-5:]
                        print(f"First 5: {head_values}")
                        print(f"Last 5: {tail_values}")
                    else:
                        print(value)
                elif isinstance(value, dict):
                    print("Dictionary contents:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(value)


def main():
    parser = argparse.ArgumentParser(description="View LanceDB table contents")
    parser.add_argument(
        "--db-path", default="data/lancedb", help="Path to LanceDB database"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List tables command
    subparsers.add_parser("list", help="List all tables in the database")

    # Show sample command
    sample_parser = subparsers.add_parser(
        "sample", help="Show sample rows from a table"
    )
    sample_parser.add_argument("table", help="Table name")
    sample_parser.add_argument(
        "--rows", type=int, default=5, help="Number of rows to show"
    )

    # Show stats command
    stats_parser = subparsers.add_parser("stats", help="Show table statistics")
    stats_parser.add_argument("table", help="Table name")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search table contents")
    search_parser.add_argument("table", help="Table name")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to show"
    )

    inspect_parser = subparsers.add_parser("inspect", help="Inspect rows of the table")
    inspect_parser.add_argument("table", help="Table name")
    inspect_parser.add_argument(
        "--start", type=int, default=0, help="Starting row index (default: 0)"
    )
    inspect_parser.add_argument(
        "--rows", type=int, default=1, help="Number of rows to show (default: 1)"
    )
    args = parser.parse_args()

    try:
        viewer = TableViewer(args.db_path)

        if args.command == "list":
            tables = viewer.list_tables()
            print("\nAvailable tables:")
            for table in tables:
                print(f"- {table}")

        elif args.command == "sample":
            table = viewer.get_table(args.table)
            print(f"\nShowing {args.rows} sample rows from table '{args.table}':")
            print(viewer.show_sample(table, args.rows))

        elif args.command == "stats":
            table = viewer.get_table(args.table)
            stats = viewer.show_stats(table)
            print(f"\nStatistics for table '{args.table}':")
            print(f"Total rows: {stats['total_rows']}")
            print(f"Unique files: {stats['unique_files']}")
            print(f"Total pages: {stats['total_pages']}")

        elif args.command == "search":
            table = viewer.get_table(args.table)
            print(f"\nSearching '{args.table}' for '{args.query}':")
            results = viewer.search(table, args.query, args.limit)
            print(results)

        elif args.command == "inspect":
            table = viewer.get_table(args.table)
            viewer.inspect_rows(table, args.start, args.rows)

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
