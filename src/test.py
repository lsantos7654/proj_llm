import lancedb


def inspect_schema(db_path: str, table_name: str):
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    print("Table Schema:")
    print(table.schema)

    # Print each field and its type
    print("\nDetailed Fields:")
    for field in table.schema:
        print(f"Field: {field.name}")
        print(f"Type: {field.type}")
        print("-" * 40)


if __name__ == "__main__":
    inspect_schema("data/lancedb", "docling")
