import os
import json
import argparse
from lightrag.utils import xml_to_json
from neo4j import GraphDatabase


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert XML graph data to JSON and import to Neo4j"
    )

    # File paths
    parser.add_argument(
        "--xml-file",
        default="graph_chunk_entity_relation.graphml",
        help="Path to input XML graph file (default: %(default)s)",
    )
    parser.add_argument(
        "--json-file",
        default="graph_data.json",
        help="Path to output JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--working-dir",
        default=".",
        help="Working directory for relative paths (default: current directory)",
    )

    # Neo4j connection parameters
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: %(default)s)",
    )
    parser.add_argument(
        "--neo4j-user", default="neo4j", help="Neo4j username (default: %(default)s)"
    )
    parser.add_argument(
        "--neo4j-password", default="your_password", help="Neo4j password"
    )

    # Processing parameters
    parser.add_argument(
        "--batch-size-nodes",
        type=int,
        default=500,
        help="Batch size for node processing (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size-edges",
        type=int,
        default=100,
        help="Batch size for edge processing (default: %(default)s)",
    )

    # Operation flags
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip XML to JSON conversion and use existing JSON file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Convert data without importing to Neo4j"
    )

    return parser.parse_args()


def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON file created: {output_path}")
        return json_data
    else:
        print("Failed to create JSON data")
        return None


def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Resolve paths
    xml_file = os.path.join(args.working_dir, args.xml_file)
    json_file = os.path.join(args.working_dir, args.json_file)

    # Convert XML to JSON or load existing JSON
    json_data = None
    if args.skip_conversion and os.path.exists(json_file):
        print(f"Using existing JSON file: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    else:
        json_data = convert_xml_to_json(xml_file, json_file)

    if json_data is None:
        return

    # Load nodes and edges
    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])

    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")

    if args.dry_run:
        print("Dry run completed, skipping Neo4j import")
        return

    # Neo4j queries
    create_nodes_query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id
    REMOVE e:Entity
    WITH e, node
    CALL apoc.create.addLabels(e, [node.entity_type]) YIELD node AS labeledNode
    RETURN count(*)
    """

    create_edges_query = """
    UNWIND $edges AS edge
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})
    WITH source, target, edge,
         CASE
            WHEN edge.keywords CONTAINS 'lead' THEN 'lead'
            WHEN edge.keywords CONTAINS 'participate' THEN 'participate'
            WHEN edge.keywords CONTAINS 'uses' THEN 'uses'
            WHEN edge.keywords CONTAINS 'located' THEN 'located'
            WHEN edge.keywords CONTAINS 'occurs' THEN 'occurs'
           ELSE REPLACE(SPLIT(edge.keywords, ',')[0], '\"', '')
         END AS relType
    CALL apoc.create.relationship(source, relType, {
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }, target) YIELD rel
    RETURN count(*)
    """

    set_displayname_and_labels_query = """
    MATCH (n)
    SET n.displayName = n.id
    WITH n
    CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
    RETURN count(*)
    """

    # Create a Neo4j driver
    print(f"Connecting to Neo4j at {args.neo4j_uri}")
    driver = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)
    )

    try:
        # Execute queries in batches
        with driver.session() as session:
            # Insert nodes in batches
            print(
                f"Importing {len(nodes)} nodes in batches of {args.batch_size_nodes}..."
            )
            session.execute_write(
                process_in_batches, create_nodes_query, nodes, args.batch_size_nodes
            )

            # Insert edges in batches
            print(
                f"Importing {len(edges)} edges in batches of {args.batch_size_edges}..."
            )
            session.execute_write(
                process_in_batches, create_edges_query, edges, args.batch_size_edges
            )

            # Set displayName and labels
            print("Setting displayName and labels...")
            session.run(set_displayname_and_labels_query)

            print("Import completed successfully")

    except Exception as e:
        print(f"Error occurred during Neo4j operations: {e}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
