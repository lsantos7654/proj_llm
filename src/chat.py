import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx
import tempfile
import pandas as pd
from io import StringIO


# Initialize LightRAG with proper embedding configuration
# @st.cache_resource
def init_rag():
    return LightRAG(
        working_dir="data/default_db",
        embedding_func=EmbeddingFunc(
            embedding_dim=1536, max_token_size=8192, func=openai_embed
        ),
        llm_model_func=gpt_4o_mini_complete,
    )


def visualize_graph(G):
    """Create an interactive visualization of the knowledge graph."""
    net = Network(notebook=True, height="500px", width="100%")
    net.from_nx(G)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        return f.name


def parse_entities(text):
    """Parse entities CSV section from the text."""
    try:
        # Check if the text contains the entities section
        if "-----Entities-----" not in text:
            return pd.DataFrame()

        # Extract the entities section
        entities_section = text.split("-----Entities-----")[1].split(
            "-----Relationships-----"
        )[0]

        # Clean up the CSV text
        if "```csv" in entities_section:
            csv_text = entities_section.split("```csv")[1].split("```")[0].strip()
        else:
            csv_text = entities_section.strip()

        # Manual parsing to handle inconsistent field counts
        lines = csv_text.split("\n")
        header_parts = lines[0].strip().split(",")
        header = [part.strip() for part in header_parts]

        # Create a list to store processed rows
        rows = []
        for line in lines[1:]:
            if not line.strip():  # Skip empty lines
                continue

            # Extract data between quotes and handle the special formatting
            row_data = []
            current_field = ""
            in_quotes = False

            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                    current_field += char
                elif char == "," and not in_quotes:
                    row_data.append(current_field.strip())
                    current_field = ""
                else:
                    current_field += char

            # Add the last field
            if current_field:
                row_data.append(current_field.strip())

            # Make sure we have the right number of columns
            if len(row_data) >= len(header):
                rows.append(row_data[: len(header)])
            else:
                # Pad with empty values if needed
                rows.append(row_data + [""] * (len(header) - len(row_data)))

        # Create DataFrame from processed data
        df = pd.DataFrame(rows, columns=header)
        return df

    except Exception as e:
        st.error(f"Error parsing entities: {e}")
        # Display raw text as a fallback
        raw_df = pd.DataFrame({"Raw Data": [entities_section]})
        return raw_df


def parse_relationships(text):
    """Parse relationships CSV section from the text."""
    try:
        # Extract the relationships section
        relationships_section = text.split("-----Relationships-----")[1].split(
            "-----Sources-----"
        )[0]
        # Clean up the CSV text
        csv_text = (
            relationships_section.replace("```csv", "").replace("```", "").strip()
        )
        # Use pandas to read the CSV text
        df = pd.read_csv(StringIO(csv_text))
        return df
    except Exception as e:
        st.error(f"Error parsing relationships: {e}")
        return pd.DataFrame()


def parse_sources(text):
    """Parse sources CSV section from the text."""
    try:
        # Check if the text contains the sources section
        if "-----Sources-----" not in text:
            return pd.DataFrame()

        # Extract the sources section
        sources_section = text.split("-----Sources-----")[1]

        # Clean up the CSV text
        if "```csv" in sources_section:
            csv_text = sources_section.split("```csv")[1].split("```")[0].strip()
        else:
            csv_text = sources_section.strip()

        # Manual parsing to handle the complex structure
        lines = csv_text.split("\n")
        header_parts = lines[0].strip().split(",")
        header = [part.strip() for part in header_parts]

        # Create a list to store data rows
        data_rows = []

        # Simple two-column parsing as fallback
        for i in range(1, len(lines)):
            if not lines[i].strip():  # Skip empty lines
                continue

            # Find the first comma to split id and content
            parts = lines[i].split(",", 1)
            if len(parts) >= 2:
                id_val = parts[0].strip()
                content_val = parts[1].strip()
                data_rows.append({"id": id_val, "content": content_val})
            else:
                # If there's no comma, treat the whole line as content
                data_rows.append({"id": f"row_{i}", "content": lines[i].strip()})

        # Create DataFrame
        df = pd.DataFrame(data_rows)
        return df

    except Exception as e:
        st.error(f"Error parsing sources: {e}")
        # Display raw text as a fallback
        raw_df = pd.DataFrame({"Raw Data": [sources_section]})
        return raw_df


# CSS for styling
st.markdown(
    """
    <style>
    .search-result {
        margin: 15px 0;
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .search-result:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .search-result h4 {
        margin: 0 0 10px 0;
        color: #1e88e5;
        font-size: 1.1em;
        font-weight: 600;
    }
    .search-result .content {
        margin-bottom: 10px;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .search-result .metadata {
        color: #666;
        font-size: 0.9em;
        padding-top: 8px;
        border-top: 1px solid #eee;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 4px;
        color: #000;
        border: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1e88e5;
        color: white;
    }
    </style>
    <style>
        [data-testid="stSidebar"][aria-expanded="true"] {
            display: none;
        }
        div[data-testid="stSidebarNav"] {
            display: none;
        }
        button[kind="secondary"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Streamlit app
st.title("📚 Document Q&A with Knowledge Graph")
if st.button("Toggle Sidebar"):
    st.session_state.sidebar_visible = not st.session_state.get(
        "sidebar_visible", False
    )

# Initialize RAG
rag = init_rag()

# Sidebar for controls
with st.sidebar:
    st.header("Controls")

    # Search mode selection
    search_mode = st.selectbox(
        "Search Mode",
        ["hybrid", "local", "global", "naive", "mix"],
        help="Select the search mode for query processing",
    )

    if st.session_state.get("sidebar_visible", False):
        with st.sidebar:
            st.header("Controls")

    # Show graph visualization
    if st.button("Show Knowledge Graph"):
        with st.spinner("Generating graph visualization..."):
            G = nx.read_graphml(
                f"{rag.working_dir}/graph_chunk_entity_relation.graphml"
            )
            html_path = visualize_graph(G)
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=600)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Create a dedicated dropdown for search results
    with st.expander("🔍 Searching Documents", expanded=False):
        # Get context first
        context = rag.query(
            prompt,
            param=QueryParam(
                mode=search_mode,
                response_type="Multiple Paragraphs",
                only_need_context=True,
                top_k=5,
            ),
        )

        # Create tabs for different types of information
        tab_context, tab_entities, tab_relationships, tab_sources = st.tabs(
            ["📑 Context", "👤 Entities", "🔗 Relationships", "📚 Sources"]
        )

        # Context Tab
        with tab_context:
            if isinstance(context, list):
                for idx, ctx in enumerate(context, 1):
                    st.markdown(
                        f"""
                        <div class="search-result">
                            <h4>Context {idx}</h4>
                            <div class="content">{ctx}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"""
                    <div class="search-result">
                        <h4>Context</h4>
                        <div class="content">{context}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Entities Tab
        with tab_entities:
            st.markdown("### Retrieved Entities")
            entities_df = parse_entities(context)
            if not entities_df.empty:
                st.dataframe(entities_df, use_container_width=True, hide_index=True)

        # Relationships Tab
        with tab_relationships:
            st.markdown("### Retrieved Relationships")
            relationships_df = parse_relationships(context)
            if not relationships_df.empty:
                st.dataframe(
                    relationships_df, use_container_width=True, hide_index=True
                )

        # Sources Tab
        with tab_sources:
            st.markdown("### Source Documents")
            sources_df = parse_sources(context)
            if not sources_df.empty:
                st.dataframe(sources_df, use_container_width=True, hide_index=True)

    # Generate and display response
    with st.chat_message("assistant"):
        response = rag.query(
            prompt,
            param=QueryParam(mode=search_mode, response_type="Multiple Paragraphs"),
        )
        st.write(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display debug information
    with st.expander("🛠️ Debug Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Working Directory:", rag.working_dir)
        with col2:
            st.write("Search Mode:", search_mode)
        st.write("Raw Context:")
        st.code(context)
        st.write("Response Structure:")
        st.json(response)

# Display basic debug info outside the chat input block
with st.expander("System Information"):
    st.write(f"Working Directory: {rag.working_dir}")
    st.write(f"Current Search Mode: {search_mode}")
