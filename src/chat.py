import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx
import tempfile


# Initialize LightRAG with proper embedding configuration
# @st.cache_resource
def init_rag():
    return LightRAG(
        working_dir="lightrag_cache",
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


# CSS for styling
st.markdown(
    """
    <style>
    .search-result {
        margin: 10px 0;
        padding: 15px;
        border-radius: 5px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    .search-result summary {
        font-weight: 600;
        color: #1e88e5;
        cursor: pointer;
    }
    .search-result .metadata {
        color: #666;
        font-size: 0.9em;
        margin: 5px 0;
    }
    .edge-info {
        margin-top: 10px;
        padding-left: 10px;
        border-left: 3px solid #1e88e5;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize Streamlit app
st.title("📚 Document Q&A with Knowledge Graph")

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

    with st.status("Searching document...", expanded=False) as status:
        context = rag.query(
            prompt,
            param=QueryParam(
                mode=search_mode,
                response_type="Multiple Paragraphs",
                only_need_context=True,
                top_k=5,
            ),
        )
        st.write("Raw context_result content:")
        st.code(context)
    with st.chat_message("assistant"):
        response = rag.query(
            prompt,
            param=QueryParam(mode=search_mode, response_type="Multiple Paragraphs"),
        )
        st.write(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display debug information
    with st.expander("Debug Information"):
        st.write(f"Working Directory: {rag.working_dir}")
        st.write(f"Current Search Mode: {search_mode}")
        st.write("Context Result Structure:")
        st.json(response)

# Display basic debug info outside the chat input block
with st.expander("System Information"):
    st.write(f"Working Directory: {rag.working_dir}")
    st.write(f"Current Search Mode: {search_mode}")
