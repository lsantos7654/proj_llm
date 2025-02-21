"""Streamlit chat interface for document Q&A using Weaviate and OpenAI."""

import streamlit as st
import weaviate
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


# Initialize Weaviate connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        Weaviate client object
    """
    return weaviate.Client("http://localhost:8080")


def get_context(query: str, client, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        client: Weaviate client object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = (
        client.query.get(
            "DocumentChunk",
            [
                "text",
                "title",
                "summary",
                "sourceId",
                "sourceType",
            ],
        )
        .with_hybrid(query=query, properties=["text"], alpha=0.5)
        .with_additional(["score"])
        .with_limit(num_results)
        .do()
    )

    contexts = []

    for chunk in results["data"]["Get"]["DocumentChunk"]:
        # Extract metadata
        source_id = chunk.get("sourceId", "Unknown source")
        source_type = chunk.get("sourceType", "Unknown type")
        title = chunk.get("title", "Untitled section")
        summary = chunk.get("summary", "")
        text = chunk.get("text", "")

        # Build source citation
        source_info = f"\nSource: {source_id} ({source_type})"
        if title:
            source_info += f"\nTitle: {title}"
        if summary:
            source_info += f"\nSummary: {summary}"

        contexts.append(f"{text}{source_info}")

    return "\n\n".join(contexts)


def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are a helpful assistant that answers questions based on the
        provided context. Use only the information from the context to answer questions.
        If you're unsure or the context doesn't contain the relevant information,
        say so.

        Context:
        {context}
        """

    messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4-turbo-preview",  # Updated to latest model
        messages=messages_with_context,
        temperature=0.7,
        stream=True,
    )

    # Use Streamlit's built-in streaming capability
    response = st.write_stream(stream)
    return response


# Initialize Streamlit app
st.title("📚 Document Q&A")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
weaviate_client = init_db()

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

    # Get relevant context
    with st.status("Searching document...", expanded=False) as status:
        context = get_context(prompt, weaviate_client)
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")
        for chunk in context.split("\n\n"):
            # Split into text and metadata parts
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {
                line.split(": ")[0]: line.split(": ")[1]
                for line in parts[1:]
                if ": " in line
            }

            source = metadata.get("Source", "Unknown source")
            title = metadata.get("Title", "Untitled section")
            summary = metadata.get("Summary", "")

            summary_div = (
                f'<div class="metadata">Summary: {summary}</div>' if summary else ""
            )
            html_content = f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Title: {title}</div>
                        {summary_div}
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

    # Display assistant response
    with st.chat_message("assistant"):
        # Get model response with streaming
        response = get_chat_response(st.session_state.messages, context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
