Multimodal AI Chatbot with RAG

This project is a Streamlit-based chatbot that utilizes OpenAI GPT models, FAISS vector database, and LangChain to retrieve and process information from uploaded documents. It supports multimodal inputs, including text, tables, and images.

Features

Retrieves and processes document-based queries using RAG (Retrieval-Augmented Generation).

Supports text, tables, and images as sources of information.

Uses FAISS for vector-based similarity search.

Provides structured, verifiable responses with citations.

Implements document upload, processing, and querying functionalities.

Project Structure

|-- chat_interface.py    # Streamlit chat interface
|-- document_manager.py  # Document upload and management
|-- rag_backend.py       # Backend for RAG, vector search, and query execution
|-- .env                 # Environment variables
|-- requirements.txt     # Python dependencies
|-- data/                # Directory for uploaded documents and vector store

Installation

1. Clone the Repository

git clone https://github.com/yourrepo/multimodal-chatbot.git
cd multimodal-chatbot

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows

3. Install Dependencies

pip install -r requirements.txt

4. Set Up Environment Variables

Create a .env file with the following:

OPENAI_API_KEY="your_openai_api_key"

Usage

1. Run the Streamlit Applications

# Run the chat interface
streamlit run chat_interface.py

# Run the document manager separately
streamlit run document_manager.py

2. Upload Documents

PDFs and Excel files can be uploaded via the sidebar in the UI.

Processed data is stored in a FAISS vector database.

3. Ask Questions

Enter questions in the chat interface.

The chatbot will retrieve relevant document sections and generate a structured response.
