# ChatWithPDF

An interactive Streamlit application that allows users to upload PDF documents and engage in natural conversations about their content using advanced language models and document retrieval techniques.

## Features

- **PDF Document Processing**: Upload and process multiple PDF documents simultaneously
- **Interactive Chat Interface**: Ask questions about your documents and receive contextual answers
- **Smart Document Retrieval**: Utilizes FAISS for efficient similarity search and retrieval
- **Context Highlighting**: Automatically highlights relevant sections in the PDF that were used to generate answers
- **Chat History**: Maintains a record of all questions and answers during the session
- **Document Management**: Easy-to-use interface for uploading and processing documents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-chat-app.git
cd pdf-chat-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root and add your HuggingFace API key:
```
HUGGINGFACE_API_KEY=your_api_key_here
```

## Dependencies

- streamlit
- langchain
- langchain-community
- PyPDF2
- python-dotenv
- PyMuPDF (fitz)
- sentence-transformers
- faiss-cpu
- huggingface_hub

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload your PDF documents using the file uploader

4. Click "Process Documents" to initialize the chat system

5. Start asking questions about your documents in the chat interface

## How It Works

1. **Document Processing**:
   - Extracts text from uploaded PDFs
   - Splits text into manageable chunks
   - Creates embeddings using the `sentence-transformers` model
   - Stores embeddings in a FAISS vector database

2. **Question Answering**:
   - Uses Mistral-7B-Instruct-v0.2 model for generating responses
   - Retrieves relevant context from the vector store
   - Maintains conversation history for context-aware responses

3. **Visualization**:
   - Highlights relevant sections in the original PDF
   - Displays both the answer and supporting context
   - Shows chat history for reference

## Configuration

The application uses several configurable parameters:

- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Model: mistralai/Mistral-7B-Instruct-v0.2
- Maximum new tokens: 512
- Temperature: 0.7
- Number of retrieved documents: 3

## Technical Details

- **Text Splitting**: Uses RecursiveCharacterTextSplitter for intelligent text chunking
- **Embeddings**: Utilizes the all-mpnet-base-v2 model from sentence-transformers
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: Mistral-7B-Instruct-v0.2 through HuggingFace Hub
- **Memory**: ConversationBufferMemory for maintaining chat history
- **PDF Processing**: Combines PyPDF2 for text extraction and PyMuPDF for highlighting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with Streamlit
- Powered by Langchain
- Uses HuggingFace's models and hub
- Built on FAISS similarity search
