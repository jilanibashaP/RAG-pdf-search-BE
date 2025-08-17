# 🚀 AskPDF

A lightweight RAG (Retrieval-Augmented Generation) application that allows you to upload PDF files, extract their content, generate embeddings, and perform semantic search over them using OpenAI and Weaviate.

## 🛠️ Getting Started

### Prerequisites
- Node.js (v14 or higher)
- OpenAI API key
- Weaviate instance (local or cloud)

### Installation
```bash
npm install
npm start
```

The application will start running on **port 5000**.

## 📦 Features

- 📄 **PDF Upload & Indexing**: Upload and automatically index PDF documents
- 🔍 **Smart Search**: Perform semantic or keyword-based search across your documents
- 📊 **System Stats**: View detailed statistics about your document collection
- 📁 **File Management**: Browse and manage your uploaded files
- ❤️ **Health Monitoring**: Built-in health check endpoint for system monitoring

## 🔗 API Endpoints

### Core Functionality
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload PDF files for processing and indexing |
| `/search` | POST | Search through indexed documents using semantic or keyword queries |
| `/stats` | GET | Retrieve system statistics and document metrics |
| `/files` | GET | List all uploaded and indexed files |
| `/check` | GET | Health check endpoint to verify system status |

### Usage Examples

#### Upload a PDF
```bash
curl -X POST -F "file=@document.pdf" http://localhost:5000/upload
```

#### Search Documents
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query here"}' \
  http://localhost:5000/search
```

#### Get System Stats
```bash
curl http://localhost:5000/stats
```

#### List Files
```bash
curl http://localhost:5000/files
```

#### Health Check
```bash
curl http://localhost:5000/check
```

## ⚙️ Configuration

Make sure to set up your environment variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
WEAVIATE_URL=your_weaviate_instance_url
WEAVIATE_API_KEY=your_weaviate_api_key (if required)
PORT=5000
```

## 🏗️ Technology Stack

- **Backend**: Node.js
- **Vector Database**: Weaviate
- **AI/ML**: OpenAI Embeddings API
- **Document Processing**: PDF parsing libraries
- **Search**: Semantic similarity using vector embeddings

## 📝 How It Works

1. **Upload**: PDFs are uploaded via the `/upload` endpoint
2. **Processing**: Text is extracted from PDFs and split into chunks
3. **Embedding**: OpenAI generates vector embeddings for each text chunk
4. **Storage**: Embeddings and metadata are stored in Weaviate
5. **Search**: User queries are converted to embeddings and matched against stored documents
6. **Results**: Relevant document segments are returned with similarity scores

## 🚀 Next Steps

- Add support for more file formats (DOCX, TXT, etc.)
- Implement user authentication
- Add document management features (delete, update)
- Create a web interface for easier interaction
- Add batch processing capabilities