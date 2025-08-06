const fs = require('fs');
const path = require('path');
const pdf = require('pdf-parse');
const OpenAI = require('openai');
const weaviate = require('weaviate-ts-client').default;
const express = require('express');
const multer = require('multer');
const { fuseSearchResults,searchPDFContent,searchPDFContentAdvanced } = require('./searchMethodsCopy.js'); // Import custom fusion logic
import {openai} from "./searchMethodsCopy.js";

// Load environment variables - adjust path if needed
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });

// Initialize Express app
const app = express();
app.use(express.json());

// Configure multer for file uploads
const upload = multer({
  dest: 'uploads/',
  fileFilter: (req, file, cb) => {
    // Initial file type check
    if (file.mimetype === 'application/pdf' || path.extname(file.originalname).toLowerCase() === '.pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed!'), false);
    }
  },
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  }
});

// // Initialize Weaviate client - Update with your cluster details
// console.log('Environment check:');
// console.log('WEAVIATE_HOST:', process.env.WEAVIATE_HOST);
// console.log('WEAVIATE_API_KEY:', process.env.WEAVIATE_API_KEY ? 'Set' : 'Not set');
// console.log('OPENAI_API_KEY:', process.env.OPENAI_API_KEY ? 'Set' : 'Not set');

// if (!process.env.WEAVIATE_HOST) {
//   console.error('❌ WEAVIATE_HOST environment variable is missing!');
//   console.log('Please check your .env file contains:');
//   console.log('WEAVIATE_HOST=5i7dishidkoccpa0jrq.c0.asia-southeast1.gcp.weaviate.cloud');
//   process.exit(1);
// }

if (!process.env.WEAVIATE_API_KEY) {
  console.error('❌ WEAVIATE_API_KEY environment variable is missing!');
  process.exit(1);
}

if (!process.env.OPENAI_API_KEY) {
  console.error('❌ OPENAI_API_KEY environment variable is missing!');
  process.exit(1);
}

const client = weaviate.client({
  scheme: 'https', // Use https for cloud instances
  host: process.env.WEAVIATE_HOST, // Your REST endpoint without https://
  apiKey: new weaviate.ApiKey(process.env.WEAVIATE_API_KEY), // Your API key
});

// Initialize OpenAI client
// const openai = new OpenAI({
//   apiKey: process.env.OPENAI_API_KEY,
// });

// Define schema for PDF documents
const pdfSchema = {
  class: 'PDFDocument',
  description: 'PDF document chunks with vector embeddings',
  vectorizer: 'none', // Manual vectorization
  properties: [
    { name: 'content', dataType: ['text'], description: 'Chunk text content' },
    { name: 'filename', dataType: ['string'], description: 'Original filename' },
    { name: 'savedFilename', dataType: ['string'], description: 'Saved filename with timestamp' },
    { name: 'pageNumber', dataType: ['int'], description: 'Page number' },
    { name: 'chunkIndex', dataType: ['int'], description: 'Chunk index' },
    { name: 'totalPages', dataType: ['int'], description: 'Total pages in document' },
    { name: 'totalChunks', dataType: ['int'], description: 'Total chunks in document' },
    { name: 'chunkStart', dataType: ['int'], description: 'Chunk start position' },
    { name: 'chunkEnd', dataType: ['int'], description: 'Chunk end position' },
    { name: 'uploadDate', dataType: ['date'], description: 'Upload timestamp' },
    { name: 'filePath', dataType: ['string'], description: 'Local file path' },
  ],
};

// 1. File validation function
function validatePDFFile(filePath, originalName) {
  try {
    // Check file extension
    const ext = path.extname(originalName).toLowerCase();
    if (ext !== '.pdf') {
      throw new Error('Invalid file extension. Only PDF files are supported.');
    }

    // Check if file exists
    if (!fs.existsSync(filePath)) {
      throw new Error('Uploaded file not found.');
    }

    // Check file size (additional validation)
    const stats = fs.statSync(filePath);
    if (stats.size === 0) {
      throw new Error('Empty PDF file.');
    }

    // Try to read the PDF header
    const buffer = fs.readFileSync(filePath, { start: 0, end: 4 });
    const header = buffer.toString('ascii');
    if (!header.startsWith('%PDF')) {
      throw new Error('Invalid PDF file format.');
    }

    return true;
  } catch (error) {
    throw new Error(`PDF validation failed: ${error.message}`);
  }
}

// 2. Text chunking with boundary overlap
function chunkTextWithOverlap(text, maxChunkSize = 1000, overlap = 200) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    let end = Math.min(start + maxChunkSize, text.length);

    // Find a good break point at sentence boundaries
    if (end < text.length) {
      const sentenceEnds = ['.', '!', '?', '\n'];
      let bestBreak = -1;

      for (let i = end; i > start + maxChunkSize * 0.5; i--) {
        if (sentenceEnds.includes(text[i])) {
          bestBreak = i + 1;
          break;
        }
      }

      if (bestBreak > -1) {
        end = bestBreak;
      }
    }

    const chunkText = text.slice(start, end).trim();
    if (chunkText.length > 50) { // Only include meaningful chunks
      chunks.push({
        text: chunkText,
        start,
        end
      });
    }

    // Move start position with overlap
    start = Math.max(start + 1, end - overlap);
    if (start >= text.length) break;
  }

  return chunks;
}

// 3. Generate embeddings using OpenAI
async function generateEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small', // Updated model
      input: text.substring(0, 8000), // Limit input length
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    throw new Error(`Embedding generation failed: ${error.message}`);
  }
}

// Extract text from PDF
async function extractPDFText(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const data = await pdf(buffer);
    return {
      text: data.text,
      numPages: data.numpages
    };
  } catch (error) {
    throw new Error(`PDF text extraction failed: ${error.message}`);
  }
}

// Create user_uploads directory if it doesn't exist
const USER_UPLOADS_DIR = './user_uploads';
if (!fs.existsSync(USER_UPLOADS_DIR)) {
  fs.mkdirSync(USER_UPLOADS_DIR, { recursive: true });
  console.log('📁 Created user_uploads directory');
}

// 4. Store PDF vectors in Weaviate and save file locally
async function storePDFInWeaviate(filePath, originalFilename) {
  try {
    const { text, numPages } = await extractPDFText(filePath);
    console.log(`Extracted ${numPages} pages from PDF: ${originalFilename}`);
    const chunks = chunkTextWithOverlap(text, 800, 150);

    console.log(`Processing ${chunks.length} chunks from ${originalFilename}`);

    // Save PDF to user_uploads folder
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const savedFilename = `${timestamp}_${originalFilename}`;
    const permanentPath = path.join(USER_UPLOADS_DIR, savedFilename);

    // Copy file to permanent location
    fs.copyFileSync(filePath, permanentPath);
    console.log(`📄 PDF saved to: ${permanentPath}`);

    // Process in smaller batches to handle rate limits
    const batchSize = 5;
    let storedChunks = 0;

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      const objects = [];

      for (let j = 0; j < batch.length; j++) {
        const chunk = batch[j];
        const embedding = await generateEmbedding(chunk.text);

        objects.push({
          class: 'PDFDocument',
          properties: {
            content: chunk.text,
            filename: originalFilename,
            savedFilename: savedFilename, // Store the saved filename with timestamp
            pageNumber: Math.floor((chunk.start / text.length) * numPages) + 1,
            chunkIndex: i + j,
            totalPages: numPages,
            totalChunks: chunks.length,
            chunkStart: chunk.start,
            chunkEnd: chunk.end,
            uploadDate: new Date().toISOString(),
            filePath: permanentPath // Store the full path for future reference
          },
          vector: embedding
        });
      }

      await client.batch.objectsBatcher()
        .withObjects(...objects)
        .do();

      storedChunks += objects.length;
      console.log(`Stored ${storedChunks}/${chunks.length} chunks`);

      // Rate limiting delay
      await new Promise(resolve => setTimeout(resolve, 200));
    }

    return {
      success: true,
      chunksStored: chunks.length,
      filename: originalFilename,
      savedFilename: savedFilename,
      savedPath: permanentPath,
      pages: numPages
    };
  } catch (error) {
    throw new Error(`Failed to store PDF: ${error.message}`);
  }
}

// 5. Query processing with LLM enhancement
async function enhanceQuery(userQuery) {
  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        {
          role: 'system',
          content: 'You are a helpful assistant that enhances search queries for document retrieval. Transform the user query into a more specific and detailed search query that would help find relevant information in documents. Keep it concise but comprehensive.'
        },
        {
          role: 'user',
          content: `Enhance this search query for better document retrieval: "${userQuery}"`
        }
      ],
      max_tokens: 100,
      temperature: 0.3
    });

    return response.choices[0].message.content.trim();
  } catch (error) {
    console.log('Query enhancement failed, using original query:', error.message);
    return userQuery; // Fallback to original query
  }
}

// Search function
async function basicSearchPDFContent(query, limit = 5) {
  try {
    // Enhance the query using LLM
    const enhancedQuery = await enhanceQuery(query);
    console.log(`Original query: "${query}"`);
    console.log(`Enhanced query: "${enhancedQuery}"`);

    // Generate embedding for the enhanced query
    const queryEmbedding = await generateEmbedding(enhancedQuery);
    console.log('Generated embedding for query====>', queryEmbedding);

    // const result = await client.graphql
    //   .get()
    //   .withClassName('PDFDocument')
    //   .withFields('content filename savedFilename pageNumber chunkIndex totalPages uploadDate filePath')
    //   .withNearVector({ vector: queryEmbedding })
    //   .withLimit(limit)
    //   .withAdditional(['certainty', 'distance'])
    //   .do();


    const result = await client.graphql
      .get()
      .withClassName('PDFDocument')
      .withFields('content filename savedFilename pageNumber chunkIndex totalPages uploadDate filePath _additional { certainty distance }')
      .withNearVector({ vector: queryEmbedding })
      .withLimit(limit)
      .do();

    // console.log('Search results:', result.data);
    // stringify the result for better readability
    console.log('Search results:', JSON.stringify(result.data, null, 2));

    return {
      originalQuery: query,
      enhancedQuery: enhancedQuery,
      results: result.data.Get.PDFDocument || []
    };
  } catch (error) {
    throw new Error(`Search failed: ${error.message}`);
  }
}

// Initialize schema
async function initializeSchema() {
  try {
    const schema = await client.schema.getter().do();
    const classExists = schema.classes.some(cls => cls.class === 'PDFDocument');

    if (!classExists) {
      await client.schema.classCreator().withClass(pdfSchema).do();
      console.log('✅ Schema created successfully');
    } else {
      console.log('ℹ️ Schema already exists');
    }
  } catch (error) {
    console.error('Schema initialization failed:', error);
    throw error;
  }
}

// Routes

// Upload and process PDF
app.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    console.log('Upload request received');
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded or invalid file type'
      });
    }

    // Validate PDF file
    validatePDFFile(req.file.path, req.file.originalname);
    console.log(`📥 Received file: ${req.file.originalname}`);
    // Process and store in Weaviate
    const result = await storePDFInWeaviate(req.file.path, req.file.originalname);
    console.log(`✅ PDF processed: ${result.savedFilename} with ${result.chunksStored} chunks`);
    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({
      success: true,
      message: 'PDF processed and stored successfully',
      data: result
    });

  } catch (error) {

    // Clean up file if it exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(400).json({
      success: false,
      error: error
    });
  }
});

// Search route
app.post('/search', async (req, res) => {
  try {
    console.log('Search request received');
    const { query, limit = 5 } = req.body;

    if (!query) {
      return res.status(400).json({
        success: false,
        error: 'Query is required'
      });
    }
    // Basic search
    // const searchResults = await basicSearchPDFContent(query, limit);

    // search reaults with hybrid approach
    // ------------------------------------------------------------------
    // Basic hybrid search (70% vector, 30% keyword)
    const results1 = await searchPDFContent(query, 10, 0.7);

    // Keyword-focused search for specific terms
    // const results2 = await searchPDFContent("API key configuration", 10, 0.3);

    // Smart search that auto-adjusts parameters
    // const results3 = await smartSearchPDFContent("What are the benefits of cloud computing?");

    res.json({
      success: true,
      data: results1
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get document statistics
app.get('/stats', async (req, res) => {
  try {
    const totalResult = await client.graphql
      .aggregate()
      .withClassName('PDFDocument')
      .withFields('meta { count }')
      .do();

    const uniqueResult = await client.graphql
      .aggregate()
      .withClassName('PDFDocument')
      .withFields('filename { count }')
      .withGroupBy(['filename'])
      .do();

    res.json({
      success: true,
      data: {
        totalChunks: totalResult.data.Aggregate.PDFDocument?.[0]?.meta?.count || 0,
        uniqueFiles: uniqueResult.data.Aggregate.PDFDocument?.length || 0,
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.get('/check', (req, res) => {
  res.json({
    success: true,
    message: 'API is running',
  });
})
// List uploaded files
app.get('/files', async (req, res) => {
  try {
    // Get files from local directory
    const localFiles = fs.readdirSync(USER_UPLOADS_DIR).map(filename => {
      const filePath = path.join(USER_UPLOADS_DIR, filename);
      const stats = fs.statSync(filePath);
      return {
        filename,
        size: stats.size,
        uploadDate: stats.birthtime.toISOString(),
        path: filePath
      };
    });

    // Get unique files from Weaviate
    const weaviateResult = await client.graphql
      .get()
      .withClassName('PDFDocument')
      .withFields('filename savedFilename uploadDate totalPages')
      .withLimit(1000)
      .do();

    const weaviateFiles = [];
    const seen = new Set();

    if (weaviateResult.data.Get.PDFDocument) {
      weaviateResult.data.Get.PDFDocument.forEach(doc => {
        if (!seen.has(doc.filename)) {
          seen.add(doc.filename);
          weaviateFiles.push({
            originalFilename: doc.filename,
            savedFilename: doc.savedFilename,
            uploadDate: doc.uploadDate,
            totalPages: doc.totalPages
          });
        }
      });
    }

    res.json({
      success: true,
      data: {
        localFiles,
        processedFiles: weaviateFiles,
        totalLocalFiles: localFiles.length,
        totalProcessedFiles: weaviateFiles.length
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});


// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        error: 'File too large. Maximum size is 50MB.'
      });
    }
  }

  res.status(400).json({
    success: false,
    error: error.message
  });
});

// Start server
async function startServer() {
  try {
    await initializeSchema();

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`🚀 Graviti Reg Search running on port ${PORT}`);
      console.log(`📄 Upload PDFs: POST /upload`);
      console.log(`🔍 Search: POST /search`);
      console.log(`📊 Stats: GET /stats`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}
// Start the application
startServer();

// module.exports = {
//   app,
//   client,
//   openai,
//   enhanceQuery,
//   basicSearchPDFContent,
//   searchPDFContentAdvanced
// };