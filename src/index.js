const fs = require('fs');
const path = require('path');
const pdf = require('pdf-parse');
const OpenAI = require('openai');
const weaviate = require('weaviate-ts-client').default;
const express = require('express');
const multer = require('multer');

// Import from searchMethods.js
const {
  client,
  openai,
  generateEmbedding,
  enhanceQuery,
  searchPDFContent,
  searchPDFContentAdvanced,
  smartSearchPDFContent,
  basicSearchPDFContent,
  fuseSearchResults
} = require('./utility');

// Load environment variables - adjust path if needed
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });

// Initialize Express app
const app = express();
app.use(express.json());

// add CORS headers
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  next();
});

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

if (!process.env.WEAVIATE_API_KEY) {
  console.error('❌ WEAVIATE_API_KEY environment variable is missing!');
  process.exit(1);
}

if (!process.env.OPENAI_API_KEY) {
  console.error('❌ OPENAI_API_KEY environment variable is missing!');
  process.exit(1);
}

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
    // here it should check the mime
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
function chunkTextWithOverlap(text, maxChunkSize = 1000, overlap = 100) {
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
    const chunks = chunkTextWithOverlap(text, 800, 200);

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

async function storePDFInWeaviateParallel(filePath, originalFilename) {
  try {
    const { text, numPages } = await extractPDFText(filePath);
    console.log(`Extracted ${numPages} pages from PDF: ${originalFilename}`);
    const chunks = chunkTextWithOverlap(text, 1000, 100);

    console.log(`Processing ${chunks.length} chunks from ${originalFilename}`);

    // Save PDF to user_uploads folder
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const savedFilename = `${timestamp}_${originalFilename}`;
    const permanentPath = path.join(USER_UPLOADS_DIR, savedFilename);

    fs.copyFileSync(filePath, permanentPath);
    console.log(`📄 PDF saved to: ${permanentPath}`);

    // Process with controlled concurrency
    const concurrencyLimit = 3; // Adjust based on your API limits
    const batchSize = 5;

    // Create batches
    const batches = [];
    for (let i = 0; i < chunks.length; i += batchSize) {
      batches.push(chunks.slice(i, i + batchSize));
    }

    let storedChunks = 0;

    // Process batches with limited concurrency
    const processBatch = async (batch, batchIndex) => {
      const objects = [];

      // Generate embeddings in parallel for this batch
      const embeddingPromises = batch.map(chunk => generateEmbedding(chunk.text));
      const embeddings = await Promise.all(embeddingPromises);

      // Create objects with embeddings
      batch.forEach((chunk, j) => {
        objects.push({
          class: 'PDFDocument',
          properties: {
            content: chunk.text,
            filename: originalFilename,
            savedFilename: savedFilename,
            pageNumber: Math.floor((chunk.start / text.length) * numPages) + 1,
            chunkIndex: batchIndex * batchSize + j,
            totalPages: numPages,
            totalChunks: chunks.length,
            chunkStart: chunk.start,
            chunkEnd: chunk.end,
            uploadDate: new Date().toISOString(),
            filePath: permanentPath
          },
          vector: embeddings[j]
        });
      });

      // Store batch in Weaviate
      await client.batch.objectsBatcher()
        .withObjects(...objects)
        .do();

      storedChunks += objects.length;
      console.log(`Stored ${storedChunks}/${chunks.length} chunks`);

      return objects.length;
    };

    // Process batches with controlled concurrency
    for (let i = 0; i < batches.length; i += concurrencyLimit) {
      const currentBatches = batches.slice(i, i + concurrencyLimit);
      const batchPromises = currentBatches.map((batch, idx) =>
        processBatch(batch, i + idx)
      );

      await Promise.all(batchPromises);

      // Small delay between concurrent batch groups if needed
      if (i + concurrencyLimit < batches.length) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
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
    // const result = await storePDFInWeaviate(req.file.path, req.file.originalname);
    const result = await storePDFInWeaviateParallel(req.file.path, req.file.originalname);
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
      error: error.message
    });
  }
});

// Search endpoint
app.post('/search', async (req, res) => {
  try {
    console.log('Search request received');
    const { query, limit = 5, searchType = 'hybrid' } = req.body;

    // Validate query input early
    if (!query || typeof query !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Query is required and must be a string.'
      });
    }

    // Enhance the user query using your enhancement function
    const enhancedQuery = await enhanceQuery(query);
    console.log(`Search query: "${query}", Enhanced query: "${enhancedQuery}", Limit: ${limit}, Search Type: ${searchType}`);

    let searchResults;

    // Determine which search strategy to use
    switch (searchType.toLowerCase()) {
      case 'basic':
        searchResults = await basicSearchPDFContent(enhancedQuery, limit);
        break;
      case 'advanced':
        searchResults = await searchPDFContentAdvanced(enhancedQuery, limit, 0.7);
        break;
      case 'smart':
        searchResults = await smartSearchPDFContent(enhancedQuery, limit);
        break;
      case 'hybrid':
      default:
        searchResults = await searchPDFContent(enhancedQuery, limit, 0.7);
        break;
    }

    console.log(`Search completed. Results found: ${searchResults ? searchResults.length : 0}`);

    // Debug: Log search results structure
    if (searchResults && searchResults.length > 0) {
      console.log('Sample search result:', JSON.stringify(searchResults[0], null, 2));
    }


    // const  = fuseSearchResults(searchResults, query);
    const finalSearchResults = searchResults.results.map(result => ({
      content: result.content,
      // filename: result.filename,
      // savedFilename: result.savedFilename,
      // pageNumber: result.pageNumber,
      // chunkIndex: result.chunkIndex,
      // totalPages: result.totalPages,
      // totalChunks: result.totalChunks,
      // chunkStart: result.chunkStart,
      // chunkEnd: result.chunkEnd,
      // uploadDate: result.uploadDate,
      // filePath: result.filePath
    }));


    // If search results exist, ask LLM to generate a response
    let llmResponse = null;

    if (finalSearchResults && finalSearchResults.length > 0) {
      try {
        console.log('Sending request to OpenAI...');

        const completion = await openai.chat.completions.create({
          model: 'gpt-4',
          messages: [
            {
              role: 'system',
              content: `You are an assistant that must answer strictly based on the content of the provided PDF document search results.
Do not use any external knowledge, assumptions, paraphrasing, or inferred logic.
Only use exact or clearly stated information from the search results.
If the answer is not explicitly present, respond with:
"Answer not found in the provided document."
Do not attempt to guess, expand, or provide helpful context beyond what is given.`
            },
            {
              role: 'user',
              content: `Original Query: "${query}"
Enhanced Query: "${enhancedQuery}"

Final Search Results: ${JSON.stringify(finalSearchResults, null, 2)}

Your task is to answer the user's question ONLY based on the provided context, i.e., the final search results. 
If the answer is not found in the provided context, you must respond with: "Answer not found in the provided document.". Please provide a concise and accurate answer.`
            }
          ],
          temperature: 0.7,
          max_tokens: 1000
        });

        console.log('OpenAI API response received');
        console.log('Completion object:', JSON.stringify(completion, null, 2));

        llmResponse = completion.choices?.[0]?.message?.content;

        if (!llmResponse) {
          console.log('LLM response is null or undefined');
          console.log('Choices array:', completion.choices);
        } else {
          console.log('LLM response length:', llmResponse.length);
        }

      } catch (llmError) {
        console.error('Error generating response from LLM:', llmError);
        console.error('LLM Error details:', {
          message: llmError.message,
          status: llmError.status,
          code: llmError.code
        });

        // Provide a fallback response
        llmResponse = `I found ${searchResults.length} relevant results for your query "${query}", but encountered an error while generating a comprehensive response. Please review the search results directly.`;
      }
    } else {
      console.log('No search results found, skipping LLM call');
      llmResponse = `No relevant results were found for your query "${query}". Try rephrasing your question or using different keywords.`;
    }


    // Respond with results
    return res.json({
      success: true,
      data: {
        originalQuery: query,
        enhancedQuery,
        // searchResults: searchResults || [],
        aiResponse: llmResponse,
        // resultsCount: searchResults ? searchResults.length : 0
      }
    });

  } catch (error) {
    console.error('Search endpoint error:', error);
    return res.status(500).json({
      success: false,
      error: 'An error occurred while processing the search request.',
      details: error.message
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


    console.log('Total documents:', JSON.stringify(totalResult, null, 2));

    const uniqueResult = await client.graphql
      .aggregate()
      .withClassName('PDFDocument')
      .withFields('filename { count }')
      .withGroupBy(['filename'])
      .do();


      console.log('Total uniqueResult:=====>', JSON.stringify(uniqueResult, null, 2));

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

// Health check route
app.get('/check', (req, res) => {
  res.json({
    success: true,
    message: 'API is running',
    timestamp: new Date().toISOString()
  });
});

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
      console.log(`📁 Files: GET /files`);
      console.log(`❤️ Health Check: GET /check`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the application
startServer();

module.exports = {
  app,
  client,
  openai,
  enhanceQuery,
  basicSearchPDFContent,
  searchPDFContentAdvanced,
  searchPDFContent,
  smartSearchPDFContent,
  validatePDFFile,
  extractPDFText,
  storePDFInWeaviate,
  chunkTextWithOverlap
};