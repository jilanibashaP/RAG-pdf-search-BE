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

// Load environment variables
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });

// Initialize Express app
const app = express();
app.use(express.json());

// CORS headers
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

// Environment validation
if (!process.env.WEAVIATE_API_KEY || !process.env.OPENAI_API_KEY) {
  console.error('❌ Missing required environment variables!');
  process.exit(1);
}

// Enhanced schema with metadata for better ranking
const pdfSchema = {
  class: 'PDFDocument',
  description: 'PDF document chunks with vector embeddings and enhanced metadata',
  vectorizer: 'none',
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
    // Enhanced metadata for better ranking
    { name: 'documentType', dataType: ['string'], description: 'Document type/category' },
    { name: 'keywords', dataType: ['string[]'], description: 'Extracted keywords' },
    { name: 'summary', dataType: ['text'], description: 'Chunk summary' },
    { name: 'importance', dataType: ['number'], description: 'Content importance score' },
    { name: 'wordCount', dataType: ['int'], description: 'Word count in chunk' },
  ],
};

// Enhanced text chunking with smart boundaries and metadata extraction
function enhancedChunkTextWithOverlap(text, maxChunkSize = 1000, overlap = 200) {
  const chunks = [];
  let start = 0;

  // Extract potential keywords and important phrases
  const extractKeywords = (text) => {
    const words = text.toLowerCase().match(/\b\w{4,}\b/g) || [];
    const freq = {};
    words.forEach(word => freq[word] = (freq[word] || 0) + 1);
    return Object.entries(freq)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([word]) => word);
  };

  // Calculate importance score based on various factors
  const calculateImportance = (text) => {
    let score = 0;
    
    // Headers and titles (ALL CAPS, numbered sections)
    if (/^[A-Z\s\d\.]{10,}$/m.test(text)) score += 0.3;
    
    // Numbers and data
    if (/\d+%|\$\d+|\d+\.\d+/.test(text)) score += 0.2;
    
    // Keywords indicating importance
    const importantWords = ['important', 'key', 'significant', 'critical', 'essential', 'conclusion', 'summary'];
    importantWords.forEach(word => {
      if (text.toLowerCase().includes(word)) score += 0.1;
    });
    
    // Questions and answers
    if (/\?/.test(text)) score += 0.1;
    
    return Math.min(score, 1.0);
  };

  while (start < text.length) {
    let end = Math.min(start + maxChunkSize, text.length);

    // Find better break points
    if (end < text.length) {
      const breakPoints = ['\n\n', '. ', '! ', '? ', '\n'];
      let bestBreak = -1;

      for (const breakPoint of breakPoints) {
        const breakIndex = text.lastIndexOf(breakPoint, end);
        if (breakIndex > start + maxChunkSize * 0.5) {
          bestBreak = breakIndex + breakPoint.length;
          break;
        }
      }

      if (bestBreak > -1) {
        end = bestBreak;
      }
    }

    const chunkText = text.slice(start, end).trim();
    if (chunkText.length > 50) {
      const keywords = extractKeywords(chunkText);
      const importance = calculateImportance(chunkText);
      const wordCount = chunkText.split(/\s+/).length;
      
      chunks.push({
        text: chunkText,
        start,
        end,
        keywords,
        importance,
        wordCount,
        summary: chunkText.length > 200 ? chunkText.substring(0, 200) + '...' : chunkText
      });
    }

    start = Math.max(start + 1, end - overlap);
    if (start >= text.length) break;
  }

  return chunks;
}

// Advanced result ranking system
class ResultRanker {
  static calculateRelevanceScore(result, query, queryEmbedding) {
    let score = result._additional?.distance ? (1 - result._additional.distance) : 0.5;
    
    // Boost based on keyword matches
    const queryWords = query.toLowerCase().split(/\s+/);
    const contentWords = result.content.toLowerCase();
    const keywordMatches = queryWords.filter(word => contentWords.includes(word)).length;
    score += (keywordMatches / queryWords.length) * 0.3;
    
    // Boost based on importance
    if (result.importance) {
      score += result.importance * 0.2;
    }
    
    // Boost for exact phrase matches
    if (contentWords.includes(query.toLowerCase())) {
      score += 0.3;
    }
    
    // Penalize very short or very long chunks
    const wordCount = result.wordCount || result.content.split(/\s+/).length;
    if (wordCount < 20 || wordCount > 500) {
      score *= 0.8;
    }
    
    return Math.min(score, 1.0);
  }

  static diversifyResults(results, maxSimilarity = 0.8) {
    if (results.length <= 1) return results;
    
    const diversified = [results[0]]; // Always include the top result
    
    for (let i = 1; i < results.length; i++) {
      const candidate = results[i];
      let shouldInclude = true;
      
      // Check similarity with already selected results
      for (const selected of diversified) {
        const similarity = this.calculateTextSimilarity(candidate.content, selected.content);
        if (similarity > maxSimilarity) {
          shouldInclude = false;
          break;
        }
      }
      
      if (shouldInclude) {
        diversified.push(candidate);
      }
    }
    
    return diversified;
  }

  static calculateTextSimilarity(text1, text2) {
    const words1 = new Set(text1.toLowerCase().match(/\b\w+\b/g) || []);
    const words2 = new Set(text2.toLowerCase().match(/\b\w+\b/g) || []);
    
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
  }
}

// Multi-document synthesis class
class DocumentSynthesizer {
  static async synthesizeResults(results, query, maxTokens = 2000) {
    if (!results || results.length === 0) return null;

    // Group results by document
    const docGroups = {};
    results.forEach(result => {
      const docKey = result.filename || 'unknown';
      if (!docGroups[docKey]) {
        docGroups[docKey] = [];
      }
      docGroups[docKey].push(result);
    });

    // Create synthesis prompt
    const synthesis = Object.entries(docGroups).map(([filename, chunks]) => {
      const content = chunks.map(chunk => chunk.content).join('\n\n');
      return `Document: ${filename}\nContent: ${content.substring(0, 1000)}...`;
    }).join('\n\n---\n\n');

    try {
      const completion = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: `You are an expert document analyst. Synthesize information from multiple document excerpts to provide a comprehensive answer. 
            
            Rules:
            1. Combine related information from different documents
            2. Identify and resolve any contradictions
            3. Highlight information that appears in multiple sources
            4. Note when information comes from specific documents
            5. If no relevant information is found, state this clearly
            6. Provide citations by mentioning document names`
          },
          {
            role: 'user',
            content: `Query: "${query}"

Multiple Document Excerpts:
${synthesis}

Please provide a comprehensive synthesis that answers the query using information from all relevant documents. Include document citations where appropriate.`
          }
        ],
        temperature: 0.3,
        max_tokens: maxTokens
      });

      return completion.choices?.[0]?.message?.content;
    } catch (error) {
      console.error('Synthesis error:', error);
      return null;
    }
  }
}

// Query refinement system
class QueryRefiner {
  static async generateQueryVariations(originalQuery) {
    try {
      const completion = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'Generate 3-5 alternative phrasings of the given query to improve search results. Focus on synonyms, different perspectives, and more specific terms.'
          },
          {
            role: 'user',
            content: `Original query: "${originalQuery}"\n\nGenerate alternative queries:`
          }
        ],
        temperature: 0.7,
        max_tokens: 200
      });

      const variations = completion.choices?.[0]?.message?.content
        ?.split('\n')
        .filter(line => line.trim())
        .map(line => line.replace(/^\d+\.\s*/, '').trim())
        .slice(0, 5) || [];

      return [originalQuery, ...variations];
    } catch (error) {
      console.error('Query refinement error:', error);
      return [originalQuery];
    }
  }

  static async expandQuery(query) {
    try {
      const completion = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'Expand the given query with related terms, synonyms, and context that might help find relevant information in documents.'
          },
          {
            role: 'user',
            content: `Expand this query with related terms: "${query}"`
          }
        ],
        temperature: 0.5,
        max_tokens: 100
      });

      return completion.choices?.[0]?.message?.content?.trim() || query;
    } catch (error) {
      console.error('Query expansion error:', error);
      return query;
    }
  }
}

// Enhanced search function with all improvements
async function enhancedSearchPDFContent(query, options = {}) {
  const {
    limit = 10,
    maxResults = 20,
    enableRanking = true,
    enableDiversification = true,
    enableSynthesis = true,
    enableQueryRefinement = true,
    searchType = 'hybrid'
  } = options;

  try {
    console.log(`Enhanced search: "${query}" with options:`, options);
    
    let allResults = [];
    let queryVariations = [query];

    // Query refinement
    if (enableQueryRefinement) {
      queryVariations = await QueryRefiner.generateQueryVariations(query);
      console.log('Query variations:', queryVariations);
    }

    // Search with multiple query variations
    for (const searchQuery of queryVariations.slice(0, 3)) { // Limit to first 3 variations
      let searchResults;

      switch (searchType) {
        case 'basic':
          searchResults = await basicSearchPDFContent(searchQuery, Math.ceil(maxResults / queryVariations.length));
          break;
        case 'advanced':
          searchResults = await searchPDFContentAdvanced(searchQuery, Math.ceil(maxResults / queryVariations.length), 0.7);
          break;
        case 'smart':
          searchResults = await smartSearchPDFContent(searchQuery, Math.ceil(maxResults / queryVariations.length));
          break;
        case 'hybrid':
        default:
          searchResults = await searchPDFContent(searchQuery, Math.ceil(maxResults / queryVariations.length), 0.7);
          break;
      }

      if (searchResults && searchResults.results) {
        allResults.push(...searchResults.results);
      }
    }

    // Remove duplicates based on content similarity
    const uniqueResults = [];
    const seenContent = new Set();

    for (const result of allResults) {
      const contentHash = result.content.substring(0, 100);
      if (!seenContent.has(contentHash)) {
        seenContent.add(contentHash);
        uniqueResults.push(result);
      }
    }

    // Enhanced ranking
    if (enableRanking && uniqueResults.length > 0) {
      const queryEmbedding = await generateEmbedding(query);
      
      uniqueResults.forEach(result => {
        result.relevanceScore = ResultRanker.calculateRelevanceScore(result, query, queryEmbedding);
      });

      uniqueResults.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));
    }

    // Result diversification
    let finalResults = uniqueResults;
    if (enableDiversification) {
      finalResults = ResultRanker.diversifyResults(uniqueResults.slice(0, maxResults));
    }

    // Limit final results
    finalResults = finalResults.slice(0, limit);

    // Multi-document synthesis
    let synthesis = null;
    if (enableSynthesis && finalResults.length > 1) {
      synthesis = await DocumentSynthesizer.synthesizeResults(finalResults, query);
    }

    return {
      results: finalResults,
      synthesis,
      metadata: {
        totalFound: uniqueResults.length,
        queryVariations: queryVariations.length,
        searchType,
        options
      }
    };

  } catch (error) {
    console.error('Enhanced search error:', error);
    throw error;
  }
}

// File validation function
function validatePDFFile(filePath, originalName) {
  try {
    const ext = path.extname(originalName).toLowerCase();
    if (ext !== '.pdf') {
      throw new Error('Invalid file extension. Only PDF files are supported.');
    }

    if (!fs.existsSync(filePath)) {
      throw new Error('Uploaded file not found.');
    }

    const stats = fs.statSync(filePath);
    if (stats.size === 0) {
      throw new Error('Empty PDF file.');
    }

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

// Create user_uploads directory
const USER_UPLOADS_DIR = './user_uploads';
if (!fs.existsSync(USER_UPLOADS_DIR)) {
  fs.mkdirSync(USER_UPLOADS_DIR, { recursive: true });
  console.log('📁 Created user_uploads directory');
}

// Enhanced PDF storage with metadata
async function enhancedStorePDFInWeaviate(filePath, originalFilename) {
  try {
    const { text, numPages } = await extractPDFText(filePath);
    console.log(`Extracted ${numPages} pages from PDF: ${originalFilename}`);
    
    const chunks = enhancedChunkTextWithOverlap(text, 800, 200);
    console.log(`Processing ${chunks.length} chunks from ${originalFilename}`);

    // Save PDF to user_uploads folder
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const savedFilename = `${timestamp}_${originalFilename}`;
    const permanentPath = path.join(USER_UPLOADS_DIR, savedFilename);

    fs.copyFileSync(filePath, permanentPath);
    console.log(`📄 PDF saved to: ${permanentPath}`);

    // Detect document type
    const detectDocumentType = (text) => {
      const lowerText = text.toLowerCase();
      if (lowerText.includes('contract') || lowerText.includes('agreement')) return 'legal';
      if (lowerText.includes('financial') || lowerText.includes('budget')) return 'financial';
      if (lowerText.includes('technical') || lowerText.includes('specification')) return 'technical';
      if (lowerText.includes('research') || lowerText.includes('study')) return 'research';
      return 'general';
    };

    const documentType = detectDocumentType(text);

    // Process with enhanced metadata
    const concurrencyLimit = 3;
    const batchSize = 5;
    const batches = [];
    
    for (let i = 0; i < chunks.length; i += batchSize) {
      batches.push(chunks.slice(i, i + batchSize));
    }

    let storedChunks = 0;

    const processBatch = async (batch, batchIndex) => {
      const objects = [];
      const embeddingPromises = batch.map(chunk => generateEmbedding(chunk.text));
      const embeddings = await Promise.all(embeddingPromises);

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
            filePath: permanentPath,
            // Enhanced metadata
            documentType,
            keywords: chunk.keywords,
            summary: chunk.summary,
            importance: chunk.importance,
            wordCount: chunk.wordCount
          },
          vector: embeddings[j]
        });
      });

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
      pages: numPages,
      documentType
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
      console.log('✅ Enhanced schema created successfully');
    } else {
      console.log('ℹ️ Schema already exists');
    }
  } catch (error) {
    console.error('Schema initialization failed:', error);
    throw error;
  }
}

// ROUTES

// Enhanced upload route
app.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    console.log('Upload request received');
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded or invalid file type'
      });
    }

    validatePDFFile(req.file.path, req.file.originalname);
    console.log(`📥 Received file: ${req.file.originalname}`);

    const result = await enhancedStorePDFInWeaviate(req.file.path, req.file.originalname);
    console.log(`✅ PDF processed: ${result.savedFilename} with ${result.chunksStored} chunks`);

    fs.unlinkSync(req.file.path);

    res.json({
      success: true,
      message: 'PDF processed and stored successfully with enhanced metadata',
      data: result
    });

  } catch (error) {
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(400).json({
      success: false,
      error: error.message
    });
  }
});

// Enhanced search route
app.post('/search', async (req, res) => {
  try {
    console.log('Enhanced search request received');
    const {
      query,
      limit = 10,
      maxResults = 20,
      searchType = 'hybrid',
      enableRanking = true,
      enableDiversification = true,
      enableSynthesis = true,
      enableQueryRefinement = true
    } = req.body;

    if (!query || typeof query !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Query is required and must be a string.'
      });
    }

    const searchOptions = {
      limit,
      maxResults,
      enableRanking,
      enableDiversification,
      enableSynthesis,
      enableQueryRefinement,
      searchType
    };

    console.log(`Enhanced search: "${query}"`, searchOptions);

    const enhancedResults = await enhancedSearchPDFContent(query, searchOptions);

    // Generate AI response
    let aiResponse = null;
    
    if (enhancedResults.results && enhancedResults.results.length > 0) {
      try {
        const contextContent = enhancedResults.results
          .slice(0, 5) // Use top 5 results for context
          .map(result => ({
            content: result.content,
            filename: result.filename,
            pageNumber: result.pageNumber,
            relevanceScore: result.relevanceScore
          }));

        const completion = await openai.chat.completions.create({
          model: 'gpt-4',
          messages: [
            {
              role: 'system',
              content: `You are an intelligent document assistant. Use the provided search results to answer questions accurately and comprehensively.
              
              Guidelines:
              1. Base your response primarily on the search results
              2. If synthesis is available, use it as additional context
              3. Mention document sources when relevant
              4. If information is incomplete, acknowledge this
              5. Provide specific, actionable answers when possible
              6. Use relevance scores to prioritize information`
            },
            {
              role: 'user',
              content: `Query: "${query}"

Search Results: ${JSON.stringify(contextContent, null, 2)}

${enhancedResults.synthesis ? `Multi-Document Synthesis: ${enhancedResults.synthesis}` : ''}

Please provide a comprehensive answer based on this information.`
            }
          ],
          temperature: 0.3,
          max_tokens: 1500
        });

        aiResponse = completion.choices?.[0]?.message?.content;

      } catch (llmError) {
        console.error('LLM response error:', llmError);
        aiResponse = enhancedResults.synthesis || 
          `Found ${enhancedResults.results.length} relevant results for your query "${query}". Please review the search results for detailed information.`;
      }
    } else {
      aiResponse = `No relevant results found for your query "${query}". Try rephrasing your question or using different keywords.`;
    }

    return res.json({
      success: true,
      data: {
        query,
        aiResponse,
        synthesis: enhancedResults.synthesis,
        results: enhancedResults.results.map(result => ({
          content: result.content.substring(0, 500) + (result.content.length > 500 ? '...' : ''),
          filename: result.filename,
          pageNumber: result.pageNumber,
          relevanceScore: result.relevanceScore,
          importance: result.importance
        })),
        metadata: enhancedResults.metadata
      }
    });

  } catch (error) {
    console.error('Enhanced search endpoint error:', error);
    return res.status(500).json({
      success: false,
      error: 'An error occurred while processing the enhanced search request.',
      details: error.message
    });
  }
});

// Query refinement endpoint
app.post('/refine-query', async (req, res) => {
  try {
    const { query } = req.body;
    
    if (!query) {
      return res.status(400).json({
        success: false,
        error: 'Query is required'
      });
    }

    const variations = await QueryRefiner.generateQueryVariations(query);
    const expanded = await QueryRefiner.expandQuery(query);

    res.json({
      success: true,
      data: {
        originalQuery: query,
        variations,
        expanded
      }
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Document statistics with enhanced metadata
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

    const typeResult = await client.graphql
      .aggregate()
      .withClassName('PDFDocument')
      .withFields('documentType { count }')
      .withGroupBy(['documentType'])
      .do();

    res.json({
      success: true,
      data: {
        totalChunks: totalResult.data.Aggregate.PDFDocument?.[0]?.meta?.count || 0,
        uniqueFiles: uniqueResult.data.Aggregate.PDFDocument?.length || 0,
        documentTypes: typeResult.data.Aggregate.PDFDocument || []
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Health check
app.get('/check', (req, res) => {
  res.json({
    success: true,
    message: 'Enhanced PDF Search API is running',
    timestamp: new Date().toISOString(),
    features: [
      'Enhanced result ranking',
      'Multi-document synthesis',
      'Query refinement',
      'Result diversification',
      'Smart metadata extraction'
    ]
  });
});

// List files with enhanced metadata
app.get('/files', async (req, res) => {
  try {
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

    const weaviateResult = await client.graphql
      .get()
      .withClassName('PDFDocument')
      .withFields('filename savedFilename uploadDate totalPages documentType')
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
            totalPages: doc.totalPages,
            documentType: doc.documentType
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

// Advanced search analytics endpoint
app.get('/search-analytics', async (req, res) => {
  try {
    const { timeframe = '7d' } = req.query;
    
    // This would typically connect to a search analytics database
    // For demo purposes, we'll return sample analytics
    const analytics = {
      totalSearches: 156,
      avgResultsPerSearch: 8.3,
      topQueries: [
        { query: "contract terms", count: 23 },
        { query: "financial data", count: 18 },
        { query: "technical specifications", count: 15 }
      ],
      searchTypes: {
        hybrid: 67,
        smart: 45,
        advanced: 32,
        basic: 12
      },
      avgResponseTime: "1.2s",
      synthesisUsage: "78%"
    };

    res.json({
      success: true,
      data: analytics,
      timeframe
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Batch search endpoint for multiple queries
app.post('/batch-search', async (req, res) => {
  try {
    const { queries, options = {} } = req.body;
    
    if (!Array.isArray(queries) || queries.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Queries array is required and must not be empty'
      });
    }

    if (queries.length > 10) {
      return res.status(400).json({
        success: false,
        error: 'Maximum 10 queries allowed per batch'
      });
    }

    console.log(`Processing batch search for ${queries.length} queries`);

    const batchResults = [];
    const defaultOptions = {
      limit: 5,
      maxResults: 10,
      enableRanking: true,
      enableDiversification: true,
      enableSynthesis: false, // Disable for batch to save tokens
      enableQueryRefinement: false,
      searchType: 'hybrid'
    };

    const searchOptions = { ...defaultOptions, ...options };

    for (const query of queries) {
      try {
        const result = await enhancedSearchPDFContent(query, searchOptions);
        batchResults.push({
          query,
          success: true,
          results: result.results.slice(0, searchOptions.limit),
          metadata: result.metadata
        });
      } catch (error) {
        batchResults.push({
          query,
          success: false,
          error: error.message,
          results: []
        });
      }
    }

    res.json({
      success: true,
      data: {
        batchResults,
        totalQueries: queries.length,
        successfulQueries: batchResults.filter(r => r.success).length
      }
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Document comparison endpoint
app.post('/compare-documents', async (req, res) => {
  try {
    const { query, documentIds } = req.body;
    
    if (!query || !Array.isArray(documentIds) || documentIds.length < 2) {
      return res.status(400).json({
        success: false,
        error: 'Query and at least 2 document IDs are required'
      });
    }

    console.log(`Comparing documents for query: "${query}"`);

    const comparisons = [];
    
    for (const docId of documentIds) {
      // Search within specific document
      const docSpecificSearch = await client.graphql
        .get()
        .withClassName('PDFDocument')
        .withWhere({
          path: ['filename'],
          operator: 'Equal',
          valueString: docId
        })
        .withNearText({ concepts: [query] })
        .withFields('content filename pageNumber relevanceScore: _additional { distance }')
        .withLimit(3)
        .do();

      const results = docSpecificSearch.data.Get.PDFDocument || [];
      
      comparisons.push({
        documentId: docId,
        relevantSections: results.map(result => ({
          content: result.content.substring(0, 300) + '...',
          pageNumber: result.pageNumber,
          relevance: 1 - (result._additional?.distance || 0.5)
        })),
        avgRelevance: results.length > 0 
          ? results.reduce((sum, r) => sum + (1 - (r._additional?.distance || 0.5)), 0) / results.length 
          : 0
      });
    }

    // Generate comparison summary
    const comparisonSummary = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'Compare how different documents address the given query. Highlight similarities, differences, and unique insights from each document.'
        },
        {
          role: 'user',
          content: `Query: "${query}"

Document Comparisons: ${JSON.stringify(comparisons, null, 2)}

Provide a comparative analysis of how each document addresses this query.`
        }
      ],
      temperature: 0.3,
      max_tokens: 800
    });

    res.json({
      success: true,
      data: {
        query,
        comparisons,
        summary: comparisonSummary.choices?.[0]?.message?.content,
        documentsCompared: documentIds.length
      }
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Export search results endpoint
app.post('/export-results', async (req, res) => {
  try {
    const { query, format = 'json', includeMetadata = false } = req.body;
    
    if (!query) {
      return res.status(400).json({
        success: false,
        error: 'Query is required'
      });
    }

    const searchResults = await enhancedSearchPDFContent(query, {
      limit: 50, // More results for export
      enableSynthesis: true,
      enableRanking: true
    });

    const exportData = {
      query,
      timestamp: new Date().toISOString(),
      results: searchResults.results.map(result => ({
        content: result.content,
        filename: result.filename,
        pageNumber: result.pageNumber,
        relevanceScore: result.relevanceScore,
        ...(includeMetadata && {
          chunkIndex: result.chunkIndex,
          importance: result.importance,
          keywords: result.keywords,
          wordCount: result.wordCount
        })
      })),
      synthesis: searchResults.synthesis,
      metadata: searchResults.metadata
    };

    if (format === 'csv') {
      // Convert to CSV format
      const csvHeader = ['Content', 'Filename', 'Page Number', 'Relevance Score'];
      const csvRows = searchResults.results.map(result => [
        `"${result.content.replace(/"/g, '""')}"`,
        result.filename,
        result.pageNumber,
        result.relevanceScore || 0
      ]);
      
      const csvContent = [csvHeader.join(','), ...csvRows.map(row => row.join(','))].join('\n');
      
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="search-results-${Date.now()}.csv"`);
      res.send(csvContent);
    } else {
      res.json({
        success: true,
        data: exportData
      });
    }

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
      console.log(`🚀 Enhanced Graviti Reg Search running on port ${PORT}`);
      console.log('\n📋 Available Endpoints:');
      console.log(`📄 Upload PDFs: POST /upload`);
      console.log(`🔍 Enhanced Search: POST /search`);
      console.log(`🔄 Query Refinement: POST /refine-query`);
      console.log(`📊 Batch Search: POST /batch-search`);
      console.log(`⚖️  Compare Documents: POST /compare-documents`);
      console.log(`📤 Export Results: POST /export-results`);
      console.log(`📈 Search Analytics: GET /search-analytics`);
      console.log(`📊 Stats: GET /stats`);
      console.log(`📁 Files: GET /files`);
      console.log(`❤️  Health Check: GET /check`);
      console.log('\n✨ Enhanced Features:');
      console.log('• Advanced result ranking with relevance scoring');
      console.log('• Multi-document synthesis and cross-reference');
      console.log('• Intelligent query refinement and expansion');
      console.log('• Result diversification to avoid duplicates');
      console.log('• Configurable search strategies');
      console.log('• Batch processing capabilities');
      console.log('• Document comparison tools');
      console.log('• Export functionality with multiple formats');
    });
  } catch (error) {
    console.error('Failed to start enhanced server:', error);
    process.exit(1);
  }
}

// Start the enhanced application
startServer();

module.exports = {
  app,
  client,
  openai,
  enhanceQuery,
  enhancedSearchPDFContent,
  ResultRanker,
  DocumentSynthesizer,
  QueryRefiner,
  validatePDFFile,
  extractPDFText,
  enhancedStorePDFInWeaviate,
  enhancedChunkTextWithOverlap
};