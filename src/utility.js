const OpenAI = require('openai');
const weaviate = require('weaviate-ts-client').default;
const path = require('path');

// Load environment variables with proper path resolution
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });

// Validate environment variables before initializing clients
if (!process.env.OPENAI_API_KEY) {
  console.error('❌ OPENAI_API_KEY environment variable is missing!');
  console.log('Please add OPENAI_API_KEY=your_key_here to your .env file');
  process.exit(1);
}

if (!process.env.WEAVIATE_API_KEY) {
  console.error('❌ WEAVIATE_API_KEY environment variable is missing!');
  console.log('Please add WEAVIATE_API_KEY=your_key_here to your .env file');
  process.exit(1);
}

if (!process.env.WEAVIATE_HOST) {
  console.error('❌ WEAVIATE_HOST environment variable is missing!');
  console.log('Please add WEAVIATE_HOST=your_weaviate_host to your .env file');
  process.exit(1);
}

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Weaviate client
const client = weaviate.client({
  scheme: 'https',
  host: process.env.WEAVIATE_HOST,
  apiKey: new weaviate.ApiKey(process.env.WEAVIATE_API_KEY),
});

// Generate embeddings using OpenAI
async function generateEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text.substring(0, 8000), // Limit input length
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    throw new Error(`Embedding generation failed: ${error.message}`);
  }
}

// Query enhancement using LLM
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

// Hybrid Search function - combines vector and keyword search
async function searchPDFContent(query, limit = 5, alpha = 0.7) {
    try {
        // Enhance the query using LLM
        const enhancedQuery = await enhanceQuery(query);
        console.log(`Original query: "${query}"`);
        console.log(`Enhanced query: "${enhancedQuery}"`);

        // Generate embedding for the enhanced query
        const queryEmbedding = await generateEmbedding(enhancedQuery);
        console.log('Generated embedding for query====>', queryEmbedding);

        // Hybrid search combining vector similarity and keyword matching
        const result = await client.graphql
            .get()
            .withClassName('PDFDocument')
            .withFields('content filename savedFilename pageNumber chunkIndex totalPages uploadDate filePath _additional { certainty distance score }')
            .withHybrid({
                query: enhancedQuery,
                vector: queryEmbedding,
                alpha: alpha, // 0 = pure keyword, 1 = pure vector, 0.7 = balanced toward vector
                fusionType: 'rankedFusion' // or 'relativeScore'
            })
            .withLimit(limit)
            .do();

        console.log('Hybrid search results:', JSON.stringify(result.data, null, 2));

        return {
            originalQuery: query,
            enhancedQuery: enhancedQuery,
            searchType: 'hybrid',
            alpha: alpha,
            results: result.data.Get.PDFDocument || []
        };
    } catch (error) {
        throw new Error(`Hybrid search failed: ${error.message}`);
    }
}

// Alternative: Separate vector and keyword searches with custom fusion
async function searchPDFContentAdvanced(query, limit = 5, vectorWeight = 0.7) {
    try {
        const enhancedQuery = await enhanceQuery(query);
        console.log(`Original query: "${query}"`);
        console.log(`Enhanced query: "${enhancedQuery}"`);

        const queryEmbedding = await generateEmbedding(enhancedQuery);

        // Perform both searches in parallel
        const [vectorResults, keywordResults] = await Promise.all([
            // Vector search
            client.graphql
                .get()
                .withClassName('PDFDocument')
                .withFields('content filename savedFilename pageNumber chunkIndex totalPages uploadDate filePath _additional { certainty distance }')
                .withNearVector({ vector: queryEmbedding })
                .withLimit(limit * 2) // Get more results for fusion
                .do(),

            // Keyword search (BM25)
            client.graphql
                .get()
                .withClassName('PDFDocument')
                .withFields('content filename savedFilename pageNumber chunkIndex totalPages uploadDate filePath _additional { score }')
                .withBm25({
                    query: enhancedQuery,
                    properties: ['content', 'filename'] // Search in these fields
                })
                .withLimit(limit * 2)
                .do()
        ]);

        // Custom fusion of results
        const fusedResults = fuseSearchResults(
            vectorResults.data.Get.PDFDocument || [],
            keywordResults.data.Get.PDFDocument || [],
            vectorWeight,
            limit
        );

        console.log('Advanced hybrid search results:', JSON.stringify(fusedResults, null, 2));

        return {
            originalQuery: query,
            enhancedQuery: enhancedQuery,
            searchType: 'advanced_hybrid',
            vectorWeight: vectorWeight,
            vectorResultsCount: vectorResults.data.Get.PDFDocument?.length || 0,
            keywordResultsCount: keywordResults.data.Get.PDFDocument?.length || 0,
            results: fusedResults
        };
    } catch (error) {
        throw new Error(`Advanced hybrid search failed: ${error.message}`);
    }
}

// Helper function to fuse vector and keyword search results
function fuseSearchResults(vectorResults, keywordResults, vectorWeight = 0.7, limit = 5) {
    const keywordWeight = 1 - vectorWeight;
    const resultMap = new Map();

    // Normalize and add vector results
    const maxCertainty = Math.max(...vectorResults.map(r => r._additional?.certainty || 0));
    vectorResults.forEach((result, index) => {
        const normalizedScore = (result._additional?.certainty || 0) / maxCertainty;
        const key = `${result.filename}_${result.pageNumber}_${result.chunkIndex}`;

        resultMap.set(key, {
            ...result,
            hybridScore: normalizedScore * vectorWeight,
            vectorScore: normalizedScore,
            keywordScore: 0,
            vectorRank: index + 1
        });
    });

    // Normalize and merge keyword results
    const maxBM25Score = Math.max(...keywordResults.map(r => r._additional?.score || 0));
    keywordResults.forEach((result, index) => {
        const normalizedScore = maxBM25Score > 0 ? (result._additional?.score || 0) / maxBM25Score : 0;
        const key = `${result.filename}_${result.pageNumber}_${result.chunkIndex}`;

        if (resultMap.has(key)) {
            // Merge with existing vector result
            const existing = resultMap.get(key);
            existing.hybridScore += normalizedScore * keywordWeight;
            existing.keywordScore = normalizedScore;
            existing.keywordRank = index + 1;
        } else {
            // Add as new keyword-only result
            resultMap.set(key, {
                ...result,
                hybridScore: normalizedScore * keywordWeight,
                vectorScore: 0,
                keywordScore: normalizedScore,
                keywordRank: index + 1
            });
        }
    });

    // Sort by hybrid score and return top results
    return Array.from(resultMap.values())
        .sort((a, b) => b.hybridScore - a.hybridScore)
        .slice(0, limit);
}

// Utility function to adjust search parameters based on query type
function getOptimalSearchParams(query) {
    const queryLower = query.toLowerCase();

    // More keyword-focused for specific terms, names, numbers
    if (/\b\d+\b/.test(query) || /[A-Z]{2,}/.test(query) || query.includes('"')) {
        return { alpha: 0.3, vectorWeight: 0.3 }; // Favor keyword search
    }

    // More vector-focused for conceptual queries
    if (queryLower.includes('what') || queryLower.includes('how') || queryLower.includes('why')) {
        return { alpha: 0.8, vectorWeight: 0.8 }; // Favor vector search
    }

    // Balanced approach for general queries
    return { alpha: 0.7, vectorWeight: 0.7 };
}

// Smart hybrid search that adjusts parameters based on query
async function smartSearchPDFContent(query, limit = 5) {
    const { alpha, vectorWeight } = getOptimalSearchParams(query);

    console.log(`Smart search detected parameters - alpha: ${alpha}, vectorWeight: ${vectorWeight}`);

    // Use the basic hybrid search with optimized parameters
    return await searchPDFContent(query, limit, alpha);
}

// Basic search function
// async function basicSearchPDFContent(query, limit = 5) {
//   try {
//     // Enhance the query using LLM
//     const enhancedQuery = await enhanceQuery(query);
//     console.log(`Original query: "${query}"`);
//     console.log(`Enhanced query: "${enhancedQuery}"`);

//     // Generate embedding for the enhanced query
//     const queryEmbedding = await generateEmbedding(enhancedQuery);
//     console.log('Generated embedding for query====>', queryEmbedding);

//     const result = await client.graphql
//       .get()
//       .withClassName('PDFDocument')
//       .withFields('content filename savedFilename pageNumber chunkIndex totalPages uploadDate filePath _additional { certainty distance }')
//       // .withNearVector({ vector: queryEmbedding })
//       .withLimit(limit)
//       .do();

//     console.log('Search results:', JSON.stringify(result.data, null, 2));

//     return {
//       originalQuery: query,
//       enhancedQuery: enhancedQuery,
//       results: result.data.Get.PDFDocument || []
//     };
//   } catch (error) {
//     throw new Error(`Search failed: ${error.message}`);
//   }
// }

async function basicSearchPDFContent(query, limit = 5) {
  try {
    console.log('Starting search for:', query);
    
    // Check if data exists first
    const dataCheck = await client.graphql
      .get()
      .withClassName('PDFDocument')
      .withFields('content filename')
      .withLimit(1)
      .do();
    
    console.log('Data exists:', dataCheck.data.Get.PDFDocument?.length > 0);
    
    // Enhance the query using LLM
    const enhancedQuery = await enhanceQuery(query);
    console.log(`Original query: "${query}"`);
    console.log(`Enhanced query: "${enhancedQuery}"`);

    // Generate embedding for the enhanced query
    const queryEmbedding = await generateEmbedding(enhancedQuery);
    console.log('Query embedding length:', queryEmbedding?.length);
    console.log('First 5 values:', queryEmbedding?.slice(0, 5));

    // Try search without nearVector first (should return all results)
    const allResults = await client.graphql
      .get()
      .withClassName('PDFDocument')
      .withFields('content filename savedFilename pageNumber chunkIndex _additional { certainty }')
      .withLimit(limit)
      .do();
    
    console.log('All results (no vector search):', allResults.data.Get.PDFDocument?.length || 0);

    // Now try with vector search
    const result = await client.graphql
      .get()
      .withClassName('PDFDocument')
      .withFields('content filename savedFilename pageNumber chunkIndex totalPages uploadDate filePath _additional { certainty distance }')
      .withNearVector({ vector: queryEmbedding })
      .withLimit(limit)
      .do();

    console.log('Vector search results:', result.data.Get.PDFDocument?.length || 0);
    console.log('Search results:', JSON.stringify(result.data, null, 2));

    return {
      originalQuery: query,
      enhancedQuery: enhancedQuery,
      results: result.data.Get.PDFDocument || []
    };
  } catch (error) {
    console.error('Search error details:', error);
    throw new Error(`Search failed: ${error.message}`);
  }
}

module.exports = {
    client,
    openai,
    generateEmbedding,
    enhanceQuery,
    searchPDFContent,
    searchPDFContentAdvanced,
    smartSearchPDFContent,
    basicSearchPDFContent,
    fuseSearchResults,
    getOptimalSearchParams
};