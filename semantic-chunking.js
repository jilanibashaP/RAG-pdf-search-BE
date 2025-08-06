// Method 1: Sentence-based chunking with overlap
function chunkBySentences(text, maxChunkSize = 1500, overlapSentences = 2) {
  // Split text into sentences using multiple delimiters
  const sentences = text.match(/[^\.!?]+[\.!?]+/g) || [text];
  
  const chunks = [];
  let currentChunk = [];
  let currentLength = 0;
  
  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i].trim();
    const sentenceLength = sentence.length;
    
    // If adding this sentence would exceed max size and we have content
    if (currentLength + sentenceLength > maxChunkSize && currentChunk.length > 0) {
      // Save current chunk
      const chunkText = currentChunk.join(' ').trim();
      chunks.push({
        text: chunkText,
        start: chunks.length === 0 ? 0 : chunks[chunks.length - 1].end,
        end: chunks.length === 0 ? chunkText.length : chunks[chunks.length - 1].end + chunkText.length,
        sentenceCount: currentChunk.length
      });
      
      // Start new chunk with overlap (keep last N sentences)
      const overlapStart = Math.max(0, currentChunk.length - overlapSentences);
      currentChunk = currentChunk.slice(overlapStart);
      currentLength = currentChunk.join(' ').length;
    }
    
    currentChunk.push(sentence);
    currentLength += sentenceLength;
  }
  
  // Add final chunk if it has content
  if (currentChunk.length > 0) {
    const chunkText = currentChunk.join(' ').trim();
    chunks.push({
      text: chunkText,
      start: chunks.length === 0 ? 0 : chunks[chunks.length - 1].end,
      end: chunks.length === 0 ? chunkText.length : chunks[chunks.length - 1].end + chunkText.length,
      sentenceCount: currentChunk.length
    });
  }
  
  return chunks;
}

// Method 2: Paragraph-based chunking
function chunkByParagraphs(text, maxChunkSize = 2000, overlapParagraphs = 1) {
  // Split by double newlines (paragraph breaks)
  const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
  
  const chunks = [];
  let currentChunk = [];
  let currentLength = 0;
  
  for (let i = 0; i < paragraphs.length; i++) {
    const paragraph = paragraphs[i].trim();
    const paragraphLength = paragraph.length;
    
    // If single paragraph is too long, split it by sentences
    if (paragraphLength > maxChunkSize) {
      // Handle oversized paragraph
      const sentenceChunks = chunkBySentences(paragraph, maxChunkSize, 1);
      
      // Add any accumulated paragraphs first
      if (currentChunk.length > 0) {
        const chunkText = currentChunk.join('\n\n').trim();
        chunks.push({
          text: chunkText,
          start: chunks.length === 0 ? 0 : chunks[chunks.length - 1].end,
          end: chunks.length === 0 ? chunkText.length : chunks[chunks.length - 1].end + chunkText.length,
          paragraphCount: currentChunk.length,
          type: 'paragraph'
        });
        currentChunk = [];
        currentLength = 0;
      }
      
      // Add sentence chunks from oversized paragraph
      sentenceChunks.forEach(chunk => {
        chunks.push({
          ...chunk,
          type: 'sentence',
          paragraphCount: 1
        });
      });
      continue;
    }
    
    // Check if adding this paragraph would exceed max size
    if (currentLength + paragraphLength > maxChunkSize && currentChunk.length > 0) {
      // Save current chunk
      const chunkText = currentChunk.join('\n\n').trim();
      chunks.push({
        text: chunkText,
        start: chunks.length === 0 ? 0 : chunks[chunks.length - 1].end,
        end: chunks.length === 0 ? chunkText.length : chunks[chunks.length - 1].end + chunkText.length,
        paragraphCount: currentChunk.length,
        type: 'paragraph'
      });
      
      // Start new chunk with overlap
      const overlapStart = Math.max(0, currentChunk.length - overlapParagraphs);
      currentChunk = currentChunk.slice(overlapStart);
      currentLength = currentChunk.join('\n\n').length;
    }
    
    currentChunk.push(paragraph);
    currentLength += paragraphLength + 2; // +2 for \n\n
  }
  
  // Add final chunk
  if (currentChunk.length > 0) {
    const chunkText = currentChunk.join('\n\n').trim();
    chunks.push({
      text: chunkText,
      start: chunks.length === 0 ? 0 : chunks[chunks.length - 1].end,
      end: chunks.length === 0 ? chunkText.length : chunks[chunks.length - 1].end + chunkText.length,
      paragraphCount: currentChunk.length,
      type: 'paragraph'
    });
  }
  
  return chunks;
}

// Method 3: Advanced semantic chunking with NLP-like features
function advancedSemanticChunking(text, maxChunkSize = 1800) {
  // Improved sentence splitting regex that handles more cases
  const sentenceRegex = /(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?]["'])\s+(?=[A-Z])/g;
  const sentences = text.split(sentenceRegex).filter(s => s.trim().length > 0);
  
  const chunks = [];
  let currentChunk = [];
  let currentLength = 0;
  
  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i].trim();
    
    // Skip very short sentences (likely artifacts)
    if (sentence.length < 10) continue;
    
    const sentenceLength = sentence.length;
    
    // Determine if this is a good breaking point
    const isGoodBreakPoint = isNaturalBreakPoint(sentence, sentences[i + 1]);
    
    // If we're near the limit and found a good break point
    if (currentLength + sentenceLength > maxChunkSize * 0.8 && isGoodBreakPoint && currentChunk.length > 0) {
      // Create chunk
      const chunkText = currentChunk.join(' ').trim();
      chunks.push({
        text: chunkText,
        start: calculateStart(chunks, chunkText),
        end: calculateEnd(chunks, chunkText),
        sentenceCount: currentChunk.length,
        type: 'semantic',
        coherenceScore: calculateCoherence(currentChunk)
      });
      
      // Start new chunk with smart overlap
      const overlapSentences = determineOverlap(currentChunk, sentence);
      currentChunk = currentChunk.slice(-overlapSentences);
      currentLength = currentChunk.join(' ').length;
    }
    
    currentChunk.push(sentence);
    currentLength += sentenceLength;
  }
  
  // Final chunk
  if (currentChunk.length > 0) {
    const chunkText = currentChunk.join(' ').trim();
    chunks.push({
      text: chunkText,
      start: calculateStart(chunks, chunkText),
      end: calculateEnd(chunks, chunkText),
      sentenceCount: currentChunk.length,
      type: 'semantic',
      coherenceScore: calculateCoherence(currentChunk)
    });
  }
  
  return chunks;
}

// Helper functions for advanced chunking
function isNaturalBreakPoint(currentSentence, nextSentence) {
  if (!nextSentence) return true;
  
  // Check for topic transition indicators
  const transitionWords = ['however', 'moreover', 'furthermore', 'in addition', 'on the other hand', 'meanwhile', 'subsequently'];
  const nextLower = nextSentence.toLowerCase();
  
  // New paragraph indicators
  if (currentSentence.endsWith('\n') || nextSentence.startsWith('\n')) return true;
  
  // Transition words at start of next sentence
  if (transitionWords.some(word => nextLower.startsWith(word))) return true;
  
  // Headers (sentences that are short and end with colons or are all caps)
  if (nextSentence.length < 50 && (nextSentence.endsWith(':') || nextSentence === nextSentence.toUpperCase())) return true;
  
  return false;
}

function determineOverlap(currentChunk, nextSentence) {
  // More overlap for technical content, less for narrative
  const technicalWords = ['therefore', 'thus', 'consequently', 'furthermore', 'moreover'];
  const hasTechnical = currentChunk.some(sentence => 
    technicalWords.some(word => sentence.toLowerCase().includes(word))
  );
  
  return hasTechnical ? 2 : 1;
}

function calculateCoherence(sentences) {
  // Simple coherence score based on word overlap between sentences
  if (sentences.length < 2) return 1;
  
  let totalOverlap = 0;
  for (let i = 0; i < sentences.length - 1; i++) {
    const words1 = new Set(sentences[i].toLowerCase().split(/\W+/));
    const words2 = new Set(sentences[i + 1].toLowerCase().split(/\W+/));
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    totalOverlap += intersection.size / Math.min(words1.size, words2.size);
  }
  
  return totalOverlap / (sentences.length - 1);
}

function calculateStart(chunks, chunkText) {
  return chunks.length === 0 ? 0 : chunks[chunks.length - 1].end;
}

function calculateEnd(chunks, chunkText) {
  return chunks.length === 0 ? chunkText.length : chunks[chunks.length - 1].end + chunkText.length;
}

// Method 4: Hybrid approach - combines multiple strategies
function hybridSemanticChunking(text, options = {}) {
  const {
    maxChunkSize = 1500,
    preferParagraphs = true,
    minChunkSize = 200,
    maxOverlapRatio = 0.2
  } = options;
  
  // First, try paragraph-based chunking
  if (preferParagraphs && text.includes('\n\n')) {
    const paragraphChunks = chunkByParagraphs(text, maxChunkSize, 1);
    
    // Check if paragraph chunks are reasonable sizes
    const reasonableChunks = paragraphChunks.every(chunk => 
      chunk.text.length >= minChunkSize && chunk.text.length <= maxChunkSize
    );
    
    if (reasonableChunks) {
      return paragraphChunks;
    }
  }
  
  // Fall back to sentence-based chunking
  return advancedSemanticChunking(text, maxChunkSize);
}

// Updated main function for your PDF processing
function chunkTextWithSemanticMeaning(text, maxChunkSize = 1500, options = {}) {
  const {
    method = 'hybrid', // 'sentences', 'paragraphs', 'semantic', 'hybrid'
    overlapSentences = 2,
    overlapParagraphs = 1
  } = options;
  
  switch (method) {
    case 'sentences':
      return chunkBySentences(text, maxChunkSize, overlapSentences);
    case 'paragraphs':
      return chunkByParagraphs(text, maxChunkSize, overlapParagraphs);
    case 'semantic':
      return advancedSemanticChunking(text, maxChunkSize);
    case 'hybrid':
    default:
      return hybridSemanticChunking(text, { maxChunkSize, ...options });
  }
}

// Example usage in your PDF storage function
async function storePDFInWeaviateWithSemanticChunking(filePath, originalFilename) {
  try {
    const { text, numPages } = await extractPDFText(filePath);
    console.log(`Extracted ${numPages} pages from PDF: ${originalFilename}`);
    
    // Use semantic chunking instead of fixed character chunking
    const chunks = chunkTextWithSemanticMeaning(text, 1500, {
      method: 'hybrid',
      preferParagraphs: true,
      minChunkSize: 300
    });

    console.log(`Processing ${chunks.length} semantic chunks from ${originalFilename}`);
    
    // Rest of your existing code...
    // The chunks will now have better semantic boundaries
    
    return chunks;
  } catch (error) {
    throw new Error(`Failed to process PDF semantically: ${error.message}`);
  }
}