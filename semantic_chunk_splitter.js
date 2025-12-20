class SemanticChunkSplitter {
  constructor(options = {}) {
    this.maxChunkSize = options.maxChunkSize || 1000;
    this.minChunkSize = options.minChunkSize || 100;
    this.overlapSize = options.overlapSize || 50;
    this.semanticThreshold = options.semanticThreshold || 0.5;
  }

  // Main split method with semantic awareness
  async split(text) {
    if (!text || typeof text !== 'string') {
      return [];
    }

    text = text.trim().replace(/\r\n/g, '\n');
    
    // Split into sentences
    const sentences = this.splitIntoSentences(text);
    
    if (sentences.length === 0) return [];
    
    // Calculate semantic similarity between consecutive sentences
    const similarities = await this.calculateSemanticSimilarities(sentences);
    
    // Group sentences based on similarity with sentence limits
    const semanticUnits = this.groupBySimilarity(sentences, similarities);
    
    // Combine units into properly sized chunks
    const chunks = this.createChunks(semanticUnits);
    
    return chunks.filter(c => c.length >= this.minChunkSize);
  }

  // Split text into sentences
  splitIntoSentences(text) {
    // More sophisticated sentence splitting
    const sentences = [];
    const pattern = /([^.!?]+[.!?]+(?:\s+|$))/g;
    let match;
    
    while ((match = pattern.exec(text)) !== null) {
      const sentence = match[1].trim();
      if (sentence) sentences.push(sentence);
    }
    
    // Handle text without sentence endings
    if (sentences.length === 0 && text.trim()) {
      sentences.push(text.trim());
    }
    
    return sentences;
  }

  // Calculate semantic similarity using TF-IDF and cosine similarity
  async calculateSemanticSimilarities(sentences) {
    const vectors = sentences.map(s => this.createTFIDFVector(s, sentences));
    const similarities = [];
    
    for (let i = 0; i < vectors.length - 1; i++) {
      const sim = this.cosineSimilarity(vectors[i], vectors[i + 1]);
      similarities.push(sim);
    }
    
    return similarities;
  }

  // Create TF-IDF vector for a sentence
  createTFIDFVector(sentence, allSentences) {
    const words = this.tokenize(sentence);
    const wordFreq = {};
    
    // Term frequency
    words.forEach(word => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });
    
    // Calculate TF-IDF
    const vector = {};
    for (const word in wordFreq) {
      const tf = wordFreq[word] / words.length;
      const idf = this.calculateIDF(word, allSentences);
      vector[word] = tf * idf;
    }
    
    return vector;
  }

  // Calculate inverse document frequency
  calculateIDF(word, sentences) {
    const docCount = sentences.filter(s => 
      this.tokenize(s).includes(word)
    ).length;
    
    return Math.log((sentences.length + 1) / (docCount + 1));
  }

  // Tokenize and normalize text
  tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 2 && !this.isStopWord(w));
  }

  // Common stop words
  isStopWord(word) {
    const stopWords = new Set([
      'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 
      'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 
      'does', 'did', 'but', 'if', 'or', 'because', 'as', 'until',
      'while', 'of', 'for', 'with', 'about', 'against', 'between',
      'into', 'through', 'during', 'before', 'after', 'above',
      'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
      'off', 'over', 'under', 'again', 'further', 'then', 'once',
      'and', 'can', 'will', 'just', 'should', 'now', 'this', 'that',
      'these', 'those', 'am', 'what', 'who', 'when', 'where', 'why', 'how'
    ]);
    return stopWords.has(word);
  }

  // Calculate cosine similarity between two vectors
  cosineSimilarity(vec1, vec2) {
    const allKeys = new Set([...Object.keys(vec1), ...Object.keys(vec2)]);
    let dotProduct = 0;
    let mag1 = 0;
    let mag2 = 0;
    
    allKeys.forEach(key => {
      const v1 = vec1[key] || 0;
      const v2 = vec2[key] || 0;
      dotProduct += v1 * v2;
      mag1 += v1 * v1;
      mag2 += v2 * v2;
    });
    
    const magnitude = Math.sqrt(mag1) * Math.sqrt(mag2);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  // Group sentences based on similarity scores with sentence limits
  groupBySimilarity(sentences, similarities) {
    const units = [];
    let currentUnit = [sentences[0]];
    const maxSimilarSentences = 5;
    const maxDissimilarSentences = 2;
    
    // Calculate threshold for determining similarity
    const mean = similarities.length > 0 
      ? similarities.reduce((a, b) => a + b, 0) / similarities.length 
      : 0.5;
    const threshold = mean * 0.8; // 80% of mean similarity
    
    for (let i = 0; i < similarities.length; i++) {
      const similarity = similarities[i];
      const nextSentence = sentences[i + 1];
      
      // IF consecutive sentences are semantically similar
      if (similarity >= threshold) {
        currentUnit.push(nextSentence);
        
        // Maximum 5 sentences when similar
        if (currentUnit.length >= maxSimilarSentences) {
          units.push(currentUnit.join(' '));
          currentUnit = [];
          // Start next unit with current sentence for context
          if (i + 1 < sentences.length - 1) {
            currentUnit.push(nextSentence);
            i++; // Skip the sentence we just added
          }
        }
      } 
      // IF NOT semantically similar
      else {
        currentUnit.push(nextSentence);
        
        // Maximum 2 sentences when dissimilar
        if (currentUnit.length >= maxDissimilarSentences) {
          units.push(currentUnit.join(' '));
          currentUnit = [];
        }
      }
    }
    
    // Add remaining sentences
    if (currentUnit.length > 0) {
      units.push(currentUnit.join(' '));
    }
    
    return units;
  }

  // Create final chunks respecting size constraints
  createChunks(semanticUnits) {
    const chunks = [];
    let currentChunk = '';
    let previousOverlap = '';
    
    for (const unit of semanticUnits) {
      // If unit is too large, split it further
      if (unit.length > this.maxChunkSize) {
        if (currentChunk.trim()) {
          chunks.push((previousOverlap + ' ' + currentChunk).trim());
          previousOverlap = this.extractOverlap(currentChunk);
          currentChunk = '';
        }
        
        const subChunks = this.splitBySize(unit);
        subChunks.forEach((subChunk, i) => {
          if (i === 0 && previousOverlap) {
            chunks.push((previousOverlap + ' ' + subChunk).trim());
          } else {
            chunks.push(subChunk.trim());
          }
          previousOverlap = this.extractOverlap(subChunk);
        });
      } else {
        const potentialChunk = currentChunk 
          ? currentChunk + ' ' + unit 
          : unit;
        
        if (potentialChunk.length > this.maxChunkSize && currentChunk) {
          chunks.push((previousOverlap + ' ' + currentChunk).trim());
          previousOverlap = this.extractOverlap(currentChunk);
          currentChunk = unit;
        } else {
          currentChunk = potentialChunk;
        }
      }
    }
    
    if (currentChunk.trim()) {
      const finalChunk = previousOverlap 
        ? (previousOverlap + ' ' + currentChunk).trim()
        : currentChunk.trim();
      chunks.push(finalChunk);
    }
    
    return chunks;
  }

  // Split oversized text by size
  splitBySize(text) {
    const sentences = this.splitIntoSentences(text);
    const chunks = [];
    let current = '';
    
    sentences.forEach(sentence => {
      if ((current + ' ' + sentence).length > this.maxChunkSize && current) {
        chunks.push(current.trim());
        current = sentence;
      } else {
        current = current ? current + ' ' + sentence : sentence;
      }
    });
    
    if (current) chunks.push(current.trim());
    return chunks;
  }

  // Extract overlap text
  extractOverlap(text) {
    if (!text || text.length < this.overlapSize) return '';
    
    const sentences = this.splitIntoSentences(text);
    if (sentences.length > 1) {
      return sentences[sentences.length - 1];
    }
    
    return text.slice(-this.overlapSize);
  }

  // Get statistics
  getStats(chunks) {
    if (!chunks || chunks.length === 0) {
      return { count: 0, avgSize: 0, minSize: 0, maxSize: 0 };
    }
    
    const sizes = chunks.map(c => c.length);
    return {
      count: chunks.length,
      avgSize: Math.round(sizes.reduce((a, b) => a + b, 0) / sizes.length),
      minSize: Math.min(...sizes),
      maxSize: Math.max(...sizes)
    };
  }
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SemanticChunkSplitter;
}

// Export for ES6 modules
if (typeof exports !== 'undefined') {
  exports.SemanticChunkSplitter = SemanticChunkSplitter;
}