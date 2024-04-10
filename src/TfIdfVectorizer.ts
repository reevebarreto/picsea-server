export class TfIdfVectorizer {
  private vocabulary: string[] = [];
  private documentFrequencies: Record<string, number> = {};
  private idfWeights: Record<string, number> = {};

  public fit(documents: string[]): void {
    this.vocabulary = this.buildVocabulary(documents);
    this.calculateDocumentFrequencies(documents);
    this.calculateIdfWeights();
  }

  public transform(document: string): number[] {
    const termFrequencies = this.calculateTermFrequencies(document);
    return this.calculateTfIdf(termFrequencies);
  }

  // Helper functions for core TF-IDF calculations

  private buildVocabulary(documents: string[]): string[] {
    const uniqueWords = new Set<string>();
    for (const doc of documents) {
      const words = doc.split(/\s+/); // Split on whitespace
      for (const word of words) {
        uniqueWords.add(word.toLowerCase()); // Lowercase for case-insensitivity
      }
    }

    return Array.from(uniqueWords);
  }

  private calculateDocumentFrequencies(documents: string[]): void {
    for (const word of this.vocabulary) {
      let documentCount = 0;
      for (const doc of documents) {
        if (doc.toLowerCase().indexOf(word) !== -1) {
          documentCount++;
        }
      }
      this.documentFrequencies[word] = documentCount;
    }
  }

  private calculateIdfWeights(): void {
    const totalDocuments = Object.keys(this.documentFrequencies).length;
    for (const word in this.documentFrequencies) {
      const documentFrequency = this.documentFrequencies[word];
      this.idfWeights[word] = Math.log(
        (totalDocuments + 1) / (documentFrequency + 1)
      );
    }
  }

  private calculateTermFrequencies(document: string): Record<string, number> {
    const termCounts: Record<string, number> = {};
    const words = document.split(/\s+/);
    for (const word of words) {
      const lowerWord = word.toLowerCase();
      termCounts[lowerWord] = (termCounts[lowerWord] || 0) + 1; // Count word occurrences
    }

    return termCounts;
  }

  private calculateTfIdf(termFrequencies: Record<string, number>): number[] {
    const tfIdfValues: number[] = [];
    for (const word of this.vocabulary) {
      const termFrequency = termFrequencies[word] || 0; // Handle potential out-of-vocabulary words
      const idfWeight = this.idfWeights[word] || 0; // Handle potential out-of-vocabulary words
      tfIdfValues.push(termFrequency * idfWeight);
    }

    return tfIdfValues;
  }
}
