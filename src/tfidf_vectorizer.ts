import natural from "natural";
import * as fs from "fs";
import { preprocessDocument } from "./utils/preprocessDocument";

export default class TFIDFVectorizer {
  private tfidf: natural.TfIdf;
  public summaryVectors: { [id: string]: number[] } = {};

  constructor() {
    this.tfidf = new natural.TfIdf();
  }

  // documents are to be of type json string arrays
  fit(documents: Map<string, string>): void {
    // Preprocess documents (tokenization, stop word removal, etc.)
    for (const [id, summaryText] of documents.entries()) {
      const preprocessedText = preprocessDocument(summaryText);
      this.tfidf.addDocument(preprocessedText);
    }

    for (const [id, summaryText] of documents.entries()) {
      const preprocessedText = preprocessDocument(summaryText);
      // Calculate and store summary vector during fit
      this.summaryVectors[id] = this.transform(preprocessedText);

      console.log(`Finished fitting document ${id}`);
    }
  }

  transform(document: string): number[] {
    // Get TF-IDF vectors for each preprocessed document
    return this.tfidf.tfidfs(preprocessDocument(document));
  }
}
