import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import { MongoClient, ServerApiVersion } from "mongodb";
import similarity from "compute-cosine-similarity";
import { TfIdfVectorizer } from "./TfIdfVectorizer";
import { preprocessDocument } from "./utils/preprocessDocument";

const app = express();
const port = process.env.PORT || 3000; // Use environment variable for port

const uri =
  process.env.URI ||
  "mongodb+srv://colab:wellitscolab@dcu.32sdeqs.mongodb.net/?retryWrites=true&w=majority";
const databaseName = "MoS";
const collectionName = "images";

let vectorizer: TfIdfVectorizer;
let summaryVectors: Record<string, number[]> = {};
let docs: { index: string; summary: string; image_url: string }[] = [];

async function connectToDbAndFitVectorizer() {
  try {
    const client = await MongoClient.connect(uri, {
      serverApi: {
        version: ServerApiVersion.v1,
        strict: true,
        deprecationErrors: true,
      },
    });
    const db = client.db(databaseName);
    const collection = db.collection(collectionName);

    const imageDocs = await collection
      .find({})
      // .limit(1000)
      .toArray();
    const summaries = imageDocs.map((doc) => doc.summary);

    docs = imageDocs.map((doc) => ({
      index: doc.index,
      summary: doc.summary,
      image_url: doc.image_url,
    }));

    vectorizer = new TfIdfVectorizer();
    vectorizer.fit(summaries.map(preprocessDocument));

    summaryVectors = {}; // Clear any existing vectors
    for (const doc of imageDocs) {
      const summaryVector = vectorizer.transform(
        preprocessDocument(doc.summary)
      );
      summaryVectors[doc.index] = summaryVector;

      console.log("Fitted ", doc.index, "/", imageDocs.length);
    }

    console.log("TF-IDF vectorizer fit and summary vectors pre-computed");
    client.close();
  } catch (error) {
    console.error("Error connecting to database:", error);
    process.exit(1); // Exit on error
  }
}

connectToDbAndFitVectorizer();

app.use(bodyParser.json());
app.use(cors({ origin: "*" }));

app.get("/", (req, res) => {
  res.send("Welcome to the document search engine!");
});

app.post("/search", async (req, res) => {
  const query: string = req.body.query;

  if (!query) {
    return res.status(400).json({ error: "Missing 'query' parameter" });
  }

  const preprocessedQuery = preprocessDocument(query); // Implement your preprocessing logic (tokenization, etc.)

  // Calculate query vector using the same vectorizer
  const queryVector = vectorizer.transform(preprocessedQuery);

  // Similarity calculation (replace with your preferred method)
  const similarityScores: { index: string; score: number }[] = [];
  for (const [index, summaryVector] of Object.entries(summaryVectors)) {
    const score = similarity(queryVector, summaryVector) || 0; // Implement your similarity function (e.g., cosine similarity)
    similarityScores.push({ index, score });
  }

  // Sort results by similarity (descending) and select top N
  const topNSimilar = similarityScores
    .filter((result) => result.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 100);

  const topResults = topNSimilar.map((result) => {
    const matchingDoc = docs.find((doc) => doc.index == result.index);
    return matchingDoc || {}; // Handle potential missing documents
  });

  res.json(topResults);
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

module.exports = app; // Export the Express app for potential unit testing
