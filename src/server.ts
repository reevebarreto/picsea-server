import express from "express";
import bodyParser from "body-parser";
import { MongoClient, ServerApiVersion } from "mongodb";
import TFIDFVectorizer from "./tfidf_vectorizer";
import similarity from "compute-cosine-similarity";

const app = express();
const port = process.env.PORT || 3000; // Use environment variable for port

const uri = process.env.URI as string;

const client = new MongoClient(uri, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  },
});

// Pre-trained TF-IDF vectorizer instance
const vectorizer = new TFIDFVectorizer();

// Create a dictionary to store image data
const imageData: {
  [index: string]: { image_url: string; summary: string };
} = {};

async function connectToDBAndFit() {
  try {
    await client.connect();
    console.log("Connected to MongoDB database!");

    // Retrieve images from MongoDB
    const db = client.db("MoS");
    const collection = db.collection("images");

    const imageDocs = await collection
      .find({})
      // .limit(1000)
      .toArray();

    // Process image documents, fit vectorizer, and populate dictionary
    for (const imageDoc of imageDocs) {
      const { index, image_url, summary } = imageDoc;

      // Store image data in dictionary
      imageData[index] = { image_url, summary };
    }

    // Fit the TF-IDF vectorizer with image summaries
    vectorizer.fit(
      new Map(
        Object.entries(imageData).map(([index, data]) => [index, data.summary])
      )
    );

    console.log(
      "Finished fitting TF-IDF vectorizer and populating image data!"
    );
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
    process.exit(1); // Exit the application on connection failure
  }
}

connectToDBAndFit(); // Call the connection function

app.use(bodyParser.json());

app.get("/", (req, res) => {
  res.send("Welcome to the document search engine!");
});

app.post("/search", async (req, res) => {
  const query: string = req.body.query;

  if (!query) {
    return res.status(400).json({ error: "Missing 'query' parameter" });
  }

  // Convert the query into a TF-IDF vector
  const queryVector = vectorizer.transform(query);

  // Calculate cosine similarities
  const cosineSimilarities: { [id: string]: number } = {};
  for (const [id, summaryVector] of Object.entries(vectorizer.summaryVectors)) {
    cosineSimilarities[id] = similarity(queryVector, summaryVector) || 0;
  }

  // Sort documents and their cosine similarities in descending order
  const sortedResults = Object.entries(cosineSimilarities)
    .sort((a, b) => b[1] - a[1])
    .map(([id, score]) => ({ id, score }))
    .slice(0, 10); // Return top 10 results by default

  res.json({
    images: sortedResults.map((result) => {
      const matchingImage = imageData[result.id];
      if (matchingImage) {
        return {
          index: result.id,
          image_url: matchingImage.image_url,
          summary: matchingImage.summary,
        };
      } else {
        console.warn(`Image with ID "${result.id}" not found in imageData`);
        return null;
      }
    }),
  });
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

module.exports = app; // Export the Express app for potential unit testing
