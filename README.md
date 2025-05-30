# Picsea Server

The backend server for Picsea, an image search engine. This server handles the indexing of image annotations using a custom-built Tf-Idf Vector Space Model (VSM) and processes search queries to return relevant images based on semantic similarity. The server is built with Node.js and TypeScript and is hosted on AWS for scalability and availability.

## 🚀 Features
- Processes and indexes image annotations combining textual descriptions and ResNet-generated tags.
- Implements a Tf-Idf Vector Space Model for efficient search indexing.
- Handles search queries by converting them to vectors and retrieving ranked image results using cosine similarity.
- Serves search results via a JSON API consumed by the Picsea frontend.

## 🛠️ Tech Stack
- Node.js and TypeScript
- AWS (for hosting)
- MongoDB Atlas (for storing image data and annotations)
