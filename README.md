# F1GPT: Formula 1 RAG Chatbot

An intelligent Next.js chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about Formula 1 racing. It searches an offline vector database for context and generates accurate, grounded answers.

## Tech Stack
- **Frontend**: Next.js 14 (App Router), React, Tailwind CSS
- **Vector Database**: MongoDB Atlas (Vector Search)
- **Embeddings**: Local HuggingFace Transformers (`Xenova/all-MiniLM-L6-v2`)
- **LLM / Answer Generation**: Google Gemini 1.5 Flash

## Prerequisites
- Node.js 18+
- MongoDB Atlas cluster (free tier works)
- Google Gemini API key

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment variables:**
   Copy `.env.example` to `.env` and fill in your details:
   ```env
   GEMINI_API_KEY="your-google-gemini-key"
   MONGODB_URI="mongodb+srv://<username>:<password>@cluster0.lktqp0g.mongodb.net/?appName=Cluster0"
   MONGODB_DB_NAME="f1gpt"
   MONGODB_COLLECTION="f1_embeddings"
   EMBEDDING_MODEL="Xenova/all-MiniLM-L6-v2"
   ```

3. **Seed the Database:**
   To populate your MongoDB cluster with F1 data, run the seed script. This scrapes Wikipedia articles, chunks them, creates vector embeddings locally, and saves them to MongoDB:
   ```bash
   npm run seed
   ```

4. **Run the Development Server:**
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

## How It Works
1. **Input**: You ask a Formula 1 question in the chat interface.
2. **Embedding**: Your question is converted into a vector embedding locally using HuggingFace.
3. **Retrieval**: The app queries MongoDB to find the 5 most mathematically similar text chunks to your question's vector.
4. **Generation**: The retrieved F1 facts are passed to the Gemini 1.5 model along with your question. Gemini formulates a natural, concise, and accurate answer based *only* on the provided context.
