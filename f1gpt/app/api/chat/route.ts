import { pipeline } from "@huggingface/transformers";
import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import { MongoClient } from "mongodb";
import OpenAI from "openai";

export const runtime = "nodejs";

type ChatMessage = {
  content: string;
  role: "assistant" | "user";
};

type StoredF1Document = {
  chunkIndex?: number;
  content?: string;
  embedding?: number[];
  sourceUrl?: string;
};

type ScoredDocument = StoredF1Document & {
  score: number;
};

const {
  EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2",
  MONGODB_COLLECTION = "f1_embeddings",
  MONGODB_DB_NAME = "f1gpt",
  MONGODB_URI,
  OPENAI_API_KEY,
} = process.env;

const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : undefined;

const globalStore = globalThis as typeof globalThis & {
  f1ExtractorPromise?: Promise<FeatureExtractionPipeline>;
  mongoClientPromise?: Promise<MongoClient>;
};

function getMongoClient() {
  if (!MONGODB_URI) {
    throw new Error("Missing MONGODB_URI in f1gpt/.env");
  }

  globalStore.mongoClientPromise ??= new MongoClient(MONGODB_URI, {
    serverSelectionTimeoutMS: 5_000,
  })
    .connect()
    .catch((error) => {
      globalStore.mongoClientPromise = undefined;
      throw error;
    });

  return globalStore.mongoClientPromise;
}

function getExtractor() {
  globalStore.f1ExtractorPromise ??= (
    pipeline("feature-extraction", EMBEDDING_MODEL) as Promise<FeatureExtractionPipeline>
  ).catch((error) => {
    globalStore.f1ExtractorPromise = undefined;
    throw error;
  });

  return globalStore.f1ExtractorPromise;
}

async function createEmbedding(content: string) {
  const extractor = await getExtractor();
  const output = await extractor(content, {
    pooling: "mean",
    normalize: true,
  });

  return Array.from(output.data as Iterable<number>);
}

function scoreDocument(queryEmbedding: number[], documentEmbedding?: number[]) {
  if (!documentEmbedding || documentEmbedding.length !== queryEmbedding.length) {
    return Number.NEGATIVE_INFINITY;
  }

  return documentEmbedding.reduce(
    (score, value, index) => score + value * queryEmbedding[index],
    0
  );
}

async function retrieveMongoContext(question: string) {
  const [client, queryEmbedding] = await Promise.all([
    getMongoClient(),
    createEmbedding(question),
  ]);

  const collection = client
    .db(MONGODB_DB_NAME)
    .collection<StoredF1Document>(MONGODB_COLLECTION);

  const documents = await collection
    .find(
      { content: { $type: "string" }, embedding: { $type: "array" } },
      {
        limit: 1_000,
        projection: {
          _id: 0,
          chunkIndex: 1,
          content: 1,
          embedding: 1,
          sourceUrl: 1,
        },
      }
    )
    .toArray();

  return documents
    .map((document) => ({
      ...document,
      score: scoreDocument(queryEmbedding, document.embedding),
    }))
    .filter((document) => Number.isFinite(document.score))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
}

function createContext(matches: ScoredDocument[]) {
  if (!matches.length) return "No matching MongoDB context found.";

  return matches
    .map((match, index) => {
      const source = match.sourceUrl ? `Source: ${match.sourceUrl}` : "Source: MongoDB";
      return `[${index + 1}] ${source}\n${match.content}`;
    })
    .join("\n\n---\n\n");
}

function createFallbackAnswer(question: string, matches: ScoredDocument[]) {
  if (!matches.length) {
    return "I could not find matching Formula 1 context in MongoDB yet. Try running npm run seed again, then ask another F1 question.";
  }

  const snippets = matches
    .slice(0, 3)
    .map((match, index) => {
      const content = match.content?.replace(/\s+/g, " ").trim() ?? "";
      const source = match.sourceUrl ? ` Source: ${match.sourceUrl}` : "";
      return `${index + 1}. ${content.slice(0, 360)}${content.length > 360 ? "..." : ""}${source}`;
    })
    .join("\n\n");

  return `I found relevant MongoDB context for: "${question}"\n\n${snippets}`;
}

function streamResponse(write: (send: (text: string) => void) => Promise<void>) {
  const encoder = new TextEncoder();

  const readable = new ReadableStream({
    async start(controller) {
      const send = (text: string) => {
        controller.enqueue(encoder.encode(`0:${JSON.stringify(text)}\n`));
      };

      try {
        await write(send);
      } catch (error) {
        console.error("Chat stream error:", error);
        send("Sorry, I could not read from the MongoDB data right now.");
      } finally {
        controller.enqueue(encoder.encode(`d:{"finishReason":"stop"}\n`));
        controller.close();
      }
    },
  });

  return new Response(readable, {
    headers: {
      "Cache-Control": "no-cache",
      "Content-Type": "text/event-stream",
      "X-Vercel-AI-Data-Stream": "v1",
    },
  });
}

export async function POST(req: Request) {
  const body = (await req.json()) as { messages?: ChatMessage[] };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const latestMessage =
    [...messages].reverse().find((message) => message.role === "user")?.content ??
    "";
  const question = latestMessage.trim();

  if (!question) {
    return streamResponse(async (send) => {
      send("Ask me a Formula 1 question and I will search your MongoDB data.");
    });
  }

  let matches: ScoredDocument[];

  try {
    matches = await retrieveMongoContext(question);
  } catch (error) {
    console.error("MongoDB retrieval error:", error);
    return streamResponse(async (send) => {
      send(
        "I could not connect to the MongoDB data right now. Check MONGODB_URI and confirm your MongoDB cluster is reachable, then try again."
      );
    });
  }

  const docContext = createContext(matches);

  if (!openai) {
    return streamResponse(async (send) => {
      send(createFallbackAnswer(question, matches));
    });
  }

  const systemPrompt = `You are F1GPT, an AI assistant specialized in Formula 1 racing.
Use the MongoDB context below first. If the context is not enough, say what is missing and answer carefully from general F1 knowledge.

MongoDB context:
${docContext}

Keep the answer concise and helpful.`;

  return streamResponse(async (send) => {
    try {
      const stream = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: systemPrompt },
          ...messages.map((message) => ({
            role: message.role,
            content: message.content,
          })),
        ],
        stream: true,
      });

      for await (const chunk of stream) {
        const text = chunk.choices[0]?.delta?.content ?? "";
        if (text) send(text);
      }
    } catch (error) {
      console.error("OpenAI response error:", error);
      send(createFallbackAnswer(question, matches));
    }
  });
}
