import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { MongoClient } from "mongodb";

export const runtime = "nodejs";
export const maxDuration = 60;

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

type RetrievalResult = {
  error?: string;
  matches: ScoredDocument[];
};

const {
  EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2",
  GEMINI_API_KEY,
  GEMINI_MODEL = "gemini-2.5-flash",
  MONGODB_COLLECTION = "f1_embeddings",
  MONGODB_DB_NAME = "f1gpt",
  MONGODB_URI,
} = process.env;

const gemini = GEMINI_API_KEY
  ? new GoogleGenerativeAI(GEMINI_API_KEY)
  : undefined;

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
    import("@huggingface/transformers").then(({ pipeline }) =>
      pipeline("feature-extraction", EMBEDDING_MODEL)
    ) as Promise<FeatureExtractionPipeline>
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
    return `I could not find matching Formula 1 context in MongoDB for: "${question}". Check your MongoDB connection, then run npm run seed again if the collection is empty.`;
  }

  const snippets = matches
    .slice(0, 3)
    .map((match, index) => {
      const content = match.content?.replace(/\s+/g, " ").trim() ?? "";
      const source = match.sourceUrl ? ` Source: ${match.sourceUrl}` : "";
      return `${index + 1}. ${content.slice(0, 360)}${content.length > 360 ? "..." : ""}${source}`;
    })
    .join("\n\n");

  return `Here is relevant Formula 1 context for: "${question}"\n\n${snippets}`;
}

function explainMongoError(error: unknown) {
  if (error instanceof Error) {
    if (error.message.includes("querySrv")) {
      return "MongoDB DNS lookup failed for your mongodb+srv URL. Check the Atlas cluster hostname, internet/DNS access, and whether the cluster is active.";
    }

    if (error.message.includes("Missing MONGODB_URI")) {
      return "MONGODB_URI is missing in f1gpt/.env.";
    }

    if (
      error.message.includes("@huggingface/transformers") ||
      error.message.includes("onnxruntime") ||
      error.message.includes("fetch failed")
    ) {
      return "The embedding model could not load in the Vercel function. The app can still answer with Gemini, but MongoDB retrieval is unavailable for this request.";
    }

    return error.message;
  }

  return "MongoDB retrieval failed.";
}

function explainGeminiError(error: unknown) {
  if (!(error instanceof Error)) {
    return "Gemini response failed.";
  }

  if (error.message.includes("API key not valid")) {
    return "Your GEMINI_API_KEY is invalid. Create a new key in Google AI Studio and update it in Vercel.";
  }

  if (error.message.includes("not found") || error.message.includes("not supported")) {
    return `The Gemini model "${GEMINI_MODEL}" is unavailable for this API key. Try setting GEMINI_MODEL to gemini-2.5-flash in Vercel.`;
  }

  if (error.message.includes("quota") || error.message.includes("429")) {
    return "Gemini quota was exceeded for this API key. Check billing/quota in Google AI Studio or try another key.";
  }

  return error.message;
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
        send("Sorry, I could not generate a response right now. Please try again.");
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
  let body: { messages?: ChatMessage[] };

  try {
    body = (await req.json()) as { messages?: ChatMessage[] };
  } catch (error) {
    console.error("Invalid chat request body:", error);
    return streamResponse(async (send) => {
      send("The chat request was invalid. Please refresh the page and try again.");
    });
  }

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

  // --- MongoDB retrieval ---
  const retrieval: RetrievalResult = { matches: [] };

  try {
    retrieval.matches = await retrieveMongoContext(question);
  } catch (error) {
    console.error("MongoDB retrieval error:", error);
    retrieval.error = explainMongoError(error);
  }

  const docContext = retrieval.error
    ? `MongoDB context is unavailable: ${retrieval.error}`
    : createContext(retrieval.matches);

  // --- No Gemini key: return raw context snippets ---
  if (!gemini) {
    return streamResponse(async (send) => {
      if (retrieval.error) {
        send(
          `${retrieval.error}\n\nSet GEMINI_API_KEY in f1gpt/.env to enable AI-generated answers.`
        );
        return;
      }

      send(createFallbackAnswer(question, retrieval.matches));
    });
  }

  // --- Build system prompt ---
  const systemPrompt = `You are F1GPT, an expert AI assistant specialised in Formula 1 racing.
Use the MongoDB context below as your primary source of truth when it is available.
If MongoDB context is unavailable, answer from general Formula 1 knowledge without mentioning internal retrieval errors.

MongoDB context:
${docContext}

STRICT RESPONSE GUIDELINES:
- YOUR ENTIRE ANSWER MUST BE A MAXIMUM OF 3 TO 4 SENTENCES.
- Be extremely concise, direct, and to the point.
- Do not add conversational fluff or greetings (e.g., skip "Here is the answer...").
- Use bullet points ONLY if it fits within the 3-4 sentence limit.
- If the context does not contain enough information, supplement carefully from your general F1 knowledge and briefly state you are doing so.`;

  // --- Stream Gemini response ---
  return streamResponse(async (send) => {
    try {
      const model = gemini.getGenerativeModel({ model: GEMINI_MODEL });

      // Build conversation history for Gemini (exclude the latest user message)
      const history = messages.slice(0, -1).map((msg) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: msg.content }],
      }));

      const chat = model.startChat({
        history,
        systemInstruction: { role: "system", parts: [{ text: systemPrompt }] },
      });

      const result = await chat.sendMessageStream(question);

      for await (const chunk of result.stream) {
        const text = chunk.text();
        if (text) send(text);
      }
    } catch (error) {
      console.error("Gemini response error:", error);

      if (retrieval.error) {
        send(explainGeminiError(error));
        return;
      }

      // Gemini failed but we have MongoDB context — show raw snippets
      send(createFallbackAnswer(question, retrieval.matches));
    }
  });
}
