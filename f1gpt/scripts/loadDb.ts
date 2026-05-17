import { MongoClient } from "mongodb"
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { pipeline } from "@huggingface/transformers"

import dotenv from "dotenv"
import path from "path"
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
dotenv.config({ path: path.resolve(__dirname, "../.env") })

const {
    MONGODB_URI,
    MONGODB_DB_NAME = "f1gpt",
    MONGODB_COLLECTION = "f1_embeddings",
    EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2",
} = process.env

const missingEnv = Object.entries({ MONGODB_URI })
    .filter(([, value]) => !value)
    .map(([key]) => key)

if (missingEnv.length) {
    throw new Error(`Missing required environment variables in f1gpt/.env: ${missingEnv.join(", ")}`)
}

const f1data = [
    "https://en.wikipedia.org/wiki/Formula_One",
    "https://en.wikipedia.org/wiki/History_of_Formula_One",
    "https://de.wikipedia.org/wiki/Lewis_Hamilton",
    "https://en.wikipedia.org/wiki/Lewis_Hamilton",
    "https://en.wikipedia.org/wiki/Max_Verstappen",
    "https://en.wikipedia.org/wiki/Michael_Schumacher",
    "https://en.wikipedia.org/wiki/Sebastian_Vettel",
    "https://www.forbes.com/sites/brettknight/2025/12/09/formula-1s-highest-paid-drivers-2025/",
    "https://www.formula1.com/en/latest/article.inside-the-business-of-formula-1.70b7k09mQ1rJ9Msm3mK4bY.html",
]

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 100,
})

const createMongoClient = () => new MongoClient(MONGODB_URI, {
    serverSelectionTimeoutMS: 5_000,
})

let extractorPromise: Promise<any> | undefined

const getExtractor = () => {
    extractorPromise ??= pipeline("feature-extraction", EMBEDDING_MODEL)
    return extractorPromise
}

const createEmbedding = async (content: string) => {
    const extractor = await getExtractor()
    const output = await extractor(content, {
        pooling: "mean",
        normalize: true,
    })

    return Array.from(output.data)
}

const prepareCollection = async (client: MongoClient) => {
    const collection = client
        .db(MONGODB_DB_NAME)
        .collection(MONGODB_COLLECTION)

    await collection.createIndex(
        { sourceUrl: 1, chunkIndex: 1 },
        { unique: true, name: "source_chunk_unique" }
    )
    await collection.createIndex({ sourceUrl: 1 }, { name: "source_url" })

    return collection
}

const loadSampleData = async () => {
    const client = createMongoClient()

    try {
        await client.connect().catch((error) => {
            throw new Error(`Could not connect to MongoDB using MONGODB_URI from f1gpt/.env.`, { cause: error })
        })
        console.log(`Connected to MongoDB database "${MONGODB_DB_NAME}".`)

        const collection = await prepareCollection(client)

        for (const url of f1data) {
            console.log(`Scraping ${url}`)
            const content = await scrapePage(url)
            const chunks = await splitter.splitText(content)

            for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
                const content = chunks[chunkIndex]
                const embedding = await createEmbedding(content)

                const result = await collection.updateOne(
                    { sourceUrl: url, chunkIndex },
                    {
                        $set: {
                            content,
                            embedding,
                            embeddingDimensions: embedding.length,
                            embeddingModel: EMBEDDING_MODEL,
                            sourceUrl: url,
                            chunkIndex,
                            updatedAt: new Date(),
                        },
                        $setOnInsert: {
                            createdAt: new Date(),
                        },
                    },
                    { upsert: true }
                )

                const status = result.upsertedCount ? "inserted" : "updated"
                console.log(`${status} chunk ${chunkIndex + 1}/${chunks.length} for ${url}`)
            }
        }

        console.log("Seed completed.")
    } finally {
        await client.close()
    }
}

const scrapePage = async (url: string) => {
    const loader = new PuppeteerWebBaseLoader(url, {
        launchOptions: {
            headless: true,
        },
        gotoOptions: {
            waitUntil: "domcontentloaded",
        },
        evaluate: async (page, browser) => {
            const result = await page.evaluate(() => document.body.innerText)
            await browser.close()
            return result
        },
    })

    return (await loader.scrape()).replace(/<[^>]*>n\?/gm, "")
}

loadSampleData().catch((error) => {
    console.error(error)
    process.exit(1)
})
