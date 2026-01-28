require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { Pinecone } = require('@pinecone-database/pinecone');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
app.use(cors());
app.use(express.json());

// --- ERROR HANDLING MIDDLEWARE ---
app.use((err, req, res, next) => {
    if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
        console.error(`❌ Bad JSON Request: ${err.message}`);
        return res.status(400).json({ error: "Invalid JSON format. Please check your request body." });
    }
    next();
});

// --- CONFIGURATION CHECKS ---
const requiredEnv = ['GOOGLE_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX'];
const missingEnv = requiredEnv.filter(key => !process.env[key]);

if (missingEnv.length > 0) {
    console.error("❌ MISSING API KEYS:", missingEnv.join(", "));
    console.error("Please fill them in your .env file.");
    process.exit(1);
}

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX);
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);

/**
 * Generates embeddings using Google's gemini-embedding-001 model via REST.
 * Forces output to 768 dimensions to match the existing Pinecone index.
 */
async function generateEmbedding(text) {
    const apiKey = process.env.GOOGLE_API_KEY;
    const modelName = "gemini-embedding-001";

    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:embedContent?key=${apiKey}`;
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                content: { parts: [{ text: text }] },
                output_dimensionality: 3072
            })
        });

        const data = await response.json();
        if (data.embedding && data.embedding.values) {
            return data.embedding.values;
        } else {
            throw new Error(`Google API Prediction Error: ${JSON.stringify(data.error || data)}`);
        }
    } catch (error) {
        console.error("Embedding Generation Error:", error.message);
        throw error;
    }
}

// --- CHAT ENDPOINT ---
// --- HEALTH CHECK FOR AWS APP RUNNER ---
app.get('/health', (req, res) => {
    res.status(200).json({ status: "ok", service: "medical-chatbot" });
});

app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });

        console.log(`\n💬 User Query: "${message}"`);

        // 1. Convert user query to vector
        const queryVector = await generateEmbedding(message);

        // 2. Query Pinecone (Harrison's Medical Manual)
        console.log("Searching medical knowledge base...");
        const searchResponse = await index.query({
            vector: queryVector,
            topK: 5,
            includeMetadata: true
        });

        // 3. Extract relevant medical text
        const contexts = searchResponse.matches
            .map(match => match.metadata.text)
            .filter(text => text !== undefined)
            .join("\n\n---\n\n");

        console.log(`✅ Retrieved ${searchResponse.matches.length} relevant sections.`);

        // 4. Generate structured response with Gemini
        console.log("⚡ Synthesizing answer with Gemini (gemini-2.0-flash)...");
        const chatModel = genAI.getGenerativeModel({
            model: "gemini-2.0-flash",
            systemInstruction: `You are an expert medical assistant and knowledgeable AI chatbot specializing in internal medicine. Your goal is to assist users by providing accurate, professional, and concise medical information based strictly on the provided context from Harrison's Principles of Internal Medicine.

Instructions:
1. Use a professional and empathetic tone.
2. Answer ONLY using the context provided below. Do not hallucinate information.
3. If the answer is not found in the context, explicitly state: "I cannot find that information in the provided medical records."
4. Format your response clearly using Markdown (bullet points or numbered lists).`
        });

        const prompt = `Context from medical records:\n${contexts}\n\nUser Question: ${message}`;

        const startTime = Date.now();
        const result = await chatModel.generateContent(prompt);
        const response = await result.response;
        const text = response.text();
        const duration = ((Date.now() - startTime) / 1000).toFixed(2);

        console.log(`\n🤖 Bot Response (Generated in ${duration}s):`);
        console.log("----------------------------------------");
        console.log(text);
        console.log("----------------------------------------");

        res.json({
            answer: text,
            citations: searchResponse.matches.map(m => m.metadata.source || `Medical Manual P.${m.metadata.page || "?"}`)
        });

    } catch (error) {
        console.error("❌ CHAT ERROR:", error.message);
        res.status(500).json({
            error: "An error occurred while processing your request.",
            details: error.message
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`\n🚀 Medical Chatbot Server Live on port ${PORT}`);
    console.log(`📡 Connected to Pinecone Index: ${process.env.PINECONE_INDEX}`);
});
