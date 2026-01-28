require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { Pinecone } = require('@pinecone-database/pinecone');
const { CohereClient } = require('cohere-ai');

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
const requiredEnv = ['GOOGLE_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX', 'COHERE_API_KEY'];
const missingEnv = requiredEnv.filter(key => !process.env[key]);

if (missingEnv.length > 0) {
    console.error("❌ MISSING API KEYS:", missingEnv.join(", "));
    console.error("Please fill them in your .env file.");
    process.exit(1);
}

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX);
const cohere = new CohereClient({ token: process.env.COHERE_API_KEY });

/**
 * Generates embeddings using Google's gemini-embedding-001 model.
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
                output_dimensionality: 768 // IMPORTANT: Matches your medical database
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
app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
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
            .filter(text => text !== undefined);

        console.log(`✅ Retrieved ${contexts.length} relevant sections.`);

        // 4. Generate structured response with Cohere
        console.log("Synthesizing answer...");
        const chatResponse = await cohere.chat({
            model: 'command-r-08-2024',
            message: message,
            documents: contexts.map((text, i) => ({
                title: `Medical Context ${i + 1}`,
                snippet: text
            })),
            preamble: `You are an expert medical assistant and knowledgeable AI chatbot specializing in internal medicine. Your goal is to assist users by providing accurate, professional, and concise medical information based strictly on the provided context from Harrison's Principles of Internal Medicine.

Instructions:
1. Use a professional and empathetic tone.
2. Answer ONLY using the context provided below. Do not hallucinate information.
3. If the answer is not found in the context, explicitly state: "I cannot find that information in the provided medical records."
4. Format your response clearly using Markdown (bullet points or numbered lists).`
        });

        console.log("\n🤖 Response Generated Successfully.");

        res.json({
            answer: chatResponse.text,
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

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`\n🚀 Medical Chatbot Server Live on port ${PORT}`);
    console.log(`📡 Connected to Pinecone Index: ${process.env.PINECONE_INDEX}`);
});
