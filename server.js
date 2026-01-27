require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { Pinecone } = require('@pinecone-database/pinecone');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { CohereClient } = require('cohere-ai');

const app = express();
app.use(cors());
app.use(express.json());

// --- CONFIGURATION ---
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX);

// Google is ONLY used for embedding (searching), not chatting
const googleAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedModel = googleAI.getGenerativeModel({ model: "text-embedding-004" });

// Cohere is used for the actual chatting
const cohere = new CohereClient({ token: process.env.COHERE_API_KEY });

// --- HELPER: RETRY LOGIC FOR GOOGLE ---
// If Google says "429 Too Many Requests", this function waits and tries again.
async function safeEmbed(text, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const result = await embedModel.embedContent(text);
            return result.embedding.values;
        } catch (error) {
            if (error.message.includes('429') || error.message.includes('Quota')) {
                console.log(`⚠️ Google Rate Limit hit. Waiting 2s... (Attempt ${i + 1})`);
                await new Promise(r => setTimeout(r, 2000 * (i + 1))); // Wait 2s, 4s, 6s
            } else {
                throw error;
            }
        }
    }
    throw new Error("Google API is too busy. Please try again later.");
}

// --- CHAT ENDPOINT ---
app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
        console.log(`\n💬 User asked: "${message}"`);

        // 1. EMBED (Search Preparation)
        // We convert the user's question into numbers using Google
        const queryVector = await safeEmbed(message);

        // 2. SEARCH (Pinecone)
        // We look for the most relevant pages in your book
        const searchResponse = await index.query({
            vector: queryVector,
            topK: 5, // Get top 5 pages
            includeMetadata: true
        });

        // 3. EXTRACT CONTEXT
        const contexts = searchResponse.matches
            .map(match => match.metadata.text)
            .filter(text => text !== undefined);

        console.log(`✅ Found ${contexts.length} relevant pages in Harrison's Manual.`);

        // 4. GENERATE ANSWER (Cohere)
        console.log("⚡ Sending query to Cohere (Model: command-r-08-2024)...");
        // We send the question + the book pages to Cohere
        const chatResponse = await cohere.chat({
            model: 'command-r-08-2024', // High quality model
            message: message,
            documents: contexts.map(text => ({ snippet: text })), // RAG capability
            preamble: "You are a helpful medical assistant. Answer the user's question using ONLY the context provided below. If the answer is not in the text, say 'I cannot find that information in Harrison's Manual'."
        });

        console.log("\n🤖 Generated Answer:");
        console.log(chatResponse.text);

        // 5. SEND BACK TO FRONTEND
        res.json({
            answer: chatResponse.text,
            citations: searchResponse.matches.map(m => `Page ${m.metadata.page}`)
        });

    } catch (error) {
        console.error("❌ ERROR:", error.message);
        res.status(500).json({ error: "Thinking failed. Please try again." });
    }
});

// Start Server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server is running on port ${PORT}`));
