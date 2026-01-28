// Native fetch is available in Node > 18
// const fetch = require('node-fetch');

async function testChat() {
    try {
        const response = await fetch('http://localhost:3000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: "What is hypertension?" })
        });
        const data = await response.json();
        console.log("Status:", response.status);
        console.log("Response:", JSON.stringify(data, null, 2));
    } catch (error) {
        console.error("Request failed:", error.message);
    }
}

testChat();
