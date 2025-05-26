from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from flask import send_file

app = Flask(__name__)
CORS(app)

@app.route('/usage-log')
def get_usage_log():
    return send_file('usage.log', as_attachment=True)

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone client (new API)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "book-chunks-openai"
index = pc.Index(index_name)

def log_interaction(ip, question, response, usage):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cost = (usage.prompt_tokens * 0.10/1_000_000) + (usage.completion_tokens * 0.40/1_000_000)
    
    log_entry = f"""
[{timestamp}] {ip}
Q: {question}
A: {response}
Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out
Cost: ${cost:.6f}
---------------------"""
    
    with open("usage.log", "a") as f:
        f.write(log_entry)
    return cost

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    ip = request.remote_addr
    data = request.json
    question = data['question']

    # Generate embedding for question using OpenAI
    embeddings_response = client.embeddings.create(
        input=question,
        model="text-embedding-ada-002"  # or "text-embedding-3-small" if you used that for upsert
    )
    question_emb = embeddings_response.data[0].embedding

    # Query Pinecone for top 2 matches
    results = index.query(vector=question_emb, top_k=2, include_metadata=True)

    # Combine retrieved chunks as context
    context = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    # Generate response
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system", 
                "content": f"Answer as Gurudev would. If a piece of text exists in the book that is relevant to the question asked by the user, quote it. Along with quoting relevant pieces of text from the book, it is important to supplement them with a simple interpretation of that piece of text with the goal of answering the question asked by the user. It is important to evidently provide a clear answer to the User at the end of the response. Answer in first person, as if Gurudev is speaking. Try to understand his tone, he is friendly and adds a bit wittiness to his responses and makes people smile or laugh. Don't be boring, but don't be extra at the same time, be natural. Speak like a human. Begin with: 'We believe this is how Gurudev would respond:'\n{context}"
            },
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    # Log interaction
    log_interaction(ip, question, response.choices[0].message.content, response.usage)
    
    return jsonify({
        "response": response.choices[0].message.content
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
