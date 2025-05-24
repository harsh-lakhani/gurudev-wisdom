from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from datetime import datetime
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup ChromaDB
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="book_rag",
    embedding_function=embedding_func
)

# Load book chunks
with open("book_chunks.txt", "r") as file:
    chunks = file.read().split("\n\n---CHUNK---\n\n")
for i, chunk in enumerate(chunks):
    collection.add(documents=[chunk], ids=[str(i)])

# Rate limiting storage
question_counts = {}

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
    
    # 5-question daily limit
    date = datetime.now().strftime("%Y-%m-%d")
    key = f"{ip}-{date}"
    question_counts[key] = question_counts.get(key, 0) + 1
    
    if question_counts[key] > 5:
        return jsonify({
            "response": "Daily limit reached (5 questions). Please return tomorrow.",
            "limit_reached": True
        })
    
    # Get book context
    results = collection.query(query_texts=[question], n_results=2)
    context = "\n\n".join(results["documents"][0])
    
    # Generate response
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system", 
                "content": f"Answer as Gurudev would. If a similar answer exists in this context, quote it prominently. Begin with: 'We believe this is how Gurudev would respond:'\n{context}"
            },
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    # Log interaction
    log_interaction(ip, question, response.choices[0].message.content, response.usage)
    
    return jsonify({
        "response": response.choices[0].message.content,
        "questions_left": 5 - question_counts[key]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))