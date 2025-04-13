from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from backend.rag_business_reviews import ReviewRAG
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from typing import Optional
import os
import shutil

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ReviewRAG (LLM + VectorDB)
rag = ReviewRAG(
    llm_model_name="meta-llama/Llama-3.2-1B",
    embedding_model_name="all-MiniLM-L6-v2",
    db_path="./review_db",
    use_mps=False
)

# Set up generator pipeline for text generation
rag.generator = pipeline(
    "text-generation",
    model=rag.llm,
    tokenizer=rag.llm_tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

# Auto-index from file if database is empty
if rag.collection.count() == 0:
    file_path = os.path.join("backend", "business_reviews.txt")
    if os.path.exists(file_path):
        print("📦 Auto-indexing reviews from business_reviews.txt...")
        reviews = rag.parse_business_reviews(file_path, max_reviews=10000)
        rag.index_reviews(reviews)
        print(f"✅ Indexed {len(reviews)} reviews from file.")
    else:
        print(f"⚠️ business_reviews.txt not found at {file_path}. Cannot auto-index.")

# Format helper like CLI output
def format_retrieved_reviews(retrieved, detail_level=1):
    if not retrieved or not retrieved.get("documents"):
        return []

    formatted = []
    docs = retrieved["documents"][0]
    metas = retrieved["metadatas"][0]

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        entry = {
            "index": i + 1,
            "business_name": meta.get("business_name", "Unknown"),
            "categories": meta.get("category", "N/A"),
            "rating": meta.get("rating", "N/A")
        }
        if detail_level > 1:
            entry["review"] = meta.get("review", "")
            entry["response"] = meta.get("response", "")
        formatted.append(entry)

    return formatted

# === ROUTES ===

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/reset_db")
def reset_db():
    rag.reset_vector_db()
    return {"message": "Vector database has been reset."}

@app.post("/index_reviews")
async def index_reviews(data_file: UploadFile = File(...), max_reviews: Optional[int] = 10000):
    os.makedirs("./temp_uploads", exist_ok=True)
    temp_path = f"./temp_uploads/{data_file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(data_file.file, buffer)

    reviews = rag.parse_business_reviews(temp_path, max_reviews=max_reviews)
    rag.index_reviews(reviews)
    os.remove(temp_path)

    return {"message": f"Indexed {len(reviews)} reviews from {data_file.filename}"}

class QueryInput(BaseModel):
    prompt: str

@app.post("/recommend")
async def recommend(query: QueryInput):
    print(f"✅ /recommend endpoint hit\n👉 Prompt received: {query.prompt}")
    result = rag.process_query(query.prompt, n_results=3)
    formatted_reviews = format_retrieved_reviews(result["retrieved_reviews"], detail_level=1)
    print("✅ Query processed and formatted")

    return {
        "answer": result["answer"],
        "reviews": formatted_reviews,
        "retrieval_time": result["retrieval_time"],
        "generation_time": result["generation_time"]
    }

@app.post("/safe_reset_db")
def safe_reset_db():
    import shutil
    rag.collection = None  # remove Chroma client reference
    del rag.collection
    del rag.client
    shutil.rmtree("./review_db", ignore_errors=True)
    return {"message": "DB reset complete"}