from collections import defaultdict
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag_business_reviews import ReviewRAG
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from typing import Optional, List, Dict
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

rag = ReviewRAG(
    llm_model_name="meta-llama/Llama-3.2-1B",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    db_path="review_db_2",
    use_mps=True
)

rag.generator = pipeline(
    "text-generation",
    model=rag.llm,
    tokenizer=rag.llm_tokenizer,
    min_new_tokens=100,
    max_new_tokens=500,
    temperature=0.5,
    top_p=0.9,
    repetition_penalty=1.1
)

def format_retrieved_reviews(retrieved):
    if not retrieved or not retrieved.get("documents"):
        return []

    metas = retrieved["metadatas"][0]

    return [
        {
            "business_name": meta.get("business_name", "Unknown"),
            "category": meta.get("category", "N/A"),
            "rating": meta.get("rating", "N/A")
        }
        for meta in metas
    ]

@app.get("/")
def home():
    return {"message": "Backend is running!"}

def deduplicate_reviews(reviews: List[Dict]) -> List[Dict]:
    seen = set()
    unique_reviews = []
    for review in reviews:
        review_text = review.get("review", "").strip().lower()
        if review_text and review_text not in seen:
            seen.add(review_text)
            unique_reviews.append(review)
    return unique_reviews

@app.post("/index_reviews")
async def index_reviews(data_file: UploadFile = File(...), max_reviews: Optional[int] = 10000):
    os.makedirs("./temp_uploads", exist_ok=True)
    temp_path = f"./temp_uploads/{data_file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(data_file.file, buffer)

    reviews = rag.parse_business_reviews(temp_path, max_reviews=max_reviews)
    reviews = deduplicate_reviews(reviews)
    rag.index_reviews(reviews)
    os.remove(temp_path)

    return {"message": f"Indexed {len(reviews)} reviews from {data_file.filename}"}

class QueryInput(BaseModel):
    prompt: str

@app.post("/recommend")
async def recommend(query: QueryInput):
    print(f"✅ /recommend endpoint hit\n👉 Prompt received: {query.prompt}")
    raw_result = rag.process_query(query.prompt, n_results=20)
    if not raw_result["retrieved_reviews"] or not raw_result["retrieved_reviews"].get("documents"):
        return {
            "answer": "No reviews are available yet. Please upload and index reviews first.",
            "reviews": [],
            "retrieval_time": 0,
            "generation_time": 0
        }

    formatted_reviews = format_retrieved_reviews(raw_result["retrieved_reviews"])

    print("✅ Query processed and formatted")

    return {
        "answer": raw_result["answer"],
        "reviews": formatted_reviews,
        "retrieval_time": raw_result["retrieval_time"],
        "generation_time": raw_result["generation_time"]
    }