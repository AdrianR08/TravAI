import os
import re
import time
from collections import defaultdict
import argparse
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import chromadb
from chromadb.config import Settings

class CustomSentenceTransformerEF:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def __call__(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

class ReviewRAG:
    def __init__(self,
                 llm_model_name="meta-llama/Llama-3.2-1B",
                 embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                 use_mps=True,
                 db_path="./review_db_2",
                 verbose=True):

        self.verbose = verbose
        self.use_mps = use_mps and torch.backends.mps.is_available()
        self.print("Initializing ReviewRAG system...")

        self.print("Loading embedding model...")
        start_time = time.time()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to('cuda')
        elif self.use_mps:
            self.embedding_model = self.embedding_model.to('mps')
        self.print(f"Embedding model loaded in {time.time() - start_time:.2f}s")

        self.print("Loading language model...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=300,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
        )

        self.db_path = db_path
        self.setup_vector_db()
        self.print("ReviewRAG system initialized successfully")

    def print(self, message):
        if self.verbose:
            print(message)

    def setup_vector_db(self):
        self.print(f"Setting up vector database at {self.db_path}...")
        self.db = chromadb.PersistentClient(path=self.db_path, settings=Settings(allow_reset=True))
        device = "mps" if self.use_mps else "cpu"
        self.print(f"Using embedding device: {device}")
        self.ef = CustomSentenceTransformerEF(self.embedding_model, device=device)

        try:
            self.collection = self.db.get_collection("business_reviews")
        except:
            self.collection = self.db.create_collection("business_reviews")

    def reset_vector_db(self):
        self.print("Resetting vector database...")
        self.db.reset()
        self.setup_vector_db()

    def parse_business_reviews(self, file_path, max_reviews=None):
        self.print(f"Reading business reviews from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        reviews_raw = content.split('---')
        self.print(f"Found {len(reviews_raw)} raw review entries")

        reviews = []
        pattern = re.compile(
            r'Business Name: (.*?)\nCategory: (.*?)\nReview: (.*?)(?:\nRating: (\d+))?(?:\nResponse: (.*?))?(?:\nAddress: (.*?))?(?:\nRating: ([\d\.-]+))?(?:\nGMAP ID: (.*?))?(?:\nLatitude: ([\d\.-]+))?(?:\nLongitude: ([\d\.-]+))?(?:\nDescription: (.*?))?(?:\nAverage Rating: ([\d\.-]+))?(?:\nPrice: (.*?))?(?:\nHours: (.*?))?(?:\n|$)',
            re.DOTALL)

        review_count = min(len(reviews_raw), max_reviews if max_reviews else len(reviews_raw))

        for i, review_text in enumerate(tqdm(reviews_raw[:review_count], desc="Parsing reviews")):
            if not review_text.strip():
                continue
            match = pattern.search(review_text.strip())
            if match:
                reviews.append({
                    "business_name": (match.group(1) or "").strip(),
                    "category": (match.group(2) or "").strip(),
                    "review": (match.group(3) or "").strip(),
                    "rating": (match.group(4) or "").strip(),
                    "address": (match.group(6) or "").strip(),
                    "avg_rating": (match.group(7) or "").strip(),
                    "description": (match.group(11) or "").strip()
                })

        return reviews

    def format_review_for_embedding(self, review):
        categories = review["category"].replace("'", "").replace("[", "").replace("]", "")
        review_text = review["review"][:300]
        description = review["description"][:200]

        return f"""Business: {review['business_name']}\nCategories: {categories}\nDescription: {description}\nAddress: {review['address']}\nAverage Rating: {review['avg_rating']}\nReview: {review_text}\nRating: {review['rating']}"""

    def batch_embed_texts(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def index_reviews(self, reviews, batch_size=512):
        self.print(f"Indexing {len(reviews)} reviews into the vector database...")
        total_batches = (len(reviews) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Indexing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(reviews))
            batch = reviews[start_idx:end_idx]
            ids = [f"review_{start_idx + i}" for i in range(len(batch))]
            documents = [self.format_review_for_embedding(r) for r in batch]
            metadatas = [r for r in batch]
            embeddings = self.batch_embed_texts(documents)

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

    def query(self, query_text, n_results=5):
        query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=False).tolist()
        return self.collection.query(query_embeddings=[query_embedding], n_results=n_results)

    def generate_answer(self, query, retrieved_reviews):
        prompt = f"""
    You are TravAI, a helpful and friendly local travel assistant. Use the real customer reviews below to recommend **one specific place** for the user’s request.

    Only recommend **one business** and give a short, enthusiastic reason based on what the reviews said. Be natural, like you're talking to a friend planning a trip.

    ---

    User's Question: "{query}"

    Top Reviews:
    """

        # Use only up to 3 reviews for clarity
        selected_reviews = zip(
            retrieved_reviews["documents"][0][:3],
            retrieved_reviews["metadatas"][0][:3]
        )

        for doc, meta in selected_reviews:
            name = meta.get("business_name", "Unknown")
            category = meta.get("category", "N/A")
            rating = meta.get("rating", "N/A")
            review = meta.get("review", "No review.")
            prompt += f"- {name} ({category}, ⭐ {rating}): {review.strip()[:200]}...\n"

        prompt += "\nNow based on the reviews above, recommend ONE place and explain briefly why it's the best fit:\nAnswer:"

        response = self.generator(
            prompt,
            return_full_text=False,
            do_sample=True,
            temperature=0.6,
            max_new_tokens=250
        )[0]["generated_text"].strip()

        return response if response else "I couldn’t generate a confident answer. Try rephrasing your question!"

    def process_query(self, query, n_results=5):
        self.print(f"Processing query: {query}")
        if self.collection.count() == 0:
            return {"query": query, "answer": "No documents found. Please index reviews first.", "retrieved_reviews": None, "retrieval_time": 0, "generation_time": 0}

        start = time.time()
        retrieved = self.query(query, n_results)
        retrieval_time = time.time() - start

        raw_reviews = list(zip(retrieved["documents"][0], retrieved["metadatas"][0]))
        grouped = defaultdict(list)
        for doc, meta in raw_reviews:
            grouped[meta.get("business_name", "Unknown")].append((doc, meta))

        top_reviews = []
        for business, entries in grouped.items():
            top_reviews.append(entries[0])
            if len(top_reviews) >= 3:
                break

        docs = [doc for doc, _ in top_reviews]
        metas = [meta for _, meta in top_reviews]

        start = time.time()
        answer = self.generate_answer(query, retrieved)
        generation_time = time.time() - start

        return {
            "query": query,
            "answer": answer,
            "retrieved_reviews": {
                "documents": [docs],
                "metadatas": [metas]
            },
            "retrieval_time": retrieval_time,
            "generation_time": generation_time
        }
