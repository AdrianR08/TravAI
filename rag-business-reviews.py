#!/usr/bin/env python3
import os
import re
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer,pipeline, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class ReviewRAG:

    def __init__(self,
                 llm_model_name="meta-llama/Llama-3.2-1B",
                 embedding_model_name="MPNet-base-v2",
                 use_mps=True,
                 db_path="./review_db_2",
                 verbose=True):

        self.verbose = verbose
        self.use_mps = use_mps and torch.backends.mps.is_available()
        self.print("Initializing ReviewRAG system...")

        # Set up embedding model
        self.print("Loading embedding model...")
        start_time = time.time()
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        if self.use_mps:
            self.embedding_model = self.embedding_model.to('mps')
        self.print(f"Embedding model loaded in {time.time() - start_time:.2f}s")

        # Set up LLM for generation
        self.print(f"Loading language model {llm_model_name}...")
        start_time = time.time()

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        # Move to MPS if available
        if self.use_mps:
            try:
                self.llm = self.llm.to("mps")
                self.print("Model successfully moved to MPS")
            except Exception as e:
                self.print(f"Failed to move model to MPS: {e}")
                self.print("Continuing with CPU")

        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

        self.print(f"Language model loaded in {time.time() - start_time:.2f}s")

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
            self.collection = self.db.get_collection("business_reviews_2")
            doc_count = self.collection.count()
            self.print(f"Found existing collection with {doc_count} documents")
            if doc_count == 0:
                self.print("Warning: Collection exists but contains 0 documents")
        except Exception as e:
            self.print(f"Collection not found: {e}")
            self.print("Creating new collection 'business_reviews_2'")
            self.collection = self.db.create_collection("business_reviews_2")

    def reset_vector_db(self):
        self.print("Resetting vector database...")
        self.db.reset()
        self.setup_vector_db()

    def parse_business_reviews(self, file_path, max_reviews=None):
        self.print(f"Reading business reviews from {file_path}")
        start_time = time.time()

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
                business_name = match.group(1) or ""
                category = match.group(2) or ""
                review = match.group(3) or ""
                rating = match.group(4) or ""
                response = match.group(5) or ""
                address = match.group(6) or ""
                avg_rating = match.group(7) or ""
                description = match.group(11) or ""

                reviews.append({
                    "business_name": business_name.strip(),
                    "category": str(category.strip()),
                    "review": review.strip(),
                    "rating": str(rating.strip()),
                    "response": "None" if response is None or response.strip() == "None" else response.strip(),
                    "address": address.strip(),
                    "avg_rating": str(avg_rating.strip()),
                    "description": description.strip()
                })
            else:
                if i % 1000 == 0:  
                    self.print(f"Failed to parse review {i}")

        elapsed = time.time() - start_time
        items_per_second = len(reviews) / elapsed
        self.print(f"Parsed {len(reviews)} reviews in {elapsed:.2f}s ({items_per_second:.2f} reviews/second)")

        return reviews

    def format_review_for_embedding(self, review):
        categories = review["category"].replace("'", "").replace("[", "").replace("]", "")

        formatted = f"""Business: {review["business_name"]}
Categories: {categories}
Description: {review["description"]}
Address: {review["address"]}
Average Rating: {review["avg_rating"]}
Review: {review["review"]}
Rating: {review["rating"]}"""

        if review["response"] and review["response"] != "None":
            formatted += f"\nResponse: {review['response']}"

        return formatted

    def format_review_for_display(self, review):
        categories = review["category"].replace("'", "").replace("[", "").replace("]", "")

        formatted = f"""Business: {review["business_name"]}
Categories: {categories}
Description: {review["description"]}
Address: {review["address"]}
Average Rating: {review["avg_rating"]}
Review: {review["review"]}
Rating: {review["rating"]}"""

        if review["response"] and review["response"] != "None":
            formatted += f"\nResponse: {review['response']}"

        return formatted

    def batch_embed_texts(self, texts, batch_size=32):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def index_reviews(self, reviews, batch_size=100):
        self.print(f"Indexing {len(reviews)} reviews into the vector database...")
        start_time = time.time()

        total_batches = (len(reviews) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(reviews))
            batch = reviews[start_idx:end_idx]

            self.print(f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")

            ids = [f"review_{start_idx + i}" for i in range(len(batch))]
            documents = [self.format_review_for_embedding(review) for review in batch]

            metadatas = []
            for review in batch:
                metadata = {}
                for key, value in review.items():
                    if isinstance(value, str):
                        metadata[key] = value[:1000] 
                    else:
                        metadata[key] = str(value)[:1000]
                metadatas.append(metadata)

            embeddings = self.batch_embed_texts(documents)

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

        elapsed = time.time() - start_time
        items_per_second = len(reviews) / elapsed
        self.print(f"Indexed {len(reviews)} reviews in {elapsed:.2f}s ({items_per_second:.2f} reviews/second)")

    def query(self, query_text, n_results=5):
        query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=False).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results

    def generate_answer(self, query, retrieved_reviews):
        # Format prompt with retrieved reviews
        prompt = f"""You are a reviews assistant. You have access to Google reviews data. 
Answer the following question based on the information in the reviews provided below.
If the reviews don't contain relevant information to answer the question directly, synthesize what you can learn from them.
Provide a clear, concise response based on the review content.

Question: {query}

Relevant Reviews:
"""

        for i, review in enumerate(retrieved_reviews["documents"][0]):
            prompt += f"\n--- Review {i + 1} ---\n{review}\n"

        prompt += "\nBased on these reviews, here's the answer to the question:"

        self.print("Generating answer...")
        response = self.generator(
            prompt,
            return_full_text=False,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=250
        )[0]["generated_text"]

        response = response.strip()

        if not response or response.isspace():
            response = "I couldn't generate a specific answer from the reviews. Please try reformulating your question."

        return response

    def process_query(self, query, n_results=5):
        self.print(f"Processing query: {query}")

        if self.collection.count() == 0:
            self.print("Error: No documents in the collection. Please index reviews first.")
            return {
                "query": query,
                "answer": "No reviews have been indexed yet. Please run with --data_file and --reset options first.",
                "retrieved_reviews": None,
                "retrieval_time": 0,
                "generation_time": 0
            }

        start_time = time.time()
        retrieved = self.query(query, n_results=n_results)
        retrieval_time = time.time() - start_time
        self.print(f"Retrieved {len(retrieved['documents'][0])} relevant reviews in {retrieval_time:.2f}s")

        start_time = time.time()
        answer = self.generate_answer(query, retrieved)
        generation_time = time.time() - start_time
        self.print(f"Generated answer in {generation_time:.2f}s")

        return {
            "query": query,
            "answer": answer,
            "retrieved_reviews": retrieved,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time
        }


class CustomSentenceTransformerEF:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def __call__(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)


def display_review_details(result, detail_level=0):
    if not result or not result.get("retrieved_reviews"):
        return

    reviews = result["retrieved_reviews"]
    print("\n=== RETRIEVED REVIEWS ===")

    for i, (doc, metadata) in enumerate(zip(reviews["documents"][0], reviews["metadatas"][0])):
        print(f"\n[Review {i + 1}] {metadata.get('business_name', 'Unknown Business')}")
        if detail_level > 0:
            print(f"Categories: {metadata.get('category', 'N/A')}")
            print(f"Rating: {metadata.get('rating', 'N/A')}")
            if detail_level > 1:
                print(f"Review: {metadata.get('review', 'N/A')}")
                if metadata.get('response') and metadata.get('response') != 'None':
                    print(f"Response: {metadata.get('response')}")


def main():
    parser = argparse.ArgumentParser(description="ReviewRAG: Retrieval-Augmented Generation for Business Reviews")
    parser.add_argument("--data_file", type=str, help="Path to business reviews file")
    parser.add_argument("--db_path", type=str, default="./review_db_2", help="Path to vector database")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-1B", help="Language model to use")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="Embedding model to use")
    parser.add_argument("--max_reviews", type=int, default=10000, help="Maximum number of reviews to index")
    parser.add_argument("--reset", action="store_true", help="Reset the vector database")
    parser.add_argument("--index_only", action="store_true", help="Only index the reviews, don't query")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode after indexing")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--use_mps", action="store_true", default=True, help="Use MPS (Metal) if available")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage even if MPS is available")
    parser.add_argument("--n_results", type=int, default=5, help="Number of results to retrieve for each query")
    parser.add_argument("--show_reviews", action="store_true", help="Show details of retrieved reviews")
    parser.add_argument("--review_details", type=int, default=0, choices=[0, 1, 2],
                        help="Level of review details to show (0=minimal, 1=basic, 2=full)")
    args = parser.parse_args()

    if args.reset and not args.data_file:
        print("Error: --reset requires --data_file to be specified")
        return

    if args.index_only and not args.data_file:
        print("Error: --index_only requires --data_file to be specified")
        return

    use_mps = args.use_mps and not args.use_cpu and torch.backends.mps.is_available()

    os.makedirs(args.db_path, exist_ok=True)

    try:
        rag = ReviewRAG(
            llm_model_name=args.llm_model,
            embedding_model_name=args.embedding_model,
            use_mps=use_mps,
            db_path=args.db_path
        )

        if args.reset:
            rag.reset_vector_db()

        if args.data_file:
            reviews = rag.parse_business_reviews(args.data_file, max_reviews=args.max_reviews)
            rag.index_reviews(reviews)

            if args.index_only:
                print("Indexing complete. Exiting.")
                return

        if args.query:
            result = rag.process_query(args.query, n_results=args.n_results)
            print("\n=== ANSWER ===")
            if result["answer"]:
                print(result["answer"])
            else:
                print("No answer generated. There might be an issue with the language model or the retrieved reviews.")

            if args.show_reviews or args.review_details > 0:
                display_review_details(result, args.review_details)

        if args.interactive:
            print("\nEntering interactive mode. Type 'exit' to quit.")
            while True:
                query = input("\nEnter your question: ")
                if query.lower() in ["exit", "quit", "q"]:
                    break

                result = rag.process_query(query, n_results=args.n_results)
                print("\n=== ANSWER ===")
                if result["answer"]:
                    print(result["answer"])
                else:
                    print("No answer generated. Try reformulating your question.")

                if args.show_reviews or args.review_details > 0:
                    display_review_details(result, args.review_details)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Exiting.")
    except Exception as e:
        print(f"\nError: {e}")
        print("An unexpected error occurred. Check your inputs and try again.")


if __name__ == "__main__":
    main()
