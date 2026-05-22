import os
import faiss
import pickle
import numpy as np

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from transformers import pipeline


# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# =========================
# LOAD FAISS INDEX
# =========================

index = faiss.read_index(
    "nigeria_states.index"
)

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


# =========================
# EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# BM25 SETUP
# =========================

tokenized_chunks = [
    chunk.split(" ")
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# =========================
# LOAD OPEN-SOURCE LLM
# =========================

# Phi-3 Mini is lightweight
# and works well on HF Spaces

text_generator = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    token=HF_TOKEN
)


# =========================
# DENSE RETRIEVAL
# =========================

def dense_retrieve(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = [
        chunks[idx]
        for idx in indices[0]
    ]

    return results


# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(query, top_k=5):

    dense_results = dense_retrieve(
        query,
        top_k
    )

    tokenized_query = query.split(" ")

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:top_k]

    bm25_results = [
        chunks[idx]
        for idx in bm25_indices
    ]

    merged_results = list(
        dict.fromkeys(
            dense_results + bm25_results
        )
    )

    return merged_results[:top_k]


# =========================
# PROMPT ENGINEERING
# =========================

def build_prompt(query, contexts):

    context_text = "\n\n".join(contexts)

    prompt = f"""
    You are a Nigeria State Intelligence Assistant.

    Use ONLY the provided context.

    Context:
    {context_text}

    Question:
    {query}

    Answer:
    """

    return prompt


# =========================
# GENERATION
# =========================

def generate_response(query):

    contexts = hybrid_retrieve(query)

    prompt = build_prompt(
        query,
        contexts
    )

    response = text_generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return response[0]["generated_text"]