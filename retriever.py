# =========================================================
# retriever.py
# SentinelAI Industrial RAG Retriever
# =========================================================

from pathlib import Path
import os

# =========================================================
# LOCAL HF CACHE (Windows-safe)
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

HF_CACHE_DIR = SCRIPT_DIR / "hf_cache"

HF_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)

from sentence_transformers import SentenceTransformer

import faiss
import json
import re


# =========================================================
# CONFIG
# =========================================================

TOP_K_DEFAULT = 5

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEBUG = True


# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS = {

    "the",
    "is",
    "a",
    "an",
    "of",
    "to",
    "in",
    "and",
    "or",
    "for",
    "on",
    "with",
    "after",
    "before",
    "under",
    "into",
    "from",
    "by",
    "at",
    "as",
    "that",
    "this",
    "it",
    "be",
    "are"
}


# =========================================================
# SECTION WEIGHTS
# =========================================================

SECTION_WEIGHTS = {

    "alarm_fault": 0.08,

    "troubleshooting": 0.07,

    "repair": 0.05,

    "maintenance": 0.04,

    "electrical": 0.03,

    "inspection": -0.01,

    "specification": -0.06,

    "safety": -0.04,

    "legal": -0.08
}


# =========================================================
# TROUBLESHOOTING INTENT KEYWORDS
# =========================================================

TROUBLESHOOTING_KEYWORDS = {

    "fault",
    "failure",
    "issue",
    "problem",
    "alarm",
    "overheating",
    "temperature",
    "vibration",
    "noise",
    "leak",
    "jam",
    "pressure",
    "warning",
    "abnormal",
    "error"
}


# =========================================================
# DOMAIN KEYWORDS
# =========================================================

DOMAIN_KEYWORDS = {

    "bearing",
    "lubrication",
    "coolant",
    "hydraulic",
    "pressure",
    "temperature",
    "vibration",
    "spindle",
    "motor",
    "roller",
    "belt",
    "alignment",
    "cylinder",
    "oil",
    "welding",
    "servo",
    "current"
}


# =========================================================
# SOURCE QUALITY WEIGHTS
# =========================================================

SOURCE_WEIGHTS = {

    "CNC2.pdf": 0.05,

    "Press3.pdf": 0.04,

    "Belt2.pdf": 0.03,

    "Arm1.pdf": 0.03
}


# =========================================================
# PATHS
# =========================================================

KB_DIR = SCRIPT_DIR / "kb"

INDEX_PATH = KB_DIR / "index.faiss"

METADATA_PATH = KB_DIR / "metadata.json"


# =========================================================
# RETRIEVER CLASS
# =========================================================

class Retriever:

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self):

        if DEBUG:
            print("\n[Retriever] Initializing retriever...\n")

        # -------------------------------------------------
        # LOAD FAISS INDEX
        # -------------------------------------------------

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        # -------------------------------------------------
        # LOAD METADATA
        # -------------------------------------------------

        with open(METADATA_PATH, "r", encoding="utf-8") as f:

            self.metadata = json.load(f)

        # -------------------------------------------------
        # SEARCH ENTIRE CORPUS
        # -------------------------------------------------

        self.initial_k = len(self.metadata)

        # -------------------------------------------------
        # LOAD EMBEDDING MODEL
        # -------------------------------------------------

        self.embedding_model = SentenceTransformer(
            EMBED_MODEL_NAME
        )

        if DEBUG:

            print(f"[Retriever] Metadata records     : {len(self.metadata)}")

            print(f"[Retriever] FAISS vectors        : {self.index.ntotal}")

            print(f"[Retriever] Initial retrieval K : {self.initial_k}")

            print("[Retriever] Ready.\n")


    # =====================================================
    # QUERY NORMALIZATION
    # =====================================================

    def normalize_query(self, query):

        query = query.lower().strip()

        replacements = {

            "temp": "temperature",
            "temps": "temperature",

            "vibe": "vibration",
            "vibes": "vibration",

            "press": "pressure",

            "motor current": "current",

            "overheat": "overheating"
        }

        for old, new in replacements.items():

            query = re.sub(
                rf"\b{re.escape(old)}\b",
                new,
                query
            )

        query = re.sub(r"\s+", " ", query)

        return query


    # =====================================================
    # JACCARD SCORE
    # =====================================================

    def compute_jaccard(self, query, text):

        query_words = {

            word
            for word in query.lower().split()
            if word not in STOPWORDS
        }

        text_words = {

            word
            for word in text.lower().split()
            if word not in STOPWORDS
        }

        if len(query_words) == 0:
            return 0.0

        intersection = query_words.intersection(text_words)

        union = query_words.union(text_words)

        score = len(intersection) / len(union)

        return round(score, 4)


    # =====================================================
    # METADATA BOOSTS
    # =====================================================

    def compute_metadata_boost(self, result, query):

        boost = 0.0

        query_lower = query.lower()

        query_words = set(query_lower.split())

        section = result["section_type"]

        source = result["source"]

        text = result["text"].lower()

        # -------------------------------------------------
        # SECTION WEIGHTS
        # -------------------------------------------------

        boost += SECTION_WEIGHTS.get(section, 0.0)

        # -------------------------------------------------
        # QUERY INTENT DETECTION
        # -------------------------------------------------

        has_troubleshooting_intent = any(

            keyword in query_words

            for keyword in TROUBLESHOOTING_KEYWORDS
        )

        if has_troubleshooting_intent:

            if section in {

                "alarm_fault",
                "troubleshooting",
                "repair"
            }:

                boost += 0.03

        # -------------------------------------------------
        # DOMAIN KEYWORD BOOSTING
        # -------------------------------------------------

        keyword_hits = 0

        for keyword in DOMAIN_KEYWORDS:

            if keyword in query_words and keyword in text:

                keyword_hits += 1

        boost += min(keyword_hits * 0.02, 0.08)

        # -------------------------------------------------
        # SIGNAL BOOSTS
        # -------------------------------------------------

        signals = result.get("signals", [])

        for signal in signals:

            if signal.lower() in query_lower:

                boost += 0.05

        # -------------------------------------------------
        # SOURCE QUALITY WEIGHTS
        # -------------------------------------------------

        boost += SOURCE_WEIGHTS.get(source, 0.0)

        return round(boost, 4)


    # =====================================================
    # MAIN RETRIEVAL
    # =====================================================

    def retrieve(
        self,
        query,
        machine_type,
        top_k=TOP_K_DEFAULT
    ):

        # -------------------------------------------------
        # NORMALIZE QUERY
        # -------------------------------------------------

        query = self.normalize_query(query)

        if DEBUG:

            print("=" * 60)
            print("[Retriever] QUERY")
            print("=" * 60)

            print(query)

            print("\n[Retriever] MACHINE FILTER:")

            print(machine_type)

        # -------------------------------------------------
        # EMBED QUERY
        # -------------------------------------------------

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # -------------------------------------------------
        # GLOBAL SEARCH
        # -------------------------------------------------

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            self.initial_k
        )

        scores = scores[0]

        indices = indices[0]

        if DEBUG:

            print(f"\n[Retriever] Initial candidates: {len(indices)}")

        # -------------------------------------------------
        # MACHINE FILTER
        # -------------------------------------------------

        filtered_results = []

        for cosine_score, idx in zip(scores, indices):

            meta = self.metadata[idx]

            # ---------------------------------------------
            # STRICT MACHINE FILTER
            # ---------------------------------------------

            if meta["machine_type"] != machine_type:
                continue
            # ---------------------------------------------
            # MINIMUM COSINE QUALITY
            # ---------------------------------------------

            if cosine_score < 0.18:
                continue
            # ---------------------------------------------
            # JACCARD SCORE
            # ---------------------------------------------

            jaccard_score = self.compute_jaccard(
                query,
                meta["text"]
            )

            # ---------------------------------------------
            # METADATA BOOST
            # ---------------------------------------------

            metadata_boost = self.compute_metadata_boost(
                meta,
                query
            )

            # ---------------------------------------------
            # FINAL SCORE
            # ---------------------------------------------

            final_score = (
                (0.8 * float(cosine_score))
                +
                (0.2 * float(jaccard_score))
                +
                metadata_boost
            )


            # ---------------------------------------------
            # MINIMUM FINAL QUALITY
            # ---------------------------------------------

            if final_score < 0.20:
                continue

            result = {

                "text": meta["text"],

                "source": meta["source"],

                "page": meta["page"],

                "machine_type": meta["machine_type"],

                "section_type": meta["section_type"],

                "signals": meta.get("signals", []),

                "failure_modes": meta.get("failure_modes", []),

                "cosine_score": round(float(cosine_score), 4),

                "jaccard_score": round(float(jaccard_score), 4),

                "metadata_boost": round(float(metadata_boost), 4),

                "final_score": round(float(final_score), 4)
            }

            filtered_results.append(result)

        if DEBUG:

            print(f"[Retriever] After machine filter: {len(filtered_results)}")

        # -------------------------------------------------
        # SORT BY FINAL SCORE
        # -------------------------------------------------

        filtered_results = sorted(
            filtered_results,
            key=lambda x: x["final_score"],
            reverse=True
        )

        # -------------------------------------------------
        # SOURCE + PAGE DEDUPING
        # MAX:
        # - 2 chunks per source
        # - 1 chunk per page
        # -------------------------------------------------

        deduped_results = []

        source_counter = {}

        seen_pages = set()

        for result in filtered_results:

            source = result["source"]

            page = result["page"]

            page_key = (source, page)

            # ---------------------------------------------
            # PAGE-LEVEL DEDUPE
            # ---------------------------------------------

            if page_key in seen_pages:
                continue

            # ---------------------------------------------
            # SOURCE LIMIT
            # ---------------------------------------------

            current_count = source_counter.get(source, 0)

            if current_count >= 2:
                continue

            # ---------------------------------------------
            # ACCEPT RESULT
            # ---------------------------------------------

            deduped_results.append(result)

            source_counter[source] = current_count + 1

            seen_pages.add(page_key)

            # ---------------------------------------------
            # STOP AT TOP_K
            # ---------------------------------------------

            if len(deduped_results) >= top_k:
                break
        
        # -------------------------------------------------
        # DEBUG OUTPUT
        # -------------------------------------------------

        if DEBUG:

            print(f"[Retriever] Final results: {len(deduped_results)}")

            print("\nTOP RESULTS:\n")

            for idx, r in enumerate(deduped_results, start=1):

                print("-" * 50)

                print(f"RANK            : {idx}")

                print(f"SOURCE          : {r['source']}")

                print(f"PAGE            : {r['page']}")

                print(f"SECTION         : {r['section_type']}")

                print(f"COSINE          : {r['cosine_score']}")

                print(f"JACCARD         : {r['jaccard_score']}")

                print(f"BOOST           : {r['metadata_boost']}")

                print(f"FINAL           : {r['final_score']}")

                print("\nPREVIEW:")

                print(r["text"][:250])

                print()

        return deduped_results


# =========================================================
# GLOBAL SINGLETON
# =========================================================

retriever = Retriever()


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    query = "hydraulic pressure overheating"

    machine_type = "hydraulic_press"

    results = retriever.retrieve(
        query=query,
        machine_type=machine_type,
        top_k=5
    )

    print("\n")
    print("=" * 60)
    print("FINAL RETRIEVAL COMPLETE")
    print("=" * 60)