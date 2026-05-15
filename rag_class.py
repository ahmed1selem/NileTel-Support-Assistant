import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi  
DATA_PATH = "E:\\ITI\\Rag_pro\\last\\data"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class TelecomRAG:
    def __init__(self):
        print("Initializing TelecomRAG system...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.metadata = []
        self.index = None
        self.bm25 = None
        self.client = Groq(api_key=GROQ_API_KEY)
        self.load_data()

    # ================== LOAD DATA ==================
    def load_data(self):
        print("Loading and processing documents...")

        for file in os.listdir(DATA_PATH):
            if file.endswith(".md"):
                with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
                    text = f.read()

                doc_chunks = self.chunk_text(text)

                for chunk in doc_chunks:
                    self.chunks.append(chunk)
                    self.metadata.append({"source": file})

        if not self.chunks:
            print("ERROR: No .md files found or files were empty! Check your DATA_PATH.")
            return

        print(f" Successfully loaded {len(self.chunks)} chunks. Building indexes...")

        tokenized_corpus = [self._tokenize(chunk) for chunk in self.chunks]  
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build FAISS Semantic Index
        embeddings = self.create_embeddings(self.chunks)
        self.index = self.build_faiss_index(embeddings)

    # ================== CHUNKING ==================
    def chunk_text(self, text):
        """
        Splits text recursively based on paragraphs/lines, adds a 150-char overlap,
        and injects the document's title into every chunk for LLM context.
        """
        # 1. Extract the title (the first non-empty line of your files)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        doc_title = lines[0] if lines else "NileTel Document"

        # 2. Set up the recursive splitter with OVERLAP
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )

        raw_chunks = text_splitter.split_text(text)
        final_chunks = []

        # 3. Inject the title into every chunk
        for chunk in raw_chunks:
            if chunk.startswith(doc_title):
                enriched_chunk = chunk.strip()
            else:
                enriched_chunk = f"[{doc_title}]\n{chunk.strip()}"
            final_chunks.append(enriched_chunk)

        return final_chunks

    # ================== EMBEDDINGS ==================
    def create_embeddings(self, chunks):
        print("Creating embeddings...")
        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return np.array(embeddings).astype("float32")

    # ================== FAISS ==================
    def build_faiss_index(self, embeddings):
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        print(f"FAISS ready! Total vectors: {index.ntotal}")
        return index

    # ================== RETRIEVE ==================
    def reciprocal_rank_fusion(self, semantic_ranks, lexical_ranks, k=60):
        """Combines ranks from FAISS and BM25."""
        rrf_scores = {}

        for rank, (idx, _) in enumerate(semantic_ranks):
            if idx not in rrf_scores: rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (k + rank + 1)

        for rank, (idx, _) in enumerate(lexical_ranks):
            if idx not in rrf_scores: rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (k + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        return sorted_indices, rrf_scores

    def retrieve(self, query, top_k=6):
        print(f"Running Hybrid Search for: {query}")

        # 1. Semantic Search (FAISS)
        query_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(query_emb, 20)
        semantic_candidates = list(zip(indices[0], distances[0]))

        # 2. Lexical Search (BM25)
        tokenized_query = self._tokenize(query)  # ✅ now works
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:20]
        lexical_candidates = [(idx, bm25_scores[idx]) for idx in top_bm25_indices if bm25_scores[idx] > 0]

        # 3. Apply Reciprocal Rank Fusion (RRF)
        sorted_indices, rrf_scores = self.reciprocal_rank_fusion(semantic_candidates, lexical_candidates)

        # 4. Format Results
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "score": float(rrf_scores[idx])
            })

        print(f"Found {len(results)} hybrid chunks")
        return results

    # ================== TOKENIZER ==================
    @staticmethod
    def _tokenize(text: str):

        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)  #
        return text.split()

    # ================== ROUTING HELPERS ==================
    @staticmethod
    def normalize_arabic(text):
        text = text.lower().strip()
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _llm_route(self, query):
        prompt = f"""You are the intelligent router for NileTel, an Internet Service Provider (ISP).
Your job is to analyze the customer's query and classify it into EXACTLY ONE of the following three categories.

### Category Definitions:
1. "ticket": Use this if the customer is reporting a problem, internet outage, technical issue, router malfunction, or explicitly asking for an engineer/technical support.
2. "chat": Use this for simple greetings, pleasantries, saying thank you, or general polite conversation.
3. "out_of_scope": Use this if the customer asks about ANYTHING unrelated to internet, telecom, or NileTel services (e.g., sports, movies, cooking, coding, politics).

### Examples:
Query: النت قاطع عندي في البيت من الصبح ومحتاج مهندس
Classification: ticket

Query: رشحلي فيلم سهرة حلو اتفرج عليه
Classification: out_of_scope

Query: شكرا جدا ليك يا فندم
Classification: chat

Query: اللمبة الحمرا بتنور في الراوتر
Classification: ticket

### Task:
Now classify the following customer query. Output ONLY the classification word and nothing else.
Query: {query}
Classification:"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, 
                max_tokens=10
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"LLM Routing Error: {e}")
            return "chat"

    # ================== MAIN ROUTING ==================
    def route_query(self, query: str):
        q = self.normalize_arabic(query)

        TICKET_PATTERNS = [
            r"(اعمل|افتح|ارفع)\s*(لي)?\s*تذكره", r"عايز\s*(اعمل|افتح|ارفع)\s*تذكره",
            r"(محتاج|عايز)\s*مهندس", r"(حد|حد\s*منكم)\s*يجي",
            r"(ابعت|ابعث)\s*حد", r"في\s*عطل",
            r"النت\s*(واقع|فاصل|مش\s*شغال|مقطوع)", r"(عايز|محتاج)\s*تصعيد",
            r"المشكله\s*مستمره",
        ]
        OUT_OF_SCOPE_PATTERNS = [
            r"(فيلم|افلام|مسلسل|مسلسلات)", r"(ماتش|كوره|لاعب|مباراه)",
            r"(اكل|طبخ|وصفه|مطعم)", r"(سياسه|انتخابات|حكومه)",
            r"(اغنيه|موسيقي|فنان)", r"رشحلي\s*(فيلم|مسلسل|اغنيه)",
            r"افضل\s*(فيلم|مطعم|مسلسل)",
        ]
        CHAT_PATTERNS = [
            r"^(ازيك|عامل\s*ايه|hello|hi)$", r"^(صباح\s*الخير|مساء\s*الخير)$",
            r"(شكرا|متشكر|thanks)",
        ]

        for pattern in [re.compile(p) for p in TICKET_PATTERNS]:
            if pattern.search(q): return "ticket"

        out_count = sum(1 for pattern in [re.compile(p) for p in OUT_OF_SCOPE_PATTERNS] if pattern.search(q))
        telecom_keywords = ["نت", "واي فاي", "باقة", "سرعة", "تحميل", "فايبر", "5g", "4g", "راوتر", "اونتى"]

        if out_count >= 1 and not any(k in q for k in telecom_keywords):
            return "out_of_scope"

        for pattern in [re.compile(p) for p in CHAT_PATTERNS]:
            if pattern.search(q): return "chat"

        if len(q.split()) > 5: return self._llm_route(query)
        return "rag"


    # ================== GENERATE ==================
    def generate_answer(self, query, retrieved_results):
        if not retrieved_results:
            return {
                "answer": "مش متأكد من البيانات المتاحة يا فندم.",
                "needs_action": "NO",
                "sources": [],
                "displayed_source": "Unknown"
            }

        context = "\n\n".join([f"Source: {res['source']}\n{res['text']}" for res in retrieved_results])

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "أنت مساعد دعم عملاء محترف في شركة NileTel للاتصالات.\n"
                        "قواعدك:\n"
                        "1. أجب باللهجة المصرية الطبيعية مع الاحترام (يا فندم، تمام، هنحلها...).\n"
                        "2. استخدم فقط المعلومات الموجودة في السياق المقدم. لا تؤلف بيانات.\n"
                        "3. إذا طلب المستخدم إنشاء تذكرة أو رفع تذكرة أو إرسال مهندس أو تصعيد، انهي الرد بـ 'needs_action: YES'.\n"
                        "4. خرج دائماً needs_action: في أول سطر، ثم سطر فارغ، ثم الإجابة.\n"
                        "5. لا تختلق أرقام تذاكر أو تفاصيل وهمية."
                    )
                },
                {"role": "user", "content": f"السياق:\n{context}\n\nالسؤال: {query}"}
            ],
            temperature=0.2,
            max_tokens=800
        )

        text = response.choices[0].message.content.strip()

        needs_action_match = re.search(r'needs[_\s-]*action\s*[:=]\s*(yes|no)', text, re.IGNORECASE)
        if needs_action_match:
            needs_action = needs_action_match.group(1).upper()
        else:
            needs_action = "YES" if re.search(r"(تذكره|مهندس|تصعيد)", text) else "NO"

        answer_match = re.search(r'answer\s*[:=]\s*(.*?)(?:\n\s*needs[_\s-]*action|$)', text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            clean_answer = answer_match.group(1).strip()
        else:
            clean_answer = re.sub(r'needs[_\s-]*action\s*[:=]\s*(yes|no)', '', text, flags=re.IGNORECASE).strip()

        best = max(retrieved_results, key=lambda x: x["score"])

        return {
            "answer": clean_answer,
            "needs_action": needs_action,
            "sources": [r["source"] for r in retrieved_results],
            "displayed_source": best["source"]
        }

    # ================== PIPELINE ==================
    def run_rag_pipeline(self, query):
        print(f"\n{'='*60}\nQuery: {query}")

        route = self.route_query(query)

        if route == "chat":
            return {
                "answer": "أهلاً يا فندم 😊، تحت أمرك في أي استفسار عن خدمات NileTel.",
                "needs_action": "NO",
                "sources": [],
                "displayed_source": "General"
            }

        if route == "out_of_scope":
            return {
                "answer": "آسف يا فندم، مش هقدر أساعدك في الموضوع ده. أنا متخصص في دعم عملاء NileTel.",
                "needs_action": "NO",
                "sources": [],
                "displayed_source": "Unknown"
            }

        if route == "ticket":
            return {
                "answer": "تمام يا فندم، هبدأ في إنشاء التذكرة حالاً. مهندس الدعم الفني هيتواصل مع حضرتك قريباً.",
                "needs_action": "YES",
                "sources": [],
                "displayed_source": "Ticket System"
            }

        results = self.retrieve(query)
        return self.generate_answer(query, results)


