import requests
import os
import time
import json
import numpy as np
from difflib import SequenceMatcher  # for fuzzy name matching

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"


# ============================
# Helper functions
# ============================

def _similarity(a: str, b: str) -> float:
    """Return a similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _detect_list_query(text: str) -> bool:
    """Detect if the user is asking for a list of all faculty."""
    t = text.lower()
    triggers = [
        "list faculty",
        "list all faculty",
        "all faculty",
        "faculty list",
        "all professors",
        "show faculty",
        "show professors",
        "advisor list",
        "which professors are there",
        "who are the faculty",
        "who are the professors",
    ]
    return any(trig in t for trig in triggers)


def _fuzzy_match_name(text: str, names, threshold: float = 0.65):
    """
    Try to match a possibly misspelled name to the closest faculty name.
    Returns the best matching name or None.
    """
    best_name = None
    best_score = 0.0

    for name in names:
        score = _similarity(text, name)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is not None and best_score >= threshold:
        return best_name
    return None


class ResponseEngine:
    """
    Response engine using Groq API with Llama 3
    plus retrieval-augmented generation (RAG)
    over BSU CS faculty profiles.
    """

    def __init__(self):
        """Initialize Groq API connection and RAG resources."""

        # ---------- LLM / Groq setup ----------
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            print("WARNING: No GROQ_API_KEY found!")
            raise ValueError("GROQ_API_KEY required in .env file")

        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.model = MODEL_NAME

        # General persona (used for non-RAG answers)
        # Adjusted to stop asking too many clarifying questions
        self.system_prompt = (
            "You are the BSU Graduate Advisor AI Assistant for Computer Science "
            "students at Boise State University.\n\n"
            "Your role:\n"
            "- Help students find suitable research advisors based on their interests, skills, and goals\n"
            "- Provide information about faculty research areas and general availability\n"
            "- Guide students through the advisor selection process\n"
            "- Answer questions about BSU CS graduate programs\n"
            "- Be direct and concise (2 to 4 sentences)\n"
            "- Only ask a clarifying question if the student's request is genuinely ambiguous\n"
            "- When possible, make the best recommendation from available information instead of asking many follow up questions\n\n"
            "When you are provided with faculty profiles in the context, you MUST rely on that "
            "data and not invent additional details."
        )

        print(f"Groq API initialized with {self.model}")

        # ---------- RAG resources (embeddings + profiles) ----------
        self._load_rag_resources()

    # =================================================================
    # RAG INITIALIZATION
    # =================================================================

    def _load_rag_resources(self):
        """
        Load faculty embeddings and metadata for retrieval.
        Expects:
            - embeddings.npy
            - faculty_ids.json
            - faculty_texts.json
        in the current working directory.
        """
        try:
            print("[RAG] Loading faculty embeddings and metadata...")
            self.embeddings = np.load("embeddings.npy")

            with open("faculty_ids.json", "r", encoding="utf-8") as f:
                self.faculty_ids = json.load(f)

            with open("faculty_texts.json", "r", encoding="utf-8") as f:
                self.faculty_texts = json.load(f)

            if len(self.embeddings) != len(self.faculty_ids):
                print(
                    f"[RAG] WARNING: embeddings count ({len(self.embeddings)}) "
                    f"!= ids count ({len(self.faculty_ids)})"
                )

            # SentenceTransformers model for query encoding
            self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            # Ensure embeddings are L2-normalized (just in case)
            self.embeddings = normalize(self.embeddings)

            print(f"[RAG] Loaded {len(self.faculty_ids)} faculty profiles for retrieval.")
        except Exception as e:
            print(f"[RAG] WARNING: could not load RAG resources: {e}")
            self.embeddings = None
            self.faculty_ids = None
            self.faculty_texts = None
            self.embed_model = None

    # =================================================================
    # BASE LLM CALL (non-RAG)
    # =================================================================

    def generate_answer(self, user_query, history=None):
        """
        Plain LLM answer using only the static system_prompt.
        (Old behavior, still available.)
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add conversation history (keep last 6 messages for context)
        if history:
            for msg in history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        messages.append({
            "role": "user",
            "content": user_query
        })

        answer = self._query_groq(messages)
        return answer

    # Convenience alias if you ever want engine.ask(...)
    def ask(self, user_query, history=None, use_rag=False):
        """
        If use_rag is True, use the RAG pipeline with all the special
        handling for listing faculty and fuzzy name matching.
        Otherwise fall back to plain LLM.
        """
        if use_rag:
            return self.generate_rag_answer(user_query, history=history)
        return self.generate_answer(user_query, history=history)

    # =================================================================
    # RAG: RETRIEVAL + GENERATION
    # =================================================================

    def retrieve_faculty(self, query, top_k=3):
        """
        Retrieve top_k most relevant faculty profiles for a given query.
        Returns a list of dicts with {name, score, profile_text}.
        """
        if self.embed_model is None or self.embeddings is None:
            print("[RAG] Retrieval requested but RAG resources are not loaded.")
            return []

        # Encode and normalize query
        q_emb = self.embed_model.encode([query])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

        # Cosine similarity because embeddings are normalized
        sims = self.embeddings @ q_emb

        top_k = min(top_k, len(sims))
        idxs = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in idxs:
            results.append({
                "name": self.faculty_ids[idx],
                "score": float(sims[idx]),
                "profile_text": self.faculty_texts[idx]
            })

        return results

    def _list_all_faculty_text(self):
        """Return a human readable list of all faculty names."""
        if not self.faculty_ids:
            return "I do not have any faculty data loaded right now."
        lines = [f"- {name}" for name in self.faculty_ids]
        return (
            "Here is the list of CS faculty I know about:\n\n"
            + "\n".join(lines)
            + "\n\nYou can ask me about any specific person, or tell me your interests and I will recommend a few advisors."
        )

    def _answer_for_specific_faculty(self, faculty_name, history=None):
        """
        Build a focused prompt for one matched faculty member.
        This is used when we fuzzy match a misspelled name.
        """
        if not self.faculty_ids or not self.faculty_texts:
            return "I could not load the faculty profiles right now."

        try:
            idx = self.faculty_ids.index(faculty_name)
        except ValueError:
            return "I could not find that faculty in my profiles."

        profile = self.faculty_texts[idx]

        single_faculty_prompt = (
            "You are the BSU Graduate Advisor AI Assistant for Computer Science students at Boise State University.\n\n"
            "You are given one faculty profile. Summarize who they are, what they work on, "
            "and what kind of student or interests they are a good match for. "
            "Be direct and concise (2 to 4 sentences) and do not ask follow up questions.\n\n"
            f"FACULTY PROFILE:\n{profile}\n"
        )

        messages = [
            {"role": "system", "content": single_faculty_prompt},
            {"role": "user", "content": f"Tell me about {faculty_name} as a potential advisor for me."}
        ]

        # Optionally include short history, but it is not critical here
        if history:
            for msg in history[-3:]:
                messages.insert(1, {  # insert after system
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return self._query_groq(messages)

    def generate_rag_answer(self, user_query, history=None, top_k=3):
        """
        RAG mode:
        1) Special handling: list-all queries and fuzzy name matches.
        2) Otherwise retrieve top_k matching faculty profiles.
        3) Inject them into a system message.
        4) Ask Llama to answer using ONLY that faculty context.
        """

        # Special case 1: list of all faculty
        if _detect_list_query(user_query) and self.faculty_ids:
            return self._list_all_faculty_text()

        # Special case 2: user typed something that looks like a faculty name
        if self.faculty_ids:
            fuzzy_name = _fuzzy_match_name(user_query, self.faculty_ids, threshold=0.7)
            if fuzzy_name is not None:
                # Handle as "tell me about this specific faculty" even if misspelled
                return self._answer_for_specific_faculty(fuzzy_name, history=history)

        # Normal RAG retrieval
        retrieved = self.retrieve_faculty(user_query, top_k=top_k)

        # If nothing retrieved, give a clear fallback instead of silence
        if not retrieved:
            return (
                "I could not match your question to any specific faculty profiles. "
                "Try telling me your research interests, for example: "
                "\"I am interested in AI and machine learning\" or "
                "\"I want to work on cybersecurity and privacy\"."
            )

        context_blocks = []
        for i, r in enumerate(retrieved, start=1):
            block = (
                f"FACULTY MATCH {i}:\n"
                f"Name: {r['name']}\n"
                f"Relevance score: {r['score']:.3f}\n"
                f"Profile:\n{r['profile_text']}\n"
            )
            context_blocks.append(block)
        faculty_context = "\n---\n".join(context_blocks)

        rag_system_prompt = (
            "You are the BSU Graduate Advisor AI Assistant for Computer Science students "
            "at Boise State University.\n\n"
            "You are connected to a factual database of BSU CS faculty profiles.\n"
            "Below you are given the top retrieved faculty profiles that are relevant "
            "to the student's question.\n\n"
            "=== FACULTY CONTEXT START ===\n"
            f"{faculty_context}\n"
            "=== FACULTY CONTEXT END ===\n\n"
            "Instructions:\n"
            "- When recommending advisors, rely ONLY on the information in the faculty context.\n"
            "- Recommend 1 to 3 specific faculty that best match the student's interests.\n"
            "- Briefly explain why each recommended faculty member is a good match.\n"
            "- Do NOT ask unnecessary clarifying questions. Make the best recommendation with the information you have.\n"
            "- If the context is insufficient, say you are not sure and suggest contacting the department.\n"
            "- Keep answers concise (2 to 4 sentences) and supportive."
        )

        messages = [
            {"role": "system", "content": rag_system_prompt}
        ]

        # Optional: include short history for conversational feel
        if history:
            for msg in history[-4:]:
                if msg.get("role") in ("user", "assistant"):
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        messages.append({
            "role": "user",
            "content": user_query
        })

        return self._query_groq(messages)

    # =================================================================
    # LOW LEVEL GROQ CALL
    # =================================================================

    def _query_groq(self, messages, max_retries=3):
        """Query Groq API with retry logic."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300,
            "top_p": 0.9,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"].strip()
                    return answer

                elif response.status_code == 401:
                    return "Authentication error. Check your GROQ_API_KEY in the .env file."

                elif response.status_code == 429:
                    print(f"Rate limit, waiting... (attempt {attempt + 1})")
                    time.sleep(2)
                    continue

                else:
                    print(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                print(f"Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

        return "I'm having trouble connecting right now. Please try again in a moment."
# import requests
# import os
# import time
# import json
# import numpy as np

# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import normalize

# load_dotenv()

# MODEL_NAME = "llama-3.3-70b-versatile"

# class ResponseEngine:
#     """
#     Response engine using Groq API with Llama 3
#     plus retrieval-augmented generation (RAG)
#     over BSU CS faculty profiles.
#     """

#     def __init__(self):
#         """Initialize Groq API connection and RAG resources."""

#         # ---------- LLM / Groq setup ----------
#         self.api_key = os.getenv("GROQ_API_KEY", "")
#         if not self.api_key:
#             print("WARNING: No GROQ_API_KEY found!")
#             raise ValueError("GROQ_API_KEY required in .env file")

#         self.api_url = "https://api.groq.com/openai/v1/chat/completions"
#         self.headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#         self.model = MODEL_NAME

#         # General persona (used for non-RAG answers)
#         self.system_prompt = (
#             "You are the BSU Graduate Advisor AI Assistant for Computer Science "
#             "students at Boise State University.\n\n"
#             "Your role:\n"
#             "- Help students find suitable research advisors based on their interests\n"
#             "- Provide information about faculty research areas and general availability\n"
#             "- Guide students through the advisor selection process\n"
#             "- Answer questions about BSU CS graduate programs\n"
#             "- Be friendly, concise (2–4 sentences), and proactive in asking clarifying questions\n\n"
#             "When you are provided with faculty profiles in the context, you MUST rely on that "
#             "data and not invent additional details."
#         )

#         print(f"Groq API initialized with {self.model}")

#         # ---------- RAG resources (embeddings + profiles) ----------
#         self._load_rag_resources()

#     # =================================================================
#     # RAG INITIALIZATION
#     # =================================================================

#     def _load_rag_resources(self):
#         """
#         Load faculty embeddings and metadata for retrieval.
#         Expects:
#             - embeddings.npy
#             - faculty_ids.json
#             - faculty_texts.json
#         in the current working directory.
#         """
#         try:
#             print("[RAG] Loading faculty embeddings and metadata...")
#             self.embeddings = np.load("embeddings.npy")

#             with open("faculty_ids.json", "r", encoding="utf-8") as f:
#                 self.faculty_ids = json.load(f)

#             with open("faculty_texts.json", "r", encoding="utf-8") as f:
#                 self.faculty_texts = json.load(f)

#             if len(self.embeddings) != len(self.faculty_ids):
#                 print(
#                     f"[RAG] WARNING: embeddings count ({len(self.embeddings)}) "
#                     f"!= ids count ({len(self.faculty_ids)})"
#                 )

#             # SentenceTransformers model for query encoding
#             self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#             # Ensure embeddings are L2-normalized (just in case)
#             self.embeddings = normalize(self.embeddings)

#             print(f"[RAG] Loaded {len(self.faculty_ids)} faculty profiles for retrieval.")
#         except Exception as e:
#             print(f"[RAG] WARNING: could not load RAG resources: {e}")
#             self.embeddings = None
#             self.faculty_ids = None
#             self.faculty_texts = None
#             self.embed_model = None

#     # =================================================================
#     # BASE LLM CALL (non-RAG)
#     # =================================================================

#     def generate_answer(self, user_query, history=None):
#         """
#         Plain LLM answer using only the static system_prompt.
#         (Old behavior, still available.)
#         """
#         messages = [
#             {"role": "system", "content": self.system_prompt}
#         ]

#         # Add conversation history (keep last 6 messages for context)
#         if history:
#             for msg in history[-6:]:
#                 messages.append({
#                     "role": msg["role"],
#                     "content": msg["content"]
#                 })

#         messages.append({
#             "role": "user",
#             "content": user_query
#         })

#         answer = self._query_groq(messages)
#         return answer

#     # Convenience alias if you ever want engine.ask(...)
#     def ask(self, user_query, history=None, use_rag=False):
#         if use_rag:
#             return self.generate_rag_answer(user_query, history=history)
#         return self.generate_answer(user_query, history=history)

#     # =================================================================
#     # RAG: RETRIEVAL + GENERATION
#     # =================================================================

#     def retrieve_faculty(self, query, top_k=3):
#         """
#         Retrieve top_k most relevant faculty profiles for a given query.
#         Returns a list of dicts with {name, score, profile_text}.
#         """
#         if self.embed_model is None or self.embeddings is None:
#             print("[RAG] Retrieval requested but RAG resources are not loaded.")
#             return []

#         # Encode and normalize query
#         q_emb = self.embed_model.encode([query])[0]
#         q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

#         # Cosine similarity because embeddings are normalized
#         sims = self.embeddings @ q_emb

#         top_k = min(top_k, len(sims))
#         idxs = np.argsort(sims)[::-1][:top_k]

#         results = []
#         for idx in idxs:
#             results.append({
#                 "name": self.faculty_ids[idx],
#                 "score": float(sims[idx]),
#                 "profile_text": self.faculty_texts[idx]
#             })

#         return results

#     def generate_rag_answer(self, user_query, history=None, top_k=3):
#         """
#         RAG mode:
#         1) Retrieve top_k matching faculty profiles.
#         2) Inject them into a system message.
#         3) Ask Llama to answer using ONLY that faculty context.
#         """
#         retrieved = self.retrieve_faculty(user_query, top_k=top_k)

#         if retrieved:
#             context_blocks = []
#             for i, r in enumerate(retrieved, start=1):
#                 block = (
#                     f"FACULTY MATCH {i}:\n"
#                     f"Name: {r['name']}\n"
#                     f"Relevance score: {r['score']:.3f}\n"
#                     f"Profile:\n{r['profile_text']}\n"
#                 )
#                 context_blocks.append(block)
#             faculty_context = "\n---\n".join(context_blocks)
#         else:
#             faculty_context = (
#                 "No matching faculty profiles were retrieved for this query. "
#                 "You should apologize and suggest that the student contact the department "
#                 "or check the official faculty directory."
#             )

#         rag_system_prompt = (
#             "You are the BSU Graduate Advisor AI Assistant for Computer Science students "
#             "at Boise State University.\n\n"
#             "You are connected to a factual database of BSU CS faculty profiles.\n"
#             "Below you are given the top retrieved faculty profiles that are relevant "
#             "to the student's question.\n\n"
#             "=== FACULTY CONTEXT START ===\n"
#             f"{faculty_context}\n"
#             "=== FACULTY CONTEXT END ===\n\n"
#             "Instructions:\n"
#             "- When recommending advisors, rely ONLY on the information in the faculty context.\n"
#             "- Recommend 1–3 specific faculty that best match the student's interests.\n"
#             "- Briefly explain why each recommended faculty member is a good match.\n"
#             "- If the context is insufficient, say you are not sure and suggest contacting the department.\n"
#             "- Keep answers concise (2–4 sentences) and supportive."
#         )

#         messages = [
#             {"role": "system", "content": rag_system_prompt}
#         ]

#         # Optional: include short history for conversational feel
#         if history:
#             for msg in history[-4:]:
#                 if msg.get("role") in ("user", "assistant"):
#                     messages.append({
#                         "role": msg["role"],
#                         "content": msg["content"]
#                     })

#         messages.append({
#             "role": "user",
#             "content": user_query
#         })

#         return self._query_groq(messages)

#     # =================================================================
#     # LOW LEVEL GROQ CALL
#     # =================================================================

#     def _query_groq(self, messages, max_retries=3):
#         """Query Groq API with retry logic."""

#         payload = {
#             "model": self.model,
#             "messages": messages,
#             "temperature": 0.7,
#             "max_tokens": 300,
#             "top_p": 0.9,
#         }

#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     self.api_url,
#                     headers=self.headers,
#                     json=payload,
#                     timeout=30,
#                 )

#                 if response.status_code == 200:
#                     result = response.json()
#                     answer = result["choices"][0]["message"]["content"].strip()
#                     return answer

#                 elif response.status_code == 401:
#                     return "Authentication error. Check your GROQ_API_KEY in the .env file."

#                 elif response.status_code == 429:
#                     print(f"Rate limit, waiting... (attempt {attempt + 1})")
#                     time.sleep(2)
#                     continue

#                 else:
#                     print(f"API Error {response.status_code}: {response.text}")

#             except Exception as e:
#                 print(f"Error: {e}")
#                 if attempt < max_retries - 1:
#                     time.sleep(1)
#                     continue

#         return "I'm having trouble connecting right now. Please try again in a moment."





# # import requests
# # import os
# # import time

# # from dotenv import load_dotenv
# # load_dotenv()

# # MODEL_NAME = "llama-3.3-70b-versatile"
# # class ResponseEngine:
# #     """
# #     Response engine using Groq API with Llama 3
# #     """
    
# #     def __init__(self):
# #         """Initialize Groq API connection."""
        
# #         # Load Groq API key from environment
# #         self.api_key = os.getenv("GROQ_API_KEY", "")
        
# #         if not self.api_key:
# #             print("WARNING: No GROQ_API_KEY found!")
# #             raise ValueError("GROQ_API_KEY required in .env file")
        
# #         # Groq API endpoint
# #         self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
# #         self.headers = {
# #             "Authorization": f"Bearer {self.api_key}",
# #             "Content-Type": "application/json"
# #         }
        
# #         # Model to use
# #         self.model = MODEL_NAME
        
# #         # define the AI's role and knowledge
# #         self.system_prompt = """You are the BSU Graduate Advisor AI Assistant for Computer Science students at Boise State University.

# # Your knowledge includes:

# # BSU CS Faculty:
# # - Dr. Jun Zhuang: Artificial Intelligence, Machine Learning, Human-Centered Computing (Available Spring 2026)
# # - Dr. Gaby Dagher: Cybersecurity, Privacy, Blockchain Technology (Available Fall 2025)
# # - Dr. Jerry Alan Fails: Human-Computer Interaction, CS Education, Child-Computer Interaction (Available Now)
# # - Dr. Elisa Barney Smith: Computer Vision, Pattern Recognition (Available Spring 2026)
# # - Dr. Hoda Mehrpouyan: AI Ethics, Sustainability, Systems Engineering (Not taking students currently)

# # Important Guidelines:
# # - Students must find a permanent advisor by the end of their 2nd semester
# # - Students should attend faculty research talks and office hours
# # - International students often face challenges in the advisor selection process
# # - Consider research interests alignment when choosing an advisor

# # Your role:
# # - Help students find suitable research advisors based on their interests
# # - Provide information about faculty research areas and availability
# # - Guide students through the advisor selection process
# # - Answer questions about BSU CS graduate programs
# # - Be friendly, conversational, and proactive in asking follow-up questions

# # Keep responses concise (2-4 sentences) but helpful. Always try to guide the conversation toward helping them find the right advisor."""
        
# #         print(f"Groq API initialized with {self.model}")
    
# #     def generate_answer(self, user_query, history=None):
# #         """
# #         Generate intelligent response using Groq API.
        
# #         Args:
# #             user_query: Current user question
# #             history: Previous conversation messages
            
# #         Returns:
# #             Generated response string
# #         """
        
# #         # Build messages array for the API
# #         messages = [
# #             {"role": "system", "content": self.system_prompt}
# #         ]
        
# #         # Add conversation history (keep last 6 messages for context)
# #         if history:
# #             for msg in history[-6:]:
# #                 messages.append({
# #                     "role": msg["role"],
# #                     "content": msg["content"]
# #                 })
        
# #         # Add current user query
# #         messages.append({
# #             "role": "user",
# #             "content": user_query
# #         })
        
# #         # Call Groq API
# #         answer = self._query_groq(messages)
# #         return answer
    
# #     def _query_groq(self, messages, max_retries=3):
# #         """Query Groq API with retry logic."""
        
# #         payload = {
# #             "model": self.model,
# #             "messages": messages,
# #             "temperature": 0.7,
# #             "max_tokens": 300,
# #             "top_p": 0.9
# #         }
        
# #         for attempt in range(max_retries):
# #             try:
# #                 response = requests.post(
# #                     self.api_url,
# #                     headers=self.headers,
# #                     json=payload,
# #                     timeout=30
# #                 )
                
# #                 if response.status_code == 200:
# #                     result = response.json()
# #                     answer = result["choices"][0]["message"]["content"].strip()
# #                     return answer
                
# #                 elif response.status_code == 401:
# #                     return "Authentication error. Check your GROQ_API_KEY in the .env file."
                
# #                 elif response.status_code == 429:
# #                     print(f"Rate limit, waiting... (attempt {attempt + 1})")
# #                     time.sleep(2)
# #                     continue
                
# #                 else:
# #                     print(f"API Error {response.status_code}: {response.text}")
                    
# #             except Exception as e:
# #                 print(f"Error: {e}")
# #                 if attempt < max_retries - 1:
# #                     time.sleep(1)
# #                     continue
        
# #         return "I'm having trouble connecting right now. Please try again in a moment."
