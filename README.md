# 🪞 MindMirror: An LLM-Powered Reflective Coach for Stressor Logging and Coping Support

MindMirror is an intelligent agentic system that helps users reflect on daily stressors and proactively prepare for future ones using Large Language Models (LLMs) and semantic memory retrieval.

The app enables users to log stressor entries throughout the day, retrieve similar past experiences via vector search, and generate personalized, actionable reflections powered by Mistral-7B.

---

## 📸 UI Preview

### ✏️ Stressor Logging Interface
![Main UI](/assets/image_1.png)

### 🤖 Personalized LLM Reflection Output
![Reflection UI](/assets/image_2.png)

---

## 🚀 Features

- 🧠 **LLM-Powered Coaching**: Uses `Mistral-7B-Instruct` to produce emotionally intelligent, reflective responses.
- 🔁 **Vector Memory Retrieval**: Stores user-specific summaries and retrieves top-k past experiences using `FAISS` and `SentenceTransformer`.
- 🧾 **Dynamic Stress Logging**: Users log stressors via a single-entry UI that accumulates entries in real-time with Streamlit session state.
- 👤 **Per-User Personalization**: Stores and retrieves memory on a per-user basis for personalized history and coping strategy generation.
- 💬 **Coping Plan Recommendation**: Reflections include guidance on how to prepare for recurring stressors.

---

## 🧠 Technical Stack

| Layer         | Technology                        |
|---------------|-----------------------------------|
| Frontend UI   | Streamlit                         |
| Language Model| Mistral-7B-Instruct via 🤗 Transformers |
| Embedding     | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Vector Memory | FAISS                             |
| Storage       | Per-user JSON memory files        |
| Language      | Python 3.9+                        |

---


