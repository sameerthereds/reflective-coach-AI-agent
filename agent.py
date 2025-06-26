# Full Agentic Reflective Coach Pipeline with FAISS Vector Memory + Mistral-compatible Prompt + LLM Inference + Streamlit UI

import os
import json
import pandas as pd
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# ----------------------- CONFIGURATION ----------------------- #
EMBED_MODEL = 'all-MiniLM-L6-v2'
MEMORY_DIR = 'user_memory_store'
LLM_MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

# ------------------ MODELS ------------------ #
os.makedirs(MEMORY_DIR, exist_ok=True)

# ------------------ MODELS ------------------ #
embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto", torch_dtype="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=400)

# ------------------ SESSION STATE ------------------ #
if "entries" not in st.session_state:
    st.session_state.entries = []
if "time" not in st.session_state:
    st.session_state.time = datetime.now().time()
if "rating" not in st.session_state:
    st.session_state.rating = 3
if "stressor" not in st.session_state:
    st.session_state.stressor = ""
if "location" not in st.session_state:
    st.session_state.location = ""

# ------------------ UI ------------------ #
st.title("🧘 Reflective Coach Agent")
st.markdown("Log each stressor entry for today, then generate a personalized reflection.")

user_id = st.text_input("User ID", "U01")
today_date = st.date_input("Date", datetime.today()).strftime("%Y-%m-%d")

# Single-entry inputs
col1, col2 = st.columns(2)
with col1:
    st.session_state.time = st.time_input("Time", st.session_state.time)
with col2:
    st.session_state.rating = st.slider("Stress Rating (1-5)", 1, 5, st.session_state.rating)

st.session_state.stressor = st.text_input("Stressor", st.session_state.stressor)
st.session_state.location = st.text_input("Location", st.session_state.location)

if st.button("➕ Add Stressor Entry"):
    st.session_state.entries.append({
        "user_id": user_id,
        "date": today_date,
        "time": st.session_state.time.strftime("%H:%M"),
        "rating": st.session_state.rating,
        "stressor": st.session_state.stressor,
        "location": st.session_state.location
    })
    # Reset input fields
    st.session_state.time = datetime.now().time()
    st.session_state.rating = 3
    st.session_state.stressor = ""
    st.session_state.location = ""

# Show current entry list
if st.session_state.entries:
    st.markdown("### 📋 Entries So Far")
    st.dataframe(pd.DataFrame(st.session_state.entries))

# ------------------ GENERATE REFLECTION ------------------ #
if st.button("✨ Generate Reflection") and st.session_state.entries:
    entries_df = pd.DataFrame(st.session_state.entries)
    entries_df["datetime"] = pd.to_datetime(entries_df["date"] + " " + entries_df["time"])
    entries_df.sort_values("datetime", inplace=True)

    def summarize_day_entries(df_day):
        date = df_day['date'].iloc[0]
        stressful_entries = df_day[df_day['rating'] >= 3]
        if stressful_entries.empty:
            summary = "Low stress day with no major stressors reported."
            theme = "Calm"
        else:
            stressors = stressful_entries['stressor'].dropna().unique()
            locations = stressful_entries['location'].dropna().unique()
            top_stressors = ', '.join(stressors)
            top_locations = ', '.join(locations)
            summary = f"Stressful day involving {top_stressors} at {top_locations}."
            theme_prompt = (
                f"You are an assistant that classifies stressor summaries into high-level themes.\n"
                f"Summary: {summary}\n"
                f"Output the most appropriate high-level theme."
            )
            theme_response = generator(theme_prompt)[0]['generated_text']
            theme = theme_response.split("\n")[0].strip()
        return {"date": date, "summary": summary, "theme": theme}

    today_memory_entry = summarize_day_entries(entries_df)

    # -------- Load or create per-user memory -------- #
    memory_file = os.path.join(MEMORY_DIR, f"{user_id}_memory.json")
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            memory_metadata = json.load(f)
    else:
        memory_metadata = []

    if today_memory_entry['date'] not in [m['date'] for m in memory_metadata]:
        memory_metadata.append(today_memory_entry)
        with open(memory_file, 'w') as f:
            json.dump(memory_metadata, f, indent=2)

    summaries = [m['summary'] for m in memory_metadata]
    embeddings = embedder.encode(summaries, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    def retrieve_similar_memories(query_summary, k=2):
        query_embedding = embedder.encode([query_summary])
        distances, indices = index.search(query_embedding, k)
        return [memory_metadata[i] for i in indices[0]]

    similar_memories = retrieve_similar_memories(today_memory_entry['summary'], k=2)

    def generate_prompt_with_memory(df_day, memory):
        date = df_day['date'].iloc[0]
        prompt = f"You are a compassionate reflective coach AI.\nUse the day's stressor data and similar past reflections to help the user reflect and prepare.\n\n"
        prompt += f"Today's Data ({date}):\n"
        for _, row in df_day.iterrows():
            line = f"- {row['time']}: Rating = {row['rating']}"
            if row['stressor']: line += f", Stressor = \"{row['stressor']}\""
            if row['location']: line += f", Location = {row['location']}"
            prompt += line + "\n"
        if memory:
            prompt += "\nSimilar Past Reflections:\n"
            for m in memory:
                prompt += f"- {m['date']}: {m['summary']} (Theme: {m['theme']})\n"
        prompt += (
            "\nAs a compassionate AI coach, respond with a personalized reflection based on the stressors and patterns above.\n"
            "Only include the reflection. Do not repeat instructions. Your response should be as a list.\n"
            "Your response should cover:\n"
            "- A summary of emotional impact today\n"
          
            "- Likely stressors that may occur again\n"
            "- A preparation or coping plan they can try\n"
        )
        return prompt

    final_prompt = generate_prompt_with_memory(entries_df, similar_memories)
    st.markdown("### ✨ Generated Reflection")
    with st.spinner("Generating personalized reflection using Mistral..."):
        raw_output = generator(final_prompt)[0]['generated_text']
        output = raw_output.split(final_prompt)[-1].strip()
        st.write(output)

    st.markdown("---")
    with st.expander("Prompt Used"):
        st.text_area("LLM Prompt", final_prompt, height=300)

    st.session_state.entries = []
