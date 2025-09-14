# app.py
import os
import streamlit as st
from dotenv import load_dotenv
import requests
from io import StringIO
from textwrap import shorten

# Use the same groq client approach as the script version
try:
    from groq import Groq
except Exception as e:
    st.error("Missing dependency 'groq'. See requirements.txt. Run `pip install -r requirements.txt`.")
    raise

# ---------- Config ----------
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # change if your Groq key uses a different model
CHUNK_WORDS = 2500
SUMMARY_SENTENCE_TARGET = "3-4 sentences"

# ---------- Helpers ----------
def load_api_key_from_env_or_input(key_input: str):
    # priority: explicit input > env var > .env
    if key_input:
        return key_input.strip()
    load_dotenv()
    return os.environ.get("GROQ_API_KEY", "").strip()

def get_client(api_key: str):
    return Groq(api_key=api_key)

def simple_fetch_url(url: str):
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = r.text
        # best-effort: strip html tags if obvious; keep simple fallback to whole page
        # quick and dirty: if there are many tags, try to extract visible text.
        if "<html" in text.lower():
            # remove scripts/styles
            import re
            text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", text)
            # remove tags
            text = re.sub(r"(?s)<.*?>", " ", text)
            # collapse whitespace
            text = " ".join(text.split())
        return text
    except Exception as e:
        return None

def stats(text: str):
    words = len(text.split())
    chars = len(text)
    return words, chars

def chunk_text_by_words(text: str, max_words=CHUNK_WORDS):
    words = text.split()
    if len(words) <= max_words:
        yield text
        return
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def call_summary(client, article_text, temperature, model=DEFAULT_MODEL):
    prompt = (
        f"Summarize the article below in {SUMMARY_SENTENCE_TARGET}. "
        "Be factual, concise, and avoid adding information not in the article.\n\n"
        f"Article:\n{article_text}"
    )
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a concise news summarizer."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def combine_chunk_summaries(client, summaries, temperature, model=DEFAULT_MODEL):
    joined = "\n\n".join(summaries)
    prompt = (
        f"Combine the following chunk summaries into one {SUMMARY_SENTENCE_TARGET} summary. "
        "Be factual and concise.\n\nChunk summaries:\n" + joined
    )
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def ask_qa(client, article_text, question, model=DEFAULT_MODEL, temperature=0.1):
    prompt = (
        f"Based on the article below, answer the question concisely. Use only information present in the article.\n\n"
        f"Question: {question}\n\nArticle:\n{article_text}"
    )
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise news analyst. Answer concisely and factually."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def make_observations_md(article_stats, summaries, qas):
    lines = []
    lines.append("# Observations on Temperature Tuning\n")
    lines.append(f"Article stats: words = {article_stats[0]}, characters = {article_stats[1]}\n")
    for t, s in summaries.items():
        lines.append(f"## Summary @ temperature {t}\n\n")
        lines.append(s + "\n\n---\n")
    lines.append("## Quick analysis (automatic notes)\n")
    lines.append(
        "- **0.1**: most factual and concise.\n"
        "- **0.7**: balanced, more natural.\n"
        "- **1.0**: more creative; watch for rephrasing/hallucinations.\n\n"
    )
    lines.append("## Q&A\n")
    for q, a in qas:
        lines.append(f"**Q:** {q}\n\n**A:** {a}\n\n")
    return "\n".join(lines)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI News Summarizer + Q&A", layout="wide")
st.title("AI News Summarizer & Q&A — Streamlit")

# Sidebar: API key + model
st.sidebar.header("Config")
api_key_input = st.sidebar.text_input("Groq API Key (or leave blank to use GROQ_API_KEY env var)", type="password")
model_input = st.sidebar.text_input("Model (change if unavailable)", value=DEFAULT_MODEL)
api_key = load_api_key_from_env_or_input(api_key_input)
if not api_key:
    st.sidebar.warning("No API key provided. Provide a Groq key to use the app.")
    st.stop()

try:
    client = get_client(api_key)
except Exception as e:
    st.sidebar.error(f"Failed to create Groq client: {e}")
    st.stop()

# Content input
st.subheader("1) Provide article text")
col1, col2 = st.columns([2,1])
with col1:
    input_mode = st.radio("Input method", ["Paste text", "Upload .txt", "Fetch URL"], index=0)
    if input_mode == "Paste text":
        article_text = st.text_area("Paste full article text here", height=280)
    elif input_mode == "Upload .txt":
        uploaded = st.file_uploader("Upload a text file (.txt)", type=["txt"])
        article_text = ""
        if uploaded:
            try:
                article_text = uploaded.getvalue().decode("utf-8")
            except:
                article_text = uploaded.getvalue().decode("latin-1")
    else:
        url = st.text_input("Article URL (http(s)://... )")
        article_text = ""
        if url:
            st.info("Attempting to fetch the URL (simple fetch). This may not extract article body cleanly.")
            fetched = simple_fetch_url(url)
            if fetched:
                # Show first 2k chars preview
                st.success("Fetched content — preview below. Trim or edit before summarizing.")
                st.text_area("Fetched preview (edit as needed)", value=shorten(fetched, width=4000, placeholder="..."), height=240)
                article_text = fetched
            else:
                st.error("Failed to fetch URL. Try pasting text or uploading a file.")

with col2:
    st.markdown("**Quick tips**:\n- Paste the article body only (no navigation menus). \n- If the article is long, chunking will be used automatically.\n- Free tier keys may have rate limits; don't hammer the API.")

if not article_text:
    st.info("Provide article text to enable summarization.")
    st.stop()

words, chars = stats(article_text)
st.markdown(f"**Article length:** {words} words, {chars} characters")

# Summarization
st.subheader("2) Generate summaries (temps: 0.1, 0.7, 1.0)")
if st.button("Generate Summaries"):
    temps = [0.1, 0.7, 1.0]
    summaries = {}
    placeholder = st.empty()
    for t in temps:
        placeholder.info(f"Generating summary at temperature {t} ...")
        if words > CHUNK_WORDS:
            chunk_summaries = []
            for idx, chunk in enumerate(chunk_text_by_words(article_text, CHUNK_WORDS), start=1):
                placeholder.write(f"Summarizing chunk {idx} (~{len(chunk.split())} words)")
                s = call_summary(client, chunk, temperature=t, model=model_input)
                chunk_summaries.append(s)
            combined = combine_chunk_summaries(client, chunk_summaries, temperature=t, model=model_input)
            summaries[t] = combined
        else:
            s = call_summary(client, article_text, temperature=t, model=model_input)
            summaries[t] = s
        st.success(f"Summary @ {t}")
        st.text_area(f"Summary (temp={t})", value=summaries[t], height=140)
    st.session_state["summaries"] = summaries
    st.session_state["article_text"] = article_text
    st.success("Done generating summaries. You can now ask questions below.")

# Q&A
st.subheader("3) Interactive Q&A")
if "article_text" not in st.session_state:
    st.info("Generate summaries first (or paste article and press 'Generate Summaries').")
else:
    q_col, a_col = st.columns([3,2])
    with q_col:
        question = st.text_input("Ask a question about the article", key="question_input")
        if st.button("Ask"):
            if not question.strip():
                st.warning("Type a question first.")
            else:
                # Use low temperature for answers
                try:
                    answer = ask_qa(client, st.session_state["article_text"], question, model=model_input, temperature=0.1)
                except Exception as e:
                    answer = f"[ERROR] {e}"
                # store history
                history = st.session_state.get("qa_history", [])
                history.append((question, answer))
                st.session_state["qa_history"] = history
                st.rerun()
    with a_col:
        st.markdown("**Last answers**")
        for q,a in reversed(st.session_state.get("qa_history", [])[-5:]):
            st.markdown(f"**Q:** {q}\n\n**A:** {a}\n---")

# Export observations
st.subheader("4) Export / Download")
if "summaries" in st.session_state:
    summaries = st.session_state["summaries"]
    qas = st.session_state.get("qa_history", [])
    observations_md = make_observations_md((words, chars), summaries, qas)
    st.download_button("Download observations.md", data=observations_md, file_name="observations.md", mime="text/markdown")
    for t, s in summaries.items():
        fname = f"summary_temp_{str(t).replace('.', '_')}.txt"
        st.download_button(f"Download summary (temp={t})", data=s, file_name=fname, mime="text/plain")
else:
    st.info("No generated summaries to export yet.")

# Footer: warnings
st.write("---")
st.write("Notes: this app uses your Groq API key. Be mindful of rate limits and free tier quotas. If you receive auth or model errors, change the model ID in the sidebar to one available for your key.")
# ---------- End of app.py ----------