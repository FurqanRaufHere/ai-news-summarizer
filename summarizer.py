"""
summarizer.py
- Reads article text (default: article.txt)
- Produces 3 summaries at temperatures 0.1, 0.7, 1.0
- Interactive Q&A (3 questions)
- Writes observations.md and summary files

Requirements:
  pip install groq python-dotenv
Set your Groq API key in env var GROQ_API_KEY (or create a .env file).
"""

import os
import sys
import argparse
from textwrap import shorten
from dotenv import load_dotenv

try:
    from groq import Groq
    import groq as groq_exceptions
except Exception as e:
    print("Import error: make sure you installed the 'groq' package (pip install groq python-dotenv).")
    raise

# Config: change MODEL if unavailable on your Groq account
MODEL = "llama-3.3-70b-versatile"   # common example model; change if needed
CHUNK_WORDS = 2500            # chunk size (words) for very long articles
SUMMARY_SENTENCE_TARGET = "3-4 sentences"

def load_api_key():
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set. Put it in environment or a .env file.")
    return key

def read_article(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def stats(text):
    words = len(text.split())
    chars = len(text)
    return words, chars

def chunk_text_by_words(text, max_words=CHUNK_WORDS):
    words = text.split()
    if len(words) <= max_words:
        yield text
        return
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def get_client():
    api_key = load_api_key()
    client = Groq(api_key=api_key)
    return client

def call_summary(client, article_text, temperature, model=MODEL):
    prompt = (
        f"Summarize the article below in {SUMMARY_SENTENCE_TARGET}. "
        "Be factual, concise, and avoid adding information not in the article.\n\n"
        f"Article:\n{article_text}"
    )
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a concise news summarizer."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=temperature,
            # max_tokens can be set if you want tighter control; graceful fallback if unsupported
        )
        summary = resp.choices[0].message.content.strip()
        return summary
    except Exception as e:
        raise

def combine_chunk_summaries(client, summaries, temperature, model=MODEL):
    joined = "\n\n".join(summaries)
    prompt = (
        f"The article was long and has been summarized into the following chunk summaries. "
        f"Combine them into a single {SUMMARY_SENTENCE_TARGET} summary, factual and concise.\n\n"
        f"Chunk summaries:\n{joined}"
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

def ask_qa(client, article_text, question, model=MODEL, temperature=0.1):
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

def write_file(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    parser = argparse.ArgumentParser(description="News summarizer + Q&A using Groq API")
    parser.add_argument("--file", "-f", default="article.txt", help="Path to article text file")
    parser.add_argument("--model", "-m", default=MODEL, help="Model id to use")
    args = parser.parse_args()

    article_path = args.file
    model = args.model

    if not os.path.exists(article_path):
        print(f"File not found: {article_path}")
        sys.exit(1)

    article = read_article(article_path)
    w, c = stats(article)
    print(f"\nArticle loaded: {article_path} — Words: {w}, Characters: {c}\n")

    client = get_client()

    temps = [0.1, 0.7, 1.0]
    summaries = {}

    for t in temps:
        print(f"Generating summary @ temperature {t} ...")
        if w > CHUNK_WORDS:
            # chunk -> summarize each chunk -> combine
            chunk_summaries = []
            for i, chunk in enumerate(chunk_text_by_words(article, CHUNK_WORDS), start=1):
                print(f"  Summarizing chunk {i} (approx {len(chunk.split())} words)...")
                s = call_summary(client, chunk, temperature=t, model=model)
                chunk_summaries.append(s)
            combined = combine_chunk_summaries(client, chunk_summaries, temperature=t, model=model)
            summaries[t] = combined
        else:
            s = call_summary(client, article, temperature=t, model=model)
            summaries[t] = s

        fname = f"summary_temp_{str(t).replace('.', '_')}.txt"
        write_file(fname, summaries[t])
        print(f"  -> written {fname}\n")

    # Print summaries
    print("=== SUMMARIES ===")
    for t in temps:
        print(f"\n--- temp={t} ---\n{summaries[t]}\n")

    # Interactive Q&A (3 questions minimum). You can type 'done' to exit early.
    answers = []
    print("\nYou can now ask at least 3 questions about the article. Type 'done' to quit early.")
    for i in range(1, 6):  # allow up to 5 but stop when user types 'done' after >=3
        q = input(f"Question {i}: ").strip()
        if not q:
            print("Empty question, try again.")
            continue
        if q.lower() == "done":
            if i <= 3:
                print("You must ask at least 3 questions. Continue.")
                continue
            else:
                break
        try:
            ans = ask_qa(client, article, q, model=model, temperature=0.1)
        except Exception as e:
            print("API error while answering question:", e)
            ans = f"[ERROR] {e}"
        print(f"Answer:\n{ans}\n")
        answers.append((q, ans))
        if i >= 3 and input("Ask another? (y/N): ").strip().lower() != "y":
            break

    # Build observations.md automatically (includes the three summaries and notes)
    obs_lines = []
    obs_lines.append("# Observations on Temperature Tuning\n")
    obs_lines.append("Article stats: words = {}, characters = {}\n".format(w, c))
    for t in temps:
        obs_lines.append(f"## Summary @ temperature {t}\n")
        obs_lines.append(summaries[t] + "\n")
        obs_lines.append("---\n")
    obs_lines.append("## Quick analysis (automatic notes)\n")
    obs_lines.append(
        "- **0.1**: tends to be the most factual and concise — shorter sentences, less speculation.\n"
        "- **0.7**: balanced — more natural phrasing and good tradeoff between completeness and brevity.\n"
        "- **1.0**: creative — may rephrase more, include optional framing or colorful language; check for hallucinations.\n\n"
        "Which temperature worked best for you depends on whether you prefer strict factual brevity (0.1) or a slightly more readable summary (0.7).\n"
    )
    obs_lines.append("## Q&A (questions & answers)\n")
    for q, a in answers:
        obs_lines.append(f"**Q:** {q}\n\n**A:** {a}\n\n")

    observations_text = "\n".join(obs_lines)
    write_file("observations.md", observations_text)
    print("\nWrote observations.md and summary files. Done.")

if __name__ == "__main__":
    main()
