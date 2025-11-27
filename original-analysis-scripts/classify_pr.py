#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import backoff
import openai
import pandas as pd
import tiktoken

# -----------------------------------------------------------------------------
# 1. Setup & Configuration
# -----------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Attempt to import helper, create dir if missing
try:
    from helper import CSV_DIR

    CSV_DIR.mkdir(exist_ok=True)
except ImportError:
    # Fallback if helper.py is missing in a different env
    CSV_DIR = Path("results")
    CSV_DIR.mkdir(exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    # Warning only, allows script to run if only doing regex checks
    print("WARNING: OPENAI_API_KEY not set. LLM classification will fail.")

MODEL = "gpt-4.1-mini"  # Ensure this model supports response_format in your tier
ENC = tiktoken.encoding_for_model("gpt-4")

# -----------------------------------------------------------------------------
# 2. Definitions (Regex & LLM Schema)
# -----------------------------------------------------------------------------
TYPES = {
    "feat": "A new feature",
    "fix": "A bug fix",
    "docs": "Documentation only changes",
    "style": "Changes that do not affect the meaning of the code (white-space, formatting, etc)",
    "refactor": "A code change that neither fixes a bug nor adds a feature",
    "perf": "A code change that improves performance",
    "test": "Adding missing tests or correcting existing tests",
    "build": "Changes that affect the build system or external dependencies",
    "ci": "Changes to our CI configuration files and scripts",
    "chore": "Changes to the build process or auxiliary tools",
    "other": "Any other changes that do not fit the above categories",
    "revert": "Reverts a previous commit",
}

# Compile Regex Patterns based on keys
PATTERNS = {
    t: re.compile(rf"^{t}(\([^)]*\))?!?(?=\W|$)", flags=re.IGNORECASE)
    for t in TYPES.keys()
}

JSON_SCHEMA = {
    "name": "classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "A brief explanation for why this commit type was chosen"
            },
            "output": {
                "type": "string",
                "enum": list(TYPES.keys()),
                "description": "One of the allowed Conventional Commit types"
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Confidence score (1-10)"
            }
        },
        "required": ["reason", "output", "confidence"],
        "additionalProperties": False
    }
}


# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------
def load_prs(source: str) -> pd.DataFrame:
    if source.startswith("hf://"):
        return pd.read_parquet(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def title_label(title: str) -> str | None:
    """Stage 1: Regex match."""
    if not title or not isinstance(title, str):
        return None
    first = title.splitlines()[0]
    for typ, pattern in PATTERNS.items():
        if pattern.match(first):
            return typ.lower()
    return None


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    toks = ENC.encode(text)
    return text if len(toks) <= max_tokens else ENC.decode(toks[:max_tokens])


# -----------------------------------------------------------------------------
# 4. LLM Classification Logic
# -----------------------------------------------------------------------------
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError), max_tries=5)
def classify_with_gpt(title: str, body: str) -> tuple[str, str, int]:
    """
    Stage 2: LLM Classification.
    Returns (reason, output, confidence).
    """
    types_str = "\n".join([f"{k}: {v}" for k, v in TYPES.items()])

    system_msg = {
        "role": "system",
        "content": (
            "You are a Conventional Commit classifier. "
            "Given a PR title and body, pick **exactly one** label from:\n"
            f"{types_str}\n"
            "Respond in JSON with schema: {reason, output, confidence (1-10)}."
        )
    }

    # Truncate body to prevent context window overflow (conservatively 10k tokens)
    safe_body = truncate_to_tokens(body or "", 10000)

    user_msg = {
        "role": "user",
        "content": f"Title:\n{title}\n\nBody:\n{safe_body}"
    }

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[system_msg, user_msg],
        temperature=0.0,
        max_tokens=4096,
        response_format={
            "type": "json_schema",
            "json_schema": JSON_SCHEMA
        },
    )

    data = json.loads(resp.choices[0].message.content)
    return data["reason"], data["output"], int(data["confidence"])


# -----------------------------------------------------------------------------
# 5. Main Processing Flow
# -----------------------------------------------------------------------------
def classify_agent_prs(df: pd.DataFrame, agent: str) -> None:
    out_fp = CSV_DIR / f"{agent}_pr_task_type.csv"

    if out_fp.exists():
        result_df = pd.read_csv(out_fp)
    else:
        result_df = pd.DataFrame(columns=["agent", "id", "title", "reason", "type", "confidence"])

    # Filter out already processed IDs
    done_ids = set(result_df["id"].astype(str))
    df = df[~df["id"].astype(str).isin(done_ids)].copy()

    if df.empty:
        print(f"All PRs already processed for {agent}")
        return

    # -------- Stage 1: Title-based Labeling --------
    df["type"] = df["title"].apply(title_label)
    df["reason"] = df["type"].apply(
        lambda x: "title provides conventional commit label" if pd.notnull(x) else None
    )
    df["confidence"] = df["type"].apply(lambda x: 10 if pd.notnull(x) else None)

    title_labeled = df[df["type"].notnull()]
    llm_needed = df[df["type"].isnull()]

    print(f"[{agent}] {len(title_labeled)} labeled via title, {len(llm_needed)} sent to LLM")

    # Save Stage 1 results immediately
    rows = title_labeled.assign(agent=agent).to_dict(orient="records")
    if rows:
        result_df = pd.concat([result_df, pd.DataFrame(rows)], ignore_index=True)
        result_df.to_csv(out_fp, index=False)

    if llm_needed.empty:
        return

    # -------- Stage 2: LLM Labeling with ThreadPool --------
    def process_row(idx, row):
        try:
            title = row.get("title", "") or ""
            body = row.get("body", "") or ""
            pid = str(row["id"])

            reason, label, confidence = classify_with_gpt(title, body)
            print(f"[{pid}] → {label}: {reason[:50]}... (conf {confidence})")

            return {
                "agent": agent,
                "id": pid,
                "title": title,
                "reason": reason,
                "type": label,
                "confidence": confidence,
            }
        except Exception as e:
            print(f"Error processing {row.get('id')}: {e}")
            return None

    buffer = []
    # Adjust max_workers based on your rate limits
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {
            executor.submit(process_row, idx, row): idx
            for idx, row in llm_needed.iterrows()
        }

        for i, future in enumerate(as_completed(future_to_idx), 1):
            result = future.result()
            if result:
                buffer.append(result)

            # Checkpoint every 50
            if len(buffer) >= 50:
                result_df = pd.concat([result_df, pd.DataFrame(buffer)], ignore_index=True)
                result_df.to_csv(out_fp, index=False)
                print(f"Saved 50 LLM-labeled PRs to {out_fp}")
                buffer = []

    # Final Save
    if buffer:
        result_df = pd.concat([result_df, pd.DataFrame(buffer)], ignore_index=True)
        result_df.to_csv(out_fp, index=False)
        print(f"Saved remaining {len(buffer)} LLM-labeled PRs to {out_fp}")


def main(source: str, agent: str | None = None) -> None:
    print(f"Loading data from {source}...")
    df = load_prs(source)
    agents: Iterable[str] = [agent] if agent else sorted(df["agent"].dropna().unique())

    for a in agents:
        print(f"Processing agent: {a}")
        sub = df[df["agent"] == a]
        if sub.empty:
            print(f"No PRs found for {a}")
            continue
        classify_agent_prs(sub, a)


if __name__ == "__main__":
    agent_arg = sys.argv[1] if len(sys.argv) > 1 else None
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else "hf://datasets/hao-li/AIDev/pull_request.parquet"
    main(dataset_path, agent_arg)