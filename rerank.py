import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict

from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.cortex import Complete
from tqdm import tqdm

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANS_OPEN = "<answer>"
ANS_CLOSE = "</answer>"

PROMPT_SYSTEM = (
    "You are RankLLM, an assistant for *listwise* passage reranking.\n"
    "You MUST respond in two parts:\n"
    f"1) {THINK_OPEN} your private reasoning here {THINK_CLOSE}\n"
    f"2) {ANS_OPEN} a ranking like [3] > [1] > [2] ... {ANS_CLOSE}\n"
    "- The numbers in brackets refer to the order of passages the user enumerates.\n"
    "- Include *all* passages once, from most relevant to least relevant.\n"
    "- Do NOT include any extra text outside the tags."
)

PROMPT_USER_TEMPLATE = (
    "I will provide you with {num} passages, each indicated by an identifier [i].\n"
    "Please rank ALL passages by relevance to the query.\n"
    "Return ONLY:\n"
    f"{THINK_OPEN} ... {THINK_CLOSE}\n"
    f"{ANS_OPEN} [best] > [next] > ... {ANS_CLOSE}\n\n"
    "Query: {query}\n\n"
    "Passages:\n"
    "{passages_block}\n\n"
    "Remember: only output the two tagged blocks. No extra commentary.\n"
)

@dataclass
class InputItem:
    qid: str
    query: str
    candidates: List[Dict[str, str]]  # each has {docid, text}

def build_messages(query: str, passages: List[str]) -> List[Dict[str, str]]:
    passages_block = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    user = PROMPT_USER_TEMPLATE.format(num=len(passages), query=query, passages_block=passages_block)
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": user},
    ]

def parse_answer(raw_text: str, n: int) -> List[int]:
    """
    Parse model output containing:
    <think>...</think>
    <answer>[3] > [1] > [2] ...</answer>
    Returns 1-based indices list. Falls back to [1..n] if parsing fails.
    """
    ans_match = re.search(re.escape(ANS_OPEN) + r"(.*?)" + re.escape(ANS_CLOSE), raw_text, flags=re.S)
    answer = ans_match.group(1) if ans_match else ""
    ids = re.findall(r"\[(\d+)\]", answer)
    order = []
    for s in ids:
        try:
            k = int(s)
            if 1 <= k <= n and k not in order:
                order.append(k)
        except ValueError:
            pass
    remaining = [i for i in range(1, n+1) if i not in order]
    final = order + remaining
    seen = set()
    dedup = []
    for x in final:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    if len(dedup) != n:
        dedup = list(range(1, n+1))
    return dedup

def extract_think(raw_text: str) -> str:
    m = re.search(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), raw_text, flags=re.S)
    return (m.group(1).strip() if m else "").strip()

def cortex_complete(session: Session, model: str, messages: List[Dict[str, str]],
                    max_tokens: int, temperature: float, top_p: float) -> str:
    resp = Complete(
        model=model,
        prompt=messages,
        options={"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
        session=session,
    )
    return str(resp)

def to_trec(qid: str, ranking_docids: List[str], tag: str) -> List[str]:
    lines = []
    N = len(ranking_docids)
    for rank, docid in enumerate(ranking_docids, start=1):
        score = float(N - rank + 1)
        lines.append(f"{qid} Q0 {docid} {rank} {score:.4f} {tag}")
    return lines

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL with qid, query, candidates[{docid,text}]")
    parser.add_argument("--output_dir", default="runs", help="Directory for outputs")
    parser.add_argument("--trec_tag", default="cortex-rerank", help="Run tag in TREC file")
    parser.add_argument("--model", default=os.environ.get("CORTEX_MODEL", "mistral-large2"))
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    connection_params = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_USER_PASSWORD"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
    }
    connection_params = {k: v for k, v in connection_params.items() if v}
    session = Session.builder.configs(connection_params).create()

    os.makedirs(args.output_dir, exist_ok=True)
    trec_lines_all: List[str] = []
    jsonl_out = []

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]

    for item in tqdm(lines, desc="Reranking"):
        qid = item["qid"]
        query = item["query"]
        cands = item["candidates"]
        passages = [c["text"] for c in cands]
        docids = [c["docid"] for c in cands]

        messages = build_messages(query=query, passages=passages)
        raw = cortex_complete(
            session=session,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        order_1based = parse_answer(raw, n=len(passages))
        ordered_docids = [docids[i-1] for i in order_1based]

        trec_lines_all.extend(to_trec(qid, ordered_docids, args.trec_tag))
        jsonl_out.append({
            "qid": qid,
            "ranking": ordered_docids,
            "think": extract_think(raw),
            "raw_answer": raw,
        })

    trec_path = os.path.join(args.output_dir, "run.trec")
    with open(trec_path, "w", encoding="utf-8") as f:
        for line in trec_lines_all:
            f.write(line + "\n")

    jsonl_path = os.path.join(args.output_dir, "run.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in jsonl_out:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote: {trec_path}")
    print(f"Wrote: {jsonl_path}")

if __name__ == "__main__":
    main()
