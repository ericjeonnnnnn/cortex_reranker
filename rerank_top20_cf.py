import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.cortex import complete, CompleteOptions
from tqdm import tqdm

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANS_OPEN = "<answer>"
ANS_CLOSE = "</answer>"

PROMPT_SYSTEM_INITIAL = (
    "You are RankLLM, an assistant for *listwise* passage reranking.\n"
    "Produce DETAILED reasoning, then a ranking of ALL passages.\n"
    f"Return ONLY two blocks:\n"
    f"1) {THINK_OPEN} your private reasoning here {THINK_CLOSE}\n"
    f"2) {ANS_OPEN} a ranking like [3] > [1] > [2] ... {ANS_CLOSE}\n"
    "- The bracketed numbers refer to the enumerated passages the user provides.\n"
    "- Include all passages exactly once, from most relevant to least relevant.\n"
    "- Do NOT include any extra text outside the tags."
)

PROMPT_USER_TEMPLATE_INITIAL = (
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

PROMPT_SYSTEM_DEVIL = (
    "You are DevilAgent, a *critical* adversarial reviewer for listwise reranking.\n"
    "You will be given the initial model's *reasoning* and *ranking*. Identify flaws, biases, and overlooked evidence.\n"
    "Propose a counterfactual perspective and produce your OWN revised ranking.\n"
    f"Output ONLY:\n"
    f"{THINK_OPEN} your critique and counterfactual rationale {THINK_CLOSE}\n"
    f"{ANS_OPEN} [best] > [next] > ... {ANS_CLOSE}\n"
    "- Use the SAME passage numbering as provided.\n"
    "- Include all passages exactly once."
)

PROMPT_USER_TEMPLATE_DEVIL = (
    "Task: Critically attack the INITIAL ranking below and propose a better one.\n\n"
    "Query: {query}\n\n"
    "Passages (numbering is authoritative):\n"
    "{passages_block}\n\n"
    "INITIAL reasoning:\n{initial_think}\n\n"
    "INITIAL ranking:\n{initial_ans}\n\n"
    "Produce ONLY the two tagged blocks."
)

PROMPT_SYSTEM_ANGEL = (
    "You are AngelAgent, a *defender* of the INITIAL ranking under the premise that the INITIAL ranking is correct.\n"
    "Read the DevilAgent's critique and DEFEND the INITIAL ranking with evidence from the passages.\n"
    "If DevilAgent exposed a genuine mistake, minimally correct it, but prefer the INITIAL ranking when reasonable.\n"
    f"Output ONLY:\n"
    f"{THINK_OPEN} your defense and evidence {THINK_CLOSE}\n"
    f"{ANS_OPEN} [best] > [next] > ... {ANS_CLOSE}\n"
    "- Use the SAME passage numbering as provided.\n"
    "- Include all passages exactly once."
)

PROMPT_USER_TEMPLATE_ANGEL = (
    "Task: Defend the INITIAL ranking against the DevilAgent's critique.\n\n"
    "Premise: The INITIAL ranking is presumed correct unless there is clear evidence otherwise.\n\n"
    "Query: {query}\n\n"
    "Passages (numbering is authoritative):\n"
    "{passages_block}\n\n"
    "INITIAL reasoning:\n{initial_think}\n\n"
    "INITIAL ranking:\n{initial_ans}\n\n"
    "DEVIL reasoning:\n{devil_think}\n\n"
    "DEVIL ranking:\n{devil_ans}\n\n"
    "Produce ONLY the two tagged blocks."
)

PROMPT_SYSTEM_JUDGE = (
    "You are JudgeAgent. Read the entire debate (INITIAL → DEVIL → ANGEL) and decide the FINAL ranking.\n"
    "Judge based on evidence from the passages and quality of arguments.\n"
    f"Output ONLY:\n"
    f"{THINK_OPEN} concise adjudication {THINK_CLOSE}\n"
    f"{ANS_OPEN} [best] > [next] > ... {ANS_CLOSE}\n"
    "- Use the SAME passage numbering as provided.\n"
    "- Include all passages exactly once."
)

PROMPT_USER_TEMPLATE_JUDGE = (
    "Task: Decide the FINAL ranking after reading the debate.\n\n"
    "Query: {query}\n\n"
    "Passages (numbering is authoritative):\n"
    "{passages_block}\n\n"
    "INITIAL reasoning:\n{initial_think}\n\n"
    "INITIAL ranking:\n{initial_ans}\n\n"
    "DEVIL reasoning:\n{devil_think}\n\n"
    "DEVIL ranking:\n{devil_ans}\n\n"
    "ANGEL reasoning:\n{angel_think}\n\n"
    "ANGEL ranking:\n{angel_ans}\n\n"
    "Produce ONLY the two tagged blocks."
)

@dataclass
class InputItem:
    qid: str
    query: str
    candidates: List[Dict[str, str]]  # each has {docid, text}

def _passages_block(passages: List[str]) -> str:
    return "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))

def build_messages(role_system: str, role_user: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": role_system},
        {"role": "user", "content": role_user},
    ]

def parse_answer(raw_text: str, n: int) -> List[int]:
    ans_match = re.search(re.escape(ANS_OPEN) + r"(.*?)" + re.escape(ANS_CLOSE), raw_text, flags=re.S)
    answer = ans_match.group(1) if ans_match else ""
    ids = re.findall(r"\[(\d+)\]", answer)
    order: List[int] = []
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
    dedup: List[int] = []
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

def cortex_complete(
    session: Session,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    opts: CompleteOptions = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp: str = complete(
        model=model,
        prompt=messages,
        options=opts,
        session=session,
    )
    return resp

def to_trec(qid: str, ranking_docids: List[str], tag: str) -> List[str]:
    lines = []
    N = len(ranking_docids)
    for rank, docid in enumerate(ranking_docids, start=1):
        score = float(N - rank + 1)
        lines.append(f"{qid} Q0 {docid} {rank} {score:.4f} {tag}")
    return lines

def run_debate_for_query(
    session: Session,
    model: str,
    query: str,
    passages: List[str],
    max_tokens: int,
    temps: Tuple[float, float, float, float],
    top_p: float,
):
    t_init, t_devil, t_angel, t_judge = temps

    pb = _passages_block(passages)

    user_initial = PROMPT_USER_TEMPLATE_INITIAL.format(
        num=len(passages), query=query, passages_block=pb
    )
    messages_initial = build_messages(PROMPT_SYSTEM_INITIAL, user_initial)
    raw_initial = cortex_complete(session, model, messages_initial, max_tokens, t_init, top_p)
    init_order = parse_answer(raw_initial, n=len(passages))
    init_ans_str = re.search(re.escape(ANS_OPEN) + r"(.*?)" + re.escape(ANS_CLOSE), raw_initial, flags=re.S)
    init_ans = init_ans_str.group(1).strip() if init_ans_str else ""
    init_think = extract_think(raw_initial)

    user_devil = PROMPT_USER_TEMPLATE_DEVIL.format(
        query=query, passages_block=pb, initial_think=init_think, initial_ans=init_ans
    )
    messages_devil = build_messages(PROMPT_SYSTEM_DEVIL, user_devil)
    raw_devil = cortex_complete(session, model, messages_devil, max_tokens, t_devil, top_p)
    devil_order = parse_answer(raw_devil, n=len(passages))
    devil_ans = re.search(re.escape(ANS_OPEN) + r"(.*?)" + re.escape(ANS_CLOSE), raw_devil, flags=re.S)
    devil_ans = devil_ans.group(1).strip() if devil_ans else ""
    devil_think = extract_think(raw_devil)

    user_angel = PROMPT_USER_TEMPLATE_ANGEL.format(
        query=query, passages_block=pb,
        initial_think=init_think, initial_ans=init_ans,
        devil_think=devil_think, devil_ans=devil_ans,
    )
    messages_angel = build_messages(PROMPT_SYSTEM_ANGEL, user_angel)
    raw_angel = cortex_complete(session, model, messages_angel, max_tokens, t_angel, top_p)
    angel_order = parse_answer(raw_angel, n=len(passages))
    angel_ans = re.search(re.escape(ANS_OPEN) + r"(.*?)" + re.escape(ANS_CLOSE), raw_angel, flags=re.S)
    angel_ans = angel_ans.group(1).strip() if angel_ans else ""
    angel_think = extract_think(raw_angel)

    user_judge = PROMPT_USER_TEMPLATE_JUDGE.format(
        query=query, passages_block=pb,
        initial_think=init_think, initial_ans=init_ans,
        devil_think=devil_think, devil_ans=devil_ans,
        angel_think=angel_think, angel_ans=angel_ans,
    )
    messages_judge = build_messages(PROMPT_SYSTEM_JUDGE, user_judge)
    raw_judge = cortex_complete(session, model, messages_judge, max_tokens, t_judge, top_p)
    judge_order = parse_answer(raw_judge, n=len(passages))
    judge_ans = re.search(re.escape(ANS_OPEN) + r"(.*?)" + re.escape(ANS_CLOSE), raw_judge, flags=re.S)
    judge_ans = judge_ans.group(1).strip() if judge_ans else ""
    judge_think = extract_think(raw_judge)

    return {
        "initial": {"raw": raw_initial, "order": init_order, "think": init_think, "ans": init_ans},
        "devil": {"raw": raw_devil, "order": devil_order, "think": devil_think, "ans": devil_ans},
        "angel": {"raw": raw_angel, "order": angel_order, "think": angel_think, "ans": angel_ans},
        "judge": {"raw": raw_judge, "order": judge_order, "think": judge_think, "ans": judge_ans},
    }

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL with qid, query, candidates[{docid,text}]")
    parser.add_argument("--output_dir", default="runs", help="Directory for outputs")
    parser.add_argument("--trec_tag", default="cortex-rerank-cf", help="Run tag in TREC file")
    parser.add_argument("--model", default=os.environ.get("CORTEX_MODEL", "mistral-large2"))
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temp_initial", type=float, default=0.0)
    parser.add_argument("--temp_devil", type=float, default=0.3)
    parser.add_argument("--temp_angel", type=float, default=0.0)
    parser.add_argument("--temp_judge", type=float, default=0.0)
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
        items = [json.loads(x) for x in f if x.strip()]

    for item in tqdm(items, desc="CF Debate Reranking"):
        qid: str = item["qid"]
        query: str = item["query"]
        cands: List[Dict[str, str]] = item["candidates"]
        passages = [c["text"] for c in cands]
        docids = [c["docid"] for c in cands]

        try:
            debate = run_debate_for_query(
                session=session,
                model=args.model,
                query=query,
                passages=passages,
                max_tokens=args.max_tokens,
                temps=(args.temp_initial, args.temp_devil, args.temp_angel, args.temp_judge),
                top_p=args.top_p,
            )
        except AttributeError as e:
            print(f"[SKIP] qid={qid} - cortex_complete AttributeError: {e}")
            continue
        except Exception as e:
            print(f"[SKIP] qid={qid} - debate failed: {type(e).__name__}: {e}")
            continue

        final_order_1based = debate["judge"]["order"]
        ordered_docids = [docids[i-1] for i in final_order_1based]

        trec_lines_all.extend(to_trec(qid, ordered_docids, args.trec_tag))
        jsonl_out.append({
            "qid": qid,
            "ranking_final": ordered_docids,
            "steps": {
                "initial": {
                    "order": debate["initial"]["order"],
                    "think": debate["initial"]["think"],
                    "ans": debate["initial"]["ans"],
                    "raw": debate["initial"]["raw"],
                },
                "devil": {
                    "order": debate["devil"]["order"],
                    "think": debate["devil"]["think"],
                    "ans": debate["devil"]["ans"],
                    "raw": debate["devil"]["raw"],
                },
                "angel": {
                    "order": debate["angel"]["order"],
                    "think": debate["angel"]["think"],
                    "ans": debate["angel"]["ans"],
                    "raw": debate["angel"]["raw"],
                },
                "judge": {
                    "order": debate["judge"]["order"],
                    "think": debate["judge"]["think"],
                    "ans": debate["judge"]["ans"],
                    "raw": debate["judge"]["raw"],
                },
            },
        })

    trec_path = os.path.join(args.output_dir, "run_cf.trec")
    with open(trec_path, "w", encoding="utf-8") as f:
        for line in trec_lines_all:
            f.write(line + "\n")

    jsonl_path = os.path.join(args.output_dir, "run_cf.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in jsonl_out:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote: {trec_path}")
    print(f"Wrote: {jsonl_path}")

if __name__ == "__main__":
    main()