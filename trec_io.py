# Optional tiny helpers if you want to convert between formats.
import json

def candidates_from_trec_and_corpus(trec_path: str, corpus_jsonl: str, out_jsonl: str, max_per_q: int = 100):
    """
    Convert (TREC run + corpus.jsonl) -> flat MVP input JSONL
    corpus.jsonl lines: {"id": "docid", "contents": "text"}
    TREC: qid Q0 docid rank score tag
    """
    # load corpus
    id2text = {}
    with open(corpus_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            id2text[str(j.get('id'))] = j.get('contents', '')

    # group trec
    qid2docs = {}
    with open(trec_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _Q0, docid, rank, score, tag = parts[:6]
            qid2docs.setdefault(qid, []).append((int(rank), docid))

    # write MVP input
    with open(out_jsonl, 'w', encoding='utf-8') as out:
        for qid, arr in qid2docs.items():
            arr.sort(key=lambda x: x[0])
            arr = arr[:max_per_q]
            cands = []
            for _, docid in arr:
                cands.append({'docid': docid, 'text': id2text.get(docid, '')})
            out.write(json.dumps({'qid': qid, 'query': '', 'candidates': cands}, ensure_ascii=False) + '\n')
