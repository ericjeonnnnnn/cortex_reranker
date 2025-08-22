import argparse
import math
from collections import defaultdict

def load_qrels(path: str):
    """Load qrels.tsv with columns: qid\tdocid\trel"""
    qrels = defaultdict(dict)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) == 4:
                # some qrels have qid \t 0 \t docid \t rel
                qid, _zero, docid, rel = parts
            else:
                continue
            try:
                qrels[qid][docid] = int(rel)
            except ValueError:
                pass
    return qrels

def load_trec_run(path: str):
    """Load TREC run: qid Q0 docid rank score tag -> dict[qid]=[docid,...] in rank order"""
    run = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _Q0, docid, rank, score, tag = parts[:6]
            run[qid].append((int(rank), docid))
    # sort by rank
    out = {}
    for qid, arr in run.items():
        arr.sort(key=lambda x: x[0])
        out[qid] = [d for _, d in arr]
    return out

def dcg(labels):
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(labels))

def ndcg_at_k(ranked_docs, rels_map, k=10):
    gains = [rels_map.get(d, 0) for d in ranked_docs[:k]]
    ideal = sorted(rels_map.values(), reverse=True)[:k]
    idcg = dcg(ideal) or 1.0
    return dcg(gains) / idcg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='TREC run file (output of rerank_cortex.py)')
    ap.add_argument('--qrels', required=True, help='qrels.tsv (qid\tdocid\trel)')
    ap.add_argument('--k', type=int, default=10, help='cutoff for nDCG@k')
    ap.add_argument("--dataset", default=None, help="ex. data/bright/economics")
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_trec_run(args.run)

    scores = []
    for qid, ranked in run.items():
        rels_map = qrels.get(qid, {})
        if not rels_map:
            continue
        scores.append(ndcg_at_k(ranked, rels_map, k=args.k))

    if not scores:
        print('No overlapping qids between run and qrels.')
        return

    macro = sum(scores) / len(scores)
    print(f'nDCG@{args.k}: {macro:.4f}  (over {len(scores)} queries)')

if __name__ == '__main__':
    main()
