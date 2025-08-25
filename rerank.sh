python rerank_window.py \
  --input input/bright_economics_rerank_input.jsonl \
  --output_dir runs \
  --trec_tag cortex-rerank \
  --max_tokens 1500 --top_p 0.9