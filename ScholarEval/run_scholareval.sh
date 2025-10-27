#!/bin/bash
python ScholarEval.py \
  --research_idea "data/idea.txt" \
  --cutoff_date "2025-10-10" \
  --llm_engine_name "claude-sonnet-4" \
  --save_to "data/results/" \
  --litellm_name "claude-sonnet-4-20250514"