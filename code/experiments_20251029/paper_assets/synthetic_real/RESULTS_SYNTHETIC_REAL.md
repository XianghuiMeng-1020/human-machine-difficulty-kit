# Synthetic (real-model names) multi-model divergence

- Input questions: `gen_questions.jsonl` (same as synthetic line)
- Inference runs: `runs/synthetic_llm_runs.csv` (models: gpt4o → mapped to "deep", gpt4o-mini → "mini", qwen3 → "qwen")
- Divergence output: `synthetic_separators.json` (18 questions on which at least one model is right and another is wrong)

**Key point.** Even on small, LLM-generated items without human labels, different LLM families still disagree on a non-trivial subset (18/30 here, stochastic run), which supports our RQ2 claim: *“认知代沟不仅存在于人工题库（RACE/Eedi），在合成题上也能被稳定激活。”*
