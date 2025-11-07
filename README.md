# Human–Machine Difficulty Kit

Open-source code release for our forthcoming paper on Human–Machine Difficulty Alignment in educational QA.
This repo provides a complete, reproducible pipeline to: (1) prepare RACE & Eedi data, (2) score MCQs with LLMs, (3) compute Data Maps and calibration (ECE / temperature scaling), and (4) quantify alignment between human-perceived and model-perceived difficulty.

⸻

✨ TL;DR
	•	Goal: Measure how well model-perceived difficulty (confidence, variability, NLL) aligns with human difficulty (error rate, wrong-with-confidence, IRT-ready signals).
	•	Datasets: RACE (reading comprehension) and Eedi NeurIPS 2020 Task 3/4 (K12 multiple-choice with images).
	•	Models: Plug any API model (OpenAI/Qwen/DeepSeek/…); a dummy client is included to validate the pipeline.
	•	Methods:
	•	Scoring uses per-option probabilities p(A…D) with temperature=0 (no sampling).
	•	Data Map from per-item mean/std of p(\text{correct}) across rounds.
	•	Calibration (ECE, Brier, temperature scaling).
	•	Alignment via rank-correlation & contingency analysis against human difficulty.
	•	Repro first: Everything runs from a single CLI: runner.py.

⸻

🗂 Repository Layout

human_machine_difficulty_kit/
├─ README.md
├─ requirements.txt
├─ config.example.yaml
├─ runner.py
├─ sample_data/
│  ├─ race_demo.jsonl
│  └─ eedi_interactions_demo.csv
├─ src/
│  ├─ utils/
│  │  └─ io.py
│  ├─ data/
│  │  ├─ prepare_race.py
│  │  └─ prepare_eedi.py
│  ├─ scoring/
│  │  ├─ interface.py
│  │  ├─ prompts.py
│  │  ├─ dummy_client.py
│  │  └─ openai_client_stub.py
│  └─ analysis/
│     ├─ calibration.py
│     ├─ datamap.py
│     ├─ alignment.py
│     └─ tables_figures.py


⸻

🔧 Installation

git clone <your-repo-url>
cd human_machine_difficulty_kit
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

Optional (for RACE auto-download):

pip install datasets


⸻

🚀 Quick Start (Demo without any external data)

Validate the entire pipeline with a tiny synthetic set and a random scorer:

python runner.py demo --rounds 3

Outputs (check outputs/demo/):
	•	scores.csv (per-round p(correct) & correctness),
	•	datamap.csv (mean/std per item).

⸻

📦 Data Preparation

RACE (ReAding Comprehension)

Use the Hugging Face dataset (recommended):

python -m src.data.prepare_race --split test
# writes: data/race/processed/race_test.jsonl

Each record has:

{
  "id": "passageId#Qk",
  "passage": "...",
  "question": "...",
  "options": {"A":"...","B":"...","C":"...","D":"..."},
  "answer": "A|B|C|D"
}

Eedi NeurIPS 2020 (Task 3/4)

Download the official CSVs per license and place them at:

data/eedi/raw/
  interactions_task34.csv   # must contain: question_id, IsCorrect, AnswerValue, CorrectAnswer, Confidence
  questions_task34.csv      # optional: stem/options/image_path for richer prompts

Preprocess:

python -m src.data.prepare_eedi \
  --in data/eedi/raw/interactions_task34.csv \
  --out data/eedi/processed \
  [--questions data/eedi/raw/questions_task34.csv]

This produces:
	•	question_summary_task34.csv with human-side signals:
	•	error_rate = 1 - mean(IsCorrect)
	•	wrong_conf = mean((1 - IsCorrect) * Confidence/100)  ← emphasizes “wrong with confidence”
	•	diff_conf_old (legacy kept for reference)
	•	option_entropy (student choice dispersion)
	•	human_label_v2 via quantiles over wrong_conf (simpleH|mediumH|hardH)
	•	task34_for_llm.jsonl unified MCQ list with optional image_path.

Note: For Eedi, many questions rely on the image. Use a vision-capable model and pass the image during scoring to ensure human–model comparability.

⸻

🤖 Scoring Models (Plug-in Interface)

Implement your model in src/scoring/openai_client_stub.py or add new clients.
Contract (src/scoring/interface.py):
	•	Input: prompt + options.
	•	Output: per-option probabilities p(A..D) (normalized), and the chosen label.

Critical requirements (to avoid common pitfalls):
	•	Compute per-option scores and softmax across A/B/C/D.
Do NOT use “first generated token probability” as confidence.
	•	Disable sampling (temperature=0) for scoring.
	•	For multimodal items (Eedi), make sure to include the image in the scoring request.

Environment variables (as needed):

OPENAI_API_KEY=...
DASHSCOPE_API_KEY=...      # Qwen
DEEPSEEK_API_KEY=...       # DeepSeek


⸻

🧪 Running Experiments

1) Score a dataset

# Example: RACE test with your model, 5 rounds for Data Map
python runner.py score --dataset race --split test \
  --model openai:gpt-4o --rounds 5 --out outputs/race_gpt4o

This writes outputs/<run>/scores.csv with:
	•	question_id, run_id, chosen, p_correct, correct

2) Build a Data Map

python runner.py datamap \
  --inp outputs/race_gpt4o/scores.csv \
  --out outputs/race_gpt4o \
  --quantile

	•	Aggregates to per-item mean_p, std_p, acc, n.
	•	Assigns regions (easy|ambiguous|hard|impossible) using adaptive quantiles (or fixed thresholds if you omit --quantile).

3) Calibrate Confidence (ECE / Temperature Scaling)

python runner.py calibrate \
  --inp outputs/race_gpt4o/scores.csv \
  --out outputs/race_gpt4o

Outputs calibration.json with:
	•	ece_before/after, brier_before/after, temperature T, NLL changes.

Use calibrated probabilities when comparing thresholds across models (to avoid unfair comparisons due to calibration drift).

4) Human–Machine Alignment (Eedi)

python runner.py align \
  --scores outputs/eedi_gpt4o/scores.csv \
  --human  data/eedi/processed/question_summary_task34.csv \
  --out    outputs/eedi_gpt4o

	•	Produces merged table and reports Spearman/Kendall between human difficulty (wrong_conf) and model difficulty (1 - mean_p(correct)).

⸻

📈 Metrics & Definitions
	•	Model-side
	•	p(correct): probability assigned to the gold option.
	•	Data Map: per-item mean_p and std_p across rounds; regioning via thresholds or quantiles.
	•	ECE (bin=10) and Brier score; optional temperature scaling.
	•	Stability: number of unique answers across rounds (not exposed by default; easy to add).
	•	Human-side (Eedi)
	•	error_rate = 1 - mean(IsCorrect)
	•	wrong_conf = mean((1 - IsCorrect) * Confidence/100) ← captures “confidently wrong”
	•	option_entropy across student choices
	•	human_label_v2 via quantiles over wrong_conf (simpleH|mediumH|hardH)

We keep diff_conf_old = 1 - mean(IsCorrect * Confidence/100) for completeness, but analyses should prefer wrong_conf when the goal is to weight confidence in mistakes.

⸻

📤 Typical Outputs
	•	scores.csv — per round predictions and p_correct
	•	datamap.csv — per item mean_p, std_p, acc, region
	•	calibration.json — ECE/Brier before/after, temperature T
	•	human_machine_merge.csv — for correlation analyses (Eedi)

You can add plots via src/analysis/tables_figures.py (reliability diagrams, Data Map scatter, heatmaps).

⸻

🧭 Best Practices & Pitfalls (Read before you run!)
	•	Don’t use first-token logprob as “confidence”. Always compute per-option probabilities and normalize across A–D.
	•	Turn off sampling for scoring (temperature=0). Use multiple rounds only when you want variability estimates (e.g., Data Map).
	•	Calibrate before comparing models with a fixed threshold (e.g., τ=0.8). Alternatively, compare at model-specific quantiles to equalize support.
	•	Feed images for Eedi items that require them; text-only inputs are not comparable to human performance.
	•	Thresholds for Data Map regions are data- and model-dependent; prefer quantile-based or clustering approaches for robust partitions.

⸻

🛠 Extending / Replacing Models
	•	Implement ModelClient.score_mcq(prompt, options) → {probs, chosen}.
	•	For OpenAI-like APIs, two robust scoring patterns:
	1.	Per-option continuation scoring (average token log-likelihood over the option text).
	2.	Label-only next-token scoring (restrict logits to A|B|C|D via logit bias and normalize).
	•	Add your client under src/scoring/ and switch with --model your_client:your_model_name.

⸻

🔁 Reproducibility Checklist
	•	Random seeds specified in clients and sampling (if used).
	•	Rounds --rounds N documented for each run.
	•	Calibration split (80/20) fixed in calibrate.
	•	Dataset splits fixed (RACE test or your chosen split).
	•	All scripts produce versioned outputs under outputs/<run>/.

⸻

📜 License & Data Terms
	•	Code is released under MIT License (see LICENSE if included).
	•	Datasets (RACE, Eedi) are subject to their original licenses; please obtain and use them accordingly.
	•	Respect privacy and data-use restrictions for any student interaction logs.
