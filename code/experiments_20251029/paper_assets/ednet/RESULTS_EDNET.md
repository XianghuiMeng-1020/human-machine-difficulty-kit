# EdNet-KT1 (sampled 200 users) → proxy human difficulty

- Source: `data/ednet/EdNet-KT1.zip` (public KT benchmark, per-user CSV under `KT1/`)
- We sampled 200 user-files → 51,956 interactions → 9,473 unique item_ids.
- Because this slice of EdNet is *heavily skewed to hard items*, most items have high error (err > 0.7).
- We applied the same MV-HMDA proxy procedure with a low-coverage bucket:
  - low-coverage (<5 attempts): **5,837**
  - hard (err ≥ 0.7): **3,636**
  - no items fell into easy / medium under this slice
- Takeaway: our pipeline still runs on KT-style logs, but to recover a 3-way difficulty we need a wider slice (more users or longer time span).
