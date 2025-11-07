# Leakage & Bias Audit (Auto)

## Near-duplicate detection (Jaccard on unigram tokens)
- RACE near-duplicates (Jaccard≥0.90): 21 pairs
- EEDI near-duplicates (Jaccard≥0.90): 1 pairs
- Cross-dataset high matches (≥0.95): 207 pairs

### RACE top pairs
```
      qid_1       qid_2  jaccard
    high_30   high_2944 1.000000
  high_1947   high_1966 1.000000
 middle_158  middle_200 1.000000
 middle_148  middle_200 1.000000
 middle_148  middle_158 1.000000
 middle_146  middle_155 1.000000
  high_2113   high_2142 1.000000
  high_2097   high_2122 1.000000
middle_1076 middle_1089 1.000000
  high_1935   high_1965 1.000000
   high_927  middle_165 1.000000
   high_165    high_204 0.967742
```

### EEDI top pairs
```
  qid_1    qid_2  jaccard
eedi_92 eedi_107 0.956522
```

### Cross-dataset top pairs
```
qid_race_raw qid_eedi_raw  match_score
      high_0     eedi_198          1.0
  middle_525     eedi_126          1.0
  middle_418     eedi_102          1.0
  middle_427     eedi_199          1.0
  middle_432       eedi_9          1.0
  middle_460      eedi_16          1.0
  middle_465     eedi_173          1.0
  middle_466      eedi_74          1.0
  middle_472     eedi_185          1.0
  middle_513     eedi_100          1.0
  middle_519     eedi_144          1.0
  middle_520      eedi_66          1.0
```

## Length bias (per-question error vs token length, Spearman)
```
dataset                                   model   n  spearman_err_vs_len
   RACE                               drobertaB 600             0.042847
   RACE                                robertaL 600            -0.017819
   RACE                                  stage3 600            -0.015220
   EEDI                  eedi_gpt4o_tau08_joint  71             0.110388
   EEDI             eedi_gpt4o_tau08_model_tags  71             0.110388
   EEDI     eedi_gpt4o_tau08_model_tags_aligned  71             0.110388
   EEDI              eedi_gpt4omini_tau08_joint  71            -0.045204
   EEDI         eedi_gpt4omini_tau08_model_tags  71            -0.045204
   EEDI eedi_gpt4omini_tau08_model_tags_aligned  71            -0.045204
   EEDI      stage2_model_tags_eedi_gpt4o_tau08  71             0.110388
   EEDI  stage2_model_tags_eedi_gpt4omini_tau08  71            -0.045204
```

_Notes_: Tokenization is whitespace; thresholds chosen for quick audit. Full audit may use character n-gram cosine with stringent thresholds and manual spot checks.