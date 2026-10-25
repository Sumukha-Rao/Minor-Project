# Dataset analysis (before training)

Original files are read only. See inventory.json for schemas, missing values, lengths, label examples, duplicate counts, and cross-file overlaps.

| File | Rows | Labels |
|---|---:|---|
| prompt injection dataset/hugging face/data/test-00000-of-00001-701d16158af87368.csv | 116 | {'label': {'1': 60, '0': 56}} |
| prompt injection dataset/hugging face/data/test-00000-of-00001-701d16158af87368.parquet | 116 | {'label': {'1': 60, '0': 56}} |
| prompt injection dataset/hugging face/data/train-00000-of-00001-9564e8b05b4757ab.csv | 546 | {'label': {'0': 343, '1': 203}} |
| prompt injection dataset/hugging face/data/train-00000-of-00001-9564e8b05b4757ab.parquet | 546 | {'label': {'0': 343, '1': 203}} |
| prompt injection dataset/memoryInjection/C.A.M.S/dataset-cams.csv | 80000 | {'label': {'ambiguous': 20000, 'legitimate': 20000, 'minja': 20000, 'mextra': 20000}, 'subcategory': {'vague_query': 20000, 'standard_lookup': 7000, 'multi_patient_comparison': 6500, 'conversational_followup': 6500, 'stage2_compressed': 5000, 'direct_probe': 5000, 'paraphrased': 5000, 'stage1_full_bridge': 5000, 'stage3_further_compressed': 5000, 'indirect': 5000, 'web_agent': 5000, 'medication_remap': 2500, 'stage4_minimal': 2500}} |
| prompt injection dataset/prompt-injection-attack-dataset/complete_dataset.csv | 3747 | {'target_label': {'1': 2000, '0': 1747}, 'inject_label': {'0': 2000, '1': 1747}} |
| prompt injection dataset/prompt-injection-safety/data/test-00000-of-00001.parquet | 10000 | {'label': {'0': 4557, '1': 4443, '2': 1000}} |
| prompt injection dataset/prompt-injection-safety/data/train-00000-of-00001.parquet | 50000 | {'label': {'0': 23414, '1': 22586, '2': 4000}} |
| prompt injection dataset/prompt-injections-benchmark/data/test-00000-of-00001.parquet | 5000 | {'label': {'benign': 2997, 'jailbreak': 2003}} |
| prompt injection dataset/prompt-injections-benchmark/test.csv | 5000 | {'label': {'benign': 3001, 'jailbreak': 1999}} |
| prompt injection dataset/prompt_injections/prompt_injections.csv | 1034 | {} |

## Verified admission policy

- Benchmark: benign → 0; jailbreak → 1, per local README. Parquet is the canonical test revision. CSV is audit-only: conflicting labels on shared texts are quarantined; CSV-only rows are not added. Jailbreak detection is a related but broader construct.
- prompt_injections: all rows → 1, per local README.
- C.A.M.S: literal legitimate → 0; ambiguous/mextra/minja quarantined. Patient number variants share groups.
- Safety: 0/1/2 have no documented meanings in local README; entire source excluded.
- hugging face: 0/1 lack documented meanings and examples include ordinary instructions labeled 1; excluded.
- Attack dataset: target_label and inject_label are task outcomes, not detector labels. All attack variants and task metadata inventoried; excluded pending an explicit derived-label protocol.

## Leakage policy

Exact raw-text label conflicts are quarantined. Duplicate rows retain provenance. NFKC/case/whitespace canonicalization is used only for grouping. Near duplicates are connected components of verified character 5-shingle Jaccard ≥ 0.85. The prefix join is exhaustive under that lexical definition; semantic paraphrases may remain. Known numeric templates are grouped. Training text is not normalized.
Official test rows stay test. Development rows connected to test are quarantined. Remaining data is split by stratified groups. TF-IDF vocabulary and hyperparameters use training folds only. Thresholds use validation only.

## Critical limitations

**Development class and source are confounded:** currently admitted benign training data comes from C.A.M.S and malicious training data from prompt_injections. High validation metrics may reflect source identification. The external benchmark tests domain shift, not comprehensive prompt-injection security. Additional documented binary data is required for credible conclusions.
No language, encoded-payload, long-context, educational-benign, or subtle-attack labels are invented. Available C.A.M.S subcategories are preserved but currently contain only admitted benign examples. Family-held-out attack evaluation is unavailable with this admission policy.

## Preparation results

```json
{
  "admitted_rows_before_dedup": 26034,
  "unverified_rows_in_selected_sources": 60000,
  "cross_split_rows_quarantined": 0,
  "exact_duplicates_removed": 9694,
  "conflict_texts": 8,
  "exact_cross_split_texts": 0,
  "near_duplicate_edges": 489,
  "groups": 5954,
  "audit_only_nonconflicting_rows": 4992,
  "mixed_label_groups": 0,
  "largest_groups": {
    "0008ddc6314b613d21e8cb4cc4e53b214465e90e327b799812d7d886132d5de8": 854,
    "00cd9702a20b0376eab32a36f888f37ca22ff0e607a5a69bccf3a398c503c1d5": 839,
    "004699eb7f288d6bfee5117bcfdc50e7faac6e826cbb9c8c98fde02daf9680a7": 829,
    "0091977b4cc3228a0d28011124b4ce46096c75b407f55898552aeb6824be7367": 810,
    "00047491f83dd93fbbad859992ce76c3c2dfbe4eec1b6816772354bd2c43f184": 782,
    "003d997f1db1265845506d7f3e26b43ad56f6c5fc0dec62aa0434fba0634a8b2": 779,
    "000880b27348529a840ab64d63dfd029d03225f4fffaf6739e22b824b5ceee75": 769,
    "00136917bd062d006478fcf75f5667ea0aeb2d495ae798b516597abc06da1a9f": 769,
    "00714557c44c2eb03be6fcc6908a566413252807ef672a5ea40efd6ad0fe3071": 199,
    "02b8188cdaa5b03bb6a4b275ffc0a0974b57e10a1cd6d1a898f1aa082c20e0fb": 198
  },
  "fingerprint": "dc8c56e4ad4bdebd4c80f3801858a614d43a1130f2b076d08fe8eb58f3dcbabd",
  "partitions": {
    "test": {
      "rows": 4992,
      "labels": {
        "0": 2995,
        "1": 1997
      },
      "groups": 4951,
      "source_by_label": {
        "('prompt injection dataset/prompt-injections-benchmark/data/test-00000-of-00001.parquet', 0)": 2995,
        "('prompt injection dataset/prompt-injections-benchmark/data/test-00000-of-00001.parquet', 1)": 1997
      }
    },
    "train": {
      "rows": 9249,
      "labels": {
        "0": 8468,
        "1": 781
      },
      "groups": 804,
      "source_by_label": {
        "('prompt injection dataset/memoryInjection/C.A.M.S/dataset-cams.csv', 0)": 8468,
        "('prompt injection dataset/prompt_injections/prompt_injections.csv', 1)": 781
      }
    },
    "validation": {
      "rows": 2091,
      "labels": {
        "0": 1898,
        "1": 193
      },
      "groups": 199,
      "source_by_label": {
        "('prompt injection dataset/memoryInjection/C.A.M.S/dataset-cams.csv', 0)": 1898,
        "('prompt injection dataset/prompt_injections/prompt_injections.csv', 1)": 193
      }
    }
  }
}
```