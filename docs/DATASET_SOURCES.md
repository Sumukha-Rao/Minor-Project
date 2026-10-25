# Dataset sources

Dataset directories are tracked as ordinary files in this repository. Commit dataset changes from the project root; no submodule initialization is needed. The source revisions below identify the original snapshots. The two CSV exports in `hugging face/data/` were already present locally and are included in this repository.

| Directory | Upstream | Source revision |
| --- | --- | --- |
| `prompt injection dataset/hugging face` | https://huggingface.co/datasets/deepset/prompt-injections | `4f61ecb038e9c3fb77e21034b22511b523772cdd` |
| `prompt injection dataset/memoryInjection/C.A.M.S` | https://github.com/Dhiv123/C.A.M.S.git | `43d8d016240f45784eb79566d4b1e4c88f36c4bc` |
| `prompt injection dataset/prompt-injection-attack-dataset` | https://huggingface.co/datasets/xxz224/prompt-injection-attack-dataset | `3b49e90e56cdab5552ccec971d487ad4f2d393ad` |
| `prompt injection dataset/prompt-injection-safety` | https://huggingface.co/datasets/jayavibhav/prompt-injection-safety | `c0f55c8817fb2d4da72ac7b06deba6033da6fad6` |
| `prompt injection dataset/prompt-injections-benchmark` | https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark | `9ef1aa46a7e5eedb096be0481be8011ede1e72e8` |
| `prompt injection dataset/prompt_injections` | https://huggingface.co/datasets/yanismiraoui/prompt_injections | `bd55359f2f332afc35f277ac3dd08f7111b024c9` |

Upstream Git metadata was preserved locally under `.git/nested-repository-backups/` during conversion; it is not part of a clone. Dataset content and existing license files are preserved. Local attribute overrides store files directly in Git rather than depending on upstream Git LFS objects.
