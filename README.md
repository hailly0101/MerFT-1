## MerFT

🧠 RoMQD: A Multimodal Meme Reasoning Dataset for Social Conflict Interpretation
RoMQD is a multimodal QA dataset focused on social conflict memes, and MerFT (Meme Exploration via Multimodal Retrieval-Augmented Fine-tuning) is a reasoning framework that leverages images, captions, and documents for robust interpretation in a complex sociocultural context.

📝 Overview
Social media is a key platform for expressing and spreading social conflicts. In particular, memes have emerged as a powerful medium that conveys complex sociocultural messages through visual satire and symbolism.

This project introduces:

RoMQD: A novel multimodal dataset tailored for interpreting socially controversial memes via QA pairs.

MerFT: A robust Retrieval-Augmented Generation (RAG)-based framework that jointly uses image, caption, and document-level context to understand memes with high reasoning quality.

MerFT incorporates distractor-aware fine-tuning and citation-based Chain-of-Thought (CoT) reasoning to enable robust inference even in environments with noisy or mixed information.

### 📁 Project Structure

```
MerFT/
├── README.md                      # English docs (this file)
├── README_ko.md                   # Korean docs
├── merft_wsdm___Copy_.pdf|.txt    # Paper and extracted text
├── data/                          # API datasets (kept from Gorilla; optional)
├── raft/
│   ├── requirements.txt
│   ├── logconf.py, logging.yaml
│   ├── client_utils.py            # OpenAI/Embeddings helpers
│   ├── multimodal_cluster.py      # CLIP embeddings, KMeans, materialization
│   ├── format.py                  # HF/completion/chat/eval format conversion
│   ├── meme_qa_generator.py       # QA generation with distractor strategies
│   ├── meme_data/
│   │   ├── docs/
│   │   │   ├── environment/*.txt  # Labeled topic docs
│   │   │   └── unlabeled/         # Unlabeled doc groups (optional)
│   │   ├── memes/
│   │   │   ├── environment/...    # Labeled topic images
│   │   │   └── unlabeled/         # Unlabeled image groups (optional)
│   │   └── scripts/
│   │       ├── generate_meme_dataset_json.py  # Build input JSON
│   │       └── organize_unlabeled.py          # Semi-supervised organizing
│   ├── raft.py, raft_local.py, eval.py        # General RAFT utilities (optional)
│   └── openai_patch.py, checkpointing.py, env_config.py (optional)
└── backup/                         # Only if created during maintenance
```

### ⚙️ Installation & Execution

1. Install dependencies
   ```bash
   pip install -r raft/requirements.txt
   ```
2. Prepare the input dataset (JSON format)
   ```bash
   python raft/meme_data/scripts/generate_meme_dataset_json.py --help
   ```
3. Generate QA pairs using OpenAI models
   
   Base (random distractors within category):
   
   ```bash
    python raft/meme_qa_generator.py \
     --input /path/to/meme_dataset.json \
     --documents_root /path/to/docs_root \
     --output /path/to/output.json \
     --openai_key $OPENAI_API_KEY \
     --model gpt-4o-mini \
     --workers 2 \
     --distractor_strategy random
   ```

   Similarity-based (dense retrieval with multimodal embeddings; supports unlabeled docs):
   
   ```bash
    python raft/meme_qa_generator.py \
     --input /path/to/meme_dataset.json \
     --documents_root /path/to/docs_root \
     --unlabeled_docs_root /path/to/unlabeled_docs \
     --unlabeled_memes_root /path/to/unlabeled_images \
     --embedder_model clip-ViT-B-32 \
     --distractor_strategy similarity \
     --output /path/to/output.json \
     --openai_key $OPENAI_API_KEY \
     --model gpt-4o-mini \
     --workers 2
   ```

   Clustering-based (as in the paper: cluster oracle+pool and select hard negatives from the same cluster; optionally materialize clusters):
   
   ```bash
    python raft/meme_qa_generator.py \
     --input /path/to/meme_dataset.json \
     --documents_root /path/to/docs_root \
     --unlabeled_docs_root /path/to/unlabeled_docs \
     --unlabeled_memes_root /path/to/unlabeled_images \
     --embedder_model clip-ViT-B-32 \
     --distractor_strategy clustering \
     --cluster_k 32 \
     --cluster_output /path/to/clustered_output \
     --output /path/to/output.json \
     --openai_key $OPENAI_API_KEY \
     --model gpt-4o-mini \
     --workers 2
   ```

   Replace $OPENAI_API_KEY with your actual OpenAI API key.

### New Features (MerFT paper-aligned)

- Distractor strategies:
  - random: previous behavior (category pool + random sampling)
  - similarity: multimodal dense retrieval (text+image via CLIP) top-k similar, excluding near-duplicates
  - clustering: multimodal KMeans; select hard negatives from the same cluster as the oracle
- Semi-supervised pooling:
  - Provide `--unlabeled_root` to include unlabeled `.txt` docs in the pool, enabling cluster-based selection with pseudo-labeled data, per paper Section 2.4/2.5
- Cluster materialization:
  - Set `--cluster_output` to export `cluster_{id}/docs/*.txt` for inspection or downstream processing; folders are auto-created when sufficient items exist.

## End-to-End Quickstart (one-shot)

1) Build input JSON (scan subfolders recursively)

```bash
python raft/meme_data/scripts/generate_meme_dataset_json.py \
  --images_dir raft/meme_data/memes \
  --docs_dir raft/meme_data/docs/environment \
  --output raft/meme_data/scripts/environment.json \
  --recursive
```

2) Organize unlabeled groups into labeled topics (copy + mark done)

```bash
python raft/meme_data/scripts/organize_unlabeled.py \
  --labeled_docs_root raft/meme_data/docs \
  --labeled_memes_root raft/meme_data/memes \
  --unlabeled_docs_root raft/meme_data/docs/unlabeled \
  --unlabeled_memes_root raft/meme_data/memes/unlabeled \
  --embedder_model clip-ViT-B-32 \
  --assign_method centroid \
  --mark_done --skip_done
```

3) Generate QA with clustering-based hard-negative selection and cluster materialization

```bash
python raft/meme_qa_generator.py \
  --input raft/meme_data/scripts/environment.json \
  --documents_root raft/meme_data/docs/environment \
  --unlabeled_docs_root raft/meme_data/docs/unlabeled \
  --unlabeled_memes_root raft/meme_data/memes/unlabeled \
  --embedder_model clip-ViT-B-32 \
  --distractor_strategy clustering \
  --cluster_k 32 \
  --cluster_output clustered_output \
  --cluster_min_size 10 \
  --output output.json \
  --openai_key $OPENAI_API_KEY \
  --model gpt-4o-mini \
  --workers 2
```

Variation: choose `--distractor_strategy similarity` for dense-retrieval negatives or `random` for baseline.

## Why MerFT? What problems does it solve?

MerFT targets robust multimodal meme understanding in retrieval-aware settings where documents may include irrelevant or misleading evidence. It is purpose-built to:

- Decode culturally nuanced memes by combining image, caption, and external documents
- Train for robustness via distractor-aware fine-tuning and Chain-of-Thought reasoning with explicit document attribution
- Leverage semi-supervised clustering to incorporate unlabeled data and select hard negatives that are semantically close yet answer-irrelevant (near-miss distractors)

This system is useful for:
- Social media analysis teams studying satire, ideology, and conflict narratives
- Researchers building retrieval-augmented multimodal models robust to noisy evidence
- Practitioners needing explainable, document-grounded QA over memes and cultural artifacts

## How this matches the paper

This repository operationalizes the paper’s design:

- Multimodal inputs (image + caption + documents) with three modes (Base/Caption/Both)
- Distractor-aware training with variable distractor counts (k=0..5), omitting oracle at k=5
- Semi-supervised clustering: unlabeled documents and images are embedded jointly (CLIP), grouped, and used both to organize topic folders and to select hard negatives from semantically close clusters
- Citation-based Chain-of-Thought prompting with explicit reasoning spans and final answer markers

The result is a robust meme reasoning pipeline that stays accurate under noisy retrieval and produces interpretable outputs.

## Command-line options (detailed)

### Dataset JSON builder: `raft/meme_data/scripts/generate_meme_dataset_json.py`

- **--images_dir Path (required)**: Meme images directory. If `--recursive` is set, all immediate subfolders are scanned and combined.
- **--docs_dir Path (required)**: Directory containing `.txt` documents where the first line is a link and the rest is text. File stem must match the image folder keyword after normalization.
- **--output Path (required)**: Output JSON path containing entries `[title, keyword, image_path, doc_link, doc_text]`.
- **--recursive (flag)**: Recursively scan `--images_dir` subfolders and aggregate all images into a single JSON.

Notes:
- Keyword normalization removes numeric prefixes like `10_` and suffixes like `_meme_pinterest` or `_meme_reddit` to match doc filenames.

### Unlabeled organizer: `raft/meme_data/scripts/organize_unlabeled.py`

- **--labeled_docs_root Path (required)**: Root with labeled topic subfolders for documents (e.g., `docs/environment`, `docs/politics`, ...). Used to build topic centroids.
- **--labeled_memes_root Path (optional)**: Root with labeled topic subfolders for images; improves multimodal centroids if present.
- **--unlabeled_docs_root Path (required)**: Root with unlabeled docs organized as subfolders per group/topic candidate (e.g., `docs/unlabeled/<group>/*.txt`).
- **--unlabeled_memes_root Path (optional)**: Root with unlabeled images mirroring doc subfolders (e.g., `memes/unlabeled/<group>/*.{jpg,png}`) for multimodal assignment.
- **--output_docs_root Path (optional)**: Destination for copied docs; defaults to `--labeled_docs_root`.
- **--output_memes_root Path (optional)**: Destination for copied images; defaults to `--labeled_memes_root`.
- **--embedder_model str (default: clip-ViT-B-32)**: Sentence-Transformers model used for multimodal embeddings.
- **--assign_method str [centroid|kmeans] (default: centroid)**: How to assign unlabeled groups to topics. `centroid` uses nearest labeled-topic centroid. `kmeans` uses unlabeled clustering to refine (requires `--k`).
- **--k int (default: 0)**: Number of clusters for `kmeans` assignment; 0 means auto.
- **--mark_done (flag)**: After copying, rename processed unlabeled subfolders with `[end]_` prefix to skip next runs.
- **--skip_done (flag)**: Skip unlabeled subfolders starting with `[end]_`.

Behavior:
- Builds topic centroids from labeled data (semi-supervised anchor).
- Assigns each unlabeled group to the closest topic using multimodal similarity.
- Copies docs and images into the decided topic folders; original groups can be marked as done.

### QA generator: `raft/meme_qa_generator.py`

Required I/O
- **--input Path (required)**: Input meme JSON generated by the dataset JSON builder.
- **--documents_root Path (required)**: Root folder with topic/category docs. Each category in this project maps to reasoning categories inside the script.
- **--output Path (required)**: Output dataset path; the script will also generate per-distractor-count files.

OpenAI and execution
- **--openai_key str (required)**: OpenAI API key.
- **--model str (default: gpt-4-mini)**: Chat model to call.
- **--workers int (default: 4)**: Thread pool size.

Distractor selection strategies
- **--distractor_strategy [random|similarity|clustering] (default: random)**
  - random: Sample distractors randomly from the pool (excluding near-duplicates of the oracle by first 100 chars).
  - similarity: Use multimodal embeddings to pick top-k most similar items to the oracle (image+text fused).
  - clustering: KMeans on oracle+pool embeddings, then pick hard negatives from the oracle’s cluster.

Semi-supervised and multimodal resources
- **--unlabeled_docs_root Path (optional)**: Include unlabeled docs into the candidate pool. Each `.txt` is added; its folder/file stem is used to pair images if available.
- **--unlabeled_memes_root Path (optional)**: Include unlabeled images; paired to docs by normalized folder/filename stem for joint embeddings.
- **--embedder_model str (default: clip-ViT-B-32)**: CLIP-based Sentence-Transformers model for multimodal embeddings.

Clustering controls
- **--cluster_k int (default: 20)**: KMeans cluster count for `clustering` strategy (auto-adjusted if pool is small).
- **--cluster_output Path (optional)**: If set, materializes clusters to `cluster_{id}/docs` and `cluster_{id}/images` for inspection (copies, not moves).
- **--cluster_min_size int (default: 10)**: Minimum cluster size to keep when materializing; smaller clusters can be filtered.

Other behaviors
- Distractor counts are generated for all values in `[0,1,2,3,4,5]`. When `k=5`, the oracle document is omitted by design.
- If the candidate pool is smaller than requested, dummy documents are auto-filled (never identical to oracle start).

📊 Experimental Highlights
Input Type Accuracy Reasoning Quality Robustness to Distractors
Base RAG Moderate Unstable Vulnerable
MerFT (Ours) High Strong Robust

Performance was evaluated across various input configurations (Image-only, Caption-only, Image+Caption) and distractor densities.

MerFT consistently outperformed baselines in both accuracy and reasoning clarity, even under noisy input conditions.

🔍 Key Contributions
A new QA-style dataset (RoMQD) for the interpretation of socially symbolic memes.

A multimodal RAG framework (MerFT) with fine-tuning strategies tailored to satire, irony, and complex social context.

Experiments on modality ablations, category-wise weaknesses, and distractor robustness.
