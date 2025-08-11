from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover - optional import error message
    SentenceTransformer = None  # type: ignore


@dataclass
class PoolItem:
    path: Optional[str]
    content: str
    image_path: Optional[str] = None


class MultimodalEmbedder:
    """
    Simple multimodal embedder using a CLIP model from sentence-transformers.
    - Texts are encoded via the text encoder
    - Images are encoded via the image encoder
    - For joint representations, we L2-normalize and average available modalities
    """

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required. Please install it (pip install sentence-transformers)."
            )
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 512), dtype=np.float32)
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return emb.astype(np.float32)

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, 512), dtype=np.float32)
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                # Fallback: blank image embedding if loading fails
                images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
        emb = self.model.encode(images, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return emb.astype(np.float32)

    def embed_joint(self, texts: List[str], image_paths: Optional[List[Optional[str]]] = None) -> np.ndarray:
        """
        Compute joint embeddings by averaging normalized text and image embeddings when both are available.
        If image_paths[i] is None or invalid, use text-only.
        """
        text_emb = self.embed_texts(texts)
        if image_paths is None:
            return text_emb

        # Prepare image list aligned with texts
        aligned_image_paths: List[str] = []
        has_image_mask: List[bool] = []
        for p in image_paths:
            if p and Path(p).exists():
                aligned_image_paths.append(p)
                has_image_mask.append(True)
            else:
                aligned_image_paths.append("")
                has_image_mask.append(False)

        # Compute image embeddings only for valid paths; fill zeros otherwise
        valid_indices = [i for i, ok in enumerate(has_image_mask) if ok]
        image_emb = np.zeros_like(text_emb, dtype=np.float32)
        if valid_indices:
            valid_paths = [aligned_image_paths[i] for i in valid_indices]
            valid_emb = self.embed_images(valid_paths)
            for j, i in enumerate(valid_indices):
                image_emb[i] = valid_emb[j]

        # Average and renormalize
        joint = text_emb + image_emb
        # L2 normalize
        norms = np.linalg.norm(joint, axis=1, keepdims=True) + 1e-12
        joint = joint / norms
        return joint.astype(np.float32)


def build_unlabeled_doc_pool(root: Path) -> List[PoolItem]:
    items: List[PoolItem] = []
    if not root or not root.exists():
        return items
    for p in root.rglob("*.txt"):
        try:
            content = p.read_text(encoding="utf-8")
            items.append(PoolItem(path=str(p), content=content))
        except Exception:
            continue
    return items


def kmeans_cluster(embeddings: np.ndarray, k: int, random_state: int = 42) -> Tuple[KMeans, np.ndarray]:
    if len(embeddings) < k:
        # Reduce k if not enough points
        k = max(1, len(embeddings))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(embeddings)
    return model, labels


def topk_similar_indices(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    top_k: int,
    exclude: Optional[List[int]] = None,
    sim_range: Tuple[float, float] = (0.4, 0.97),
) -> List[int]:
    if candidate_vecs.size == 0:
        return []
    sims = cosine_similarity(query_vec.reshape(1, -1), candidate_vecs).ravel()
    idx = np.argsort(-sims)  # descending
    selected: List[int] = []
    for i in idx:
        if exclude and i in exclude:
            continue
        if sims[i] < sim_range[0] or sims[i] > sim_range[1]:
            continue
        selected.append(i)
        if len(selected) >= top_k:
            break
    # If not enough, relax thresholds
    if len(selected) < top_k:
        for i in idx:
            if exclude and i in exclude:
                continue
            if i in selected:
                continue
            selected.append(i)
            if len(selected) >= top_k:
                break
    return selected


def materialize_clusters(
    output_dir: Path,
    labels: np.ndarray,
    pool_items: List[PoolItem],
    copy_files: bool = True,
) -> None:
    """
    Persist clustered documents to disk for reproducibility.
    For items with a path, copy the file; otherwise, write the content to a new file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_clusters = int(labels.max()) + 1 if labels.size else 0
    for cid in range(num_clusters):
        cluster_dir = output_dir / f"cluster_{cid}"
        docs_dir = cluster_dir / "docs"
        imgs_dir = cluster_dir / "images"
        docs_dir.mkdir(parents=True, exist_ok=True)
        imgs_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(pool_items):
        cid = int(labels[i]) if labels.size else 0
        base_dir = output_dir / f"cluster_{cid}"
        docs_dir = base_dir / "docs"
        imgs_dir = base_dir / "images"
        if item.path and os.path.exists(item.path) and copy_files:
            # Copy preserving filename
            try:
                dst = docs_dir / Path(item.path).name
                if str(Path(item.path).resolve()) != str(dst.resolve()):
                    # Avoid copying onto itself
                    from shutil import copy2
                    copy2(item.path, dst)
            except Exception:
                # Fallback to write content
                (docs_dir / f"doc_{i}.txt").write_text(item.content, encoding="utf-8")
        else:
            (docs_dir / f"doc_{i}.txt").write_text(item.content, encoding="utf-8")

        # Copy paired image if present
        if item.image_path and os.path.exists(item.image_path) and copy_files:
            try:
                from shutil import copy2
                img_dst = imgs_dir / Path(item.image_path).name
                if str(Path(item.image_path).resolve()) != str(img_dst.resolve()):
                    copy2(item.image_path, img_dst)
            except Exception:
                pass


