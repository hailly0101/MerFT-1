from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import numpy as np

from multimodal_cluster import MultimodalEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("organize_unlabeled")


def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Organize unlabeled docs/images into labeled topic folders using multimodal embeddings.")
    p.add_argument("--labeled_docs_root", type=Path, required=True, help="Root with labeled topic subfolders for documents")
    p.add_argument("--labeled_memes_root", type=Path, default=None, help="Optional root with labeled topic subfolders for images")
    p.add_argument("--unlabeled_docs_root", type=Path, required=True, help="Root with unlabeled docs (topic subfolders allowed)")
    p.add_argument("--unlabeled_memes_root", type=Path, required=False, default=None, help="Root with unlabeled images mirroring docs structure")
    p.add_argument("--output_docs_root", type=Path, required=False, help="Where to copy docs; defaults to labeled_docs_root")
    p.add_argument("--output_memes_root", type=Path, required=False, help="Where to copy images; defaults to labeled_memes_root if provided")
    p.add_argument("--embedder_model", type=str, default="clip-ViT-B-32")
    p.add_argument("--assign_method", type=str, default="centroid", choices=["centroid", "kmeans"], help="Assign method for topics")
    p.add_argument("--k", type=int, default=0, help="Number of clusters if using kmeans (0 -> auto)")
    p.add_argument("--mark_done", action="store_true", help="Rename processed unlabeled subfolders by prefixing [end]_ to skip next time")
    p.add_argument("--skip_done", action="store_true", help="Skip unlabeled subfolders that already start with [end]_")
    return p.parse_args()


def list_topic_subfolders(root: Path, skip_done: bool) -> List[Path]:
    subs: List[Path] = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        if skip_done and sub.name.startswith("[end]_"):
            continue
        subs.append(sub)
    return subs


def build_topic_centroids(embedder: MultimodalEmbedder, labeled_docs_root: Path, labeled_memes_root: Optional[Path]) -> Tuple[List[str], np.ndarray]:
    topic_names: List[str] = []
    topic_centroids: List[np.ndarray] = []
    for topic_dir in sorted([d for d in labeled_docs_root.iterdir() if d.is_dir()]):
        docs = list(topic_dir.glob("*.txt"))
        if not docs:
            continue
        texts = [read_txt(p) for p in docs]
        # Optional: pair images by topic folder name under labeled_memes_root
        image_paths: Optional[List[Optional[str]]] = None
        if labeled_memes_root and (labeled_memes_root / topic_dir.name).exists():
            img_dir = labeled_memes_root / topic_dir.name
            images_all = [str(p) for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            # Repeat or trim to match doc count for simple pairing
            if images_all:
                reps = max(1, len(texts) // len(images_all) + (1 if len(texts) % len(images_all) else 0))
                paired = (images_all * reps)[: len(texts)]
                image_paths = paired
        vecs = embedder.embed_joint(texts, image_paths)
        centroid = vecs.mean(axis=0, keepdims=True)
        topic_names.append(topic_dir.name)
        topic_centroids.append(centroid)
    if not topic_centroids:
        raise RuntimeError("No labeled topics with documents found to build centroids.")
    centroids = np.vstack(topic_centroids)
    return topic_names, centroids


def copy_group_to_topic(
    group_dir_docs: Path,
    group_dir_images: Optional[Path],
    out_docs_topic_dir: Path,
    out_images_topic_dir: Optional[Path],
) -> None:
    out_docs_topic_dir.mkdir(parents=True, exist_ok=True)
    if out_images_topic_dir:
        out_images_topic_dir.mkdir(parents=True, exist_ok=True)
    # Copy docs
    for f in group_dir_docs.glob("*.txt"):
        try:
            shutil.copy2(str(f), str(out_docs_topic_dir / f.name))
        except Exception:
            continue
    # Copy images
    if group_dir_images and group_dir_images.exists():
        for p in group_dir_images.glob("*"):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            try:
                shutil.copy2(str(p), str(out_images_topic_dir / p.name))  # type: ignore[arg-type]
            except Exception:
                continue


def main():
    args = get_args()
    output_docs_root = args.output_docs_root or args.labeled_docs_root
    output_memes_root = args.output_memes_root or args.labeled_memes_root

    embedder = MultimodalEmbedder(args.embedder_model)
    topic_names, topic_centroids = build_topic_centroids(embedder, args.labeled_docs_root, args.labeled_memes_root)
    logger.info(f"Built {len(topic_names)} topic centroids: {topic_names}")

    # Iterate unlabeled groups as subfolders under unlabeled_docs_root
    for group in list_topic_subfolders(args.unlabeled_docs_root, args.skip_done):
        docs = list(group.glob("*.txt"))
        if not docs:
            logger.info(f"Skipping empty group: {group}")
            continue
        texts = [read_txt(p) for p in docs]
        # Pair images by same named subfolder under unlabeled_memes_root
        images_group_dir: Optional[Path] = None
        if args.unlabeled_memes_root:
            maybe = args.unlabeled_memes_root / group.name
            if maybe.exists():
                images_group_dir = maybe
        image_paths: Optional[List[Optional[str]]] = None
        if images_group_dir and any(images_group_dir.glob("*")):
            imgs_all = [str(p) for p in images_group_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            if imgs_all:
                reps = max(1, len(texts) // len(imgs_all) + (1 if len(texts) % len(imgs_all) else 0))
                image_paths = (imgs_all * reps)[: len(texts)]

        vecs = embedder.embed_joint(texts, image_paths)
        # Assign by nearest centroid
        sims = (vecs @ topic_centroids.T)
        assign_idx = int(np.argmax(sims.mean(axis=0)))  # group-level assignment by mean similarity
        topic = topic_names[assign_idx]
        logger.info(f"Assign group '{group.name}' -> topic '{topic}'")

        out_docs_dir = output_docs_root / topic
        out_images_dir = (output_memes_root / topic) if output_memes_root else None
        copy_group_to_topic(group, images_group_dir, out_docs_dir, out_images_dir)

        # Mark done by renaming folder names with [end]_ prefix
        if args.mark_done:
            try:
                group.rename(group.parent / ("[end]_" + group.name))
            except Exception:
                pass
            if images_group_dir and images_group_dir.exists():
                try:
                    images_group_dir.rename(images_group_dir.parent / ("[end]_" + images_group_dir.name))
                except Exception:
                    pass

    logger.info("Organization complete.")


if __name__ == "__main__":
    main()


