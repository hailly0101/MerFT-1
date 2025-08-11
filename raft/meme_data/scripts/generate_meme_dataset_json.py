#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=Path, required=True, help="Meme images directory (or a parent directory to scan subfolders)")
    p.add_argument("--docs_dir", type=Path, required=True, help="Documents directory containing *.txt")
    p.add_argument("--output", type=Path, required=True, help="Output JSON path")
    p.add_argument("--recursive", action="store_true", help="If set, scan images_dir recursively and generate a single combined JSON")
    return p.parse_args()


def clean_keyword(folder_name):
    """
    폴더 이름에서 키워드만 추출하는 전처리 함수.
    예: '1_earth-chan_meme_pinterest' -> 'earth-chan'
    """
    # 숫자, 밈, 플랫폼 이름 등을 제거 (필요시 정교화 가능)
    for prefix in ["1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_", "9_", "10_"]:
        if folder_name.startswith(prefix):
            folder_name = folder_name[len(prefix):]
    for suffix in ["_meme_pinterest", "_meme_reddit"]:
        if folder_name.endswith(suffix):
            folder_name = folder_name[:-len(suffix)]
    return folder_name


def main():
    args = get_args()
    images_dir = args.images_dir
    docs_dir = args.docs_dir
    output_json = args.output

    if not images_dir.is_dir() or not docs_dir.is_dir():
        print("❌ Missing directories", file=sys.stderr)
        sys.exit(1)

    # 문서 로딩
    doc_map = {}
    for doc in docs_dir.glob("*.txt"):
        keyword = doc.stem
        try:
            lines = doc.read_text(encoding='utf-8').splitlines()
            if not lines:
                continue
            doc_link = lines[0].strip()
            doc_text = "\n".join(lines[1:]).strip()
            doc_map[keyword] = (doc_text, doc_link)
        except Exception as e:
            print(f"⚠️ Skipping doc {doc.name}: {e}", file=sys.stderr)

    entries = []

    def build_entries_for_folder(folder: Path) -> list[dict]:
        keyword = clean_keyword(folder.name)
        if keyword not in doc_map:
            print(f"⚠️ No matching document for keyword: {keyword}", file=sys.stderr)
            return []
        doc_text, doc_link = doc_map[keyword]
        image_files = sorted([f for f in folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        local = []
        for idx, img in enumerate(image_files, start=1):
            local.append({
                "title": f"{keyword}_{idx}",
                "keyword": keyword,
                "image_path": str(img.resolve()),
                "doc_link": doc_link,
                "doc_text": doc_text
            })
        return local

    if args.recursive:
        for sub in images_dir.iterdir():
            if sub.is_dir():
                entries.extend(build_entries_for_folder(sub))
    else:
        entries.extend(build_entries_for_folder(images_dir))

    if not entries:
        print("❌ No entries found.", file=sys.stderr)
        sys.exit(1)

    output_json.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✅ {len(entries)} image entries written to {output_json}")


if __name__ == '__main__':
    main()
