#!/usr/bin/env python3
"""将 data/sample 下的 Markdown 切块并写入本地 LanceDB 向量库。"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path

import lancedb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sample"
DB_DIR = ROOT / "lancedb"
TABLE = "alevel_quadratic"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# 单块过长时再按字符切
MAX_CHARS = 500
OVERLAP = 60


def split_by_headings(text: str) -> list[str]:
    """按 ## / ### 标题切段，保留标题行，便于按知识点检索。"""
    parts = re.split(r"(?m)(?=^#{2,3} )", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_long(text: str, size: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def load_chunks(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    files = sorted(data_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"未找到笔记：{data_dir}/*.md")
    for path in files:
        raw = path.read_text(encoding="utf-8")
        sections = split_by_headings(raw)
        flat: list[str] = []
        for sec in sections:
            flat.extend(chunk_long(sec))
        for i, chunk in enumerate(flat):
            digest = hashlib.md5(f"{path.name}:{i}:{chunk[:40]}".encode()).hexdigest()[:12]
            rows.append(
                {
                    "id": f"{path.stem}_{i}_{digest}",
                    "source": path.name,
                    "chunk_index": i,
                    "text": chunk,
                }
            )
    return rows


def main() -> int:
    print(f"[ingest] 数据目录: {DATA_DIR}")
    print(f"[ingest] 向量库: {DB_DIR}")
    print(f"[ingest] Embedding: {EMBED_MODEL}")

    rows = load_chunks(DATA_DIR)
    print(f"[ingest] 切块数: {len(rows)}")
    print("[ingest] 加载 embedding 模型（首次会下载）…")
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([r["text"] for r in rows], normalize_embeddings=True)
    for r, vec in zip(rows, vectors):
        r["vector"] = vec.tolist()

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(DB_DIR))
    table = db.create_table(TABLE, data=rows, mode="overwrite")
    print(f"[ingest] 完成：表 `{TABLE}` 共 {table.count_rows()} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
