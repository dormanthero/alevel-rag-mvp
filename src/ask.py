#!/usr/bin/env python3
"""CLI：提问 → 检索 → LLM 或模板按固定结构输出。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lancedb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DB_DIR = ROOT / "lancedb"
TABLE = "alevel_quadratic"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 3

IN_SCOPE_HINTS = (
    "二次", "方程", "函数", "判别式", "求根", "配方", "因式", "顶点",
    "抛物线", "开口", "对称轴", "quadratic", "discriminant", "vertex",
    "factoris", "completing the square", "root",
)

SYSTEM_PROMPT = """你是 A-level 纯数学学习助手。当前 MVP **仅覆盖**「二次方程 / 二次函数」。

规则：
1. 只根据下方「笔记片段」回答；不要编造笔记里没有的公式或结论。
2. 若问题明显超出本单元，明确说：当前 MVP 仅覆盖二次方程/二次函数。
3. 输出必须使用以下 Markdown 小节（标题一字不差）：
## 回答
（简要说明 + 在文中用 [1][2]… 标注引用）
## 引用片段
（按 [1] 来源 `…` + 短摘录 列出你用到的笔记）
## 下一步提示
（2～4 条可执行下一步）
## 相关知识点
（3 条左右要点）
## 检查理解的小问
（1 道小题，便于自测）
4. 用简体中文回答（公式可保留英文/符号）。
"""

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    if not DB_DIR.exists():
        raise SystemExit("向量库未就绪。请先运行: python src/ingest.py")
    db = lancedb.connect(str(DB_DIR))
    try:
        table = db.open_table(TABLE)
    except Exception as e:
        raise SystemExit(f"打不开表 `{TABLE}`：{e}\n请先运行: python src/ingest.py") from e
    if table.count_rows() == 0:
        raise SystemExit("表为空，请先运行: python src/ingest.py")

    qvec = get_model().encode([query], normalize_embeddings=True)[0].tolist()
    hits = table.search(qvec).limit(min(top_k, table.count_rows())).to_list()
    return hits


def looks_in_scope(query: str) -> bool:
    q = query.lower()
    return any(h.lower() in q for h in IN_SCOPE_HINTS)


def _format_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        src = hit.get("source", "?")
        text = hit.get("text", "").strip()
        parts.append(f"[{i}] 来源 `{src}`\n{text}")
    return "\n\n".join(parts) if parts else "（无检索结果）"


def _resolve_llm_client() -> tuple[object, str] | None:
    """返回 (OpenAI client, model) 或 None（表示无法调用）。"""
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()
    ollama_host = (os.getenv("OLLAMA_HOST") or "").strip().rstrip("/")

    if not api_key and ollama_host:
        api_key = "ollama"
        if not base_url:
            base_url = f"{ollama_host}/v1"
        if not model:
            model = "qwen2.5:7b"
    elif not api_key:
        return None

    if not base_url:
        base_url = "https://api.openai.com/v1"
    if not model:
        model = "gpt-4o-mini"

    try:
        from openai import OpenAI
    except ImportError:
        print("[ask] 未安装 openai，请 pip install openai", file=sys.stderr)
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def call_llm(query: str, hits: list[dict]) -> str | None:
    """调用 OpenAI 兼容 Chat Completions；失败或无密钥时返回 None。"""
    resolved = _resolve_llm_client()
    if resolved is None:
        return None
    client, model = resolved

    user_content = (
        f"用户问题：{query}\n\n"
        f"笔记片段：\n{_format_context(hits)}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception as e:
        print(f"[ask] LLM 调用失败，将回退模板：{e}", file=sys.stderr)
        return None


def build_answer(query: str, hits: list[dict]) -> str:
    """用检索片段 + 模板拼出固定结构（无 LLM 时的回退）。"""
    if not hits:
        return (
            "未检索到相关笔记。请确认已运行 ingest，或换一个本单元内的问法。\n"
            "范围：A-level 纯数学 · 二次方程/二次函数。"
        )

    primary = hits[0]["text"].replace("\n", " ").strip()
    if len(primary) > 280:
        primary = primary[:280] + "…"

    cites = []
    for i, hit in enumerate(hits, 1):
        src = hit.get("source", "?")
        snippet = hit["text"].replace("\n", " ").strip()
        if len(snippet) > 160:
            snippet = snippet[:160] + "…"
        dist = hit.get("_distance")
        dist_s = f"，距离={dist:.4f}" if isinstance(dist, (int, float)) else ""
        cites.append(f"  [{i}] 来源 `{src}`{dist_s}\n      「{snippet}」")

    scope_note = ""
    if not looks_in_scope(query):
        scope_note = (
            "\n> 提示：当前 MVP **仅覆盖**二次方程/二次函数。"
            "若你的问题属于其他章节，请先改成本单元相关问法。\n"
        )

    return f"""## 回答
{scope_note}根据笔记，与「{query}」最相关的要点如下：

{primary}

## 引用片段
{chr(10).join(cites)}

## 下一步提示
1. 先确认题目是「求根」还是「看图像（顶点/开口/交点）」。
2. 求根：优先尝试因式分解 → 不行再用求根公式，并先算判别式 Δ。
3. 看图像：写出顶点式或用 x=-b/(2a) 找顶点，再标与轴交点。

## 相关知识点
- 标准形式 ax^2+bx+c=0 与二次函数 y=ax^2+bx+c
- 判别式 Δ 与实根个数
- 顶点 / 对称轴 / 开口方向

## 检查理解的小问
若 y = x^2 - 4x + 3：开口方向？顶点坐标？与 x 轴交点各是什么？
（答完可再问我：「帮我核对这道题」）
"""


def resolve_mode(cli_mode: str | None) -> str:
    """返回 auto | llm | template。"""
    if cli_mode:
        return cli_mode
    env = (os.getenv("LLM_MODE") or "auto").strip().lower()
    if env not in ("auto", "llm", "template"):
        print(f"[ask] 未知 LLM_MODE={env!r}，改用 auto", file=sys.stderr)
        return "auto"
    return env


def answer(query: str, hits: list[dict], mode: str) -> tuple[str, str]:
    """返回 (正文, 实际使用的模式标签)。"""
    if mode == "template":
        return build_answer(query, hits), "template"

    if mode == "llm":
        text = call_llm(query, hits)
        if text is None:
            raise SystemExit(
                "LLM 模式失败：请设置 OPENAI_API_KEY（或 OLLAMA_HOST），"
                "或改用 --mode auto / --mode template。"
            )
        return text, "llm"

    # auto
    text = call_llm(query, hits)
    if text is not None:
        return text, "llm"
    print("[ask] 无可用 LLM 密钥或调用失败，已回退模板模式。", file=sys.stderr)
    return build_answer(query, hits), "template (auto 回退)"


def main() -> int:
    parser = argparse.ArgumentParser(description="A-level 二次方程 RAG 提问 CLI")
    parser.add_argument("question", nargs="?", help="你的问题（中文或英文）")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="检索条数")
    parser.add_argument(
        "--mode",
        choices=("auto", "llm", "template"),
        default=None,
        help="覆盖环境变量 LLM_MODE：auto（默认）| llm | template",
    )
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("请输入问题: ").strip()
    if not question:
        print("问题不能为空", file=sys.stderr)
        return 1

    mode = resolve_mode(args.mode)
    print(f"[ask] 检索中…（模型 {EMBED_MODEL}，模式 {mode}）")
    hits = retrieve(question, top_k=args.top_k)
    body, used = answer(question, hits, mode)
    footer = f"\n---\n*模式：{used}。配置见 `.env.example` / `--mode`。*\n"
    print(body.rstrip() + footer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
