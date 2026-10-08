# A-level 学习助手 RAG · 周末 MVP

带检索增强（RAG）的 A-level **纯数学 · 二次方程/二次函数** 学习助手起步仓库。  
目标：学生提问 → 从已入库笔记检索 → 按固定结构回答（下一步提示 / 关键知识点 / 检查理解小问 + 出处）。

## 范围（用户问什么 → 答什么）

| 问 | 答 |
|----|----|
| 只围绕「二次方程 / 二次函数」的概念、解法、图像、卡点 | 检索笔记 → **简要回答 + 引用片段** → **下一步提示** → **关键知识点** → **检查理解的小问** |
| 其他科目/章节 | 明确告知当前 MVP 仅覆盖本单元 |

详见 [`SCOPE.md`](./SCOPE.md)。

## 架构一览

```
笔记 Markdown (data/sample)
        │
        ▼
   ingest.py  切块 → 本地 embedding (BAAI/bge-small-zh-v1.5)
        │
        ▼
   LanceDB 持久化 (lancedb/)
        │
        ▼
   ask.py  提问 → 检索 Top-K → OpenAI 兼容 LLM（或模板回退）
              固定结构：回答+引用 / 下一步 / 相关知识点 / 小问
```

- **向量库**：LanceDB（本地目录；在 Python 3.13 上比 Chroma 更易安装）
- **Embedding**：`sentence-transformers` + `BAAI/bge-small-zh-v1.5`（完全本地）
- **编排**：纯 Python（未引入 LangChain/LlamaIndex，减少依赖）
- **UI**：周末后可接 Streamlit/Gradio；当前为 CLI

## 如何跑

```bash
# 克隆后进入仓库根目录（GitHub 仓库名一般为 alevel-rag-mvp）
cd alevel-rag-mvp

# 1. 安装依赖（首次会较慢：torch + 模型）
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 灌入示例笔记
python src/ingest.py

# 3. 提问
python src/ask.py "判别式大于零说明什么？"
# 或交互：
python src/ask.py
```

无需 `.env` 即可跑通（默认 `LLM_MODE=auto`，无密钥时自动用模板）。

### 接真实 LLM（可选）

依赖已含 `openai` + `python-dotenv`。复制环境变量模板后二选一：

```bash
cp .env.example .env
# 编辑 .env，不要把真实密钥提交进 git
```

**方式 A：OpenAI / 兼容网关**

```bash
# .env
LLM_MODE=auto
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1   # 可改成其他兼容地址
OPENAI_MODEL=gpt-4o-mini
```

```bash
python src/ask.py "判别式大于零说明什么？"
python src/ask.py --mode llm "顶点坐标怎么求？"   # 强制走 LLM
```

**方式 B：本地 Ollama（无需 API Key）**

```bash
ollama pull qwen2.5:7b
# .env
LLM_MODE=auto
OLLAMA_HOST=http://127.0.0.1:11434
OPENAI_MODEL=qwen2.5:7b
```

**仅模板（不调任何 API）**

```bash
python src/ask.py --mode template "判别式大于零说明什么？"
```

回答末尾 footer 会标明实际使用的模式（`llm` / `template` / `template (auto 回退)`）。

## 周六 / 周日检查清单（摘要）

**周六**
- [x] 定窄范围（见 SCOPE.md）
- [x] 最小栈：LanceDB + bge-small-zh + CLI
- [x] 灌库并跑通「提问 → 检索 → 带出处回答」
- [x] 确认回答里已有「下一步 / 相关知识点 / 小问」

**周日**
- [ ] 卡点分析：根据错误答案给出回退步骤（可在模板上加分支）
- [ ] 做一个知识点讲解卡片（文字即可）
- [ ] 自测 5 题，修切块/prompt
- [ ] 写下明天 3 个改进点（不扩功能）

## 先不要做（3 件）

1. 不要做多科目 / 全考纲覆盖  
2. 不要上复杂动画引擎或视频生成  
3. 不要自建账号体系、支付、多用户权限  

## 目录

```
alevel-rag-mvp/
├── README.md
├── SCOPE.md
├── requirements.txt
├── .env.example
├── data/sample/          # 示例中文笔记
├── src/ingest.py         # 切块入库
├── src/ask.py            # CLI 提问
└── lancedb/              # 本地向量库（ingest 后生成）
```
