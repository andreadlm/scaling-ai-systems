# 📬 Scaling AI Systems — Part II: Email Research Agent

> 🎓 Companion notebook for the **"Scaling AI Systems" Part II** *Show Me How* session.  
> Inspired by the [OpenPipe ART e-mail agent](https://openpipe.ai/blog/art-e-mail-agent) blog post.

## 🧭 What this is

A step-by-step, interactive [Marimo](https://marimo.io) notebook that walks through the full lifecycle of a production-grade LLM agent: from a working prototype, through instrumented observability, to a model that improves itself via reinforcement learning.

The agent answers natural-language questions about an email inbox by autonomously deciding which emails to search and read — a task that requires multi-step reasoning, not just retrieval.

The notebook is structured around three progressive stages, each building on the previous one:

```
Question → [ Agent (LangChain) ] → Answer
                  ↕ tool calls
         search_emails / read_email
                  ↓
         Langfuse (traces + scores)
                  ↓
         OpenPipe ART (RL training)
                  ↓
         updated model weights → Agent
```

### 🤖 Stage 1 · Agent

Define the environment (a SQLite database built from the [Enron email corpus](https://en.wikipedia.org/wiki/Enron_Corpus)), expose two tools to the agent (`search_emails`, `read_email`), and wire everything together with a LangChain agent and a dynamic system prompt that injects runtime context (inbox address, current date).

You can interact with the tools and the agent directly from the notebook UI.

### 📊 Stage 2 · Observability

Integrate [Langfuse](https://langfuse.com) via a single `CallbackHandler` to capture structured **traces** of every agent run — every LLM call, every tool invocation, timings, and token counts — without modifying the agent code.

Observability is a prerequisite for systematic improvement: you cannot fix what you cannot see.

### 🏋️ Stage 3 · Training

Close the loop with **reinforcement learning** using [OpenPipe ART](https://art.openpipe.ai).

Rather than a binary correct/wrong reward, the notebook implements **reward shaping**: a rubric that awards partial credit for good intermediate behaviour (locating the right email, reading it) and penalises inefficiency even when the final answer is correct. This gives the model a much denser training signal.

| Outcome | Base reward | Partial bonuses | Range |
|---|---|---|---|
| ✅ Correct answer | +1.0 | found · read · efficiency | [1.0, 1.3] |
| ❌ Wrong answer | −1.0 | found · read | [−1.0, −0.8] |
| 🚫 No answer / error | 0.0 | found · read | [0.0, 0.2] |

The training loop runs batches of parallel rollouts, scores each trajectory, and feeds the results to ART for policy gradient updates.

## ✅ Prerequisites

- 🐍 Python ≥ 3.12
- 🐳 [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Langfuse)
- 🔑 An OpenAI-compatible API key (or a local endpoint via `OPENAI_BASE_URL`)
- ⚡ [uv](https://docs.astral.sh/uv/) — fast Python package manager

## 🚀 Setup

**1. Install dependencies**

```bash
uv sync
```

This creates a `.venv` and installs all dependencies declared in `pyproject.toml` (including the `dev` group).

> 💡 To add a new dependency: `uv add <package>`. To add a dev-only one: `uv add --group dev <package>`.

**2. Configure environment variables**

For a standard local setup, `.env.example` already contains pre-configured values that work out of the box — just copy it:

```bash
cp .env.example .env
```

The only value you need to fill in is your API key:

```env
OPENAI_API_KEY=sk-...
```

> 💡 All other values (Langfuse URLs, credentials, project ID, model name) are pre-set in `.env.example` to match the Docker Compose stack and the notebook defaults.

**3. Launch the notebook**

```bash
uv run marimo edit .\notebook.py
```

The notebook will handle starting Langfuse (via `docker compose up`) and downloading the Enron dataset automatically when you reach the relevant cells.

## 🧹 Code quality

The project uses two tools from the [Astral](https://astral.sh) ecosystem for code quality, both configured in `pyproject.toml` / `ruff.toml`.

### Ruff — linter & formatter

[Ruff](https://docs.astral.sh/ruff/) is an extremely fast Python linter and formatter.

```bash
# Check for lint issues
uv run ruff check .

# Auto-fix what can be fixed automatically
uv run ruff check --fix .

# Format the codebase
uv run ruff format .
```

The project configuration (`ruff.toml`) silences rule `F541` (f-strings without placeholders), which Marimo notebooks occasionally generate.

### ty — static type checker

[ty](https://github.com/astral-sh/ty) is Astral's next-generation Python type checker, configured to use the project's virtual environment.

```bash
# Run a full type check
uv run ty check
```

## 📁 Project structure

```
notebook.py          # 📓 Main Marimo notebook — start here
agent/
  tools.py           # 🔧 search_emails and read_email LangChain tools
data/
  local_db.py        # 🗄️  SQLite database setup from the Enron corpus
  query_iterators.py # 📥 Loads the evaluation dataset from Hugging Face
  types.py           # 🏷️  Shared data types
docker-compose.yml   # 🐳 Langfuse stack (Postgres + web UI)
pyproject.toml       # 📦 Dependencies and tool configuration
ruff.toml            # 🧹 Ruff lint rules
```

## 🔑 Key technologies

| Layer | Tool |
|---|---|
| 📓 Notebook runtime | [Marimo](https://marimo.io) |
| 🤖 Agent framework | [LangChain](https://www.langchain.com) |
| 💬 LLM provider | OpenAI-compatible API |
| 📊 Observability | [Langfuse](https://langfuse.com) |
| 🏋️ RL training | [OpenPipe ART](https://art.openpipe.ai) |
| 📧 Email dataset | [Enron Corpus](https://en.wikipedia.org/wiki/Enron_Corpus) via Hugging Face |
| 🧪 Evaluation queries | [`corbt/enron_emails_sample_questions`](https://huggingface.co/datasets/corbt/enron_emails_sample_questions) |
| ⚡ Package manager | [uv](https://docs.astral.sh/uv/) |
| 🧹 Linter / formatter | [Ruff](https://docs.astral.sh/ruff/) |
| 🔍 Type checker | [ty](https://github.com/astral-sh/ty) |
