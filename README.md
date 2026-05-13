# MicroAI Chat

> Your private AI assistant. Runs locally. Knows the world.

MicroAI Chat is a lightweight conversational AI agent powered by [Ollama](https://ollama.com/) and [LangChain](https://www.langchain.com/). It thinks, searches, calculates, and answers — all on your machine, with no data leaving your device.

---

## Why MicroAI Chat?

**Privacy first.** No API keys. No cloud. Your conversations stay on your hardware.

**Actually useful.** Ask about the weather, do math, look up facts, read Wikipedia, catch the latest news — all in one chat window.

**Speaks your language.** Ukrainian, English, or whatever you type — MicroAI responds in kind.

---

## What it can do

| Skill | Example |
|-------|---------|
| Live weather | *"Яка погода у Львові?"* |
| Math | *"What is 15% of 4800?"* |
| Wikipedia lookup | *"Tell me about Nikola Tesla"* |
| Latest news | *"Give me today's top news"* |
| File analysis | *"Summarize tools/weather.py"* |
| General knowledge | *"Who founded Apple?"* |

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running on your machine
- Recommended model: `qwen2.5:14b`

> **Performance tip:** Run Ollama on the **host machine** (not inside Docker) so it can access your GPU directly via Metal (macOS) or CUDA (Linux/Windows). Models running inside a container lose GPU acceleration and fall back to CPU — significantly slower.
>
> `qwen2.5:14b` on an M2 Max with Metal runs ~40 tok/s. The same model on CPU: ~4 tok/s.

---

## Quick Start

```bash
pip install -r requirements-ai.txt
ollama pull qwen2.5:14b
python chat.py
```

```
AI Agent
Type /exit to quit

You: What's the weather in Kyiv?
AI: Kyiv is currently sunny, +18°C — a great day to be outside.

You: Latest news?
AI: • Ukraine and EU sign new cooperation agreement (Tue, 13 May 2026)
    • ...

You: /exit
Bye!
```

---

## Project Structure

```
AIAgent/
├── chat.py               # Interactive chat loop
├── main.py               # Standalone weather assistant
├── tools/
│   ├── weather.py        # Live weather via wttr.in
│   ├── calculator.py     # Safe math evaluator
│   ├── file_reader.py    # File summarizer
│   └── web_search.py     # Google News RSS + Wikipedia
└── requirements-ai.txt
```

---

## Configuration

```python
# chat.py
OLLAMA_MODEL    = "qwen2.5:14b"
OLLAMA_BASE_URL = "http://host.docker.internal:11434"
```

---

## Tools

| Tool | Source | API Key |
|------|--------|---------|
| Weather | wttr.in | None |
| Calculator | built-in | None |
| News search | Google News RSS | None |
| Wiki search | Wikipedia API | None |
| File reader | local filesystem | None |

Zero external dependencies. Zero cost.
