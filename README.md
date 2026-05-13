# AI Agent

A conversational AI agent built with [LangChain](https://www.langchain.com/) and [Ollama](https://ollama.com/), running locally with tool-augmented ReAct reasoning.

## Features

- ReAct (Reasoning + Acting) agent loop
- Local LLM via Ollama — no cloud required
- Built-in tools: weather lookup, calculator
- Multilingual — responds in the user's language

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (default: `http://host.docker.internal:11434`)
- Model pulled in Ollama (e.g. `llama3.1:8b` or `qwen2.5:7b`)

## Installation

```bash
pip install -r requirements-ai.txt
```

Pull a model:

```bash
ollama pull llama3.1:8b
```

## Usage

```bash
python chat.py
```

```
AI Agent
Type /exit to quit

You: What is the weather in Kyiv?
AI: Kyiv: ☀️ +18°C

You: /exit
Bye!
```

## Project Structure

```
AIAgent/
├── chat.py             # Entry point — interactive chat loop
├── main.py             # Alternative entry point
├── ollama_client.py    # Ollama LLM client setup
├── tools/
│   ├── weather.py      # Weather tool
│   └── calculator.py   # Calculator tool
├── agents/             # Agent definitions
├── prompts/            # Prompt templates
├── memory/             # Conversation memory
├── tests/              # Tests
└── requirements-ai.txt
```

## Configuration

Edit `chat.py` to change the model or Ollama URL:

```python
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://host.docker.internal:11434",
    temperature=0,
)
```

## Tools

| Tool        | Description                        |
|-------------|------------------------------------|
| `get_weather` | Returns current weather for a city |
| `calculator`  | Evaluates math expressions         |
