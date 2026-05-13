import requests


OLLAMA_URL = "http://host.docker.internal:11434/api/generate"


def ask_llm(prompt: str, model: str = "qwen2.5:7b") -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]