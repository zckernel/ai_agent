import requests


OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"


def ask_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]
