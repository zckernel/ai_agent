import ollama

from tools.weather import get_weather


OLLAMA_BASE_URL = "http://host.docker.internal:11434"
OLLAMA_MODEL = "llama3.1:8b"

client = ollama.Client(host=OLLAMA_BASE_URL)


def ask_weather(city: str) -> str:
    weather_data = get_weather(city)

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"You are a weather assistant.\n\n"
                f"Weather data:\n{weather_data}\n\n"
                f"Answer naturally and briefly."
            ),
        }],
    )
    return response.message.content


if __name__ == "__main__":
    city = input("Location: ")
    print("\nAI Response:\n")
    print(ask_weather(city))
