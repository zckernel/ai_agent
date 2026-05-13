import ollama

from app.AIAgent.tools.weather import get_weather


client = ollama.Client(
    host='http://host.docker.internal:11434'
)

city = input("Location: ")

weather_data = get_weather(city)

prompt = f"""
You are a weather assistant.

Weather data:
{weather_data}

Answer user naturally and shortly.
"""

response = client.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ]
)

print("\nAI Response:\n")
print(response["message"]["content"])