from langchain.tools import tool

import requests


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    """

    url = f"https://wttr.in/{city}?format=3"

    response = requests.get(url, timeout=10)

    return response.text