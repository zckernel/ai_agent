from langchain.tools import tool

import requests


@tool
def get_weather(city: str) -> str:
    """
    Get current weather and 2-day forecast for a city.
    Returns today's conditions and tomorrow's forecast.
    """
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url, timeout=10)
    data = response.json()

    days = data["weather"]
    today = days[0]
    tomorrow = days[1]

    def fmt_day(label: str, d: dict) -> str:
        desc = d["hourly"][4]["weatherDesc"][0]["value"]
        max_c = d["maxtempC"]
        min_c = d["mintempC"]
        return f"{label}: {desc}, {min_c}°C – {max_c}°C"

    current = data["current_condition"][0]
    current_desc = current["weatherDesc"][0]["value"]
    current_temp = current["temp_C"]

    return "\n".join([
        f"Now: {current_desc}, {current_temp}°C",
        fmt_day("Today", today),
        fmt_day("Tomorrow", tomorrow),
    ])