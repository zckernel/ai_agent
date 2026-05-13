import requests
import xml.etree.ElementTree as ET
from langchain_core.tools import tool


@tool
def search_news(query: str) -> str:
    """Search Google News for recent news and current events on any topic. Use 'latest news' if no specific topic."""
    if not query or not query.strip():
        query = "latest news"
    try:
        resp = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "uk", "gl": "UA", "ceid": "UA:uk"},
            timeout=10,
        )
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            return "Could not parse news feed. Try a more specific query."
        items = root.findall(".//item")[:5]
        if not items:
            return "No news found."
        results = []
        for item in items:
            title = item.findtext("title", "").strip()
            pub_date = item.findtext("pubDate", "").strip()
            results.append(f"• {title} ({pub_date})")
        return "\n".join(results)
    except Exception as e:
        return f"Error: {e}"


HEADERS = {"User-Agent": "AIAgent/1.0 (educational project; contact@example.com)"}


def _wiki_summary(lang: str, query: str) -> str | None:
    search_resp = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        headers=HEADERS,
        timeout=10,
    )
    results = search_resp.json().get("query", {}).get("search", [])
    if not results:
        return None
    title = results[0]["title"]
    from urllib.parse import quote
    summary_resp = requests.get(
        f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
        headers=HEADERS,
        timeout=10,
    )
    try:
        data = summary_resp.json()
    except Exception:
        return None
    extract = data.get("extract")
    return f"{title}: {extract}" if extract else None


@tool
def search_wiki(query: str) -> str:
    """Search Wikipedia for facts, definitions, and background information on any topic."""
    for lang in ("uk", "en"):
        try:
            result = _wiki_summary(lang, query)
            if result:
                return result
        except Exception:
            continue
    return "No Wikipedia results found."
