from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

def _tavily_search(query: str, max_results: int ) -> list[dict]:
    """
    Uses TavilySearchResults if installed and TAVILY_API_KEY is set.
    Returns list of dict with common fields. Note: published date is often missing.
    """
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})["results"]
    normalized: list[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "source": r.get("source") or "",
            }
        )
    return normalized

