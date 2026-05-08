"""
AgentDataset Discovery Agent
Search & Fetch Research Documents
"""

import os
from typing import List
from duckduckgo_search import DDGS
import trafilatura
from agentdataset.models.schemas import DiscoveryResult

class DiscoveryAgent:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[DiscoveryResult]:
        """Search web for relevant documents."""
        results = []
        with DDGS() as ddgs:
            # Search for PDFs specifically
            pdf_query = f"{query} filetype:pdf"
            for r in ddgs.text(pdf_query, max_results=self.max_results):
                results.append(DiscoveryResult(
                    title=r['title'],
                    url=r['href'],
                    source_type="pdf",
                    relevance_score=1.0, # Placeholder
                    snippet=r['body']
                ))
            
            # General web search for HTML
            for r in ddgs.text(query, max_results=self.max_results):
                if not r['href'].endswith(".pdf"):
                    results.append(DiscoveryResult(
                        title=r['title'],
                        url=r['href'],
                        source_type="html",
                        relevance_score=0.8, # Placeholder
                        snippet=r['body']
                    ))
        return results

    def fetch_content(self, result: DiscoveryResult) -> str:
        """Fetch and convert content to Markdown."""
        if result.source_type == "html":
            downloaded = trafilatura.fetch_url(result.url)
            if downloaded:
                return trafilatura.extract(downloaded)
        elif result.source_type == "pdf":
            # For now, we just return the URL for the PDF processor to handle
            return result.url
        return ""
