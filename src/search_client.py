"""Search Client - Use Serper API for Real Search"""
import os
import requests
from typing import Dict, List, Any
from loguru import logger


class SerperSearchClient:
    """Serper Search Client"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Serper search client
        
        Args:
            api_key: Serper API key
        """
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("Serper API key required. Please set SERPER_API_KEY environment variable.")
        
        self.base_url = "https://google.serper.dev/search"
        logger.info("Serper search client initialized successfully")
    
    def search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Execute search query
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Search results dictionary
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results,
        }
        
        try:
            logger.info(f"🔍 SerperSearch: {query}")
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"SearchReturn {len(data.get('organic', []))} entriesResult")
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"SearchTimeout: {query}")
            return {"error": "SearchTimeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"SearchFailed: {e}")
            return {"error": str(e)}
    
    def format_results(self, search_data: Dict[str, Any]) -> str:
        """
        FormatSearchResultasText
        
        Args:
            search_data: SearchResultData
            
        Returns:
            FormatText
        """
        if "error" in search_data:
            return f"SearchError: {search_data['error']}"
        
        organic_results = search_data.get("organic", [])
        if not organic_results:
            return "Not found relevantResult"
        
        formatted = []
        for i, result in enumerate(organic_results[:5], 1):  
            title = result.get("title", "NoneTitle")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            date = result.get("date", "")
            
            formatted.append(f"{i}. {title}")
            if date:
                formatted.append(f"   Date: {date}")
            formatted.append(f"   {snippet}")
            formatted.append(f"   Link: {link}")
            formatted.append("")
        
        knowledge_graph = search_data.get("knowledgeGraph")
        if knowledge_graph:
            formatted.insert(0, "===  recognizeGraph ===")
            formatted.insert(1, f"Title: {knowledge_graph.get('title', '')}")
            formatted.insert(2, f"Description: {knowledge_graph.get('description', '')}")
            formatted.insert(3, "")
        
        return "\n".join(formatted)


class OpenDeepSearchClient:
    """OpenDeepSearchClient（If installed）"""
    
    def __init__(self):
        """InitializeOpenDeepSearchClient"""
        try:
            from opendeepsearch import OpenDeepSearchTool
            
            self.search_tool = OpenDeepSearchTool(
                model_name="openrouter/google/gemini-2.0-flash-001",
                reranker="jina",
                search_provider="serper",
            )
            
            if not self.search_tool.is_initialized:
                self.search_tool.setup()
            
            self.available = True
            logger.info("OpenDeepSearchClientInitializeSuccess")
            
        except ImportError:
            self.available = False
            logger.warning("OpenDeepSearchNot installed，Will use simpleSerperSearch")
    
    def search(self, query: str) -> str:
        """
        UseOpenDeepSearchExecuteSearch
        
        Args:
            query: SearchQuery
            
        Returns:
            SearchResultText
        """
        if not self.available:
            raise RuntimeError("OpenDeepSearch not available")
        
        try:
            logger.info(f"🔍 OpenDeepSearch: {query}")
            result = self.search_tool.forward(query)
            logger.debug(f"SearchResultLength: {len(result)} Character")
            return result
        except Exception as e:
            logger.error(f"OpenDeepSearchSearchFailed: {e}")
            return f"SearchFailed: {str(e)}"


def create_search_client(use_opendeepsearch: bool = False):
    """
    CreateSearchClient
    
    Args:
        use_opendeepsearch: Whether to try usingOpenDeepSearch
        
    Returns:
        SearchClientInstance
    """
    if use_opendeepsearch:
        try:
            client = OpenDeepSearchClient()
            if client.available:
                logger.info("Use OpenDeepSearch（Advanced mode）")
                return client
        except Exception as e:
            logger.warning(f"OpenDeepSearchInitializeFailed: {e}")
    
    logger.info("Use Serper API（Standard mode）")
    return SerperSearchClient()

