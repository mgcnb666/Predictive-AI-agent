"""General Prediction AI Agent - Supports Multi-Domain Prediction"""
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from .openai_client import create_openrouter_client
from .search_client import create_search_client
from .domains.sports import SportsPredictionDomain
from .domains.weather import WeatherPredictionDomain
from .domains.election import ElectionPredictionDomain
from .domains.general import GeneralPredictionDomain


class UniversalPredictionAgent:
    """General Prediction AI Agent - Supports Multi-Domain Prediction"""
    
    DOMAINS = {
        "sports": SportsPredictionDomain,
        "weather": WeatherPredictionDomain,
        "election": ElectionPredictionDomain,
        "general": GeneralPredictionDomain,  
    }
    
    def __init__(
        self,
        model_name: str = "google/gemini-2.0-flash-001",
        use_opendeepsearch: bool = False,
    ):
        """
        Initializegeneral predictionAgent
        
        Args:
            model_name: LLM model name
            use_opendeepsearch: WhetherUseOpenDeepSearch（NeedInstall）
        """
        logger.info("InitializeGeneral predictionAI Agent...")
        
        self.client = create_openrouter_client(model=model_name)
        
        try:
            self.search_client = create_search_client(use_opendeepsearch=use_opendeepsearch)
            logger.info("✅ SearchClientInitializeSuccess -  UseSerper/Jina.aiPerform realSearch")
        except Exception as e:
            logger.error(f"SearchClientInitializeFailed: {e}")
            self.search_client = None
        
        logger.info(f"SupportedPrediction domain: {list(self.DOMAINS.keys())}")
    
    def predict(
        self,
        domain: str,
        params: Dict[str, Any],
        use_search: bool = True
    ) -> Dict[str, Any]:
        """
        Execute prediction
        
        Args:
            domain: Prediction domain (sports/weather/electionetc)
            params: Prediction parameters
            use_search: WhetherUseSearchGetData
        
        Returns:
            Prediction Result
        """
        if domain not in self.DOMAINS:
            raise ValueError(f"Unsupported domain: {domain}。Support: {list(self.DOMAINS.keys())}")
        
        logger.info(f"Start {domain} domain prediction")
        logger.info(f"Parameters: {params}")
        
        domain_class = self.DOMAINS[domain]
        
        data = {}
        if use_search:
            data = self._collect_data(domain_class, params)
        
        prediction = self._generate_prediction(domain_class, data, params)
        
        result = domain_class.format_prediction(prediction)
        result["timestamp"] = datetime.now().isoformat()
        result["parameters"] = params
        
        logger.info(f"{domain} PredictionComplete")
        
        return result
    
    def _collect_data(
        self,
        domain_class,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ SetData - UseRealSearchAPI"""
        logger.info("StartDatacollectSet...")
        
        queries = domain_class.get_search_queries(params)
        logger.info(f"Generate  {len(queries)}  SearchQuery")
        
        search_results = {}
        
        if self.search_client is None:
            logger.warning("SearchClient Initialize，UseLLM recognizeLibrary（May not be new enough）")
            for query in queries:
                try:
                    response = self.client.simple_query(
                        f"extractProvide aboutwithDownInnercontentLatestInformationandData：{query}",
                        temperature=0.3
                    )
                    search_results[query] = response
                    logger.debug(f"LLMQueryComplete: {query}")
                except Exception as e:
                    logger.error(f"QueryFailed {query}: {e}")
                    search_results[query] = f"QueryFailed: {str(e)}"
        else:
            logger.info("🔍 Use Serper API Perform realSearch...")
            for i, query in enumerate(queries, 1):
                try:
                    logger.info(f"[{i}/{len(queries)}] Search: {query}")
                    
                    if hasattr(self.search_client, 'search') and hasattr(self.search_client, 'format_results'):
                        search_data = self.search_client.search(query)
                        result_text = self.search_client.format_results(search_data)
                    else:
                        result_text = self.search_client.search(query)
                    
                    search_results[query] = result_text
                    logger.debug(f"✅ QueryComplete: {query}")
                    
                except Exception as e:
                    logger.error(f"❌ SearchFailed {query}: {e}")
                    search_results[query] = f"SearchFailed: {str(e)}"
        
        logger.info(f"DatacollectSetComplete，Total receivedSet {len(search_results)} entriesResult")
        return search_results
    
    def _generate_prediction(
        self,
        domain_class,
        data: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """GeneratePrediction"""
        logger.info("GeneratePrediction...")
        
        system_prompt = domain_class.get_system_prompt()
        
        if hasattr(domain_class, 'get_prediction_prompt'):
            prediction_prompt = domain_class.get_prediction_prompt(data, params)
        else:
            prediction_prompt = f"""
Based onDownDataPerform prediction analysis：

Parameters：{json.dumps(params, ensure_ascii=False)}

Data：{json.dumps(data, ensure_ascii=False)}

Please provide detailed prediction analysis，withJSONFormatReturnResult。
"""
        
        try:
            response = self.client.simple_query(
                query=prediction_prompt,
                system_prompt=system_prompt,
                temperature=0.5
            )
            
            logger.debug(f"LLMResponse: {response[:200]}...")
            
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    prediction = json.loads(json_match.group())
                    logger.info(f"SuccessParsePrediction Result")
                except json.JSONDecodeError as e:
                    logger.error(f"JSONParseFailed: {e}")
                    logger.debug(f"OriginalResponse: {response}")
                    prediction = {
                        "analysis": response,
                        "confidence": 0.5
                    }
            else:
                logger.warning("Responsenot found intoJSONFormat")
                logger.debug(f"CompleteResponse: {response}")
                prediction = {
                    "analysis": response,
                    "confidence": 0.5
                }
            
            return prediction
            
        except Exception as e:
            logger.error(f"PredictionGenerateFailed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "confidence": 0.0
            }
    
    def list_domains(self) -> List[str]:
        """List supported domains"""
        return list(self.DOMAINS.keys())
    
    def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """GetDomainInformation"""
        if domain not in self.DOMAINS:
            return {"error": "Unsupported domain"}
        
        domain_class = self.DOMAINS[domain]
        return {
            "name": domain,
            "class": domain_class.__name__,
            "system_prompt": domain_class.get_system_prompt(),
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = UniversalPredictionAgent()
    
    print("\n" + "="*60)
    print("Testing1: Sports Prediction")
    print("="*60)
    result = agent.predict(
        domain="sports",
        params={
            "team1": "Barcelona",
            "team2": "Real Madrid",
            "league": "La Liga"
        },
        use_search=False  
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("Testing2: weatherPrediction")
    print("="*60)
    result = agent.predict(
        domain="weather",
        params={
            "location": "Beijing",
            "date": "2025-10-20",
            "days_ahead": 7
        },
        use_search=False
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("Testing3: Election Prediction")
    print("="*60)
    result = agent.predict(
        domain="election",
        params={
            "election": "2024 US Presidential Election",
            "region": "United States",
            "candidates": ["Candidate A", "Candidate B"]
        },
        use_search=False
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

