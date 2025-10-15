"""Data Collection Module - Use OpenDeepSearch for Deep Search"""
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

try:
    from opendeepsearch import OpenDeepSearchTool
except ImportError:
    logger.warning("OpenDeepSearchNot installed， Run: pip install opendeepsearch")
    OpenDeepSearchTool = None


class SportsDataCollector:
    """body DatacollectSeter"""
    
    def __init__(
        self,
        model_name: str = "openrouter/google/gemini-2.0-flash-001",
        reranker: str = "jina",
        search_provider: str = "serper",
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None,
        use_pro_mode: bool = True
    ):
        """
        InitializeDatacollectSeter
        
        Args:
            model_name: LLM model name
            reranker: Reranker (jina/infinity)
            search_provider: Search provider (serper/searxng)
            searxng_instance_url: SearXNGInstanceURL
            searxng_api_key: SearXNG API key
            use_pro_mode: WhetherUseProMode（DepthSearch）
        """
        if OpenDeepSearchTool is None:
            raise ImportError("Please install firstOpenDeepSearch: pip install opendeepsearch")
        
        self.use_pro_mode = use_pro_mode
        
        search_config = {
            "model_name": model_name,
            "reranker": reranker,
            "search_provider": search_provider,
        }
        
        if search_provider == "searxng":
            if searxng_instance_url:
                search_config["searxng_instance_url"] = searxng_instance_url
            if searxng_api_key:
                search_config["searxng_api_key"] = searxng_api_key
        
        self.search_agent = OpenDeepSearchTool(**search_config)
        
        if not self.search_agent.is_initialized:
            self.search_agent.setup()
        
        logger.info(f"DatacollectSeterInitializeComplete - Model: {model_name}, Search: {search_provider}")
    
    def search(self, query: str) -> str:
        """ExecuteSearchQuery"""
        try:
            logger.info(f"SearchQuery: {query}")
            result = self.search_agent.forward(query)
            logger.debug(f"SearchResultLength: {len(result)} Character")
            return result
        except Exception as e:
            logger.error(f"SearchFailed: {e}")
            return f"SearchFailed: {str(e)}"
    
    def get_match_data(
        self, 
        team1: str, 
        team2: str, 
        league: str,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        GetMatchRelatedData
        
        Args:
            team1: Home teamName
            team2: Away teamName
            league: LeagueName
            date: MatchDate（Optional）
        
        Returns:
            PackageIncludingClassMatchDataDictionary
        """
        logger.info(f"GetMatchData: {team1} vs {team2} ({league})")
        
        date_str = f"on {date}" if date else "upcoming"
        
        queries = {
            "head_to_head": f"{team1} vs {team2} head to head statistics last 5 matches history",
            "team1_form": f"{team1} recent form last 10 matches {league} results standings",
            "team2_form": f"{team2} recent form last 10 matches {league} results standings",
            "team1_injuries": f"{team1} injury news lineup squad {date_str}",
            "team2_injuries": f"{team2} injury news lineup squad {date_str}",
            "betting_odds": f"{team1} vs {team2} {date_str} betting odds comparison bookmakers",
            "expert_predictions": f"{team1} vs {team2} {date_str} expert predictions analysis",
            "match_preview": f"{team1} vs {team2} {league} {date_str} match preview tactical analysis",
        }
        
        results = {}
        for key, query in queries.items():
            try:
                result = self.search(query)
                results[key] = {
                    "query": query,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Query {key} Failed: {e}")
                results[key] = {
                    "query": query,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return results
    
    def get_team_statistics(self, team_name: str, league: str) -> Dict[str, Any]:
        """
        GetTeamStatisticsData
        
        Args:
            team_name: TeamName
            league: LeagueName
        
        Returns:
            TeamStatisticsData
        """
        logger.info(f"GetTeamStatistics: {team_name} ({league})")
        
        queries = {
            "season_stats": f"{team_name} {league} season statistics goals scored conceded",
            "home_away": f"{team_name} home away record {league} current season",
            "key_players": f"{team_name} key players top scorers assists {league}",
            "recent_results": f"{team_name} last 10 matches results {league}",
        }
        
        results = {}
        for key, query in queries.items():
            try:
                result = self.search(query)
                results[key] = result
            except Exception as e:
                logger.error(f"Query {key} Failed: {e}")
                results[key] = f"QueryFailed: {str(e)}"
        
        return results
    
    def get_market_sentiment(
        self, 
        team1: str, 
        team2: str,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get fieldsituation andBettingTrend
        
        Args:
            team1: Home team
            team2: Away team
            date: MatchDate
        
        Returns:
             fieldsituation Data
        """
        logger.info(f"Get fieldsituation : {team1} vs {team2}")
        
        date_str = f"on {date}" if date else ""
        
        queries = {
            "betting_trends": f"{team1} vs {team2} {date_str} betting trends public money",
            "odds_movement": f"{team1} vs {team2} {date_str} odds movement line movement",
            "social_sentiment": f"{team1} vs {team2} {date_str} Twitter sentiment fan predictions",
            "sharp_money": f"{team1} vs {team2} {date_str} sharp money professional bettors",
        }
        
        results = {}
        for key, query in queries.items():
            try:
                result = self.search(query)
                results[key] = result
            except Exception as e:
                logger.error(f"Query {key} Failed: {e}")
                results[key] = f"QueryFailed: {str(e)}"
        
        return results
    
    def get_live_odds(
        self, 
        team1: str, 
        team2: str,
        bet_type: str = "1x2"
    ) -> Dict[str, Any]:
        """
        GetReal-timeOdds
        
        Args:
            team1: Home team
            team2: Away team
            bet_type: BettingType (1x2, asian_handicap, over_underetc)
        
        Returns:
            Real-timeOddsData
        """
        logger.info(f"GetReal-timeOdds: {team1} vs {team2} ({bet_type})")
        
        query = f"{team1} vs {team2} live odds {bet_type} bookmakers comparison best odds"
        
        try:
            result = self.search(query)
            return {
                "match": f"{team1} vs {team2}",
                "bet_type": bet_type,
                "odds_data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"GetOddsFailed: {e}")
            return {
                "match": f"{team1} vs {team2}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_historical_odds(
        self,
        team1: str,
        team2: str,
        lookback_matches: int = 5
    ) -> Dict[str, Any]:
        """
        GetHistoricalVersusOddsData
        
        Args:
            team1: Home team
            team2: Away team
            lookback_matches: BacktrackMatchfield 
        
        Returns:
            HistoricalOddsData
        """
        logger.info(f"GetHistoricalOdds: {team1} vs {team2} (Recent{lookback_matches}field)")
        
        query = f"{team1} vs {team2} historical betting odds last {lookback_matches} matches closing odds results"
        
        try:
            result = self.search(query)
            return {
                "match": f"{team1} vs {team2}",
                "lookback": lookback_matches,
                "historical_data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"GetHistoricalOddsFailed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_weather_conditions(
        self,
        stadium: str,
        date: str
    ) -> Dict[str, Any]:
        """
        GetMatchweathersituation 
        
        Args:
            stadium:  fieldNameorCity
            date: MatchDate
        
        Returns:
            weatherData
        """
        logger.info(f"GetweatherInformation: {stadium} on {date}")
        
        query = f"weather forecast {stadium} {date} temperature wind rain conditions"
        
        try:
            result = self.search(query)
            return {
                "stadium": stadium,
                "date": date,
                "weather": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"GetweatherFailed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class CryptoDataCollector:
    """Cryptocurrency data collector (can be extended to other prediction markets)"""
    
    def __init__(
        self,
        model_name: str = "openrouter/google/gemini-2.0-flash-001",
        reranker: str = "jina",
        search_provider: str = "serper"
    ):
        """InitializeEncryptionCurrencyDatacollectSeter"""
        if OpenDeepSearchTool is None:
            raise ImportError("Please install firstOpenDeepSearch: pip install opendeepsearch")
        
        self.search_agent = OpenDeepSearchTool(
            model_name=model_name,
            reranker=reranker,
            search_provider=search_provider
        )
        
        if not self.search_agent.is_initialized:
            self.search_agent.setup()
        
        logger.info(f"EncryptionCurrencyDatacollectSeterInitializeComplete")
    
    def get_token_data(self, token_symbol: str) -> Dict[str, Any]:
        """GetTokenData"""
        logger.info(f"GetTokenData: {token_symbol}")
        
        queries = {
            "price_analysis": f"{token_symbol} price prediction technical analysis",
            "on_chain": f"{token_symbol} on-chain metrics whale activity",
            "news": f"{token_symbol} latest news developments announcements",
            "social": f"{token_symbol} social sentiment Twitter community",
        }
        
        results = {}
        for key, query in queries.items():
            try:
                result = self.search_agent.forward(query)
                results[key] = result
            except Exception as e:
                logger.error(f"Query {key} Failed: {e}")
                results[key] = f"QueryFailed: {str(e)}"
        
        return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    collector = SportsDataCollector(
        search_provider=os.getenv("SEARCH_PROVIDER", "serper"),
        reranker=os.getenv("RERANKER", "jina")
    )
    
    match_data = collector.get_match_data(
        team1="Manchester United",
        team2="Liverpool",
        league="Premier League"
    )
    
    print(json.dumps(match_data, indent=2, ensure_ascii=False))

