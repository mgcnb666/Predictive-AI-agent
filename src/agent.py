"""Main AI Agent - Integrate All Components"""
import json
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .data_collector import SportsDataCollector
from .feature_extractor import FeatureExtractor
from .prediction_engine import PredictionEngine, RiskManager


class PredictionAgent:
    """PredictionAI Agent - Complete prediction process"""
    
    def __init__(
        self,
        model_name: str = "openrouter/google/gemini-2.0-flash-001",
        search_provider: str = "serper",
        reranker: str = "jina",
        initial_bankroll: float = 10000,
        **kwargs
    ):
        """
        Initialize prediction agent
        
        Args:
            model_name: LLM model name
            search_provider: Search provider
            reranker: Reranker
            initial_bankroll: Initial bankroll
            **kwargs: Other configuration parameters
        """
        logger.info("Initializing prediction AI agent...")
        
        self.data_collector = SportsDataCollector(
            model_name=model_name,
            search_provider=search_provider,
            reranker=reranker,
            **kwargs
        )
        
        self.feature_extractor = FeatureExtractor(
            model_name=model_name
        )
        
        self.prediction_engine = PredictionEngine(
            model_name=model_name,
            **kwargs
        )
        
        self.risk_manager = RiskManager(
            initial_bankroll=initial_bankroll
        )
        
        logger.info("PredictionAI AgentInitializeComplete")
    
    def analyze_match(
        self,
        team1: str,
        team2: str,
        league: str,
        date: Optional[str] = None,
        market_odds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Complete match analysis process
        
        Args:
            team1: Home team
            team2: Away team
            league: League
            date: MatchDate
            market_odds: Market odds {"home": 2.0, "draw": 3.5, "away": 2.5}
        
        Returns:
            Complete analysisReport
        """
        logger.info(f"StartAnalysisMatch: {team1} vs {team2}")
        
        start_time = datetime.now()
        
        try:
            logger.info("Step1: DatacollectSet...")
            match_data = self.data_collector.get_match_data(
                team1=team1,
                team2=team2,
                league=league,
                date=date
            )
            
            sentiment_data = self.data_collector.get_market_sentiment(
                team1=team1,
                team2=team2,
                date=date
            )
            
            if market_odds is None:
                odds_data = self.data_collector.get_live_odds(
                    team1=team1,
                    team2=team2
                )
            else:
                odds_data = {"odds_data": market_odds}
            
            logger.info("Step2: Feature extraction...")
            features = self.feature_extractor.extract_all_features(
                match_data=match_data,
                odds_data=odds_data
            )
            
            logger.info("Step3: ExecutePrediction...")
            prediction = self.prediction_engine.predict(
                features=features,
                team1_name=team1,
                team2_name=team2
            )
            
            logger.info("Step4: Expected priceValueAnalysis...")
            
            if market_odds:
                ev_analysis = self.prediction_engine.calculate_expected_value(
                    prediction=prediction,
                    market_odds=market_odds
                )
            else:
                try:
                    extracted_odds = self._extract_odds_from_data(odds_data)
                    ev_analysis = self.prediction_engine.calculate_expected_value(
                        prediction=prediction,
                        market_odds=extracted_odds
                    )
                except Exception as e:
                    logger.warning(f"NoneUnable to proceedEVAnalysis: {e}")
                    ev_analysis = {"error": "Odds data not available"}
            
            if ev_analysis.get("should_bet", False):
                bet_size_pct = ev_analysis["best_bet"]["bet_size_percentage"]
                confidence = prediction["confidence"]
                
                bet_amount = self.risk_manager.calculate_bet_amount(
                    bet_size_percentage=bet_size_pct,
                    confidence=confidence
                )
                
                ev_analysis["recommended_bet_amount"] = bet_amount
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            report = {
                "match_info": {
                    "team1": team1,
                    "team2": team2,
                    "league": league,
                    "date": date or "Unspecified"
                },
                "prediction": {
                    "home_win_probability": prediction["home_win_prob"],
                    "draw_probability": prediction["draw_prob"],
                    "away_win_probability": prediction["away_win_prob"],
                    "confidence": prediction["confidence"],
                    "expected_score": prediction.get("expected_score", "Not predicted")
                },
                "analysis": {
                    "summary": prediction.get("analysis", ""),
                    "key_factors": prediction.get("key_factors", []),
                    "risks": prediction.get("risks", [])
                },
                "betting_analysis": ev_analysis,
                "features_summary": {
                    "form_differential": features.get("derived_features", {}).get("form_differential", 0),
                    "home_advantage": features.get("derived_features", {}).get("home_advantage", 0),
                    "overall_score": features.get("derived_features", {}).get("overall_advantage_score", 0)
                },
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "elapsed_time_seconds": elapsed_time,
                    "data_quality": self._assess_data_quality(match_data)
                }
            }
            
            logger.info(f"AnalysisComplete，Time-consuming {elapsed_time:.2f}  ")
            
            return report
            
        except Exception as e:
            logger.error(f"AnalysisFailed: {e}", exc_info=True)
            return {
                "error": str(e),
                "match_info": {
                    "team1": team1,
                    "team2": team2,
                    "league": league
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def quick_predict(
        self,
        team1: str,
        team2: str,
        league: str = "Unspecified",
        market_odds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Quick prediction（Simplified version）
        
        Args:
            team1: Home team
            team2: Away team
            league: League
            market_odds: Market odds
        
        Returns:
            SimplifyPrediction Result
        """
        logger.info(f"Quick prediction: {team1} vs {team2}")
        
        full_report = self.analyze_match(
            team1=team1,
            team2=team2,
            league=league,
            market_odds=market_odds
        )
        
        if "error" in full_report:
            return full_report
        
        return {
            "match": f"{team1} vs {team2}",
            "prediction": full_report["prediction"],
            "recommendation": full_report["betting_analysis"].get("recommendation", "NoneRecommendation"),
            "confidence": full_report["prediction"]["confidence"],
            "key_insight": full_report["analysis"]["summary"][:200] + "..."
        }
    
    def batch_analyze(
        self,
        matches: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Batch analysismultiplefieldMatch
        
        Args:
            matches: MatchList，EveryPackage  team1, team2, leagueetc
        
        Returns:
            Analysis resultList
        """
        logger.info(f"Batch analysis {len(matches)} fieldMatch")
        
        results = []
        for match in matches:
            try:
                result = self.analyze_match(
                    team1=match["team1"],
                    team2=match["team2"],
                    league=match.get("league", "Unspecified"),
                    date=match.get("date"),
                    market_odds=match.get("market_odds")
                )
                results.append(result)
            except Exception as e:
                logger.error(f"AnalysisFailed: {match}, Error: {e}")
                results.append({
                    "error": str(e),
                    "match": match
                })
        
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """GetAgentStatus"""
        return {
            "status": "active",
            "bankroll": {
                "current": self.risk_manager.current_bankroll,
                "initial": self.risk_manager.initial_bankroll,
                "daily_pnl": self.risk_manager.daily_pnl
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_odds_from_data(self, odds_data: Dict[str, Any]) -> Dict[str, float]:
        """fromOddsDataExtract standard format from"""
        if "odds_data" in odds_data and isinstance(odds_data["odds_data"], dict):
            return odds_data["odds_data"]
        
        return {"home": 2.0, "draw": 3.5, "away": 2.5}
    
    def _assess_data_quality(self, match_data: Dict[str, Any]) -> str:
        """EvaluationData Quality"""
        key_fields = ["head_to_head", "team1_form", "team2_form", "betting_odds"]
        
        available = sum(
            1 for field in key_fields 
            if field in match_data and "error" not in match_data[field]
        )
        
        quality_ratio = available / len(key_fields)
        
        if quality_ratio >= 0.8:
            return "high"
        elif quality_ratio >= 0.5:
            return "medium"
        else:
            return "low"


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    agent = PredictionAgent(
        search_provider=os.getenv("SEARCH_PROVIDER", "serper"),
        reranker=os.getenv("RERANKER", "jina")
    )
    
    result = agent.analyze_match(
        team1="Manchester United",
        team2="Liverpool",
        league="Premier League",
        market_odds={"home": 2.5, "draw": 3.4, "away": 2.8}
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

