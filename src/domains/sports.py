"""body match Prediction domain"""
from typing import Dict, Any, List
from loguru import logger


class SportsPredictionDomain:
    """body match Prediction domain"""
    
    domain_name = "sports"
    
    @staticmethod
    def get_search_queries(params: Dict[str, Any]) -> List[str]:
        """GenerateSearchQuery"""
        team1 = params.get("team1")
        team2 = params.get("team2")
        league = params.get("league", "")
        
        return [
            f"{team1} vs {team2} head to head statistics",
            f"{team1} recent form {league}",
            f"{team2} recent form {league}",
            f"{team1} injury news",
            f"{team2} injury news",
            f"{team1} vs {team2} betting odds",
            f"{team1} vs {team2} expert predictions",
        ]
    
    @staticmethod
    def get_system_prompt() -> str:
        """GetSystem promptWord"""
        return """You are a professional sports event analyst。
AnalysisMatchWhen considering：
1. TeamRecentStatusandPerformance
2. Historical match record
3. Home/awayAdvantage
4. Injurysituation 
5. Tactical arrangement
6. PsychologyFactors
Please give objective、DataDriven prediction。"""
    
    @staticmethod
    def get_prediction_prompt(data: Dict[str, Any], params: Dict[str, Any]) -> str:
        """GeneratePrediction prompt"""
        team1 = params.get("team1", "Home team")
        team2 = params.get("team2", "Away team")
        league = params.get("league", "")
        
        return f"""
Based onDownDataPrediction {team1} vs {team2} ({league}) matchResult：

SearchData：
{data}

Please provide detailed prediction analysis，withJSONFormatReturn：

{{
  "home_win_prob": 0.0,
  "draw_prob": 0.0,
  "away_win_prob": 0.0,
  "confidence": 0.0,
  "analysis": "Detailed analysis...",
  "key_factors": ["Factors1", "Factors2"],
  "risks": ["Risk1", "Risk2"]
}}

Requirement：
1.   ProbabilitySum must equal1.0
2. home_win_probis{team1}WinProbability
3. away_win_probis{team2}WinProbability
4. Based onSearchDataObjective analysis
5. confidenceIndicatePredictionConfidence(0-1)
6. OnlyReturnJSON，No otherInnercontent
"""
    
    @staticmethod
    def format_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """FormatPrediction result"""
        return {
            "domain": "sports",
            "prediction_type": "match_outcome",
            "outcomes": {
                "home_win": prediction.get("home_win_prob", 0),
                "draw": prediction.get("draw_prob", 0),
                "away_win": prediction.get("away_win_prob", 0),
            },
            "confidence": prediction.get("confidence", 0),
            "analysis": prediction.get("analysis", ""),
            "key_factors": prediction.get("key_factors", []),
            "risks": prediction.get("risks", []),
        }

