"""Feature extractionModule - fromSearchResultExtract structured features from"""
import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

try:
    from smolagents import CodeAgent, LiteLLMModel
except ImportError:
    logger.warning("SmolAgents not installed, some functions will not be available")
    CodeAgent = None
    LiteLLMModel = None


class FeatureExtractor:
    """Feature extractioner - UseLLMfromUnstructuredDataextract fromFeatures"""
    
    def __init__(
        self,
        model_name: str = "openrouter/google/gemini-2.0-flash-001",
        temperature: float = 0.1
    ):
        """
        InitializeFeature extractioner
        
        Args:
            model_name: LLM model name
            temperature: TemperatureParameters（ TemperatureEnsure stable output）
        """
        if LiteLLMModel is None:
            raise ImportError("Please install firstsmolagents: pip install smolagents")
        
        self.model = LiteLLMModel(model_name, temperature=temperature)
        self.agent = CodeAgent(tools=[], model=self.model)
        logger.info(f"Feature extractionerInitializeComplete - Model: {model_name}")
    
    def extract_match_features(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        fromMatchDataextract fromFeatures
        
        Args:
            match_data: OriginalMatchData
        
        Returns:
            Structured featuresDictionary
        """
        logger.info("Startextract MatchFeatures")
        
        prompt = f"""
AnalysiswithDownFootballMatchData，extractGet keyKeyFeatures withJSONFormatReturn。

OriginalData：
{json.dumps(match_data, indent=2, ensure_ascii=False)}

 extract withDownFeatures（ReturnStrictJSONFormat）：
{{
  "team1": {{
    "name": "Home teamName",
    "recent_form": {{
      "wins": 0,
      "draws": 0,
      "losses": 0,
      "win_rate": 0.0,
      "goals_scored_avg": 0.0,
      "goals_conceded_avg": 0.0
    }},
    "home_record": {{
      "wins": 0,
      "draws": 0,
      "losses": 0
    }},
    "injuries": {{
      "key_players_out": [],
      "severity": "low/medium/high"
    }},
    "league_position": 0,
    "form_trend": "improving/stable/declining"
  }},
  "team2": {{
    "name": "Away teamName",
    "recent_form": {{
      "wins": 0,
      "draws": 0,
      "losses": 0,
      "win_rate": 0.0,
      "goals_scored_avg": 0.0,
      "goals_conceded_avg": 0.0
    }},
    "away_record": {{
      "wins": 0,
      "draws": 0,
      "losses": 0
    }},
    "injuries": {{
      "key_players_out": [],
      "severity": "low/medium/high"
    }},
    "league_position": 0,
    "form_trend": "improving/stable/declining"
  }},
  "head_to_head": {{
    "total_matches": 0,
    "team1_wins": 0,
    "team2_wins": 0,
    "draws": 0,
    "avg_goals": 0.0
  }},
  "betting_odds": {{
    "team1_win": 0.0,
    "draw": 0.0,
    "team2_win": 0.0,
    "implied_prob_team1": 0.0,
    "implied_prob_draw": 0.0,
    "implied_prob_team2": 0.0
  }},
  "expert_consensus": {{
    "predicted_winner": "team1/team2/draw",
    "confidence": "low/medium/high"
  }},
  "external_factors": {{
    "weather": "good/poor",
    "stadium_advantage": "significant/moderate/none"
  }}
}}

OnlyReturnJSON，Do not add any explanation。IfSomeDataMissing，UseReasonableDefaultValue。
"""
        
        try:
            result = self.agent.run(prompt)
            if isinstance(result, str):
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    features = json.loads(json_match.group())
                else:
                    features = json.loads(result)
            else:
                features = result
            
            logger.info("Feature extractionSuccess")
            return features
        
        except Exception as e:
            logger.error(f"Feature extractionFailed: {e}")
            return self._get_default_features()
    
    def extract_odds_features(self, odds_data: Dict[str, Any]) -> Dict[str, float]:
        """
        fromOddsDataextract fromFeatures
        
        Args:
            odds_data: OddsData
        
        Returns:
            OddsFeatures
        """
        logger.info("extract OddsFeatures")
        
        prompt = f"""
fromwithDownOddsDataextract from KeyInformation，ReturnJSONFormat：

{json.dumps(odds_data, indent=2, ensure_ascii=False)}

ReturnFormat：
{{
  "home_odds": 0.0,
  "draw_odds": 0.0,
  "away_odds": 0.0,
  "home_implied_prob": 0.0,
  "draw_implied_prob": 0.0,
  "away_implied_prob": 0.0,
  "bookmaker_margin": 0.0,
  "odds_movement": "up/down/stable",
  "sharp_money_indicator": "home/away/none"
}}

OnlyReturnJSON，Do not addExplanation。
"""
        
        try:
            result = self.agent.run(prompt)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(result)
        except Exception as e:
            logger.error(f"OddsFeature extractionFailed: {e}")
            return {
                "home_odds": 2.0,
                "draw_odds": 3.5,
                "away_odds": 2.5,
                "home_implied_prob": 0.333,
                "draw_implied_prob": 0.286,
                "away_implied_prob": 0.400,
                "bookmaker_margin": 0.019,
                "odds_movement": "stable",
                "sharp_money_indicator": "none"
            }
    
    def calculate_derived_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        CalculateDerived features
        
        Args:
            features: BasicFeatures
        
        Returns:
            Derived features
        """
        logger.info("CalculateDerived features")
        
        derived = {}
        
        try:
            team1_form = features.get("team1", {}).get("recent_form", {})
            team2_form = features.get("team2", {}).get("recent_form", {})
            
            derived["form_differential"] = (
                team1_form.get("win_rate", 0.5) - team2_form.get("win_rate", 0.5)
            )
            
            derived["goals_differential"] = (
                team1_form.get("goals_scored_avg", 1.0) - 
                team2_form.get("goals_conceded_avg", 1.0)
            )
            
            team1_home = features.get("team1", {}).get("home_record", {})
            team2_away = features.get("team2", {}).get("away_record", {})
            
            team1_home_total = sum(team1_home.values()) or 1
            team2_away_total = sum(team2_away.values()) or 1
            
            derived["home_advantage"] = (
                team1_home.get("wins", 0) / team1_home_total
            )
            derived["away_strength"] = (
                team2_away.get("wins", 0) / team2_away_total
            )
            
            h2h = features.get("head_to_head", {})
            total_h2h = h2h.get("total_matches", 0) or 1
            
            derived["h2h_advantage"] = (
                (h2h.get("team1_wins", 0) - h2h.get("team2_wins", 0)) / total_h2h
            )
            
            odds = features.get("betting_odds", {})
            derived["odds_value_home"] = (
                1.0 / odds.get("team1_win", 2.0) if odds.get("team1_win", 0) > 0 else 0.5
            )
            derived["odds_value_away"] = (
                1.0 / odds.get("team2_win", 2.0) if odds.get("team2_win", 0) > 0 else 0.5
            )
            
            derived["overall_advantage_score"] = (
                derived["form_differential"] * 0.3 +
                derived["home_advantage"] * 0.2 +
                derived["h2h_advantage"] * 0.2 +
                (derived["odds_value_home"] - derived["odds_value_away"]) * 0.3
            )
            
            logger.info(f"Derived featuresCalculateComplete: {len(derived)}  Features")
            
        except Exception as e:
            logger.error(f"Derived featuresCalculateFailed: {e}")
        
        return derived
    
    def _get_default_features(self) -> Dict[str, Any]:
        """ReturnDefaultFeaturesStructure"""
        return {
            "team1": {
                "name": "Unknown",
                "recent_form": {
                    "wins": 5,
                    "draws": 3,
                    "losses": 2,
                    "win_rate": 0.5,
                    "goals_scored_avg": 1.5,
                    "goals_conceded_avg": 1.2
                },
                "home_record": {"wins": 6, "draws": 2, "losses": 2},
                "injuries": {"key_players_out": [], "severity": "low"},
                "league_position": 10,
                "form_trend": "stable"
            },
            "team2": {
                "name": "Unknown",
                "recent_form": {
                    "wins": 5,
                    "draws": 3,
                    "losses": 2,
                    "win_rate": 0.5,
                    "goals_scored_avg": 1.5,
                    "goals_conceded_avg": 1.2
                },
                "away_record": {"wins": 4, "draws": 3, "losses": 3},
                "injuries": {"key_players_out": [], "severity": "low"},
                "league_position": 10,
                "form_trend": "stable"
            },
            "head_to_head": {
                "total_matches": 10,
                "team1_wins": 4,
                "team2_wins": 4,
                "draws": 2,
                "avg_goals": 2.5
            },
            "betting_odds": {
                "team1_win": 2.0,
                "draw": 3.5,
                "team2_win": 2.5,
                "implied_prob_team1": 0.4,
                "implied_prob_draw": 0.25,
                "implied_prob_team2": 0.35
            },
            "expert_consensus": {
                "predicted_winner": "draw",
                "confidence": "low"
            },
            "external_factors": {
                "weather": "good",
                "stadium_advantage": "moderate"
            }
        }
    
    def extract_all_features(
        self, 
        match_data: Dict[str, Any],
        odds_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract all features（Basic + Derived）
        
        Args:
            match_data: MatchData
            odds_data: OddsData
        
        Returns:
            CompleteFeaturesSet
        """
        logger.info("StartextractGet completeFeaturesSet")
        
        base_features = self.extract_match_features(match_data)
        
        if odds_data:
            odds_features = self.extract_odds_features(odds_data)
            base_features["betting_odds"] = odds_features
        
        derived_features = self.calculate_derived_features(base_features)
        
        all_features = {
            "base_features": base_features,
            "derived_features": derived_features,
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        logger.info("Feature extractionComplete")
        return all_features


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    extractor = FeatureExtractor()
    
    test_match_data = {
        "head_to_head": {"result": "Last 5 matches: Team A won 3, Team B won 2"},
        "team1_form": {"result": "Team A: Last 10 matches - 7 wins, 2 draws, 1 loss"},
        "team2_form": {"result": "Team B: Last 10 matches - 5 wins, 3 draws, 2 losses"},
    }
    
    features = extractor.extract_match_features(test_match_data)
    print(json.dumps(features, indent=2, ensure_ascii=False))

