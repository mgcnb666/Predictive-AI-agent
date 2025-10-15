"""Prediction engine - Core prediction logic"""
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from loguru import logger

try:
    from smolagents import CodeAgent, LiteLLMModel
except ImportError:
    logger.warning("SmolAgentsNot installed")
    CodeAgent = None
    LiteLLMModel = None


class PredictionEngine:
    """Prediction engine - CombineLLMReasoningandStatisticsModel"""
    
    def __init__(
        self,
        model_name: str = "openrouter/google/gemini-2.0-flash-001",
        temperature: float = 0.3,
        confidence_threshold: float = 0.6,
        kelly_fraction: float = 0.25,
        max_bet_percentage: float = 0.05
    ):
        """
        InitializePrediction engine
        
        Args:
            model_name: LLM model name
            temperature: TemperatureParameters
            confidence_threshold: ConfidenceThresholdValue
            kelly_fraction: KellyCriteriaScore
            max_bet_percentage: Maximum betRatio
        """
        if LiteLLMModel is None:
            raise ImportError("Please install firstsmolagents: pip install smolagents")
        
        self.model = LiteLLMModel(model_name, temperature=temperature)
        self.agent = CodeAgent(tools=[], model=self.model)
        self.confidence_threshold = confidence_threshold
        self.kelly_fraction = kelly_fraction
        self.max_bet_percentage = max_bet_percentage
        
        logger.info(f"Prediction engineInitializeComplete - Model: {model_name}")
    
    def predict(
        self, 
        features: Dict[str, Any],
        team1_name: str,
        team2_name: str
    ) -> Dict[str, Any]:
        """
        Execute prediction
        
        Args:
            features: FeaturesData
            team1_name: Home teamName
            team2_name: Away teamName
        
        Returns:
            Prediction Result
        """
        logger.info(f"StartPrediction: {team1_name} vs {team2_name}")
        
        llm_prediction = self._llm_predict(features, team1_name, team2_name)
        
        stat_prediction = self._statistical_predict(features)
        
        final_prediction = self._ensemble_predictions(llm_prediction, stat_prediction)
        
        final_prediction["team1_name"] = team1_name
        final_prediction["team2_name"] = team2_name
        final_prediction["prediction_timestamp"] = datetime.now().isoformat()
        
        logger.info(
            f"PredictionComplete - Home win: {final_prediction['home_win_prob']:.2%}, "
            f"Draw: {final_prediction['draw_prob']:.2%}, "
            f"Away win: {final_prediction['away_win_prob']:.2%}"
        )
        
        return final_prediction
    
    def _llm_predict(
        self, 
        features: Dict[str, Any],
        team1_name: str,
        team2_name: str
    ) -> Dict[str, Any]:
        """UseLLMPerform reasoning prediction"""
        logger.info("LLMReasoning prediction in progress...")
        
        prompt = f"""
You are a professionalSports event predictionAnalysis 。Based onDownDataAnalysis{team1_name} vs {team2_name}matchResult。

FeaturesData：
{json.dumps(features, indent=2, ensure_ascii=False)}

Please provide detailed prediction analysis， withJSONFormatReturn：
{{
  "home_win_prob": 0.0,  // Home teamWinProbability (0-1)
  "draw_prob": 0.0,      // DrawProbability (0-1)
  "away_win_prob": 0.0,  // Away teamWinProbability (0-1)
  "confidence": 0.0,      // PredictionConfidence (0-1)
  "analysis": "Detailed analysis...",
  "key_factors": [        //  KeyInfluencing factors
    "Factors1",
    "Factors2"
  ],
  "risks": [              // Risk factors
    "Risk1",
    "Risk2"
  ],
  "expected_score": "1-1" // Expected score
}}

Requirement：
1.   ProbabilitySum must equal1
2. Based onDataObjective analysis，Do not be biased
3. Consider allKey factors：RecentStatus、HistoricalVersus、Home/away、Injury、Oddsetc
4. ConfidenceShould reflectDataReliability and consistency
5. OnlyReturnJSON，Do not addOtherInnercontent
"""
        
        try:
            result = self.agent.run(prompt)
            
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                prediction = json.loads(json_match.group())
            else:
                prediction = json.loads(result)
            
            total_prob = (
                prediction.get("home_win_prob", 0) +
                prediction.get("draw_prob", 0) +
                prediction.get("away_win_prob", 0)
            )
            
            if abs(total_prob - 1.0) > 0.01:  
                prediction["home_win_prob"] /= total_prob
                prediction["draw_prob"] /= total_prob
                prediction["away_win_prob"] /= total_prob
            
            return prediction
            
        except Exception as e:
            logger.error(f"LLMPredictionFailed: {e}")
            return self._get_default_prediction()
    
    def _statistical_predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Based onStatisticsModel prediction"""
        logger.info("StatisticsModel prediction in progress...")
        
        try:
            derived = features.get("derived_features", {})
            base = features.get("base_features", {})
            
            form_diff = derived.get("form_differential", 0)
            home_adv = derived.get("home_advantage", 0.5)
            h2h_adv = derived.get("h2h_advantage", 0)
            overall_score = derived.get("overall_advantage_score", 0)
            
            
            odds = base.get("betting_odds", {})
            base_home_prob = odds.get("implied_prob_team1", 0.35)
            base_draw_prob = odds.get("implied_prob_draw", 0.30)
            base_away_prob = odds.get("implied_prob_team2", 0.35)
            
            adjustment = (
                form_diff * 0.15 +
                home_adv * 0.10 +
                h2h_adv * 0.08 +
                overall_score * 0.12
            )
            
            home_prob = np.clip(base_home_prob + adjustment, 0.1, 0.8)
            away_prob = np.clip(base_away_prob - adjustment, 0.1, 0.8)
            draw_prob = 1.0 - home_prob - away_prob
            draw_prob = np.clip(draw_prob, 0.1, 0.5)
            
            total = home_prob + draw_prob + away_prob
            home_prob /= total
            draw_prob /= total
            away_prob /= total
            
            return {
                "home_win_prob": float(home_prob),
                "draw_prob": float(draw_prob),
                "away_win_prob": float(away_prob),
                "confidence": 0.65  
            }
            
        except Exception as e:
            logger.error(f"StatisticsPredictionFailed: {e}")
            return {
                "home_win_prob": 0.35,
                "draw_prob": 0.30,
                "away_win_prob": 0.35,
                "confidence": 0.5
            }
    
    def _ensemble_predictions(
        self,
        llm_pred: Dict[str, Any],
        stat_pred: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combine multiplePrediction Result"""
        logger.info("FusionPrediction result...")
        
        llm_weight = 0.6
        stat_weight = 0.4
        
        home_prob = (
            llm_pred.get("home_win_prob", 0.33) * llm_weight +
            stat_pred.get("home_win_prob", 0.33) * stat_weight
        )
        draw_prob = (
            llm_pred.get("draw_prob", 0.33) * llm_weight +
            stat_pred.get("draw_prob", 0.33) * stat_weight
        )
        away_prob = (
            llm_pred.get("away_win_prob", 0.34) * llm_weight +
            stat_pred.get("away_win_prob", 0.34) * stat_weight
        )
        
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        confidence = (
            llm_pred.get("confidence", 0.5) * llm_weight +
            stat_pred.get("confidence", 0.5) * stat_weight
        )
        
        return {
            "home_win_prob": float(home_prob),
            "draw_prob": float(draw_prob),
            "away_win_prob": float(away_prob),
            "confidence": float(confidence),
            "analysis": llm_pred.get("analysis", "NoneDetailed analysis"),
            "key_factors": llm_pred.get("key_factors", []),
            "risks": llm_pred.get("risks", []),
            "expected_score": llm_pred.get("expected_score", "Not predicted"),
            "llm_prediction": llm_pred,
            "statistical_prediction": stat_pred
        }
    
    def calculate_expected_value(
        self,
        prediction: Dict[str, Any],
        market_odds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        CalculateExpected priceValue（Expected Value）
        
        Args:
            prediction: Prediction result
            market_odds: Market odds {"home": 2.0, "draw": 3.5, "away": 2.5}
        
        Returns:
            EVAnalysis result
        """
        logger.info("CalculateExpected priceValue...")
        
        home_prob = prediction["home_win_prob"]
        draw_prob = prediction["draw_prob"]
        away_prob = prediction["away_win_prob"]
        
        home_odds = market_odds.get("home", 2.0)
        draw_odds = market_odds.get("draw", 3.5)
        away_odds = market_odds.get("away", 2.5)
        
        home_ev = (home_prob * home_odds) - 1
        draw_ev = (draw_prob * draw_odds) - 1
        away_ev = (away_prob * away_odds) - 1
        
        outcomes = [
            ("Home win", home_ev, home_prob, home_odds),
            ("Draw", draw_ev, draw_prob, draw_odds),
            ("Away win", away_ev, away_prob, away_odds)
        ]
        
        best_bet = max(outcomes, key=lambda x: x[1])
        
        ev_threshold = 0.05  
        confidence_threshold = self.confidence_threshold
        
        should_bet = (
            best_bet[1] > ev_threshold and
            prediction["confidence"] >= confidence_threshold
        )
        
        if should_bet:
            bet_size = self._calculate_kelly_bet(
                best_bet[2],  
                best_bet[3]   
            )
            
            recommendation = (
                f"RecommendBetting【{best_bet[0]}】\n"
                f"Expected priceValue: {best_bet[1]:.2%}\n"
                f"Betting recommendationRatio: {bet_size:.2%}\n"
                f"PredictionWin probability: {best_bet[2]:.2%}\n"
                f"Market odds: {best_bet[3]:.2f}"
            )
        else:
            reason = []
            if best_bet[1] <= ev_threshold:
                reason.append(f"most EV as{best_bet[1]:.2%}，Less thanThresholdValue{ev_threshold:.2%}")
            if prediction["confidence"] < confidence_threshold:
                reason.append(f"Confidence{prediction['confidence']:.2%}Too low")
            
            recommendation = f"notBetting recommendation - {'; '.join(reason)}"
            bet_size = 0.0
        
        return {
            "should_bet": should_bet,
            "recommendation": recommendation,
            "best_bet": {
                "outcome": best_bet[0],
                "ev": best_bet[1],
                "probability": best_bet[2],
                "odds": best_bet[3],
                "bet_size_percentage": bet_size
            },
            "all_evs": {
                "home_ev": home_ev,
                "draw_ev": draw_ev,
                "away_ev": away_ev
            },
            "market_efficiency": self._calculate_market_efficiency(market_odds)
        }
    
    def _calculate_kelly_bet(self, win_prob: float, odds: float) -> float:
        """
        UseKellyCriteriaCalculatemost BettingRatio
        
        Args:
            win_prob: Win probability
            odds: Odds
        
        Returns:
            BettingRatio（0-1）
        """
        
        q = 1 - win_prob
        b = odds - 1  
        
        if b <= 0:
            return 0.0
        
        kelly = (win_prob * b - q) / b
        
        fractional_kelly = kelly * self.kelly_fraction
        
        bet_size = max(0, min(fractional_kelly, self.max_bet_percentage))
        
        return bet_size
    
    def _calculate_market_efficiency(self, market_odds: Dict[str, float]) -> Dict[str, Any]:
        """CalculateMarket efficiency（Throughbookmaker margin）"""
        home_odds = market_odds.get("home", 2.0)
        draw_odds = market_odds.get("draw", 3.5)
        away_odds = market_odds.get("away", 2.5)
        
        implied_total = (1/home_odds + 1/draw_odds + 1/away_odds)
        margin = implied_total - 1.0
        
        efficiency = "high" if margin < 0.05 else "medium" if margin < 0.08 else "low"
        
        return {
            "bookmaker_margin": margin,
            "efficiency": efficiency,
            "implied_probability_total": implied_total
        }
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """ReturnDefaultPrediction"""
        return {
            "home_win_prob": 0.35,
            "draw_prob": 0.30,
            "away_win_prob": 0.35,
            "confidence": 0.5,
            "analysis": "Datanot ，UseDefaultPrediction",
            "key_factors": ["Datanot "],
            "risks": ["Prediction reliability is low"],
            "expected_score": "Unknown"
        }


class RiskManager:
    """RiskManagementer"""
    
    def __init__(
        self,
        initial_bankroll: float = 10000,
        max_risk_per_bet: float = 0.05,
        max_daily_loss: float = 0.10
    ):
        """
        InitializeRiskManagementer
        
        Args:
            initial_bankroll: Initial bankroll
            max_risk_per_bet: Single maximum riskRatio
            max_daily_loss: Daily maximum lossRatio
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_risk_per_bet = max_risk_per_bet
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
        
        logger.info(f"RiskManagementerInitialize - Fund: {initial_bankroll}")
    
    def calculate_bet_amount(
        self,
        bet_size_percentage: float,
        confidence: float
    ) -> float:
        """
        CalculateActual betting amount
        
        Args:
            bet_size_percentage: KellyRecommendationBettingRatio
            confidence: PredictionConfidence
        
        Returns:
            Actual betting amount
        """
        adjusted_percentage = bet_size_percentage * confidence
        
        adjusted_percentage = min(adjusted_percentage, self.max_risk_per_bet)
        
        if abs(self.daily_pnl) >= self.max_daily_loss * self.initial_bankroll:
            logger.warning("Daily loss limit reached，StopBetting")
            return 0.0
        
        bet_amount = self.current_bankroll * adjusted_percentage
        
        logger.info(f"CalculateBettingAmount: {bet_amount:.2f} ({adjusted_percentage:.2%})")
        
        return bet_amount
    
    def update_bankroll(self, pnl: float):
        """UpdateFund """
        self.current_bankroll += pnl
        self.daily_pnl += pnl
        logger.info(f"FundUpdate: {self.current_bankroll:.2f} (Daily profit/loss: {self.daily_pnl:.2f})")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    engine = PredictionEngine()
    
    test_features = {
        "base_features": {
            "team1": {"name": "Team A"},
            "team2": {"name": "Team B"},
            "betting_odds": {
                "team1_win": 2.0,
                "draw": 3.5,
                "team2_win": 2.8
            }
        },
        "derived_features": {
            "form_differential": 0.2,
            "home_advantage": 0.6,
            "h2h_advantage": 0.1,
            "overall_advantage_score": 0.15
        }
    }
    
    prediction = engine.predict(test_features, "Team A", "Team B")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    
    market_odds = {"home": 2.0, "draw": 3.5, "away": 2.8}
    ev_analysis = engine.calculate_expected_value(prediction, market_odds)
    print(json.dumps(ev_analysis, indent=2, ensure_ascii=False))

