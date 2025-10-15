"""SimpleUseExample"""
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import PredictionAgent


def example1_basic_prediction():
    """Example1: Basic prediction"""
    print("\n" + "="*60)
    print("Example1: Basic prediction")
    print("="*60)
    
    agent = PredictionAgent()
    
    result = agent.quick_predict(
        team1="Barcelona",
        team2="Real Madrid",
        league="La Liga"
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))


def example2_detailed_analysis():
    """Example2: Detailed analysis"""
    print("\n" + "="*60)
    print("Example2: Detailed analysis（ Odds）")
    print("="*60)
    
    agent = PredictionAgent()
    
    result = agent.analyze_match(
        team1="Manchester City",
        team2="Arsenal",
        league="Premier League",
        market_odds={
            "home": 1.8,
            "draw": 3.6,
            "away": 4.2
        }
    )
    
    print(f"\nMatch: {result['match_info']['team1']} vs {result['match_info']['team2']}")
    print(f"\nPrediction result:")
    pred = result['prediction']
    print(f"  Home win: {pred['home_win_probability']:.2%}")
    print(f"  Draw: {pred['draw_probability']:.2%}")
    print(f"  Away win: {pred['away_win_probability']:.2%}")
    print(f"  Confidence: {pred['confidence']:.2%}")
    
    print(f"\nBettingAnalysis:")
    ba = result['betting_analysis']
    print(f"  WhetherBetting recommendation: {ba.get('should_bet', False)}")
    print(f"  Recommendation: {ba.get('recommendation', 'None')}")
    
    if 'best_bet' in ba:
        bb = ba['best_bet']
        print(f"  most Betting: {bb['outcome']}")
        print(f"  Expected priceValue: {bb['ev']:.2%}")
        print(f"  Betting recommendationRatio: {bb['bet_size_percentage']:.2%}")


def example3_batch_analysis():
    """Example3: Batch analysis"""
    print("\n" + "="*60)
    print("Example3: Batch analysismultiplefieldMatch")
    print("="*60)
    
    agent = PredictionAgent()
    
    matches = [
        {
            "team1": "Liverpool",
            "team2": "Chelsea",
            "league": "Premier League",
            "market_odds": {"home": 2.1, "draw": 3.4, "away": 3.2}
        },
        {
            "team1": "Bayern Munich",
            "team2": "Borussia Dortmund",
            "league": "Bundesliga",
            "market_odds": {"home": 1.7, "draw": 3.8, "away": 4.5}
        }
    ]
    
    results = agent.batch_analyze(matches)
    
    for i, result in enumerate(results, 1):
        if "error" in result:
            print(f"\nMatch{i}: AnalysisFailed - {result['error']}")
        else:
            info = result['match_info']
            pred = result['prediction']
            print(f"\nMatch{i}: {info['team1']} vs {info['team2']}")
            print(f"  Prediction: Home win{pred['home_win_probability']:.1%} | "
                  f"Draw{pred['draw_probability']:.1%} | "
                  f"Away win{pred['away_win_probability']:.1%}")


def example4_agent_status():
    """Example4: CheckAgentStatus"""
    print("\n" + "="*60)
    print("Example4: AgentStatusMonitoring")
    print("="*60)
    
    agent = PredictionAgent(initial_bankroll=5000)
    
    status = agent.get_agent_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        example1_basic_prediction()
    except Exception as e:
        print(f"Example1Failed: {e}")
    
    try:
        example2_detailed_analysis()
    except Exception as e:
        print(f"Example2Failed: {e}")
    
    try:
        example3_batch_analysis()
    except Exception as e:
        print(f"Example3Failed: {e}")
    
    try:
        example4_agent_status()
    except Exception as e:
        print(f"Example4Failed: {e}")
    
    print("\n" + "="*60)
    print("AllExampleExecuteComplete")
    print("="*60)

