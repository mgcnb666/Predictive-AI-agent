"""APIClientUseExample"""
import requests
import json


class PredictionAPIClient:
    """PredictionAPIClient"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        InitializeAPIClient
        
        Args:
            base_url: APIBase URL
        """
        self.base_url = base_url.rstrip("/")
    
    def health_check(self):
        """HealthCheck"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def analyze_match(
        self,
        team1: str,
        team2: str,
        league: str = "Unspecified",
        date: str = None,
        market_odds: dict = None
    ):
        """
        Analyze match
        
        Args:
            team1: Home team
            team2: Away team
            league: League
            date: Date
            market_odds: Market odds
        
        Returns:
            Analysis result
        """
        url = f"{self.base_url}/api/v1/analyze"
        
        payload = {
            "team1": team1,
            "team2": team2,
            "league": league
        }
        
        if date:
            payload["date"] = date
        if market_odds:
            payload["market_odds"] = market_odds
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def quick_predict(
        self,
        team1: str,
        team2: str,
        league: str = "Unspecified",
        market_odds: dict = None
    ):
        """
        Quick prediction
        
        Args:
            team1: Home team
            team2: Away team
            league: League
            market_odds: Market odds
        
        Returns:
            Prediction result
        """
        url = f"{self.base_url}/api/v1/quick-predict"
        
        payload = {
            "team1": team1,
            "team2": team2,
            "league": league
        }
        
        if market_odds:
            payload["market_odds"] = market_odds
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def batch_analyze(self, matches: list):
        """
        Batch analysis
        
        Args:
            matches: MatchList
        
        Returns:
            BatchAnalysis result
        """
        url = f"{self.base_url}/api/v1/batch-analyze"
        
        payload = {"matches": matches}
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_status(self):
        """GetAgentStatus"""
        url = f"{self.base_url}/api/v1/status"
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def get_leagues(self):
        """GetSupportedLeagueList"""
        url = f"{self.base_url}/api/v1/leagues"
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()


def main():
    """ Function - DemoAPIUse"""
    
    client = PredictionAPIClient(base_url="http://localhost:8000")
    
    print("="*60)
    print("PredictionAI Agent - APIClientExample")
    print("="*60)
    
    print("\n1. HealthCheck")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Agent Initialize: {health['agent_initialized']}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Please ensureAPIService hasStart: python main.py api")
        return
    
    print("\n2. GetSupportedLeague")
    try:
        leagues = client.get_leagues()
        print(f"   Support {leagues['total']}  League")
        for league in leagues['leagues'][:3]:
            print(f"   - {league['name']} ({league['code']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Quick prediction")
    try:
        result = client.quick_predict(
            team1="Manchester United",
            team2="Liverpool",
            league="Premier League"
        )
        print(f"   Match: {result['match']}")
        pred = result['prediction']
        print(f"   Prediction: Home win{pred['home_win_probability']:.1%} | "
              f"Draw{pred['draw_probability']:.1%} | "
              f"Away win{pred['away_win_probability']:.1%}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Recommendation: {result['recommendation']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Detailed analysis（ Odds）")
    try:
        result = client.analyze_match(
            team1="Barcelona",
            team2="Real Madrid",
            league="La Liga",
            market_odds={
                "home": 2.3,
                "draw": 3.4,
                "away": 3.0
            }
        )
        
        print(f"   Match: {result['match_info']['team1']} vs {result['match_info']['team2']}")
        pred = result['prediction']
        print(f"   Prediction:")
        print(f"     Home win: {pred['home_win_probability']:.2%}")
        print(f"     Draw: {pred['draw_probability']:.2%}")
        print(f"     Away win: {pred['away_win_probability']:.2%}")
        print(f"     Confidence: {pred['confidence']:.2%}")
        
        ba = result['betting_analysis']
        print(f"   BettingAnalysis:")
        print(f"     Betting recommendation: {ba.get('should_bet', False)}")
        if ba.get('best_bet'):
            bb = ba['best_bet']
            print(f"     Best option: {bb['outcome']}")
            print(f"     Expected priceValue: {bb['ev']:.2%}")
            print(f"     Betting recommendationRatio: {bb['bet_size_percentage']:.2%}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n5. Batch analysis")
    try:
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
        
        result = client.batch_analyze(matches)
        print(f"   Analysis  {result['total_matches']} fieldMatch")
        
        for i, match_result in enumerate(result['results'], 1):
            if 'error' not in match_result:
                info = match_result['match_info']
                pred = match_result['prediction']
                print(f"   Match{i}: {info['team1']} vs {info['team2']}")
                print(f"     Prediction: Home win{pred['home_win_probability']:.1%} | "
                      f"Draw{pred['draw_probability']:.1%} | "
                      f"Away win{pred['away_win_probability']:.1%}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n6. AgentStatus")
    try:
        status = client.get_status()
        print(f"   Status: {status['status']}")
        bankroll = status['bankroll']
        print(f"   whenBeforeFund: ${bankroll['current']:.2f}")
        print(f"   Initial bankroll: ${bankroll['initial']:.2f}")
        print(f"   Daily profit/loss: ${bankroll['daily_pnl']:.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*60)
    print("ExampleExecuteComplete")
    print("="*60)


if __name__ == "__main__":
    main()

