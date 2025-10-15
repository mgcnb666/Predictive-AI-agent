"""Main Program Entry - Command Line Interface"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import PredictionAgent


def setup_logging(log_level: str = "INFO", log_file: str = "logs/agent.log"):
    """Configure logging"""
    os.makedirs("logs", exist_ok=True)
    
    logger.remove()  
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level
    )
    logger.add(
        log_file,
        rotation="1 day",
        retention="7 days",
        level=log_level
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Prediction AI Agent - Sports Prediction System Based on OpenDeepSearch"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze match")
    analyze_parser.add_argument("--team1", required=True, help="Home team name")
    analyze_parser.add_argument("--team2", required=True, help="Away team name")
    analyze_parser.add_argument("--league", default="Unspecified", help="League name")
    analyze_parser.add_argument("--date", help="Match date (YYYY-MM-DD)")
    analyze_parser.add_argument("--odds-home", type=float, help="Home team odds")
    analyze_parser.add_argument("--odds-draw", type=float, help="Draw odds")
    analyze_parser.add_argument("--odds-away", type=float, help="Away team odds")
    analyze_parser.add_argument("--output", help="Output file path (JSON)")
    
    predict_parser = subparsers.add_parser("predict", help="Quick prediction")
    predict_parser.add_argument("--team1", required=True, help="Home team name")
    predict_parser.add_argument("--team2", required=True, help="Away team name")
    predict_parser.add_argument("--league", default="Unspecified", help="League name")
    
    api_parser = subparsers.add_parser("api", help="Start API service")
    api_parser.add_argument("--host", default="0.0.0.0", help="Listen address")
    api_parser.add_argument("--port", type=int, default=8000, help="Port")
    api_parser.add_argument("--reload", action="store_true", help="Auto reload")
    
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--search-provider", help="Search provider (serper/searxng)")
    
    args = parser.parse_args()
    
    setup_logging(log_level=args.log_level)
    
    if args.command == "analyze":
        logger.info(f"Starting analysis: {args.team1} vs {args.team2}")
        
        agent_config = {}
        if args.model:
            agent_config["model_name"] = args.model
        if args.search_provider:
            agent_config["search_provider"] = args.search_provider
        
        agent = PredictionAgent(**agent_config)
        
        market_odds = None
        if args.odds_home and args.odds_draw and args.odds_away:
            market_odds = {
                "home": args.odds_home,
                "draw": args.odds_draw,
                "away": args.odds_away
            }
        
        result = agent.analyze_match(
            team1=args.team1,
            team2=args.team2,
            league=args.league,
            date=args.date,
            market_odds=market_odds
        )
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {args.output}")
        else:
            print("\n" + "="*80)
            print(f"Match Analysis: {args.team1} vs {args.team2}")
            print("="*80)
            
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
            else:
                pred = result["prediction"]
                print(f"\n📊 Prediction Result:")
                print(f"  Home Win Probability: {pred['home_win_probability']:.2%}")
                print(f"  Draw Probability: {pred['draw_probability']:.2%}")
                print(f"  Away Win Probability: {pred['away_win_probability']:.2%}")
                print(f"  Confidence: {pred['confidence']:.2%}")
                print(f"  Expected Score: {pred['expected_score']}")
                
                if "betting_analysis" in result:
                    ba = result["betting_analysis"]
                    print(f"\n💰 Betting Recommendation:")
                    print(f"  {ba.get('recommendation', 'No recommendation')}")
                
                analysis = result.get("analysis", {})
                print(f"\n📝 Analysis Summary:")
                print(f"  {analysis.get('summary', 'None')[:300]}")
                
                if analysis.get("key_factors"):
                    print(f"\n🔑 Key Factors:")
                    for factor in analysis["key_factors"][:5]:
                        print(f"  - {factor}")
                
                print("\n" + "="*80)
                print(f"Full results saved, use --output parameter to export")
    
    elif args.command == "predict":
        logger.info(f"Quick prediction: {args.team1} vs {args.team2}")
        
        agent_config = {}
        if args.model:
            agent_config["model_name"] = args.model
        if args.search_provider:
            agent_config["search_provider"] = args.search_provider
        
        agent = PredictionAgent(**agent_config)
        
        result = agent.quick_predict(
            team1=args.team1,
            team2=args.team2,
            league=args.league
        )
        
        print(f"\n⚡ Quick Prediction: {result['match']}")
        print(f"Home Win: {result['prediction']['home_win_probability']:.2%} | "
              f"Draw: {result['prediction']['draw_probability']:.2%} | "
              f"Away Win: {result['prediction']['away_win_probability']:.2%}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nRecommendation: {result['recommendation']}")
    
    elif args.command == "api":
        logger.info(f"Starting API service: http://{args.host}:{args.port}")
        
        import uvicorn
        from src.api import app
        
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

