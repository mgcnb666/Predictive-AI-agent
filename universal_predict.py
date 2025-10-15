"""General Prediction CLI - Supports Multi-Domain Prediction"""
import argparse
import json
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.universal_agent import UniversalPredictionAgent


def predict_weather(args):
    """Weather prediction"""
    agent = UniversalPredictionAgent()
    
    params = {
        "location": args.location,
        "date": args.date,
        "days_ahead": args.days,
    }
    
    if args.event:
        params["event"] = args.event
    
    result = agent.predict(
        domain="weather",
        params=params,
        use_search=args.use_search
    )
    
    return result


def predict_election(args):
    """Election Prediction"""
    agent = UniversalPredictionAgent()
    
    params = {
        "election": args.election,
        "region": args.region,
        "candidates": args.candidates,
    }
    
    if args.focus_states:
        params["focus_states"] = args.focus_states
    
    result = agent.predict(
        domain="election",
        params=params,
        use_search=args.use_search
    )
    
    return result


def predict_sports(args):
    """Sports Prediction (Universal Interface)"""
    agent = UniversalPredictionAgent()
    
    params = {
        "team1": args.team1,
        "team2": args.team2,
        "league": args.league,
    }
    
    if args.date:
        params["date"] = args.date
    
    result = agent.predict(
        domain="sports",
        params=params,
        use_search=args.use_search
    )
    
    return result


def list_domains(args):
    """List supported domains"""
    agent = UniversalPredictionAgent()
    
    domains = agent.list_domains()
    
    print("\n" + "="*60)
    print("Supported Prediction Domains")
    print("="*60)
    
    for domain in domains:
        info = agent.get_domain_info(domain)
        print(f"\n📊 {domain.upper()}")
        print(f"Class: {info['class']}")
        print(f"Description:\n{info['system_prompt']}")
        print("-"*60)
    
    return {"domains": domains}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Universal Prediction AI Agent - Supports Weather, Election, Sports and other multi-domain predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python universal_predict.py weather --location "Beijing" --date "2025-10-20" --days 7
  
  python universal_predict.py election --election "2024 Election" --region "United States" --candidates "Candidate A" "Candidate B"
  
  python universal_predict.py sports --team1 "Barcelona" --team2 "Real Madrid" --league "La Liga"
  
  python universal_predict.py list
        """
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (JSON format)"
    )
    
    parser.add_argument(
        "--no-search",
        dest="use_search",
        action="store_false",
        help="Do not use search function (only use LLM knowledge)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Prediction domain")
    
    weather_parser = subparsers.add_parser("weather", help="Weather prediction")
    weather_parser.add_argument("--location", required=True, help="Location")
    weather_parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    weather_parser.add_argument("--days", type=int, default=7, help="Prediction days")
    weather_parser.add_argument("--event", help="Specific event")
    
    election_parser = subparsers.add_parser("election", help="Election prediction")
    election_parser.add_argument("--election", required=True, help="Election name")
    election_parser.add_argument("--region", required=True, help="Region")
    election_parser.add_argument("--candidates", nargs="+", required=True, help="Candidate list")
    election_parser.add_argument("--focus-states", nargs="+", help="Focus states/regions")
    
    sports_parser = subparsers.add_parser("sports", help="Sports prediction")
    sports_parser.add_argument("--team1", required=True, help="Team 1")
    sports_parser.add_argument("--team2", required=True, help="Team 2")
    sports_parser.add_argument("--league", required=True, help="League")
    sports_parser.add_argument("--date", help="Match date")
    
    list_parser = subparsers.add_parser("list", help="List supported prediction domains")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    logger.info(f"Executing {args.command} prediction")
    
    try:
        if args.command == "weather":
            result = predict_weather(args)
        elif args.command == "election":
            result = predict_election(args)
        elif args.command == "sports":
            result = predict_sports(args)
        elif args.command == "list":
            result = list_domains(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return
        
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        print("\n" + "="*60)
        print("Prediction Result")
        print("="*60)
        print(output_json)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_json)
            logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

