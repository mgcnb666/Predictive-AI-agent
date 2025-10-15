"""Smart Prediction - Using Natural Language Input"""
import sys
import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.nlp_parser import NLPParser
from src.universal_agent import UniversalPredictionAgent
from src.context_manager import ContextManager


def main():
    """Main function"""
    print("="*70)
    print("🤖 Smart Prediction System - Natural Language Version (with Context Support)")
    print("="*70)
    print("\n💡 Usage:")
    print("  - Enter your question directly, the system will automatically understand")
    print("  - System remembers your preferences and query history")
    print("  - Examples:")
    print("    • Predict tomorrow's weather in New York")
    print("    • Who will win Barcelona vs Real Madrid")
    print("    • What about the day after tomorrow? (will automatically use the previous city)")
    print("    • Set default city to Shanghai")
    print("\n  Special Commands:")
    print("    • history - view conversation history")
    print("    • context - view current context")
    print("    • clear - clear context")
    print("    • set <key> <value> - set preference")
    print("    • quit/exit - exit\n")
    
    parser = NLPParser()
    agent = UniversalPredictionAgent()
    ctx = ContextManager()
    
    while True:
        try:
            print("="*70)
            user_input = input("🎯 Enter your prediction question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'history':
                print("\n📜 Conversation History:")
                for i, msg in enumerate(ctx.conversation_history[-10:], 1):
                    print(f"  {i}. [{msg['role']}] {msg['content'][:50]}...")
                continue
            
            if user_input.lower() == 'context':
                print("\n📊 Current Context:")
                print(json.dumps(ctx.summarize(), indent=2, ensure_ascii=False))
                continue
            
            if user_input.lower() == 'clear':
                ctx.clear()
                print("\n✅ Context cleared")
                continue
            
            if user_input.lower().startswith('set '):
                parts = user_input[4:].split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    ctx.set_preference(key, value)
                    print(f"\n✅ Set: {key} = {value}")
                else:
                    print("\n❌ Format error, please use: set <key> <value>")
                continue
            
            print()
            
            ctx.add_message("user", user_input)
            
            print("🔍 Understanding your question...")
            parsed = parser.parse(user_input)
            
            if "error" in parsed and parsed.get("confidence", 0) == 0:
                print(f"❌ Sorry, unable to understand your question: {parsed['error']}")
                print("💡 Please try to describe your prediction needs more clearly")
                continue
            
            domain = parsed.get('domain')
            params = parsed.get('params', {})
            
            if domain and params:
                params = ctx.smart_complete_params(domain, params)
                parsed['params'] = params
            
            print(f"\n✅ Understanding successful!")
            print(f"  Domain: {domain}")
            print(f"  Parameters: {json.dumps(params, ensure_ascii=False)}")
            print(f"  Confidence: {parsed.get('confidence', 0):.0%}")
            
            if parsed.get('confidence', 0) < 0.5:
                print(f"\n⚠️  Understanding confidence is low, results may be inaccurate")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    continue
            
            if not domain or not params:
                print("❌ Cannot extract prediction parameters")
                continue
            
            print(f"\n🚀 Making prediction...")
            result = agent.predict(
                domain=domain,
                params=params,
                use_search=True
            )
            
            ctx.add_prediction(domain, params, result)
            
            print("\n" + "="*70)
            print("📊 Prediction Result")
            print("="*70)
            
            assistant_reply = generate_summary(domain, result)
            ctx.add_message("assistant", assistant_reply)
            
            if domain == "weather":
                display_weather_result(result)
            elif domain == "sports":
                display_sports_result(result)
            elif domain == "election":
                display_election_result(result)
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()


def display_weather_result(result: dict):
    """Display weather prediction result"""
    forecast = result.get('forecast', {})
    
    print(f"\n🌤️  Weather Forecast")
    print(f"  Location: {result['parameters'].get('location')}")
    print(f"  Date: {result['parameters'].get('date')}")
    print()
    
    if forecast:
        temp = forecast.get('temperature_range', {})
        print(f"  🌡️  Temperature: {temp.get('low')}°C - {temp.get('high')}°C")
        print(f"  ☁️  Weather: {forecast.get('weather_condition', 'Unknown')}")
        print(f"  💧 Precipitation: {forecast.get('precipitation_prob', 0):.0%}")
        print(f"  💨 Wind Speed: {forecast.get('wind_speed', {}).get('speed', 0)} km/h")
        print(f"  💦 Humidity: {forecast.get('humidity', 0):.0%}")
    
    print(f"\n  📈 Confidence: {result.get('confidence', 0):.0%}")
    
    if result.get('analysis'):
        print(f"\n  📝 Analysis:")
        print(f"     {result['analysis'][:200]}...")


def display_sports_result(result: dict):
    """Display sports prediction result"""
    outcomes = result.get('outcomes', {})
    params = result.get('parameters', {})
    
    print(f"\n⚽ Match Prediction")
    print(f"  {params.get('team1')} vs {params.get('team2')}")
    print(f"  League: {params.get('league')}")
    print()
    
    print(f"  🏆 {params.get('team1')} Win: {outcomes.get('home_win', 0):.1%}")
    print(f"  🤝 Draw: {outcomes.get('draw', 0):.1%}")
    print(f"  🏆 {params.get('team2')} Win: {outcomes.get('away_win', 0):.1%}")
    
    print(f"\n  📈 Confidence: {result.get('confidence', 0):.0%}")
    
    if result.get('analysis'):
        print(f"\n  📝 Analysis:")
        print(f"     {result['analysis'][:300]}...")
    
    if result.get('key_factors'):
        print(f"\n  🔑 Key Factors:")
        for factor in result['key_factors'][:3]:
            print(f"     • {factor}")


def display_election_result(result: dict):
    """Display election prediction result"""
    predictions = result.get('predictions', {})
    params = result.get('parameters', {})
    
    print(f"\n🗳️  Election Prediction")
    print(f"  Election: {params.get('election')}")
    print(f"  Region: {params.get('region')}")
    print()
    
    print(f"  📊 Winning Probability:")
    for candidate, prob in predictions.items():
        print(f"     • {candidate}: {prob:.1%}")
    
    print(f"\n  📈 Confidence: {result.get('confidence', 0):.0%}")
    
    if result.get('analysis'):
        print(f"\n  📝 Analysis:")
        print(f"     {result['analysis'][:300]}...")


def generate_summary(domain: str, result: dict) -> str:
    """Generate prediction result summary"""
    if domain == "weather":
        forecast = result.get('forecast', {})
        location = result.get('parameters', {}).get('location')
        condition = forecast.get('weather_condition', 'Unknown')
        return f"{location} weather: {condition}"
    
    elif domain == "sports":
        outcomes = result.get('outcomes', {})
        params = result.get('parameters', {})
        winner = max(outcomes, key=outcomes.get)
        prob = outcomes[winner]
        return f"{params.get('team1')} vs {params.get('team2')}: {winner} ({prob:.0%})"
    
    elif domain == "election":
        predictions = result.get('predictions', {})
        winner = max(predictions, key=predictions.get) if predictions else "Unknown"
        prob = predictions.get(winner, 0)
        return f"Election Prediction: {winner} leading ({prob:.0%})"
    
    return "Prediction complete"


if __name__ == "__main__":
    main()

