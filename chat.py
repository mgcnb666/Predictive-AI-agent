"""
Smart Prediction Chat - Simplified Launch Commands

Usage:
    python3 chat
"""
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


def print_welcome():
    """Print welcome message"""
    print("="*70)
    print("💬 Smart Prediction Chat")
    print("="*70)
    print("\n🎯 Enter your question directly, AI will automatically understand and predict!")
    print("\n📝 Examples:")
    print("  • predict tomorrow's weather in Beijing")
    print("  • what about the day after? (remembers last city)")
    print("  • who will win Barcelona vs Real Madrid")
    print("  • will Bitcoin go up? will AI replace programmers? (general prediction)")
    print("  • set default city to Shanghai")
    print("\n💡 AI will show understanding process (keyword extraction)")
    print("\n⚙️  Commands:")
    print("  • /help    - show help")
    print("  • /history - view conversation history")
    print("  • /context - view context")
    print("  • /clear   - clear context")
    print("  • /set <key> <value> - set preference")
    print("  • /quit or /exit - exit")
    print("\n" + "="*70 + "\n")


def print_help():
    """Print help message"""
    print("\n📖 Help Information")
    print("="*70)
    print("\n🌤️  Weather Prediction:")
    print("  • predict tomorrow's weather in Beijing")
    print("  • will it rain in Shanghai the day after tomorrow")
    print("  • how's the weather in Shenzhen today")
    print("\n⚽ Sports Prediction:")
    print("  • who will win Barcelona vs Real Madrid")
    print("  • Liverpool vs Manchester United match result")
    print("  • Serbia vs Latvia")
    print("\n🗳️  Election Prediction:")
    print("  • who will win 2024 US election Trump or Biden")
    print("\n⚙️  System Commands:")
    print("  • /help    - show this help")
    print("  • /history - view conversation history")
    print("  • /context - view current context")
    print("  • /clear   - clear all context")
    print("  • /set default_location Shanghai - set default city")
    print("  • /quit or /exit - exit program")
    print("="*70 + "\n")


def display_result(domain: str, result: dict):
    """Display prediction result (concise version)"""
    print("\n" + "-"*70)
    
    if domain == "weather":
        forecast = result.get('forecast', {})
        params = result.get('parameters', {})
        
        location = params.get('location', 'Unknown')
        condition = forecast.get('weather_condition', 'Unknown')
        temp = forecast.get('temperature_range', {})
        temp_low = temp.get('low', '?')
        temp_high = temp.get('high', '?')
        precip = forecast.get('precipitation_prob', 0)
        
        print(f"🌤️  {location} Weather: {condition}")
        print(f"   Temperature: {temp_low}-{temp_high}°C")
        print(f"   Precipitation: {precip:.0%}")
        
    elif domain == "sports":
        outcomes = result.get('outcomes', {})
        params = result.get('parameters', {})
        
        team1 = params.get('team1', 'Team1')
        team2 = params.get('team2', 'Team2')
        
        home_win = outcomes.get('home_win', 0)
        draw = outcomes.get('draw', 0)
        away_win = outcomes.get('away_win', 0)
        
        print(f"⚽ {team1} vs {team2}")
        print(f"   {team1} Win: {home_win:.0%}")
        print(f"   Draw: {draw:.0%}")
        print(f"   {team2} Win: {away_win:.0%}")
        
    elif domain == "election":
        predictions = result.get('predictions', {})
        params = result.get('parameters', {})
        total_candidates = result.get('total_candidates', 0)
        main_contenders = result.get('main_contenders', [])
        
        election = params.get('election', 'Unknown Election')
        region = params.get('region', '')
        
        print(f"🗳️  {region} {election}")
        if total_candidates:
            print(f"   Total Candidates: {total_candidates}")
        
        if predictions:
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            print(f"\n   Winning Probability:")
            for i, (candidate, prob) in enumerate(sorted_predictions, 1):
                marker = "⭐" if candidate in main_contenders else "  "
                print(f"   {marker}{i}. {candidate}: {prob:.1%}")
        else:
            print(f"   No candidate probability data available")
        
        vote_share = result.get('vote_share', {})
        if vote_share and len(vote_share) > 2:
            print(f"\n   Estimated Vote Share (Top 3):")
            sorted_votes = sorted(vote_share.items(), key=lambda x: x[1], reverse=True)[:3]
            for candidate, share in sorted_votes:
                print(f"   • {candidate}：{share:.1%}")
        
        swing_factors = result.get('swing_factors', [])
        if swing_factors:
            print(f"\n   Key Factors：")
            for factor in swing_factors[:3]:
                print(f"   • {factor}")
    
    elif domain == "general":
        prediction = result.get('result', '')
        probability = result.get('probability', 0)
        data_date = result.get('data_date', '')
        data_quality = result.get('data_quality', '')
        top_contenders = result.get('top_contenders', {})
        
        print(f"🔮 general prediction")
        
        if data_date:
            print(f"   📅 Data Date：{data_date}")
        if data_quality:
            print(f"   📊 Data Quality：{data_quality}")
        
        print(f"\n   Prediction Result：{prediction}")
        print(f"   Probability：{probability:.0%}")
        
        if top_contenders:
            print(f"\n   🏆 Top Contenders：")
            sorted_contenders = sorted(top_contenders.items(), key=lambda x: x[1], reverse=True)
            for i, (contender, prob) in enumerate(sorted_contenders, 1):
                marker = "⭐" if i == 1 else "  "
                print(f"   {marker}{i}. {contender}：{prob:.1%}")
        
        scenarios = result.get('scenarios', {})
        if scenarios:
            print(f"\n   Scenario Analysis：")
            if 'likely_case' in scenarios:
                print(f"   Most Likely：{scenarios['likely_case']}")
            if 'dark_horse' in scenarios:
                print(f"   Dark Horse：{scenarios['dark_horse']}")
            if 'best_case' in scenarios:
                print(f"   Best Case：{scenarios['best_case']}")
            if 'worst_case' in scenarios:
                print(f"   Worst Case：{scenarios['worst_case']}")
    
    confidence = result.get('confidence', 0)
    print(f"\n   Confidence：{confidence:.0%}")
    print("-"*70)


def main():
    """Main function"""
    print_welcome()
    
    parser = NLPParser()
    agent = UniversalPredictionAgent()
    ctx = ContextManager()
    
    while True:
        try:
            user_input = input("💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input in ['/quit', '/exit', 'quit', 'exit']:
                print("\n👋 Goodbye！")
                break
            
            if user_input == '/help':
                print_help()
                continue
            
            if user_input == '/history':
                print("\n📜 Conversation History（Recent10entries）：")
                print("-"*70)
                for i, msg in enumerate(ctx.conversation_history[-10:], 1):
                    role = "You" if msg['role'] == 'user' else "AI"
                    content = msg['content']
                    print(f"{i}. [{role}] {content[:60]}...")
                print("-"*70 + "\n")
                continue
            
            if user_input == '/context':
                print("\n📊 Current Context：")
                print("-"*70)
                summary = ctx.summarize()
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                print("-"*70 + "\n")
                continue
            
            if user_input == '/clear':
                ctx.clear()
                print("\n✅ Context cleared\n")
                continue
            
            if user_input.startswith('/set '):
                parts = user_input[5:].split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    ctx.set_preference(key, value)
                    print(f"\n✅ Set：{key} = {value}\n")
                else:
                    print("\n❌ Format error，please use：/set <key> <value>\n")
                continue
            
            ctx.add_message("user", user_input)
            
            print("🤔 AI is thinking...", end='', flush=True)
            parsed = parser.parse(user_input)
            print("\r" + " "*30 + "\r", end='', flush=True)  
            
            if "error" in parsed and parsed.get("confidence", 0) == 0:
                print(f"🤖 AI: Sorry, I didn't understand you. You can try saying:")
                print("     • predict tomorrow's weather in Beijing")
                print("     • who will win Barcelona vs Real Madrid\n")
                continue
            
            domain = parsed.get('domain')
            params = parsed.get('params', {})
            confidence = parsed.get('confidence', 0)
            
            print(f"💡 AI Understanding: ", end='')
            if domain == "weather":
                location = params.get('location', '?')
                date = params.get('date', '?')
                print(f"Query weather in {location}")
            elif domain == "sports":
                team1 = params.get('team1', '?')
                team2 = params.get('team2', '?')
                print(f"Predict {team1} vs {team2} match")
            elif domain == "election":
                election = params.get('election', '?')
                print(f"Predict {election}'s result")
            elif domain == "general":
                query = params.get('query', '?')
                print(f"General prediction: {query}")
            else:
                print(f"{domain} domain prediction")
            
            print(f"   Keywords：{json.dumps(params, ensure_ascii=False)}")
            print(f"   Confidence：{confidence:.0%}")
            
            if domain and params:
                original_params = params.copy()
                params = ctx.smart_complete_params(domain, params)
                parsed['params'] = params
                
                if params != original_params:
                    print(f"   📝 Auto-completed：", end='')
                    for key, value in params.items():
                        if key not in original_params or original_params[key] != value:
                            print(f"{key}={value} ", end='')
                    print()
            
            if parsed.get('confidence', 0) < 0.5:
                print(f"⚠️  Understanding confidence is low ({parsed.get('confidence', 0):.0%})")
                confirm = input("   Continue? (y/n): ").strip().lower()
                if confirm not in ['y', 'yes', '']:
                    print()
                    continue
            
            if not domain or not params:
                print("🤖 AI: Cannot extract prediction parameters, please describe differently\n")
                continue
            
            print("🔮 Predicting...", end='', flush=True)
            result = agent.predict(
                domain=domain,
                params=params,
                use_search=True
            )
            print("\r" + " "*20 + "\r", end='', flush=True)  
            
            ctx.add_prediction(domain, params, result)
            
            display_result(domain, result)
            
            if domain == "weather":
                forecast = result.get('forecast', {})
                condition = forecast.get('weather_condition', 'Unknown')
                assistant_reply = f"Weather prediction: {condition}"
            elif domain == "sports":
                outcomes = result.get('outcomes', {})
                winner = max(outcomes, key=outcomes.get) if outcomes else "Unknown"
                assistant_reply = f"Match prediction complete"
            else:
                assistant_reply = "Prediction complete"
            
            ctx.add_message("assistant", assistant_reply)
            print()  
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()

