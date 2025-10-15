"""SmartPredictionExample - Demonstrates how to use natural language for prediction"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp_parser import NLPParser
from src.universal_agent import UniversalPredictionAgent


def example1_weather():
    """Example1: Weather prediction"""
    print("\n" + "="*70)
    print("Example1: Use natural language to predict weather")
    print("="*70)
    
    user_input = "Predict tomorrow's weather in New York"
    print(f"User input: {user_input}")
    
    parser = NLPParser()
    parsed = parser.parse(user_input)
    print(f"\nParseResult: {parsed}")
    
    agent = UniversalPredictionAgent()
    result = agent.predict(
        domain=parsed['domain'],
        params=parsed['params'],
        use_search=False  
    )
    
    print(f"\nPrediction result:")
    print(f"  Dayweather: {result['forecast']['weather_condition']}")
    print(f"  Temperature: {result['forecast']['temperature_range']}")
    print(f"  Confidence: {result['confidence']:.0%}")


def example2_sports():
    """Example2: Sports prediction"""
    print("\n" + "="*70)
    print("Example2: UseNatural languagePredictionMatch")
    print("="*70)
    
    user_inputs = [
        "Who will win Barcelona vs Real Madrid",
        "Liverpool vs Manchester United matchResult",
        "Serbia vs Latvia World Cup Qualifiers",
    ]
    
    parser = NLPParser()
    agent = UniversalPredictionAgent()
    
    for user_input in user_inputs:
        print(f"\nUser input: {user_input}")
        
        parsed = parser.parse(user_input)
        print(f"  recognize : {parsed['params'].get('team1')} vs {parsed['params'].get('team2')}")
        
        result = agent.predict(
            domain=parsed['domain'],
            params=parsed['params'],
            use_search=False
        )
        
        outcomes = result['outcomes']
        print(f"  Prediction: Home win{outcomes['home_win']:.0%} | "
              f"Draw{outcomes['draw']:.0%} | "
              f"Away win{outcomes['away_win']:.0%}")


def example3_multiple_questions():
    """Example3: BatchProcessmultiple Question"""
    print("\n" + "="*70)
    print("Example3: BatchProcessmultiple PredictionQuestion")
    print("="*70)
    
    questions = [
        "Predict tomorrow's weather in Beijing",
        "AfterDayUphai DownWill it rain",
        "Who will win Barcelona vs Real Madrid",
        "2024Who will win the US election Trump or Biden",
    ]
    
    parser = NLPParser()
    agent = UniversalPredictionAgent()
    
    for question in questions:
        print(f"\n❓ {question}")
        
        try:
            parsed = parser.parse(question)
            domain = parsed['domain']
            
            print(f"   recognize Domain: {domain}")
            
            result = agent.predict(
                domain=domain,
                params=parsed['params'],
                use_search=False
            )
            
            if domain == 'weather':
                print(f"   ✅ {result['forecast']['weather_condition']}")
            elif domain == 'sports':
                outcomes = result['outcomes']
                winner = max(outcomes, key=outcomes.get)
                print(f"   ✅ mostPossibleResult: {winner} ({outcomes[winner]:.0%})")
            elif domain == 'election':
                preds = result['predictions']
                winner = max(preds, key=preds.get)
                print(f"   ✅ LeadingCandidate: {winner} ({preds[winner]:.0%})")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def example4_conversational():
    """Example4: ConversationStyle interaction"""
    print("\n" + "="*70)
    print("Example4: Conversation Prediction（Simulation）")
    print("="*70)
    
    conversation = [
        ("User", "TomorrowDayDayHow's the weather"),
        ("System", "May I ask what you wantQuery  CityDayweather？"),
        ("User", "Beijing"),
    ]
    
    for speaker, message in conversation:
        if speaker == "User":
            print(f"\n👤 {speaker}: {message}")
        else:
            print(f"🤖 {speaker}: {message}")
    
    print("\n💡 Prompt：CompleteConversationStyle functionNeedSessionManagementandContextTrack")


def main():
    """RunAllExample"""
    print("="*70)
    print("🤖 SmartPredictionSystem - UseExample")
    print("="*70)
    
    examples = [
        ("Weather prediction", example1_weather),
        ("Sports prediction", example2_sports),
        ("BatchProcess", example3_multiple_questions),
        ("ConversationStyle interaction", example4_conversational),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nExample [{name}] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ AllExampleExecuteComplete")
    print("="*70)
    print("\n💡 Prompt：Run python3 smart_predict.py StartInteractivePredictionSystem")


if __name__ == "__main__":
    main()

