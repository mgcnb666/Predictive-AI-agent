"""ContextShareExample - Demonstrates how to share between different agentsContext"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.context_manager import ContextManager, SharedContextManager
from src.universal_agent import UniversalPredictionAgent


def example1_basic_context():
    """Example1: BasicContextUse"""
    print("\n" + "="*70)
    print("Example1: BasicContext - RememberUserPreference")
    print("="*70)
    
    ctx = ContextManager(session_id="user-001")
    agent = UniversalPredictionAgent()
    
    ctx.set_preference("default_location", "Uphai")
    print(f"✅ SettingDefaultCity: Uphai")
    
    print("\nQuery1: Predict tomorrow's weather in Beijing")
    params = {"location": "Beijing", "date": "2025-10-16", "days_ahead": 1}
    result = agent.predict("weather", params, use_search=False)
    ctx.add_prediction("weather", params, result)
    print(f"  Result: {result['forecast']['weather_condition']}")
    
    print("\nQuery2: AfterDay？（Not specifiedCity）")
    incomplete_params = {"date": "2025-10-17", "days_ahead": 1}
    completed_params = ctx.smart_complete_params("weather", incomplete_params)
    print(f"  Auto completeLocation: {completed_params['location']}")
    
    print("\nQuery3: TomorrowDayDayweather？（CompletelynotSpecified）")
    empty_params = {"date": "2025-10-18", "days_ahead": 1}
    completed_params = ctx.smart_complete_params("weather", empty_params)
    print(f"  UseDefaultCity: {completed_params['location']}")


def example2_conversation_context():
    """Example2: ConversationContext"""
    print("\n" + "="*70)
    print("Example2: ConversationContext - multiple Conversation")
    print("="*70)
    
    ctx = ContextManager(session_id="user-002")
    
    conversation = [
        ("user", "Predict tomorrow's weather in New York"),
        ("assistant", "New York will be cloudy to sunny tomorrow，15-22 "),
        ("user", "AfterDay"),
        ("assistant", "AfterDayNew Yorksunny，16-24 "),
        ("user", "Who will win Barcelona vs Real Madrid"),
        ("assistant", "PredictionBarcelonaWinProbability65%"),
    ]
    
    for role, content in conversation:
        ctx.add_message(role, content)
        print(f"  [{role}] {content}")
    
    print("\nConversationSummary:")
    summary = ctx.get_conversation_context(max_turns=3)
    print(summary)


def example3_shared_context():
    """Example3: ShareContext - Multiple agents access the sameContext"""
    print("\n" + "="*70)
    print("Example3: ShareContext - multipleAgentShare")
    print("="*70)
    
    print("\n🤖 Agent 1: Weather prediction")
    ctx1 = SharedContextManager.get_context("user-003")
    ctx1.set_preference("default_location", "Shenzhen")
    ctx1.set_preference("timezone", "Asia/Shanghai")
    print(f"  SettingDefaultCity: Shenzhen")
    
    print("\n🤖 Agent 2: Sports prediction")
    ctx2 = SharedContextManager.get_context("user-003")
    location = ctx2.get_preference("default_location")
    print(f"  ReadDefaultCity: {location}")
    print(f"  ✅ SuccessShareContext！")
    
    print(f"\nValidate: ctx1 is ctx2 = {ctx1 is ctx2}")
    
    print(f"\nActiveSession: {SharedContextManager.list_sessions()}")


def example4_context_persistence():
    """Example4: ContextPersistence"""
    print("\n" + "="*70)
    print("Example4: ContextPersistence - SaveandLoad")
    print("="*70)
    
    ctx = ContextManager(session_id="user-004")
    ctx.set_preference("default_location", "Guangzhou")
    ctx.add_message("user", "Predict tomorrow's weather in Guangzhou")
    ctx.add_message("assistant", "GuangzhouTomorrowDayCloudy，25-32 ")
    
    ctx.add_prediction(
        "weather",
        {"location": "Guangzhou", "date": "2025-10-16"},
        {"forecast": {"weather_condition": "Cloudy"}}
    )
    
    print("📁 SaveContextto file...")
    ctx.save_to_file("test_context.json")
    print("  ✅ SaveSuccess")
    
    print("\n📂 LoadContext...")
    new_ctx = ContextManager(session_id="new-session")
    new_ctx.load_from_file("test_context.json")
    print("  ✅ LoadSuccess")
    
    print(f"\nRestoreContext:")
    print(f"  SessionID: {new_ctx.session_id}")
    print(f"  Conversation : {len(new_ctx.conversation_history)}")
    print(f"  Preference: {new_ctx.preferences}")
    
    import os
    if os.path.exists("test_context.json"):
        os.remove("test_context.json")
        print("\n  🗑️  CleanTesting item")


def example5_smart_completion():
    """Example5: SmartCompleteScenario"""
    print("\n" + "="*70)
    print("Example5: SmartComplete - ActualUseScenario")
    print("="*70)
    
    ctx = ContextManager(session_id="user-005")
    agent = UniversalPredictionAgent()
    
    print("\n📍 Scenario1: ContinuousQuerynot TimeSameLocationDayweather")
    print("  User: Predict tomorrow's weather in Beijing")
    ctx.add_prediction("weather", {"location": "Beijing", "date": "2025-10-16"}, {})
    
    print("  User: AfterDay？")
    params = ctx.smart_complete_params("weather", {"date": "2025-10-17"})
    print(f"  ✅ Auto complete: location={params['location']}")
    
    print("  User:  AfterDay？")
    params = ctx.smart_complete_params("weather", {"date": "2025-10-18"})
    print(f"  ✅ Auto complete: location={params['location']}")
    
    print("\n📍 Scenario2: Switchto City")
    print("  User: UphaiDayweather？")
    ctx.add_prediction("weather", {"location": "Uphai", "date": "2025-10-16"}, {})
    
    print("  User: TomorrowDay？")
    params = ctx.smart_complete_params("weather", {"date": "2025-10-16"})
    print(f"  ✅ Auto complete: location={params['location']} (SwitchedtoUphai)")


def example6_multi_domain():
    """Example6: Multi-domainContext"""
    print("\n" + "="*70)
    print("Example6: Multi-domain - not DomainIndependentContext")
    print("="*70)
    
    ctx = ContextManager(session_id="user-006")
    
    print("\n🌤️  DayweatherDomain:")
    ctx.add_prediction("weather", {"location": "Beijing", "date": "2025-10-16"}, {})
    ctx.add_prediction("weather", {"location": "Uphai", "date": "2025-10-17"}, {})
    
    print("\n⚽ body Domain:")
    ctx.add_prediction("sports", {"team1": "Barcelona", "team2": "Real Madrid"}, {})
    ctx.add_prediction("sports", {"team1": "Liverpool", "team2": "Manchester United"}, {})
    
    print("\n📊  DomainHistoricalRecord:")
    for domain, records in ctx.domain_history.items():
        print(f"  {domain}: {len(records)} records")
        for i, record in enumerate(records, 1):
            params = record['params']
            print(f"    {i}. {params}")


def main():
    """RunAllExample"""
    print("="*70)
    print("🔄 ContextShareSystem - UseExample")
    print("="*70)
    
    examples = [
        ("BasicContext", example1_basic_context),
        ("ConversationContext", example2_conversation_context),
        ("ShareContext", example3_shared_context),
        ("ContextPersistence", example4_context_persistence),
        ("SmartComplete", example5_smart_completion),
        ("Multi-domainContext", example6_multi_domain),
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
    print("\n💡 Prompt：")
    print("  - Run python3 smart_predict.py body CompleteContextFunction")
    print("  - Use 'history' Command to viewConversation history")
    print("  - Use 'context' Command to viewwhenBeforeContext")
    print("  - Use 'set <key> <value>' SettingPreference")


if __name__ == "__main__":
    main()

