"""Natural Language Parser - Use LLM to Understand User Intent"""
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from .openai_client import create_openrouter_client


class NLPParser:
    """Natural languageParseer"""
    
    def __init__(self):
        """InitializeParseer"""
        self.client = create_openrouter_client()
        logger.info("NLPParseerInitializeSuccess")
    
    def parse(self, user_input: str) -> Dict[str, Any]:
        """
        Parse user input，Extract prediction intent andParameters
        
        Args:
            user_input: User's natural language input
        
        Returns:
            ParseResultDictionary
        """
        logger.info(f"Parse user input: {user_input}")
        
        prompt = f"""
Please analyze toDownUser input，Extract prediction intent and KeyParameters。

User input：{user_input}

SupportedPrediction domain：
1. sports - Sports match prediction（Need：team1, team2, league）**Only for specific match-ups**
2. weather - Weather prediction（Need：location, date, days_ahead）
3. election - Election prediction（Need：election, region, candidates）
4. general - General prediction（Need：query, topic）- Can predict anything，**Package ChampionshipPrediction、Tournament winneretc**

**Important note**：
- Ifis"Who will winChampionship"、"Championship"、"Winner"etcQuestion，should be classified asClassas**general**instead ofsports
- sportsDomain is only for specific two-team matches（ "BarcelonavsReal Madrid"）
- ChampionshipPrediction、Tournament predictions shouldUsegeneralDomain

Please useJSONFormatReturn：
{{
  "domain": "Prediction domain(sports/weather/election)",
  "params": {{
    // According to domainReturnCorrespondingParameters
  }},
  "confidence": 0.0-1.0,
  "raw_dates": {{
    // IfhaveDateExpression，extract OriginalExpression
    "relative": "TomorrowDay/AfterDay/DownWeeketc",
    "absolute": "2025-10-16etc"
  }}
}}

DateProcessRules：
- "TomorrowDay" -> whenBeforeDate+1Day
- "AfterDay" -> whenBeforeDate+2Day
- "DownWeek " -> DownWeekWeek 
-  bodyDateDirectUse

Example1：
User input："Predict tomorrow's weather in New York"
Return：
{{
  "domain": "weather",
  "params": {{
    "location": "New York",
    "date": "TomorrowDay",
    "days_ahead": 1
  }},
  "confidence": 0.95,
  "raw_dates": {{
    "relative": "TomorrowDay"
  }}
}}

Example2：
User input："Who will win Barcelona vs Real Madrid"
Return：
{{
  "domain": "sports",
  "params": {{
    "team1": "Barcelona",
    "team2": "Real Madrid",
    "league": "La Liga"
  }},
  "confidence": 0.9
}}

Example3：
User input："2024Who will win the US election Trump or Biden"
Return：
{{
  "domain": "election",
  "params": {{
    "election": "2024 US Presidential Election",
    "region": "United States",
    "candidates": ["Donald Trump", "Joe Biden"]
  }},
  "confidence": 0.85
}}

Example4：
User input："Will Bitcoin rise"
Return：
{{
  "domain": "general",
  "params": {{
    "query": "Will Bitcoin rise",
    "topic": "Bitcoin price trend"
  }},
  "confidence": 0.8
}}

Example5：
User input："World Series Champion 2025"
Return：
{{
  "domain": "general",
  "params": {{
    "query": "World Series Champion 2025",
    "topic": "MLB World Series 2025 champion prediction"
  }},
  "confidence": 0.85
}}

Example6：
User input："Who will win2025yearNBAChampionship"
Return：
{{
  "domain": "general",
  "params": {{
    "query": "Who will win2025yearNBAChampionship",
    "topic": "NBA Championship 2025 winner"
  }},
  "confidence": 0.85
}}

Note：
- Ifdoes not belong tosports/weather/election，then classify asClassasgeneral
- generalCan predict anything：Stock、Economic、Technology、Social eventsetc
- **ChampionshipPrediction、Tournament winner predictions should be classified asClassasgeneral，notissports**
- sportsOnly for specific two-team matches

OnlyReturnJSON，No otherInnercontent。
"""
        
        try:
            response = self.client.simple_query(
                query=prompt,
                system_prompt="You are a professional natural language understanding assistant，Good at extracting prediction-related keysKeyInformation。",
                temperature=0.3
            )
            
            logger.debug(f"LLMResponse: {response[:200]}...")
            
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                result = self._process_dates(result)
                
                logger.info(f"ParseSuccess - Domain: {result.get('domain')}, Confidence: {result.get('confidence')}")
                return result
            else:
                logger.error("NoneUnable toResponseextract fromJSON")
                return self._create_error_response("NoneUnable to understand input")
                
        except Exception as e:
            logger.error(f"ParseFailed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_error_response(str(e))
    
    def _process_dates(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """ProcessRelativeDateExpression"""
        if result.get("domain") == "weather" and "params" in result:
            params = result["params"]
            
            today = datetime.now()
            
            date_str = params.get("date", "")
            
            if "TomorrowDay" in date_str or "tomorrow" in date_str.lower():
                actual_date = today + timedelta(days=1)
                params["date"] = actual_date.strftime("%Y-%m-%d")
                logger.info(f"ConvertDate: TomorrowDay -> {params['date']}")
                
            elif "AfterDay" in date_str:
                actual_date = today + timedelta(days=2)
                params["date"] = actual_date.strftime("%Y-%m-%d")
                logger.info(f"ConvertDate: AfterDay -> {params['date']}")
                
            elif " Day" in date_str or "today" in date_str.lower():
                params["date"] = today.strftime("%Y-%m-%d")
                logger.info(f"ConvertDate:  Day -> {params['date']}")
                
            elif "DownWeek" in date_str:
                actual_date = today + timedelta(days=7)
                params["date"] = actual_date.strftime("%Y-%m-%d")
                logger.info(f"ConvertDate: DownWeek -> {params['date']}")
            
            if not params.get("date"):
                params["date"] = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        
        return result
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """CreateErrorResponse"""
        return {
            "error": error_msg,
            "domain": None,
            "params": {},
            "confidence": 0.0
        }


def parse_natural_language(user_input: str) -> Dict[str, Any]:
    """
    QuickFunction：ParseNatural languageInput
    
    Args:
        user_input: User's natural language input
    
    Returns:
        ParseResult
    """
    parser = NLPParser()
    return parser.parse(user_input)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    test_inputs = [
        "Predict tomorrow's weather in New York",
        "Who will win Barcelona vs Real Madrid",
        "2024Who will win the US election Trump or Biden",
        "AfterDayBeijingDayHow's the weather",
        "Liverpool vs Manchester United matchResult",
    ]
    
    parser = NLPParser()
    
    for user_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"Input: {user_input}")
        print('='*60)
        
        result = parser.parse(user_input)
        print(json.dumps(result, indent=2, ensure_ascii=False))

