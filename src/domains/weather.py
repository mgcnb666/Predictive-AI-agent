"""weatherPrediction domain"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger


class WeatherPredictionDomain:
    """weatherPrediction domain"""
    
    domain_name = "weather"
    
    @staticmethod
    def get_search_queries(params: Dict[str, Any]) -> List[str]:
        """GenerateSearchQuery"""
        from datetime import datetime
        
        location = params.get("location")
        date = params.get("date", "")
        days_ahead = params.get("days_ahead", 7)
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        queries = [
            f"{location} weather forecast {date} latest update",
            f"{location} weather prediction next {days_ahead} days current",
            f"{location} real-time weather forecast {date}",
            f"{location} weather trends analysis latest",
            f"{location} meteorological data {current_date}",
            f"weather models prediction {location} updated",
        ]
        
        event = params.get("event")
        if event:
            queries.append(f"{location} weather forecast for {event} latest")
        
        return queries
    
    @staticmethod
    def get_system_prompt() -> str:
        """GetSystem promptWord"""
        return """You are a professional meteorological analyst。
AnalysisweatherWhen considering：
1. Historicalweather Dataand trends
2. whenBeforeweather entriesitem
3. SeasonalFactors
4. Geographic location impact
5. Meteorological model prediction
6. ExtremeweatherProbability
 Based onScienceDataProvide reasonable prediction。"""
    
    @staticmethod
    def format_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """FormatPrediction Result"""
        return {
            "domain": "weather",
            "prediction_type": "weather_forecast",
            "forecast": {
                "temperature_range": prediction.get("temperature", {}),
                "precipitation_prob": prediction.get("precipitation_prob", 0),
                "weather_condition": prediction.get("condition", "unknown"),
                "wind_speed": prediction.get("wind_speed", {}),
                "humidity": prediction.get("humidity", 0),
            },
            "confidence": prediction.get("confidence", 0),
            "analysis": prediction.get("analysis", ""),
            "key_factors": prediction.get("key_factors", []),
            "warnings": prediction.get("warnings", []),
        }
    
    @staticmethod
    def get_prediction_prompt(data: Dict[str, Any], params: Dict[str, Any]) -> str:
        """GeneratePrediction prompt"""
        location = params.get("location")
        date = params.get("date", "Unspecified")
        
        return f"""
Based onDownDataPrediction {location}   {date} Dayweathersituation ：

Data：{data}

Please useJSONFormatReturnPrediction：
{{
  "temperature": {{
    "high": 0,
    "low": 0,
    "unit": "celsius"
  }},
  "precipitation_prob": 0.0,
  "condition": "sunny/Cloudy/rain/snowetc",
  "wind_speed": {{
    "speed": 0,
    "unit": "km/h"
  }},
  "humidity": 0.0,
  "confidence": 0.0,
  "analysis": "Detailed analysis",
  "key_factors": ["Factors1", "Factors2"],
  "warnings": ["Warning1"]
}}
"""

