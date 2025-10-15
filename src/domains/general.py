"""general predictionDomain - Supports arbitraryTypePrediction"""
from typing import Dict, Any, List
from loguru import logger


class GeneralPredictionDomain:
    """general predictionDomain - Can predict anything"""
    
    domain_name = "general"
    
    @staticmethod
    def get_search_queries(params: Dict[str, Any]) -> List[str]:
        """GenerateSearchQuery"""
        from datetime import datetime
        
        query = params.get("query", "")
        topic = params.get("topic", "")
        
        current_date = datetime.now().strftime("%Y-%m")
        current_year = datetime.now().year
        
        is_championship = any(keyword in query.lower() for keyword in [
            'champion', 'championship', 'winner', 'trophy', 'title', 
            'Championship', 'Championship', 'world series', 'nba finals', 'super bowl'
        ])
        
        if is_championship:
            queries = [
                f"{topic or query} odds {current_year}",
                f"{topic or query} predictions {current_year}",
                f"{topic or query} favorites {current_year}",
                f"{topic or query} contenders analysis {current_year}",
                f"{topic or query} betting odds latest",
                f"{topic or query} expert predictions {current_date}",
            ]
        else:
            queries = [
                f"{topic or query} {current_year} latest update",
                f"{topic or query} {current_date} recent news",
                f"{topic or query} prediction {current_year}",
                f"{topic or query} forecast latest analysis",
                f"{topic or query} current trends {current_year}",
                f"{topic or query} latest data {current_date}",
            ]
        
        return queries
    
    @staticmethod
    def get_system_prompt() -> str:
        """GetSystem promptWord"""
        return """You are a versatile prediction analyst。
Can analyze and predict issues in any domain，PackageIncluding but not limited to：
- Economic trends
- Technology development
- Social events
- Natural phenomena
- Character behavior
- Market trends

When analyzing, should：
1. Based on haveDataand trends
2. Consider multipleProbability
3. Provide reasonableProbability
4. DescriptionUncertainty factors
Please give objective、rational prediction。"""
    
    @staticmethod
    def get_prediction_prompt(data: Dict[str, Any], params: Dict[str, Any]) -> str:
        """GeneratePrediction prompt"""
        from datetime import datetime
        
        query = params.get("query", "UnknownQuestion")
        topic = params.get("topic", query)
        
        current_datetime = datetime.now().strftime("%Yyear%m %d ")
        current_year = datetime.now().year
        current_month = datetime.now().strftime("%Y-%m")
        
        is_championship = any(keyword in query.lower() for keyword in [
            'champion', 'championship', 'winner', 'trophy', 'title', 
            'Championship', 'Championship', 'world series', 'nba finals', 'super bowl'
        ])
        
        if is_championship:
            return f"""
**whenBeforeDate：{current_datetime}**

Based onDownDataPrediction：{query}

SearchData：
{data}

**This is aChampionship/Tournament prediction**，Please identify and analyze all possible candidates（Team/Contender）。

Please useJSONFormatReturnPrediction：

{{
  "prediction": "PredictionChampionship/Winner",
  "probability": 0.0-1.0,
  "confidence": 0.0-1.0,
  "data_date": "DataLatestDate（ 2025-10）",
  "top_contenders": {{
    "Candidates1": 0.35,
    "Candidates2": 0.25,
    "Candidates3": 0.20,
    "Candidates4": 0.12,
    "Other": 0.08
  }},
  "analysis": "Detailed analysis of each major candidate's strengths and weaknesses",
  "factors": ["Key factors1", "Key factors2", "Key factors3"],
  "scenarios": {{
    "best_case": "most CandidatesAdvantageScenario",
    "likely_case": "Most likelyChampionshipandReason",
    "dark_horse": "Dark horseCandidates"
  }},
  "risks": ["Risk1", "Risk2"],
  "timeline": "Championshipmatch/match TimeRange",
  "data_quality": "DataQuality assessment"
}}

Requirement：
1. prediction is the most likely winner
2. probability is the winner'sWinProbability
3. top_contenders List at least3-5  needCandidatesAnd itsProbability
4. top_contenders  AllProbabilitysum should be1.0
5. Based on{current_year}yearLatestOdds、Rank、Expert prediction
6. OnlyReturnJSON，No otherInnercontent
"""
        else:
            return f"""
**whenBeforeDate：{current_datetime}**

Based onDownDataPrediction：{query}

SearchData：
{data}

**Important note**：
1. whenBeforeis {current_datetime}，Please prioritizeUse {current_year} yearLatestData
2. IfSearchData Package not TimeInformation， TomorrowAccurate annotationDataDate
3. Ignore outdatedInformation，Focus on recent monthsData
4. IfDatanotNew enough，Please in the analysisDescription

Please useJSONFormatReturnPrediction：

{{
  "prediction": " needPrediction result（Tomorrow Based onLatestData）",
  "probability": 0.0-1.0,
  "confidence": 0.0-1.0,
  "data_date": "DataLatestDate（ 2025-10）",
  "analysis": "Detailed analysis（Based on{current_year}yearLatestData）",
  "factors": ["Factors1", "Factors2", "Factors3"],
  "scenarios": {{
    "best_case": "Best case",
    "likely_case": "Most likely scenario",
    "worst_case": "Worst case"
  }},
  "risks": ["Risk1", "Risk2"],
  "timeline": "PredictionTimeRange",
  "data_quality": "DataQuality assessment（WhetherLatest、WhetherSufficient）"
}}

Requirement：
1. prediction needTomorrowEnsure havebody，Based on{current_year}yearLatestData
2. probability isPrediction resultpossibility（0-1）
3. confidence is your predictionConfidence（0-1）
4. data_date LabelSearchDataLatestDate
5. Consider multiplesituation 
6. data_quality EvaluationDataWhethernew and accurate enough
7. OnlyReturnJSON，No otherInnercontent
"""
    
    @staticmethod
    def format_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """FormatPrediction result"""
        return {
            "domain": "general",
            "prediction_type": "general_forecast",
            "result": prediction.get("prediction", ""),
            "probability": prediction.get("probability", 0),
            "confidence": prediction.get("confidence", 0),
            "data_date": prediction.get("data_date", "Unknown"),
            "data_quality": prediction.get("data_quality", ""),
            "top_contenders": prediction.get("top_contenders", {}),  
            "analysis": prediction.get("analysis", ""),
            "factors": prediction.get("factors", []),
            "scenarios": prediction.get("scenarios", {}),
            "risks": prediction.get("risks", []),
            "timeline": prediction.get("timeline", ""),
        }

