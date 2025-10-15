"""Election PredictionDomain"""
from typing import Dict, Any, List
from loguru import logger


class ElectionPredictionDomain:
    """Election PredictionDomain"""
    
    domain_name = "election"
    
    @staticmethod
    def get_search_queries(params: Dict[str, Any]) -> List[str]:
        """GenerateSearchQuery"""
        election = params.get("election")
        region = params.get("region", "")
        candidates = params.get("candidates") or []  
        
        queries = [
            f"{election} {region} all candidates list",
            f"{election} {region} polls latest all candidates",
            f"{election} {region} voting intention survey",
            f"{election} {region} election forecast all runners",
            f"{election} {region} candidates comparison full list",
            f"{election} {region} demographic analysis",
            f"{election} {region} historical voting patterns",
        ]
        
        if candidates and isinstance(candidates, list):
            for candidate in candidates:
                queries.extend([
                    f"{candidate} approval rating {region}",
                    f"{candidate} campaign strategy {election}",
                    f"{candidate} policy positions {election}",
                ])
        
        return queries
    
    @staticmethod
    def get_system_prompt() -> str:
        """GetSystem promptWord"""
        return """You are a professional election analyst and political scientist。
When analyzing elections, consider：
1. LatestPollDataand trends
2. HistoricalElection mode
3. PopulationStatistics Factors
4. EconomicMetric
5. CandidateSupport rateandImage
6.  KeyIssue impact
7. RegionDifference
8. Voter turnoutPrediction
 Based onObjectiveDataPerformAnalysis，Avoid political bias。"""
    
    @staticmethod
    def format_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """FormatPrediction Result"""
        return {
            "domain": "election",
            "prediction_type": "election_outcome",
            "predictions": prediction.get("candidate_probabilities", {}),
            "vote_share": prediction.get("vote_share", {}),
            "total_candidates": prediction.get("total_candidates", 0),
            "main_contenders": prediction.get("main_contenders", []),
            "confidence": prediction.get("confidence", 0),
            "swing_factors": prediction.get("swing_factors", []),
            "analysis": prediction.get("analysis", ""),
            "key_regions": prediction.get("key_regions", []),
            "uncertainties": prediction.get("uncertainties", []),
        }
    
    @staticmethod
    def get_prediction_prompt(data: Dict[str, Any], params: Dict[str, Any]) -> str:
        """GeneratePrediction prompt"""
        election = params.get("election")
        region = params.get("region")
        candidates = params.get("candidates") or []  
        
        if candidates and isinstance(candidates, list):
            candidates_str = ", ".join(candidates)
            candidates_info = f" needCandidate：{candidates_str}"
        else:
            candidates_info = "CandidateInformationNeedfromSearchDataextract from"
        
        return f"""
Based onDownDataPrediction {region} {election} Result：

{candidates_info}

SearchData：
{data}

 AnalysisElectionResult，withJSONFormatReturnPrediction。

**Important note**：
 1. Please identify **all candidates** from search data, not just focus on the top 2
2. as**Each Candidate**All giveWinProbability
3. Even minor partiesCandidateor independentCandidate needPackage 

ReturnFormat：
{{
  "candidate_probabilities": {{
    "Candidate1Name": 0.45,
    "Candidate2Name": 0.35,
    "Candidate3Name": 0.10,
    "Candidate4Name": 0.05,
    "Candidate5Name": 0.03,
    "OtherCandidate": 0.02
  }},
  "vote_share": {{
    "Candidate1Name": 0.45,
    "Candidate2Name": 0.35,
    "Candidate3Name": 0.10,
    "Candidate4Name": 0.05,
    "Candidate5Name": 0.03,
    "Other": 0.02
  }},
  "total_candidates": 6,
  "main_contenders": ["Candidate1", "Candidate2"],
  "confidence": 0.75,
  "swing_factors": ["Key factors1", "Key factors2", "Key factors3"],
  "analysis": "Detailed analysisAllCandidateCampaign performance、PollData、SupportBasicetc",
  "key_regions": [" KeyRegion1", " KeyRegion2"],
  "uncertainties": ["Uncertainty factors1", "Uncertainty factors2"]
}}

Requirement：
1. **Must list allSearchDataappears inCandidate**，At least3-5 
2. candidate_probabilities isWinProbability，AllValue andMustas1.0
3. vote_share isExpectedVote share，AllValuesum should be1.0
4. total_candidates isrecognize toCandidateTotal
 5. main_contenders should be the main competitors (top 2-3)
6. IfDataNot sufficient to identify allCandidate，At leastneedPackage 3 withUp
7. Based onPoll、ExpertAnalysis、HistoricalDataObjectivePrediction
8. OnlyReturnJSON，No otherInnercontent
"""

