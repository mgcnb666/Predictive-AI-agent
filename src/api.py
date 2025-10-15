"""FastAPI Interface - RESTful API Service"""
import os
import sys
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import PredictionAgent
from config.settings import get_settings, Settings



class MatchAnalysisRequest(BaseModel):
    """MatchAnalysisRequest"""
    team1: str = Field(..., description="Home teamName")
    team2: str = Field(..., description="Away teamName")
    league: str = Field(default="Unspecified", description="LeagueName")
    date: Optional[str] = Field(None, description="MatchDate (YYYY-MM-DD)")
    market_odds: Optional[dict] = Field(
        None,
        description="Market odds",
        example={"home": 2.0, "draw": 3.5, "away": 2.5}
    )


class QuickPredictRequest(BaseModel):
    """Quick predictionRequest"""
    team1: str = Field(..., description="Home teamName")
    team2: str = Field(..., description="Away teamName")
    league: str = Field(default="Unspecified", description="LeagueName")
    market_odds: Optional[dict] = Field(None, description="Market odds")


class BatchAnalysisRequest(BaseModel):
    """Batch analysisRequest"""
    matches: List[MatchAnalysisRequest] = Field(
        ...,
        description="MatchList"
    )


class PredictionResponse(BaseModel):
    """PredictionResponse"""
    match_info: dict
    prediction: dict
    analysis: dict
    betting_analysis: dict
    features_summary: dict
    metadata: dict


class QuickPredictResponse(BaseModel):
    """Quick predictionResponse"""
    match: str
    prediction: dict
    recommendation: str
    confidence: float
    key_insight: str


class StatusResponse(BaseModel):
    """StatusResponse"""
    status: str
    bankroll: dict
    timestamp: str



app = FastAPI(
    title="PredictionAI Agent API",
    description="Based onOpenDeepSearchSports event predictionAPI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



prediction_agent: Optional[PredictionAgent] = None



def get_agent() -> PredictionAgent:
    """GetAgentInstance"""
    if prediction_agent is None:
        raise HTTPException(
            status_code=503,
            detail="PredictionAgent Initialize"
        )
    return prediction_agent



@app.on_event("startup")
async def startup_event():
    """Start item - InitializeAgent"""
    global prediction_agent
    
    logger.info("In progressInitializePredictionAI Agent...")
    
    settings = get_settings()
    
    try:
        prediction_agent = PredictionAgent(
            model_name=settings.litellm_model_id,
            search_provider=settings.search_provider,
            reranker=settings.reranker,
            initial_bankroll=10000  
        )
        
        logger.info("PredictionAI AgentInitializeSuccess")
        
    except Exception as e:
        logger.error(f"AgentInitializeFailed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close item"""
    logger.info("ClosePredictionAI Agent...")



@app.get("/")
async def root():
    """Root path"""
    return {
        "name": "PredictionAI Agent API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """HealthCheck"""
    return {
        "status": "healthy",
        "agent_initialized": prediction_agent is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/analyze", response_model=PredictionResponse)
async def analyze_match(
    request: MatchAnalysisRequest,
    agent: PredictionAgent = Depends(get_agent)
):
    """
    CompleteMatchAnalysis
    
    ReturnDetailedPrediction Result、Feature analysis and betting recommendations
    """
    try:
        logger.info(f" toAnalysisRequest: {request.team1} vs {request.team2}")
        
        result = agent.analyze_match(
            team1=request.team1,
            team2=request.team2,
            league=request.league,
            date=request.date,
            market_odds=request.market_odds
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AnalysisFailed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"AnalysisFailed: {str(e)}"
        )


@app.post("/api/v1/quick-predict", response_model=QuickPredictResponse)
async def quick_predict(
    request: QuickPredictRequest,
    agent: PredictionAgent = Depends(get_agent)
):
    """
    Quick prediction
    
    ReturnSimplifyPrediction Result，Suitable for quickQuery
    """
    try:
        logger.info(f" toQuick predictionRequest: {request.team1} vs {request.team2}")
        
        result = agent.quick_predict(
            team1=request.team1,
            team2=request.team2,
            league=request.league,
            market_odds=request.market_odds
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick predictionFailed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"PredictionFailed: {str(e)}"
        )


@app.post("/api/v1/batch-analyze")
async def batch_analyze(
    request: BatchAnalysisRequest,
    agent: PredictionAgent = Depends(get_agent)
):
    """
    Batch analysismultiplefieldMatch
    
    Analyze multiple matches at onceMatch，ReturnResultList
    """
    try:
        logger.info(f" toBatch analysisRequest: {len(request.matches)} fieldMatch")
        
        matches = [match.dict() for match in request.matches]
        results = agent.batch_analyze(matches)
        
        return {
            "total_matches": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch analysisFailed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysisFailed: {str(e)}"
        )


@app.get("/api/v1/status", response_model=StatusResponse)
async def get_status(agent: PredictionAgent = Depends(get_agent)):
    """
    GetAgentStatus
    
    ReturnwhenBeforeAgentRunStatusandFundInformation
    """
    try:
        status = agent.get_agent_status()
        return status
    except Exception as e:
        logger.error(f"GetStatusFailed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"GetStatusFailed: {str(e)}"
        )


@app.get("/api/v1/leagues")
async def get_supported_leagues():
    """
    GetSupportedLeagueList
    
    ReturnCommon footballLeague
    """
    leagues = [
        {"name": "Premier League", "code": "Premier League", "country": "England"},
        {"name": "La Liga", "code": "La Liga", "country": "Spain"},
        {"name": "Bundesliga", "code": "Bundesliga", "country": "Germany"},
        {"name": "Serie A", "code": "Serie A", "country": "Italy"},
        {"name": "Ligue 1", "code": "Ligue 1", "country": "France"},
        {"name": "Champions League", "code": "Champions League", "country": "Europe"},
        {"name": "Europa League", "code": "Europa League", "country": "Europe"},
        {"name": "Chinese Super League", "code": "Chinese Super League", "country": "China"},
    ]
    
    return {
        "leagues": leagues,
        "total": len(leagues)
    }



@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404ErrorProcess"""
    return {
        "error": "Not Found",
        "message": "RequestResourcenotExist",
        "path": str(request.url),
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """500ErrorProcess"""
    logger.error(f"Inner Error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "ServiceerInner Error",
        "timestamp": datetime.now().isoformat()
    }



if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    logger.add(
        settings.log_file,
        rotation="1 day",
        retention="7 days",
        level=settings.log_level
    )
    
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

