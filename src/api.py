"""FastAPI Interface - OpenAPI 3.0 Compliant API Service"""
import os
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import PredictionAgent
from src.universal_agent import UniversalPredictionAgent
from config.settings import get_settings, Settings

# ============================================================================
# Error Response Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail model"""
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Agent not initialized",
                "code": "SERVICE_UNAVAILABLE",
                "details": {"reason": "Prediction agent is still initializing"}
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    path: Optional[str] = Field(None, description="Request path that caused the error")
    timestamp: str = Field(..., description="Error timestamp in ISO format")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "path": "/api/v1/analyze",
                "timestamp": "2024-01-20T10:30:00.000Z"
            }
        }
    )


# ============================================================================
# Request Models
# ============================================================================

class MarketOdds(BaseModel):
    """Market odds for a match"""
    home: float = Field(..., gt=1.0, description="Home win odds")
    draw: Optional[float] = Field(None, gt=1.0, description="Draw odds")
    away: float = Field(..., gt=1.0, description="Away win odds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "home": 2.0,
                "draw": 3.5,
                "away": 2.5
            }
        }
    )


class MatchAnalysisRequest(BaseModel):
    """Match analysis request with full details"""
    team1: str = Field(..., min_length=1, description="Home team name", examples=["Manchester United", "Barcelona"])
    team2: str = Field(..., min_length=1, description="Away team name", examples=["Liverpool", "Real Madrid"])
    league: str = Field(default="Unspecified", description="League name", examples=["Premier League", "La Liga"])
    date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Match date in YYYY-MM-DD format", examples=["2024-10-26"])
    market_odds: Optional[MarketOdds] = Field(None, description="Current market odds for the match")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "team1": "Manchester United",
                "team2": "Liverpool",
                "league": "Premier League",
                "date": "2024-10-26",
                "market_odds": {
                    "home": 2.5,
                    "draw": 3.4,
                    "away": 2.8
                }
            }
        }
    )


class QuickPredictRequest(BaseModel):
    """Quick prediction request with minimal details"""
    team1: str = Field(..., min_length=1, description="Home team name", examples=["Barcelona"])
    team2: str = Field(..., min_length=1, description="Away team name", examples=["Real Madrid"])
    league: str = Field(default="Unspecified", description="League name", examples=["La Liga"])
    market_odds: Optional[MarketOdds] = Field(None, description="Current market odds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "team1": "Barcelona",
                "team2": "Real Madrid",
                "league": "La Liga",
                "market_odds": {
                    "home": 2.2,
                    "draw": 3.3,
                    "away": 3.1
                }
            }
        }
    )


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request for multiple matches"""
    matches: List[MatchAnalysisRequest] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of matches to analyze (max 10)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "matches": [
                    {
                        "team1": "Liverpool",
                        "team2": "Chelsea",
                        "league": "Premier League",
                        "market_odds": {"home": 1.8, "draw": 3.6, "away": 4.2}
                    },
                    {
                        "team1": "Bayern Munich",
                        "team2": "Dortmund",
                        "league": "Bundesliga",
                        "market_odds": {"home": 1.5, "draw": 4.0, "away": 6.0}
                    }
                ]
            }
        }
    )


# ============================================================================
# Response Models
# ============================================================================

class MatchInfo(BaseModel):
    """Match information"""
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    league: str = Field(..., description="League name")
    date: Optional[str] = Field(None, description="Match date")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "home_team": "Manchester United",
                "away_team": "Liverpool",
                "league": "Premier League",
                "date": "2024-10-26"
            }
        }
    )


class PredictionResult(BaseModel):
    """Prediction probabilities and outcome"""
    home_win_probability: float = Field(..., ge=0.0, le=1.0, description="Home win probability (0-1)")
    draw_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Draw probability (0-1)")
    away_win_probability: float = Field(..., ge=0.0, le=1.0, description="Away win probability (0-1)")
    predicted_outcome: str = Field(..., description="Most likely outcome", examples=["home", "draw", "away"])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score (0-1)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "home_win_probability": 0.45,
                "draw_probability": 0.25,
                "away_win_probability": 0.30,
                "predicted_outcome": "home",
                "confidence": 0.78
            }
        }
    )


class AnalysisDetail(BaseModel):
    """Detailed analysis information"""
    summary: str = Field(..., description="Analysis summary")
    key_factors: List[str] = Field(default_factory=list, description="Key factors influencing the prediction")
    strengths: Optional[Dict[str, Any]] = Field(None, description="Team strengths analysis")
    weaknesses: Optional[Dict[str, Any]] = Field(None, description="Team weaknesses analysis")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary": "Manchester United has strong home advantage but Liverpool's recent form is excellent",
                "key_factors": [
                    "Home advantage for Manchester United",
                    "Liverpool's 5-game winning streak",
                    "Head-to-head history favors Liverpool"
                ],
                "strengths": {
                    "home_team": ["Strong defense", "Home record"],
                    "away_team": ["Attacking prowess", "Recent form"]
                }
            }
        }
    )


class BettingRecommendation(BaseModel):
    """Betting recommendation details"""
    recommended_bet: Optional[str] = Field(None, description="Recommended bet type", examples=["home", "away", "draw", "none"])
    stake_percentage: float = Field(..., ge=0.0, le=100.0, description="Recommended stake as percentage of bankroll")
    expected_value: float = Field(..., description="Expected value of the bet")
    kelly_criterion: Optional[float] = Field(None, description="Kelly criterion calculation")
    reasoning: str = Field(..., description="Reasoning for the recommendation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommended_bet": "home",
                "stake_percentage": 2.5,
                "expected_value": 0.15,
                "kelly_criterion": 0.025,
                "reasoning": "Positive expected value with home odds at 2.5"
            }
        }
    )


class FeaturesSummary(BaseModel):
    """Summary of extracted features"""
    total_features: int = Field(..., ge=0, description="Total number of features extracted")
    feature_categories: List[str] = Field(default_factory=list, description="Categories of features")
    key_metrics: Optional[Dict[str, Any]] = Field(None, description="Key metrics and statistics")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_features": 25,
                "feature_categories": ["recent_form", "head_to_head", "home_away_stats", "injuries"],
                "key_metrics": {
                    "home_form_score": 7.5,
                    "away_form_score": 8.2
                }
            }
        }
    )


class Metadata(BaseModel):
    """Response metadata"""
    timestamp: str = Field(..., description="Response generation timestamp")
    processing_time_ms: Optional[float] = Field(None, ge=0, description="Processing time in milliseconds")
    model_version: Optional[str] = Field(None, description="Prediction model version")
    data_sources: Optional[List[str]] = Field(None, description="Data sources used")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-20T10:30:00.000Z",
                "processing_time_ms": 1250.5,
                "model_version": "v1.0.0",
                "data_sources": ["OpenDeepSearch", "Market Data API"]
            }
        }
    )


class PredictionResponse(BaseModel):
    """Complete prediction response with all details"""
    match_info: MatchInfo = Field(..., description="Match information")
    prediction: PredictionResult = Field(..., description="Prediction results")
    analysis: AnalysisDetail = Field(..., description="Detailed analysis")
    betting_analysis: BettingRecommendation = Field(..., description="Betting recommendations")
    features_summary: FeaturesSummary = Field(..., description="Features summary")
    metadata: Metadata = Field(..., description="Response metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match_info": {
                    "home_team": "Manchester United",
                    "away_team": "Liverpool",
                    "league": "Premier League",
                    "date": "2024-10-26"
                },
                "prediction": {
                    "home_win_probability": 0.45,
                    "draw_probability": 0.25,
                    "away_win_probability": 0.30,
                    "predicted_outcome": "home",
                    "confidence": 0.78
                },
                "analysis": {
                    "summary": "Close match with slight home advantage",
                    "key_factors": ["Home advantage", "Recent form"]
                },
                "betting_analysis": {
                    "recommended_bet": "home",
                    "stake_percentage": 2.5,
                    "expected_value": 0.15,
                    "reasoning": "Positive EV bet"
                },
                "features_summary": {
                    "total_features": 25,
                    "feature_categories": ["form", "h2h"]
                },
                "metadata": {
                    "timestamp": "2024-01-20T10:30:00.000Z",
                    "processing_time_ms": 1250.5
                }
            }
        }
    )


class QuickPredictResponse(BaseModel):
    """Quick prediction response with minimal details"""
    match: str = Field(..., description="Match description")
    prediction: PredictionResult = Field(..., description="Prediction results")
    recommendation: str = Field(..., description="Brief betting recommendation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    key_insight: str = Field(..., description="Key insight from the analysis")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match": "Barcelona vs Real Madrid (La Liga)",
                "prediction": {
                    "home_win_probability": 0.42,
                    "draw_probability": 0.28,
                    "away_win_probability": 0.30,
                    "predicted_outcome": "home",
                    "confidence": 0.72
                },
                "recommendation": "Consider home win bet with 2% stake",
                "confidence": 0.72,
                "key_insight": "Barcelona's home record is strong this season"
            }
        }
    )


class Bankroll(BaseModel):
    """Bankroll information"""
    current: float = Field(..., ge=0, description="Current bankroll amount")
    initial: float = Field(..., ge=0, description="Initial bankroll amount")
    total_bets: int = Field(..., ge=0, description="Total number of bets placed")
    winning_bets: int = Field(..., ge=0, description="Number of winning bets")
    roi: Optional[float] = Field(None, description="Return on investment percentage")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current": 10500.0,
                "initial": 10000.0,
                "total_bets": 50,
                "winning_bets": 28,
                "roi": 5.0
            }
        }
    )


class StatusResponse(BaseModel):
    """Agent status response"""
    status: str = Field(..., description="Agent status", examples=["ready", "busy", "initializing"])
    bankroll: Bankroll = Field(..., description="Bankroll information")
    timestamp: str = Field(..., description="Status check timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ready",
                "bankroll": {
                    "current": 10500.0,
                    "initial": 10000.0,
                    "total_bets": 50,
                    "winning_bets": 28,
                    "roi": 5.0
                },
                "timestamp": "2024-01-20T10:30:00.000Z"
            }
        }
    )


class LeagueInfo(BaseModel):
    """League information"""
    name: str = Field(..., description="League name")
    code: str = Field(..., description="League code")
    country: str = Field(..., description="Country")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Premier League",
                "code": "Premier League",
                "country": "England"
            }
        }
    )


class BatchAnalysisResponse(BaseModel):
    """Batch analysis response"""
    total_matches: int = Field(..., ge=0, description="Total number of matches analyzed")
    results: List[Dict[str, Any]] = Field(..., description="Analysis results for each match")
    timestamp: str = Field(..., description="Batch processing timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_matches": 2,
                "results": [
                    {"match": "Liverpool vs Chelsea", "status": "success"},
                    {"match": "Bayern vs Dortmund", "status": "success"}
                ],
                "timestamp": "2024-01-20T10:30:00.000Z"
            }
        }
    )
# ============================================================================
# OpenAPI Tags
# ============================================================================

tags_metadata = [
    {
        "name": "Health",
        "description": "Health check and status endpoints for monitoring the API service",
    },
    {
        "name": "Predictions",
        "description": "Sports match prediction endpoints with AI-powered analysis",
    },
    {
        "name": "Batch Operations",
        "description": "Batch processing endpoints for analyzing multiple matches",
    },
    {
        "name": "Agent Management",
        "description": "Agent status and configuration management",
    },
    {
        "name": "Reference Data",
        "description": "Reference data such as supported leagues and sports",
    },
]


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="PredictionAI Agent API",
    description="""
## Sports Event Prediction API

AI-powered sports prediction API built with OpenDeepSearch and advanced machine learning models.

### Features

* **Complete Match Analysis** - In-depth analysis with betting recommendations
* **Quick Predictions** - Fast predictions for quick decision making
* **Batch Processing** - Analyze multiple matches in a single request
* **Kelly Criterion** - Advanced bankroll management and bet sizing
* **Multi-League Support** - Support for major football leagues worldwide

### Prediction Models

This API combines multiple prediction approaches:
- **LLM-based Analysis** (60% weight) - Using advanced language models for contextual understanding
- **Statistical Models** (40% weight) - Traditional statistical analysis
- **Feature Engineering** - 25+ features including form, head-to-head, and advanced metrics

### Betting Features

- Expected Value (EV) calculations
- Kelly Criterion for optimal bet sizing
- Risk management and bankroll tracking
- Market odds integration

### Data Sources

- OpenDeepSearch for comprehensive sports data
- Real-time market odds
- Historical match statistics
- Team performance metrics
    """,
    version="1.0.0",
    terms_of_service="https://example.com/terms/",
    contact={
        "name": "PredictionAI Support",
        "url": "https://example.com/contact/",
        "email": "support@predictionai.example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")



prediction_agent: Optional[PredictionAgent] = None
universal_agent: Optional[UniversalPredictionAgent] = None



def get_agent() -> PredictionAgent:
    """GetAgentInstance"""
    if prediction_agent is None:
        raise HTTPException(
            status_code=503,
            detail="PredictionAgent Initialize"
        )
    return prediction_agent


def adapt_agent_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert agent response format to API response format"""
    # Convert match_info
    match_info = result.get("match_info", {})
    adapted_match_info = {
        "home_team": match_info.get("team1", match_info.get("home_team", "")),
        "away_team": match_info.get("team2", match_info.get("away_team", "")),
        "league": match_info.get("league", "Unspecified"),
        "date": match_info.get("date")
    }

    # Convert prediction
    prediction = result.get("prediction", {})
    adapted_prediction = {
        "home_win_probability": prediction.get("home_win_probability", 0.0),
        "draw_probability": prediction.get("draw_probability"),
        "away_win_probability": prediction.get("away_win_probability", 0.0),
        "predicted_outcome": _get_predicted_outcome(prediction),
        "confidence": prediction.get("confidence", 0.0)
    }

    # Convert analysis
    analysis = result.get("analysis", {})
    adapted_analysis = {
        "summary": analysis.get("summary", ""),
        "key_factors": analysis.get("key_factors", []),
        "strengths": None,
        "weaknesses": None
    }

    # Convert betting_analysis
    betting = result.get("betting_analysis", {})
    best_bet = betting.get("best_bet", {})
    adapted_betting = {
        "recommended_bet": best_bet.get("outcome") if betting.get("should_bet") else None,
        "stake_percentage": best_bet.get("bet_size_percentage", 0.0) * 100 if best_bet else 0.0,
        "expected_value": best_bet.get("ev", 0.0) if best_bet else 0.0,
        "kelly_criterion": best_bet.get("bet_size_percentage") if best_bet else None,
        "reasoning": betting.get("recommendation", "")
    }

    # Convert features_summary
    features = result.get("features_summary", {})
    adapted_features = {
        "total_features": 25,  # Default value
        "feature_categories": ["recent_form", "head_to_head", "home_away_stats"],
        "key_metrics": {
            "form_differential": features.get("form_differential", 0.0),
            "home_advantage": features.get("home_advantage", 0.0),
            "overall_score": features.get("overall_score", 0.0)
        }
    }

    # Convert metadata
    meta = result.get("metadata", {})
    adapted_metadata = {
        "timestamp": meta.get("analysis_timestamp", meta.get("timestamp", datetime.now().isoformat())),
        "processing_time_ms": meta.get("elapsed_time_seconds", 0.0) * 1000,
        "model_version": "v1.0.0",
        "data_sources": ["OpenDeepSearch"]
    }

    return {
        "match_info": adapted_match_info,
        "prediction": adapted_prediction,
        "analysis": adapted_analysis,
        "betting_analysis": adapted_betting,
        "features_summary": adapted_features,
        "metadata": adapted_metadata
    }


def _get_predicted_outcome(prediction: Dict[str, Any]) -> str:
    """Determine the predicted outcome from probabilities"""
    home_prob = prediction.get("home_win_probability", 0.0)
    draw_prob = prediction.get("draw_probability", 0.0)
    away_prob = prediction.get("away_win_probability", 0.0)

    max_prob = max(home_prob, draw_prob, away_prob)

    if max_prob == home_prob:
        return "home"
    elif max_prob == away_prob:
        return "away"
    else:
        return "draw"


def adapt_quick_predict_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert quick predict response format"""
    prediction = result.get("prediction", {})
    adapted_prediction = {
        "home_win_probability": prediction.get("home_win_probability", 0.0),
        "draw_probability": prediction.get("draw_probability"),
        "away_win_probability": prediction.get("away_win_probability", 0.0),
        "predicted_outcome": _get_predicted_outcome(prediction),
        "confidence": prediction.get("confidence", 0.0)
    }

    return {
        "match": result.get("match", ""),
        "prediction": adapted_prediction,
        "recommendation": result.get("recommendation", ""),
        "confidence": result.get("confidence", 0.0),
        "key_insight": result.get("key_insight", "")
    }


def adapt_status_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert status response format"""
    bankroll = result.get("bankroll", {})
    current = bankroll.get("current", 10000.0)
    initial = bankroll.get("initial", 10000.0)
    daily_pnl = bankroll.get("daily_pnl", 0.0)

    # Calculate ROI
    roi = ((current - initial) / initial * 100) if initial > 0 else 0.0

    adapted_bankroll = {
        "current": current,
        "initial": initial,
        "total_bets": 0,  # Not tracked currently
        "winning_bets": 0,  # Not tracked currently
        "roi": roi
    }

    return {
        "status": result.get("status", "unknown"),
        "bankroll": adapted_bankroll,
        "timestamp": result.get("timestamp", datetime.now().isoformat())
    }



@app.on_event("startup")
async def startup_event():
    """Start item - InitializeAgent"""
    global prediction_agent, universal_agent

    logger.info("Initializing Prediction AI Agents...")

    settings = get_settings()

    try:
        prediction_agent = PredictionAgent(
            model_name=settings.litellm_model_id,
            search_provider=settings.search_provider,
            reranker=settings.reranker,
            initial_bankroll=10000
        )
        logger.info("✅ PredictionAI Agent initialized")

        # Initialize universal agent
        universal_agent = UniversalPredictionAgent(
            model_name=settings.openrouter_model,
            use_opendeepsearch=True
        )
        logger.info("✅ Universal Prediction Agent initialized")

    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close item"""
    logger.info("ClosePredictionAI Agent...")



@app.get(
    "/",
    response_class=FileResponse,
    include_in_schema=False,
)
async def root():
    """Serve the frontend application"""
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "name": "PredictionAI Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "timestamp": datetime.now().isoformat()
    }


@app.get(
    "/api",
    tags=["Health"],
    summary="API Root",
    description="Get basic API information and available endpoints",
    response_description="API information including version and documentation links",
)
async def api_root():
    """
    Get API root information.

    Returns basic information about the API including:
    - API name and version
    - Status
    - Links to documentation
    - Current timestamp
    """
    return {
        "name": "PredictionAI Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "timestamp": datetime.now().isoformat()
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Check if the API service and prediction agent are healthy and ready",
    response_description="Health status including agent initialization state",
)
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
    - Service health status
    - Agent initialization state
    - Current timestamp

    Use this endpoint for:
    - Load balancer health checks
    - Monitoring systems
    - Readiness probes
    """
    return {
        "status": "healthy" if prediction_agent is not None else "initializing",
        "agent_initialized": prediction_agent is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post(
    "/api/v1/analyze",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Complete Match Analysis",
    description="Perform comprehensive match analysis with detailed predictions and betting recommendations",
    response_description="Detailed prediction with analysis, betting recommendations, and features",
    responses={
        200: {
            "description": "Successful analysis",
            "model": PredictionResponse,
        },
        500: {
            "description": "Analysis failed",
            "model": ErrorResponse,
        },
        503: {
            "description": "Service unavailable - Agent not initialized",
            "model": ErrorResponse,
        },
    },
    status_code=status.HTTP_200_OK,
)
async def analyze_match(
    request: MatchAnalysisRequest,
    agent: PredictionAgent = Depends(get_agent)
):
    """
    Perform complete match analysis.

    This endpoint provides comprehensive match analysis including:
    - **Prediction Probabilities**: Win/Draw/Loss probabilities for both teams
    - **Detailed Analysis**: Key factors, strengths, weaknesses
    - **Betting Recommendations**: Expected value, Kelly criterion, stake suggestions
    - **Features Summary**: All extracted features and metrics
    - **Metadata**: Processing time, data sources, model version

    ### How it works:
    1. Collects data from multiple sources using OpenDeepSearch
    2. Extracts 25+ features including recent form, head-to-head, and advanced metrics
    3. Runs hybrid prediction model (60% LLM + 40% statistical)
    4. Calculates expected value and optimal bet sizing using Kelly criterion
    5. Returns comprehensive analysis with actionable recommendations

    ### Use cases:
    - Pre-match analysis for informed betting decisions
    - Deep dive into match dynamics and key factors
    - Understanding team strengths and weaknesses
    - Risk-managed betting with Kelly criterion

    ### Example Request:
    ```json
    {
        "team1": "Manchester United",
        "team2": "Liverpool",
        "league": "Premier League",
        "date": "2024-10-26",
        "market_odds": {
            "home": 2.5,
            "draw": 3.4,
            "away": 2.8
        }
    }
    ```
    """
    try:
        logger.info(f"Analysis request: {request.team1} vs {request.team2}")

        result = agent.analyze_match(
            team1=request.team1,
            team2=request.team2,
            league=request.league,
            date=request.date,
            market_odds=request.market_odds.model_dump() if request.market_odds else None
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        # Adapt the response format to match Pydantic models
        adapted_result = adapt_agent_response(result)
        return adapted_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post(
    "/api/v1/quick-predict",
    response_model=QuickPredictResponse,
    tags=["Predictions"],
    summary="Quick Match Prediction",
    description="Get fast prediction results with minimal details for quick decision making",
    response_description="Simplified prediction with key insights",
    responses={
        200: {
            "description": "Successful prediction",
            "model": QuickPredictResponse,
        },
        500: {
            "description": "Prediction failed",
            "model": ErrorResponse,
        },
        503: {
            "description": "Service unavailable - Agent not initialized",
            "model": ErrorResponse,
        },
    },
    status_code=status.HTTP_200_OK,
)
async def quick_predict(
    request: QuickPredictRequest,
    agent: PredictionAgent = Depends(get_agent)
):
    """
    Get quick match prediction.

    This endpoint provides fast predictions with minimal details, suitable for:
    - Quick decision making
    - Rapid scanning of multiple matches
    - Mobile applications with limited screen space
    - Real-time updates

    Returns:
    - Match description
    - Prediction probabilities
    - Brief recommendation
    - Confidence score
    - One key insight

    **Note**: For comprehensive analysis with betting recommendations and detailed features,
    use the `/api/v1/analyze` endpoint instead.

    ### Example Request:
    ```json
    {
        "team1": "Barcelona",
        "team2": "Real Madrid",
        "league": "La Liga",
        "market_odds": {
            "home": 2.2,
            "draw": 3.3,
            "away": 3.1
        }
    }
    ```
    """
    try:
        logger.info(f"Quick prediction request: {request.team1} vs {request.team2}")

        result = agent.quick_predict(
            team1=request.team1,
            team2=request.team2,
            league=request.league,
            market_odds=request.market_odds.model_dump() if request.market_odds else None
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        # Adapt the response format
        adapted_result = adapt_quick_predict_response(result)
        return adapted_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/api/v1/batch-analyze",
    response_model=BatchAnalysisResponse,
    tags=["Batch Operations"],
    summary="Batch Match Analysis",
    description="Analyze multiple matches in a single request (max 10 matches)",
    response_description="Batch analysis results for all matches",
    responses={
        200: {
            "description": "Successful batch analysis",
            "model": BatchAnalysisResponse,
        },
        500: {
            "description": "Batch analysis failed",
            "model": ErrorResponse,
        },
        503: {
            "description": "Service unavailable - Agent not initialized",
            "model": ErrorResponse,
        },
    },
    status_code=status.HTTP_200_OK,
)
async def batch_analyze(
    request: BatchAnalysisRequest,
    agent: PredictionAgent = Depends(get_agent)
):
    """
    Analyze multiple matches at once.

    This endpoint allows you to analyze up to 10 matches in a single request, useful for:
    - Scanning multiple matches in a league round
    - Comparing predictions across different games
    - Building a betting portfolio
    - Weekend fixture analysis

    **Limitations**:
    - Maximum 10 matches per request
    - Processing time scales with number of matches
    - Individual match failures don't fail the entire batch

    ### Example Request:
    ```json
    {
        "matches": [
            {
                "team1": "Liverpool",
                "team2": "Chelsea",
                "league": "Premier League",
                "market_odds": {"home": 1.8, "draw": 3.6, "away": 4.2}
            },
            {
                "team1": "Bayern Munich",
                "team2": "Dortmund",
                "league": "Bundesliga",
                "market_odds": {"home": 1.5, "draw": 4.0, "away": 6.0}
            }
        ]
    }
    ```

    Returns analysis results for each match with success/error status.
    """
    try:
        logger.info(f"Batch analysis request: {len(request.matches)} matches")

        matches = [match.model_dump() for match in request.matches]
        results = agent.batch_analyze(matches)

        # Adapt each result in the batch
        adapted_results = []
        for result in results:
            if "error" in result:
                adapted_results.append(result)
            else:
                adapted_results.append(adapt_agent_response(result))

        return {
            "total_matches": len(adapted_results),
            "results": adapted_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )


@app.get(
    "/api/v1/status",
    response_model=StatusResponse,
    tags=["Agent Management"],
    summary="Get Agent Status",
    description="Get current status of the prediction agent and bankroll information",
    response_description="Agent status including bankroll, bets, and ROI",
    responses={
        200: {
            "description": "Agent status retrieved successfully",
            "model": StatusResponse,
        },
        500: {
            "description": "Failed to get status",
            "model": ErrorResponse,
        },
        503: {
            "description": "Service unavailable - Agent not initialized",
            "model": ErrorResponse,
        },
    },
    status_code=status.HTTP_200_OK,
)
async def get_status(agent: PredictionAgent = Depends(get_agent)):
    """
    Get current agent status.

    Returns information about:
    - **Agent State**: Current operational status (ready/busy/initializing)
    - **Bankroll**: Current balance, initial balance, profit/loss
    - **Betting History**: Total bets placed, winning bets, win rate
    - **Performance**: Return on investment (ROI) percentage

    Use this endpoint to:
    - Monitor agent health
    - Track betting performance
    - Check available bankroll
    - Calculate profitability

    The bankroll tracking uses Kelly criterion for optimal bet sizing and risk management.
    """
    try:
        agent_status = agent.get_agent_status()
        # Adapt the response format
        adapted_status = adapt_status_response(agent_status)
        return adapted_status
    except Exception as e:
        logger.error(f"Get status failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get status failed: {str(e)}"
        )


@app.get(
    "/api/v1/leagues",
    tags=["Reference Data"],
    summary="Get Supported Leagues",
    description="Get list of all supported football leagues",
    response_description="List of supported leagues with codes and countries",
    responses={
        200: {
            "description": "List of supported leagues",
            "content": {
                "application/json": {
                    "example": {
                        "leagues": [
                            {"name": "Premier League", "code": "Premier League", "country": "England"},
                            {"name": "La Liga", "code": "La Liga", "country": "Spain"}
                        ],
                        "total": 8
                    }
                }
            },
        },
    },
    status_code=status.HTTP_200_OK,
)
async def get_supported_leagues():
    """
    Get supported leagues.

    Returns a list of all football leagues supported by the prediction API.

    Each league includes:
    - **Name**: Full league name
    - **Code**: League code to use in prediction requests
    - **Country**: Country or region

    ### Supported Leagues:
    - **England**: Premier League
    - **Spain**: La Liga
    - **Germany**: Bundesliga
    - **Italy**: Serie A
    - **France**: Ligue 1
    - **Europe**: Champions League, Europa League
    - **China**: Chinese Super League

    Use the league **code** field when making prediction requests to ensure accurate analysis.

    **Note**: While these leagues are officially supported, the API can analyze matches from other
    leagues as well. Specify "Unspecified" as the league if your league is not listed.
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


@app.post(
    "/api/v1/predict",
    tags=["Predictions"],
    summary="Universal Prediction",
    description="AI-powered universal prediction for any domain",
    status_code=status.HTTP_200_OK,
)
async def universal_predict(request: Dict[str, Any]):
    """
    Universal prediction endpoint.

    Supports multiple domains:
    - sports: Sports events prediction
    - weather: Weather forecasting
    - election: Election outcomes
    - general: General purpose predictions

    Request body:
    ```json
    {
        "domain": "general",
        "params": {
            "question": "Will it rain tomorrow in Beijing?"
        },
        "use_search": true
    }
    ```
    """
    try:
        domain = request.get("domain", "general")
        params = request.get("params", {})
        use_search = request.get("use_search", True)

        if universal_agent is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Universal prediction agent not initialized"
            )

        logger.info(f"Universal prediction request - Domain: {domain}, Params: {params}")

        result = universal_agent.predict(
            domain=domain,
            params=params,
            use_search=use_search
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Universal prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )



@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 Not Found errors"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Not Found",
            "message": "The requested resource does not exist",
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 Internal Server Error"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An internal server error occurred",
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )



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

