"""Domain-specific predictionModule"""
from .sports import SportsPredictionDomain
from .weather import WeatherPredictionDomain
from .election import ElectionPredictionDomain

__all__ = [
    'SportsPredictionDomain',
    'WeatherPredictionDomain',
    'ElectionPredictionDomain',
]

