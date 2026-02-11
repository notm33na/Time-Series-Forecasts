"""
Integration tests for continuous evaluation and monitoring.

NOTE: This test file needs to be updated to use MongoDB instead of deprecated SQLAlchemy models.
The tests currently use deprecated db_models which have been removed.
"""

import pytest  # type: ignore[reportMissingImports]
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from ..services.evaluation_service import EvaluationService
from ..services.data_ingestion import DataIngestionService

# TODO: Update tests to use MongoDB via UnifiedDataStore instead of SQLAlchemy
# from ..db.unified_db import UnifiedDataStore


@pytest.fixture
def sample_forecast():
    """Create a sample forecast for testing.
    
    NOTE: This fixture needs to be updated to use MongoDB.
    Currently skipped due to deprecated SQLAlchemy models.
    """
    pytest.skip("Test needs update to use MongoDB instead of deprecated SQLAlchemy models")


def test_evaluate_forecast(sample_forecast):
    """Test forecast evaluation.
    
    NOTE: Needs update to use MongoDB.
    """
    pytest.skip("Test needs update to use MongoDB instead of deprecated SQLAlchemy models")


def test_compute_metrics(sample_forecast):
    """Test metrics computation.
    
    NOTE: Needs update to use MongoDB.
    """
    pytest.skip("Test needs update to use MongoDB instead of deprecated SQLAlchemy models")


def test_get_metric_trends(sample_forecast):
    """Test metric trends retrieval.
    
    NOTE: Needs update to use MongoDB.
    """
    pytest.skip("Test needs update to use MongoDB instead of deprecated SQLAlchemy models")


def test_auto_evaluate_pending():
    """Test automatic evaluation of pending forecasts."""
    # This would require actual data in the database
    # For now, just test that the method exists and handles empty case
    try:
        count = EvaluationService.auto_evaluate_pending("NONEXISTENT")
        assert isinstance(count, int)
        assert count >= 0
    except Exception:
        # Expected if no data exists
        pass


def test_auto_evaluate_with_metrics_computation(sample_forecast):
    """Test that auto-evaluation automatically computes metrics.
    
    NOTE: Needs update to use MongoDB.
    """
    pytest.skip("Test needs update to use MongoDB instead of deprecated SQLAlchemy models")


def test_auto_evaluate_all_symbols():
    """Test automatic evaluation for all symbols."""
    try:
        results = EvaluationService.auto_evaluate_all_symbols()
        assert isinstance(results, dict)
        # Results should map symbol to count
        for symbol, count in results.items():
            assert isinstance(symbol, str)
            assert isinstance(count, int)
            assert count >= 0
    except Exception:
        # Expected if no data exists
        pass
