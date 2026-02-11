"""
Main FastAPI application with all routers and dashboard integration.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import data, evaluation, models, portfolio, forecast, retraining
from .config import get_settings
from .db import get_mongo_db
from .utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    import asyncio
    from .services.evaluation_service import EvaluationService
    
    # Startup: connect to MongoDB
    logger.info("Starting application...")
    try:
        mongo_db = get_mongo_db()
        if mongo_db is not None:
            logger.info(f"Connected to MongoDB: {settings.mongo_db}")
        else:
            raise RuntimeError("Failed to connect to MongoDB")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise
    
    # Start continuous evaluation background task
    continuous_eval_task = None
    try:
        async def continuous_evaluation_loop():
            """Background task for continuous evaluation."""
            check_interval = getattr(settings, 'continuous_eval_interval_minutes', 15)  # Default: 15 minutes
            while True:
                try:
                    await asyncio.sleep(check_interval * 60)  # Convert minutes to seconds
                    logger.debug("Running continuous evaluation check...")
                    results = EvaluationService.auto_evaluate_all_symbols()
                    if results:
                        logger.info(f"Auto-evaluated forecasts: {results}")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in continuous evaluation loop: {e}", exc_info=True)
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
        
        continuous_eval_task = asyncio.create_task(continuous_evaluation_loop())
        logger.info("Continuous evaluation background task started")
    except Exception as e:
        logger.warning(f"Could not start continuous evaluation task: {e}")
    
    # Start scheduled retraining service (optional, can be started via API)
    try:
        from .services.scheduled_retraining import get_retraining_service
        retraining_service = get_retraining_service()
        # Auto-start if enabled via environment variable
        auto_start_retraining = getattr(settings, 'auto_start_retraining', False)
        if auto_start_retraining:
            await retraining_service.start()
            logger.info("Scheduled retraining service auto-started")
        else:
            logger.info("Scheduled retraining service available (start via /api/retraining/start)")
    except Exception as e:
        logger.warning(f"Could not initialize retraining service: {e}")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    
    # Cancel continuous evaluation task
    if continuous_eval_task:
        continuous_eval_task.cancel()
        try:
            await continuous_eval_task
        except asyncio.CancelledError:
            pass
        logger.info("Continuous evaluation task stopped")
    
    # Stop retraining service
    try:
        from .services.scheduled_retraining import get_retraining_service
        retraining_service = get_retraining_service()
        await retraining_service.stop()
    except Exception:
        pass


app = FastAPI(
    title="Adaptive Forecasting System",
    description="Continuous learning forecasting system with portfolio management",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router)
app.include_router(models.router)
app.include_router(forecast.router)  # Enhanced forecast API
app.include_router(evaluation.router)
app.include_router(portfolio.router)
app.include_router(retraining.router)


@app.get("/")
async def root():
    """Root endpoint."""
    # Collect all registered routes
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else []
            })
    
    return {
        "message": "Adaptive Forecasting System API",
        "version": "1.0.0",
        "endpoints": {
            "data": "/api/data",
            "models": "/api/models",
            "forecast": "/api/forecast",
            "evaluation": "/api/evaluation",
            "portfolio": "/api/portfolio",
            "dashboard": "/dashboard",
        },
        "all_routes": routes[:20],  # Show first 20 routes
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Mount dashboard (will be created separately)
try:
    from .dashboard_enhanced import create_dashboard
    from starlette.middleware.wsgi import WSGIMiddleware

    dashboard_app = create_dashboard()
    app.mount("/dashboard", WSGIMiddleware(dashboard_app.server))
    logger.info("Dashboard mounted at /dashboard")
except ImportError as e:
    logger.warning(f"Enhanced dashboard not available, trying basic dashboard: {e}")
    try:
        from .dashboard import create_dashboard
        from starlette.middleware.wsgi import WSGIMiddleware
        dashboard_app = create_dashboard()
        app.mount("/dashboard", WSGIMiddleware(dashboard_app.server))
        logger.info("Basic dashboard mounted at /dashboard")
    except ImportError as e2:
        logger.warning(f"Dashboard not available: {e2}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

