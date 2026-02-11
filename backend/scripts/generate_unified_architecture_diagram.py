"""
Generate architecture diagram for the unified FinTech forecasting platform.
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import Users
from diagrams.onprem.frontend import React
from diagrams.onprem.container import Docker
from diagrams.programming.language import Python
from diagrams.onprem.database import MongoDB, SQLite
from diagrams.onprem.compute import Server
from diagrams.programming.framework import FastAPI
from diagrams.generic.compute import Rack
from diagrams.onprem.monitoring import Prometheus

def generate_architecture_diagram():
    """Generate the unified architecture diagram."""
    
    with Diagram("Unified FinTech Forecasting Platform Architecture", 
                 filename="docs/unified_architecture", 
                 show=False, 
                 direction="TB"):
        
        # Frontend Layer
        with Cluster("Frontend Layer"):
            react_app = React("React Dashboard\n(FinSight + forecast-pro-dash)")
            users = Users("Users")
            users >> react_app
        
        # API Gateway
        with Cluster("API Layer"):
            fastapi = FastAPI("FastAPI REST API\n/forecast\n/data\n/evaluation\n/portfolio")
            react_app >> Edge(label="HTTP/REST") >> fastapi
        
        # Backend Services
        with Cluster("Backend Services"):
            data_service = Python("Data Ingestion\nService")
            model_service = Python("Multi-Model\nService")
            eval_service = Python("Evaluation\nService")
            portfolio_service = Python("Portfolio\nService")
            adaptive_service = Python("Adaptive Learning\nService")
            
            fastapi >> data_service
            fastapi >> model_service
            fastapi >> eval_service
            fastapi >> portfolio_service
            fastapi >> adaptive_service
        
        # ML Models
        with Cluster("ML Models"):
            traditional = Rack("Traditional\n(ARIMA, VAR,\nExp Smoothing)")
            neural = Rack("Neural\n(LSTM, GRU,\nTransformer)")
            ensemble = Rack("Ensemble\n(Adaptive)")
            
            model_service >> traditional
            model_service >> neural
            model_service >> ensemble
        
        # Data Layer
        with Cluster("Data Persistence"):
            mongodb = MongoDB("MongoDB\n(prices, forecasts,\nmodels, metrics)")
            sqlite = SQLite("SQLite\n(backup, local)")
            
            data_service >> mongodb
            data_service >> sqlite
            model_service >> mongodb
            eval_service >> mongodb
            portfolio_service >> mongodb
        
        # External Data Sources
        with Cluster("External Data"):
            yfinance = Server("Yahoo Finance\nAPI")
            data_service >> Edge(label="Fetch OHLC") >> yfinance
        
        # Monitoring
        with Cluster("Monitoring & Logging"):
            monitoring = Prometheus("Performance\nMonitoring")
            eval_service >> monitoring
            adaptive_service >> monitoring

if __name__ == "__main__":
    generate_architecture_diagram()
    print("Architecture diagram generated at docs/unified_architecture.png")

