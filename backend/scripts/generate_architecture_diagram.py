"""
Generate architecture and data flow diagram using the diagrams package.
"""

from pathlib import Path

try:
    from diagrams import Diagram, Cluster, Edge
    from diagrams.onprem.client import Client
    from diagrams.onprem.database import MongoDB, SQLite
    from diagrams.programming.language import Python
    from diagrams.onprem.compute import Server
    from diagrams.onprem.analytics import Spark
    from diagrams.onprem.monitoring import Grafana

    HAS_DIAGRAMS = True
except ImportError:
    HAS_DIAGRAMS = False
    print("Warning: 'diagrams' package not installed. Install with: pip install diagrams")
    print("Also requires Graphviz: https://graphviz.org/download/")


def generate_diagram(output_path: Path):
    """Generate architecture diagram."""
    if not HAS_DIAGRAMS:
        print("Skipping diagram generation (diagrams package not available)")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Diagram(
        "Adaptive Forecasting System Architecture",
        filename=str(output_path.with_suffix("")),
        show=False,
        direction="LR",
    ):
        client = Client("User / Dashboard")

        with Cluster("API Layer"):
            api = Server("FastAPI\nMain App")

        with Cluster("Services"):
            ingestion = Python("Data\nIngestion")
            model_service = Python("Model\nTraining/\nAdaptation")
            evaluation = Python("Evaluation\n& Metrics")
            portfolio = Python("Portfolio\nManagement")

        with Cluster("Data Layer"):
            db = SQLite("SQLite\nDatabase")
            artifacts = Server("Model\nArtifacts\n(joblib)")

        with Cluster("Models"):
            sgd = Python("SGD\nRegressor")
            ensemble = Python("Ensemble\nForecaster")

        client >> Edge(label="HTTP/REST") >> api

        api >> ingestion
        api >> model_service
        api >> evaluation
        api >> portfolio

        ingestion >> db
        model_service >> db
        model_service >> artifacts
        model_service >> sgd
        model_service >> ensemble
        evaluation >> db
        portfolio >> db

    print(f"✓ Architecture diagram generated: {output_path}")
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directories to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from backend.config import get_paths

    paths = get_paths()
    diagram_path = paths.diagrams_dir / "architecture"
    generate_diagram(diagram_path)

