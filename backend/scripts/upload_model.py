"""
Utility script to upload trained model weights to Hugging Face Hub.

Supports:
- Sklearn models (.pkl files)
- Keras models (.h5 weights + .pkl metadata)
- Automatic detection of model type
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import HfApi, login, create_repo
    from huggingface_hub.utils import HfHubHTTPError
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

from ..config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def upload_model(
    model_path: Path,
    repo_id: str,
    commit_message: str = "Upload model",
    token: Optional[str] = None,
    private: bool = False,
    create_repo_if_not_exists: bool = True,
) -> str:
    """
    Upload a model to Hugging Face Hub.
    
    Args:
        model_path: Path to model file (.pkl for sklearn, .h5 for keras)
        repo_id: Hugging Face repository ID (e.g., "username/repo-name")
        commit_message: Commit message for the upload
        token: Hugging Face token (optional, will use environment or prompt)
        private: Whether to create a private repository
        create_repo_if_not_exists: Create repository if it doesn't exist
    
    Returns:
        URL of the uploaded model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model type cannot be determined
    """
    if not HAS_HF_HUB:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Uploading model from {model_path} to {repo_id}")
    
    # Initialize HF API
    api = HfApi()
    
    # Login if token provided
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    else:
        # Try to login with environment token or saved token
        try:
            # Check for HF_TOKEN environment variable first
            import os
            if os.getenv("HF_TOKEN"):
                login(token=os.getenv("HF_TOKEN"))
                logger.info("Logged in to Hugging Face Hub (using HF_TOKEN)")
            else:
                login()  # Try saved token
                logger.info("Logged in to Hugging Face Hub (using saved token)")
        except Exception as e:
            logger.warning(f"Could not auto-login: {e}")
            logger.info("You may need to run: huggingface-cli login")
            logger.info("Or set HF_TOKEN environment variable")
    
    # Create repository if needed
    if create_repo_if_not_exists:
        try:
            create_repo(repo_id=repo_id, private=private, exist_ok=True)
            logger.info(f"Repository {repo_id} ready")
        except HfHubHTTPError as e:
            if "already exists" in str(e).lower():
                logger.info(f"Repository {repo_id} already exists")
            else:
                raise
    
    # Determine model type and upload files
    uploaded_files = []
    
    if model_path.suffix == '.pkl':
        # Sklearn model - upload single .pkl file
        logger.info(f"Uploading sklearn model: {model_path.name}")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_path.name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        uploaded_files.append(model_path.name)
        logger.info(f"✓ Uploaded {model_path.name}")
        
    elif model_path.suffix == '.h5':
        # Keras model - upload both .h5 weights and .pkl metadata
        weights_path = model_path
        metadata_path = model_path.with_suffix('.pkl')
        
        logger.info(f"Uploading Keras model weights: {weights_path.name}")
        api.upload_file(
            path_or_fileobj=str(weights_path),
            path_in_repo=weights_path.name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        uploaded_files.append(weights_path.name)
        logger.info(f"✓ Uploaded {weights_path.name}")
        
        if metadata_path.exists():
            logger.info(f"Uploading model metadata: {metadata_path.name}")
            api.upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo=metadata_path.name,
                repo_id=repo_id,
                commit_message=commit_message,
            )
            uploaded_files.append(metadata_path.name)
            logger.info(f"✓ Uploaded {metadata_path.name}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}. Uploading weights only.")
    
    else:
        # Try to upload as-is
        logger.info(f"Uploading file: {model_path.name}")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_path.name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        uploaded_files.append(model_path.name)
        logger.info(f"✓ Uploaded {model_path.name}")
    
    # Construct model URL
    model_url = f"https://huggingface.co/{repo_id}/resolve/main/{uploaded_files[0]}"
    
    logger.info(f"Successfully uploaded model to {repo_id}")
    logger.info(f"Model URL: {model_url}")
    logger.info(f"Uploaded files: {', '.join(uploaded_files)}")
    
    return model_url


def upload_model_directory(
    model_dir: Path,
    repo_id: str,
    commit_message: str = "Upload model",
    token: Optional[str] = None,
    private: bool = False,
    pattern: str = "*",
) -> list[str]:
    """
    Upload all model files from a directory to Hugging Face Hub.
    
    Args:
        model_dir: Directory containing model files
        repo_id: Hugging Face repository ID
        commit_message: Commit message
        token: Hugging Face token
        private: Whether repository is private
        pattern: Glob pattern to match files (default: "*")
    
    Returns:
        List of uploaded file URLs
    """
    if not HAS_HF_HUB:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
    
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Directory not found: {model_dir}")
    
    logger.info(f"Uploading models from {model_dir} to {repo_id}")
    
    api = HfApi()
    
    if token:
        login(token=token)
    else:
        try:
            login()
        except Exception:
            logger.warning("Could not auto-login. You may need to run: huggingface-cli login")
    
    # Find all model files
    model_files = list(model_dir.glob(pattern))
    model_files = [f for f in model_files if f.is_file()]
    
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} matching pattern {pattern}")
    
    logger.info(f"Found {len(model_files)} files to upload")
    
    # Upload all files
    uploaded_urls = []
    for model_file in model_files:
        try:
            api.upload_file(
                path_or_fileobj=str(model_file),
                path_in_repo=model_file.name,
                repo_id=repo_id,
                commit_message=commit_message,
            )
            file_url = f"https://huggingface.co/{repo_id}/resolve/main/{model_file.name}"
            uploaded_urls.append(file_url)
            logger.info(f"✓ Uploaded {model_file.name}")
        except Exception as e:
            logger.error(f"Failed to upload {model_file.name}: {e}")
    
    logger.info(f"Successfully uploaded {len(uploaded_urls)} files to {repo_id}")
    return uploaded_urls


def main():
    """Command-line interface for uploading models."""
    parser = argparse.ArgumentParser(
        description="Upload trained model weights to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a single model
  python -m backend.scripts.upload_model \\
    --model-path artifacts/AAPL_arima_20240101_120000.pkl \\
    --repo-id username/forecast-models \\
    --commit-message "Upload ARIMA model for AAPL"
  
  # Upload a Keras model (will upload both .h5 and .pkl)
  python -m backend.scripts.upload_model \\
    --model-path artifacts/AAPL_lstm_20240101_120000.h5 \\
    --repo-id username/forecast-models \\
    --commit-message "Upload LSTM model for AAPL"
  
  # Upload all models from a directory
  python -m backend.scripts.upload_model \\
    --model-dir artifacts \\
    --repo-id username/forecast-models \\
    --commit-message "Upload all models"
  
  # Upload with token
  python -m backend.scripts.upload_model \\
    --model-path artifacts/model.pkl \\
    --repo-id username/forecast-models \\
    --token YOUR_HF_TOKEN
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model file (.pkl for sklearn, .h5 for keras)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing model files (uploads all matching files)"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/repo-name')"
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model",
        help="Commit message for the upload"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN environment variable)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern for directory uploads (default: '*')"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_path and not args.model_dir:
        parser.error("Either --model-path or --model-dir must be provided")
    
    if args.model_path and args.model_dir:
        parser.error("Cannot specify both --model-path and --model-dir")
    
    try:
        if args.model_path:
            # Upload single model
            model_path = Path(args.model_path)
            url = upload_model(
                model_path=model_path,
                repo_id=args.repo_id,
                commit_message=args.commit_message,
                token=args.token,
                private=args.private,
            )
            print(f"\n✓ Success! Model uploaded to: {url}")
        else:
            # Upload directory
            model_dir = Path(args.model_dir)
            urls = upload_model_directory(
                model_dir=model_dir,
                repo_id=args.repo_id,
                commit_message=args.commit_message,
                token=args.token,
                private=args.private,
                pattern=args.pattern,
            )
            print(f"\n✓ Success! Uploaded {len(urls)} files")
            for url in urls:
                print(f"  - {url}")
    
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        print(f"\n✗ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

