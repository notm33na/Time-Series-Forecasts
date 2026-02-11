"""
Utility module for downloading model weights from remote sources.
Supports Hugging Face Hub, Google Drive, and S3.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from ..config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 8192,
    show_progress: bool = True,
) -> Path:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        chunk_size: Size of chunks to download
        show_progress: Whether to show progress bar
    
    Returns:
        Path to downloaded file
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading from {url} to {destination}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if show_progress and total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        
        logger.info(f"Successfully downloaded {destination.name} ({total_size:,} bytes)")
        return destination
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download from {url}: {e}")
        if destination.exists():
            destination.unlink()  # Remove partial download
        raise


def download_from_huggingface(
    repo_id: str,
    filename: str,
    destination: Path,
    token: Optional[str] = None,
) -> Path:
    """
    Download a file from Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (e.g., "username/repo-name")
        filename: Name of the file to download
        destination: Local path to save the file
        token: Hugging Face token (optional, for private repos)
    
    Returns:
        Path to downloaded file
    """
    # Hugging Face Hub URL format
    base_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    logger.info(f"Downloading from Hugging Face Hub: {repo_id}/{filename}")
    
    try:
        response = requests.get(base_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        logger.info(f"Successfully downloaded {filename} from Hugging Face Hub")
        return destination
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download from Hugging Face Hub: {e}")
        if destination.exists():
            destination.unlink()
        raise


def download_from_google_drive(file_id: str, destination: Path) -> Path:
    """
    Download a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        destination: Local path to save the file
    
    Returns:
        Path to downloaded file
    """
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    logger.info(f"Downloading from Google Drive: {file_id}")
    
    # Handle large files that require confirmation
    session = requests.Session()
    response = session.get(url, stream=True, timeout=30)
    
    # Check for virus scan warning
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True, timeout=30)
            break
    
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f:
        if total_size > 0:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    logger.info(f"Successfully downloaded {destination.name} from Google Drive")
    return destination


def download_from_s3(url: str, destination: Path, aws_access_key: Optional[str] = None, aws_secret_key: Optional[str] = None) -> Path:
    """
    Download a file from S3.
    
    Args:
        url: S3 URL (s3://bucket/path or https://bucket.s3.region.amazonaws.com/path)
        destination: Local path to save the file
        aws_access_key: AWS access key (optional)
        aws_secret_key: AWS secret key (optional)
    
    Returns:
        Path to downloaded file
    """
    # Convert s3:// URL to HTTPS if needed
    if url.startswith('s3://'):
        # Parse s3://bucket/path
        match = re.match(r's3://([^/]+)/(.+)', url)
        if match:
            bucket, key = match.groups()
            url = f"https://{bucket}.s3.amazonaws.com/{key}"
        else:
            raise ValueError(f"Invalid S3 URL format: {url}")
    
    logger.info(f"Downloading from S3: {url}")
    
    headers = {}
    if aws_access_key and aws_secret_key:
        # For private S3 buckets, you'd use boto3, but for public buckets, simple GET works
        pass
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        logger.info(f"Successfully downloaded {destination.name} from S3")
        return destination
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download from S3: {e}")
        if destination.exists():
            destination.unlink()
        raise


def ensure_model_file(
    local_path: Path,
    remote_url: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_filename: Optional[str] = None,
    gdrive_id: Optional[str] = None,
    s3_url: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Path:
    """
    Ensure a model file exists locally, downloading from remote if needed.
    
    Tries sources in order:
    1. Local file (if exists, returns immediately)
    2. Hugging Face Hub (if hf_repo and hf_filename provided)
    3. Google Drive (if gdrive_id provided)
    4. S3 (if s3_url provided)
    5. Generic URL (if remote_url provided)
    
    Args:
        local_path: Local path where model should be stored
        remote_url: Generic URL to download from
        hf_repo: Hugging Face repository ID (e.g., "username/repo-name")
        hf_filename: Filename in Hugging Face repo
        gdrive_id: Google Drive file ID
        s3_url: S3 URL
        hf_token: Hugging Face token (for private repos)
    
    Returns:
        Path to local model file
    
    Raises:
        FileNotFoundError: If file doesn't exist locally and all download attempts fail
    """
    # Check if file already exists locally
    if local_path.exists():
        logger.info(f"Model file found locally: {local_path}")
        return local_path
    
    logger.info(f"Model file not found locally: {local_path}")
    logger.info("Attempting to download from remote sources...")
    
    # Try Hugging Face Hub first
    if hf_repo and hf_filename:
        try:
            logger.info(f"Trying Hugging Face Hub: {hf_repo}/{hf_filename}")
            return download_from_huggingface(hf_repo, hf_filename, local_path, token=hf_token)
        except Exception as e:
            logger.warning(f"Hugging Face download failed: {e}")
    
    # Try Google Drive
    if gdrive_id:
        try:
            logger.info(f"Trying Google Drive: {gdrive_id}")
            return download_from_google_drive(gdrive_id, local_path)
        except Exception as e:
            logger.warning(f"Google Drive download failed: {e}")
    
    # Try S3
    if s3_url:
        try:
            logger.info(f"Trying S3: {s3_url}")
            return download_from_s3(s3_url, local_path)
        except Exception as e:
            logger.warning(f"S3 download failed: {e}")
    
    # Try generic URL
    if remote_url:
        try:
            logger.info(f"Trying generic URL: {remote_url}")
            return download_file(remote_url, local_path)
        except Exception as e:
            logger.warning(f"Generic URL download failed: {e}")
    
    # All attempts failed
    raise FileNotFoundError(
        f"Model file not found locally at {local_path} and all download attempts failed. "
        f"Please provide one of: hf_repo+hf_filename, gdrive_id, s3_url, or remote_url."
    )


def parse_model_url(model_path_or_url: str) -> dict:
    """
    Parse a model path or URL to extract source information.
    
    Supports:
    - Local paths: ./models/weights/lstm.h5
    - Hugging Face: hf://username/repo-name/lstm.h5
    - Google Drive: gdrive://FILE_ID
    - S3: s3://bucket/path or https://bucket.s3.amazonaws.com/path
    - Generic URLs: https://example.com/model.h5
    
    Args:
        model_path_or_url: Model path or URL string
    
    Returns:
        dict with parsed information
    """
    result = {
        "local_path": None,
        "remote_url": None,
        "hf_repo": None,
        "hf_filename": None,
        "gdrive_id": None,
        "s3_url": None,
    }
    
    # Hugging Face format: hf://username/repo-name/filename
    if model_path_or_url.startswith("hf://"):
        parts = model_path_or_url[5:].split("/", 2)
        if len(parts) >= 3:
            result["hf_repo"] = f"{parts[0]}/{parts[1]}"
            result["hf_filename"] = parts[2]
        elif len(parts) == 2:
            result["hf_repo"] = parts[0]
            result["hf_filename"] = parts[1]
        return result
    
    # Google Drive format: gdrive://FILE_ID
    if model_path_or_url.startswith("gdrive://"):
        result["gdrive_id"] = model_path_or_url[9:]
        return result
    
    # S3 format: s3://bucket/path
    if model_path_or_url.startswith("s3://"):
        result["s3_url"] = model_path_or_url
        return result
    
    # Check if it's a URL
    parsed = urlparse(model_path_or_url)
    if parsed.scheme in ("http", "https"):
        if "s3.amazonaws.com" in parsed.netloc or "s3." in parsed.netloc:
            result["s3_url"] = model_path_or_url
        else:
            result["remote_url"] = model_path_or_url
        return result
    
    # Assume it's a local path
    result["local_path"] = model_path_or_url
    return result

