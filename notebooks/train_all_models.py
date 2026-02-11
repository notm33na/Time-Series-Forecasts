"""
Batch training script to train all models for multiple symbols locally.

This script trains:
- Adaptive Forecaster
- ARIMA
- LSTM
- GRU
- Transformer
- Exponential Smoothing

For all specified symbols with maximum time period and 50 epochs for neural models.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Default symbols to train
DEFAULT_SYMBOLS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Google
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "META",   # Meta
    "NVDA",   # NVIDIA
    "JPM",    # JPMorgan
    "V",      # Visa
    "JNJ",    # Johnson & Johnson
]

# Model training commands
MODELS = {
    "adaptive_forecaster": {
        "script": "train_adaptive_forecaster.py",
        "args": ["--allow-local-training", "--skip-db"],
        "epochs": None,  # Not applicable
    },
    "arima": {
        "script": "train_arima.py",
        "args": [],
        "epochs": None,  # Not applicable
    },
    "lstm": {
        "script": "train_lstm.py",
        "args": ["--epochs", "5", "--batch-size", "16"],  # Updated to match i221500_A02_nlp
        "epochs": 5,  # Updated to match i221500_A02_nlp
    },
    "gru": {
        "script": "train_gru.py",
        "args": ["--epochs", "50"],
        "epochs": 50,
    },
    "transformer": {
        "script": "train_transformer.py",
        "args": ["--epochs", "50"],
        "epochs": 50,
    },
    "exponential_smoothing": {
        "script": "train_exponential_smoothing.py",
        "args": [],
        "epochs": None,  # Not applicable
    },
}


def train_model(symbol: str, model_name: str, model_config: dict) -> bool:
    """
    Train a single model for a symbol.
    
    Args:
        symbol: Stock symbol
        model_name: Name of the model
        model_config: Model configuration dict
        
    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / model_config["script"]
    
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_path}")
        print(f"   Skipping {model_name.upper()} for {symbol}")
        return False
    
    cmd = [
        sys.executable,
        str(script_path),
        "--symbol", symbol,
    ] + model_config["args"]
    
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} for {symbol}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run with real-time output but capture stderr for error reporting
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=True,
            timeout=3600  # 1 hour timeout per model
        )
        
        print(f"\n✅ {model_name.upper()} training completed for {symbol}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n❌ {model_name.upper()} training timed out for {symbol} (>1 hour)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_name.upper()} training failed for {symbol} (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"\n❌ Python executable not found: {sys.executable}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error training {model_name.upper()} for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_all_models(
    symbols: list = None,
    models: list = None,
    max_period: str = "max"
):
    """
    Train all models for all symbols.
    
    Args:
        symbols: List of symbols to train (default: DEFAULT_SYMBOLS)
        models: List of model names to train (default: all models)
        max_period: Time period for data (default: "max" for maximum available)
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    if models is None:
        models = list(MODELS.keys())
    
    # Filter to only existing models
    available_models = {k: v for k, v in MODELS.items() if k in models}
    
    # Verify all scripts exist before starting
    missing_scripts = []
    for model_name, model_config in available_models.items():
        script_path = Path(__file__).parent / model_config["script"]
        if not script_path.exists():
            missing_scripts.append((model_name, script_path))
    
    if missing_scripts:
        print("❌ ERROR: Some training scripts are missing:")
        for model_name, script_path in missing_scripts:
            print(f"   - {model_name}: {script_path}")
        print("\nPlease ensure all training scripts exist before running batch training.")
        return {}
    
    print("="*70)
    print("BATCH TRAINING - ALL MODELS")
    print("="*70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Models: {', '.join(available_models.keys())}")
    print(f"Period: {max_period}")
    print(f"Epochs: 50 (for neural network models)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    results = {}
    total_tasks = len(symbols) * len(available_models)
    completed = 0
    failed = 0
    
    for symbol in symbols:
        results[symbol] = {}
        for model_name, model_config in available_models.items():
            success = train_model(symbol, model_name, model_config)
            results[symbol][model_name] = success
            completed += 1
            
            if success:
                print(f"✓ Progress: {completed}/{total_tasks} completed")
            else:
                failed += 1
                print(f"✗ Progress: {completed}/{total_tasks} completed ({failed} failed)")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total tasks: {total_tasks}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    if total_tasks > 0:
        success_rate = ((completed - failed) / total_tasks * 100)
        print(f"Success rate: {success_rate:.1f}%")
    else:
        print(f"Success rate: N/A (no tasks)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Detailed results
    print("\nDetailed Results:")
    print("-"*70)
    for symbol in symbols:
        print(f"\n{symbol}:")
        for model_name in available_models.keys():
            status = "✅" if results[symbol][model_name] else "❌"
            print(f"  {status} {model_name}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train all models for multiple symbols locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models for all default symbols
  python train_all_models.py

  # Train specific models for specific symbols
  python train_all_models.py --symbols AAPL,MSFT --models adaptive_forecaster,arima,lstm

  # Train all models for a single symbol
  python train_all_models.py --symbols AAPL

  # Train with custom epochs (for neural networks)
  python train_all_models.py --epochs 100
        """
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,JPM,V,JNJ)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to train (default: all). Options: adaptive_forecaster,arima,lstm,gru,transformer,exponential_smoothing"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for neural network models (default: 50)"
    )
    
    parser.add_argument(
        "--period",
        type=str,
        default="max",
        help="Time period for data (default: max). Options: 1y, 2y, 5y, max"
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = DEFAULT_SYMBOLS
    
    # Parse models
    if args.models:
        models = [m.strip().lower() for m in args.models.split(",")]
        # Validate models
        invalid_models = [m for m in models if m not in MODELS]
        if invalid_models:
            print(f"❌ Invalid models: {', '.join(invalid_models)}")
            print(f"Available models: {', '.join(MODELS.keys())}")
            sys.exit(1)
    else:
        models = None  # All models
    
    # Update epochs for neural network models
    if args.epochs != 50:
        for model_name in ["lstm", "gru", "transformer"]:
            if model_name in MODELS:
                # Update the args list
                if "--epochs" in MODELS[model_name]["args"]:
                    idx = MODELS[model_name]["args"].index("--epochs")
                    MODELS[model_name]["args"][idx + 1] = str(args.epochs)
                else:
                    MODELS[model_name]["args"].extend(["--epochs", str(args.epochs)])
                MODELS[model_name]["epochs"] = args.epochs
    
    # Run training
    results = train_all_models(
        symbols=symbols,
        models=models,
        max_period=args.period
    )
    
    # Exit with error code if any failed
    total_failed = sum(1 for symbol_results in results.values() 
                      for success in symbol_results.values() if not success)
    
    if total_failed > 0:
        print(f"\n⚠️  {total_failed} training task(s) failed")
        sys.exit(1)
    else:
        print("\n✅ All training tasks completed successfully!")
        sys.exit(0)

