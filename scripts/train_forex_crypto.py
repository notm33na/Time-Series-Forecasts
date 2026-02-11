"""
Training script for Forex and Crypto pairs.
Trains all models (ARIMA, LSTM, GRU, Transformer, Exponential Smoothing) for:
- 2 Forex pairs: EURUSD, GBPUSD
- 2 Crypto pairs: BTC-USD, ETH-USD

This script uses the improved/fixed training scripts where available.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Forex and Crypto symbols
FOREX_SYMBOLS = [
    "EURUSD=X",  # EUR/USD
    "GBPUSD=X",  # GBP/USD
]

CRYPTO_SYMBOLS = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
]

ALL_SYMBOLS = FOREX_SYMBOLS + CRYPTO_SYMBOLS

# Model training configurations
# Using improved/fixed scripts where available
MODELS = {
    "arima": {
        "script": "notebooks/train_arima.py",
        "args": [],
        "epochs": None,
    },
    "lstm": {
        "script": "notebooks/train_lstm.py",
        "args": ["--epochs", "30", "--batch-size", "16"],
        "epochs": 30,
    },
    "gru": {
        "script": "scripts/retrain_gru_fixed.py",  # Use improved GRU script
        "args": ["--epochs", "50", "--lag-window", "90", "--scaler", "standard"],
        "epochs": 50,
    },
    "transformer": {
        "script": "scripts/retrain_transformer_fixed.py",  # Use improved Transformer script
        "args": ["--epochs", "30", "--lag-window", "60", "--learning-rate", "1e-4"],
        "epochs": 30,
    },
    "exponential_smoothing": {
        "script": "notebooks/train_exponential_smoothing.py",
        "args": [],
        "epochs": None,
    },
}


def train_model(symbol: str, model_name: str, model_config: dict) -> bool:
    """
    Train a single model for a symbol.
    
    Args:
        symbol: Trading pair symbol
        model_name: Name of the model
        model_config: Model configuration dict
        
    Returns:
        True if successful, False otherwise
    """
    script_path = project_root / model_config["script"]
    
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
        # Run with real-time output
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            timeout=7200  # 2 hour timeout per model (crypto/forex may need more data)
        )
        
        print(f"\n✅ {model_name.upper()} training completed for {symbol}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n❌ {model_name.upper()} training timed out for {symbol} (>2 hours)")
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
    skip_ensemble: bool = False
):
    """
    Train all models for all symbols.
    
    Args:
        symbols: List of symbols to train (default: ALL_SYMBOLS)
        models: List of model names to train (default: all models)
        skip_ensemble: Skip ensemble training (default: False)
    """
    if symbols is None:
        symbols = ALL_SYMBOLS
    
    if models is None:
        models = list(MODELS.keys())
    
    # Filter to only existing models
    available_models = {k: v for k, v in MODELS.items() if k in models}
    
    # Verify all scripts exist before starting
    missing_scripts = []
    for model_name, model_config in available_models.items():
        script_path = project_root / model_config["script"]
        if not script_path.exists():
            missing_scripts.append((model_name, script_path))
    
    if missing_scripts:
        print("❌ ERROR: Some training scripts are missing:")
        for model_name, script_path in missing_scripts:
            print(f"   - {model_name}: {script_path}")
        print("\nPlease ensure all training scripts exist before running batch training.")
        return {}
    
    print("="*70)
    print("FOREX & CRYPTO BATCH TRAINING")
    print("="*70)
    print(f"Forex Pairs: {', '.join(FOREX_SYMBOLS)}")
    print(f"Crypto Pairs: {', '.join(CRYPTO_SYMBOLS)}")
    print(f"Total Symbols: {len(symbols)}")
    print(f"Models: {', '.join(available_models.keys())}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    results = {}
    total_tasks = len(symbols) * len(available_models)
    completed = 0
    failed = 0
    
    for symbol in symbols:
        results[symbol] = {}
        print(f"\n{'#'*70}")
        print(f"# Processing {symbol}")
        print(f"{'#'*70}")
        
        for model_name, model_config in available_models.items():
            success = train_model(symbol, model_name, model_config)
            results[symbol][model_name] = success
            completed += 1
            
            if success:
                print(f"✓ Progress: {completed}/{total_tasks} completed")
            else:
                failed += 1
                print(f"✗ Progress: {completed}/{total_tasks} completed ({failed} failed)")
    
    # Train ensemble models if requested
    if not skip_ensemble:
        print(f"\n{'='*70}")
        print("TRAINING ENSEMBLE MODELS")
        print(f"{'='*70}")
        
        ensemble_script = project_root / "notebooks" / "train_ensemble.py"
        if ensemble_script.exists():
            for symbol in symbols:
                print(f"\nTraining Ensemble for {symbol}...")
                cmd = [
                    sys.executable,
                    str(ensemble_script),
                    "--symbol", symbol,
                    "--horizon", "24",
                ]
                
                try:
                    subprocess.run(
                        cmd,
                        cwd=project_root,
                        check=True,
                        timeout=1800  # 30 min timeout for ensemble
                    )
                    print(f"✅ Ensemble training completed for {symbol}")
                    results[symbol]["ensemble"] = True
                except Exception as e:
                    print(f"❌ Ensemble training failed for {symbol}: {e}")
                    results[symbol]["ensemble"] = False
        else:
            print("⚠️  Ensemble training script not found, skipping...")
    
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
            status = "✅" if results[symbol].get(model_name, False) else "❌"
            print(f"  {status} {model_name}")
        if "ensemble" in results[symbol]:
            status = "✅" if results[symbol]["ensemble"] else "❌"
            print(f"  {status} ensemble")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train all models for Forex and Crypto pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models for all forex and crypto pairs
  python scripts/train_forex_crypto.py

  # Train specific models for specific symbols
  python scripts/train_forex_crypto.py --symbols EURUSD=X,BTC-USD --models arima,lstm,gru

  # Train only forex pairs
  python scripts/train_forex_crypto.py --forex-only

  # Train only crypto pairs
  python scripts/train_forex_crypto.py --crypto-only

  # Skip ensemble training
  python scripts/train_forex_crypto.py --skip-ensemble
        """
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: EURUSD=X,GBPUSD=X,BTC-USD,ETH-USD)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to train (default: all). Options: arima,lstm,gru,transformer,exponential_smoothing"
    )
    
    parser.add_argument(
        "--forex-only",
        action="store_true",
        help="Train only forex pairs (EURUSD=X, GBPUSD=X)"
    )
    
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Train only crypto pairs (BTC-USD, ETH-USD)"
    )
    
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Skip ensemble training (default: False)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for neural network models"
    )
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.forex_only:
        symbols = FOREX_SYMBOLS
    elif args.crypto_only:
        symbols = CRYPTO_SYMBOLS
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = ALL_SYMBOLS
    
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
    
    # Update epochs if specified
    if args.epochs:
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
        skip_ensemble=args.skip_ensemble
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

