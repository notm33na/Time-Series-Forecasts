"""
Comprehensive diagnostic script for GRU model.
Calculates quantitative metrics, generates diagnostic plots, and provides interpretation.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path.parent))

from backend.services.model_loader import ModelLoader
from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy import stats
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    mdates = None
    stats = None

logger = get_logger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate comprehensive error metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Remove any NaN or inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid data points after filtering NaN/inf")
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    bias = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "bias": float(bias),
        "r2": float(r2),
        "correlation": float(corr),
        "n_samples": len(y_true)
    }


def generate_predictions_on_test_set(
    model,
    prices: pd.Series,
    test_size: float = 0.2,
    min_test_samples: int = 30
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Generate predictions on test set using walk-forward approach.
    Returns: (predictions, actuals, dates)
    """
    lag_window = model.lag_window
    n_total = len(prices)
    n_test = max(int(n_total * test_size), min_test_samples)
    n_train = n_total - n_test
    
    if n_train < lag_window:
        raise ValueError(
            f"Insufficient data: need at least {lag_window} points for training, "
            f"got {n_train} (total: {n_total})"
        )
    
    predictions = []
    actuals = []
    dates = []
    
    # Walk-forward: predict one step ahead at each point in test set
    for i in range(n_train, n_total):
        # Use all data up to current point
        historical = prices.iloc[:i]
        
        if len(historical) < lag_window:
            continue
        
        try:
            # Predict next step
            pred = model.predict(historical, steps=1)[0]
            actual = prices.iloc[i]
            
            predictions.append(pred)
            actuals.append(actual)
            dates.append(prices.index[i])
        except Exception as e:
            logger.warning(f"Prediction failed at index {i}: {e}")
            continue
    
    return np.array(predictions), np.array(actuals), pd.DatetimeIndex(dates)


def plot_predicted_vs_true(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = None):
    """Scatter plot: Predicted vs True values."""
    if not HAS_PLOTTING:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib scipy")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Price ($)', fontsize=12)
    ax.set_ylabel('Predicted Price ($)', fontsize=12)
    ax.set_title('Predicted vs True Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = None):
    """Residual plot: residuals vs true values."""
    if not HAS_PLOTTING:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib scipy")
    residuals = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_true, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    
    # Add trend line if there's a pattern
    if len(residuals) > 2:
        z = np.polyfit(y_true, residuals, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), "g--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
    
    ax.set_xlabel('True Price ($)', fontsize=12)
    ax.set_ylabel('Residual (Pred - True) ($)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = None):
    """Histogram of residuals."""
    if not HAS_PLOTTING:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib scipy")
    residuals = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    ax.axvline(np.mean(residuals), color='green', linestyle='--', lw=2, 
               label=f'Mean: ${np.mean(residuals):.2f}')
    
    ax.set_xlabel('Residual (Pred - True) ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_qq_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = None):
    """Q-Q plot of residuals to check normality."""
    if not HAS_PLOTTING:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib scipy")
    residuals = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Residuals (Normality Check)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_time_series(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.DatetimeIndex, 
                     save_path: Path = None):
    """Time series plot: True vs Predicted over time."""
    if not HAS_PLOTTING:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib scipy")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(dates, y_true, label='True Price', alpha=0.7, linewidth=1.5)
    ax.plot(dates, y_pred, label='Predicted Price', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('True vs Predicted Time Series', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_rolling_metrics(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.DatetimeIndex,
                         window: int = 30, save_path: Path = None):
    """Rolling MAE and RMSE over time."""
    if not HAS_PLOTTING:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib scipy")
    if len(y_true) < window:
        window = max(5, len(y_true) // 3)
    
    rolling_mae = []
    rolling_rmse = []
    rolling_dates = []
    
    for i in range(window, len(y_true)):
        window_true = y_true[i-window:i]
        window_pred = y_pred[i-window:i]
        
        mae = mean_absolute_error(window_true, window_pred)
        rmse = np.sqrt(mean_squared_error(window_true, window_pred))
        
        rolling_mae.append(mae)
        rolling_rmse.append(rmse)
        rolling_dates.append(dates[i])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(rolling_dates, rolling_mae, label=f'Rolling MAE (window={window})', linewidth=2)
    ax.plot(rolling_dates, rolling_rmse, label=f'Rolling RMSE (window={window})', linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Error ($)', fontsize=12)
    ax.set_title('Rolling Error Metrics Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def diagnose_gru(
    symbol: str = "GOOGL",
    test_size: float = 0.2,
    generate_plots: bool = True,
    save_plots: bool = False,
    plots_dir: Path = None
):
    """
    Comprehensive GRU model diagnosis.
    
    Args:
        symbol: Stock symbol
        test_size: Fraction of data to use for testing
        generate_plots: Whether to generate diagnostic plots
        save_plots: Whether to save plots to files
        plots_dir: Directory to save plots (if save_plots=True)
    """
    print(f"\n{'='*70}")
    print(f"GRU Model Comprehensive Diagnosis for {symbol}")
    print(f"{'='*70}\n")
    
    if save_plots and plots_dir:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
    elif save_plots:
        plots_dir = Path(__file__).parent.parent / "diagnostics" / f"{symbol}_gru"
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load GRU model
        print("Step 1: Loading GRU model...")
        try:
            model, model_version = ModelLoader.load_latest_model(symbol, "GRU")
            print(f"   ✓ GRU model loaded")
            print(f"   Model version: {model_version.get('id', 'N/A')}")
            print(f"   Lag window: {model.lag_window}")
            print(f"   Units: {model.units}")
        except ValueError as e:
            print(f"   ❌ Failed to load GRU model: {e}")
            return False
        
        # Step 2: Get historical data
        print("\nStep 2: Loading historical data...")
        data_service = DataIngestionService()
        df = data_service.get_prices(symbol, limit=1000)  # Get more data for testing
        if df.empty:
            print(f"   ❌ No data found for {symbol}")
            return False
        
        # Ensure date column exists and create Series with date as index
        if 'date' not in df.columns:
            print(f"   ❌ No 'date' column found. Available columns: {list(df.columns)}")
            return False
        
        # Create prices Series with date as index
        df['date'] = pd.to_datetime(df['date'], utc=True)
        prices = pd.Series(df['close'].values, index=df['date'])
        prices = prices.sort_index()  # Ensure chronological order
        
        print(f"   ✓ Loaded {len(prices)} data points")
        print(f"   Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"   Current price: ${prices.iloc[-1]:.2f}")
        
        # Step 3: Check scaler range (CRITICAL DIAGNOSIS)
        print("\nStep 3: Checking scaler configuration...")
        if not hasattr(model, 'scaler'):
            print("   ❌ Model missing scaler!")
            return False
        
        scaler = model.scaler
        is_minmax = hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_')
        is_standard = hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_')
        
        current_price = float(prices.iloc[-1])
        price_min = float(prices.min())
        price_max = float(prices.max())
        
        scaler_issue_detected = False
        
        if is_minmax:
            scaler_min = float(scaler.data_min_[0])
            scaler_max = float(scaler.data_max_[0])
            scaler_range = scaler_max - scaler_min
            print(f"   Scaler type: MinMaxScaler")
            print(f"   Scaler min: ${scaler_min:.2f}")
            print(f"   Scaler max: ${scaler_max:.2f}")
            print(f"   Scaler range: ${scaler_range:.2f}")
            print(f"\n   Current data range: ${price_min:.2f} - ${price_max:.2f}")
            print(f"   Current price: ${current_price:.2f}")
            
            # Check if current price is outside scaler range
            if current_price > scaler_max:
                pct_outside = ((current_price - scaler_max) / scaler_max * 100)
                print(f"\n   🔴 CRITICAL: Current price ${current_price:.2f} is {pct_outside:.1f}% ABOVE scaler max ${scaler_max:.2f}")
                print(f"      → Input normalization compresses everything beyond ${scaler_max:.2f} into '1.0'")
                print(f"      → Model never saw such values during training (extrapolation)")
                print(f"      → This causes flattened predictions and systematic underprediction bias")
                scaler_issue_detected = True
            elif current_price < scaler_min:
                pct_outside = ((scaler_min - current_price) / current_price * 100)
                print(f"\n   🔴 CRITICAL: Current price ${current_price:.2f} is {pct_outside:.1f}% BELOW scaler min ${scaler_min:.2f}")
                print(f"      → Input normalization compresses everything below ${scaler_min:.2f} into '0.0'")
                print(f"      → Model never saw such values during training (extrapolation)")
                scaler_issue_detected = True
            else:
                # Check if data range extends beyond scaler
                if price_max > scaler_max:
                    pct_beyond = ((price_max - scaler_max) / scaler_max * 100)
                    print(f"\n   ⚠️  WARNING: Data range extends {pct_beyond:.1f}% beyond scaler max")
                    print(f"      → Some historical data is outside training range")
                if price_min < scaler_min:
                    pct_below = ((scaler_min - price_min) / price_min * 100)
                    print(f"\n   ⚠️  WARNING: Data range extends {pct_below:.1f}% below scaler min")
                
                # Check overlap
                overlap_pct = ((min(price_max, scaler_max) - max(price_min, scaler_min)) / scaler_range * 100)
                if overlap_pct < 50:
                    print(f"\n   ⚠️  WARNING: Only {overlap_pct:.1f}% overlap between data range and scaler range")
                    scaler_issue_detected = True
                else:
                    print(f"\n   ✓ Current price is within scaler range")
                    print(f"   ✓ Data range overlap: {overlap_pct:.1f}%")
                    
        elif is_standard:
            scaler_mean = float(scaler.mean_[0])
            scaler_std = float(scaler.scale_[0])
            print(f"   Scaler type: StandardScaler")
            print(f"   Mean: ${scaler_mean:.2f}")
            print(f"   Std: ${scaler_std:.2f}")
            print(f"   Typical range (±3 std): [${scaler_mean - 3*scaler_std:.2f}, ${scaler_mean + 3*scaler_std:.2f}]")
            print(f"\n   Current data range: ${price_min:.2f} - ${price_max:.2f}")
            print(f"   Current price: ${current_price:.2f}")
            
            z_score = (current_price - scaler_mean) / scaler_std
            print(f"\n   Current price z-score: {z_score:.2f}")
            
            if abs(z_score) > 3:
                print(f"\n   🔴 CRITICAL: Current price is {abs(z_score):.2f} std from mean")
                print(f"      → Model is extrapolating beyond typical training range")
                print(f"      → This may cause unreliable predictions")
                scaler_issue_detected = True
            elif abs(z_score) > 2:
                print(f"\n   ⚠️  WARNING: Current price is {abs(z_score):.2f} std from mean")
                print(f"      → Approaching extrapolation territory")
            else:
                print(f"\n   ✓ Current price is within ±2 std of mean (typical range)")
        else:
            print("   ⚠️  Unknown scaler type")
        
        # Step 4: Generate predictions on test set
        print("\nStep 4: Generating predictions on test set...")
        try:
            y_pred, y_true, dates = generate_predictions_on_test_set(
                model, prices, test_size=test_size
            )
            print(f"   ✓ Generated {len(y_pred)} predictions")
            print(f"   Test period: {dates[0]} to {dates[-1]}")
        except Exception as e:
            print(f"   ❌ Failed to generate predictions: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 5: Calculate metrics
        print("\nStep 5: Calculating error metrics...")
        print(f"{'='*70}")
        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"\n📊 Basic Error Metrics:")
        print(f"   MAE (Mean Absolute Error):     ${metrics['mae']:.2f}")
        print(f"   RMSE (Root Mean Squared Error): ${metrics['rmse']:.2f}")
        print(f"   MAPE (Mean Absolute % Error):  {metrics['mape']:.2f}%")
        print(f"   Bias (Mean Error):              ${metrics['bias']:.2f}")
        
        print(f"\n📈 Fit & Correlation Metrics:")
        print(f"   R² (Coefficient of Determination): {metrics['r2']:.4f}")
        print(f"   Pearson Correlation (r):           {metrics['correlation']:.4f}")
        
        print(f"\n📋 Sample Statistics:")
        print(f"   Number of predictions: {metrics['n_samples']}")
        print(f"   True price range:      ${y_true.min():.2f} - ${y_true.max():.2f}")
        print(f"   Pred price range:      ${y_pred.min():.2f} - ${y_pred.max():.2f}")
        
        # Step 6: Interpretation
        print(f"\n{'='*70}")
        print("🔍 Interpretation:")
        print(f"{'='*70}")
        
        # Bias analysis
        if metrics['bias'] < -5:
            print(f"   🔴 LARGE NEGATIVE BIAS (${metrics['bias']:.2f})")
            print(f"      → Model systematically UNDERPREDICTS (always predicts lower than reality)")
            print(f"      → This is NOT random noise - it's STRUCTURAL BIAS")
            print(f"      → Root causes:")
            print(f"         • Out-of-range scaler (prices beyond training range compressed)")
            print(f"         • Training data skew (insufficient examples near current price range)")
            print(f"         • Possible activation saturation (sigmoid/tanh near upper bound)")
        elif metrics['bias'] > 5:
            print(f"   🔴 LARGE POSITIVE BIAS (${metrics['bias']:.2f})")
            print(f"      → Model systematically OVERPREDICTS")
            print(f"      → Possible causes: scaler range mismatch or training data issues")
        else:
            print(f"   ✓ Bias is relatively small (${metrics['bias']:.2f})")
        
        # Variance analysis
        if metrics['rmse'] > metrics['mae'] * 1.5:
            print(f"   ⚠️  HIGH VARIANCE (RMSE >> MAE)")
            print(f"      → Model has inconsistent predictions (some large errors)")
            print(f"      → Possible causes: overfitting, unstable training, or data noise")
        else:
            print(f"   ✓ Variance is controlled (RMSE ≈ MAE)")
        
        # MAPE analysis
        if metrics['mape'] > 10:
            print(f"   ⚠️  HIGH MAPE ({metrics['mape']:.2f}%)")
            print(f"      → Poor calibration or scaling issues")
        elif metrics['mape'] > 5:
            print(f"   ⚠️  MODERATE MAPE ({metrics['mape']:.2f}%)")
            print(f"      → Acceptable but could be improved")
        else:
            print(f"   ✓ MAPE is good ({metrics['mape']:.2f}%)")
        
        # Correlation vs RMSE
        if metrics['correlation'] >= 0.9 and metrics['rmse'] > 10:
            print(f"   ⚠️  SCALING PROBLEM DETECTED")
            print(f"      → High correlation ({metrics['correlation']:.3f}) but large RMSE (${metrics['rmse']:.2f})")
            print(f"      → Model tracks direction but misses magnitude")
            print(f"      → Possible fix: adjust scaler or retrain with more recent data")
        elif metrics['correlation'] < 0.7:
            print(f"   ⚠️  LOW CORRELATION ({metrics['correlation']:.3f})")
            print(f"      → Partial tracking (r={metrics['correlation']:.3f}) but magnitude/direction are off")
            print(f"      → Model doesn't follow trend reversals or peaks well")
            print(f"      → Possible causes:")
            print(f"         • Sequence length too short (current: {model.lag_window})")
            print(f"         • Missing derivative features (price change %, RSI, moving averages)")
            print(f"         • Insufficient training epochs or learning rate issues")
        else:
            print(f"   ✓ Good correlation ({metrics['correlation']:.3f})")
        
        # R² analysis
        if metrics['r2'] < 0:
            print(f"   🔴 NEGATIVE R² ({metrics['r2']:.4f})")
            print(f"      → Model is WORSE than predicting the mean!")
            print(f"      → Model performance is worse than a naive baseline")
            print(f"      → This indicates severe model failure or data mismatch")
        elif metrics['r2'] < 0.5:
            print(f"   ⚠️  LOW R² ({metrics['r2']:.4f})")
            print(f"      → Model explains less than 50% of variance")
            print(f"      → Model struggles to capture price dynamics")
        elif metrics['r2'] < 0.8:
            print(f"   ⚠️  MODERATE R² ({metrics['r2']:.4f})")
            print(f"      → Model explains {metrics['r2']*100:.1f}% of variance")
        else:
            print(f"   ✓ Good R² ({metrics['r2']:.4f}) - explains {metrics['r2']*100:.1f}% of variance")
        
        # Step 7: Generate diagnostic plots
        if generate_plots:
            if not HAS_PLOTTING:
                print(f"\n{'='*70}")
                print("⚠️  Plotting libraries not available")
                print(f"{'='*70}")
                print("   Install matplotlib and scipy to generate plots:")
                print("   pip install matplotlib scipy")
                print("   Or run with --no-plots to skip plotting")
            else:
                print(f"\n{'='*70}")
                print("📊 Generating diagnostic plots...")
                print(f"{'='*70}\n")
                
                try:
                    plot_path = plots_dir / f"{symbol}_gru_predicted_vs_true.png" if save_plots else None
                    plot_predicted_vs_true(y_true, y_pred, save_path=plot_path)
                    print("   ✓ Predicted vs True scatter plot")
                    
                    plot_path = plots_dir / f"{symbol}_gru_residuals.png" if save_plots else None
                    plot_residuals(y_true, y_pred, save_path=plot_path)
                    print("   ✓ Residual plot")
                    
                    plot_path = plots_dir / f"{symbol}_gru_residual_hist.png" if save_plots else None
                    plot_residual_histogram(y_true, y_pred, save_path=plot_path)
                    print("   ✓ Residual histogram")
                    
                    plot_path = plots_dir / f"{symbol}_gru_qq_plot.png" if save_plots else None
                    plot_qq_residuals(y_true, y_pred, save_path=plot_path)
                    print("   ✓ Q-Q plot (normality check)")
                    
                    plot_path = plots_dir / f"{symbol}_gru_time_series.png" if save_plots else None
                    plot_time_series(y_true, y_pred, dates, save_path=plot_path)
                    print("   ✓ Time series plot")
                    
                    plot_path = plots_dir / f"{symbol}_gru_rolling_metrics.png" if save_plots else None
                    plot_rolling_metrics(y_true, y_pred, dates, window=30, save_path=plot_path)
                    print("   ✓ Rolling metrics plot")
                    
                except Exception as e:
                    print(f"   ⚠️  Error generating plots: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Step 8: Recommendations
        print(f"\n{'='*70}")
        print("💡 Recommendations:")
        print(f"{'='*70}")
        
        recommendations = []
        
        # Priority 1: Scaler range issues (most critical)
        if scaler_issue_detected:
            recommendations.append("   🔴 CRITICAL: Scaler Range Mismatch")
            recommendations.append("      → The GRU was trained on prices ≤ ${:.2f}, but now seeing ${:.2f}+".format(
                scaler_max if is_minmax else scaler_mean + 3*scaler_std, current_price))
            recommendations.append("      → Input normalization compresses out-of-range values")
            recommendations.append("      → This causes severe extrapolation and flattened predictions")
            recommendations.append("")
            recommendations.append("      ✅ IMMEDIATE FIX:")
            recommendations.append("      1. Retrain GRU with updated scaler fitted to full range:")
            recommendations.append("         - Current range: ${:.2f} - ${:.2f}".format(price_min, price_max))
            recommendations.append("         - Use StandardScaler OR MinMaxScaler fitted to $128-$290 range")
            recommendations.append("      2. Command: python notebooks/train_gru.py --symbol {}".format(symbol))
            recommendations.append("")
        
        # Priority 2: Bias issues
        if metrics['bias'] < -5:
            recommendations.append("   🔴 CRITICAL: Systematic Underprediction Bias")
            recommendations.append("      → Model always predicts lower than reality (${:.2f} bias)".format(metrics['bias']))
            recommendations.append("      → This is structural bias, not random noise")
            recommendations.append("")
            recommendations.append("      ✅ FIXES (in order of priority):")
            recommendations.append("      1. Retrain with new price range (fixes scaler issue)")
            recommendations.append("      2. Optionally use ReLU activations in dense output layers")
            recommendations.append("         (reduces sigmoid/tanh compression near bounds)")
            recommendations.append("      3. Add mean-centering or z-score normalization per window")
            recommendations.append("         (stabilizes scale across different price ranges)")
            recommendations.append("")
        
        # Priority 3: R² issues
        if metrics['r2'] < 0:
            recommendations.append("   🔴 CRITICAL: Negative R² (Model Worse Than Mean)")
            recommendations.append("      → R² = {:.4f} means model is worse than predicting the mean!".format(metrics['r2']))
            recommendations.append("      → Focus on correcting bias via retraining/rescaling FIRST")
            recommendations.append("      → Don't change architecture until scaler is fixed")
            recommendations.append("")
        
        # Priority 4: Correlation issues
        if metrics['correlation'] < 0.7:
            recommendations.append("   ⚠️  Low Correlation (r={:.3f})".format(metrics['correlation']))
            recommendations.append("      → Model partially tracks but misses magnitude/direction")
            recommendations.append("")
            recommendations.append("      ✅ FIXES:")
            recommendations.append("      1. Increase sequence length: {} → 90-120 steps".format(model.lag_window))
            recommendations.append("         (captures slower price trends)")
            recommendations.append("      2. Add derivative features:")
            recommendations.append("         - Price change % (momentum)")
            recommendations.append("         - RSI (relative strength)")
            recommendations.append("         - Moving averages (trend smoothing)")
            recommendations.append("      3. Increase training epochs or add learning rate decay")
            recommendations.append("")
        
        # Priority 5: Variance issues
        if metrics['rmse'] > metrics['mae'] * 1.5:
            recommendations.append("   ⚠️  High Variance Detected")
            recommendations.append("      → Add regularization (dropout, L2)")
            recommendations.append("      → Check for overfitting (reduce model complexity)")
            recommendations.append("")
        
        # Priority 6: MAPE issues
        if metrics['mape'] > 10:
            recommendations.append("   ⚠️  High MAPE ({:.2f}%)".format(metrics['mape']))
            recommendations.append("      → Check scaler configuration")
            recommendations.append("      → Verify data preprocessing is correct")
            recommendations.append("")
        
        # Priority 7: Scaling problems
        if metrics['correlation'] >= 0.9 and metrics['rmse'] > 10:
            recommendations.append("   ⚠️  Scaling Problem (High Correlation, Large RMSE)")
            recommendations.append("      → Model tracks direction but misses magnitude")
            recommendations.append("      → Consider switching to StandardScaler if using MinMaxScaler")
            recommendations.append("      → Retrain with more recent data")
            recommendations.append("")
        
        if not recommendations:
            recommendations.append("   ✓ Model appears to be performing well")
            recommendations.append("   → Continue monitoring for drift over time")
            recommendations.append("   → Consider periodic retraining with fresh data")
        
        for rec in recommendations:
            print(rec)
        
        print(f"\n{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive GRU model diagnosis")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Stock symbol")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--plots-dir", type=str, default=None, help="Directory to save plots")
    
    args = parser.parse_args()
    
    success = diagnose_gru(
        symbol=args.symbol,
        test_size=args.test_size,
        generate_plots=not args.no_plots,
        save_plots=args.save_plots,
        plots_dir=Path(args.plots_dir) if args.plots_dir else None
    )
    
    sys.exit(0 if success else 1)

