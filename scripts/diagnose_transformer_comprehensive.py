"""
Comprehensive Transformer diagnostic script for price forecasting.
Calculates quantitative metrics, checks for saturation, temporal patterns, and provides interpretation.
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
)
from scipy.stats import pearsonr, spearmanr

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


def transformer_diagnostics(y_true, y_pred, normalized_preds=None):
    """
    Calculate comprehensive Transformer diagnostic metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        normalized_preds: Normalized predictions (for saturation checks)
    
    Returns:
        dict with all metrics
    """
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
    
    # Basic error metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    bias = np.mean(y_pred - y_true)
    
    # Fit & correlation metrics
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    # Distribution & saturation diagnostics
    pred_range = (np.max(y_pred) - np.min(y_pred)) if len(y_pred) > 0 else 0
    true_range = (np.max(y_true) - np.min(y_true)) if len(y_true) > 0 else 0
    range_ratio = pred_range / true_range if true_range > 0 else 0
    
    # Autocorrelation of residuals
    residuals = y_pred - y_true
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
    else:
        autocorr = 0
    
    # Temporal tracking: lag correlation
    if len(y_pred) > 1:
        lag_corr = np.corrcoef(y_pred[:-1], y_true[1:])[0, 1] if len(y_pred) > 1 else 0
    else:
        lag_corr = 0
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "bias": float(bias),
        "r2": float(r2),
        "pearson_r": float(pearson_corr),
        "spearman_rho": float(spearman_corr),
        "pred_range": float(pred_range),
        "true_range": float(true_range),
        "range_ratio": float(range_ratio),
        "residual_autocorr": float(autocorr),
        "lag_correlation": float(lag_corr),
        "n_samples": len(y_true)
    }
    
    # Add normalized prediction stats if provided
    if normalized_preds is not None:
        normalized_preds = np.array(normalized_preds).flatten()
        metrics["norm_pred_mean"] = float(np.mean(normalized_preds))
        metrics["norm_pred_std"] = float(np.std(normalized_preds))
        metrics["norm_pred_min"] = float(np.min(normalized_preds))
        metrics["norm_pred_max"] = float(np.max(normalized_preds))
    
    return metrics


def generate_predictions_on_test_set(
    model,
    prices: pd.Series,
    test_size: float = 0.2,
    min_test_samples: int = 30,
    return_normalized: bool = False
) -> tuple:
    """
    Generate predictions on test set using walk-forward approach.
    Returns: (predictions, actuals, dates, normalized_predictions)
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
    normalized_preds = []
    
    # Walk-forward: predict one step ahead at each point in test set
    for i in range(n_train, n_total):
        # Use all data up to current point
        historical = prices.iloc[:i]
        
        if len(historical) < lag_window:
            continue
        
        try:
            # Preprocess to get normalized input
            scaled_input = model.preprocess_input(historical)
            window = scaled_input[-lag_window:]
            X = window.reshape((1, lag_window, 1))
            
            # Get normalized prediction
            pred_normalized = model.model.predict(X, verbose=0)[0, 0]
            normalized_preds.append(pred_normalized)
            
            # Predict next step (this will inverse transform)
            pred = model.predict(historical, steps=1)[0]
            actual = prices.iloc[i]
            
            predictions.append(pred)
            actuals.append(actual)
            dates.append(prices.index[i])
        except Exception as e:
            logger.warning(f"Prediction failed at index {i}: {e}")
            continue
    
    result = (np.array(predictions), np.array(actuals), pd.DatetimeIndex(dates))
    if return_normalized:
        result = result + (np.array(normalized_preds),)
    return result


def calculate_rolling_metrics(y_true, y_pred, window: int = 30):
    """Calculate rolling RMSE and bias over time."""
    if len(y_true) < window:
        window = max(5, len(y_true) // 3)
    
    rolling_rmse = []
    rolling_bias = []
    
    for i in range(window, len(y_true)):
        window_true = y_true[i-window:i]
        window_pred = y_pred[i-window:i]
        
        rmse = np.sqrt(mean_squared_error(window_true, window_pred))
        bias = np.mean(window_pred - window_true)
        
        rolling_rmse.append(rmse)
        rolling_bias.append(bias)
    
    return np.array(rolling_rmse), np.array(rolling_bias)


def diagnose_transformer(
    symbol: str = "GOOGL",
    test_size: float = 0.2,
    generate_plots: bool = True,
    save_plots: bool = False,
    plots_dir: Path = None,
    no_plots: bool = False
):
    """
    Comprehensive Transformer model diagnosis.
    
    Args:
        symbol: Stock symbol
        test_size: Fraction of data to use for testing
        generate_plots: Whether to generate diagnostic plots
        save_plots: Whether to save plots to files
        plots_dir: Directory to save plots (if save_plots=True)
        no_plots: Skip plotting entirely (overrides generate_plots)
    """
    print(f"\n{'='*70}")
    print(f"Transformer Model Comprehensive Diagnosis for {symbol}")
    print(f"{'='*70}\n")
    
    if save_plots and plots_dir:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
    elif save_plots:
        plots_dir = Path(__file__).parent.parent / "diagnostics" / f"{symbol}_transformer"
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load Transformer model
        print("Step 1: Loading Transformer model...")
        try:
            model, model_version = ModelLoader.load_latest_model(symbol, "Transformer")
            print(f"   ✓ Transformer model loaded")
            print(f"   Model version: {model_version.get('id', 'N/A')}")
            print(f"   Lag window: {model.lag_window}")
            print(f"   d_model: {model.d_model}")
        except ValueError as e:
            print(f"   ❌ Failed to load Transformer model: {e}")
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
            y_pred, y_true, dates, normalized_preds = generate_predictions_on_test_set(
                model, prices, test_size=test_size, return_normalized=True
            )
            print(f"   ✓ Generated {len(y_pred)} predictions")
            print(f"   Test period: {dates[0]} to {dates[-1]}")
        except Exception as e:
            print(f"   ❌ Failed to generate predictions: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 5: Calculate comprehensive metrics
        print("\nStep 5: Calculating Transformer diagnostic metrics...")
        print(f"{'='*70}")
        metrics = transformer_diagnostics(y_true, y_pred, normalized_preds)
        
        print(f"\n📊 Basic Error Metrics:")
        print(f"   MAE (Mean Absolute Error):     ${metrics['mae']:.2f}")
        print(f"   RMSE (Root Mean Squared Error): ${metrics['rmse']:.2f}")
        print(f"   MAPE (Mean Absolute % Error):  {metrics['mape']:.2f}%")
        print(f"   Bias (Mean Error):              ${metrics['bias']:.2f}")
        
        print(f"\n📈 Fit & Correlation Metrics:")
        print(f"   R² (Coefficient of Determination): {metrics['r2']:.4f}")
        print(f"   Pearson Correlation (r):           {metrics['pearson_r']:.4f}")
        print(f"   Spearman Rank Correlation (ρ):     {metrics['spearman_rho']:.4f}")
        
        print(f"\n🔍 Distribution & Saturation Diagnostics:")
        print(f"   Predicted value range:      ${metrics['pred_range']:.2f}")
        print(f"   True value range:           ${metrics['true_range']:.2f}")
        print(f"   Range ratio (pred/true):    {metrics['range_ratio']:.3f}")
        if 'norm_pred_mean' in metrics:
            print(f"   Normalized prediction mean: {metrics['norm_pred_mean']:.3f}")
            print(f"   Normalized prediction std:  {metrics['norm_pred_std']:.3f}")
            print(f"   Normalized prediction range: [{metrics['norm_pred_min']:.3f}, {metrics['norm_pred_max']:.3f}]")
        
        print(f"\n⏱️  Temporal Tracking Metrics:")
        print(f"   Residual autocorrelation:    {metrics['residual_autocorr']:.4f}")
        print(f"   Lag correlation:             {metrics['lag_correlation']:.4f}")
        
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
            print(f"         • Possible activation saturation (output layer near bounds)")
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
        
        # Saturation analysis
        if 'norm_pred_mean' in metrics:
            norm_mean = metrics['norm_pred_mean']
            norm_std = metrics['norm_pred_std']
            
            if is_minmax:
                if abs(norm_mean - 0.5) < 0.2 and norm_std < 0.1:
                    print(f"   ⚠️  PREDICTIONS NEAR MEAN (saturation detected)")
                    print(f"      → Normalized predictions clustered around {norm_mean:.3f} (mean)")
                    print(f"      → Model output is clamped near mean → needs more recent data or better scaling")
                elif norm_mean < 0.1 or norm_mean > 0.9:
                    print(f"   ⚠️  PREDICTIONS NEAR BOUNDS (saturation detected)")
                    print(f"      → Normalized predictions near {norm_mean:.3f} (near 0 or 1)")
                    print(f"      → Model may be saturating at scaler bounds")
            elif is_standard:
                if abs(norm_mean) < 0.2 and norm_std < 0.1:
                    print(f"   ⚠️  PREDICTIONS NEAR MEAN (saturation detected)")
                    print(f"      → Normalized predictions clustered around {norm_mean:.3f} (z-score near 0)")
                    print(f"      → Model output is stuck near mean → needs more recent data or better scaling")
                elif abs(norm_mean) > 2:
                    print(f"   ⚠️  PREDICTIONS FAR FROM MEAN (extrapolation)")
                    print(f"      → Normalized predictions at {norm_mean:.3f} std from mean")
                    print(f"      → Model is extrapolating beyond training range")
        
        # Range ratio analysis
        if metrics['range_ratio'] < 0.5:
            print(f"   ⚠️  COLLAPSED PREDICTION RANGE (ratio: {metrics['range_ratio']:.3f})")
            print(f"      → Model predictions have much smaller range than true values")
            print(f"      → Sign of saturation or poor scaling")
        elif metrics['range_ratio'] > 2.0:
            print(f"   ⚠️  EXCESSIVE PREDICTION RANGE (ratio: {metrics['range_ratio']:.3f})")
            print(f"      → Model predictions vary more than true values")
            print(f"      → Possible overfitting or unstable predictions")
        else:
            print(f"   ✓ Prediction range is reasonable (ratio: {metrics['range_ratio']:.3f})")
        
        # Correlation vs R² analysis
        if metrics['pearson_r'] >= 0.9 and metrics['rmse'] > 10:
            print(f"   ⚠️  SCALING PROBLEM DETECTED")
            print(f"      → High correlation ({metrics['pearson_r']:.3f}) but large RMSE (${metrics['rmse']:.2f})")
            print(f"      → Model tracks direction but misses magnitude")
            print(f"      → Possible fix: adjust scaler or retrain with more recent data")
        elif metrics['pearson_r'] < 0.7:
            print(f"   ⚠️  LOW CORRELATION ({metrics['pearson_r']:.3f})")
            print(f"      → Model doesn't follow trend reversals or peaks well")
            print(f"      → Possible causes:")
            print(f"         • Sequence length too short (current: {model.lag_window})")
            print(f"         • Missing derivative features (price change %, RSI, moving averages)")
            print(f"         • Insufficient training epochs or learning rate issues")
        else:
            print(f"   ✓ Good correlation ({metrics['pearson_r']:.3f})")
        
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
        
        # Temporal analysis
        if metrics['residual_autocorr'] > 0.3:
            print(f"   ⚠️  HIGH RESIDUAL AUTOCORRELATION ({metrics['residual_autocorr']:.3f})")
            print(f"      → Residuals are temporally correlated (not random)")
            print(f"      → Model is underfitting temporal patterns")
        else:
            print(f"   ✓ Residual autocorrelation is low ({metrics['residual_autocorr']:.3f})")
        
        if metrics['lag_correlation'] > metrics['pearson_r'] * 1.1:
            print(f"   ⚠️  LAG CORRELATION HIGHER THAN DIRECT ({metrics['lag_correlation']:.3f} vs {metrics['pearson_r']:.3f})")
            print(f"      → Predictions lag behind true data")
            print(f"      → Model is reacting to past trends rather than current state")
        else:
            print(f"   ✓ Lag correlation is reasonable ({metrics['lag_correlation']:.3f})")
        
        # Step 7: Generate diagnostic plots
        if generate_plots and not no_plots:
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
                    # Rolling metrics
                    rolling_rmse, rolling_bias = calculate_rolling_metrics(y_true, y_pred, window=30)
                    rolling_dates = dates[len(dates) - len(rolling_rmse):]
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                    
                    # Rolling RMSE
                    ax1.plot(rolling_dates, rolling_rmse, label='Rolling RMSE (window=30)', linewidth=2, color='blue')
                    ax1.set_xlabel('Date', fontsize=12)
                    ax1.set_ylabel('RMSE ($)', fontsize=12)
                    ax1.set_title('Rolling RMSE Over Time', fontsize=14, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    # Rolling Bias
                    ax2.plot(rolling_dates, rolling_bias, label='Rolling Bias (window=30)', linewidth=2, color='red')
                    ax2.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
                    ax2.set_xlabel('Date', fontsize=12)
                    ax2.set_ylabel('Bias ($)', fontsize=12)
                    ax2.set_title('Rolling Bias Over Time', fontsize=14, fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    plt.tight_layout()
                    if save_plots:
                        plot_path = plots_dir / f"{symbol}_transformer_rolling_metrics.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        print(f"   ✓ Rolling metrics plot saved")
                    plt.close()
                    
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
            recommendations.append("      → The Transformer was trained on prices outside current range")
            recommendations.append("      → Input normalization compresses out-of-range values")
            recommendations.append("      → This causes severe extrapolation and flattened predictions")
            recommendations.append("")
            recommendations.append("      ✅ IMMEDIATE FIX:")
            recommendations.append("      1. Retrain Transformer with updated scaler fitted to full range:")
            recommendations.append(f"         - Current range: ${price_min:.2f} - ${price_max:.2f}")
            if is_minmax:
                recommendations.append(f"         - Use StandardScaler OR MinMaxScaler fitted to ${price_min:.0f}-${price_max:.0f} range")
            else:
                recommendations.append(f"         - Use StandardScaler with mean/std from recent data")
            recommendations.append(f"      2. Command: python notebooks/train_transformer.py --symbol {symbol}")
            recommendations.append("")
        
        # Priority 2: Bias issues
        if metrics['bias'] < -5:
            recommendations.append("   🔴 CRITICAL: Systematic Underprediction Bias")
            recommendations.append(f"      → Model always predicts lower than reality (${metrics['bias']:.2f} bias)")
            recommendations.append("      → This is structural bias, not random noise")
            recommendations.append("")
            recommendations.append("      ✅ FIXES (in order of priority):")
            recommendations.append("      1. Retrain with new price range (fixes scaler issue)")
            recommendations.append("      2. Optionally use ReLU activations in dense output layers")
            recommendations.append("         (reduces sigmoid/tanh compression near bounds)")
            recommendations.append("      3. Add mean-centering or z-score normalization per window")
            recommendations.append("         (stabilizes scale across different price ranges)")
            recommendations.append("")
        
        # Priority 3: Saturation issues
        if 'norm_pred_mean' in metrics:
            norm_mean = metrics['norm_pred_mean']
            norm_std = metrics['norm_pred_std']
            
            if is_minmax and (abs(norm_mean - 0.5) < 0.2 and norm_std < 0.1):
                recommendations.append("   ⚠️  Saturation Detected (Predictions Near Mean)")
                recommendations.append(f"      → Normalized predictions clustered around {norm_mean:.3f}")
                recommendations.append("      → Model output is clamped near mean")
                recommendations.append("")
                recommendations.append("      ✅ FIXES:")
                recommendations.append("      1. Use StandardScaler instead of MinMaxScaler")
                recommendations.append("      2. Add last 6-12 months of high-price data to training")
                recommendations.append("      3. Optionally enable positional encoding dropout (helps generalization)")
                recommendations.append("")
            elif is_standard and (abs(norm_mean) < 0.2 and norm_std < 0.1):
                recommendations.append("   ⚠️  Saturation Detected (Predictions Near Mean)")
                recommendations.append(f"      → Normalized predictions clustered around {norm_mean:.3f} (z-score)")
                recommendations.append("      → Model output is stuck near mean")
                recommendations.append("")
                recommendations.append("      ✅ FIXES:")
                recommendations.append("      1. Add last 6-12 months of high-price data to training")
                recommendations.append("      2. Check if model architecture needs more capacity")
                recommendations.append("      3. Optionally enable positional encoding dropout")
                recommendations.append("")
        
        # Priority 4: R² issues
        if metrics['r2'] < 0:
            recommendations.append("   🔴 CRITICAL: Negative R² (Model Worse Than Mean)")
            recommendations.append(f"      → R² = {metrics['r2']:.4f} means model is worse than predicting the mean!")
            recommendations.append("      → Focus on correcting bias via retraining/rescaling FIRST")
            recommendations.append("      → Don't change architecture until scaler is fixed")
            recommendations.append("")
        
        # Priority 5: Correlation issues
        if metrics['pearson_r'] < 0.7:
            recommendations.append(f"   ⚠️  Low Correlation (r={metrics['pearson_r']:.3f})")
            recommendations.append("      → Model partially tracks but misses magnitude/direction")
            recommendations.append("")
            recommendations.append("      ✅ FIXES:")
            recommendations.append(f"      1. Increase sequence length: {model.lag_window} → 90-120 steps")
            recommendations.append("         (captures slower price trends)")
            recommendations.append("      2. Add derivative features:")
            recommendations.append("         - Price change % (momentum)")
            recommendations.append("         - RSI (relative strength)")
            recommendations.append("         - Moving averages (trend smoothing)")
            recommendations.append("      3. Increase training epochs or add learning rate decay")
            recommendations.append("")
        
        # Priority 6: Temporal issues
        if metrics['residual_autocorr'] > 0.3:
            recommendations.append("   ⚠️  High Residual Autocorrelation Detected")
            recommendations.append("      → Add regularization (dropout, L2)")
            recommendations.append("      → Check for temporal underfitting (increase sequence length)")
            recommendations.append("")
        
        if metrics['lag_correlation'] > metrics['pearson_r'] * 1.1:
            recommendations.append("   ⚠️  Predictions Lag Behind True Data")
            recommendations.append("      → Model is reacting to past trends rather than current state")
            recommendations.append("      → Consider reducing sequence length or adding recent features")
            recommendations.append("")
        
        # Priority 7: Range issues
        if metrics['range_ratio'] < 0.5:
            recommendations.append("   ⚠️  Collapsed Prediction Range")
            recommendations.append("      → Model predictions have much smaller range than true values")
            recommendations.append("      → Check for saturation or poor scaling")
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
    
    parser = argparse.ArgumentParser(description="Comprehensive Transformer model diagnosis")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Stock symbol")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--plots-dir", type=str, default=None, help="Directory to save plots")
    
    args = parser.parse_args()
    
    success = diagnose_transformer(
        symbol=args.symbol,
        test_size=args.test_size,
        generate_plots=not args.no_plots,
        save_plots=args.save_plots,
        plots_dir=Path(args.plots_dir) if args.plots_dir else None,
        no_plots=args.no_plots
    )
    
    sys.exit(0 if success else 1)

