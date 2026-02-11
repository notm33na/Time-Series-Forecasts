import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
import { CandlestickChart } from "@/components/CandlestickChart";

interface OHLCData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PriceData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ForecastData {
  arima_forecast?: number[];
  lstm_forecast?: number[];
  gru_forecast?: number[];
  transformer_forecast?: number[];
  exponential_smoothing_forecast?: number[];
  ensemble_forecast?: number[];
  predictions?: number[]; // For individual model forecasts
  data_frequency?: string; // 'daily' or 'hourly'
  forecast_steps?: number;
  symbol?: string;
  model_type?: string;
  horizon?: string;
  status?: string;
  metrics?: {
    ARIMA?: { RMSE: number; MAE: number; MAPE: number };
    LSTM?: { RMSE: number; MAE: number; MAPE: number };
    GRU?: { RMSE: number; MAE: number; MAPE: number };
    Transformer?: { RMSE: number; MAE: number; MAPE: number };
  };
}

const Dashboard = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [instrumentType, setInstrumentType] = useState<string>("stock");
  const [symbol, setSymbol] = useState<string>("AAPL");
  const [horizon, setHorizon] = useState<string>("24h");
  const [ohlcData, setOhlcData] = useState<OHLCData[]>([]);
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [errorData, setErrorData] = useState<Array<{ date: string; actual: number; predicted: number; error: number; errorPercent: number }>>([]);

  const instrumentOptions = {
    stock: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
    crypto: ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
    forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
  };

  const fetchOHLCData = useCallback(async () => {
    try {
      // Fetch from backend API instead of Supabase
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/data/prices/${symbol}?limit=500`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch price data');
      }

      const result = await response.json() as { prices?: PriceData[] };
      
      if (result.prices && result.prices.length > 0) {
        // Detect if data is hourly by checking time differences between consecutive points
        let isHourly = false;
        if (result.prices.length > 1) {
          const timeDiffs: number[] = [];
          for (let i = 1; i < Math.min(result.prices.length, 10); i++) {
            const diff = Math.abs(
              new Date(result.prices[i].timestamp).getTime() - 
              new Date(result.prices[i-1].timestamp).getTime()
            );
            timeDiffs.push(diff);
          }
          // If median time difference is less than 24 hours, it's likely hourly data
          const medianDiff = timeDiffs.sort((a, b) => a - b)[Math.floor(timeDiffs.length / 2)];
          isHourly = medianDiff < 86400000 && medianDiff > 0; // Between 0 and 24 hours
        }
        
        // Calculate time window based on selected forecast horizon
        const horizonHours = parseInt(horizon.replace('h', '')) || 24;
        const now = new Date();
        let startTime: Date;
        
        if (isHourly) {
          // For hourly data, show appropriate window based on horizon
          if (horizonHours === 1) {
            // For 1h: Show last 24 hours of hourly data (1 day)
            startTime = new Date(now.getTime() - (24 * 60 * 60 * 1000));
          } else if (horizonHours === 3) {
            // For 3h: Show last 72 hours of hourly data (3 days)
            startTime = new Date(now.getTime() - (72 * 60 * 60 * 1000));
          } else if (horizonHours === 24) {
            // For 24h: Show last 7 days of hourly data
            startTime = new Date(now.getTime() - (7 * 24 * 60 * 60 * 1000));
          } else {
            // For 72h: Show last 14 days of hourly data
            startTime = new Date(now.getTime() - (14 * 24 * 60 * 60 * 1000));
          }
        } else {
          // For daily data, show appropriate window
          if (horizonHours <= 3) {
            // For 1h, 3h: Show last 7 days (even though data is daily, show recent context)
            startTime = new Date(now.getTime() - (7 * 24 * 60 * 60 * 1000));
          } else if (horizonHours === 24) {
            // For 24h: Show last 30 days
            startTime = new Date(now.getTime() - (30 * 24 * 60 * 60 * 1000));
          } else {
            // For 72h: Show last 90 days
            startTime = new Date(now.getTime() - (90 * 24 * 60 * 60 * 1000));
          }
        }
        
        // Filter data to only show data within the time window
        const filteredPrices = result.prices.filter((item: PriceData) => {
          const itemTime = new Date(item.timestamp);
          return itemTime >= startTime;
        });
        
        const chartData: OHLCData[] = filteredPrices.map((item: PriceData) => {
          const dateObj = new Date(item.timestamp);
          // For hourly data, preserve full datetime with time; for daily, use date only
          // Use ISO string for consistency, which preserves time information
          return {
            date: isHourly ? dateObj.toISOString() : dateObj.toISOString().split('T')[0],
            open: Number(item.open),
            high: Number(item.high),
            low: Number(item.low),
            close: Number(item.close),
            volume: Number(item.volume || 0),
          };
        });
        setOhlcData(chartData);
      }
    } catch (error) {
      console.error("Error fetching OHLC data:", error);
      toast({
        title: "Data fetch failed",
        description: "Could not fetch price data. Make sure data is ingested first using the API.",
        variant: "destructive",
      });
    }
  }, [symbol, horizon, toast]);

  const fetchErrorData = useCallback(async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/evaluation/errors/${symbol}?limit=50`);
      
      if (response.ok) {
        const result = await response.json() as { errors?: Array<{ date: string; actual: number; predicted: number; error: number; errorPercent: number }> };
        if (result.errors) {
          // Preserve full datetime for error data to match historical data format
          // Check if errors have time components (hourly data)
          const hasTimeInErrors = result.errors.length > 1 && 
            Math.abs(new Date(result.errors[1].date).getTime() - 
                     new Date(result.errors[0].date).getTime()) < 86400000;
          
          const formattedErrors = result.errors.map(err => {
            const dateObj = new Date(err.date);
            // Use ISO string format consistently for Plotly compatibility
            // For hourly data, use full ISO string; for daily, also use ISO string (Plotly handles both)
            return {
              ...err,
              date: dateObj.toISOString()
            };
          });
          setErrorData(formattedErrors);
        }
      }
    } catch (error) {
      console.error("Error fetching error data:", error);
      // Don't show toast for errors, as this is optional data
    }
  }, [symbol]);

  // Refetch data when horizon or symbol changes to show appropriate time window
  useEffect(() => {
    fetchOHLCData();
  }, [horizon, symbol, fetchOHLCData]);

  useEffect(() => {
    fetchErrorData();
  }, [fetchErrorData]);

  const handleRunForecast = async () => {
    setLoading(true);

    try {
      // Fetch from backend API
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const horizonHours = parseInt(horizon.replace('h', '')) || 24;
      
      const response = await fetch(`${apiUrl}/api/forecast/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: symbol,
          model_type: 'Ensemble',
          horizon: horizon,
        }),
      });

      if (!response.ok) {
        // Try to get error details from response
        let errorMessage = `Failed to fetch forecast (${response.status} ${response.statusText})`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          // If response is not JSON, use status text
          errorMessage = `Failed to fetch forecast: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json() as ForecastData;

      // Debug: Log forecast data
      console.log("Forecast API Response:", data);
      console.log("Ensemble forecast:", data.ensemble_forecast);
      console.log("Forecast steps:", data.forecast_steps);
      console.log("Data frequency:", data.data_frequency);
      console.log("Metrics:", data.metrics);

      // Always try to fetch metrics separately (they may not be in forecast response)
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const metricsResponse = await fetch(`${apiUrl}/api/evaluation/metrics/${symbol}/latest`);
        if (metricsResponse.ok) {
          const metricsResult = await metricsResponse.json() as { latest_metrics?: Record<string, { rmse?: number; mae?: number; mape?: number; RMSE?: number; MAE?: number; MAPE?: number }> };
          console.log("Metrics API Response:", metricsResult);
          // Transform metrics data to match our interface
          if (metricsResult.latest_metrics) {
            const metrics: Record<string, { RMSE: number; MAE: number; MAPE: number }> = {};
            Object.entries(metricsResult.latest_metrics).forEach(([modelType, metricData]) => {
              const rmse = metricData.RMSE ?? metricData.rmse;
              const mae = metricData.MAE ?? metricData.mae;
              const mape = metricData.MAPE ?? metricData.mape;
              // Include metrics if at least one value exists (even if 0)
              if (rmse !== undefined || mae !== undefined || mape !== undefined) {
                metrics[modelType] = {
                  RMSE: rmse ?? 0,
                  MAE: mae ?? 0,
                  MAPE: mape ?? 0,
                };
              }
            });
            console.log("Transformed metrics:", metrics);
            // Merge with existing metrics if any
            data.metrics = { ...data.metrics, ...metrics };
          }
        } else {
          console.warn("Metrics API response not OK:", metricsResponse.status, metricsResponse.statusText);
        }
      } catch (metricsError) {
        console.warn("Could not fetch metrics:", metricsError);
      }
      
      console.log("Final forecastData with metrics:", data);

      // Forecast data is already stored in MongoDB by the backend
      setForecastData(data);

      toast({
        title: "Forecast generated",
        description: `Successfully generated ${horizon} forecast for ${symbol} (${data.forecast_steps || 0} steps)`,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to generate forecast";
      
      // Clear old forecast data if request failed (to avoid showing stale data)
      setForecastData(null);
      
      toast({
        title: "Forecast failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // Get latest OHLCV values
  const latestOHLCV = ohlcData.length > 0 ? ohlcData[ohlcData.length - 1] : null;

  // Get immediate next forecast prediction
  const getNextForecast = (): number | null => {
    if (!forecastData || ohlcData.length === 0) return null;
    
    // Get forecast values - handle both array and object formats
    let forecastValues: number[] = [];
    
    if (Array.isArray(forecastData.ensemble_forecast)) {
      forecastValues = forecastData.ensemble_forecast;
    } else if (Array.isArray(forecastData.predictions)) {
      forecastValues = forecastData.predictions.map((p: number | { prediction: number }) => {
        if (typeof p === 'object' && p !== null && 'prediction' in p) {
          return p.prediction;
        }
        return typeof p === 'number' ? p : 0;
      });
    } else if (Array.isArray(forecastData.arima_forecast)) {
      forecastValues = forecastData.arima_forecast;
    } else if (forecastData.ensemble_forecast && typeof forecastData.ensemble_forecast === 'object') {
      const obj = forecastData.ensemble_forecast as Record<string, unknown>;
      if (Array.isArray(obj.values)) {
        forecastValues = obj.values as number[];
      } else if (Array.isArray(obj.forecast)) {
        forecastValues = obj.forecast as number[];
      } else if (Array.isArray(obj.predictions)) {
        forecastValues = obj.predictions as number[];
      }
    }
    
    return forecastValues.length > 0 ? forecastValues[0] : null;
  };

  const nextForecast = getNextForecast();

  // Prepare forecast data for candlestick chart
  let forecastArray: Array<{ date: string; prediction: number }> = [];

  if (forecastData && ohlcData.length > 0) {
    // Get forecast values - handle both array and object formats
    let forecastValues: number[] = [];
    
    // Check if ensemble_forecast exists and is an array
    if (Array.isArray(forecastData.ensemble_forecast)) {
      forecastValues = forecastData.ensemble_forecast;
    } 
    // Check if predictions array exists (formatted for frontend)
    else if (Array.isArray(forecastData.predictions)) {
      // Extract prediction values from prediction objects
      forecastValues = forecastData.predictions.map((p: number | { prediction: number }) => {
        if (typeof p === 'object' && p !== null && 'prediction' in p) {
          return p.prediction;
        }
        return typeof p === 'number' ? p : 0;
      });
    }
    // Check if arima_forecast exists and is an array
    else if (Array.isArray(forecastData.arima_forecast)) {
      forecastValues = forecastData.arima_forecast;
    }
    // If ensemble_forecast is an object, try to extract values
    else if (forecastData.ensemble_forecast && typeof forecastData.ensemble_forecast === 'object') {
      // Try to extract array from object (e.g., {values: [...]})
      const obj = forecastData.ensemble_forecast as Record<string, unknown>;
      if (Array.isArray(obj.values)) {
        forecastValues = obj.values as number[];
      } else if (Array.isArray(obj.forecast)) {
        forecastValues = obj.forecast as number[];
      } else if (Array.isArray(obj.predictions)) {
        forecastValues = obj.predictions as number[];
      }
    }
    
    // Debug: Log forecast preparation
    console.log("Preparing forecast array:", {
      hasForecastData: !!forecastData,
      hasOhlcData: ohlcData.length > 0,
      forecastValuesLength: forecastValues.length,
      isArray: Array.isArray(forecastValues),
      ensembleForecast: forecastData.ensemble_forecast,
      arimaForecast: forecastData.arima_forecast,
      predictions: forecastData.predictions,
      forecastValues: forecastValues
    });

    if (forecastValues.length === 0) {
      console.warn("No forecast values available in forecastData");
    }

    // Parse the last date from historical data
    // Handle both date strings and Date objects
    const lastDataPoint = ohlcData[ohlcData.length - 1];
    let lastDate: Date;
    
    if (typeof lastDataPoint.date === 'string') {
      // Try parsing the date string - handle both locale date strings and ISO strings
      const parsed = new Date(lastDataPoint.date);
      if (isNaN(parsed.getTime())) {
        // If locale date string fails, try to parse it manually
        const parts = lastDataPoint.date.split('/');
        if (parts.length === 3) {
          // MM/DD/YYYY format
          lastDate = new Date(parseInt(parts[2]), parseInt(parts[0]) - 1, parseInt(parts[1]));
        } else {
          lastDate = new Date(); // Fallback to current date
        }
      } else {
        lastDate = parsed;
      }
    } else {
      lastDate = new Date(lastDataPoint.date);
    }
    
    // Get data frequency and forecast info from API response
    const dataFrequency = forecastData.data_frequency || 'daily';
    const forecastSteps = forecastData.forecast_steps || forecastValues.length;
    const selectedHorizonHours = parseInt(horizon.replace('h', '')) || 24;
    
    console.log("Forecast configuration:", {
      dataFrequency,
      forecastSteps,
      selectedHorizonHours,
      lastDate: lastDate.toISOString(),
      forecastValuesCount: forecastValues.length
    });
    
    // Generate timestamps based on data frequency and selected horizon
    forecastArray = forecastValues.map((value: number, i: number) => {
      const date = new Date(lastDate);
      
      if (dataFrequency === 'hourly') {
        // For hourly data, each forecast step represents 1 hour
        // Increment by 1 hour per step to maintain hourly intervals
        date.setHours(date.getHours() + (i + 1));
        // Use ISO string format to match historical data format and ensure Plotly compatibility
        return {
          date: date.toISOString(),
          prediction: value
        };
      } else {
        // For daily data, we need to show the forecast based on selected horizon
        // Even though data is daily, user selected a specific hour horizon
        if (selectedHorizonHours < 24) {
          // For 1h, 3h: Show as hours from the last data point
          // Distribute the forecast points across the selected horizon
          const hoursPerStep = selectedHorizonHours / forecastSteps;
          date.setHours(date.getHours() + (i + 1) * hoursPerStep);
          // Use ISO string format for consistency
          return {
            date: date.toISOString(),
            prediction: value
          };
        } else {
          // For 24h, 72h: Show as days (since 24h = 1 day, 72h = 3 days)
          // For better visualization, generate intermediate points even for daily data
          // This helps show the forecast progression
          if (forecastSteps === 1 && selectedHorizonHours > 24) {
            // If only 1 step but horizon is multiple days, create intermediate points
            // For 72h (3 days), create 3 points
            const numPoints = Math.max(1, Math.ceil(selectedHorizonHours / 24));
            const daysPerPoint = selectedHorizonHours / 24 / numPoints;
            const pointDate = new Date(lastDate);
            pointDate.setDate(pointDate.getDate() + (i + 1) * daysPerPoint);
            pointDate.setHours(16, 0, 0, 0); // End of trading day
            return {
              date: pointDate.toISOString(),
              prediction: value // Use same value for all points if only 1 forecast
            };
          } else {
            // Normal case: distribute forecast points across horizon
            const daysPerStep = selectedHorizonHours / 24 / forecastSteps;
            date.setDate(date.getDate() + (i + 1) * daysPerStep);
            // For multi-day forecasts, set time to end of trading day
            if (selectedHorizonHours >= 24) {
              date.setHours(16, 0, 0, 0);
            }
            // Use ISO string format for consistency
            return {
              date: date.toISOString(),
              prediction: value
            };
          }
        }
      }
    });

    console.log("Generated forecast array:", forecastArray);
  } else {
    if (!forecastData) {
      console.log("Forecast array not generated: No forecast data available");
    } else if (ohlcData.length === 0) {
      console.log("Forecast array not generated: No OHLC data available");
    } else {
      console.log("Forecast array not generated:", {
        hasForecastData: !!forecastData,
        hasOhlcData: ohlcData.length > 0,
        forecastDataKeys: forecastData ? Object.keys(forecastData) : []
      });
    }
  }

  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Controls Panel */}
        <Card className="glass-panel lg:w-80 shrink-0">
          <CardHeader>
            <CardTitle className="text-primary">Forecast Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Instrument Type</label>
              <Select value={instrumentType} onValueChange={setInstrumentType}>
                <SelectTrigger className="bg-secondary/50">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="stock">Stock</SelectItem>
                  <SelectItem value="crypto">Crypto</SelectItem>
                  <SelectItem value="forex">Forex</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Symbol</label>
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger className="bg-secondary/50">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {instrumentOptions[instrumentType as keyof typeof instrumentOptions].map((opt) => (
                    <SelectItem key={opt} value={opt}>
                      {opt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Forecast Horizon</label>
              <div className="grid grid-cols-2 gap-2">
                {["1h", "3h", "24h", "72h"].map((h) => (
                  <Button
                    key={h}
                    variant={horizon === h ? "default" : "outline"}
                    onClick={() => setHorizon(h)}
                    className={horizon === h ? "btn-primary" : ""}
                  >
                    {h}
                  </Button>
                ))}
              </div>
            </div>

            <Button
              onClick={handleRunForecast}
              disabled={loading}
              className="w-full btn-primary mt-6"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Run Forecast"
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Chart Panel */}
        <div className="flex-1">
          <CandlestickChart
            historical={ohlcData}
            forecast={forecastArray}
            errors={errorData}
            loading={loading}
            symbol={symbol}
          />
        </div>
      </div>

      {/* OHLCV and Next Forecast Display */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Latest OHLCV Values */}
        <Card className="glass-panel">
          <CardHeader>
            <CardTitle className="text-primary">Latest OHLCV Values</CardTitle>
          </CardHeader>
          <CardContent>
            {latestOHLCV ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Open</p>
                    <p className="text-2xl font-bold text-foreground">${latestOHLCV.open.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">High</p>
                    <p className="text-2xl font-bold text-green-500">${latestOHLCV.high.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Low</p>
                    <p className="text-2xl font-bold text-red-500">${latestOHLCV.low.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Close</p>
                    <p className="text-2xl font-bold text-primary">${latestOHLCV.close.toFixed(2)}</p>
                  </div>
                </div>
                <div className="pt-2 border-t border-border">
                  <p className="text-xs text-muted-foreground">
                    Date: {new Date(latestOHLCV.date).toLocaleString()}
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground">No OHLCV data available. Please select a symbol and ensure data is loaded.</p>
            )}
          </CardContent>
        </Card>

        {/* Next Forecast Prediction */}
        <Card className="glass-panel">
          <CardHeader>
            <CardTitle className="text-primary">Next Forecast Prediction</CardTitle>
          </CardHeader>
          <CardContent>
            {nextForecast !== null ? (
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Immediate Next Forecast</p>
                  <p className="text-4xl font-bold text-primary mb-2">${nextForecast.toFixed(2)}</p>
                  {latestOHLCV && (
                    <>
                      <div className="flex items-center gap-2 mt-3">
                        <p className="text-sm text-muted-foreground">Change from Close:</p>
                        <p className={`text-lg font-semibold ${
                          nextForecast >= latestOHLCV.close ? 'text-green-500' : 'text-red-500'
                        }`}>
                          {nextForecast >= latestOHLCV.close ? '+' : ''}
                          ${(nextForecast - latestOHLCV.close).toFixed(2)} (
                          {nextForecast >= latestOHLCV.close ? '+' : ''}
                          {(((nextForecast - latestOHLCV.close) / latestOHLCV.close) * 100).toFixed(2)}%)
                        </p>
                      </div>
                      {Math.abs(nextForecast - latestOHLCV.close) > latestOHLCV.close * 0.05 && (
                        <div className="mt-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-xs text-yellow-600 dark:text-yellow-400">
                          <p className="font-semibold mb-1">⚠️ Large Prediction Change</p>
                          <p>
                            The model predicts a {Math.abs(((nextForecast - latestOHLCV.close) / latestOHLCV.close) * 100).toFixed(1)}% change. 
                            This may indicate:
                          </p>
                          <ul className="list-disc list-inside mt-1 space-y-0.5">
                            <li>Recent data shows a strong trend</li>
                            <li>Model detected a pattern shift</li>
                            <li>Consider reviewing recent price movements</li>
                          </ul>
                        </div>
                      )}
                    </>
                  )}
                </div>
                <div className="pt-2 border-t border-border">
                  <p className="text-xs text-muted-foreground">
                    Forecast Horizon: {horizon} | Symbol: {symbol}
                  </p>
                  {forecastData?.model_type && (
                    <p className="text-xs text-muted-foreground">
                      Model: {forecastData.model_type}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-muted-foreground">No forecast available.</p>
                <p className="text-sm text-muted-foreground">
                  Click "Run Forecast" to generate predictions.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Individual Model Predictions Panel */}
      {forecastData && (
        <Card className="glass-panel">
          <CardHeader>
            <CardTitle className="text-primary">Individual Model Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {forecastData.arima_forecast && Array.isArray(forecastData.arima_forecast) && forecastData.arima_forecast.length > 0 ? (
                <div className="p-3 bg-secondary/20 rounded">
                  <h3 className="text-sm font-semibold mb-2">ARIMA</h3>
                  <p className="text-xs text-muted-foreground mb-1">First Prediction:</p>
                  {forecastData.arima_forecast[0] < 0 ? (
                    <div>
                      <p className="text-lg font-bold text-red-500">${forecastData.arima_forecast[0].toFixed(2)} ⚠️</p>
                      <p className="text-xs text-red-400 mt-1">ERROR: Negative price detected! This indicates a conversion bug.</p>
                    </div>
                  ) : (
                    <>
                      <p className="text-lg font-bold">${forecastData.arima_forecast[0].toFixed(2)}</p>
                      {latestOHLCV && (
                        <p className="text-xs mt-1">
                          Change: {((forecastData.arima_forecast[0] - latestOHLCV.close) / latestOHLCV.close * 100).toFixed(2)}%
                        </p>
                      )}
                    </>
                  )}
                  {forecastData.metrics?.ARIMA && (forecastData.metrics.ARIMA.RMSE > 0 || forecastData.metrics.ARIMA.MAE > 0 || forecastData.metrics.ARIMA.MAPE > 0) && (
                    <div className="mt-3 pt-3 border-t border-border/50">
                      <p className="text-xs text-muted-foreground mb-1">Metrics:</p>
                      <p className="text-xs">RMSE: {forecastData.metrics.ARIMA.RMSE > 0 ? forecastData.metrics.ARIMA.RMSE.toFixed(4) : 'N/A'}</p>
                      <p className="text-xs">MAE: {forecastData.metrics.ARIMA.MAE > 0 ? forecastData.metrics.ARIMA.MAE.toFixed(4) : 'N/A'}</p>
                      <p className="text-xs">MAPE: {forecastData.metrics.ARIMA.MAPE > 0 ? forecastData.metrics.ARIMA.MAPE.toFixed(2) : 'N/A'}%</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-3 bg-secondary/10 rounded border border-dashed">
                  <h3 className="text-sm font-semibold mb-2 text-muted-foreground">ARIMA</h3>
                  <p className="text-xs text-muted-foreground">No prediction available</p>
                </div>
              )}
              
              {forecastData.lstm_forecast && Array.isArray(forecastData.lstm_forecast) && forecastData.lstm_forecast.length > 0 ? (
                <div className="p-3 bg-secondary/20 rounded">
                  <h3 className="text-sm font-semibold mb-2">LSTM</h3>
                  <p className="text-xs text-muted-foreground mb-1">First Prediction:</p>
                  <p className="text-lg font-bold">${forecastData.lstm_forecast[0].toFixed(2)}</p>
                  {latestOHLCV && (
                    <p className="text-xs mt-1">
                      Change: {((forecastData.lstm_forecast[0] - latestOHLCV.close) / latestOHLCV.close * 100).toFixed(2)}%
                    </p>
                  )}
                  {forecastData.metrics?.LSTM && (forecastData.metrics.LSTM.RMSE > 0 || forecastData.metrics.LSTM.MAE > 0 || forecastData.metrics.LSTM.MAPE > 0) && (
                    <div className="mt-3 pt-3 border-t border-border/50">
                      <p className="text-xs text-muted-foreground mb-1">Metrics:</p>
                      <p className="text-xs">RMSE: {forecastData.metrics.LSTM.RMSE > 0 ? forecastData.metrics.LSTM.RMSE.toFixed(4) : 'N/A'}</p>
                      <p className="text-xs">MAE: {forecastData.metrics.LSTM.MAE > 0 ? forecastData.metrics.LSTM.MAE.toFixed(4) : 'N/A'}</p>
                      <p className="text-xs">MAPE: {forecastData.metrics.LSTM.MAPE > 0 ? forecastData.metrics.LSTM.MAPE.toFixed(2) : 'N/A'}%</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-3 bg-secondary/10 rounded border border-dashed">
                  <h3 className="text-sm font-semibold mb-2 text-muted-foreground">LSTM</h3>
                  <p className="text-xs text-muted-foreground">
                    {forecastData.lstm_forecast ? 'Invalid format' : 'No prediction available'}
                  </p>
                </div>
              )}
              
              {forecastData.gru_forecast && Array.isArray(forecastData.gru_forecast) && forecastData.gru_forecast.length > 0 ? (
                <div className="p-3 bg-secondary/20 rounded">
                  <h3 className="text-sm font-semibold mb-2">GRU</h3>
                  <p className="text-xs text-muted-foreground mb-1">First Prediction:</p>
                  <p className="text-lg font-bold">${forecastData.gru_forecast[0].toFixed(2)}</p>
                  {latestOHLCV && (
                    <p className="text-xs mt-1">
                      Change: {((forecastData.gru_forecast[0] - latestOHLCV.close) / latestOHLCV.close * 100).toFixed(2)}%
                    </p>
                  )}
                  {forecastData.metrics?.GRU && (forecastData.metrics.GRU.RMSE > 0 || forecastData.metrics.GRU.MAE > 0 || forecastData.metrics.GRU.MAPE > 0) && (
                    <div className="mt-3 pt-3 border-t border-border/50">
                      <p className="text-xs text-muted-foreground mb-1">Metrics:</p>
                      <p className="text-xs">RMSE: {forecastData.metrics.GRU.RMSE > 0 ? forecastData.metrics.GRU.RMSE.toFixed(4) : 'N/A'}</p>
                      <p className="text-xs">MAE: {forecastData.metrics.GRU.MAE > 0 ? forecastData.metrics.GRU.MAE.toFixed(4) : 'N/A'}</p>
                      <p className="text-xs">MAPE: {forecastData.metrics.GRU.MAPE > 0 ? forecastData.metrics.GRU.MAPE.toFixed(2) : 'N/A'}%</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-3 bg-secondary/10 rounded border border-dashed">
                  <h3 className="text-sm font-semibold mb-2 text-muted-foreground">GRU</h3>
                  <p className="text-xs text-muted-foreground">No prediction available</p>
                </div>
              )}
              
              {forecastData.transformer_forecast && Array.isArray(forecastData.transformer_forecast) && forecastData.transformer_forecast.length > 0 ? (
                <div className="p-3 bg-secondary/20 rounded">
                  <h3 className="text-sm font-semibold mb-2">Transformer</h3>
                  <p className="text-xs text-muted-foreground mb-1">First Prediction:</p>
                  <p className="text-lg font-bold">${forecastData.transformer_forecast[0].toFixed(2)}</p>
                  {latestOHLCV && (
                    <p className="text-xs mt-1">
                      Change: {((forecastData.transformer_forecast[0] - latestOHLCV.close) / latestOHLCV.close * 100).toFixed(2)}%
                    </p>
                  )}
                  {forecastData.metrics?.Transformer && (forecastData.metrics.Transformer.RMSE > 0 || forecastData.metrics.Transformer.MAE > 0 || forecastData.metrics.Transformer.MAPE > 0) && (
                    <div className="mt-3 pt-3 border-t border-border/50">
                      <p className="text-xs text-muted-foreground mb-1">Metrics:</p>
                      <p className="text-xs">RMSE: {forecastData.metrics.Transformer.RMSE > 0 ? forecastData.metrics.Transformer.RMSE.toFixed(4) : 'N/A'}</p>
                      <p className="text-xs">MAE: {forecastData.metrics.Transformer.MAE > 0 ? forecastData.metrics.Transformer.MAE.toFixed(4) : 'N/A'}</p>
                      <p className="text-xs">MAPE: {forecastData.metrics.Transformer.MAPE > 0 ? forecastData.metrics.Transformer.MAPE.toFixed(2) : 'N/A'}%</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-3 bg-secondary/10 rounded border border-dashed">
                  <h3 className="text-sm font-semibold mb-2 text-muted-foreground">Transformer</h3>
                  <p className="text-xs text-muted-foreground">No prediction available</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Metrics Panel */}
      {forecastData && forecastData.metrics && (
        <Card className="glass-panel">
          <CardHeader>
            <CardTitle className="text-primary">Evaluation Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {forecastData.metrics.ARIMA && (
                <div>
                  <h3 className="text-lg font-semibold mb-2">ARIMA Model</h3>
                  <p>RMSE: {forecastData.metrics.ARIMA.RMSE.toFixed(4)}</p>
                  <p>MAE: {forecastData.metrics.ARIMA.MAE.toFixed(4)}</p>
                  <p>MAPE: {forecastData.metrics.ARIMA.MAPE.toFixed(2)}%</p>
                </div>
              )}
              {forecastData.metrics.LSTM && (
                <div>
                  <h3 className="text-lg font-semibold mb-2">LSTM Model</h3>
                  <p>RMSE: {forecastData.metrics.LSTM.RMSE.toFixed(4)}</p>
                  <p>MAE: {forecastData.metrics.LSTM.MAE.toFixed(4)}</p>
                  <p>MAPE: {forecastData.metrics.LSTM.MAPE.toFixed(2)}%</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Dashboard;
