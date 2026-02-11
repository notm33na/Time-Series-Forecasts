import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Loader2, TrendingUp, TrendingDown, DollarSign, BarChart3 } from "lucide-react";
import Plot from "react-plotly.js";

interface PortfolioSnapshot {
  timestamp: string;
  cash: number;
  holdings: number;
  total_value: number;
  daily_return: number;
  volatility: number;
  sharpe_ratio: number;
}

interface Trade {
  id: number;
  timestamp: string;
  action: string;
  quantity: number;
  price: number;
  note?: string;
}

const Portfolio = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [symbol, setSymbol] = useState<string>("AAPL");
  const [forecastPrice, setForecastPrice] = useState<string>("");
  const [positionSize, setPositionSize] = useState<string>("0.1");
  const [portfolioHistory, setPortfolioHistory] = useState<PortfolioSnapshot[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [portfolioMetrics, setPortfolioMetrics] = useState<Record<string, unknown> | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<Record<string, unknown> | null>(null);
  const [manualTradeQuantity, setManualTradeQuantity] = useState<string>("");
  const [currentPosition, setCurrentPosition] = useState<{quantity: number} | null>(null);

  const instrumentOptions = {
    stock: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
    crypto: ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
    forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
  };

  const fetchPortfolioHistory = useCallback(async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/portfolio/history/${symbol}?limit=100`);
      
      if (response.ok) {
        const result = await response.json() as { history?: PortfolioSnapshot[] };
        if (result.history) {
          setPortfolioHistory(result.history);
        }
      }
    } catch (error) {
      console.error("Error fetching portfolio history:", error);
    }
  }, [symbol]);

  const fetchTrades = useCallback(async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/portfolio/trades/${symbol}?limit=50`);
      
      if (response.ok) {
        const result = await response.json() as { trades?: Trade[] };
        if (result.trades) {
          setTrades(result.trades);
        }
      }
    } catch (error) {
      console.error("Error fetching trades:", error);
    }
  }, [symbol]);

  const fetchCurrentPrice = useCallback(async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/data/prices/${symbol}?limit=1`);
      
      if (response.ok) {
        const result = await response.json() as { prices?: Array<{ close: number }> };
        if (result.prices && result.prices.length > 0) {
          setCurrentPrice(result.prices[0].close);
        }
      }
    } catch (error) {
      console.error("Error fetching current price:", error);
    }
  }, [symbol]);

  const fetchCurrentPosition = useCallback(async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/portfolio/positions?portfolio_id=default`);
      
      if (response.ok) {
        const result = await response.json() as { success?: boolean; data?: Array<{symbol: string; quantity: number}> };
        if (result.success && result.data) {
          const position = result.data.find(p => p.symbol === symbol);
          setCurrentPosition(position ? { quantity: position.quantity } : null);
        }
      }
    } catch (error) {
      console.error("Error fetching current position:", error);
      setCurrentPosition(null);
    }
  }, [symbol]);

  const fetchDashboardData = useCallback(async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/portfolio/dashboard/${symbol}?limit=100`);
      
      if (response.ok) {
        const result = await response.json();
        console.log("Dashboard data received:", {
          hasHistory: !!result.portfolio_history,
          historyLength: result.portfolio_history?.length || 0,
          hasTrades: !!result.trades,
          tradesLength: result.trades?.length || 0,
          hasMetrics: !!result.portfolio_metrics,
          hasPerformance: !!result.performance_metrics,
          sampleHistory: result.portfolio_history?.[0] || null
        });
        
        if (result.portfolio_history && result.portfolio_history.length > 0) {
          // Ensure timestamps are properly formatted
          const formattedHistory = result.portfolio_history.map((snapshot: PortfolioSnapshot & {date?: string; value?: number}) => ({
            ...snapshot,
            timestamp: snapshot.timestamp || snapshot.date || new Date().toISOString(),
            total_value: snapshot.total_value || snapshot.value || 0
          }));
          setPortfolioHistory(formattedHistory);
          console.log("Portfolio history set:", formattedHistory.length, "records");
        } else {
          console.warn("No portfolio history in response");
        }
        if (result.trades) {
          setTrades(result.trades);
        }
        if (result.portfolio_metrics) {
          setPortfolioMetrics(result.portfolio_metrics);
        }
        if (result.performance_metrics) {
          setPerformanceMetrics(result.performance_metrics);
        }
        if (result.current_price) {
          setCurrentPrice(result.current_price);
        }
      } else {
        console.error("Dashboard fetch failed:", response.status, response.statusText);
        // Fallback to individual fetches
        fetchPortfolioHistory();
        fetchTrades();
        fetchCurrentPrice();
      }
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
      // Fallback to individual fetches
      fetchPortfolioHistory();
      fetchTrades();
      fetchCurrentPrice();
    }
  }, [symbol, fetchPortfolioHistory, fetchTrades, fetchCurrentPrice]);

  useEffect(() => {
    fetchDashboardData();
    fetchCurrentPosition();
  }, [fetchDashboardData, fetchCurrentPosition]);

  const handleExecuteStrategy = async () => {
    if (!forecastPrice || isNaN(parseFloat(forecastPrice))) {
      toast({
        title: "Invalid forecast price",
        description: "Please enter a valid forecast price",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      
      const requestBody = {
        symbol: symbol,
        forecast_price: parseFloat(forecastPrice),
        position_size: parseFloat(positionSize),
        strategy: "momentum", // Default strategy
        portfolio_id: "default",
      };
      
      console.log("Execute Strategy Request:", {
        url: `${apiUrl}/api/portfolio/execute-strategy`,
        method: "POST",
        body: requestBody
      });
      
      const response = await fetch(`${apiUrl}/api/portfolio/execute-strategy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      console.log("Execute Strategy Response Status:", response.status, response.statusText);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error("Execute Strategy Error Response:", errorData);
        throw new Error(errorData.message || errorData.detail || 'Failed to execute strategy');
      }

      const result = await response.json();
      
      // Debug: Log full response
      console.log("Execute Strategy API Response:", result);
      console.log("Response keys:", Object.keys(result));
      
      // Check if strategy actually executed a trade
      const actionTaken = result.forecast_analysis?.action_taken || result.action_taken || "hold";
      const message = result.message || `Strategy executed: ${actionTaken}`;
      const portfolioValue = result.portfolio_value || 0;
      const success = result.success !== false; // Default to true if not specified
      
      console.log("Parsed values:", {
        actionTaken,
        message,
        portfolioValue,
        success,
        hasTransaction: !!result.transaction
      });
      
      if (!success) {
        toast({
          title: "Strategy execution failed",
          description: message || "Strategy did not execute successfully",
          variant: "destructive",
        });
      } else if (actionTaken === "hold") {
        toast({
          title: "Strategy executed (Hold)",
          description: message,
        });
      } else {
        // Trade was executed
        const transactionInfo = result.transaction 
          ? ` | ${result.transaction.action} ${result.transaction.quantity?.toFixed(4) || ''} @ $${result.transaction.price?.toFixed(2) || ''}`
          : '';
        toast({
          title: "Strategy executed",
          description: `${message}${transactionInfo} | Portfolio Value: $${portfolioValue.toFixed(2)}`,
        });
      }

      // Refresh data to show updated portfolio
      // Add small delay to ensure database is updated
      setTimeout(() => {
        fetchDashboardData();
      }, 500);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to execute strategy";
      toast({
        title: "Strategy execution failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleManualTrade = async (action: "BUY" | "SELL") => {
    if (!currentPrice) {
      toast({
        title: "No current price",
        description: "Unable to get current price for the symbol",
        variant: "destructive",
      });
      return;
    }

    // Validate quantity
    let quantity: number | undefined = undefined;
    if (manualTradeQuantity) {
      quantity = parseFloat(manualTradeQuantity);
      if (isNaN(quantity) || quantity <= 0) {
        toast({
          title: "Invalid quantity",
          description: "Please enter a valid quantity greater than 0",
          variant: "destructive",
        });
        return;
      }
      
      // For SELL, check if quantity doesn't exceed position
      if (action === "SELL" && currentPosition) {
        if (quantity > currentPosition.quantity) {
          toast({
            title: "Insufficient shares",
            description: `You only have ${currentPosition.quantity.toFixed(4)} shares. Cannot sell ${quantity.toFixed(4)}.`,
            variant: "destructive",
          });
          return;
        }
      }
    } else {
      // If no quantity specified, use default behavior (all position for SELL, 10% cash for BUY)
      if (action === "SELL" && currentPosition) {
        quantity = currentPosition.quantity;
      }
      // For BUY, quantity will be calculated by backend if not provided
    }

    setLoading(true);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const requestBody: {
        symbol: string;
        action: string;
        price: number;
        quantity?: number;
      } = {
        symbol: symbol,
        action: action,
        price: currentPrice,
      };
      
      // Only include quantity if specified
      if (quantity !== undefined) {
        requestBody.quantity = quantity;
      }
      
      const response = await fetch(`${apiUrl}/api/portfolio/trade`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || errorData.detail || `Failed to ${action}`);
      }

      const result = await response.json();
      
      // Show detailed message from API
      const message = result.message || `Successfully executed ${action} order`;
      const portfolioValue = result.portfolio_value || 0;
      
      toast({
        title: `${action} executed`,
        description: `${message} | Portfolio Value: $${portfolioValue.toFixed(2)}`,
      });

      // Clear quantity input
      setManualTradeQuantity("");

      // Refresh data to show updated portfolio
      // Add small delay to ensure database is updated
      setTimeout(() => {
        fetchDashboardData();
        fetchCurrentPosition();
      }, 500);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : `Failed to ${action}`;
      toast({
        title: `${action} failed`,
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const latestSnapshot = portfolioHistory[portfolioHistory.length - 1];

  // Prepare portfolio growth chart data
  const portfolioChartData = portfolioHistory.length > 0 ? (() => {
    // Sort by timestamp to ensure chronological order
    const sortedHistory = [...portfolioHistory].sort((a, b) => {
      const dateA = new Date(a.timestamp || 0).getTime();
      const dateB = new Date(b.timestamp || 0).getTime();
      return dateA - dateB;
    });
    
    const xValues = sortedHistory.map(s => {
      // Convert timestamp to Date object for Plotly
      const timestamp = s.timestamp;
      if (typeof timestamp === 'string') {
        const date = new Date(timestamp);
        // Check if date is valid
        if (isNaN(date.getTime())) {
          console.warn("Invalid timestamp:", timestamp);
          return new Date();
        }
        return date;
      }
      return new Date();
    });
    
    const yValues = sortedHistory.map(s => {
      const value = s.total_value || 0;
      return typeof value === 'number' ? value : parseFloat(String(value)) || 0;
    });
    
    console.log("Chart data prepared:", {
      dataPoints: xValues.length,
      firstX: xValues[0],
      firstY: yValues[0],
      lastX: xValues[xValues.length - 1],
      lastY: yValues[yValues.length - 1]
    });
    
    return [{
      x: xValues,
      y: yValues,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: 'Portfolio Value',
      line: { color: '#8b5cf6', width: 3 },
      marker: { size: 8, color: '#8b5cf6' },
      // Only fill if we have more than 2 points, otherwise it covers the line
      fill: xValues.length > 2 ? 'tozeroy' : 'none' as const,
      fillcolor: 'rgba(139, 92, 246, 0.1)',
      hovertemplate: '<b>Portfolio Value</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>',
    }];
  })() : [];

  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Controls Panel */}
        <Card className="glass-panel lg:w-80 shrink-0">
          <CardHeader>
            <CardTitle className="text-primary">Portfolio Management</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Symbol</Label>
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger className="bg-secondary/50">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.values(instrumentOptions).flat().map((opt) => (
                    <SelectItem key={opt} value={opt}>
                      {opt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {currentPrice && (
              <div className="p-3 bg-secondary/30 rounded-lg">
                <Label className="text-sm text-muted-foreground">Current Price</Label>
                <p className="text-2xl font-bold">${currentPrice.toFixed(2)}</p>
              </div>
            )}

            <div className="space-y-2">
              <Label>Forecast Price (for strategy)</Label>
              <Input
                type="number"
                step="0.01"
                value={forecastPrice}
                onChange={(e) => setForecastPrice(e.target.value)}
                placeholder="Enter forecast price"
                className="bg-secondary/50"
              />
            </div>

            <div className="space-y-2">
              <Label>Position Size (0.0 - 1.0)</Label>
              <Input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={positionSize}
                onChange={(e) => setPositionSize(e.target.value)}
                className="bg-secondary/50"
              />
            </div>

            <Button
              onClick={handleExecuteStrategy}
              disabled={loading || !forecastPrice}
              className="w-full btn-primary"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Executing...
                </>
              ) : (
                "Execute Strategy"
              )}
            </Button>

            <div className="border-t pt-4 space-y-2">
              <Label>Manual Trading</Label>
              <div className="space-y-2">
                <div className="space-y-1">
                  <Label className="text-sm">Quantity (optional)</Label>
                  <Input
                    type="number"
                    step="0.0001"
                    min="0"
                    value={manualTradeQuantity}
                    onChange={(e) => setManualTradeQuantity(e.target.value)}
                    placeholder={
                      currentPosition 
                        ? `Max: ${currentPosition.quantity.toFixed(4)} (leave empty to sell all)`
                        : "Leave empty for default"
                    }
                    className="bg-secondary/50"
                  />
                  {currentPosition && (
                    <p className="text-xs text-muted-foreground">
                      Current position: {currentPosition.quantity.toFixed(4)} shares
                    </p>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    onClick={() => handleManualTrade("BUY")}
                    disabled={loading}
                    variant="default"
                    className="bg-green-600 hover:bg-green-700"
                  >
                    Buy
                  </Button>
                  <Button
                    onClick={() => handleManualTrade("SELL")}
                    disabled={loading || (currentPosition && currentPosition.quantity <= 0)}
                    variant="default"
                    className="bg-red-600 hover:bg-red-700"
                  >
                    Sell
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Portfolio Metrics */}
        <Card className="flex-1 glass-panel">
          <CardHeader>
            <CardTitle className="text-primary">Portfolio Performance</CardTitle>
          </CardHeader>
          <CardContent>
            {latestSnapshot ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="p-4 bg-secondary/30 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <DollarSign className="h-5 w-5 text-primary" />
                    <Label className="text-sm text-muted-foreground">Total Value</Label>
                  </div>
                  <p className="text-2xl font-bold">${latestSnapshot.total_value.toFixed(2)}</p>
                </div>
                <div className="p-4 bg-secondary/30 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingUp className="h-5 w-5 text-green-500" />
                    <Label className="text-sm text-muted-foreground">Daily Return</Label>
                  </div>
                  <p className={`text-2xl font-bold ${latestSnapshot.daily_return >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {(latestSnapshot.daily_return * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground">No portfolio data available. Execute a strategy to start tracking.</p>
            )}

            {/* Performance Metrics */}
            {performanceMetrics && (
              <div className="mt-6 p-4 bg-secondary/30 rounded-lg">
                <h3 className="text-lg font-semibold mb-4">Performance Summary</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {performanceMetrics.total_return_pct !== undefined && typeof performanceMetrics.total_return_pct === 'number' && (
                    <div>
                      <Label className="text-sm text-muted-foreground">Total Return</Label>
                      <p className={`text-xl font-bold ${performanceMetrics.total_return_pct >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {performanceMetrics.total_return_pct.toFixed(2)}%
                      </p>
                    </div>
                  )}
                  {performanceMetrics.max_drawdown_pct !== undefined && typeof performanceMetrics.max_drawdown_pct === 'number' && (
                    <div>
                      <Label className="text-sm text-muted-foreground">Max Drawdown</Label>
                      <p className="text-xl font-bold text-red-500">{performanceMetrics.max_drawdown_pct.toFixed(2)}%</p>
                    </div>
                  )}
                  {performanceMetrics.win_rate !== undefined && typeof performanceMetrics.win_rate === 'number' && (
                    <div>
                      <Label className="text-sm text-muted-foreground">Win Rate</Label>
                      <p className="text-xl font-bold">{performanceMetrics.win_rate.toFixed(1)}%</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Portfolio Growth Chart */}
            {portfolioChartData.length > 0 ? (
              <div className="h-96">
                <Plot
                  data={portfolioChartData}
                  layout={{
                    title: "Portfolio Growth Over Time",
                    xaxis: { 
                      title: "Date",
                      type: 'date',
                      tickformat: '%Y-%m-%d %H:%M',
                      showgrid: true,
                      gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: { 
                      title: "Value ($)",
                      tickformat: '$,.0f',
                      showgrid: true,
                      gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    height: 400,
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    hovermode: 'x unified',
                    showlegend: true,
                    font: {
                      color: '#ffffff'
                    }
                  }}
                  config={{
                    displayModeBar: true,
                    displaylogo: false,
                    responsive: true,
                  }}
                  style={{ width: '100%', height: '100%' }}
                />
              </div>
            ) : (
              <div className="h-96 flex items-center justify-center text-muted-foreground">
                <p>No portfolio history data available for chart</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Trade History */}
      <Card className="glass-panel">
        <CardHeader>
          <CardTitle className="text-primary">Trade History</CardTitle>
        </CardHeader>
        <CardContent>
          {trades.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Time</th>
                    <th className="text-left p-2">Action</th>
                    <th className="text-left p-2">Quantity</th>
                    <th className="text-left p-2">Price</th>
                    <th className="text-left p-2">Note</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((trade) => (
                    <tr key={trade.id} className="border-b hover:bg-secondary/30">
                      <td className="p-2">{new Date(trade.timestamp).toLocaleString()}</td>
                      <td className="p-2">
                        <span className={`px-2 py-1 rounded ${
                          trade.action === 'BUY' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'
                        }`}>
                          {trade.action}
                        </span>
                      </td>
                      <td className="p-2">{trade.quantity.toFixed(4)}</td>
                      <td className="p-2">${trade.price.toFixed(2)}</td>
                      <td className="p-2 text-sm text-muted-foreground">{trade.note || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-muted-foreground">No trades yet. Execute a strategy or manual trade to see history.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Portfolio;

