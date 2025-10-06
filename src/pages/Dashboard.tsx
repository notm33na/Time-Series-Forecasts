import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import { Loader2 } from "lucide-react";

const Dashboard = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [instrumentType, setInstrumentType] = useState<string>("stock");
  const [symbol, setSymbol] = useState<string>("AAPL");
  const [horizon, setHorizon] = useState<string>("24h");
  const [ohlcData, setOhlcData] = useState<any[]>([]);
  const [forecastData, setForecastData] = useState<any>(null);

  const instrumentOptions = {
    stock: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
    crypto: ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
    forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
  };

  useEffect(() => {
    fetchOHLCData();
  }, [symbol]);

  const fetchOHLCData = async () => {
    const { data, error } = await supabase
      .from("ohlc")
      .select("*")
      .eq("instrument", symbol)
      .order("ts", { ascending: true });

    if (error) {
      console.error("Error fetching OHLC data:", error);
      return;
    }

    if (data) {
      const chartData = data.map((item) => ({
        time: new Date(item.ts).toLocaleDateString(),
        price: Number(item.close),
        high: Number(item.high),
        low: Number(item.low),
      }));
      setOhlcData(chartData);
    }
  };

  const handleRunForecast = async () => {
    setLoading(true);
    
    try {
      // Get user settings for API configuration
      const { data: userData } = await supabase.auth.getUser();
      if (!userData.user) throw new Error("Not authenticated");

      const { data: settings } = await supabase
        .from("user_settings")
        .select("model_api_url, api_key_encrypted")
        .eq("user_id", userData.user.id)
        .single();

      if (!settings?.model_api_url) {
        toast({
          title: "API not configured",
          description: "Please configure your model API URL in Settings first.",
          variant: "destructive",
        });
        setLoading(false);
        return;
      }

      // Mock forecast data for demonstration
      // In production, this would call the external API
      const mockForecast = {
        timestamps: Array.from({ length: 5 }, (_, i) => {
          const date = new Date();
          date.setDate(date.getDate() + i + 1);
          return date.toISOString();
        }),
        forecast: Array.from({ length: 5 }, () => 
          ohlcData[ohlcData.length - 1]?.price * (0.98 + Math.random() * 0.04)
        ),
        lower: Array.from({ length: 5 }, () => 
          ohlcData[ohlcData.length - 1]?.price * (0.95 + Math.random() * 0.02)
        ),
        upper: Array.from({ length: 5 }, () => 
          ohlcData[ohlcData.length - 1]?.price * (1.0 + Math.random() * 0.05)
        ),
      };

      // Store forecast in database
      await supabase.from("forecasts").insert({
        user_id: userData.user.id,
        instrument: symbol,
        horizon,
        payload: mockForecast,
      });

      setForecastData(mockForecast);

      toast({
        title: "Forecast generated",
        description: `Successfully generated ${horizon} forecast for ${symbol}`,
      });
    } catch (error: any) {
      toast({
        title: "Forecast failed",
        description: error.message || "Failed to generate forecast",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const combinedData = [...ohlcData];
  if (forecastData) {
    forecastData.timestamps.forEach((timestamp: string, i: number) => {
      combinedData.push({
        time: new Date(timestamp).toLocaleDateString(),
        price: null,
        forecast: forecastData.forecast[i],
        lower: forecastData.lower[i],
        upper: forecastData.upper[i],
      });
    });
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-col lg:flex-row gap-6">
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
        <Card className="glass-panel flex-1">
          <CardHeader>
            <CardTitle className="text-primary">
              {symbol} - Price & Forecast
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={combinedData}>
                <defs>
                  <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--accent))" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(var(--accent))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
                <XAxis
                  dataKey="time"
                  stroke="hsl(var(--chart-text))"
                  tick={{ fill: "hsl(var(--chart-text))" }}
                />
                <YAxis
                  stroke="hsl(var(--chart-text))"
                  tick={{ fill: "hsl(var(--chart-text))" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="price"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  fill="url(#colorPrice)"
                  name="Historical"
                />
                <Area
                  type="monotone"
                  dataKey="forecast"
                  stroke="hsl(var(--accent))"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  fill="url(#colorForecast)"
                  name="Forecast"
                />
                <Area
                  type="monotone"
                  dataKey="upper"
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth={1}
                  fill="none"
                  name="Upper Bound"
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth={1}
                  fill="none"
                  name="Lower Bound"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
