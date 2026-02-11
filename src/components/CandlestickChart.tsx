import React from "react";
import Plot from "react-plotly.js";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface CandlestickChartProps {
  historical: Array<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>;
  forecast: Array<{ date: string; prediction: number }>;
  errors?: Array<{ date: string; actual: number; predicted: number; error: number; errorPercent: number }>;
  loading: boolean;
  symbol: string;
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  historical,
  forecast,
  errors,
  loading,
  symbol,
}) => {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{symbol} Price Chart</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-96 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (!historical || historical.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{symbol} Price Chart</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-96 text-muted-foreground">
            No data available
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare candlestick data - Plotly expects arrays for OHLC
  // Convert date strings to Date objects for proper datetime handling
  const parseDate = (dateStr: string): Date => {
    // Try parsing as ISO string first
    const isoDate = new Date(dateStr);
    if (!isNaN(isoDate.getTime())) {
      return isoDate;
    }
    // Try parsing locale date strings (MM/DD/YYYY)
    const parts = dateStr.split('/');
    if (parts.length === 3) {
      return new Date(parseInt(parts[2]), parseInt(parts[0]) - 1, parseInt(parts[1]));
    }
    // Fallback to current date
    return new Date();
  };
  
  // Sort historical data by date to ensure proper chronological order
  const sortedHistorical = [...historical].sort((a, b) => {
    const dateA = typeof a.date === 'string' ? parseDate(a.date) : new Date(a.date);
    const dateB = typeof b.date === 'string' ? parseDate(b.date) : new Date(b.date);
    return dateA.getTime() - dateB.getTime();
  });

  const dates = sortedHistorical.map(item => {
    if (typeof item.date === 'string') {
      return parseDate(item.date);
    }
    return new Date(item.date);
  });
  const opens = sortedHistorical.map(item => item.open);
  const highs = sortedHistorical.map(item => item.high);
  const lows = sortedHistorical.map(item => item.low);
  const closes = sortedHistorical.map(item => item.close);

  const candlestickTrace = {
    x: dates,
    open: opens,
    high: highs,
    low: lows,
    close: closes,
    type: 'candlestick' as const,
    name: 'Price',
    increasing: { line: { color: '#26a69a' } },
    decreasing: { line: { color: '#ef5350' } },
    xaxis: 'x',
    yaxis: 'y'
  };

  // Add a close price line overlay for smoother visualization (like professional trading apps)
  const closePriceLineTrace = {
    x: dates,
    y: closes,
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'Close Price',
    line: { 
      color: '#3b82f6', 
      width: 2,
      shape: 'linear' as const
    },
    xaxis: 'x',
    yaxis: 'y',
    showlegend: false, // Hide from legend since it's just for visual enhancement
    hoverinfo: 'skip' as const
  };

  // Prepare forecast line - convert date strings to Date objects
  // Connect forecast to the last historical point for smooth transition
  let forecastTrace = null;
  
  console.log("CandlestickChart - Forecast data:", {
    forecastLength: forecast.length,
    forecast: forecast,
    historicalLength: historical.length,
    lastHistoricalDate: dates.length > 0 ? dates[dates.length - 1] : null,
    lastHistoricalClose: closes.length > 0 ? closes[closes.length - 1] : null
  });
  
  if (forecast.length > 0) {
    const lastHistoricalDate = dates[dates.length - 1];
    const lastHistoricalClose = closes[closes.length - 1];
    
    // Sort forecast data by date
    const sortedForecast = [...forecast].sort((a, b) => {
      const dateA = typeof a.date === 'string' ? parseDate(a.date) : new Date(a.date);
      const dateB = typeof b.date === 'string' ? parseDate(b.date) : new Date(b.date);
      return dateA.getTime() - dateB.getTime();
    });

    const forecastDates = sortedForecast.map(item => {
      if (typeof item.date === 'string') {
        return parseDate(item.date);
      }
      return new Date(item.date);
    });
    const forecastValues = sortedForecast.map(item => item.prediction);

    // Connect forecast to last historical point for smooth visualization
    const connectedForecastDates = [lastHistoricalDate, ...forecastDates];
    const connectedForecastValues = [lastHistoricalClose, ...forecastValues];

    forecastTrace = {
      x: connectedForecastDates,
      y: connectedForecastValues,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: 'Forecast',
      line: { 
        color: '#ff6b6b', 
        width: 3, 
        dash: 'dash',
        shape: 'linear' as const
      },
      marker: { 
        size: 5,
        color: '#ff6b6b'
      },
      xaxis: 'x',
      yaxis: 'y'
    };
  }

  // Prepare error overlays (when actual prices are available)
  const errorTraces: any[] = [];
  if (errors && errors.length > 0) {
    // Error bars showing difference between predicted and actual
    errors.forEach((err, idx) => {
      const errorColor = 
        Math.abs(err.errorPercent) < 2 ? '#10b981' :  // Green: < 2% error
        Math.abs(err.errorPercent) < 5 ? '#f59e0b' :  // Yellow: 2-5% error
        '#ef4444';  // Red: > 5% error
      
      // Parse error date to Date object for consistency with historical and forecast data
      const errorDate = typeof err.date === 'string' ? parseDate(err.date) : new Date(err.date);
      
      // Error bar trace
      errorTraces.push({
        x: [errorDate],
        y: [err.actual],
        type: 'scatter' as const,
        mode: 'markers' as const,
        name: idx === 0 ? 'Actual (with error)' : '',
        marker: {
          color: errorColor,
          size: 10,
          symbol: 'circle',
          line: { color: errorColor, width: 2 }
        },
        error_y: {
          type: 'data' as const,
          array: [Math.abs(err.error)],
          color: errorColor,
          thickness: 2,
          width: 8
        },
        showlegend: idx === 0,
        hovertemplate: `<b>Actual: $${err.actual.toFixed(2)}</b><br>` +
                       `Predicted: $${err.predicted.toFixed(2)}<br>` +
                       `Error: $${err.error.toFixed(2)} (${err.errorPercent.toFixed(2)}%)<extra></extra>`
      });
    });
  }

  // Build data array: candlestick first, then close price line overlay, then forecast, then errors
  const data = [candlestickTrace, closePriceLineTrace];
  if (forecastTrace) data.push(forecastTrace);
  if (errorTraces.length > 0) data.push(...errorTraces);

  // Always use 'date' type for x-axis - Plotly handles both daily and hourly data well
  // The date type automatically scales appropriately based on the data range
  const xAxisType = 'date';
  
  // Calculate x-axis range to include forecast if present
  let xAxisRange: [string, string] | undefined = undefined;
  if (forecast.length > 0 && dates.length > 0) {
    const allDates = [...dates];
    forecast.forEach(item => {
      const forecastDate = typeof item.date === 'string' ? parseDate(item.date) : new Date(item.date);
      allDates.push(forecastDate);
    });
    const minDate = new Date(Math.min(...allDates.map(d => d.getTime())));
    const maxDate = new Date(Math.max(...allDates.map(d => d.getTime())));
    // Extend range slightly to show forecast clearly
    const padding = (maxDate.getTime() - minDate.getTime()) * 0.1;
    xAxisRange = [
      new Date(minDate.getTime() - padding).toISOString(),
      new Date(maxDate.getTime() + padding).toISOString()
    ];
  }
  
  const layout = {
    title: {
      text: `${symbol} Price Chart`,
      font: { size: 18 }
    },
    xaxis: {
      title: 'Date',
      type: xAxisType as const,
      rangeslider: { visible: false },
      showgrid: true,
      gridcolor: 'rgba(128, 128, 128, 0.2)',
      ...(xAxisRange && { range: xAxisRange })
    },
    yaxis: {
      title: 'Price ($)',
      side: 'right' as const,
      showgrid: true,
      gridcolor: 'rgba(128, 128, 128, 0.2)'
    },
    hovermode: 'x unified' as const,
    showlegend: true,
    legend: {
      x: 0,
      y: 1
    },
    margin: { t: 50, r: 50, b: 50, l: 50 },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {
      color: '#374151'
    }
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{symbol} Price Chart</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <Plot
            data={data}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </CardContent>
    </Card>
  );
};

