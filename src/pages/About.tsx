import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, Database, Lock, Zap, LineChart, Code } from "lucide-react";

const About = () => {
  const features = [
    {
      icon: TrendingUp,
      title: "Multi-Asset Forecasting",
      description: "Support for stocks, cryptocurrencies, and forex instruments",
    },
    {
      icon: LineChart,
      title: "Advanced Charting",
      description: "Interactive candlestick charts with confidence intervals",
    },
    {
      icon: Zap,
      title: "External Model Integration",
      description: "Connect to your Python ML models via REST API",
    },
    {
      icon: Database,
      title: "Historical Data Storage",
      description: "OHLC data storage with PostgreSQL",
    },
    {
      icon: Lock,
      title: "Secure Authentication",
      description: "Email/password authentication with secure API key storage",
    },
    {
      icon: Code,
      title: "Modern Tech Stack",
      description: "Built with React, TypeScript, and Lovable Cloud",
    },
  ];

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <Card className="glass-panel">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className="p-4 rounded-xl bg-primary/10 glow-border">
              <TrendingUp className="h-12 w-12 text-primary" />
            </div>
          </div>
          <CardTitle className="text-3xl">About ForecastPro</CardTitle>
          <CardDescription className="text-base">
            A professional financial forecasting platform built to connect with your ML models
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3 text-primary">Overview</h3>
            <p className="text-muted-foreground leading-relaxed">
              ForecastPro is a full-stack web application designed to visualize financial market data
              and integrate with external machine learning models for price forecasting. The platform
              provides a clean, mobile-friendly dashboard where users can select instruments, choose
              forecast horizons, and view predictions with confidence intervals overlaid on historical
              price charts.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-4 text-primary">Key Features</h3>
            <div className="grid md:grid-cols-2 gap-4">
              {features.map((feature) => (
                <div
                  key={feature.title}
                  className="flex gap-3 p-4 rounded-lg bg-secondary/30 border border-border/50"
                >
                  <feature.icon className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-semibold mb-1">{feature.title}</h4>
                    <p className="text-sm text-muted-foreground">{feature.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3 text-primary">Tech Stack</h3>
            <div className="space-y-2 text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Frontend:</strong> React 18, TypeScript, Tailwind CSS
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Backend:</strong> Lovable Cloud (PostgreSQL, Auth, Storage)
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Charts:</strong> Recharts library for responsive visualizations
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>External API:</strong> REST integration for Python ML models
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-border">
            <h3 className="text-xl font-semibold mb-3 text-primary">Getting Started</h3>
            <ol className="space-y-2 text-muted-foreground list-decimal list-inside">
              <li>Configure your model API endpoint in the Settings page</li>
              <li>Choose an instrument type (Stock, Crypto, or Forex)</li>
              <li>Select a specific symbol to view historical data</li>
              <li>Pick a forecast horizon (1h, 3h, 24h, or 72h)</li>
              <li>Click "Run Forecast" to generate predictions</li>
              <li>View results with confidence bands on the interactive chart</li>
            </ol>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default About;
