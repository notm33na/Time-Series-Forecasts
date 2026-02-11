import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, LineChart, Brain, Target, RefreshCw, Activity } from "lucide-react";

// About page component - updated to reflect actual project features
const About = () => {
  const features = [
    {
      icon: TrendingUp,
      title: "Multi-Asset Forecasting",
      description: "Support for stocks, cryptocurrencies, and forex instruments with multiple forecast horizons (1h, 3h, 24h, 72h)",
    },
    {
      icon: Brain,
      title: "Advanced ML Models",
      description: "Traditional models (ARIMA) and neural networks (LSTM, GRU, Transformer) with adaptive ensemble",
    },
    {
      icon: RefreshCw,
      title: "Adaptive Learning",
      description: "Incremental learning, fine-tuning, scheduled retraining, and dynamic model reweighting based on performance",
    },
    {
      icon: Activity,
      title: "Continuous Evaluation",
      description: "Automatic evaluation when ground truth arrives, real-time metrics (MAE, RMSE, MAPE), and error visualization",
    },
    {
      icon: Target,
      title: "Portfolio Management",
      description: "Simulated trading with multiple strategies (forecast-based, MA crossover, combined) and performance tracking",
    },
    {
      icon: LineChart,
      title: "Interactive Visualization",
      description: "Candlestick charts with forecast overlays, color-coded error indicators, and real-time performance monitoring",
    },
  ];

  const models = [
    { name: "ARIMA", type: "Traditional" },
    { name: "LSTM", type: "Neural Network" },
    { name: "GRU", type: "Neural Network" },
    { name: "Transformer", type: "Neural Network" },
    { name: "Adaptive Ensemble", type: "Ensemble" },
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
            A comprehensive financial forecasting platform combining traditional time series models with modern neural network approaches
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3 text-primary">Overview</h3>
            <p className="text-muted-foreground leading-relaxed">
              ForecastPro is a professional-grade FinTech forecasting platform that provides comprehensive financial market predictions
              across multiple asset classes. The system features adaptive learning mechanisms that continuously improve model performance,
              real-time evaluation and monitoring, and integrated portfolio management capabilities. Built with a microservice architecture,
              the platform supports both batch and streaming data processing with automatic model retraining and versioning.
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
            <h3 className="text-xl font-semibold mb-3 text-primary">Forecasting Models</h3>
            <div className="grid md:grid-cols-2 gap-3">
              {models.map((model) => (
                <div
                  key={model.name}
                  className="p-3 rounded-lg bg-secondary/20 border border-border/50 flex items-center justify-between"
                >
                  <span className="font-medium">{model.name}</span>
                  <span className="text-xs text-muted-foreground bg-primary/10 px-2 py-1 rounded">
                    {model.type}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3 text-primary">Adaptive Learning Capabilities</h3>
            <div className="space-y-2 text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Incremental Learning:</strong> Online updates using SGDRegressor with partial_fit()
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Fine-Tuning:</strong> Rolling window fine-tuning for neural models (LSTM, GRU, Transformer)
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Scheduled Retraining:</strong> Automatic retraining based on time, performance degradation, or new data
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Adaptive Ensemble:</strong> Dynamic reweighting based on recent errors with exponential decay
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Model Versioning:</strong> Complete model registry with performance tracking and artifact storage
              </div>
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
                <strong>Backend:</strong> FastAPI (Python), RESTful API with auto-generated Swagger docs
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Database:</strong> MongoDB (historical data, forecasts, model versions, portfolio)
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>ML Libraries:</strong> scikit-learn, statsmodels, TensorFlow/Keras, pandas
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary"></span>
                <strong>Visualization:</strong> Plotly for charts and interactive dashboards
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-border">
            <h3 className="text-xl font-semibold mb-3 text-primary">Getting Started</h3>
            <ol className="space-y-2 text-muted-foreground list-decimal list-inside">
              <li>Choose an instrument type (Stock, Crypto, or Forex) and select a symbol</li>
              <li>Pick a forecast horizon (1h, 3h, 24h, or 72h) and model type</li>
              <li>Generate forecasts and view predictions overlaid on historical price charts</li>
              <li>Monitor model performance with real-time metrics and error visualization</li>
              <li>Set up portfolio strategies and track simulated trading performance</li>
              <li>Configure scheduled retraining for automatic model updates</li>
            </ol>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default About;
