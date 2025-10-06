-- Create profiles table for user data
CREATE TABLE public.profiles (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email text,
  full_name text,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

-- Create trigger function to handle new user signups
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name)
  VALUES (
    new.id,
    new.email,
    new.raw_user_meta_data->>'full_name'
  );
  RETURN new;
END;
$$;

-- Trigger to create profile on signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Create OHLC table for historical price data
CREATE TABLE public.ohlc (
  id bigserial PRIMARY KEY,
  instrument text NOT NULL,
  ts timestamptz NOT NULL,
  open numeric NOT NULL,
  high numeric NOT NULL,
  low numeric NOT NULL,
  close numeric NOT NULL,
  volume numeric NOT NULL,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Enable RLS on ohlc
ALTER TABLE public.ohlc ENABLE ROW LEVEL SECURITY;

-- OHLC policies - read-only for authenticated users
CREATE POLICY "Authenticated users can view OHLC data"
  ON public.ohlc FOR SELECT
  TO authenticated
  USING (true);

-- Create index for faster queries
CREATE INDEX idx_ohlc_instrument_ts ON public.ohlc(instrument, ts DESC);

-- Create forecasts table
CREATE TABLE public.forecasts (
  id bigserial PRIMARY KEY,
  user_id uuid REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
  instrument text NOT NULL,
  horizon text NOT NULL,
  model text DEFAULT 'external-api' NOT NULL,
  generated_at timestamptz DEFAULT now() NOT NULL,
  payload jsonb NOT NULL
);

-- Enable RLS on forecasts
ALTER TABLE public.forecasts ENABLE ROW LEVEL SECURITY;

-- Forecasts policies
CREATE POLICY "Users can view own forecasts"
  ON public.forecasts FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can create own forecasts"
  ON public.forecasts FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own forecasts"
  ON public.forecasts FOR DELETE
  USING (auth.uid() = user_id);

-- Create index for faster forecast queries
CREATE INDEX idx_forecasts_user_instrument ON public.forecasts(user_id, instrument, generated_at DESC);

-- Create settings table for API configuration
CREATE TABLE public.user_settings (
  user_id uuid PRIMARY KEY REFERENCES public.profiles(id) ON DELETE CASCADE,
  model_api_url text,
  api_key_encrypted text,
  updated_at timestamptz DEFAULT now() NOT NULL
);

-- Enable RLS on user_settings
ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;

-- Settings policies
CREATE POLICY "Users can view own settings"
  ON public.user_settings FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own settings"
  ON public.user_settings FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own settings"
  ON public.user_settings FOR UPDATE
  USING (auth.uid() = user_id);

-- Insert sample OHLC data for testing
INSERT INTO public.ohlc (instrument, ts, open, high, low, close, volume) VALUES
  ('AAPL', now() - interval '5 days', 150.00, 152.50, 149.00, 151.25, 1000000),
  ('AAPL', now() - interval '4 days', 151.25, 153.00, 150.50, 152.75, 1200000),
  ('AAPL', now() - interval '3 days', 152.75, 154.00, 151.50, 153.50, 1100000),
  ('AAPL', now() - interval '2 days', 153.50, 155.00, 152.00, 154.25, 1300000),
  ('AAPL', now() - interval '1 day', 154.25, 156.00, 153.50, 155.50, 1400000),
  ('BTC-USD', now() - interval '5 days', 45000.00, 46000.00, 44500.00, 45500.00, 500000),
  ('BTC-USD', now() - interval '4 days', 45500.00, 47000.00, 45000.00, 46500.00, 600000),
  ('BTC-USD', now() - interval '3 days', 46500.00, 48000.00, 46000.00, 47000.00, 550000),
  ('BTC-USD', now() - interval '2 days', 47000.00, 48500.00, 46500.00, 48000.00, 650000),
  ('BTC-USD', now() - interval '1 day', 48000.00, 49000.00, 47500.00, 48500.00, 700000);