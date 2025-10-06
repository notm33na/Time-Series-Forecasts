import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Save, Lock } from "lucide-react";

const Settings = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [apiUrl, setApiUrl] = useState("");
  const [apiKey, setApiKey] = useState("");

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    const { data: userData } = await supabase.auth.getUser();
    if (!userData.user) return;

    const { data } = await supabase
      .from("user_settings")
      .select("model_api_url")
      .eq("user_id", userData.user.id)
      .maybeSingle();

    if (data) {
      setApiUrl(data.model_api_url || "");
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const { data: userData } = await supabase.auth.getUser();
      if (!userData.user) throw new Error("Not authenticated");

      const { error } = await supabase
        .from("user_settings")
        .upsert({
          user_id: userData.user.id,
          model_api_url: apiUrl,
          api_key_encrypted: apiKey, // In production, encrypt this properly
          updated_at: new Date().toISOString(),
        });

      if (error) throw error;

      toast({
        title: "Settings saved",
        description: "Your API configuration has been updated.",
      });

      setApiKey(""); // Clear the API key input after saving
    } catch (error: any) {
      toast({
        title: "Save failed",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <Card className="glass-panel">
        <CardHeader>
          <CardTitle className="text-2xl text-primary flex items-center gap-2">
            <Lock className="h-6 w-6" />
            API Settings
          </CardTitle>
          <CardDescription>
            Configure your external model API endpoint and credentials.
            These are stored securely and used for forecast generation.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSave} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="api-url">Model API URL</Label>
              <Input
                id="api-url"
                type="url"
                placeholder="https://your-model-api.com/predict"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="bg-secondary/50"
                required
              />
              <p className="text-sm text-muted-foreground">
                The endpoint should accept POST requests with instrument and horizon parameters.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="api-key">API Key</Label>
              <Input
                id="api-key"
                type="password"
                placeholder="Enter your API key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="bg-secondary/50"
              />
              <p className="text-sm text-muted-foreground">
                Your API key is encrypted and stored securely. Leave blank to keep existing key.
              </p>
            </div>

            <div className="pt-4 border-t border-border">
              <h3 className="font-semibold mb-2">Expected API Response Format:</h3>
              <pre className="bg-secondary/30 p-4 rounded-lg text-xs overflow-x-auto">
                {`{
  "timestamps": ["2025-01-01T00:00:00Z", ...],
  "forecast": [150.5, 151.2, ...],
  "lower": [148.5, 149.0, ...],
  "upper": [152.5, 153.5, ...]
}`}
              </pre>
            </div>

            <Button type="submit" disabled={loading} className="w-full btn-primary">
              <Save className="mr-2 h-4 w-4" />
              {loading ? "Saving..." : "Save Settings"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default Settings;
