import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { User, Lock, KeyRound } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

const Settings = () => {
  const { toast } = useToast();
  const { user } = useAuth();
  const [loading, setLoading] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const validatePassword = (password: string): { valid: boolean; message?: string } => {
    if (password.length < 8) {
      return { valid: false, message: "Password must be at least 8 characters long." };
    }

    if (!/[A-Z]/.test(password)) {
      return { valid: false, message: "Password must contain at least one capital letter." };
    }

    if (!/[0-9]/.test(password)) {
      return { valid: false, message: "Password must contain at least one number." };
    }

    if (!/[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(password)) {
      return { valid: false, message: "Password must contain at least one special character." };
    }

    return { valid: true };
  };

  const handleResetPassword = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // Validate current password is provided
    if (!currentPassword) {
      toast({
        title: "Current password required",
        description: "Please enter your current password to reset it.",
        variant: "destructive",
      });
      setLoading(false);
      return;
    }

    // Validate new password strength
    const passwordValidation = validatePassword(newPassword);
    if (!passwordValidation.valid) {
      toast({
        title: "Invalid password",
        description: passwordValidation.message,
        variant: "destructive",
      });
      setLoading(false);
      return;
    }

    // Validate passwords match
    if (newPassword !== confirmPassword) {
      toast({
        title: "Passwords don't match",
        description: "New password and confirm password must match.",
        variant: "destructive",
      });
      setLoading(false);
      return;
    }

    // Simulate password reset (in production, this would call an API)
    setTimeout(() => {
      toast({
        title: "Password reset",
        description: "Your password has been successfully reset.",
      });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setLoading(false);
    }, 1000);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      {/* User Details Card */}
      <Card className="glass-panel">
        <CardHeader>
          <CardTitle className="text-2xl text-primary flex items-center gap-2">
            <User className="h-6 w-6" />
            User Details
          </CardTitle>
          <CardDescription>
            View your account information.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="user-email">Email</Label>
            <Input
              id="user-email"
              type="email"
              value={user?.email || ""}
              className="bg-secondary/50"
              disabled
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="user-name">Full Name</Label>
            <Input
              id="user-name"
              type="text"
              value={user?.name || "Not set"}
              className="bg-secondary/50"
              disabled
            />
          </div>
        </CardContent>
      </Card>

      {/* Reset Password Card */}
      <Card className="glass-panel">
        <CardHeader>
          <CardTitle className="text-2xl text-primary flex items-center gap-2">
            <KeyRound className="h-6 w-6" />
            Reset Password
          </CardTitle>
          <CardDescription>
            Change your account password. Make sure to use a strong password.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleResetPassword} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="current-password">Current Password</Label>
              <Input
                id="current-password"
                type="password"
                placeholder="Enter your current password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                className="bg-secondary/50"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="new-password">New Password</Label>
              <Input
                id="new-password"
                type="password"
                placeholder="Enter your new password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="bg-secondary/50"
                required
                minLength={8}
              />
              <p className="text-sm text-muted-foreground">
                Password must be at least 8 characters long and contain:
              </p>
              <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                <li>At least one capital letter (A-Z)</li>
                <li>At least one number (0-9)</li>
                <li>At least one special character (!@#$%^&*...)</li>
              </ul>
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirm-password">Confirm New Password</Label>
              <Input
                id="confirm-password"
                type="password"
                placeholder="Confirm your new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="bg-secondary/50"
                required
                minLength={8}
              />
            </div>

            <Button type="submit" disabled={loading} className="w-full btn-primary">
              <Lock className="mr-2 h-4 w-4" />
              {loading ? "Resetting Password..." : "Reset Password"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default Settings;
