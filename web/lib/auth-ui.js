export function applyAuthConfigView(elements, config) {
  const passwordEnabled = Boolean(config.password);
  elements.openButton.disabled = false;
  elements.googleButton.disabled = !config.google;
  elements.submitButton.disabled = !passwordEnabled;
  elements.passwordTabs.hidden = !passwordEnabled;
  elements.passwordForm.hidden = !passwordEnabled;
  elements.divider.hidden = !passwordEnabled || !config.google;
  if (!config.enabled) {
    elements.widget.classList.add("auth-unavailable");
    elements.openButton.textContent = "Account setup";
    elements.message.textContent = "Authentication is not configured yet. Add D1 and Worker secrets to enable signups.";
  } else {
    elements.widget.classList.remove("auth-unavailable");
    elements.openButton.textContent = "Sign in";
    elements.message.textContent = passwordEnabled
      ? "Analysis and local exports are open. Sign in only for optional research access and saved requests."
      : "Continue with Google only for optional research access and saved requests.";
  }
  elements.googleButton.textContent = config.google ? "Continue with Google" : "Google OAuth not configured";
}

export function applyAuthModeView(elements, mode) {
  const signup = mode === "signup";
  elements.loginTab.classList.toggle("active", !signup);
  elements.signupTab.classList.toggle("active", signup);
  elements.nameWrap.hidden = !signup;
  elements.passwordInput.autocomplete = signup ? "new-password" : "current-password";
  elements.submitButton.textContent = signup ? "Create account" : "Login";
}
