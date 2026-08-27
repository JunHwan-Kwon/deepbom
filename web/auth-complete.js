const params = new URLSearchParams(location.search);
const error = params.get("auth_error") || "";
const message = {
  type: "deepbom:auth-complete",
  ok: !error,
  error,
};

try {
  const channel = new BroadcastChannel("deepbom-auth");
  channel.postMessage(message);
  channel.close();
} catch {
  // window.opener remains the compatibility path where BroadcastChannel is unavailable.
}

if (window.opener) window.opener.postMessage(message, location.origin);

document.getElementById("authCompleteTitle").textContent = error ? "Sign-in failed" : "Sign-in complete";
document.getElementById("authCompleteMessage").textContent = error
  ? `Return to the audit and try again (${error}).`
  : "The audit remains open in the original tab. This window can be closed.";

if (!error) window.close();
