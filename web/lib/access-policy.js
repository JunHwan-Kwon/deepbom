export const ACCESS_PROFILES = Object.freeze(["anonymous", "verified", "research", "medical_ai", "admin"]);

const PROFILE_RANK = Object.freeze(Object.fromEntries(ACCESS_PROFILES.map((profile, index) => [profile, index])));

export function canonicalAccessProfile(profile, role = "user") {
  if (role === "admin" || profile === "admin") return "admin";
  const value = String(profile || "verified").trim().toLowerCase();
  return Object.hasOwn(PROFILE_RANK, value) ? value : "verified";
}

export function accessProfileForUser(user) {
  if (!user) return "anonymous";
  return canonicalAccessProfile(user.access_profile, user.role);
}

export function hasAccessProfile(current, required) {
  const currentProfile = typeof current === "object" ? accessProfileForUser(current) : canonicalAccessProfile(current);
  return PROFILE_RANK[currentProfile] >= PROFILE_RANK[required];
}
