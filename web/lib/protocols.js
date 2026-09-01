import { formatDrift } from "./format.js";

export function deepBomProtocolGroups(result = null) {
  const posture = result?.posture || null;
  const ws = result?.weight_tensor_stats || {};
  const hasWs = (ws.analyzed_count || 0) > 0;
  const anomalyCount = (result?.anomalies || []).length;
  const caveatCount = (result?.caveats || []).length;

  const summaryItems = [
    { name: "Evidence type", detail: "Experimental deploy-artifact composites over weight distributions, quantization metadata, graph topology, low-intensity mix, and predicted delegate pressure." },
    { name: "Use", detail: "Descriptive research output only. No empirical accuracy, latency, robustness, Hessian, or generalization threshold has been established." },
    hasWs ? { name: "Weight tensors", detail: `${ws.analyzed_count} constant tensor(s) analyzed from artifact bytes${ws.has_quantized ? "; includes INT8/UINT8 quantized tensors" : ""}.` } : null,
    anomalyCount ? { name: "Anomalies", status: "executed", detail: `${anomalyCount} cross-validation anomaly/anomalies flagged. Review Caveats section.` } : null,
    caveatCount ? { name: "Caveats", detail: `${caveatCount} scope/limitation note(s). See Caveats section.` } : null,
  ].filter(Boolean);

  return [
    {
      title: "Evidence Summary",
      items: summaryItems,
    },
    {
      title: "Interpretation Boundary",
      items: [
        { name: "Artifact composite", detail: "Deterministic fixed-weight summary. No validated ok/warn/risk threshold or measured stability interpretation." },
        { name: "Quant composite", detail: "Component values and formula are reproducible; the aggregate weighting is not empirically calibrated." },
        { name: "Topology composite", detail: "Predicted chain and byte descriptors are inputs; the aggregate is not an observed latency or accuracy signal." },
        { name: "Byte Entropy", detail: "watch < 4.5 bits/B; otherwise info. File-byte signal only." },
      ],
    },
  ];
}

export function perturbationProtocolGroups({ drift, baseline, perturbed, weightProbe = null, layerRobustness = [], haarSweep = null, profile }) {
  const executedInput = Boolean(drift && baseline && perturbed);
  const haarOk = Array.isArray(haarSweep) && haarSweep.some((r) => !r.error);
  const executed = [
    executedInput
      ? { name: "Input perturbation", status: "executed", detail: `Baseline vs perturbed local TFLite/LiteRT WASM output drift. Unit: ${profile.unit} (${profile.dtype}).` }
      : null,
    executedInput
      ? { name: "Output drift", status: "executed", detail: "Reports RMS, mean absolute, max absolute, cosine distance, and top-1 flip." }
      : null,
    weightProbe
      ? { name: "Weight perturbation sensitivity", status: "executed", detail: `Local perturbed model-byte copy over ${weightProbe.touchedTensors} constant tensor(s); raw weights are not exported.` }
      : null,
    layerRobustness.length
      ? { name: "Layer-wise robustness scoring", status: "executed", detail: `Swept ${layerRobustness.length} high-impact layer(s) with local weight perturbation and output drift scoring.` }
      : null,
    haarOk
      ? { name: "Haar pattern sweep", status: "executed", detail: `${haarSweep.filter((r) => !r.error).length} structured spatial patterns (DC, edge, diagonal, checkerboard at 2–16 px, stride-shifted) run as synthetic input; drift vs zero baseline.` }
      : null,
  ].filter(Boolean);
  const notRun = [
    !executedInput
      ? { name: "Input/output drift", status: "not_run", detail: "No TFLite runtime path returned baseline and perturbed outputs." }
      : null,
    !weightProbe
      ? { name: "Weight perturbation sensitivity", status: "not_applicable", detail: "This artifact did not expose supported constant weight buffer metadata for local mutation." }
      : null,
    !layerRobustness.length
      ? { name: "Layer-wise robustness scoring", status: "not_applicable", detail: "Layer sweep needs at least one supported constant weight tensor." }
      : null,
    !haarOk
      ? { name: "Haar pattern sweep", status: "not_run", detail: "Spatial pattern sweep did not complete." }
      : null,
  ].filter(Boolean);
  return [
    { title: "Executed Evidence", items: executed.length ? executed : [{ name: "No executable probe completed", status: "not_run", detail: "Run a TFLite model through the local runtime path first." }] },
    ...(notRun.length ? [{ title: "Not Run For This Artifact", items: notRun }] : []),
    {
      title: "Status Criteria",
      items: [
        { name: "RMS drift", detail: `ok <= ${formatDrift(profile.rmsOk)}; warn <= ${formatDrift(profile.rmsWarn)}; risk > ${formatDrift(profile.rmsWarn)} ${profile.unit}.` },
        { name: "Max abs drift", detail: `ok <= ${formatDrift(profile.maxOk)}; warn <= ${formatDrift(profile.maxWarn)}; risk > ${formatDrift(profile.maxWarn)} ${profile.unit}.` },
        { name: "Cosine distance", detail: "ok <= 1e-4; warn <= 1e-2; risk > 1e-2." },
        { name: "Timing", detail: "Single-run baseline/perturbed timings are variance notes only. Use Runtime Benchmark p50/p90/p95/p99 for latency decisions." },
      ],
    },
  ];
}

export function runtimeBasinProtocolGroups({ maxDrift, results = [], preprocessingDrift = null, profile }) {
  const attempted = results.length ? results.map((item) => `${item.backend}:${item.ok ? "ok" : "fail"}`).join(" / ") : "none";
  const successfulCount = results.filter((item) => item.ok).length;
  const executed = [
    results.length ? { name: "Backend availability", status: "executed", detail: `Attempted local LiteRT browser backend candidates exposed by this device: ${attempted}.` } : null,
    maxDrift ? { name: "Backend output drift", status: "executed", detail: `Compared successful backend outputs against the first successful reference backend. Unit: ${profile.unit} (${profile.dtype}).` } : null,
    preprocessingDrift ? { name: "Preprocessing drift", status: "executed", detail: "Compared prepared image tensor output against synthetic tensor output on the WASM path." } : null,
  ].filter(Boolean);
  const notRun = [
    results.length && !maxDrift ? { name: "Backend output drift", status: "not_run", detail: successfulCount < 2 ? "At least two successful backend paths are required for output-drift comparison." : "No comparable backend output drift was produced." } : null,
    !preprocessingDrift ? { name: "Preprocessing drift", status: "not_applicable", detail: "Prepare an image tensor in Runtime before running this check." } : null,
  ].filter(Boolean);
  return [
    { title: "Executed Evidence", items: executed.length ? executed : [{ name: "No backend run completed", status: "not_run", detail: "No local browser runtime path returned outputs." }] },
    ...(notRun.length ? [{ title: "Not Run For This Artifact", items: notRun }] : []),
    {
      title: "Status Criteria",
      items: [
        { name: "Backend coverage", detail: "ok = all attempted backends completed; warn = partial completion; risk = none completed." },
        { name: "Max backend drift", detail: `ok <= ${formatDrift(profile.maxOk)}; warn <= ${formatDrift(profile.maxWarn)}; risk > ${formatDrift(profile.maxWarn)} ${profile.unit}.` },
        { name: "Backend set", detail: "WebGPU/WebNN are browser/device dependent. Missing backends are reported as unavailable, not assumed." },
      ],
    },
  ];
}

export function deploymentSensitivityProtocolGroups({ basin = null, curvature = null }) {
  const executed = basin && curvature;
  return [
    {
      title: "Executed Evidence",
      items: executed
        ? [
          { name: "Deploy curvature", status: "executed", detail: "Finite-difference probes of the deployed function f(x) through the local runtime path." },
          { name: "Experimental stability composite", status: "executed", detail: "Unvalidated fixed-weight summary of perturbation radius, top-1 stability, output drift, margin, and directional finite-difference signals." },
        ]
        : [
          { name: "Deploy stability probes", status: "not_run", detail: "Run a supported TFLite artifact through the local runtime path to compute research-stage finite-difference stability evidence." },
        ],
    },
    {
      title: "Interpretation Boundary",
      items: [
        { name: "Experimental composite", detail: "No validated ok/warn/risk threshold; inspect the raw curvature, drift, margin, and rank-change observations." },
        { name: "Tested rank-stability radius", detail: "Reports only the largest tested epsilon band that preserved the first-output argmax for this synthetic direction." },
        { name: "Curvature/Lipschitz", detail: "Relative deploy-function metrics. Compare across the same model, input contract, dtype, and perturbation scale." },
      ],
    },
  ];
}
