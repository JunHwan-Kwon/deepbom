/** Stable SDK entry point; native documents retain their own schema versions. */
export type JsonDocument = Record<string, unknown>;
export interface Limits { timeoutMs?: number; maxOutputBytes?: number; }
export interface InspectOptions extends Limits {
  scan?: "auto" | "structure" | "integrity" | "full";
  expectedSha256?: string;
  gate?: "defects";
  policy?: "engineering" | "regulatory";
}
export interface AuditOptions extends InspectOptions {
  output?: "analysis" | "envelope" | "cyclonedx" | "sarif";
  sections?: string[];
}
export interface NumericalEvidenceOptions extends Limits {
  weights?: boolean;
  weightBaseline?: string;
  weightOptions?: string;
  weightMapping?: string;
  activationCapture?: string;
  activationBaseline?: string;
  expectedSha256?: string;
}
/** A selection of native numerical documents; no automatic model execution. */
export interface NumericalEvidenceResult extends JsonDocument {
  schema: "deepbom.analysis_selection.v1";
  sections: {
    model_ir: JsonDocument;
    numerical_evidence_bundle: JsonDocument;
    numerical_details: JsonDocument & { schema: "deepbom.numerical_details.v1"; numerical_details_sha256: string };
    weight_ir?: JsonDocument;
    weight_analysis?: JsonDocument;
    weight_comparison?: JsonDocument;
    activation_ir?: JsonDocument;
    activation_baseline_ir?: JsonDocument;
  };
}
export interface Finding extends JsonDocument {
  id: string; title: string; severity: string; evidence_class: string;
  source_pointers: string[]; recommendation: string | null;
}
export interface ReviewSummary extends JsonDocument {
  schema: "deepbom.review_summary.v1";
  artifact: { filename: string | null; format: string | null; sha256: string; byte_length: number | null; artifact_ir_sha256: string };
  verdict: { status: "artifact_defects_observed" | "no_artifact_defect_observed"; artifact_defect_count: number; caution_count: number; evidence_needed_count: number; scope: string };
  findings: { artifact_defects: Finding[]; cautions: Finding[]; evidence_needed: Finding[] };
  coverage: { assessed: number; partial: number; unavailable: number; /** @deprecated Alias for unavailable; does not mean all missing capabilities require external evidence. */ needs_external_evidence: number; declared: number; applicability: Record<string, number> };
  graph: { operator_count: number | null; tensor_count: number | null; total_macs: number | null; mac_confidence: "exact" | "partial" | "symbolic" | "not_applicable"; mac_assessment_status: string | null };
  storage: JsonDocument | null;
  quantization: { classification: string | null; max_risk: string; max_risk_op_index: number | null; max_risk_op_name: string | null; max_risk_detail: string | null };
  target: { id: string | null; label: string | null; binding_source: string | null };
  rulepack: { version: string | null; sha256: string | null };
  reproduction: { schema: "deepbom.reproduction_command.v1"; executable: string; argv: string[]; package_version: string; artifact_filename: string; expected_sha256: string; shell_command: string; boundary: string };
  applicability: Record<string, JsonDocument>;
  next_actions: { action: string; command_hint: string }[];
  evidence_envelope_sha256: string;
}
export interface EvidenceEnvelope extends JsonDocument {
  schema: "deepbom.artifact_evidence_envelope.v1";
  identity: { filename: string; format: string; sha256: string; hash_basis: string; byte_length: number | null; schema_or_opset: string | null };
  capabilities: { schema: string; format: string; declared_capabilities: string[]; assessed: string[]; partial: string[]; unavailable: string[]; conservation: { declared: number; classified: number; valid: boolean } };
  findings: (JsonDocument & { id: string; finding_kind: "artifact_defect" | "caution" | "evidence_gap"; severity: string })[];
  provenance: JsonDocument & { analyzer: string; version: string; command: string };
  evidence_boundary: string;
  envelope_sha256: string;
}
export class DeepBomError extends Error {
  constructor(message: string, options?: { exitCode?: number | null; document?: JsonDocument | null; cause?: unknown });
  readonly code: string;
  readonly exitCode: number | null;
  document: JsonDocument | null;
}
export class DeepBomInvocationError extends DeepBomError {}
export class DeepBomPolicyBlocked extends DeepBomError {}
export class DeepBomIncompleteBinding extends DeepBomError {}
export class DeepBomIdentityMismatch extends DeepBomError {}
export class DeepBomTimeout extends DeepBomError {}
export class DeepBomOutputTooLarge extends DeepBomError {}
export function capabilities(options?: Limits): Promise<JsonDocument>;
export function numericalEvidence(artifact: string, options: NumericalEvidenceOptions): Promise<NumericalEvidenceResult>;
export function inspect(artifact: string, options?: InspectOptions): Promise<ReviewSummary>;
export function audit(artifact: string, options?: InspectOptions & { output?: "envelope"; sections?: never }): Promise<EvidenceEnvelope>;
export function audit(artifact: string, options: AuditOptions): Promise<JsonDocument>;
export function diff(baseline: string, candidate: string, options?: Limits & { tensorsOnly?: boolean }): Promise<JsonDocument>;
export function captureContract(artifact: string, options?: Limits): Promise<JsonDocument>;
export function verifyContract(artifact: string, contract: string, options?: Limits): Promise<JsonDocument>;
export function verifyBom(artifact: string, bom: string, options?: Limits & { componentRef?: string }): Promise<JsonDocument>;
