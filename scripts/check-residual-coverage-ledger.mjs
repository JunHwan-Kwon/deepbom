import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { createCheck } from "./check-assert.mjs";
import { DYNAMIC_SHAPE_COST_SCHEMA } from "../web/lib/dynamic-shape-cost.js";
import { ONNX_SHAPE_INFERENCE_SCHEMA } from "../web/lib/onnx-shape-inference.js";

const check = createCheck("Residual coverage ledger");
const ledger = JSON.parse(readFileSync("corpus/residual-coverage-priorities.v1.json", "utf8"));
const onnx = ledger.populations?.onnx || {};
const priorities = ledger.ranked_priorities || [];
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const histogramTotal = (rows) => (rows || []).reduce((total, row) => total + Number(row.count || 0), 0);

check.expectEqual(ledger.schema, "deepbom.residual_coverage_priorities.v1.6", "ledger schema");
check.expectEqual(onnx.analyzer_contracts?.onnx_shape_inference, ONNX_SHAPE_INFERENCE_SCHEMA, "ONNX residual shape-engine freshness");
check.expectEqual(onnx.analyzer_contracts?.dynamic_shape_cost, DYNAMIC_SHAPE_COST_SCHEMA, "ONNX residual cost-engine freshness");
check.expectEqual(onnx.artifact_count, onnx.artifacts?.length, "ONNX artifact denominator");
check.expectEqual(onnx.path_record_count, onnx.artifact_count + onnx.duplicate_path_record_count, "ONNX path/byte denominator conservation");
check.expectEqual(new Set((onnx.artifacts || []).map((row) => row.artifact_sha256)).size, onnx.artifact_count, "ONNX SHA-256 deduplication");
check.expectEqual((onnx.artifacts || []).reduce((total, row) => total + row.path_records.length, 0), onnx.path_record_count, "ONNX path records");
check.expectEqual(new Set((onnx.artifacts || []).flatMap((row) => row.path_records.map((record) => record.repository_id))).size, onnx.repository_count, "ONNX repository denominator");
check.expectEqual(onnx.node_output_count, onnx.shape_contract_known_node_output_count + onnx.shape_contract_unknown_node_output_count, "ONNX node-output shape contract conservation");
check.expectEqual(onnx.shape_contract_unknown_node_output_count, onnx.invalid_node_output_count
  + onnx.conditionally_invalid_node_output_count + onnx.unresolved_nonconflict_shape_contract_node_output_count,
"ONNX invalid/conditionally-invalid/residual shape separation");
check.expectEqual(onnx.unknown_node_output_count, onnx.shape_contract_unknown_node_output_count
  + onnx.symbolic_shape_contract_node_output_count + onnx.conditional_shape_contract_node_output_count,
"ONNX concrete/symbolic/conditional output conservation");
check.expectEqual(histogramTotal(onnx.unresolved_op_histogram), onnx.shape_rule_unresolved_node_count, "unresolved-op histogram conservation");
check.expectEqual(histogramTotal(onnx.unresolved_reason_histogram), onnx.shape_rule_unresolved_node_count, "unresolved-reason histogram conservation");
check.expect(!(onnx.unresolved_reason_histogram || []).some((row) => row.name.startsWith("blocked_by_upstream_contract_conflict:")), "Priority-ledger reason classes must not duplicate artifact-specific tensor identifiers from the compressed source sweep.");
check.expectEqual(histogramTotal(onnx.unassessed_compute_op_histogram), onnx.unassessed_compute_op_count, "unassessed-compute op histogram conservation");
check.expectEqual(histogramTotal(onnx.unassessed_compute_reason_histogram), onnx.unassessed_compute_op_count, "unassessed-compute reason histogram conservation");
check.expectEqual(histogramTotal(onnx.unassessed_compute_op_reason_histogram), onnx.unassessed_compute_op_count, "unassessed-compute joint histogram conservation");
check.expectEqual(histogramTotal(onnx.algorithm_dependent_arithmetic_op_histogram), onnx.algorithm_dependent_arithmetic_op_count, "algorithm-dependent arithmetic histogram conservation");
check.expectEqual(histogramTotal(onnx.total_macs_unresolved_op_histogram), onnx.total_macs_unresolved_op_count, "total-MAC residual op histogram conservation");
check.expectEqual(histogramTotal(onnx.total_macs_unresolved_reason_histogram), onnx.total_macs_unresolved_op_count, "total-MAC residual reason histogram conservation");
check.expectEqual(histogramTotal(onnx.total_macs_unresolved_resolution_histogram), onnx.total_macs_unresolved_op_count, "total-MAC resolution-class histogram conservation");
check.expectEqual(histogramTotal(onnx.total_macs_unresolved_op_reason_histogram), onnx.total_macs_unresolved_op_count, "total-MAC residual joint histogram conservation");
check.expectEqual(onnx.total_macs_unresolved_op_count, onnx.total_macs_artifact_contract_conflict_op_count
  + onnx.total_macs_external_binding_required_op_count + onnx.total_macs_analyzer_or_contract_residual_op_count,
"total-MAC blocker class conservation");
check.expectEqual(onnx.shape_rule_unsupported_node_count, 0, "Every observed ONNX node should have a source-bound local output-shape rule in the current measured population.");
check.expectEqual(onnx.unresolved_nonconflict_shape_contract_node_output_count, 0, "Every observed incomplete ONNX output contract must be classified as an artifact conflict rather than an analyzer residual.");
check.expectEqual(onnx.conditional_unassessed_variant_count, 0, "Every observed finite conditional-shape branch must be either derived or deterministically invalid.");
check.expectEqual(onnx.total_macs_analyzer_or_contract_residual_op_count, 0, "No observed total-MAC blocker may remain attributable to a statically closable analyzer residual.");
check.expectEqual(histogramTotal(onnx.unsupported_op_histogram), onnx.shape_rule_unsupported_node_count, "unsupported-op histogram conservation");
check.expect(onnx.ort_contrib_node_count > 0 && onnx.artifacts_with_ort_contrib_nodes > 0, "The public ONNX population should expose source-registered ORT contrib nodes separately from unknown custom domains.");
check.expectEqual(onnx.ort_contrib_shape_rule_unsupported_node_count, 0, "Every observed ORT contrib node should have a pinned local output-shape rule in the current measured population.");
check.expectEqual(onnx.external_custom_domain_node_count, 0, "The current population should not relabel ORT contrib nodes as external custom-domain coverage.");
check.expect((onnx.artifacts || []).every((row) => /^[a-f0-9]{64}$/.test(row.artifact_sha256) && row.analysis_sha256s.length > 0 && row.analysis_sha256s.every((value) => /^[a-f0-9]{64}$/.test(value)) && row.path_records.every((record) => row.analysis_sha256s.includes(record.analysis_sha256))), "Every ONNX byte artifact and path record should bind analysis bytes.");
check.expect((onnx.source_sweeps || []).every((row) => /^[a-f0-9]{64}$/.test(row.sha256)), "Every measured sweep should carry a SHA-256.");

const coreMlManifest = readFileSync("corpus/coreml-mlprogram-contract-corpus.v1.json");
const coreMlPublicEvidence = readFileSync("corpus/cyclonedx-generalization-evidence.v1.json");
const coreMl = ledger.populations.coreml;
check.expectEqual(coreMl.manifest_sha256, sha256(coreMlManifest), "Core ML contract-corpus manifest SHA-256");
check.expectEqual(coreMl.artifact_count, 5, "Core ML MLProgram contract denominator");
check.expectEqual(coreMl.contract_classes.join(","), "blockwise_affine_compression,bounded_shape_range,enumerated_shape,lut_palettization_compression,static_external_blob", "Core ML contract strata");
check.expectEqual(coreMl.ecosystem_prevalence_claim, false, "Core ML generated fixtures must not claim ecosystem prevalence");
check.expectEqual(coreMl.public_population?.source_sha256, sha256(coreMlPublicEvidence), "Core ML public-population source SHA-256");
check.expectEqual(coreMl.public_population?.artifact_count, 27, "Core ML public-artifact denominator");
check.expectEqual(coreMl.public_population?.path_record_count, 27, "Core ML public path denominator");
check.expectEqual(coreMl.public_population?.catalog_source_count, 1, "Core ML public catalog-source denominator");
check.expectEqual(coreMl.public_population?.serialized_graph_present_count, 22, "Core ML decoded serialized-graph count");
check.expectEqual(coreMl.public_population?.serialized_graph_payload_not_decoded_count, 5, "Core ML unread serialized-payload boundary");
check.expectEqual(Object.values(coreMl.public_population?.architecture_strata || {}).reduce((sum, count) => sum + count, 0), 27, "Core ML architecture strata conservation");
check.expectEqual(Object.values(coreMl.public_population?.task_strata || {}).reduce((sum, count) => sum + count, 0), 27, "Core ML task strata conservation");
check.expectEqual(Object.values(coreMl.public_population?.precision_strata || {}).reduce((sum, count) => sum + count, 0), 27, "Core ML precision strata conservation");
check.expectEqual(Object.values(coreMl.public_population?.quantization_classification || {}).reduce((sum, count) => sum + count, 0), 27, "Core ML quantization-state conservation");
check.expectEqual(Object.values(coreMl.public_population?.serialized_contract_status || {}).reduce((sum, count) => sum + count, 0), 27, "Core ML serialized-contract conservation");
check.expectEqual(coreMl.public_population?.ecosystem_prevalence_claim, false, "Core ML public cohort must not claim ecosystem prevalence");
check.expectEqual(coreMl.public_population?.device_placement_claim, false, "Core ML public cohort must not claim device placement");
check.expectEqual(coreMl.compiled_plan_population?.artifact_count, 0, "Core ML compiled-plan evidence boundary");
check.expectEqual(coreMl.compiled_plan_population?.status, "runtime_evidence_required", "Core ML compiled-plan status");
const ggufManifest = readFileSync("corpus/gguf-architecture-encoding-corpus.v1.json");
const gguf = ledger.populations.gguf;
check.expectEqual(gguf.manifest_sha256, sha256(ggufManifest), "GGUF corpus manifest SHA-256");
check.expectEqual(gguf.artifact_count, 8, "GGUF corpus denominator");
check.expectEqual(gguf.architecture_count, 4, "GGUF architecture strata");
check.expectEqual(gguf.encoding_count, 10, "GGUF observed encoding strata");
check.expectEqual(gguf.unsupported_encoding_tensor_count, 0, "GGUF unsupported encoding tensors");
const safeTensorsManifest = readFileSync("corpus/safetensors-architecture-corpus.v1.json");
const safeTensors = ledger.populations.safetensors;
check.expectEqual(safeTensors.manifest_sha256, sha256(safeTensorsManifest), "SafeTensors corpus manifest SHA-256");
check.expectEqual(safeTensors.artifact_count, 3, "SafeTensors family corpus denominator");
check.expectEqual(safeTensors.architecture_classes.join(","), "dense_decoder,sparse_moe_decoder,ssm_recurrent", "SafeTensors family strata");
check.expectEqual(safeTensors.shape_mismatch_count, 0, "SafeTensors canonical tensor-shape mismatches");

check.expectEqual(priorities.map((row) => row.rank).join(","), priorities.map((_, index) => index + 1).join(","), "contiguous deterministic ranks");
check.expect(!priorities.some((row) => row.id === "onnx-shape-rule-coverage"), "A zero unsupported-shape residual must not remain in the action-priority ledger.");
check.expect(!priorities.some((row) => row.id === "onnx-ort-contrib-shape-semantics"), "A zero ORT contrib shape residual must not remain in the action-priority ledger.");
check.expect(!priorities.some((row) => row.id === "onnx-symbolic-shape-rank-propagation"), "A zero non-conflict symbolic-shape residual must not remain in the ranked priority ledger.");
check.expect(priorities.every((row) => row.evidence && row.next_action && row.status), "Every priority should preserve evidence, boundary status, and next action.");
check.expect(ledger.method?.population_boundary?.includes("not ecosystem prevalence") || ledger.method?.non_claim?.includes("not generalized"), "Single-anchor formats should carry an explicit non-generalization boundary.");
check.expect(ledger.method?.reason_histogram_projection?.includes("compressed sweep") && ledger.method.reason_histogram_projection.includes("exact counts"), "Reason-class aggregation must disclose where detailed identifiers remain and preserve count semantics.");

check.done(`${onnx.artifact_count} measured ONNX artifacts, three SafeTensors family anchors, eight GGUF strata, 27 public Core ML artifacts, and five generated Core ML contracts conserve the residual roadmap without merging evidence classes.`);
