import { evidenceContract, sealDocument } from "./evidence-identity.js";

export const EVALUATION_PROTOCOL_SCHEMA = "deepbom.evaluation_protocol.v1";
export const EVALUATION_RECORD_SCHEMA = "deepbom.evaluation_record.v1";
export function exactCount(value, label, positive = false) {
  evidenceContract("count", value, {label, positive});
  return BigInt(value);
}
export function buildEvaluationProtocol(input) {
  evidenceContract("protocol", input, {sealed:false});
  return sealDocument(structuredClone(input), "protocol_sha256");
}
export function validateEvaluationProtocol(doc) {
  evidenceContract("protocol", doc, {sealed:true});
  return doc;
}
export function buildEvaluationRecord(input) {
  evidenceContract("evaluation", input, {sealed:false});
  return sealDocument(structuredClone(input), "evaluation_sha256");
}
export function validateEvaluationRecord(doc) {
  evidenceContract("evaluation", doc, {sealed:true});
  return doc;
}
export function validateMeasurementAgainstProtocol(record, protocol) {
  evidenceContract("measurement", record, {protocol});
  return record;
}
