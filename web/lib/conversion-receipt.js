import { portableContract } from "./portable-contracts.js";
import { canonicalJson } from "./report-utils.js";
export const CONVERSION_RECEIPT_SCHEMA = "deepbom.conversion_receipt.v1";
export const BOUND_CONVERSION_RECEIPT_SCHEMA = "deepbom.bound_conversion_receipt.v1";
function receipt(operation, value, active = null, receiptFileSha256 = null) {
  canonicalJson(value); canonicalJson(active);
  return Object.freeze(portableContract("conversion_receipt",{operation,value,active,receipt_file_sha256:receiptFileSha256}));
}
export function buildConversionReceipt(input) { return receipt("build",input); }
export function validateConversionReceipt(document) { return receipt("validate",document); }
export function bindConversionReceipt(document,activeArtifact,{receiptFileSha256=null}={}) { return receipt("bind",document,activeArtifact,receiptFileSha256); }
export function validateBoundConversionReceipt(document,activeArtifact=null) { return receipt("bound",document,activeArtifact); }
export function buildConversionLineageEvidence(analysis,activeArtifact) { return receipt("lineage",analysis?.conversion_receipt ?? null,activeArtifact); }
