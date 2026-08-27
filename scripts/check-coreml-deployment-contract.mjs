import assert from "node:assert/strict";
import { File } from "node:buffer";

import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { buildCoreMlDeploymentContract, coreMlFloorLabel } from "../web/lib/coreml-deployment-contract.js";

function varint(value) { let x = BigInt.asUintN(64, BigInt(value)); const out = []; while (x > 127n) { out.push(Number(x & 127n) | 128); x >>= 7n; } out.push(Number(x)); return Buffer.from(out); }
function concat(...rows) { return Buffer.concat(rows.flat(Infinity).filter(Boolean)); }
function key(field, wire) { return varint(field * 8 + wire); }
function uint(field, value) { return concat(key(field, 0), varint(value)); }
function bytes(field, value) { const body = Buffer.from(value); return concat(key(field, 2), varint(body.length), body); }
function message(field, value) { return bytes(field, value); }
function string(field, value) { return bytes(field, Buffer.from(value, "utf8")); }
function packed(field, values) { return bytes(field, concat(values.map(varint))); }
function feature(name, type) { return concat(string(1, name), message(3, type)); }
function sizeRange(lower, upper) { return concat(uint(1, lower), uint(2, upper)); }
function identityModel(version, description) { return concat(uint(1, version), message(2, description), message(900, Buffer.alloc(0))); }

function flexibleImageType(width = 4, height = 4) {
  const imageSize = (w, h) => concat(uint(1, w), uint(2, h));
  const enumerated = concat(message(1, imageSize(4, 4)), message(1, imageSize(8, 8)));
  return message(4, concat(uint(1, width), uint(2, height), uint(3, 20), message(21, enumerated)));
}

function rangedArrayType(shape = [2, 3], dtype = 65568) {
  const ranges = concat(message(1, sizeRange(1, 4)), message(1, sizeRange(2, 8)));
  return message(5, concat(packed(1, shape), uint(2, dtype), message(31, ranges)));
}

const flexibleDescription = concat(
  message(1, feature("image", flexibleImageType())),
  message(10, feature("features", rangedArrayType())),
);
const flexible = (await readCoreMlModelFile(new File([identityModel(3, flexibleDescription)], "flexible.mlmodel"))).analysis;
assert.equal(flexible.inputs[0].constraints.flexibility.kind, "enumerated");
assert.deepEqual(flexible.inputs[0].constraints.flexibility.sizes, [{ width: 4, height: 4 }, { width: 8, height: 8 }]);
assert.equal(flexible.outputs[0].constraints.flexibility.kind, "range");
assert.deepEqual(flexible.outputs[0].constraints.flexibility.dimensions.map((range) => [range.lower_bound, range.upper_bound]), [[1, 4], [2, 8]]);
assert.equal(flexible.coreml.deployment_floor.status, "assessed");
assert.equal(flexible.coreml.deployment_floor.observed_feature_minimum_specification_version, 3);
assert.match(coreMlFloorLabel(flexible.coreml.deployment_floor.declared_load_floor), /iOS 12; macOS 10.14/);

const sequenceType = message(7, concat(message(1, Buffer.alloc(0)), message(101, sizeRange(0, 64))));
const stateArray = message(5, concat(packed(1, [1, 16]), uint(2, 65552)));
const stateType = message(8, message(1, concat(packed(1, [1, 16]), uint(2, 65552))));
const stateDescription = concat(
  message(1, feature("tokens", sequenceType)),
  message(10, feature("result", stateArray)),
  message(13, feature("cache", stateType)),
);
const state = (await readCoreMlModelFile(new File([identityModel(9, stateDescription)], "state.mlmodel"))).analysis;
assert.equal(state.inputs[0].dtype, "SEQUENCE<INT64>");
assert.deepEqual(state.inputs[0].constraints.size_range, {
  lower_bound: 0,
  upper_bound: 64,
  upper_bound_unbounded: false,
  serialized_upper_bound: 64,
});
assert.equal(state.states[0].feature_type, "state");
assert.equal(state.states[0].dtype, "FLOAT16");
assert.deepEqual(state.states[0].shape, [1, 16]);
assert.equal(state.coreml.deployment_floor.observed_feature_minimum_specification_version, 9);

const int8Description = concat(message(1, feature("input", message(5, concat(packed(1, [1, 4]), uint(2, 131080))))));
const int8 = (await readCoreMlModelFile(new File([identityModel(10, int8Description)], "int8.mlmodel"))).analysis;
assert.equal(int8.inputs[0].dtype, "INT8");
assert.equal(int8.coreml.deployment_floor.observed_feature_minimum_specification_version, 10);
assert.equal(int8.runtime_requirements.declared_os_floor.visionos, "26");

await assert.rejects(
  readCoreMlModelFile(new File([identityModel(2, flexibleDescription)], "bad-floor.mlmodel")),
  /declared specification 2 is below observed feature floor 3/,
);
await assert.rejects(
  readCoreMlModelFile(new File([identityModel(3, concat(message(1, feature("bad", flexibleImageType(5, 5)))))], "bad-default.mlmodel")),
  /default image size is not one of its enumerated sizes/,
);

const newer = buildCoreMlDeploymentContract({ specificationVersion: 11, modelType: "identity" });
assert.equal(newer.status, "not_assessed_newer_than_pinned_table");
assert.equal(newer.declared_load_floor, null);

console.log("Core ML deployment-floor and interface contracts passed (OS table, flexible shapes, sequence/state, INT8, and contradiction rejection).");
