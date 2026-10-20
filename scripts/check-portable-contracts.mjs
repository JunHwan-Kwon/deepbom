import assert from "node:assert/strict";
import { initializeNumericalWasm, numericalWasm } from "../web/lib/numerical-wasm.js";
import { aggregateExactCounts, portableContract } from "../web/lib/portable-contracts.js";
import { protobufWireFields, protobufVarint, protobufPackedFloats } from "../web/lib/portable-protobuf.js";
import { optimizationFieldChanges } from "../web/lib/native-optimization-diff.js";
import { compareNativeNodes } from "../web/lib/native-evidence.js";
import { deriveGraphTopology } from "../web/lib/graph-topology.js";
import { convTransposeAxisPairs, integerConstant, integerSymbol, integerCall, evaluateIntegerExpression, evaluateGuardedIntegerFormula, serializeGuardedIntegerFormula, equalityGuard, divisibleGuard } from "../web/lib/guarded-integer-expression.js";
import { indexModelProgram, modelGroupBoundary, partitionConnectedOperations } from "../web/lib/model-program-boundary.js";
await initializeNumericalWasm();

assert.deepEqual(aggregateExactCounts([null, "9007199254740993", "7"]), {
  decimal: "9007199254741000", number: null, status: "partial", assessed_count: 2, total_count: 3,
});
assert.equal(aggregateExactCounts([null]).decimal, null);
assert.equal(aggregateExactCounts([]).decimal, "0");
assert.equal(aggregateExactCounts([], false).decimal, null);
const symbol = integerSymbol("N"), c = integerConstant;
const formula = serializeGuardedIntegerFormula(integerCall("exact_div", symbol, c(3)), "elements", "test", [divisibleGuard(symbol,c(3))]);
assert.equal(evaluateGuardedIntegerFormula(formula,{N:"9007199254740993"}),3002399751580331n);
assert.equal(evaluateGuardedIntegerFormula(formula,{N:9007199254740992}),null);
assert.equal(evaluateGuardedIntegerFormula(formula,{N:4}),null);
assert.equal(evaluateIntegerExpression(integerCall("sub", c(2), c(7)),{}),-5n);
assert.equal(evaluateGuardedIntegerFormula(serializeGuardedIntegerFormula(c(-1),"n","test"),{}),null);
assert.equal(evaluateGuardedIntegerFormula(serializeGuardedIntegerFormula(symbol,"n","test",[equalityGuard(symbol,c(9))]),new Map([["N",9n]])),9n);
assert.equal(evaluateIntegerExpression(integerSymbol("1"),new Map([[1,3n]])),null);
for(let input=0n;input<9n;input++) for(let kernel=1n;kernel<7n;kernel++) for(let stride=1n;stride<4n;stride++) for(let pad=0n;pad<4n;pad++) {
  let expected=0n;
  for(let i=0n;i<input;i++) for(let k=0n;k<kernel;k++) if(i*stride+k*2n>=pad && i*stride+k*2n<pad+7n) expected++;
  assert.equal(convTransposeAxisPairs(input,kernel,stride,2n,pad,7n),expected);
}
assert.equal(convTransposeAxisPairs(10n**50n,1n,1n,1n,0n,10n**50n),10n**50n);
assert.equal(convTransposeAxisPairs(1000001n,1000001n,1n,1n,0n,7n),null);
for (let n = 0n; n < 500n; n += 7n) for (const width of [1n, 3n, 4n, 6n, 8n, 16n, 64n]) {
  const count = n * 9007199254740991n;
  const bits = count * width;
  const expected = bits / 8n + (bits % 8n ? 1n : 0n);
  assert.equal(portableContract("pack", { elements: String(count), bits: Number(width) }).decimal, String(expected));
}
assert.deepEqual(optimizationFieldChanges({"a/b":null, array:[1], "😀":1, "\ue000":1}, {array:[1,2],"😀":2,"\ue000":2}).map(r=>r.path), ["/a~1b","/array","/😀","/\ue000"]);
assert.equal(optimizationFieldChanges({a:null},{})[0].after_present,false);
assert.deepEqual(compareNativeNodes([{id:"a",x:1},{id:"b"}],[{id:"a",x:2},{id:"c"}]),[
  {id:"a",status:"changed"},{id:"b",status:"removed"},{id:"c",status:"added"},
]);

const operations = ["a","b","c","d"].map(id=>({id}));
const relationships = [{kind:"data_dependency",from_ref:"a",to_ref:"c"},{kind:"data_dependency",from_ref:"c",to_ref:"d"}];
assert.deepEqual(partitionConnectedOperations(operations,relationships,2).map(g=>g.map(r=>r.id)),[["a","c"],["b"],["d"]]);
const program={operations:operations.slice(0,2),values:[{id:"input",storage_refs:[]},{id:"x",storage_refs:[]},{id:"y",storage_refs:[]}],
  ports:[{id:"i",direction:"input",operation_ref:"a",value_ref:"input",position:0},
         {id:"ao",direction:"output",operation_ref:"a",value_ref:"x",position:0},
         {id:"b0",direction:"input",operation_ref:"b",value_ref:"x",position:0},
         {id:"b1",direction:"input",operation_ref:"b",value_ref:"x",position:1},
         {id:"bo",direction:"output",operation_ref:"b",value_ref:"y",position:0}],
  programs:[{id:"main",output_value_refs:["y"]}]};
const index=indexModelProgram({program});
assert.equal(index.edges.length,4);
assert.deepEqual(modelGroupBoundary(index,["b"]), {input_value_refs:["x"],output_value_refs:["y"],internal_edge_refs:[],incoming_edge_refs:["port-edge:b0","port-edge:b1"],outgoing_edge_refs:["output-edge:main:0:y"]});
assert.deepEqual(modelGroupBoundary(index,["a","b"]).internal_edge_refs,["port-edge:b0","port-edge:b1"]);
assert.throws(()=>modelGroupBoundary(index,["b","b"]));
const long = Array.from({length:20000},(_,i)=>({index:i,name:"Conv",inputs:[i],outputs:[i+1]})).reverse();
assert.equal(deriveGraphTopology(long).annotations.at(-1).depth,19999);
assert.equal(deriveGraphTopology([{index:0,inputs:[2],outputs:[1]},{index:1,inputs:[1],outputs:[2]}]).status,"invalid_cycle_detected");

const bytes=Uint8Array.from([8,150,1,18,3,97,98,99]);
assert.deepEqual(protobufWireFields(bytes).map(r=>[r.no,r.wire,r.offset,r.length,r.value]),[[1,0,1,2,"150"],[2,2,5,3,null]]);
assert.equal(protobufVarint(Uint8Array.from([255,255,255,255,255,255,255,255,255,1])).value,18446744073709551615n);
assert.deepEqual(Array.from(numericalWasm().protobuf_packed_varints(Uint8Array.from([1,150,1]))),[1n,150n]);
for(const malformed of [[0,0],[8,128],[18,5,1],[13,0],[8,255,255,255,255,255,255,255,255,255,2],[8,128,128,128,128,128,128,128,128,128,128,0]]) assert.throws(()=>protobufWireFields(Uint8Array.from(malformed)));
assert.throws(()=>protobufPackedFloats(Uint8Array.of(1,2,3),4));
const floats=new Uint8Array(12),view=new DataView(floats.buffer);view.setFloat32(0,1.25,true);view.setFloat32(4,-0,true);view.setFloat32(8,Infinity,true);
const decoded=protobufPackedFloats(floats,4);assert.equal(decoded[0],1.25);assert(Object.is(decoded[1],-0));assert.equal(decoded[2],Infinity);
console.log("Portable contracts: exact counts/unknowns, 504 packing oracles, JSON presence/order, port boundaries, connected partitions, 20,000-node topology and protobuf malformed bounds passed.");
