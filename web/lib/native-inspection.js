import { buildNativeEvidence } from './native-evidence.js';
import { buildNativeStructureProjection } from './native-optimization-report.js';
import { exactFields } from './evidence-identity.js';

// Transport-independent owner. Framework adapters supply observations; this
// projection uses the same state identities and cost rules as optimization.
export async function inspectNative(payload) {
  exactFields(payload, ['snapshot', 'probe'], [], 'native inspection');
  const evidence = await buildNativeEvidence(payload.snapshot);
  const structure = buildNativeStructureProjection(evidence, payload.probe, '1.1.0');
  return {
    schema: 'deepbom.native_inspection.v1', cost_method: '1.1.0',
    evidence, structure,
    execution: {status: 'completed', shape_probe: payload.probe.status},
    validation: {numerical_equivalence: 'not_requested', task_quality: 'not_requested', runtime_latency: 'not_measured'},
  };
}
