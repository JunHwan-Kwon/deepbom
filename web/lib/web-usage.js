import { createChatGptUsage, mountUsageControls } from './chatgpt-usage.js';

const formats = new Set(['onnx', 'tflite', 'gguf', 'safetensors', 'coreml', 'executorch', 'graphdef', 'savedmodel', 'hdf5', 'keras', 'pt2', 'pytorch_checkpoint']);
const usage = createChatGptUsage({ endpoint: new URL('/', import.meta.url), channel: 'web' });
let format = 'unknown';
let completed = false;
let generation = 0;
const normalized = (value) => formats.has(value) ? value : value === 'tensorflow_protobuf' ? 'graphdef' : 'unknown';

// Call sites supply a format or export kind, never a model object or filename.
export const webUsage = {
  mount(container) { if (container) mountUsageControls(container, usage); },
  start(value) { generation += 1; usage.beginRun(); completed = false; format = normalized(value); usage.track('analysis_started', { format }); },
  complete(value, graphRendered = false) {
    format = normalized(value); completed = true;
    usage.track('analysis_completed', { format });
    if (graphRendered) usage.track('visualization_rendered', { format, detail: 'web-graph' });
  },
  fail() { usage.track('analysis_failed', { format }); },
  export(detail) {
    if (completed && ['svg', 'png', 'word', 'cyclonedx', 'spdx', 'model_views', 'evidence_package'].includes(detail)) usage.track('export_prepared', { format, detail });
  },
  observeExport(detail) {
    const startedGeneration = generation;
    return () => { if (startedGeneration === generation) webUsage.export(detail); };
  },
  observeTextExport() {
    const svg = webUsage.observeExport('svg');
    const cyclonedx = webUsage.observeExport('cyclonedx');
    return (artifact) => {
      if (artifact.type === 'image/svg+xml') svg();
      if (artifact.type.startsWith('application/vnd.cyclonedx')) cyclonedx();
    };
  },
};
