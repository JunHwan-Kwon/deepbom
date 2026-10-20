# Monitor CPU measurement record

These are development-build measurements, not a production overhead guarantee. All 120 runs preserved the exact controlled final state and loss sequence. Setup, training and close are measured separately. A full queue skips captures explicitly; skipped work is not a speed improvement.

## Conditions

Linux x86_64 shared host (48 logical CPUs), PyTorch 2.8.0 CPU, TensorFlow 2.20.0 CPU, NumPy 2.2.6. The deterministic two-layer dense workload uses width 16 or 512, batch 8, SGD, 10 warmup updates and 100 measured updates. Three repetitions rotate/reverse condition order. Width 512 is still a synthetic model, not a large production workload. No other DEEPBOM build or test was started during these trials; unrelated host activity was not controlled.

`scalar` records supplied loss and framework collector events. Torch also observes optimizer boundaries; the manual TensorFlow scalar loop does not infer optimizer semantics. `snapshot_copy` copies and serializes snapshots without the Monitor. `snapshot_worker` also records scalar/lifecycle evidence. `activation` explicitly captures a snapshot and performs an extra deterministic forward; it is not a free hook-only observation.

## Capture saturation

Median seconds for the 100-update training interval; setup and close excluded. Every 10 updates requests ten captures; compare actual completions below.

| Framework | Width | Off | Scalar | Copy only | Copy + worker | Explicit activation | Worker completed / requested |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| torch | 16 | 0.0281 | 0.3031 | 0.0353 | 0.4200 | 0.9759 | 2/10, 2/10, 2/10 |
| torch | 512 | 0.0589 | 0.3480 | 0.1162 | 0.6237 | 5.1049 | 2/10, 2/10, 2/10 |
| tensorflow | 16 | 0.5243 | 0.6550 | 0.5329 | 0.7629 | 1.4148 | 3/10, 3/10, 3/10 |
| tensorflow | 512 | 0.5742 | 0.7049 | 0.6389 | 1.1165 | 5.5228 | 3/10, 3/10, 2/10 |

Worker setup medians: 1.110–1.122 s; close medians: 0.585–1.445 s.
All admitted worker captures completed. Copy-only and worker totals are unequal-work comparisons because the worker skipped requests before copying.

## One completed capture per run

Median seconds for the 100-update training interval; setup and close excluded. A period of 1,000 requests one capture within the 100-update run. All worker requests completed without skips or failures.

| Framework | Width | Off | Scalar | Copy only | Copy + worker | Explicit activation | Worker completed / requested |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| torch | 16 | 0.0273 | 0.3069 | 0.0281 | 0.3833 | 0.4108 | 1/1, 1/1, 1/1 |
| torch | 512 | 0.0559 | 0.3493 | 0.0691 | 0.5143 | 0.9465 | 1/1, 1/1, 1/1 |
| tensorflow | 16 | 0.5029 | 0.6363 | 0.4983 | 0.7350 | 0.7471 | 1/1, 1/1, 1/1 |
| tensorflow | 512 | 0.5532 | 0.6905 | 0.5552 | 0.9055 | 1.3128 | 1/1, 1/1, 1/1 |

Worker setup medians: 1.107–1.126 s; close medians: 0.041–0.413 s.
This equal-capture-count run measures one completed snapshot per condition. It does not make the conditions identical: scalar/lifecycle handling, serialization and analysis remain different work.

## Interpretation and reproduction

Exact producer hashes, settings, per-run timings, storage/copy/queue observations and sampled RSS are in the [machine-readable record](monitor-overhead-evidence-contracts-cpu-20261014.json). Child RSS is sampled every 10 ms and summed across processes; short peaks can be missed and shared pages counted repeatedly. It is not a limit or GPU VRAM. The [earlier shape-contract measurement](monitor-overhead-shape-contracts-cpu-20261014.json) remains separate and is not substituted for the new producer.

The basic preset avoids full weight/activation capture, but it is not negligible on these very short steps. Detailed capture and persistence can dominate the training interval. Choose a period using completed work, setup/close and your real model; these results do not establish a universal period or sub-1% overhead.

Install the named frameworks and `psutil` in the experiment environment; `psutil` is not a normal library dependency. Run against the intended installed package, using a fresh output directory for each command:

```bash
python scripts/benchmark-monitor-overhead.py --output /tmp/deepbom-saturation   --widths 16,512 --steps 100 --warmup 10 --repetitions 3 --every 10
python scripts/benchmark-monitor-overhead.py --output /tmp/deepbom-one-capture   --widths 16,512 --steps 100 --warmup 10 --repetitions 3 --every 1000
```

`experiment.json` records the protocol and engine before execution; `measurements.json` records per-run observations; `validation.json` exists only after controlled state/loss comparisons pass. GPU/distributed collection, production overhead and total host RSS limits are not qualified by this CPU experiment.
