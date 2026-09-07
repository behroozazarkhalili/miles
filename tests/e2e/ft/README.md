# Fault Tolerance E2E Tests

## Overview Table

### CI Entries

- **CI entry files**: `test_<TEST_NAME>__<mode>.py`, or `test_<TEST_NAME>__<kill>.py` when the scenario pins its own topology and takes no mode; split on the first `__` to read the scenario and the rest back out, which is why no scenario name contains one.
- **The segment after the scenario is always the kill segment**: a mode name starts with it, and an entry with no mode carries it alone, so every entry says what its run crashes without anyone opening the file.
- **Entry file content**: `register_cuda_ci(est_time=..., suite=..., labels=[...])` plus `run_ci(_MODE)` under `__main__`, no test logic.
- **Execution model**: bare `python3 <file>` from the repo root, exit code = pass/fail (`tests/ci/ci_utils.py` `run_unittest_files`).

| Scenario | Modes with an entry file |
| --- | --- |
| `scenario_trainer_no_failure` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp4_cp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer` |
| `scenario_trainer_deterministic` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp4_cp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer` |
| `scenario_trainer_with_failure` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2` |
| `scenario_rollout_deterministic` | `kill_rollout__dp4` |
| `scenario_random_crash` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer`, `kill_train_rollout__dp2_cp2`, `kill_rollout__dp4` |
| `scenario_realistic_gsm8k` | `test_realistic_gsm8k__kill_train_rollout.py`, no modes |
| `scenario_random_crash_fully_async` | `kill_train_rollout__dp2_cp2` |
| `scenario_realistic_gsm8k_fully_async` | `test_realistic_gsm8k_fully_async__kill_train_rollout.py`, no modes |
| `scenario_weight_update_all_gather` | `kill_train_rollout__dp2_tp2` |
| `scenario_weight_update_p2p_local` | `kill_train_rollout__dp2_tp2` |
| `scenario_weight_update_p2p_remote` | `kill_train_rollout__dp2_tp2` |
| `scenario_weight_update_all_gather_deadlock` | `kill_train_rollout__dp2_tp2` |
| `scenario_weight_update_p2p_local_sigstop` | `kill_train_rollout__dp2_tp2` |
| `scenario_weight_update_p2p_remote_sigstop` | `kill_train_rollout__dp2_tp2` |

- **Forced absences**, one reason each:
    - `kill_train__dp4_cp2_tp2_pp2_ep2_etp2__moe_full` is multi-node, and no multi-node CI lane exists.
    - `kill_rollout__dp4` fits only the scenarios that crash engines.
    - `scenario_rollout_deterministic` needs real engines and `ft_components == ("rollout",)` exactly.
    - The fully-async soaks reject modes without real engines or with colocation.
    - `kill_train__dp2_cp2` supersedes `kill_train__dp2_cp2__moe_5layer` in `scenario_trainer_with_failure`.
    - `scenario_trainer_with_failure` x `kill_train__dp4_cp2__fake_rollout__moe_5layer` is an authorized skip.
- **Every other absence is an unclaimed cell**, not a decision — adding an entry file is all it takes.

### Scenarios

- **Scenario logic**: `conftest_ft/scenario_<name>.py` — a typer app plus a `run_ci(mode)` runner.

| Scenario (`conftest_ft/scenario_*.py`) | Type | What it verifies |
| --- | --- | --- |
| `scenario_trainer_no_failure` | comparison | indep_dp matches normal DP when no faults |
| `scenario_trainer_with_failure` | comparison, multi-phase | indep_dp matches normal DP after fault + ckpt resume |
| `scenario_trainer_deterministic` | comparison, multi-phase | healing state transfer is bitwise-correct, on cold start and on resume from a post-healing ckpt |
| `scenario_rollout_deterministic` | comparison | engine crashes change training bits not at all |
| `scenario_random_crash` | soak | system survives random crashes without hanging |
| `scenario_realistic_gsm8k` | soak | model still reaches gsm8k accuracy under random crashes |
| `scenario_random_crash_fully_async` | soak | same, through `train_async.py --fully-async` |
| `scenario_realistic_gsm8k_fully_async` | soak | same, through `train_async.py --fully-async` |
| `scenario_weight_update_all_gather` | targeted | a trainer worker dying inside the weight update's own TP all-gather is survived |
| `scenario_weight_update_p2p_local` | targeted | a sender dying at the p2p write it is about to issue is survived |
| `scenario_weight_update_p2p_remote` | targeted | the engine a sender has just submitted a write to dying is survived, and confined to that engine |
| `scenario_weight_update_all_gather_deadlock` | targeted | the same all-gather point, with a sender that hangs holding the GIL instead of dying |
| `scenario_weight_update_p2p_local_sigstop` | targeted | the same p2p write point, with a sender frozen by SIGSTOP instead of killed |
| `scenario_weight_update_p2p_remote_sigstop` | targeted | the receiver of a submitted write frozen by SIGSTOP instead of killed |

### Modes

- **Selection**: `--mode`, defined in `conftest_ft/modes.py`; `scenario_realistic_gsm8k` takes none.
- **Mode names**: `<kill>__<parallelism>[__fake_rollout][__moe_5layer|__moe_full][__colocate]`, segments separated by `__` and joined by `_` inside a segment.
- **What a name carries**: the `kill` segment always, then only the axes that differ from the naming defaults — real sglang engines, the dense `Qwen3-0.6B`, disaggregated placement. Node counts, engine counts and cell counts are never in the name; read them from the table below.
- **Why `kill` leads**: what a run crashes is the subject of this suite, so it is the first thing the name answers, and it is a property of the mode alone — no scenario widens it at runtime.
- **The scheme is enforced, not remembered**: `compute_mode_name` derives a mode's name from its fields against an explicit naming-default table, and `tests/fast/e2e/ft/test_naming_scheme.py` fails when a name drifts from it.
- **Declared per mode**: cell count, parallelism, model, train/rollout GPU split, `colocate` (default disaggregated, i.e. training and rollout on separate nodes), `ft_components` (default `("train",)`).
- **No rollout engines**: modes with `rollout_num_engines == 0` train on pre-recorded debug rollout data.
- **No registered mode colocates**: `FTTestMode` still takes `colocate`, and the `__colocate` name segment still exists, but the last colocated mode became `kill_rollout__dp4` when the injection scenarios moved to p2p, which `validate_args` refuses under `--colocate`. A colocated mode is therefore reachable only from a scenario that transfers weights by broadcast.

| Mode | Nodes | GPUs (train + rollout) | DP cells | Parallelism | Rollout | Model | `ft_components` | Why it exists |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 2 | CP2 TP2 EP2 | debug data | 5-layer MoE | `("train",)` | TP + EP coverage |
| `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 2 | CP2 PP2 | debug data | 5-layer MoE | `("train",)` | PP coverage, via `--decoder-first-pipeline-num-layers 3 --decoder-last-pipeline-num-layers 2` |
| `kill_train__dp4_cp2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 4 | CP2 | debug data | 5-layer MoE | `("train",)` | multi-replica coverage (>= 4 cells) |
| `kill_train__dp2_cp2__moe_5layer` | 1 | 4 + 4 | 2 | CP2 | 4 engines × 1 GPU | 5-layer MoE | `("train",)` | real engines + the weight-update path |
| `kill_train__dp2_cp2` | 1 | 4 + 4 | 2 | CP2 | 4 engines × 1 GPU | dense Qwen3-0.6B | `("train",)` | `scenario_trainer_with_failure` under real generation; needs the dense model (see below) |
| `kill_train_rollout__dp2_cp2` | 1 | 4 + 4 | 2 | CP2 | 4 engines × 1 GPU | dense Qwen3-0.6B | `("train", "rollout")` | both kinds crash in the same run, sync and fully-async; disaggregated, since colocation makes the two crashes contend for the same gpus |
| `kill_train_rollout__dp2_tp2` | 1 | 4 + 4 | 2 | TP2 | 4 engines × 1 GPU | dense Qwen3-0.6B | `("train", "rollout")` | the only mode whose trainer runs a real TP all-gather in the weight update, which the targeted hook scenarios crash inside; the killed sender takes its engines with it, so both kinds must be recoverable |
| `kill_rollout__dp4` | 1 | 4 + 4 | 4 | — | 4 engines × 1 GPU | dense Qwen3-0.6B | `("rollout",)` | the only rollout-only mode: crashes engines, not trainer cells |
| `kill_train__dp4_cp2_tp2_pp2_ep2_etp2__moe_full` | 4 train + 2 rollout | 32 + 16 | 4 | CP2 TP2 PP2 EP2 ETP2 | 2 engines × 8 GPU | full MoE | `("train",)` | full model, all parallelism; multi-node, so no CI entry |

- **Batch shape**: `--rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256` everywhere — 256 samples per rollout, divisible by both 2 and 4 cells. Uneven distribution across replicas is **not** exercised.
- **Model**: 1-node modes use the 5-layer MoE `Qwen3-30B-A3B-5layer`, except the dense modes.

### Weight Transfer Protocol

- **The injection scenarios launch p2p**: `scenario_random_crash`, `scenario_realistic_gsm8k`, their fully-async shells and `scenario_rollout_deterministic` add `get_weight_transfer_args(mode)`, which is `--update-weight-transfer-mode p2p --sglang-remote-instance-weight-loader-start-seed-via-transfer-engine`.
- **Why p2p is the protocol under test**: what these scenarios crash is an engine that a weight update is talking to, and the update path that has per-cell sessions, RDMA writes and partial-target failure handling is the p2p one. Under the `broadcast` default a crashed engine tests a collective the production p2p deployments do not run.
- **Why the sglang seed flag**: p2p reads each engine's registration through `/get_remote_instance_transfer_engine_info`, which an engine that never started its remote-instance transfer engine does not serve. Every in-repo p2p launcher passes the same flag.
- **Why colocation is refused**: `validate_args` rejects `--update-weight-transfer-mode p2p` together with `--colocate` (colocated transfers move a CUDA IPC handle, not bytes), so `get_weight_transfer_args` asserts on a colocated mode rather than letting the run fail on the cluster.
- **A mode without engines names no protocol**: `rollout_num_engines == 0` pushes weights to nobody, so those modes get no transfer flag at all instead of a p2p claim nothing exercises.
- **The comparison scenarios keep the broadcast default**: `scenario_trainer_no_failure`, `scenario_trainer_with_failure` and `scenario_trainer_deterministic` do not call `get_weight_transfer_args`, so `kill_train__dp2_cp2__moe_5layer` and `kill_train__dp2_cp2` still transfer by broadcast there and by p2p under `scenario_random_crash`. The transfer protocol is a property of the scenario, not of the topology.

## Running the code

### In CI

- **Gating labels**: `run-ci-ft-short` for the comparison scenarios (minutes each), `run-ci-ft-long` for the soaks (tens of minutes to hours). Nothing here runs on an unlabelled PR.
- **Broad scopes**: `run-ci-all` includes both; the nightly cadence includes `ft-short` but not `ft-long`; `run-ci-image` excludes both.
- **Suite**: `suite="stage-c-8-gpu-h200"`, run by the job of the same name in `.github/workflows/pr-test.yml`.
- **ft-long is disabled**: every ft-long entry passes `disabled="FT soak tests pending CI infra support"`, and `tests/ci/run_suite.py` drops every test with a non-`None` `disabled`, so `run-ci-ft-long` executes nothing. Unblocked by an ft-long capable lane; nothing in the tests is known broken.
- **Fast-layer stand-in**: `tests/fast/e2e/ft/test_rollout_gated_recovery.py` covers suspend → gated relaunch → recovery on CPU meanwhile.
- **Add a `(scenario, mode)`**: copy an entry file, change `_MODE`.
- **Add a label**: an entry in `tests/ci/labels.py` plus the matching `run-ci-<key>` GitHub label; the workflow needs no edit.

### Manually

`PYTHONPATH` must point at the repo root (CI sets it automatically).

```bash
# One mode, exactly as CI runs it
PYTHONPATH=. python tests/e2e/ft/test_trainer_no_failure__kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer.py

# Any mode, including the ones with no entry file
PYTHONPATH=. python tests/e2e/ft/conftest_ft/scenario_trainer_no_failure.py run --mode kill_train__dp4_cp2__fake_rollout__moe_5layer
```

| Subcommand | Does | Available in |
| --- | --- | --- |
| `run` | full pipeline: prepare + every phase's baseline/target + compare | all scenarios |
| `baseline` / `target` | one side only, for debugging | comparison scenarios |
| `compare` | re-run the comparison on existing dumps (no GPU) | comparison scenarios |
| `generate-data` | record debug rollout data with real engines, no dumper | comparison scenarios |

- **Debugging**: prefer the individual subcommands over `run` — with a shared `--dump-dir` (plus `--phase` when multi-phase) you re-run only what changed.
- **`scenario_rollout_deterministic`**: the comparison subcommands, with the injection constants fixed in the module rather than exposed as options.
- **`scenario_random_crash`**: only `run`, with `--mode` / `--seed` / `--num-steps` / `--trainer-crash-interval-seconds` / `--rollout-crash-interval-seconds` / `--fully-async`.
- **`scenario_weight_update_all_gather`**, **`scenario_weight_update_p2p_local`**, **`scenario_weight_update_p2p_remote`**: only `run`, with `--mode` / `--num-steps` / `--failure-mode`; the hook and the action are fixed in the module, the fault is not. `--failure-mode` defaults to `sigkill`, and any other value moves the dump directory and the request id with it, so two entries of one scenario never share evidence.
- **`scenario_weight_update_all_gather_deadlock`**, **`scenario_weight_update_p2p_local_sigstop`**, **`scenario_weight_update_p2p_remote_sigstop`**: the same three runners with the fault pinned, so a hang is covered by an entry rather than by the soak happening to draw one.
- **`scenario_realistic_gsm8k`**: only `run`, with `--seed` / `--num-rollout` / `--trainer-crash-interval-seconds` / `--rollout-crash-interval-seconds` / `--metric-threshold` / `--fully-async`; no `--mode`.
- **`scenario_*_fully_async`**: only `run`, with the same options minus `--fully-async`, which they pin.
- **Dumps**: `resolve_dump_dir` in `conftest_ft/app.py` puts them under `$MILES_TEST_DUMPS_ROOT/<run_id>/<test_name>/`, falling back to `/node_public/dumps` when the cluster sets no root. A comparison scenario's `run` deletes them when it ends; the soak scenarios (`scenario_random_crash`, `scenario_realistic_gsm8k`) only clear a stale directory before starting, so a finished soak leaves its dumps behind for inspection. The run id is what stops two agents running the same test from deleting each other's dumps.

### Cluster Backend

- **Selection**: `command_utils.default_config()`, off `MILES_SCRIPT_CLUSTER_BACKEND` / `MILES_SCRIPT_NAMESPACE` / `MILES_SCRIPT_RUN_ID`, already set in the miles-workbench pod.
- **Scenarios stay backend-agnostic**: no mode declares one; the backend changes only the set of fault forms.
- **One config throughout**: the same `ExecuteTrainConfig` threads through `prepare()`, `run_training()` and `api_server_host()`; on kubernetes the api server lives on a pod named after its `run_id`, so a second config would aim the injector at a release that does not exist.
- **Side-specific releases**: a comparison may provide `config_for_side`; the pipeline applies it once before the target context and launch, which both receive that same transformed config.
- **Bounded side handoff**: after each kubernetes comparison side, including a failed one, the pipeline uninstalls its Helm release and waits at most five minutes for both the release and every release-labelled pod to disappear. The next side and the CPU comparison start only after that completes, so asynchronous chart cleanup cannot overlap their GPU reservations. Ray comparisons never call Helm.
- **Unreachable is a failure, not a skip**: `create_backend_for_run()` asserts before handing back a backend, since exiting 0 would report green for a test that never ran.
- **Namespaced probes only**: never a cluster-scoped CRD read, which the workbench's Role cannot do.

### Generate Debug Rollout Data

- **Who uses it**: modes with `has_real_rollout == False`, through `--load-debug-rollout-data --debug-train-only`.
- **Where it comes from**: `prepare()` in `conftest_ft/execution.py`, via `U.hf_download_dataset()` on `fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer`.
- **Soak reuse**: `materialize_cyclic_debug_rollout_data()` symlinks the recorded files cyclically under the shared data dir, so a soak can run more steps than were recorded and the run pod can read the links.
- **Regenerating it needs the 5-layer model**: the full model's `rollout_log_probs` are incompatible with the 5-layer training model and produce NaN GRPO gradients.

```bash
# 1. Generate with the 5-layer model and real sglang engines (no dumper)
PYTHONPATH=. python tests/e2e/ft/conftest_ft/scenario_trainer_no_failure.py generate-data \
    --mode kill_train__dp2_cp2__moe_5layer --num-steps 12 --output-dir /tmp/gen_rollout

# 2. Inspect
ls /tmp/gen_rollout/rollout_data/

# 3. Upload
hf upload --repo-type dataset fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer \
    /tmp/gen_rollout/rollout_data/
```

## Test Specifications

### Comparison Criterion

- **Dumps**: per-tensor predicates over `rel` / `max_abs` / `mean_abs`, as `compare_dumps(diff_thresholds=[(name_regex, predicate), ...])` onto the sglang comparator's `--diff-threshold`.
- **Fail-closed**: a tensor matching no regex fails, so every list ends with a `.*` catch-all and the specific families come first.
- **Model inputs**: `INPUT_TENSORS_ALLOW_FAILED_PATTERN` exempts `input_ids`, `positions`, `cu_seqlens_*`, `qkv_format`; `INPUT_TENSORS_SKIP_PATTERN` skips those plus `.*witness.*`. Nothing else is exempt.
- **Metrics**: `compare_metrics` reads `MetricEvent`s, requires `train/grad_norm` and `train/loss` in the baseline and equal event counts on both sides, and compares only the highest-attempt event per rollout id.

- **Why only some are bitwise**: baseline and target reduce over different topologies, so allreduce kernel ordering differs — unless `--deterministic-mode` and `--debug-deterministic-collective` are on.
- **Why `train/grad_norm` is exempt in `scenario_trainer_deterministic`**: it sums squared shard fragments, so its bracketing follows the dist-optimizer shard count (8 flat vs 2 per cell); a few fp32 ulps are inherent. The grads stay bitwise-checked through the dumps. It is exact in `scenario_rollout_deterministic`, where ft on rollout alone leaves one trainer topology and no shard-count bracketing to excuse.

### Targeted Fault Hooks

- **What they are**: named points in production code (`FaultHookName`) that a test can arm from outside, through `POST /api/v1/cells/<cell>/arm-fault-hook`; the first thread to reach an armed point consumes it and runs the fault there.
- **Who accepts one**: trainer cells only — the hooks live in trainer worker processes, and the only thing asked to arm one is the trainer controller that owns the named cell, so anything else is not a fault hook source at all rather than a requester left waiting for a fault nothing reaches.
- **Every arm names the incarnation it chose**: `expected_workers_hash` is required and non-empty, and it is the hash the caller saw when it picked the source. The owning trainer controller matches it against the generation it currently holds and freezes that worker's already-held handle before its first `await`, so a replacement created afterwards can only be reached through a handle this request never took. It refuses — 400, with the reason — when the cell is gone, runs another hash, has not finished `init`, or has no such `sub_index`; it never re-aims at the replacement and never retries. No handle is built for the arm and none re-handshakes: under kubernetes the frozen handle was pinned to a boot uuid when the cell ran `init`, so a process that merely reuses the endpoint is refused by the rpc boot guard before the call is submitted. The call goes out on the worker's `fault_injector` concurrency group and is bounded, so arming waits for neither the training step nor the weight update.
- **One-shot and exclusive**: a second arm at an already armed point is refused instead of replacing the first, and the slot frees itself when the fault fires.
- **A delay is part of the request**: `delay_ms` is a strict integer in `0..60000`, defaulting to `0`. At `0` the fault runs in the thread that reached the point. Above `0` the point freezes the action, the target the site named, the executor that can reach it and the weight-update span it was reached in, starts a daemon timer and returns, so the update goes on rather than sleeping inside the hook. The frozen context is what the timer fires with, so a delay can neither re-aim the fault at whatever this rank writes to later nor attribute it to the update that opened while it waited.
- **A reached request keeps holding its point**: the hook counts as pending from the moment it is reached until its fault has finished running, whether it waited or not, so a second arm is refused for that whole window and at most one fault per `FaultHookName` can be outstanding. This matters without any delay too: a `remote_inference_cell` fault travels the network from the writer thread, and a request armed while that call is in flight could otherwise be consumed by another cell's writer. A local kill never returns, and the pending slot dying with the process is the honest end of that request.
- **Scheduling is not firing**: an arm is logged as `arm`, a delayed request as `schedule`, and only the execution writes the `FaultHookFireEvent`, whose timestamp is therefore the moment the fault really ran and whose `delay_ms` says what it waited. A request that was scheduled and never ran leaves no fire, and every witness fails on it.
- **A delay that cannot be delivered is visible**: a timer that fails to start, or a site that named no target, clears the pending slot and records `not_scheduled`; a remote executor that raises records `errored` with its stacktrace, and a receiver that refuses still records `refused`. All three are outcomes a witness rejects, so a fault nobody delivered can never pass for one that landed.
| Hook | Site | Runs on | Remote-capable |
| --- | --- | --- | --- |
| `weight_update.before_all_gather` | `all_gather_params_async`, inside the `tp_size > 1` branch, immediately before the real `dist.all_gather(..., async_op=True)` | the trainer rank | no |
| `weight_update.before_p2p_write` | `_P2PInferenceCellUpdater._write_one_peer`, immediately before `batch_transfer_sync_write` | that cell's writer thread, with the peer and the span frozen at submit | yes |
| `weight_update.after_p2p_submit` | `_P2PInferenceCellUpdater.submit_write`, once the future exists and is pending | the submitting rank | yes |
| `weight_update.after_base_weights` | `UpdateWeightP2P.after_base_weights`, once every pending write has been collected | the submitting rank | no |

- **Two actions, chosen at arming time**: `local` crashes the trainer worker that reached the point; `remote_inference_cell` crashes the engine that point is writing to. Nothing else is offered — the registry takes no callback and no cell name.
- **A remote request never names its victim**: the site does, from the connection it actually established. The target is built from the `RemoteWeightInfo` of the peer being written to — its cell id, the `workers_hash` that cell was reached at and the receiver's own boot uuid, session and rank — so no request can name a cell or a rank this write never reached. The receiver's rank is not a `worker_in_cell_index`, and the event records the two separately rather than assuming the leader worker.
- **The before-write hook reads the span frozen at submit**: the writer thread can reach the transfer after the update that queued it has ended and the next one has opened its own span, so the version is captured when the write is submitted and travels with the task.
- **Refused before it is accepted**: arming `remote_inference_cell` at a hook with no single target, or in a process that cannot reach a cell at all, fails the arm rather than being dropped later.
- **Bound to the incarnation, not to the name**: the fault carries the expected `workers_hash` all the way into the operation that delivers it. On ray the worker manager compares it and submits to the captured actor inside one hold of its membership lock, so no replacement can appear between the check and the submit. Comparing the hash first and then calling an unconditional inject would leave exactly that window.
- **A refused fault is not harm**: an incarnation that is already gone yields `stale_target`, which is recorded on the fire and fails the scenario's witness. Nothing is retried blindly.
- **The receiver closes the window, not the control plane**: the engine process that holds the transfer-engine session mints a `receiver_boot_uuid` per session and serves an `/inject_fault` endpoint that only acts on a request naming that uuid, that session and that rank. Both backends check the cell incarnation first — ray inside one hold of the worker manager's membership lock, kubernetes against the observed pods — and then send the same bounded HTTP request to the control url frozen at connect time. Neither builds a worker rpc handle for it, because a handle built now pins whichever process answers.
- **The identity rides with the weights**: `receiver_identity` comes back in the same metadata response as the session id and the weight buffers, so a reader cannot pair fresh weights with a stale identity. Miles verifies the identity's session and rank against the peer it is about to write to and fails that target otherwise; an engine running without the fault-control flag publishes no identity, transfers normally, and refuses a remote fault at the site rather than falling back to killing by name.
- **The request and the answer are matched field by field**: the post carries `request_id`, the expected uuid, session and rank, and a mode the receiver actually implements. A 200 must answer `accepted` for the same request id, uuid, session and rank; a 409 naming an identity mismatch or an inactive receiver is the incarnation being gone, and its body may legitimately report the replacement's identity, so only the request id is matched there. A pending-action conflict, a payload conflict, any other status and a body that answers for another request are explicit failures, and a timeout or a disconnect is `unknown` — never a delivery and never a blind retry.
- **Accepting is not firing**: the receiver signals itself after it answers, so the fire event records `accepted` for a remote fault and `fired` only for a local one. The scenario still has to see the victim lose the incarnation the write reached and come back under a replacement.
- **`weight_update.after_base_weights` has no entry of its own**: it is a local-only point strictly later in the same update than `before_p2p_write`, so an entry there would crash a sender in the same shape `scenario_weight_update_p2p_local` already proves. It is drawn by `scenario_random_crash`, whose hook forms sample the whole reachable set rather than pinning one point.
- **A fire names the update it happened in**: `WeightUpdater.update_weights` opens a process-wide weight-update span, the fire event carries its version, and the trainer controller writes one `WeightUpdateAssignmentEvent` per sender before calling it. That pair is what lets a scenario say which engines the harmed sender owned, and it is what the p2p scenarios join their fire to as well.

### Fault Forms and Receivers

| Backend | Cell type | Forms, drawn from uniformly |
| --- | --- | --- |
| ray | actor | `inject_fault:sigkill`, `inject_fault:exit`, `inject_fault:segfault`, `inject_fault:deadlock`, `inject_fault:sigstop` |
| ray | rollout | `inject_fault:sigkill` |
| kubernetes | actor | those five faults, plus `delete_pod` |
| kubernetes | rollout | `exec_sigkill`, `delete_pod` |
| either | actor | plus `fault_hook:local`, when the soak can reach the hooks |
| either | rollout | plus `fault_hook:remote_inference_cell`, when the soak can reach the hooks |

- **A form can be unavailable rather than absent**: `BaseFaultForm.is_available` is asked about the cell that was drawn, and a kind whose every form says no defers that injection instead of forcing one. Only the hook forms ever say no, and only until the run has an assignment and a recovery source to arm against.
- **Where the soak finds its sources**: `GET /api/v1/fault-hook-sources` answers with the trainer cells of the run whether or not `--ft-components` names `train`, while `GET /api/v1/cells` keeps listing only the cell types this deployment heals. A rollout-only soak can therefore draw `fault_hook:remote_inference_cell` against a live trainer generation, and the mini FT controller still sees no trainer to suspend or resume.
- **The soak draws the delay with everything else**: the hook forms draw their source, hook, failure mode and `delay_ms` from the same seeded `random.Random` the rest of the injector uses, in that order, over `0..1000` ms through `draw_fault_hook_delay_ms`, so a failed run replays from its seed. The value goes into the arm request and into the arm record, and the fire collector rejects a fire whose `delay_ms` is not the one that was armed. The registry in the worker draws nothing itself.
- **A form can keep its own books**: `records_own_attempt` says the form writes what it did itself, so the loop does not record an `InjectionEvent` for it. Arming a hook is not harming a cell, and the one record the loop would write would claim it was.
- **Each `FailureMode` is its own form**: pod deletion is a quarter of a kubernetes trainer injection, not half of it.
- **The actor class decides what a kill means**, since an injection carries only a mode and a `sub_index`: `TrainRayActor` and `ServeActor` crash their own process, the only thing that costs torchft a member, while `CommandActor` SIGKILLs the isolated process group rooted at the engine subprocess. That includes the launch shell and every engine child it spawned, so a dead cell cannot leave an orphaned scheduler holding GPU memory while its replacement starts; the Ray actor observes the subprocess exit and reports the death as production sees it.
- **Why an engine takes sigkill alone through its supervisor**: exiting, segfaulting and deadlocking are what a process does to itself from the inside, and no signal reproduces them from outside — SIGTERM is a clean shutdown, SIGSEGV is delivered rather than provoked. Stopping the supervisor's own subprocess group would freeze the launch shell, not the engine rank holding a weight transfer, so `CommandActor` refuses everything but sigkill; a frozen engine is reached instead by the remote hook form, which signals the receiver process itself.
- **Why a trainer also takes deadlock and sigstop**: a rank wedged in a native collective and a rank stopped by a signal both stop answering without dying, which is the failure the update deadline and the control plane's confirm-dead path exist to end. Both are inflicted by the trainer worker on itself, so no outside process has to fake them.
- **What ends a wall-clock hang, which lands at no particular phase**: the heartbeat rpc runs on its own concurrency group, but a `PyDLL` sleep holds the GIL and a stopped process runs nothing, so neither answers. After the trainer heartbeat checker's grace and `failure_threshold` consecutive misses the cell reads `Healthy=False`, and the mini FT controller suspends it — on ray by stopping its actors, on kubernetes by deleting its pods. That chain needs nothing from the hung process either, and is separate from the weight-update deadline the targeted entries rely on.
- **How a kubernetes engine takes a kill**: its pod runs sglang as the entrypoint (`CommandWorkerSpec`), so no actor and no rpc server exist to receive `inject_fault`. The kill is delivered from outside instead, as a `kubectl exec` SIGKILL of the sglang processes in the engine container, and deleting the pod is the second, coarser form — the engine *is* the pod.
- **Deletion is the test layer's own `kubectl delete pod`**, timeout-bounded and selecting on release, pool and cell index. It models an outsider, and deliberately avoids the production heal path `KubernetesCellOperations.suspend`, whose bugs an injector sharing it would hide.

### `scenario_trainer_no_failure`

```
Type: comparison (baseline=normal DP, target=indep_dp)
Steps: 2 (NUM_STEPS)
Compare: dumps rel <= 0.0085; metrics rtol=1e-2, atol=1e-8

1. Baseline: normal DP on debug rollout data (real engines in a real-rollout mode)
2. Target: the same arguments plus get_ft_args(mode), which is --use-fault-tolerance
   --ft-components <the mode's ft_components> --api-server-port 0
3. Compare:
   - Tensor-level: compare_dumps (weights, grads via dumper & sglang comparator)
   - Metric-level: compare_metrics (MetricEvent, requires train/grad_norm and train/loss)
   - Rank matching: grouping_skip_keys=["rank", "dp", "edp"], the two sides differing in
     world size and DP layout

Roughly equal, not bitwise - allreduce kernel ordering differs across topologies.
```

### `scenario_trainer_with_failure`

```
Type: comparison, multi-phase (phase_a + phase_b)
Steps: phase_a 1 rollout (id 0), phase_b 3 rollouts (ids 1..3)
  --num-rollout 4: exclusive global end id, not a per-run count
Compare: phase_b dumps per rollout, rel <= 0.0085 plus the max_abs floors below;
         metrics rtol=5e-2, atol=1e-7

Phase A (both sides):
  1. Run 1 rollout
  2. Save checkpoint (--save-interval 1), exit

Phase B - baseline:
  1. Resume from the phase_a checkpoint
  2. Run 3 normal rollouts (1..3)

Phase B - target:
  1. Resume from the phase_a checkpoint
  2. Rollout 1: N cells normal
  3. Rollout 2, attempt 0: crash_before_allreduce on last cell rank 0
     -> os._exit(1) -> allreduce timeout -> should_commit=false -> retry
  4. Rollout 2, attempt 1: reconfigure to N-1 cells, commit on the degraded quorum
  5. After rollout 2: stop_cell_at_end(last) + start_cell_at_end(last)
  6. Rollout 3: heal back to N cells, train with the healed cell

Fault injection: --ci-ft-test-actions, JSON list of {at_rollout, action, cell_id, rank, attempt}
  at_rollout: rollout id; attempt: retry attempt, actor-level actions only
  stop_cell_at_end / start_cell_at_end: trainer controller, suspend/resume via cell_operations
  crash_before_allreduce: inside the targeted actor

Healing witness: target phase_b event dir, exactly two CellReconfigureEvents
  rollout 2: shrink, alive N -> N-1
  rollout 3: heal, healed = last cell, ckpt src = cell 0, alive back to N
  baseline and phase_a dirs: zero
Dump-leaf witness: {fwd_bwd/rollout_<id> leaf dirs} == {rollouts the comparison loop walks}
```

- **Why the healing witness**: without it the comparison degenerates into two fault-free runs that trivially agree; the shrink proves the injection fired.
- **Why the dump-leaf witness**: a newly added leaf dir would otherwise skip comparison unnoticed.

Grad families with a `max_abs` floor (cancellation-dominated near-zero grads; real grads sit around `1e-2`):

| Rollouts | Families | Floor |
| --- | --- | --- |
| all | MoE expert grads, QK-norm (`q_layernorm` / `k_layernorm`) grads | `max_abs <= 1e-3` |
| injected ones, real-rollout mode only | QK-norms, folded `layer_norm_weight`s, `linear_qkv` / `linear_proj` / `mlp.linear_fc[12]` weights | `max_abs <= 3e-3` |

- **Where `3e-3` comes from**: the degraded commit's ulp drift lands as <= 2.8e-3 absolute noise in those near-zero grads (40 tensors, 2026-06-12), against real grads around `1e-2`. Embedding, output, final-norm grads, every activation and every pre-fault rollout keep the strict set.

#### `kill_train__dp2_cp2` mode

`scenario_trainer_with_failure` against live generation: real sglang engines, deterministic inference, temperature 0.8.

- **Pre-fault rollouts need bitwise weights on both sides**: the fault rollout trains the target's own live samples, and one bf16 ulp in a weight flips temperature-0.8 samples so the fault rollout trains different data (observed as a 6% `train/grad_norm` gap once the sglang v0.5.18 bump changed the sampled content). Both sides therefore run `--debug-deterministic-collective` (the same fixed fold for the normal-DP 4-rank reduce and the indep_dp CP-then-cross-cell reduce, as in `scenario_trainer_deterministic`) and `--clip-grad 10.0` (clipping inactive: the dense grad norm is ~1.3, and `train/grad_norm` differs by a few fp32 ulps across shardings, which an active clip would multiply into every update). The other FT modes have grad norms below 1.0, so clipping is inactive there without the override.
- **Post-fault rollouts are injected**: `--ci-inject-rollout-data-path` replays the baseline's `--save-debug-rollout-data` recording from rollout 3 on (crash rollout + 1).
- **Why inject**: the degraded-quorum commit brackets microbatch accumulation differently, and under live sampling that ulp diff flips tokens until the two runs' rollout data diverges wholesale. It is fault-inherent -- no collective ordering removes it. Injecting makes training inputs identical by construction, keeping the comparison strict.
- **The target stays real**: engines and generation still run (samples discarded), `update_weights` fires after the degraded commit and after healing, the health monitor pauses and resumes — the whole crash → retry → heal → weight-sync path. Engine checksums are not compared here; only `scenario_trainer_deterministic` does that.
- **Generation is still asserted**: `RolloutDataInjectionUtil.assert_matches_generated` requires bitwise-identical prompt tokens per sample, plus a mean response-token match ratio above `--ci-inject-rollout-data-min-match-ratio`, set to 0.5 here (the flag's own default is 0.9). A broken `update_weights` drops that ratio by ~2 orders.
- **Not asserted**: exact post-fault sampled content beyond the ratio; pre-fault rollouts are compared for real.

Guard calibration (2026-06-12, first post-fault rollout, 256 samples, correct weights; a response counts as mismatched from its first flipped token on):

| Model | Mean response-token match | Min |
| --- | --- | --- |
| dense Qwen3-0.6B | **0.63** | 0.035 |
| 5-layer MoE | **0.19** | 0.005 |

- **Why dense**: on the truncated MoE, uncalibrated logits plus router near-ties amplify the drift to 0.19, indistinguishable from unrelated content; dense's 0.63 sits 2 orders above that, so 0.5 separates them.

### `scenario_trainer_deterministic`

```
Type: comparison, multi-phase (phase_a + phase_b)
Steps: 3 rollouts per phase - phase_a 0..2, phase_b 3..5
  --num-rollout 6: exclusive global end id
  --debug-exit-after-rollout 3: counts within the run, fires after that rollout's ckpt save
  --save-interval 3 (NUM_ROLLOUTS_PER_PHASE): one ckpt at each phase's last rollout
Compare: BOTH phases' dumps rel <= 0 (bitwise); metrics rtol=0 / atol=0, except
         train/grad_norm at rtol=1e-6

One shared builder parameterized by the phase's start rollout id P; only the start regime differs:
  phase_a: cold start (no --load, so no_load_optim/no_load_rng/finetune) - rollouts 0..2 (P=0)
  phase_b: resumes from phase_a's post-healing rollout-2 ckpt (start_rollout_id = loaded + 1
           = 3) - rollouts 3..5 (P=3)

Per-phase baseline: rollouts P..P+2 all normal, no stop/start, no healing

Per-phase target:
  1. Rollout P, P+1: all N cells normal
  2. After rollout P+1: stop_cell_at_end(last) + start_cell_at_end(last)
  3. Rollout P+2: heal at the start (recv_ckpt from cell 0), then normal execution

Determinism: --deterministic-mode, plus NCCL_ALGO=Ring, NVTE_ALLOW_NONDETERMINISTIC_ALGO=0,
  CUBLAS_WORKSPACE_CONFIG=:4096:8, SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=8192
  --debug-deterministic-collective: fixed-tree SUM folds, making normal DP's and indep_dp's
    reduction topologies bitwise-comparable

Cross-cell check: --use-fault-tolerance --ft-components train auto-enables
  --save-local-weight-checksum, --save-inference-engine-weight-checksum and --enable-event-analyzer
  cross_replica_weight_checksum: cell-to-cell bitwise equality, every rollout attempt,
    post-healing included
Engine checksum (real-rollout modes only): one InferenceEngineWeightChecksumEvent per
  update_weights, carrying every engine's checksum
  _compare, per phase: baseline and target pushed identical weights per (rollout, engine)
  inference_engine_weight_checksum_consistency: all engines of one rollout agree

Healing witness: one heal per target phase, at P+2 (healed = last cell, ckpt src = cell 0,
  alive back to N); no standalone shrink - one _refresh_cells absorbs the stop+start pair
  the event dir is snapshotted into the ckpt and restored on --load, hence:
    target phase_a: heal at rollout 2
    target phase_b: heal at rollout 2 (restored with the ckpt) + heal at rollout 5
    both baselines: zero reconfigure events
```

- **Why P+2 must exist**: healing runs at its start, so a shorter phase never executes the path under test.
- **Why zero tolerance**: a state-copy bug in healing is easy to make and an approximate check would miss it.
- **What phase_b adds**: reproducing the baseline bit-for-bit also proves the ckpt round-trips bitwise.
- **Why the healing witness**: it gates the off-by-one bug where healing never runs and the comparison passes on two fault-free runs.

### `scenario_rollout_deterministic`

```
Type: comparison; both sides run the identical command, only the target is wrapped in the
      fault injector, through the pipeline's target_side_context hook
Entry: test_rollout_deterministic__kill_rollout__dp4.py, ft-long
Steps: 8 rollouts (NUM_ROLLOUTS)
Requires: mode.has_real_rollout, ft_components == ("rollout",) exactly, and not mode.colocate
Compare: dumps rel <= 0 (bitwise); metrics rtol=0 / atol=0 over train/* and rollout/*,
         train/grad_norm included

Regime (both sides):
  - the shared deterministic rollout recipe: --sglang-enable-deterministic-inference,
    --sglang-attention-backend flashinfer and --deterministic-mode
  - --debug-deterministic-collective and scenario_trainer_deterministic's deterministic env vars
  - --sglang-disable-radix-cache
  - --rollout-health-check-interval 1
  - the p2p weight transfer recipe, so the update a crash interrupts is the p2p one
  - --update-weights-interval 1: every one of the 8 rollouts publishes weights

Injection (target side only):
  1. Rollout cells, seed 42, exponential mean CRASH_INTERVAL_SECONDS (30s)
  2. Forms drawn per (cluster backend, cell type), as in the soaks
  3. Stop accepting faults after six completed rollouts, leaving the final two rollouts for recovery
  4. Stop the injector, waiting out a mid-flight injection for at most
     STOP_AND_JOIN_TIMEOUT_SECONDS (180s), then re-use the soak's rollout witnesses: >= 2
     accepted rollout injections, each paired with one completed recovery cycle

Assertions:
  1. Reconfigure events: zero on BOTH sides - crashing an engine must not reconfigure trainer cells
  2. Metrics: rtol=atol=0 over train/* and rollout/*
  3. Dumps: rel <= 0
  4. Engine checksums: baseline and target pushed identical weights per (rollout, engine)
  5. Weights moved, per side: the engine weight checksum is not identical across all rollouts
  6. Engines rejoined, per side: no update reached more than the mode's 4 engines, and the last
     update of every policy reached all 4
```

- **Why it exists**: an engine dying and being replaced mid-generation is supposed to be invisible to training, and "invisible" is a claim about bits; the rollout soak only ever asserted survival.
- **What "rollout + weight update ft" means here**: the only fault tolerance under test is the engine side of a weight update. Trainer ft stays off, so one trainer topology reduces on both sides and every tensor can be compared at `rel <= 0`; a healed trainer cell would rebracket the reduce and force the tolerances this scenario refuses to have. Both sides therefore run the same trainer layout, the same command, and differ only in that the target is wrapped in the injector.
- **Why `--update-weights-interval 1` is pinned rather than inherited**: at a wider interval most rollouts publish nothing, so the injector's crashes land in rollouts where no update is running, the per-rollout engine checksum coverage thins out, and the audit that a wider interval is expected to disable would take the scenario's own witnesses with it. Pinning it makes the claim a property of the test rather than of a production default.
- **Why the rejoin witness (assertion 6)**: publishing to the engines still alive is exactly what partial-target weight update is for, so an update that reached three of four engines is correct in the middle of a run and a bug at the end of one. The witness allows the first and rejects the second, which the bitwise comparison cannot: an engine that never rejoined the fan-out serves stale weights that no assertion here reads. The upper bound catches the other half — a dead engine left in the fan-out beside its replacement.
- **Why not `assert_engine_count`**: the deploy suite's version demands every update cover every engine, which under injection fails on the very partial-target behaviour this scenario exists to permit.
- **Deliberately uncovered**: landing a fault at a chosen point inside a weight update. The injector fires on a wall clock, so which phase of an update a crash interrupts is luck; the typed fault hooks are what make that deterministic.
- **Why the shared deterministic recipe**: the assertion is deterministic replay across fresh inference engines, not true-on-policy training. Reusing the same FlashInfer recipe as the main deterministic trainer-FT test avoids a second, incompatible attention-backend contract.
- **Why `--sglang-disable-radix-cache`**: a replacement engine serves with a cold prefix cache where the baseline's was warm, and deterministic inference is nowhere documented as prefix-cache-length invariant.
- **Why this recipe disables batch-variant MM fallback**: a rollout worker loss changes co-batching while the pool is healing; permitting an `einsum` fallback would make the same seeded request depend on that temporary batch shape. The scenario injects the environment override without changing the production default.
- **Why `--rollout-health-check-interval 1`**: healthy generation can finish between two five-second polls; the short scenario needs at least one fresh Serving observation before it may pick a target.
- **Why this scenario polls the fault window every 0.2 seconds**: an eight-rollout run offers few chances to inject, and the generic two-second scheduler cadence spends them waiting. The interval predates disaggregation, where engines serve continuously; it is now a latency bound on noticing a fresh Serving reading rather than the only way to catch one.
- **Why the final two rollouts accept no new fault**: the scheduler keeps observing recovery but closes admission after rollout 5, so teardown cannot race a newly accepted replacement.
- **Why every namespace, not just `train/`**: an engine crash shows up first in `rollout/raw_reward` or `rollout/log_probs`. `perf/` is left out by name, being wall-clock and throughput that a relaunch moves by definition, and a metric in neither namespace fails the run rather than being dropped quietly.
- **Why the weights-moved gate**: bitwise equality is also satisfied by two runs that trained on nothing.
- **Why not a loss or reward curve**: neither is a progress signal here — the reward is `deterministic_random`, a hash of the response, and GRPO's surrogate loss is not monotone even while a run learns. Over eight rollouts neither moves for a reason worth asserting, and the weights either changed or they did not.

### `scenario_weight_update_all_gather`

```
Type: targeted (no baseline, no compare); passes if the armed hook fires and the run recovers from it
Entry: test_weight_update_all_gather__kill_train_rollout__dp2_tp2.py, ft-short
Steps: 8 (DEFAULT_NUM_STEPS)
Requires: real disaggregated engines, TP2, ft_components == ("train", "rollout"), >= 2 trainer cells,
          >= 2 engines
Regime: --update-weight-transfer-mode p2p
        --sglang-remote-instance-weight-loader-start-seed-via-transfer-engine
        --sglang-enable-p2p-fault-injection
        --update-weights-timeout 120 --update-weight-engine-request-timeout 30
        --p2p-transfer-timeout 10
        --save <dump>/ckpt --save-interval 1, --mini-ft-controller-enable

1. A background thread polls /api/v1/cells every 2s and records every snapshot
2. The arm waits for a completed training step, a checkpoint tracker naming an iteration >= 1, a
   non-empty publication of a rollout other than the startup sync, and the armed cell observed
   healthy. It then snapshots every cell's workers_hash and arms
   WEIGHT_UPDATE_BEFORE_ALL_GATHER with sigkill in worker 0 of the last trainer cell, once
3. The next weight update reaches that point in hf_weight_iterator_direct.all_gather_params_async
   and the worker dies before the collective it was about to enter

Witnesses:
  fire       -> exactly one FaultHookFireEvent with the armed request id, whose whole
                TrainProcessIdentity (component, model id, cell index, rank) is the armed worker's,
                carrying the weight version of the update it fired in
  assignment -> exactly one WeightUpdateAssignmentEvent for that version and that trainer cell,
                whose trainer_workers_hash is the incarnation that was armed
  eviction   -> the armed cell observed under another workers_hash after the fire, healthy
                again under it, and never the armed hash alive at the end
  hang       -> that first observation of another workers_hash falls between the fire's own
                timestamp and 600s after it, the time a deadlocked worker would take to return
                by itself
  healing    -> a CellReconfigureEvent after that assignment dropping the cell index, then a later
                one healing it back
  isolation  -> every engine the assignment names left the incarnation it was written to, and each
                was observed healthy and Serving under a replacement
  blast      -> >= 1 engine outside the assignment kept the incarnation it had when the hook was
                armed and was observed Serving after the harm
  progress   -> >= 1 publication carrying a non-empty checksum after that assignment
```

- **Why a named point and not a wall clock**: an update is a small share of a step, so a random injector rarely lands inside one, and never lands at a chosen line of it.
- **Why the hook sits before the collective, not after the bucket loop**: a hook placed once the gathers are done crashes a rank that already survived the thing under test.
- **Why the arm waits for a completed step, a checkpoint and a real publication**: killing a worker before those exist takes out the only source a replacement could be healed from, and the run would fail for a reason the scenario is not about. A tracker file that exists but names no iteration, and a checksum event carrying only empty dicts, are exactly the states that look ready and are not.
- **Why an arm is not a fault**: the api server answering 200 only proves the request was accepted, so the run fails unless the fire event says production reached the point.
- **Why the ack time is never used as the fault time**: the armed worker dies at the hook, so its rpc reply can be lost and the fire can be recorded before the arm is acknowledged. Every ordering the witnesses need comes from the fire event, from the assignment, or from the observations themselves; the pre-arm snapshot supplies identities, never times.
- **Why the fire is an event and not a log line**: the fault kills the process, arms and fires interleave across workers, and only a request id pairs them; `EventLogger` writes and closes per event, so the record survives the sigkill that follows it.
- **Why the whole process identity is compared**: a run can hold two roles, several trained models and many cells, and every one of them numbers its cells and ranks from zero.
- **Why an assignment event exists at all**: which engines a sender owned in one update is the controller's decision, and nothing in a cell name reproduces it. It is written after the assignment is computed and before any sender is called, so a fault fired inside the call always has it to be read against.
- **Why the fire carries a weight version**: it is the join to that assignment. The trainer worker opens a process-wide span for the duration of `WeightUpdater.update_weights` — a plain module global, because the p2p write hooks run on per-cell writer threads no contextvar would reach — and clears it in a `finally`. Updates are serial in a worker, so a second span is refused rather than nested.
- **Why the armed incarnation is compared against the assignment**: it rules out the case the observations alone cannot, where the cell was already replaced before the fault fired and a healthy-looking second hash is really a third one.
- **Why the healing witness is anchored to that assignment**: a run has other reconfigures, and picking any eviction and any healing from the whole log would let an earlier unrelated crash pay for this fault.
- **Why both ft components**: a trainer that dies mid-update takes its assigned engines down with it, so the run only recovers if engines are recoverable too.
- **What the progress witness cannot yet say**: `InferenceEngineWeightChecksumEvent` names no cell, so "an unrelated engine published after the fault" is asserted as that engine still Serving its original incarnation plus a non-empty publication of the run. Op33 gives the event a cell and a version, and this witness tightens to that engine's own publication.
- **Why the eviction is measured from the fire and not from the arm**: a hang leaves the process alive, so the only thing separating "the control plane took it out" from "it came back on its own" is when the replacement appeared relative to the fault. The arm can precede the fire by many updates, and a replacement that appeared for its own reasons before the fault must not pay for it, so the witness reads only observations at or after the fire's own timestamp.
- **Why 600s is the bound**: `FailureMode.DEADLOCK` sleeps in libc through `ctypes.PyDLL`, holding the GIL for `DEADLOCK_SLEEP_SECONDS`, and then returns. An eviction after that is indistinguishable from the worker waking up, so the witness reads the same constant the fault uses. `SIGSTOP` never returns at all, and is held to the same bound because nothing but the control plane can end it.
- **Why the deadlines are shortened for these runs**: at the production defaults (`--update-weights-timeout 3600`, `--update-weight-engine-request-timeout 600`) the controller would still be waiting when a deadlocked sender woke up, so the run could not tell the two apart. 120s / 30s / 10s cover a Qwen3-0.6B transfer to the assigned engines with room to evict, and only the tests set them.
- **Why the trainer heartbeat grace is left alone**: `--trainer-heartbeat-checker-first-wait` defaults to 300s and is re-armed on every resume, which is what keeps a legitimately initializing replacement from being read as a hang. The hang witnesses lean on the weight-update deadline, which is scoped to one update and cannot be confused with startup.
- **How a hung worker is actually removed**: `mark_errored_and_kill` asks each worker to kill itself, and a stopped or deadlocked process answers neither that rpc nor the death probe — `_probe_is_dead` counts a timeout or any error as still running. After `CONFIRM_DEAD_TIMEOUT_S` the cell falls back to `CellOperations.terminate_incarnation`, which on ray stops the captured actors of that `workers_hash` and on kubernetes deletes their pods under a uid and resourceVersion precondition and waits until they are gone. Nothing in that path needs the frozen process to cooperate. Inference cells take the same external route through `_terminate_errored_cell`.
- **Why a new generation cannot appear beside the old one**: on ray a stopped cell keeps its actors in a retiring set until every one of them is confirmed dead, and `start_cells` refuses a cell that still has any — a kill that raised, a probe that timed out, a cancelled stop and one surviving rank all keep the cell out of service instead of freeing its name. The refusal is bounded, so the heal loop retries and other cells keep starting. On kubernetes the conditional delete waits for the pods of that uid to disappear before it reports success. This is what lets the hang witnesses read a new `workers_hash` as evidence that the old incarnation is gone rather than merely replaced in the bookkeeping.

### `scenario_weight_update_p2p_local` and `scenario_weight_update_p2p_remote`

```
Type: targeted; same runner, mode, regime and readiness gate as scenario_weight_update_all_gather
Entries: test_weight_update_p2p_local__kill_train_rollout__dp2_tp2.py,
         test_weight_update_p2p_remote__kill_train_rollout__dp2_tp2.py, both ft-short
Steps: 8 (DEFAULT_NUM_STEPS)

p2p_local:  arms weight_update.before_p2p_write with sigkill, target local
  Witnesses: exactly the all-gather scenario's, at a different point -- fire and assignment join,
             the armed trainer incarnation evicted and healed, every engine that assignment names
             replaced and serving again, an engine outside it still serving its original
             incarnation, and a non-empty publication after that assignment

p2p_remote: arms weight_update.after_p2p_submit with sigkill, target remote_inference_cell
  Witnesses: fire (one, in the armed worker, target remote, outcome fired rather than stale),
             assignment join, the fire carries the receiver uuid, session and rank of the peer it
             wrote to, the victim named by the fire is one of that assignment's engines at
             the incarnation it held when the hook was armed, that incarnation gone within 600s
             of the fire and by the end, the victim Serving again under a replacement, an engine
             outside the assignment still serving its original incarnation, and a non-empty
             publication after that assignment
```

### The three hang entries

```
Type: targeted; the same three runners with --failure-mode pinned, so a hang is covered by an
      entry rather than by the soak drawing one
Entries: test_weight_update_all_gather_deadlock__kill_train_rollout__dp2_tp2.py,
         test_weight_update_p2p_local_sigstop__kill_train_rollout__dp2_tp2.py,
         test_weight_update_p2p_remote_sigstop__kill_train_rollout__dp2_tp2.py, all ft-short

all_gather_deadlock: a sender that stops answering while holding the GIL, at the collective
p2p_local_sigstop:   a sender frozen by a signal, at the write it was about to issue
p2p_remote_sigstop:  the receiver of a write in flight frozen by a signal, in the engine process
                     that holds the mooncake session

Witnesses: exactly the sigkill entry's, plus the hang bound on when the incarnation went away
```

- **Why three entries and not a product**: each answers a different boundary — a hang inside the trainer's own collective, a hang in the sender's write path, and a hang in the receiver. Every other combination repeats one of those three shapes at 8 GPU-hours each, and the soak draws them anyway.
- **Why the sigkill entries stay**: dying and hanging leave the control plane different work to do, and the entries that already cover the first are not replaced by the ones covering the second.
- **Why a delegating scenario module per entry**: the entry naming scheme reads the scenario out of the file name, so a second entry of one scenario needs a name of its own. Each delegate is a `run_ci` that pins `failure_mode` and nothing else; `compute_targeted_test_name` derives the same name the delegate declares, and a fast test fails if the two ever drift.
- **Why a remote hang cannot be a deadlock or an exit**: those are things a process does to itself from inside its own code, and the receiver's fault endpoint only raises signals. Arming a remote request for one of them is refused where it is armed, not silently downgraded, so the soak never draws a combination the receiver would reject.

- **Why these two of the four hooks**: one local and one remote case, each at the moment its kind is hardest — the sender dies with a transfer half-issued, and the target dies while a write to it is in flight. A cross product of hooks and actions would buy repeats of the same two shapes at 8 GPU-hours each.
- **Why the remote case does not assert healing of the victim's cell index**: engines are not trainer cells and produce no `CellReconfigureEvent`; the incarnation the api server reports is the evidence, exactly as in the rollout soak.
- **Why the victim is checked against the assignment and the arm snapshot**: the fire names a cell and a hash from inside the write, and both have to be the cell this update gave that sender and the incarnation the run watched serving. Otherwise a fire could name anything and a later unrelated replacement would look like harm.
- **Why a stale outcome fails rather than retries**: the fault was refused because the incarnation was already gone, so nothing was harmed. This is the one shape that must never be counted as an injection, and an unknown answer is not one either.
- **Why neither backend is ruled out**: the fault is delivered by the receiver process itself, which exists on both, so the scenario is no longer restricted to ray.
- **Why an unrelated engine has to keep serving**: a fault aimed at one target that stopped every engine would satisfy a witness that only asked whether the victim died.

### `scenario_random_crash`

```
Type: soak (no baseline, no compare); passes if training completes without hanging and the
      witnesses hold
Steps: 60 (default)
CLI: --mode, --seed (42), --num-steps (60), --trainer-crash-interval-seconds (120),
     --rollout-crash-interval-seconds (240), --fully-async (off)

Targeting and assertions follow the mode's ft_components:
  ("train",)          -> inject into "actor" cells, assert trainer healing
  ("rollout",)        -> inject into "rollout" cells, assert the recovery cycle
  ("train","rollout") -> inject into both kinds, assert both
  A mode declaring rollout ft without real engines would schedule injections into a cell kind
    that does not exist, so FTTestMode refuses to be constructed at all

Architecture (external fault injection, not inside the training loop):
  1. Start indep_dp training + api server (port 18080) + --mini-ft-controller-enable
  2. A background daemon thread iterates every 2s:
     a. GET /api/v1/cells, keeping only the targeted cell types
     b. Append that whole snapshot to the injector's event log, its only state
     c. Collect the cell kinds whose own schedule is due; stop here if none
     d. Stop here while any harmed cell is still owed a recovery, run-wide: a cell is owed one
        until it is observed in service under a workers_hash different from the one the fault
        was aimed at
     e. Stop here unless EVERY enabled kind has recovered: every replica that kind has ever
        shown is present, its current (cell_id, workers_hash) is eligible, and at least one
        spare replica survives a kill
     f. Draw a due kind, a cell of that kind and one of its fault forms that is available for
        that cell - preferring a form the log shows has never worked - apply it, record the
        attempt with the target's workers_hash, then draw that kind's next injection time
     g. Read the run's own FaultHookFireEvents back each poll and pair them to armed hooks by
        request id, so an arm and its fire reduce to one outstanding harm rather than two
  3. inject_fault() runs on the actor's own ray concurrency group thread and kills the process,
     or the test layer deletes the pod on kubernetes
  4. The health checker notices by heartbeat timeout
  5. The mini FT controller recovers it (suspend -> resume)
  6. Verify: training completes, no hangs, prod assertions pass

Per-kind schedules: exponential, mean that kind's --*-crash-interval-seconds

Eligibility, per (cell_id, workers_hash), read off the observations alone:
  in service     -> Healthy=True and Allocated=True, plus Serving=True for rollout cells;
                    trainer cells publish no Serving condition and are not asked for one.
                    Healthy=True with reason TrainerUninitialized does not count: a trainer
                    cell reports that from the moment it is allocated, before it has joined
                    the run
  grants         -> the first in-service observation of a generation makes it eligible at once
  carries        -> a weight-update pause (Healthy=Unknown, reason WeightUpdateInProgress,
                    Allocated=True, Serving=True for rollout) keeps an eligibility that
                    generation already had, and grants none it did not
  drops          -> any other reading: explicitly unhealthy, de-allocated, out of the router,
                    paused for an offload, or the cell absent from the listing
  never inherits -> a new workers_hash starts with none

Targeted fault hooks, whenever the mode has real disaggregated engines:
  regime  -> the p2p weight transfer recipe plus --sglang-enable-p2p-fault-injection, and
             --save <dump>/ckpt --save-interval 10 so a recovery source exists to arm against
  forms   -> fault_hook:local among the actor forms when ft covers train, and
             fault_hook:remote_inference_cell among the rollout forms when it covers rollout;
             a remote hook is armed in a trainer and counted against the engine it harms
  source  -> a trainer whose current workers_hash owns a non-empty WeightUpdateAssignmentEvent,
             worker 0 of it, drawn from the same seeded rng. The trainers are listed for this
             draw in their own bounded request, so a rollout-only soak - which never lists,
             targets, schedules or counts trainer cells - can still arm one
  point   -> uniformly among the hooks that source can be shown to reach: the two p2p write
             hooks and after_base_weights always, before_all_gather only at TP/ETP > 1, and for
             a remote request only the two the site names a single peer at
  action  -> drawn from the same seeded rng: sigkill, deadlock or sigstop for a local request, and
             for a remote one only what the receiver implements on itself (sigkill, sigstop). The
             mode goes into the arm request and the arm record, and the fire collector rejects a
             fire whose mode is not the one that was armed
  state   -> armed (recorded before the request leaves) -> that request id fired and was
             judged against the arm -> the victim it names back in service under another
             generation. Every state but the last holds the run-wide harm slot

Witnesses, counted per kind:
  forms   -> every form the enabled components make available succeeded at least once, so a
             soak that clears the injection floors on one form still has to draw the others;
             a hook form only counts once its fire was judged a delivery and its victim came
             back, so an arm the api server answered 200 to proves nothing on its own
  hooks   -> no armed hook is left unresolved when the run ends: an arm whose fire never
             arrived, whose fire answered stale/unknown/refused or named another point,
             source or engine, or whose victim never served again, fails the run
  train   -> >= 2 accepted actor injections, >= 2 healed cells across the
             CellReconfigureEvents, and every injected cell index paired with a healing of
             that same index - no debt left when training ends
  rollout -> >= 2 accepted rollout injections, and every injected cell observed in service
             at least once under a workers_hash other than the one it was injected at - a
             reading the killed incarnation cannot produce however stale it is

Faults are random, so beyond the witnesses neither an exact sequence nor the end-state
membership is asserted.
```

- **Why per-kind schedules and counting**: each kind's cadence stays what it would be in a single-kind soak, and the trainer assertion reads only `actor` injections while the rollout one reads only `rollout` — a mixed soak cannot let one kind's crashes pay for the other's missing heal.
- **Why rollout gets the longer interval**: the replacement pays a full sglang launch plus a weight sync before it can serve again.
- **No per-kind quota, and no redirection either**: a kind without a spare replica does not hand its budget to the other kind - it stops the run's injections until it recovers. A stretch of that shows up as a loud "too few injections" for whichever kind fell short, never as a silent pass.
- **Why recovery is proved by identity, not by waiting**: the api server reports a just-killed cell Healthy for ~95s, far longer than the poll interval, and indep_dp cannot heal from zero survivors, so a naive Healthy count would eventually kill the last replica. Waiting a fixed time is only a guess about how stale that reading may be; the `workers_hash` the api server carries names the incarnation directly, so a Serving reading under a *different* hash is evidence the killed one cannot manufacture at any age.
- **Why only one harm is outstanding run-wide**: a trainer loss takes its assigned inference cells down with it, and a second fault landing during that cascade hits a fleet that is already a replica short and leaves nobody able to say which fault the damage came from. The per-kind clocks stay independent; only the admission is serialized.
- **Why every kind, not just the due one, has to be back**: the victim of a trainer crash is the trainer, and its debt clears as soon as a replacement trainer is in service - but the inference cells that trainer was writing to are retired by the same update and are still relaunching. Checking only the due kind would let the very next draw harm the trainer again while its targets are down. The gate therefore reads every enabled kind, and a kind that is a replica short blocks the whole run rather than redirecting the fault at another kind.
- **Why eligibility is per generation, not per poll**: a pause makes health Unknown, so accepting a pause on its own would let a replacement that has never been probed pass as a spare the moment a weight update starts. Tying the qualification to `(cell_id, workers_hash)` means the pause can only carry forward what a real in-service observation of that same generation already established, and a cell that read unhealthy cannot be laundered back by the next pause.
- **Why no waiting window**: the qualification is evidence, not age. One honest in-service reading is enough, and no number of stale ones ever is.
- **A form that leaves its cell running**: `BaseFaultForm.harms_cell` is false for it, so the draw is recorded without charging that cell a recovery, and it never holds up the next injection.
- **Why a rollout target must be `Serving`, not just `Healthy`**: `Healthy` and even `Running` include a replacement that got weights but cannot answer requests yet, so treating such a replica as recovered would land the fault mid-relaunch. Trainer cells publish no `Serving` condition at all (`compute_cell_status` emits `Allocated` and `Healthy`), so demanding one there would stop every trainer soak; they are judged by `Healthy` and `Allocated`.
- **Why a trainer cell needs `TrainerUninitialized`**: `StateAllocatedUninitialized` reports `Healthy=True` — the cell exists and its workers answer — but it has not run `init()` and holds no rank in the run. Without a way to tell that apart, a replacement would clear its predecessor's recovery debt and count as a spare the instant it was allocated, licensing a second crash of the one trainer that had actually initialized. The reason is set where that status is built (`cell_monitor.HEALTH_TRAINER_UNINITIALIZED`) and only there; a cell whose `Healthy` comes from a real probe never carries it. Nothing waits a fixed time for initialization, and no `Serving` condition is invented for trainer cells.
- **Why a weight-update pause still counts as injectable**: the controller pauses health probing for the duration of an update, and a fault landing inside that window is exactly what the weight-update fault tolerance has to survive. The pause carries an explicit `WeightUpdateInProgress` reason, so it is distinguishable from a cell nobody has probed and from `EnginesOffloaded`, neither of which is a legal target.
- **Why replicas are counted against the most ever seen**: a deleted pod vanishes from the listing instead of reading unhealthy, and the survivors all serve; only the missing replica says the kind is still recovering.
- **An injection whose outcome is unknown**: a request that raised may still have landed, so the cell is owed a recovery exactly as a successful one is, and it is never retried blindly. If no replacement generation serves within `UNKNOWN_INJECTION_RESOLUTION_TIMEOUT_SECONDS`, the injector thread raises and `stop_and_join` re-raises it into the test: whether the fault landed is unknowable at that point, so the run cannot claim either outcome.
- **Why every enabled form has to land**: the floors count injections, not forms, so `inject_fault:sigkill` alone could clear them while `delete_pod` is never tried. This witness makes the draw's preference for an untried form binding.
- **Why the per-cell pairing**: a floor of ">= 2 healings" passes whenever the last crash never recovered. The default intervals are short enough that a soak reliably clears the floors.
- **Why the step budget is 60**: a rollout injection waits for the previous victim to serve again under a new generation, plus a mean-240s exponential wait, so the second accepted rollout injection the witness demands takes well over ten minutes. The budget buys that time instead of lowering the gate that keeps the injector from killing a kind's last live replica.
- **Why the rollout witness is one-sided**: the trainer witness reads the run's own CellReconfigureEvents, which miss nothing; the rollout witness reads sampled polls, which miss windows by construction. It therefore never demands seeing the down half of a recovery - it demands a Serving reading under a generation other than the one that was killed. Undercounting an intermediate recovery cannot fail the run; claiming one that never happened cannot pass it.
- **Why the soak arms hooks at all**: the wall-clock forms decide *when* a fault lands and never *where*, so the phases of a weight update they interrupt are whatever the schedule happened to hit. The hook forms put the same soak's faults at named points inside the update, at the cost of one extra draw per kind rather than a second scenario.
- **Why an arm is recorded before the request is sent**: a 500 or a timeout does not mean nothing was armed — the worker may have taken the request and lost the reply. The record is written first and the responsibility is held from then on, so an arm whose answer never came still blocks the next fault and still fails the run if it never fires. Nothing is re-armed blindly.
- **Why the arm and the fire are one outstanding harm, not two**: they are the same fault, joined by a request id the injector minted. `compute_hook_harms` folds the arm record, the fire read back from the run's events and the later observations into one entry, so the eligibility gate sees one debt and the witnesses read one outcome.
- **Why the fire is read from the event dir rather than the log**: the fault fires in a trainer worker, possibly on another node, and often kills that worker; `EventLogger` writes one line per event and closes, so the record survives. Arms and fires also interleave across workers, so the pairing is by request id and never by which line appeared first.
- **Why a fire is judged rather than counted**: `reach_fault_hook` records the point it reached, the identity it reached it in, the update it happened in and what the delivery answered. A fire naming another hook, another mode, another target, another process identity, no weight version, or — for a remote request — an engine the assignment never gave this sender or a receiver identity it never carried, is rejected. A rejected fire does not clear the arm, so the run fails rather than counting it.
- **Why the update's own sender has to be the armed generation**: a process identity repeats every generation — the same cell index and rank come back with the replacement — so identity alone cannot say which incarnation fired. The fire is joined to the one `WeightUpdateAssignmentEvent` of its version and the armed cell, and that assignment's `trainer_workers_hash` and cell index have to be the ones the arm was taken against. The arm is sent to a cell endpoint, so a replacement appearing between the listing that chose the source and the request itself would be armed under the old cell's name; this is what stops such a fire from being counted, and the arm then stays outstanding until the run fails on it. Nothing is retried, and no replacement that appeared for its own reasons can pay off an arm whose fire was rejected, because a rejected fire names no victim at all.
- **Why `accepted` and `fired` are the only deliveries**: a local fault runs in the process that reached the point, so it records `fired`; a remote one is delivered by the receiver, which answers before signalling itself, so it records `accepted`. Every other answer counts as no delivery, and none of them is retried.
- **Why a request can end without harming anything, and what that is worth**: two answers prove a request cost the run nothing. An arming request the api server refused with its own 400, 404 or 422 refusal body never reached a worker, because every one of those is raised before the arm is passed on; a remote fire answered `stale_target` reached the receiver of the very engine the update assigned this sender, at the identity the arm was taken against, and was refused there. Both release the run-wide slot the arm holds, and neither counts as a successful injection, a recovery or a publication: the form still owes a real delivery. Every other answer keeps the request outstanding — a 500, a timeout or a broken connection may have armed a worker that will fire later, and `refused`, `not_scheduled` and `errored` are exceptions inside the sender that can still reach the p2p write and retire a cell. The 900s unknown-outcome deadline still fails a run whose request never resolves.
- **Why a stale fire has to pass the same checks as a delivered one**: `stale_target` alone only says some receiver refused something. The identity of the fire — its hook, mode, target, delay, the process it fired in, the weight version, the one assignment that version gave the armed generation, the assigned engine at the assigned `workers_hash`, and the receiver's own boot uuid, session and rank — is checked exactly as it is for a delivery, and only then is the request released.
- **Why conflicting terminal evidence fails the run**: acknowledgement and pre-arm refusal cannot both describe one request, nor can pre-arm refusal and delivery or harmless refusal and delivery. The check is independent of event order. The collector therefore keeps reading fires for resolved requests, remembers each raw fire it has already read, ignores only an exact repeat, and rejects different fire contents for the same request.
- **Why the source is a trainer with a live assignment, not any trainer**: the write hooks are only reached by a sender, and `WeightUpdateAssignmentEvent` is the run's own record of which trainer generation was given targets. Matching it against the cell's current `workers_hash` also rules out arming a replacement against its predecessor's assignment. `before_all_gather` additionally needs TP or ETP above one, since below it the branch holding the hook is skipped.
- **Why worker 0**: `miles/ray/specs/train.py` passes `worker_in_cell_index` as the actor's `rank`, and `train_actor.py` uses that same rank for `RANK` and for `TrainProcessIdentity.rank_within_cell`, so `sub_index` 0 is the cell's distributed rank 0. `get_data_replica_rank_and_size` orders data-replica columns by their smallest global rank, so rank 0's gathered rank is 0, and `plan_p2p` gives source rank 0 the first target whenever the target list is non-empty. It is worker 0 because that is what the code does today, not because leaders are conventionally index 0; changing either mapping has to change this draw.
- **Why the remote form is a rollout form armed in a trainer**: the fault is delivered to an engine, so the engine's kind is what owes the recovery and what the rollout floors count. The trainer is only the trigger — a remote request does not harm it, and its cell name is never allowed to stand in for the victim. The injector lists only the kinds it crashes, so the trigger is found by a separate five-second listing of `actor` cells rather than by widening the poll: a trainer that no fault targets must not join the schedules, the spare counting or the trainer healing witness, and none of those read that listing.
- **Why the victim of a remote hook is not chosen at draw time**: the request names no cell. The site names one, from the peer it actually connected to, and the fire carries that cell id, that `workers_hash` and the receiver's own boot uuid, session and rank. A candidate picked while sampling would be a guess that a later unrelated replacement could make look true.
- **Why the soak saves checkpoints only when hooks are enabled**: the readiness gate the targeted scenarios use is reused whole — a completed training step, a checkpoint tracker naming an iteration, and a non-empty publication — so arming cannot take out the only source a replacement heals from. The interval is wide because the soak is 60 steps and only needs the tracker to exist.
- **Stopping the injector**: `stop_and_join` asserts the thread actually stopped, since a thread still mid-injection could crash a cell nothing will heal, and would race the witness being read. It then reads the fire events and the cell listing one last time, in that order, so a fault that fired just before the stop still gets its victim's final reading.

### `scenario_realistic_gsm8k`

```
Type: soak (no baseline run; reference = the baseline test's wandb curves)
Entry: test_realistic_gsm8k__kill_train_rollout.py, no mode variants
CLI: --seed (42), --num-rollout (250), --trainer-crash-interval-seconds (600),
     --rollout-crash-interval-seconds (1200), --metric-threshold (0.55), --fully-async (off);
     no --mode

Recipe: Qwen2.5-0.5B-Instruct, GRPO, 250 rollouts, over the gsm8k RL recipe of
        tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py, whose regular CI runs are the no-fault
        reference wandb curves
Layout: mirrors kill_train__dp2_cp2__moe_5layer - 2 cells x CP2 on 4 train GPUs + 4 rollout engines
        x 1 GPU, disaggregated
Faults: scenario_random_crash's injection loop (shared conftest_ft/fault_injection/), with
        --ft-components train rollout asked for outright, so both trainer cells and engines crash

Assertions:
  1. --ci-metric-checker-key eval/gsm8k against a threshold that must stay identical to the
     no-fault baseline's (0.55); passes if ANY eval reaches it
  2. assert_healing, shared with scenario_random_crash, so both the trainer reconfigure
     assertions and the rollout recovery witness apply here

Fault recovery must not cost end-to-end learning, which the comparison scenarios cannot observe.
```

- **Why the threshold does not move**: it is the entire value of this scenario, so engine crashes are paid for with a lower rollout crash rate, never with a lower bar.

### `scenario_random_crash_fully_async` and `scenario_realistic_gsm8k_fully_async`

```
Type: shells - each calls its sync twin with fully_async=True and pins nothing else
Entries: test_random_crash_fully_async__kill_train_rollout__dp2_cp2.py,
         test_realistic_gsm8k_fully_async__kill_train_rollout.py (no mode)
Differs from the twin: train_async.py instead of train.py, plus --fully-async
                       --pause-generation-mode in_place; test name gains a _fully_async suffix,
                       which separates the dump dirs and wandb runs
Same as the twin: model, parallelism, batch sizes, CLI and every assertion, by construction
```

- **Why it matters**: production fully-async keeps the engines generating across weight updates, so a crash lands the system in states no strictly-alternating soak reaches.
- **Why `--pause-generation-mode in_place`**: the default retract mode can deadlock `flush_cache` under load, and a soak whose verdict is "training finished without hanging" cannot tell that deadlock from the failure it exists to catch.
- **Asserted before the cluster comes up**: the mode has real engines and is not colocated. Recorded rollout data would prove nothing about generating while training, and `train_async.py` rejects colocation outright.
- **Deliberately uncovered**: `train_async.py` without `--fully-async`, the strictly easier case, at tens of minutes to hours of 8-GPU time per soak.
