#!/usr/bin/env python3
"""Bounded Option D Supplement — Step 2: Post-Hoc Stage-B Alignment Evaluation (3501..3505, N=45).

Evaluates 5 additional frozen systems once under the exact pit-free Stage-B contract
(pits_enabled=False, legal-option masking, jev_rl_v7.mapdata.npz, calibrated_params.json,
action_spec_hash=a65f275a8fff3740) on seeds [3501, 3502, 3503, 3504, 3505] (45 scenarios):
  1. kev_0_8b              (jaredpalmer/kev-0.8b via KevRunner)
  2. minilm_offshelf       (models/ms-marco-MiniLM-L6-v2, off-the-shelf 22.7M cross-encoder)
  3. m2_qwen_ops_lora      (models/m2_qwen3.5_0.8b_lora_s42, Software Ops LoRA, T=2.02)
  4. raw_qwen_base         (Qwen/Qwen3.5-0.8B-Base, exact base initialization, same letter-logit interface)
  5. frontier_llm_dated    (deepseek-v4.1-flash-1017 via .env QWEN_BASE_URL, full call logging)

Reuses existing 3501..3505 Stage-B episode logs for driver_only, track_rule, lookup_9cell,
nn3_racing_v1, and qwen_lora_racing_v1 after verifying exact scenario_id and hash parity.
Preserves all original frozen baselines untouched; writes strictly to reports/alignment_supplement/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "decision_eval" / "src"
KEV_VENDOR = REPO_ROOT / "decision_eval" / "vendor" / "kev"
for p in (str(REPO_ROOT), str(SRC_DIR), str(KEV_VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from decision_eval.paths import HF_CACHE_DIR, configure_bundled_hf_cache, resolve_local_model_path  # noqa: E402
from decision_eval.racing_contract import (  # noqa: E402
    SCHEMA_VERSION,
    action_spec_hash,
    generate_scenarios_for_seeds,
    render_obs_text,
)
from decision_eval.racing_rollouts import (  # noqa: E402
    DEFAULT_DRIVER,
    load_frozen_sim_assets,
    run_closed_loop_policy,
)
from decision_eval.runners import KevRunner, M2QwenLoRARunner, _softmax, parse_json_decision  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "alignment_supplement"
EXISTING_WORKERS_DIR = REPO_ROOT / "reports" / "racing_engineer_v1" / "_tmp_eval_workers"
SEEDS = [3501, 3502, 3503, 3504, 3505]
WTL_TOL_S = 0.50

Q_MAP = {
    "build": "Select the pre-race car build to maximize 4-lap completion and minimize race time.",
    "pace": "Select the driving pace & ERS mode for this lap.",
    "pit": "Select whether to stay out or pit at the end of this lap.",
}


def load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class KevStageBAdapter:
    """Pure-PyTorch Kev-0.8B loader (exact fp32 LoRA merge W += (alpha/r)*B@A, zero peft dependency)."""

    def __init__(self) -> None:
        from safetensors.torch import load_file
        from kev.checkpoint import Checkpoint
        from kev.model import DecisionModel, load_tokenizer

        kev_dir = REPO_ROOT / "models" / "modelscope" / "jaredpalmer" / "kev-0.8b"
        ck = Checkpoint(str(kev_dir))
        meta = ck.meta
        acfg = ck.adapter_config()
        self.tok = load_tokenizer(meta.base, revision=meta.base_revision)
        self.model = DecisionModel(
            meta.base, self.tok, "cpu", lora=None, revision=meta.base_revision,
            head_dim=meta.head_dim, option_isolation=meta.option_isolation, dtype=torch.float32,
        )
        adapters = load_file(str(kev_dir / "adapter_model.safetensors"))
        scale = float(acfg["lora_alpha"]) / float(acfg["r"])
        params = dict(self.model.lm.named_parameters())
        a_keys = sorted(k for k in adapters if k.endswith(".lora_A.weight"))
        for ak in a_keys:
            bk = ak.replace(".lora_A.weight", ".lora_B.weight")
            target_key = ak.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
            delta = (adapters[bk].float() @ adapters[ak].float()) * scale
            params[target_key].data.add_(delta.to(params[target_key].dtype))
        self.model.head.load_state_dict(meta.head)
        self.model.head.temperature = meta.temperature
        self.model.eval()
        self.latencies_ms: list[float] = []
        self.truncated_calls = 0

    def decide(self, kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]) -> tuple[str | None, bool, float]:
        from kev.api import Choice, SystemOneRequest, to_answers, to_record
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

        st = render_obs_text(obs, include_contract=True)
        criteria = {o: d for o, d in zip(opts, descs)}
        req = SystemOneRequest(
            state=st,
            model="jaredpalmer/kev-0.8b",
            questions={"decision": Choice(type="choice", instructions=Q_MAP[kind], criteria=criteria)},
        )
        record, meta = to_record(req)
        encoded = self.model.encode(self.tok, record, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
        if encoded.get("state_truncated"):
            self.truncated_calls += 1
        t0 = time.perf_counter()
        probs_rows = self.model.probs(encoded)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        ans = to_answers([[float(x) for x in row.tolist()] for row in probs_rows], meta)
        pmap = ans["decision"].get("probabilities") or {}
        probs = [float(pmap.get(o, 0.0)) for o in opts]
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return opts[best_idx], True, float(probs[best_idx])


class MiniLMStageBAdapter:
    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = ckpt_dir
        self.tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), local_files_only=True)
        self.model.eval()
        self.latencies_ms: list[float] = []
        self.truncated_calls = 0
        self.max_tokens_seen = 0

    @torch.inference_mode()
    def decide(self, kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]) -> tuple[str | None, bool, float]:
        st = render_obs_text(obs, include_contract=True)
        q_text = f"{Q_MAP[kind]}\n{st}"
        queries = [q_text] * len(opts)
        docs = [f"{o}: {d}" for o, d in zip(opts, descs)]
        t0 = time.perf_counter()
        enc = self.tok(queries, docs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        n_tok = int(enc["input_ids"].shape[1])
        self.max_tokens_seen = max(self.max_tokens_seen, n_tok)
        if n_tok >= 512:
            self.truncated_calls += 1
        logits = self.model(**enc).logits.squeeze(-1).float().cpu().tolist()
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        probs = _softmax([float(x) for x in logits])
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return opts[best_idx], True, float(probs[best_idx])


class M2QwenOpsStageBAdapter:
    def __init__(self, ckpt_dir: Path) -> None:
        self.runner = M2QwenLoRARunner(ckpt_dir, device="cpu")
        self.latencies_ms: list[float] = []
        self.truncated_calls = 0

    def decide(self, kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]) -> tuple[str | None, bool, float]:
        st = render_obs_text(obs, include_contract=True)
        res = self.runner.score_decision(st, Q_MAP[kind], opts, descs, max_len=512)
        self.latencies_ms.append(float(res["latency_ms"]))
        if res.get("state_truncated"):
            self.truncated_calls += 1
        return res["choice"], bool(res["parse_ok"]), float(res["confidence"])


class RawQwenBaseStageBAdapter:
    """Exact raw Qwen/Qwen3.5-0.8B-Base using identical M2QwenLoRARunner prompt & letter-ID interface."""

    LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H")
    EXPECTED_LETTER_IDS = [357, 417, 351, 414, 458]

    def __init__(self) -> None:
        configure_bundled_hf_cache()
        self.base_model_id = "Qwen/Qwen3.5-0.8B-Base"
        resolved = resolve_local_model_path(self.base_model_id)
        cache_kw: dict[str, Any] = {"cache_dir": str(HF_CACHE_DIR)} if HF_CACHE_DIR.is_dir() else {}
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True, local_files_only=True, **cache_kw)
        resolved_ids: list[int] = []
        for ch in self.LETTERS[: len(self.EXPECTED_LETTER_IDS)]:
            t_sp = self.tokenizer.encode(f" {ch}", add_special_tokens=False)
            t_pl = self.tokenizer.encode(ch, add_special_tokens=False)
            resolved_ids.append(int(t_sp[0]) if len(t_sp) == 1 else int(t_pl[0]))
        if resolved_ids != self.EXPECTED_LETTER_IDS:
            raise ValueError(f"Raw Qwen letter_ids {resolved_ids} != {self.EXPECTED_LETTER_IDS}")
        self.letter_ids = resolved_ids
        self.lid_tensor = torch.tensor(self.letter_ids, dtype=torch.long)
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=torch.float32, trust_remote_code=True, local_files_only=True, **cache_kw
        )
        self.model.eval()
        self.latencies_ms: list[float] = []
        self.truncated_calls = 0

    def _format_prompt(self, state: str, question: str, options: list[str], descs: list[str]) -> str:
        opt_lines = [f"{self.LETTERS[i]}. {options[i]} — {descs[i]}" for i in range(len(options))]
        return (
            f"Context:\n{state}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n" + "\n".join(opt_lines) + "\n\n"
            f"Select the single best option letter (A-{self.LETTERS[len(options) - 1]}).\nAnswer:"
        )

    @torch.inference_mode()
    def decide(self, kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]) -> tuple[str | None, bool, float]:
        st = render_obs_text(obs, include_contract=True)
        prompt = self._format_prompt(st, Q_MAP[kind], opts, descs)
        enc = self.tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
        if int(enc["input_ids"].shape[1]) >= 512:
            self.truncated_calls += 1
        t0 = time.perf_counter()
        out_logits = self.model(**enc).logits[0, -1, self.lid_tensor[: len(opts)]].float().cpu().tolist()
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        probs = _softmax([float(x) for x in out_logits])
        best_idx = max(range(len(opts)), key=lambda i: probs[i])
        return opts[best_idx], True, float(probs[best_idx])


class FrontierDatedStageBAdapter:
    """Dated cloud reference with full provenance, retry, token, and raw-response logging."""

    def __init__(self, base_url: str, model: str, api_key: str, log_path: Path) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.log_path = log_path
        self.log_path.write_text("", encoding="utf-8")
        self.latencies_ms: list[float] = []
        self.returned_models: Counter[str] = Counter()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.retries = 0
        self.invalid_calls = 0

    def decide(self, kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]) -> tuple[str | None, bool, float]:
        st = render_obs_text(obs, include_contract=True)
        numbered = [f"{i}: {o} ({d})" for i, (o, d) in enumerate(zip(opts, descs))]
        prompt = (
            "You are a decision assistant. Pick exactly one option index and output ONLY valid JSON "
            'with keys "choice_index" (integer) and "confidence" (float between 0.0 and 1.0).\n\n'
            f"State:\n{st}\n\nQuestion: {Q_MAP[kind]}\nOptions:\n" + "\n".join(numbered) + "\n\nJSON:"
        )
        url = self.base_url + "/chat/completions"
        body_obj: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload = json.dumps(body_obj).encode("utf-8")

        raw_json: dict[str, Any] | None = None
        err_msg: str | None = None
        lat_ms = 0.0
        for attempt in range(3):
            if attempt > 0:
                self.retries += 1
                time.sleep(1.5 * attempt)
            http_req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(http_req, timeout=60) as resp:
                    raw_json = json.loads(resp.read().decode("utf-8"))
                lat_ms = (time.perf_counter() - t0) * 1000.0
                err_msg = None
                break
            except Exception as exc:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                err_msg = f"{type(exc).__name__}: {exc}"

        self.total_calls += 1
        self.latencies_ms.append(lat_ms)
        if raw_json is None:
            self.invalid_calls += 1
            rec = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scenario_track": obs.get("track"),
                "weather_regime": obs.get("weather_regime"),
                "lap": obs.get("lap"),
                "kind": kind,
                "error": err_msg,
                "parse_ok": False,
                "choice": None,
                "latency_ms": round(lat_ms, 2),
            }
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            return None, False, 0.0

        ret_model = str(raw_json.get("model", ""))
        if ret_model:
            self.returned_models[ret_model] += 1
        usage = raw_json.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        self.total_prompt_tokens += p_tok
        self.total_completion_tokens += c_tok

        gen_text = str(raw_json.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        parse_ok, choice_idx, confidence, _probs = parse_json_decision(gen_text, len(opts))
        choice = opts[choice_idx] if parse_ok else None
        if not parse_ok:
            self.invalid_calls += 1

        rec = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_track": obs.get("track"),
            "weather_regime": obs.get("weather_regime"),
            "lap": obs.get("lap"),
            "kind": kind,
            "requested_model": self.model,
            "returned_model": ret_model,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "latency_ms": round(lat_ms, 2),
            "parse_ok": parse_ok,
            "choice": choice,
            "confidence": round(confidence, 4),
            "raw_output": gen_text,
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return choice, parse_ok, float(confidence)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ts = [float(r["finish_time_s"]) for r in rows if r["finished"] and r["finish_time_s"] is not None]
    fin = sum(1 for r in rows if r["finished"])
    fail_reasons = Counter(str(r.get("status_name") or r.get("status")) for r in rows if not r["finished"])
    return {
        "n": len(rows),
        "finishes": fin,
        "completion_rate_pct": round(100.0 * fin / max(len(rows), 1), 2),
        "mean_time_finishers_s": round(float(np.mean(ts)), 2) if ts else None,
        "pits_total": sum(int(r.get("pit_stops", 0)) for r in rows),
        "delegated_calls": sum(int(r.get("delegated_calls", 0)) for r in rows),
        "illegal_proposed": sum(int(r.get("illegal_proposed", 0)) for r in rows),
        "build_counts": dict(Counter(r["chosen_build"] for r in rows)),
        "mode_counts": dict(Counter(m for r in rows for m in r["mode_history"])),
        "failure_status_counts": dict(fail_reasons),
    }


def paired_compare_stratified(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]], n_boot: int = 4000) -> dict[str, Any]:
    assert [r["scenario_id"] for r in base_rows] == [r["scenario_id"] for r in cand_rows]
    n = len(base_rows)
    both_fin = rescue = reg = both_fail = 0
    deltas: list[float] = []
    wins = ties_fin = losses = 0
    cells: dict[str, list[int]] = {}
    comp_diffs = np.zeros(n, dtype=np.float64)
    delta_per_row = np.full(n, np.nan, dtype=np.float64)

    for i, (b, c) in enumerate(zip(base_rows, cand_rows)):
        cid = f"{b['track']}_{b['weather_regime'].lower()}"
        cells.setdefault(cid, []).append(i)
        bf, cf = bool(b["finished"]), bool(c["finished"])
        comp_diffs[i] = float(int(cf) - int(bf))
        if bf and cf:
            both_fin += 1
            d = float(c["finish_time_s"]) - float(b["finish_time_s"])
            deltas.append(d)
            delta_per_row[i] = d
            if d < -WTL_TOL_S:
                wins += 1
            elif d > WTL_TOL_S:
                losses += 1
            else:
                ties_fin += 1
        elif cf and not bf:
            rescue += 1
            wins += 1
        elif bf and not cf:
            reg += 1
            losses += 1
        else:
            both_fail += 1

    rng = np.random.default_rng(42)
    cell_lists = [np.array(idxs, dtype=np.int64) for idxs in cells.values()]
    boot_comp = np.empty(n_boot, dtype=np.float64)
    boot_time = []
    for b_i in range(n_boot):
        sampled = np.concatenate([rng.choice(arr, size=len(arr), replace=True) for arr in cell_lists])
        boot_comp[b_i] = float(np.mean(comp_diffs[sampled]) * 100.0)
        dt_vals = delta_per_row[sampled]
        dt_vals = dt_vals[~np.isnan(dt_vals)]
        if len(dt_vals) > 0:
            boot_time.append(float(np.mean(dt_vals)))

    comp_ci = [round(float(np.percentile(boot_comp, 2.5)), 2), round(float(np.percentile(boot_comp, 97.5)), 2)]
    time_ci = (
        [round(float(np.percentile(boot_time, 2.5)), 3), round(float(np.percentile(boot_time, 97.5)), 3)]
        if len(boot_time) >= 10
        else [None, None]
    )
    return {
        "n": n,
        "paired_outcomes": {"both_finish": both_fin, "rescue": rescue, "regression": reg, "both_fail": both_fail},
        "completion_diff_pp": round(float(np.mean(comp_diffs) * 100.0), 2),
        "comp_ci95_2sided_pp": comp_ci,
        "common_finishers_n": len(deltas),
        "mean_paired_delta_s": round(float(np.mean(deltas)), 3) if deltas else None,
        "median_paired_delta_s": round(float(np.median(deltas)), 3) if deltas else None,
        "time_ci95_2sided_s": time_ci,
        "wtl_0_5s": {"wins": wins, "ties_finished": ties_fin, "ties_both_failed": both_fail, "losses": losses},
    }


def load_existing_3501_3505(model_key: str) -> list[dict[str, Any]]:
    prefix = "fresh_confirm_qwen" if model_key == "qwen_lora_racing_v1" else "fresh_confirm"
    rows: list[dict[str, Any]] = []
    seed_set = set(SEEDS)
    for tr in ("parkland", "highland", "crossover"):
        fp = EXISTING_WORKERS_DIR / f"{prefix}_{model_key}_{tr}.json"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing existing Stage-B worker file: {fp}")
        data = json.loads(fp.read_text(encoding="utf-8"))
        rows.extend([r for r in data if int(r.get("seed", -1)) in seed_set])
    rows.sort(key=lambda r: r["scenario_id"])
    assert len(rows) == 45, f"Expected 45 rows for {model_key}, got {len(rows)}"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["kev_0_8b", "minilm_offshelf", "m2_qwen_ops_lora", "raw_qwen_base", "frontier_llm_dated"])
    ap.add_argument("--skip-existing-run-if-cached", action="store_true")
    args = ap.parse_args()

    t_all0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes_dir = OUT_DIR / "stage_b_45_episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    device = torch.device("cpu")
    driver_model, obs_mean, obs_std, is_residual, cfg, builds = load_frozen_sim_assets(device=device)
    scenarios = sorted(generate_scenarios_for_seeds(SEEDS), key=lambda s: s["scenario_id"])
    expected_sids = [s["scenario_id"] for s in scenarios]
    assert len(scenarios) == 45

    dotenv = load_dotenv(REPO_ROOT / ".env")
    run_meta: dict[str, Any] = {}

    for mname in args.models:
        ep_file = episodes_dir / f"{mname}_3501_3505.json"
        if args.skip_existing_run_if_cached and ep_file.is_file():
            print(f"[Stage-B Align] Using cached {ep_file.name}", flush=True)
            continue

        t0 = time.perf_counter()
        print(f"\n[Stage-B Align] Initializing {mname}...", flush=True)
        if mname == "kev_0_8b":
            adapter = KevStageBAdapter()
        elif mname == "minilm_offshelf":
            adapter = MiniLMStageBAdapter(REPO_ROOT / "models" / "ms-marco-MiniLM-L6-v2")
        elif mname == "m2_qwen_ops_lora":
            adapter = M2QwenOpsStageBAdapter(REPO_ROOT / "models" / "m2_qwen3.5_0.8b_lora_s42")
        elif mname == "raw_qwen_base":
            adapter = RawQwenBaseStageBAdapter()
        elif mname == "frontier_llm_dated":
            base_url = os.environ.get("QWEN_BASE_URL") or dotenv.get("QWEN_BASE_URL")
            model_id = os.environ.get("QWEN_MODEL") or dotenv.get("QWEN_MODEL")
            api_key = os.environ.get("QWEN_API_KEY") or dotenv.get("QWEN_API_KEY")
            if not (base_url and model_id and api_key):
                raise RuntimeError("Missing QWEN_BASE_URL / QWEN_MODEL / QWEN_API_KEY in .env or environment.")
            adapter = FrontierDatedStageBAdapter(
                base_url=base_url,
                model=model_id,
                api_key=api_key,
                log_path=OUT_DIR / "frontier_stage_b_45_calls.jsonl",
            )
        else:
            raise ValueError(f"Unknown alignment model: {mname}")

        # Non-evaluation smoke test on synthetic obs before touching 3501..3505
        smoke_obs = {
            "decision_type": "build",
            "track": "parkland", "track_length_m": 5795.8, "laps_total": 4,
            "weather_regime": "Dry", "forecast_wet_start": 0.02, "forecast_wet_end": 0.02,
            "forecast_rain_prob": 0.05, "track_temp_c": 30.0,
            "legal_options": ["dry_low_drag", "dry_balanced", "damp_mid_downforce", "wet_high_downforce"],
        }
        sc_choice, sc_ok, sc_conf = adapter.decide(
            "build", smoke_obs,
            ["dry_low_drag", "dry_balanced", "damp_mid_downforce", "wet_high_downforce"],
            ["low drag", "balanced", "mid downforce", "high downforce"],
        )
        print(f"  [Smoke] {mname}: choice={sc_choice} ok={sc_ok} conf={sc_conf:.3f}", flush=True)
        # Reset smoke latency from stats
        adapter.latencies_ms.clear()

        rows = run_closed_loop_policy(
            scenarios, adapter.decide, driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, pits_enabled=False
        )
        rows.sort(key=lambda r: r["scenario_id"])
        assert [r["scenario_id"] for r in rows] == expected_sids
        wall_s = round(time.perf_counter() - t0, 1)

        meta: dict[str, Any] = {
            "wall_s": wall_s,
            "decision_calls_measured": len(adapter.latencies_ms),
            "latency_p50_ms": round(float(np.percentile(adapter.latencies_ms, 50)), 2) if adapter.latencies_ms else None,
            "latency_p95_ms": round(float(np.percentile(adapter.latencies_ms, 95)), 2) if adapter.latencies_ms else None,
            "truncated_calls": getattr(adapter, "truncated_calls", 0),
        }
        if isinstance(adapter, FrontierDatedStageBAdapter):
            meta.update({
                "provider_endpoint": adapter.base_url,
                "requested_model": adapter.model,
                "returned_models": dict(adapter.returned_models),
                "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "prompt_tokens_total": adapter.total_prompt_tokens,
                "completion_tokens_total": adapter.total_completion_tokens,
                "retries_total": adapter.retries,
                "invalid_calls_total": adapter.invalid_calls,
                "decoding": {"temperature": 0.0, "max_tokens": 128, "response_format": "json_object", "enable_thinking": False},
            })
        run_meta[mname] = meta
        ep_file.write_text(json.dumps({"model": mname, "meta": meta, "episodes": rows}, indent=2) + "\n", encoding="utf-8")
        s = summarize_rows(rows)
        print(
            f"  [DONE] {mname:20s} | fin={s['finishes']}/45 ({s['completion_rate_pct']}%) | "
            f"mean={s['mean_time_finishers_s']}s | p50={meta['latency_p50_ms']}ms | wall={wall_s}s",
            flush=True,
        )

    # Assemble complete 10-model comparison (5 existing + 5 new)
    existing_keys = ["driver_only", "track_rule", "lookup_9cell", "nn3_racing_v1", "qwen_lora_racing_v1"]
    new_keys = ["kev_0_8b", "minilm_offshelf", "raw_qwen_base", "m2_qwen_ops_lora", "frontier_llm_dated"]
    all_episodes: dict[str, list[dict[str, Any]]] = {}
    all_meta: dict[str, Any] = {}

    for k in existing_keys:
        all_episodes[k] = load_existing_3501_3505(k)
        all_meta[k] = {"source": "reused_frozen_racing_engineer_v1_3501_3505"}
    for k in new_keys:
        fp = episodes_dir / f"{k}_3501_3505.json"
        if fp.is_file():
            blob = json.loads(fp.read_text(encoding="utf-8"))
            all_episodes[k] = blob["episodes"]
            all_meta[k] = blob["meta"]

    present_keys = list(all_episodes.keys())
    common_sids = set(expected_sids)
    for k in present_keys:
        common_sids &= {r["scenario_id"] for r in all_episodes[k] if r["finished"]}

    models_out: dict[str, Any] = {}
    for k in present_keys:
        rows = all_episodes[k]
        common_ts = [float(r["finish_time_s"]) for r in rows if r["scenario_id"] in common_sids]
        models_out[k] = {
            "summary": summarize_rows(rows),
            "meta": all_meta[k],
            "mean_time_all_model_common_s": round(float(np.mean(common_ts)), 2) if common_ts else None,
            "vs_driver_only": paired_compare_stratified(all_episodes["driver_only"], rows) if k != "driver_only" else None,
            "vs_track_rule": paired_compare_stratified(all_episodes["track_rule"], rows) if k != "track_rule" else None,
        }

    qwen_contrasts: dict[str, Any] = {}
    if {"raw_qwen_base", "m2_qwen_ops_lora", "qwen_lora_racing_v1"}.issubset(all_episodes.keys()):
        qwen_contrasts = {
            "racing_qwen_vs_raw_qwen_base": paired_compare_stratified(all_episodes["raw_qwen_base"], all_episodes["qwen_lora_racing_v1"]),
            "m2_ops_qwen_vs_raw_qwen_base": paired_compare_stratified(all_episodes["raw_qwen_base"], all_episodes["m2_qwen_ops_lora"]),
            "racing_qwen_vs_m2_ops_qwen": paired_compare_stratified(all_episodes["m2_qwen_ops_lora"], all_episodes["qwen_lora_racing_v1"]),
        }

    summary = {
        "protocol": "Bounded_Option_D_Post_Hoc_Stage_B_Alignment_45",
        "label": "Post-hoc evaluation of frozen models on the previously inspected 45-case confirmation subset (3501..3505)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "action_spec_hash": action_spec_hash(),
        "driver_weights": str(DEFAULT_DRIVER.name),
        "pits_enabled": False,
        "seeds": SEEDS,
        "n_scenarios": 45,
        "all_model_common_finishers_n": len(common_sids),
        "total_wall_s": round(time.perf_counter() - t_all0, 1),
        "models": models_out,
        "qwen_adaptation_contrasts": qwen_contrasts,
    }

    out_summary = OUT_DIR / "stage_b_alignment_45_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved 10-model Stage-B alignment summary -> {out_summary} (common n={len(common_sids)})", flush=True)


if __name__ == "__main__":
    main()
