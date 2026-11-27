"""FastAPI application hosting the local decision API, Web UI, metrics, and SVG charts."""
from __future__ import annotations

import json
import urllib.error
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from decision_eval import charts as charts_mod
from decision_eval import metrics as metrics_mod
from decision_eval import paths
from decision_eval.runners import KevRunner, M2QwenLoRARunner, OptionalNNAdapter, call_frontier_llm, create_qwen_runner


class ScoreRequest(BaseModel):
    model: Literal["m2-qwen3.5-0.8b-lora-s42", "kev-0.8b", "qwen-0.8b-ll", "qwen-0.8b-json"] = "m2-qwen3.5-0.8b-lora-s42"
    state: str
    question: str
    options: list[str] = Field(..., min_length=2)
    option_descriptions: list[str] | None = None


class CompareRequest(BaseModel):
    state: str
    question: str
    options: list[str] = Field(..., min_length=2)
    option_descriptions: list[str] | None = None
    models: list[Literal["m2-qwen3.5-0.8b-lora-s42", "kev-0.8b", "qwen-0.8b-ll", "qwen-0.8b-json"]] | None = None


class SystemOnePayload(BaseModel):
    state: str
    questions: dict[str, dict[str, Any]]
    model: str = "jaredpalmer/kev-0.8b"


class FrontierRequest(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    api_key: str
    state: str
    question: str
    options: list[str] = Field(..., min_length=2)
    option_descriptions: list[str] | None = None


class NNScoreRequest(BaseModel):
    features: list[float]
    options: list[str] = Field(..., min_length=2)


class AnalyzeRequest(BaseModel):
    records: list[dict[str, Any]]
    models: list[str]
    config: dict[str, Any] = Field(default_factory=dict)
    frontier_in_price: float = 0.15
    frontier_out_price: float = 0.60
    merge_existing: bool = True


def discover_datasets() -> dict[str, tuple[str, Path]]:
    candidates = [
        ("all_test", "Full Multi-Suite Benchmark (all_test.jsonl)", paths.BENCHMARKS_DIR / "all_test.jsonl"),
        ("all_calibrate", "Full Multi-Suite Calibrate (all_calibrate.jsonl)", paths.BENCHMARKS_DIR / "all_calibrate.jsonl"),
        ("suite1_software_ops_test", "Suite 1: Software Ops (test)", paths.BENCHMARKS_DIR / "suite1_software_ops_test.jsonl"),
        ("suite2_evidence_discipline_test", "Suite 2: Engineering Evidence Discipline (test)", paths.BENCHMARKS_DIR / "suite2_evidence_discipline_test.jsonl"),
        ("suite3_car_build_test", "Suite 3: Car-Build Enriched (test)", paths.BENCHMARKS_DIR / "suite3_car_build_test.jsonl"),
        ("suite4_stress_test", "Suite 4: Stress Test (Context Length + Paraphrase)", paths.BENCHMARKS_DIR / "suite4_stress_test.jsonl"),
        ("car_build_test", "Legacy Car-Build Benchmark (test.jsonl)", paths.CAR_BUILD_DIR / "test.jsonl"),
        ("car_build_calibrate", "Legacy Car-Build Benchmark (calibrate.jsonl)", paths.CAR_BUILD_DIR / "calibrate.jsonl"),
        ("car_build_train", "Legacy Car-Build Benchmark (train.jsonl)", paths.CAR_BUILD_DIR / "train.jsonl"),
    ]
    return {ds_id: (label, p) for ds_id, label, p in candidates if p.is_file()}


def create_app(
    kev_run: str | None,
    qwen_base_model: str | None,
    qwen_instruct_model: str | None,
    device: str,
    backend: str = "auto",
) -> FastAPI:
    state_store: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if paths.M2_QWEN_LORA_S42_DIR.is_dir():
            try:
                state_store["m2_qwen_lora_s42"] = M2QwenLoRARunner(paths.M2_QWEN_LORA_S42_DIR, device=device)
            except Exception as exc:
                state_store["m2_qwen_lora_s42_error"] = str(exc)
                print(f"[decision_eval] Warning: failed to load m2-qwen3.5-0.8b-lora-s42: {exc}", flush=True)
        if kev_run:
            try:
                state_store["kev"] = KevRunner(kev_run, device, backend=backend)
            except Exception as exc:
                state_store["kev_error"] = str(exc)
                print(f"[decision_eval] Warning: failed to load kev-0.8b ({kev_run}): {exc}", flush=True)
        if qwen_base_model:
            try:
                state_store["qwen_ll"] = create_qwen_runner(qwen_base_model, device, backend=backend)
            except Exception as exc:
                state_store["qwen_ll_error"] = str(exc)
                print(f"[decision_eval] Warning: failed to load qwen-0.8b-ll ({qwen_base_model}): {exc}", flush=True)
        if qwen_instruct_model:
            if qwen_instruct_model == qwen_base_model and "qwen_ll" in state_store:
                state_store["qwen_json"] = state_store["qwen_ll"]
            else:
                try:
                    state_store["qwen_json"] = create_qwen_runner(qwen_instruct_model, device, backend=backend)
                except Exception as exc:
                    state_store["qwen_json_error"] = str(exc)
                    print(f"[decision_eval] Warning: failed to load qwen-0.8b-json ({qwen_instruct_model}): {exc}", flush=True)
        elif "qwen_ll" in state_store:
            state_store["qwen_json"] = state_store["qwen_ll"]
        yield
        state_store.clear()

    app = FastAPI(title="Decision AI Evaluation Studio API", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        if not paths.UI_HTML.is_file():
            return HTMLResponse("<h3>ui/index.html not found</h3>", status_code=404)
        return HTMLResponse(paths.UI_HTML.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        info: dict[str, Any] = {"status": "ok", "device": device, "models": {}}
        if "m2_qwen_lora_s42" in state_store:
            m2r: M2QwenLoRARunner = state_store["m2_qwen_lora_s42"]
            info["models"]["m2-qwen3.5-0.8b-lora-s42"] = {"model_id": m2r.base_model_id, "backend": m2r.backend, "dtype": m2r.dtype, "temperature": m2r.temperature, "load_s": m2r.load_s}
        if "kev" in state_store:
            k: KevRunner = state_store["kev"]
            info["models"]["kev-0.8b"] = {"model_id": k.run_id, "backend": k.backend, "base_source": k.base_source, "load_s": k.load_s}
        if "qwen_ll" in state_store:
            q = state_store["qwen_ll"]
            info["models"]["qwen-0.8b-ll"] = {"model_id": q.model_id, "backend": getattr(q, "backend", f"torch-{q.device}"), "dtype": q.dtype, "load_s": q.load_s}
        if "qwen_json" in state_store:
            qj = state_store["qwen_json"]
            info["models"]["qwen-0.8b-json"] = {"model_id": qj.model_id, "backend": getattr(qj, "backend", f"torch-{qj.device}"), "dtype": qj.dtype, "load_s": qj.load_s}
        return info

    @app.get("/v1/datasets")
    def list_datasets() -> dict[str, Any]:
        out = []
        for ds_id, (label, path) in discover_datasets().items():
            count = sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
            out.append({"id": ds_id, "label": label, "count": count})
        return {"datasets": out}

    @app.get("/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        found = discover_datasets()
        if dataset_id not in found:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        label, path = found[dataset_id]
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {"id": dataset_id, "label": label, "rows": rows}

    @app.post("/v1/score")
    def score(req: ScoreRequest) -> dict[str, Any]:
        if req.model == "m2-qwen3.5-0.8b-lora-s42":
            if "m2_qwen_lora_s42" not in state_store:
                raise HTTPException(status_code=503, detail="m2-qwen3.5-0.8b-lora-s42 is not loaded")
            return state_store["m2_qwen_lora_s42"].score_decision(req.state, req.question, req.options, req.option_descriptions)
        if req.model == "kev-0.8b":
            if "kev" not in state_store:
                raise HTTPException(status_code=503, detail="kev-0.8b is not loaded")
            return state_store["kev"].score_unified(req.state, req.question, req.options, req.option_descriptions)
        if req.model == "qwen-0.8b-ll":
            if "qwen_ll" not in state_store:
                raise HTTPException(status_code=503, detail="qwen-0.8b-ll is not loaded")
            return state_store["qwen_ll"].score_loglik(req.state, req.question, req.options, req.option_descriptions)
        if req.model == "qwen-0.8b-json":
            if "qwen_json" not in state_store:
                raise HTTPException(status_code=503, detail="qwen-0.8b-json is not loaded")
            return state_store["qwen_json"].score_json(req.state, req.question, req.options, req.option_descriptions)
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    @app.post("/v1/compare")
    def compare(req: CompareRequest) -> dict[str, Any]:
        targets = req.models or [
            m for m, key in [
                ("m2-qwen3.5-0.8b-lora-s42", "m2_qwen_lora_s42"),
                ("kev-0.8b", "kev"),
                ("qwen-0.8b-ll", "qwen_ll"),
                ("qwen-0.8b-json", "qwen_json"),
            ]
            if key in state_store
        ]
        results = {
            m: score(ScoreRequest(model=m, state=req.state, question=req.question, options=req.options, option_descriptions=req.option_descriptions))
            for m in targets
        }
        return {"question": req.question, "options": req.options, "results": results}

    @app.post("/v1/frontier")
    def frontier_score(req: FrontierRequest) -> dict[str, Any]:
        try:
            return call_frontier_llm(req.base_url, req.model, req.api_key, req.state, req.question, req.options, req.option_descriptions)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise HTTPException(status_code=502, detail=f"Frontier API HTTP {e.code}: {body[:300]}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Frontier API error: {e}")

    @app.post("/v1/systemone")
    def systemone(req: SystemOnePayload) -> dict[str, Any]:
        if "kev" not in state_store:
            raise HTTPException(status_code=503, detail="kev-0.8b is not loaded")
        return state_store["kev"].score_systemone(req.state, req.questions)

    @app.get("/v1/nn_status")
    def nn_status() -> dict[str, Any]:
        return {
            "available": paths.CAR_BUILD_NN_NPZ.is_file(),
            "model": "nn-combo",
            "car_build_path": str(paths.CAR_BUILD_NN_NPZ),
            "driver_path": str(paths.DRIVER_NN_NPZ) if paths.DRIVER_NN_NPZ.is_file() else None,
        }

    @app.post("/v1/score_nn")
    def score_nn(req: NNScoreRequest) -> dict[str, Any]:
        if not paths.CAR_BUILD_NN_NPZ.is_file():
            raise HTTPException(status_code=404, detail="models/car_build_nn.npz not found")
        try:
            if "nn_adapter" not in state_store:
                state_store["nn_adapter"] = OptionalNNAdapter(paths.CAR_BUILD_NN_NPZ)
            raw = state_store["nn_adapter"].score_features(req.features)
            idx = int(raw["choice_index"])
            return {
                "model": "nn-combo",
                "model_id": "car_build_nn + jev_rl",
                "backend": "numpy-cpu",
                "choice_index": idx,
                "choice": req.options[idx] if 0 <= idx < len(req.options) else None,
                "probabilities": raw["probabilities"],
                "confidence": raw["confidence"],
                "parse_ok": True,
                "latency_ms": raw["latency_ms"],
                "input_tokens": 0,
                "output_tokens": 0,
                "state_truncated": False,
                "raw_output": None,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Tier 4 NN scoring failed: {exc}")

    @app.post("/v1/analyze")
    def analyze_batch(req: AnalyzeRequest) -> dict[str, Any]:
        raw_path = paths.REPORTS_DIR / "benchmark_raw.jsonl"
        sum_path = paths.REPORTS_DIR / "benchmark_summary.json"
        replaced = set(req.models)

        merged_records = list(req.records)
        merged_models: list[str] = []
        prev_summary: dict[str, Any] | None = None

        if req.merge_existing:
            if sum_path.is_file():
                try:
                    prev_summary = json.loads(sum_path.read_text(encoding="utf-8"))
                except Exception:
                    prev_summary = None
            if raw_path.is_file():
                kept_raw: list[dict[str, Any]] = []
                for ln in raw_path.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    row = json.loads(ln)
                    if row.get("model") not in replaced:
                        kept_raw.append(row)
                        m_name = row.get("model")
                        if m_name and m_name not in merged_models:
                            merged_models.append(m_name)
                merged_records = kept_raw + list(req.records)

        for m in req.models:
            if m not in merged_models:
                merged_models.append(m)

        paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as fh:
            for row in merged_records:
                fh.write(json.dumps(row) + "\n")

        summary = metrics_mod.build_summary(
            merged_records, merged_models, req.config,
            frontier_in_price=req.frontier_in_price,
            frontier_out_price=req.frontier_out_price,
        )

        if req.merge_existing and prev_summary:
            for m, m_data in (prev_summary.get("models") or {}).items():
                if m not in replaced and m not in summary.get("models", {}):
                    summary["models"][m] = m_data

        metrics_mod.write_summary_files(summary, paths.REPORTS_DIR)
        charts = charts_mod.generate_all_charts(summary, paths.CHARTS_DIR)
        return {"summary": summary, "charts": charts}

    @app.get("/v1/reports/latest")
    def latest_report() -> dict[str, Any]:
        summary_path = paths.REPORTS_DIR / "benchmark_summary.json"
        if not summary_path.is_file():
            return {"available": False, "summary": None, "charts": []}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        charts = (
            [{"filename": p.name, "title": p.stem.replace("_", " ").title()} for p in sorted(paths.CHARTS_DIR.glob("*.svg"))]
            if paths.CHARTS_DIR.is_dir() else []
        )
        return {"available": True, "summary": summary, "charts": charts}

    @app.get("/v1/reports/pace_v2")
    def get_pace_v2_report() -> dict[str, Any]:
        pace_path = paths.REPO_ROOT / "reports" / "pace_accuracy_summary.json"
        if not pace_path.is_file():
            return {"available": False, "report": None}
        return {"available": True, "report": json.loads(pace_path.read_text(encoding="utf-8"))}

    @app.post("/v1/reports/pace_v2/recompute")
    def recompute_pace_v2_report() -> dict[str, Any]:
        import importlib.util
        script_path = paths.REPO_ROOT / "scripts" / "eval_pace_accuracy.py"
        if not script_path.is_file():
            raise HTTPException(status_code=404, detail="scripts/eval_pace_accuracy.py not found")
        spec = importlib.util.spec_from_file_location("eval_pace_accuracy", script_path)
        if spec is None or spec.loader is None:
            raise HTTPException(status_code=500, detail="Failed to load eval_pace_accuracy.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        report = mod.evaluate(mod.DEFAULT_MATCHES, mod.DEFAULT_WINNERS, mod.DEFAULT_OUT)
        return {"available": True, "report": report}

    @app.get("/v1/reports/models_pace_v2")
    def get_models_pace_v2_report() -> dict[str, Any]:
        models_path = paths.REPO_ROOT / "reports" / "models_pace_v2_summary.json"
        if not models_path.is_file():
            return {"available": False, "report": None}
        return {"available": True, "report": json.loads(models_path.read_text(encoding="utf-8"))}

    @app.get("/v1/reports/race_engineer_v2")
    def get_race_engineer_v2_report() -> dict[str, Any]:
        re_path = paths.REPO_ROOT / "reports" / "race_engineer_v2_summary.json"
        if not re_path.is_file():
            return {"available": False, "report": None}
        return {"available": True, "report": json.loads(re_path.read_text(encoding="utf-8"))}

    @app.get("/v1/reports/track_rule_confirm_v2")
    def get_track_rule_confirm_v2_report() -> dict[str, Any]:
        cf_path = paths.REPO_ROOT / "reports" / "track_rule_confirm_v2.json"
        if not cf_path.is_file():
            return {"available": False, "report": None}
        return {"available": True, "report": json.loads(cf_path.read_text(encoding="utf-8"))}

    @app.get("/v1/reports/feedback_f1")
    def get_feedback_f1_report() -> dict[str, Any]:
        fb_v2 = paths.REPO_ROOT / "reports" / "feedback_f1_v2_summary.json"
        fb_v1 = paths.REPO_ROOT / "reports" / "feedback_f1_summary.json"
        fb_path = fb_v2 if fb_v2.is_file() else fb_v1
        if not fb_path.is_file():
            return {"available": False, "report": None}
        return {"available": True, "report": json.loads(fb_path.read_text(encoding="utf-8"))}

    @app.get("/v1/reports/m2_v5")
    def get_m2_v5_report() -> dict[str, Any]:
        ho_path = paths.REPO_ROOT / "reports" / "m2_v5" / "m2_v5_holdout_report.json"
        tr_path = paths.REPO_ROOT / "reports" / "m2_v5" / "m2_v5_train_summary.json"
        th_path = paths.REPO_ROOT / "reports" / "m2_v5" / "m2_v5_teacher_holdout_summary.json"
        if not ho_path.is_file():
            return {"available": False, "holdout": None, "train": None, "teacher": None}
        return {
            "available": True,
            "holdout": json.loads(ho_path.read_text(encoding="utf-8")),
            "train": json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.is_file() else None,
            "teacher": json.loads(th_path.read_text(encoding="utf-8")) if th_path.is_file() else None,
        }

    @app.get("/v1/reports/racing_engineer_v1")
    def get_racing_engineer_v1_report() -> dict[str, Any]:
        rv1_dir = paths.REPO_ROOT / "reports" / "racing_engineer_v1"
        th_path = rv1_dir / "thesis_analysis_v1.json"
        cl_path = rv1_dir / "closed_loop_eval_summary.json"
        tr_path = rv1_dir / "training_summary.json"
        m1_path = rv1_dir / "milestone1_diagnostics.json"
        curve_path = rv1_dir / "completion_by_time_curves.png"
        if not th_path.is_file() and not cl_path.is_file():
            return {"available": False}
        m1_summary = None
        if m1_path.is_file():
            m1_raw = json.loads(m1_path.read_text(encoding="utf-8"))
            ca = m1_raw.get("counterfactual_action_audit") or {}
            m1_summary = {
                "action_spec_hash": m1_raw.get("action_spec_hash"),
                "runner_contract_check": m1_raw.get("runner_contract_check"),
                "snapshot_replay_invariance": m1_raw.get("snapshot_replay_invariance"),
                "pit_snapshots_n": ca.get("pit_snapshots_n"),
                "pit_helped_n": ca.get("pit_helped_n"),
                "enable_pits_in_stage_b": ca.get("enable_pits_in_stage_b"),
                "pace_snapshots_n": ca.get("pace_snapshots_n"),
                "pace_mean_gain_vs_balanced_s": ca.get("pace_mean_gain_vs_balanced_s"),
            }
        return {
            "available": True,
            "thesis_analysis": json.loads(th_path.read_text(encoding="utf-8")) if th_path.is_file() else None,
            "closed_loop": json.loads(cl_path.read_text(encoding="utf-8")) if cl_path.is_file() else None,
            "training": json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.is_file() else None,
            "milestone1": m1_summary,
            "curve_available": curve_path.is_file(),
        }

    @app.get("/v1/reports/racing_engineer_v1/completion_curve.png")
    def get_racing_engineer_v1_curve() -> Response:
        curve_path = paths.REPO_ROOT / "reports" / "racing_engineer_v1" / "completion_by_time_curves.png"
        if not curve_path.is_file():
            raise HTTPException(status_code=404, detail="completion_by_time_curves.png not found")
        return Response(content=curve_path.read_bytes(), media_type="image/png")

    @app.get("/v1/reports/alignment_supplement")
    def get_alignment_supplement() -> dict[str, Any]:
        sup_dir = paths.REPO_ROOT / "reports" / "alignment_supplement"
        ml_path = sup_dir / "minilm_holdout_600_supplement.json"
        sb_path = sup_dir / "stage_b_alignment_45_summary.json"
        if not ml_path.is_file() and not sb_path.is_file():
            return {"available": False, "minilm_holdout_600": None, "stage_b_alignment_45": None}
        return {
            "available": True,
            "minilm_holdout_600": json.loads(ml_path.read_text(encoding="utf-8")) if ml_path.is_file() else None,
            "stage_b_alignment_45": json.loads(sb_path.read_text(encoding="utf-8")) if sb_path.is_file() else None,
        }

    @app.get("/v1/reports/csv")
    def download_report_csv() -> Response:
        csv_path = paths.REPORTS_DIR / "benchmark_summary.csv"
        if not csv_path.is_file():
            raise HTTPException(status_code=404, detail="No benchmark_summary.csv generated yet")
        return Response(content=csv_path.read_text(encoding="utf-8"), media_type="text/csv")

    @app.get("/v1/charts/{filename}")
    def get_chart(filename: str) -> Response:
        safe = Path(filename).name
        if not safe.endswith(".svg"):
            raise HTTPException(status_code=400, detail="Only .svg charts are served")
        target = paths.CHARTS_DIR / safe
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Chart not found: {safe}")
        return Response(content=target.read_text(encoding="utf-8"), media_type="image/svg+xml")

    @app.get("/v1/sim/replays")
    def list_sim_replays() -> dict[str, Any]:
        models = [
            {"id": "track_rule", "label": "track_rule (If-Else Rule + jev_rl_v7 · 596.1s · -8.5s vs driver_only)", "kind": "model"},
            {"id": "driver_only", "label": "driver_only (jev_rl_v7 Baseline · 603.8s)", "kind": "model"},
            {"id": "kev-0.8b", "label": "kev-0.8b (Best ML + jev_rl_v7 · 606.1s)", "kind": "model"},
            {"id": "qwen-0.8b-ll", "label": "qwen-0.8b-ll (+ jev_rl_v7 · 8/9 · 607.0s)", "kind": "model"},
            {"id": "nn-combo", "label": "nn-combo (Car-Build NN + jev_rl_v7 · 8/9 · 623.3s)", "kind": "model"},
            {"id": "if_else_baseline", "label": "if_else_baseline (+ jev_rl_v7 · 7/9 · 620.6s)", "kind": "model"},
            {"id": "qwen-0.8b-json", "label": "qwen-0.8b-json (+ jev_rl_v7 · 6/9 · 649.2s)", "kind": "model"},
            {"id": "self-jev-minilm", "label": "self-jev-minilm (+ jev_rl_v7 · 4/9 · 634.1s)", "kind": "model"},
            {"id": "sim_validation", "label": "Physics Check (Winner Controls vs AGX Winner)", "kind": "sim"},
        ]
        scenarios = [
            "crossover_saturated", "crossover_wetting", "crossover_dry",
            "highland_saturated", "highland_wetting", "highland_dry",
            "parkland_dry", "parkland_wetting", "parkland_saturated",
        ]
        winner_json = paths.REPO_ROOT / "mac_racing_sim" / "data" / "winner-builds-and-controls.json"
        races: list[str] = []
        if winner_json.is_file():
            try:
                doc = json.loads(winner_json.read_text(encoding="utf-8"))
                races = sorted((doc.get("winners") or {}).keys())
            except Exception:
                races = []
        return {
            "available": True,
            "default_model": "track_rule",
            "default_scenario": "crossover_saturated",
            "models": models,
            "scenarios": scenarios,
            "races": races,
        }

    @app.get("/v1/sim/replay_html", response_class=HTMLResponse)
    def get_sim_replay_html(
        model: str = "track_rule",
        race: str = "crossover_saturated",
        mode: Literal["guided", "open-loop"] = "guided",
        window_s: float = 19.0,
    ) -> HTMLResponse:
        import importlib.util

        if model != "sim_validation":
            rep_dir = paths.REPO_ROOT / "reports" / "sim_replays"
            safe_m = model.replace("/", "_")
            safe_sc = race.replace(":", "_").replace("/", "_")
            out_path = rep_dir / f"replay_{safe_m}_{safe_sc}.html"
            if not out_path.is_file():
                mod = state_store.get("model_replay_mod")
                if mod is None:
                    script_path = paths.REPO_ROOT / "scripts" / "render_model_replay.py"
                    if not script_path.is_file():
                        raise HTTPException(status_code=404, detail="scripts/render_model_replay.py not found")
                    spec = importlib.util.spec_from_file_location("self_jev_render_model_replay", script_path)
                    if spec is None or spec.loader is None:
                        raise HTTPException(status_code=500, detail="Failed to load render_model_replay.py")
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    state_store["model_replay_mod"] = mod
                mod.render_model_replay_html(model, race, out_path)
            return HTMLResponse(out_path.read_text(encoding="utf-8"))

        sim_dir = paths.REPO_ROOT / "mac_racing_sim"
        rep_dir = sim_dir / "reports"
        safe_name = race.replace(":", "_").replace("/", "_")
        out_path = rep_dir / f"replay_{safe_name}_{mode}.html"
        if not out_path.is_file():
            script_path = sim_dir / "render_replay.py"
            winner_json = sim_dir / "data" / "winner-builds-and-controls.json"
            if not script_path.is_file() or not winner_json.is_file():
                raise HTTPException(status_code=404, detail="mac_racing_sim replay generator or winner data not found")
            spec = importlib.util.spec_from_file_location("mac_sim_render_replay", script_path)
            if spec is None or spec.loader is None:
                raise HTTPException(status_code=500, detail="Failed to load mac_racing_sim/render_replay.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            doc = json.loads(winner_json.read_text(encoding="utf-8"))
            if race not in (doc.get("winners") or {}):
                raise HTTPException(status_code=404, detail=f"Unknown race: {race}")
            idx = {c: i for i, c in enumerate(doc["control_trace_columns"])}
            entry = doc["winners"][race]
            if mode == "open-loop":
                geom, frames, meta = mod.record_open_loop_frames(entry, idx, window_s=window_s)
            else:
                geom, frames, meta = mod.record_guided_4lap_frames(entry, idx)
            rep_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({"geom": geom, "frames": frames, "meta": meta}, separators=(",", ":"))
            html = mod.HTML_TEMPLATE.replace("__TITLE__", race).replace("__PAYLOAD__", payload)
            out_path.write_text(html, encoding="utf-8")
        return HTMLResponse(out_path.read_text(encoding="utf-8"))

    return app
