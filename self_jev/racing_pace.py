"""Racing pace contrasts from account J's finished highland matches.

The predeclared held-out metric is macro F1 on {table, grip}. A checkpoint may
drive a submission only when that score beats the majority-label macro F1 by
at least 0.10. Train text and held-out text must not share a 4-word window.
"""
from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

from self_jev.bundle import read_strategy
from self_jev.element_data import _ngrams
from self_jev.elements import write_archive
from self_jev.metrics import macro_f1, tempered_softmax

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "state" / "racing_pace_checkpoint.txt"
SOURCE = ROOT / "bases" / "live" / "racing-draft.zip"
METRIC = "macro_f1"
LABELS = ("table", "grip")
MARGIN = 0.10
CHOICE_MARGIN = 0.05

# Measured on the public recordings. table is the capped car. grip is a quicker
# legal finish, except where the quicker lap was deleted for an illegal command.
QUESTIONS = {
    "alpha": "Which recorded pace should the next cap follow?",
    "beta": "Select the pace supported by this sample.",
    "gamma": "Choose the pace for the coming highland entry.",
}
DESCRIPTIONS = {
    "alpha": {
        "table": "table cap that remained classified",
        "grip": "grip ceiling aligned with the winning total",
    },
    "beta": {
        "table": "table limit that kept its place",
        "grip": "grip ceiling used by first place",
    },
    "gamma": {
        "table": "table limit, skip deleted commands",
        "grip": "grip ceiling for a full-distance win",
    },
}

# (split, family, label, situation). Numbers do not cross splits.
ROWS = (
    ("train", "alpha", "grip", "Outing two. Capped qualifier 177.510 stayed classified and did not win. Four-lap winning total 634.283."),
    ("train", "alpha", "grip", "Second highland result. Cap 177.510 trailed. Winner covered four laps totaling 634.283."),
    ("train", "alpha", "grip", "Highland bout two. 177.510 was behind the classified winner. That winner four-lap sum is 634.283."),
    ("train", "alpha", "grip", "Race two. The cap posted 177.510 in qualifying and lost. Winning distance total 634.283."),
    ("train", "alpha", "table", "Outing two. Mark 132.715 drew an illegal command and was deleted. Cap 177.510 remained classified."),
    ("train", "alpha", "table", "Second highland result. 132.715 did not survive its illegal command. Classified cap stayed at 177.510."),
    ("train", "alpha", "table", "Highland bout two. Illegal command struck 132.715, so that lap vanished. 177.510 kept a classification."),
    ("train", "alpha", "table", "Race two. 132.715 ended on an illegal command. The surviving cap was 177.510."),
    ("select", "beta", "grip", "Opener sample. Limit qualifier 184.974. First-place qualifier 166.195, race sum 607.575."),
    ("select", "beta", "grip", "Opening highland. Capped race sum 693.400 lost to 607.575, the first-place four-lap sum."),
    ("select", "beta", "table", "Opener sample. 180.143 vanished after its illegal command. Limit qualifier 184.974 kept a place."),
    ("select", "beta", "table", "Opening highland. Illegal command ended 180.143. Place retained by 184.974."),
    ("test", "gamma", "grip", "Later forest card. Three-lap finisher 184.971 lost the distance to a winner whose lap mean was 158.571."),
    ("test", "gamma", "grip", "Follow-up card. Scheduled-distance winner qualifier 134.410. Slower finisher qualifier 184.971."),
    ("test", "gamma", "table", "Later forest card. 133.057 was cleared because the command was illegal, and it never became the winner."),
    ("test", "gamma", "table", "Follow-up card. An illegal command cleared 142.906 before the flag. A full-distance winner still stands."),
)

DRAFT_SITUATION = (
    "Coming highland entry. Trailing caps finished behind full-distance winners. "
    "Deleted illegal commands are not the winner. Take the quicker full-distance pace."
)


def examples() -> list[dict]:
    rows = []
    for index, (split, family, label, situation) in enumerate(ROWS):
        rng = random.Random(f"pace-{index}")
        order = list(LABELS)
        rng.shuffle(order)
        candidates = [
            {"id": f"opt{index}{key[:1]}", "key": key, "description": DESCRIPTIONS[family][key]}
            for key in order
        ]
        rows.append({
            "situation_id": f"{split}-{family}-{index}",
            "split": split,
            "render": family,
            "question": QUESTIONS[family],
            "situation": situation,
            "candidates": candidates,
            "label_key": label,
        })
    return rows


def model_text(example: dict) -> str:
    pieces = [example["question"], example["situation"]]
    for candidate in example["candidates"]:
        pieces.append(f"{candidate['id']}: {candidate['description']}")
    return " ".join(pieces)


def assert_holdout(rows: list[dict]) -> None:
    alpha, held = [], []
    for example in rows:
        text = model_text(example)
        if example["split"] == "train":
            alpha.append(text)
        else:
            held.append(text)
    shared = _shared(alpha, held + [DRAFT_SITUATION, QUESTIONS["gamma"], *DESCRIPTIONS["gamma"].values()])
    if shared:
        raise RuntimeError(f"train text shares a 4-word window with held-out text: {shared[:5]}")


def _shared(left: list[str], right: list[str]) -> list[str]:
    pool = set()
    for text in left:
        pool.update(_ngrams(text, 4))
    found = []
    seen = set()
    for text in right:
        for gram in _ngrams(text, 4):
            if gram in pool and gram not in seen:
                seen.add(gram)
                found.append(gram)
    return found


def train() -> Path:
    import torch
    from transformers import get_linear_schedule_with_warmup

    from self_jev.hashing import sha256_file
    from self_jev.model import load_encoder, pair_text, training_logits
    from self_jev.train import _epoch, _fit_temperature, _score_split

    if not torch.backends.mps.is_available():
        raise RuntimeError("Self-JEV training requires MPS")
    rows = examples()
    assert_holdout(rows)
    data_dir = ROOT / "data" / "racing_pace"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "examples.jsonl"
    with path.open("w") as handle:
        for example in rows:
            handle.write(json.dumps(example) + "\n")
    dataset_sha = sha256_file(path)
    run_dir = ROOT / "runs" / f"racing-pace-{dataset_sha[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "task": "racing-pace-contrast",
        "dataset_sha256": dataset_sha,
        "metric": METRIC,
        "labels": list(LABELS),
        "trust_rule": "held-out macro_f1 >= majority_macro_f1 + 0.10",
        "init": "ms-marco-MiniLM-L6-v2",
        "not": "Does not replace readiness or elements checkpoints.",
    }, indent=2) + "\n")
    device = torch.device("mps")
    torch.manual_seed(17)
    model, tokenizer = load_encoder(device)
    train_rows = [row for row in rows if row["split"] == "train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = ((len(train_rows) + 15) // 16) * 6
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), max(steps, 1))
    select_rows = [row for row in rows if row["split"] == "select"]
    best = None
    for epoch in range(6):
        loss = _epoch(model, tokenizer, optimizer, scheduler, train_rows, device, epoch, training_logits, pair_text, torch)
        scores = _score_split(model, tokenizer, select_rows, device, pair_text)
        metrics = _classify(select_rows, scores, temperature=1.0)
        print(f"epoch {epoch} loss {loss:.4f} select_macro_f1 {metrics['macro_f1']:.3f}")
        if best is None or metrics["macro_f1"] > best["macro_f1"] + 1e-12:
            best = {**metrics, "epoch": epoch}
            torch.save({
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "dataset_sha256": dataset_sha,
                "task": "racing-pace-contrast",
            }, run_dir / "best.pt")
    bundle = _load(run_dir, device)
    calibrate_rows = select_rows
    calibrate_scores = _score_split(bundle["model"], bundle["tokenizer"], calibrate_rows, device, pair_text)
    temperature = _fit_temperature(calibrate_rows, calibrate_scores)
    test_rows = [row for row in rows if row["split"] == "test"]
    test_scores = _score_split(bundle["model"], bundle["tokenizer"], test_rows, device, pair_text)
    report = _classify(test_rows, test_scores, temperature)
    majority = _majority([row["label_key"] for row in test_rows])
    majority_pred = [majority] * len(test_rows)
    truth = [row["label_key"] for row in test_rows]
    report["majority_label"] = majority
    report["majority_macro_f1"] = macro_f1(truth, majority_pred, LABELS)
    report["majority_accuracy"] = sum(left == right for left, right in zip(truth, majority_pred)) / len(truth)
    report["trusted"] = report["macro_f1"] >= report["majority_macro_f1"] + MARGIN
    report["temperature"] = temperature
    report["checkpoint"] = str(run_dir)
    (run_dir / "temperature.json").write_text(json.dumps({"temperature": temperature}) + "\n")
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "racing_pace.json").write_text(json.dumps(report, indent=2) + "\n")
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    POINTER.write_text(str(run_dir.relative_to(ROOT)) + "\n")
    print(
        f"held-out macro_f1 {report['macro_f1']:.3f} accuracy {report['accuracy']:.3f} "
        f"majority_macro_f1 {report['majority_macro_f1']:.3f} trusted {report['trusted']}"
    )
    print(run_dir)
    return run_dir


def _classify(rows: list[dict], scores: list[list[float]], temperature: float) -> dict:
    from self_jev.metrics import predicted_key

    truth, predicted = [], []
    for example, row_scores in zip(rows, scores):
        probabilities = tempered_softmax(row_scores, temperature)
        # predicted_key uses raw scores; temperature does not change argmax.
        truth.append(example["label_key"])
        predicted.append(predicted_key(probabilities, example["candidates"]))
    return {
        "accuracy": sum(left == right for left, right in zip(truth, predicted)) / len(truth),
        "macro_f1": macro_f1(truth, predicted, LABELS),
        "predicted": predicted,
        "truth": truth,
    }


def _majority(labels: list[str]) -> str:
    return max(LABELS, key=lambda label: (labels.count(label), label))


def _load(run_dir: Path, device):
    import torch
    from self_jev.model import load_encoder

    model, tokenizer = load_encoder(device)
    saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    model.to(device)
    model.eval()
    temperature = 1.0
    path = run_dir / "temperature.json"
    if path.exists():
        temperature = float(json.loads(path.read_text())["temperature"])
    return {"model": model, "tokenizer": tokenizer, "temperature": temperature, "device": device, "dir": run_dir}


def checkpoint_path() -> Path:
    pointer = POINTER.read_text().strip()
    path = ROOT / pointer
    if not (path / "best.pt").exists():
        raise FileNotFoundError(path)
    return path


def predict_pace(situation: str | None = None) -> dict:
    import torch
    from self_jev.model import pair_text, score_pairs

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    bundle = _load(checkpoint_path(), device)
    example = {
        "question": QUESTIONS["gamma"],
        "situation": situation or DRAFT_SITUATION,
        "candidates": [
            {"id": "table", "key": "table", "description": DESCRIPTIONS["gamma"]["table"]},
            {"id": "grip", "key": "grip", "description": DESCRIPTIONS["gamma"]["grip"]},
        ],
    }
    pairs = [pair_text(example["question"], example["situation"], candidate) for candidate in example["candidates"]]
    scores = score_pairs(bundle["model"], bundle["tokenizer"], pairs, bundle["device"])
    probabilities = tempered_softmax(scores, bundle["temperature"])
    order = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
    margin = probabilities[order[0]] - probabilities[order[1]]
    key = example["candidates"][order[0]]["key"]
    by_key = {candidate["key"]: probabilities[index] for index, candidate in enumerate(example["candidates"])}
    return {
        "choice": key,
        "probability": by_key[key],
        "margin": margin,
        "probabilities": by_key,
        "checkpoint": str(bundle["dir"]),
        "situation": example["situation"],
        "question": example["question"],
    }


def apply_pace(choice: str, dest: Path) -> dict:
    """Write a racing zip whose only new decision is the pace cap the model chose."""
    strategy = read_strategy(SOURCE)
    with zipfile.ZipFile(SOURCE) as bundle:
        agent = bundle.read("agent.py").decode()
        driver = bundle.read("formula_driver.py").decode()
    strategy = dict(strategy)
    strategy["pace_source"] = choice
    strategy["name"] = "sj4-pace"
    moved = choice != "table"
    strategy["thesis"] = (
        f"Self-JEV pace contrast chose {choice}. "
        "Highland qualifying under the speed table was 177.5s and 185.0s; "
        "the same-session winners who finished ran 134.4s and 166.2s. "
        "grip lifts the table ceiling so the grip loop, which the driver already trusts, sets the speed."
        if choice == "grip"
        else "Self-JEV pace contrast kept the table cap."
    )[:600]
    agent = _once(
        agent,
        '    "brake_gain",\n)',
        '    "brake_gain",\n    "pace_source",\n)',
    )
    driver = _once(
        driver,
        "        speeds = np.minimum(speeds, (lower * (1 - f) + upper * f) * cold_load)\n        wear = max(c['tyre_wear'])",
        "        speeds = np.minimum(speeds, (lower * (1 - f) + upper * f) * cold_load)\n"
        "        if self.ctrl.get('pace_source') == 'grip' and self.capabilities:\n"
        "            speeds = np.maximum(speeds, 90.0)\n"
        "        wear = max(c['tyre_wear'])",
    )
    driver = _once(
        driver,
        "        self.gg_active = bool(self.capabilities) and self.o['phase'] == 'race'",
        "        self.gg_active = bool(self.capabilities) and (self.o['phase'] == 'race' or self.ctrl.get('pace_source') == 'grip')",
    )
    digest = write_archive(SOURCE, "racing", strategy, agent, dest, driver)
    return {"strategy": strategy, "sha256": digest, "archive": str(dest), "moved": moved}


def draft_racing() -> dict:
    from self_jev.check import check_bundle

    decision = predict_pace()
    source = "model" if decision["margin"] >= CHOICE_MARGIN else "fallback"
    choice = decision["choice"] if source == "model" else "table"
    if source == "fallback":
        why = f"Pace model top key {decision['choice']} at {decision['probability']:.3f}, margin {decision['margin']:.3f}. The margin is below 0.05, so this is a fallback."
    else:
        why = f"Pace model chose {choice} at {decision['probability']:.3f}, margin {decision['margin']:.3f}."
    dest = ROOT / "submissions" / "racing" / "sj4-pace.zip"
    built = apply_pace(choice, dest)
    report_path = ROOT / "reports" / "racing_pace.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    critique = {
        "game": "racing",
        "name": "sj4-pace",
        "submitted": False,
        "base_name": "wet-front-sliding-steer-rate-075",
        "source_archive": str(SOURCE),
        "archive": str(dest),
        "sha256": built["sha256"],
        "checkpoint": decision["checkpoint"],
        "metric": METRIC,
        "trusted": report.get("trusted"),
        "held_out_macro_f1": report.get("macro_f1"),
        "majority_macro_f1": report.get("majority_macro_f1"),
        "elements": [{
            "id": "pace_source",
            "old": "table",
            "new": choice,
            "choice": decision["choice"],
            "probability": decision["probability"],
            "margin": decision["margin"],
            "probabilities": decision["probabilities"],
            "source": source,
            "changed": choice != "table",
            "why": why,
        }],
        "changed": ["pace_source"] if choice != "table" else [],
        "smoke": check_bundle(dest, "racing"),
    }
    dest.with_suffix(".json").write_text(json.dumps(critique, ensure_ascii=False, indent=2) + "\n")
    print(f"racing: sj4-pace {critique['smoke']} choice {choice} source {source} p {decision['probability']:.3f}")
    return critique


def _once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"pace patch missed a unique site: {old[:80]}")
    return text.replace(old, new, 1)


def main() -> None:
    train()
    draft_racing()
