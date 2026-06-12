from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False

ROOT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOLDER / "src"))
sys.path.insert(0, str(ROOT_FOLDER / "work"))

from parser_eval_data import HANDWRITTEN_EXAMPLES, build_gold_dataset
from rages_parser import ParserConfig, ParseResult, score_slots
from rages_parser import parse_intent_to_ir
from rages_sampling import IR


Counts = Dict[str, int]


def _empty_counts() -> Counts:
    return {"tp": 0, "fp": 0, "fn": 0}


def _add_counts(dst: Counts, src: Counts) -> None:
    dst["tp"] += int(src.get("tp", 0))
    dst["fp"] += int(src.get("fp", 0))
    dst["fn"] += int(src.get("fn", 0))


def _prf(counts: Counts) -> Dict[str, float]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _dataset(n: int, seed: int, include_handwritten: bool) -> List[Tuple[str, Optional[IR]]]:
    rows: List[Tuple[str, Optional[IR]]] = [(text, ir) for text, ir in build_gold_dataset(n, seed)]
    if include_handwritten:
        rows.extend(HANDWRITTEN_EXAMPLES)
    return rows


def _aggregate_slot_scores(pairs: Iterable[Tuple[Optional[IR], IR]]) -> Dict[str, object]:
    groups: Dict[str, Counts] = defaultdict(_empty_counts)
    overall = _empty_counts()
    for pred_ir, gold_ir in pairs:
        scored = score_slots(pred_ir, gold_ir)
        for group, counts in scored["groups"].items():
            _add_counts(groups[str(group)], counts)  # type: ignore[arg-type]
        _add_counts(overall, scored["overall"])  # type: ignore[arg-type]
    return {
        "groups": {
            group: {**counts, **_prf(counts)}
            for group, counts in sorted(groups.items())
        },
        "overall": {**overall, **_prf(overall)},
    }


def _format_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def _print_summary(metrics: Dict[str, object]) -> None:
    print("Parser benchmark")
    print(f"backend: {metrics['backend']}  model: {metrics['model']}")
    print(f"examples: {metrics['n_examples']}  valid: {metrics['n_valid']}  malformed: {metrics['n_malformed']}")
    print()

    print("Slot F1 by group")
    print("group        precision   recall      f1       tp   fp   fn")
    slot_metrics = metrics["slot_metrics"]  # type: ignore[index]
    for group, vals in slot_metrics["groups"].items():  # type: ignore[index, union-attr]
        print(
            f"{group:<10}  {_format_pct(vals['precision'])}  {_format_pct(vals['recall'])}  "
            f"{_format_pct(vals['f1'])}  {vals['tp']:4d} {vals['fp']:4d} {vals['fn']:4d}"
        )
    overall = slot_metrics["overall"]  # type: ignore[index, union-attr]
    print(
        f"{'overall':<10}  {_format_pct(overall['precision'])}  {_format_pct(overall['recall'])}  "
        f"{_format_pct(overall['f1'])}  {overall['tp']:4d} {overall['fp']:4d} {overall['fn']:4d}"
    )
    print()

    print("Core metrics")
    print(f"command_class_accuracy:        {_format_pct(metrics['command_class_accuracy'])}")
    print(f"abstain_precision:             {_format_pct(metrics['abstain_precision'])}")
    print(f"abstain_recall:                {_format_pct(metrics['abstain_recall'])}")
    print(f"malformed_rejection_rate:      {_format_pct(metrics['malformed_rejection_rate'])}")
    print()

    print("Per task class")
    print("task_class   n    class_acc   slot_f1")
    per_task = metrics["per_task_class"]  # type: ignore[index]
    for task_class, vals in sorted(per_task.items()):  # type: ignore[union-attr]
        print(
            f"{task_class:<10} {vals['n']:3d}  {_format_pct(vals['command_class_accuracy'])}  "
            f"{_format_pct(vals['slot_metrics']['overall']['f1'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text-to-IR parser behavior.")
    parser.add_argument("--backend", choices=("openai", "heuristic"), default="heuristic")
    parser.add_argument("--model", default=ParserConfig.model)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--n", type=int, default=100, help="Number of sampled templated examples.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-handwritten", action="store_true")
    parser.add_argument("--out", help="Optional JSON report path.")
    args = parser.parse_args()

    load_dotenv()
    config = ParserConfig(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    rows = _dataset(args.n, args.seed, include_handwritten=not args.no_handwritten)

    result_rows: List[Dict[str, object]] = []
    valid_slot_pairs: List[Tuple[Optional[IR], IR]] = []
    per_task_pairs: Dict[str, List[Tuple[Optional[IR], IR]]] = defaultdict(list)
    per_task_class_total: Dict[str, int] = defaultdict(int)
    per_task_class_correct: Dict[str, int] = defaultdict(int)

    valid_total = 0
    malformed_total = 0
    class_correct = 0
    abstain_counts = _empty_counts()

    for text, gold_ir in rows:
        result: ParseResult = parse_intent_to_ir(text, config=config)
        pred_abstain = bool(result.abstained or result.ir is None)
        gold_abstain = gold_ir is None

        if gold_abstain:
            malformed_total += 1
        else:
            valid_total += 1
            assert gold_ir is not None
            valid_slot_pairs.append((result.ir, gold_ir))
            per_task_pairs[gold_ir.g.task_class].append((result.ir, gold_ir))
            per_task_class_total[gold_ir.g.task_class] += 1
            if result.ir is not None and result.ir.g.task_class == gold_ir.g.task_class:
                class_correct += 1
                per_task_class_correct[gold_ir.g.task_class] += 1

        if pred_abstain and gold_abstain:
            abstain_counts["tp"] += 1
        elif pred_abstain and not gold_abstain:
            abstain_counts["fp"] += 1
        elif (not pred_abstain) and gold_abstain:
            abstain_counts["fn"] += 1

        result_rows.append(
            {
                "text": text,
                "gold_ir": None if gold_ir is None else gold_ir.to_dict(),
                "parse": result.to_dict(),
            }
        )

    slot_metrics = _aggregate_slot_scores(valid_slot_pairs)
    abstain_prf = _prf(abstain_counts)
    malformed_rejection_rate = (
        float(abstain_counts["tp"] / malformed_total) if malformed_total else 0.0
    )
    per_task_class = {}
    for task_class, pairs in sorted(per_task_pairs.items()):
        total = per_task_class_total[task_class]
        per_task_class[task_class] = {
            "n": total,
            "command_class_accuracy": (
                float(per_task_class_correct[task_class] / total) if total else 0.0
            ),
            "slot_metrics": _aggregate_slot_scores(pairs),
        }

    metrics: Dict[str, object] = {
        "backend": args.backend,
        "model": args.model,
        "n_examples": len(rows),
        "n_valid": valid_total,
        "n_malformed": malformed_total,
        "slot_metrics": slot_metrics,
        "command_class_accuracy": float(class_correct / valid_total) if valid_total else 0.0,
        "abstain_precision": abstain_prf["precision"],
        "abstain_recall": abstain_prf["recall"],
        "malformed_rejection_rate": malformed_rejection_rate,
        "abstain_counts": abstain_counts,
        "per_task_class": per_task_class,
    }
    report = {"metrics": metrics, "examples": result_rows}
    _print_summary(metrics)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
