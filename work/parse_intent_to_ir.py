from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False

ROOT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOLDER / "src"))

from rages_parser import ParserConfig, parse_batch, parse_intent_to_ir


def _read_inputs(args: argparse.Namespace) -> List[str]:
    inputs: List[str] = []
    if args.text:
        inputs.extend(args.text)
    if args.input_file:
        path = Path(args.input_file)
        inputs.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not inputs:
        raise ValueError("Provide --text or --input-file.")
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse free-form mission text into RAGES IR.")
    parser.add_argument("--text", action="append", help="Mission text to parse. Can be repeated.")
    parser.add_argument("--input-file", help="Path to a newline-delimited text file.")
    parser.add_argument("--backend", choices=("openai", "heuristic"), default="openai")
    parser.add_argument("--model", default=ParserConfig.model)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    load_dotenv()
    config = ParserConfig(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    texts = _read_inputs(args)
    if len(texts) == 1:
        payload: object = parse_intent_to_ir(texts[0], config=config).to_dict()
    else:
        payload = [
            {"text": text, "parse": result.to_dict()}
            for text, result in zip(texts, parse_batch(texts, config=config))
        ]

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
