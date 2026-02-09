#!/usr/bin/env python3

import argparse
import json
import readline
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a Hugging Face IMDb sentiment model to stdin text."
    )
    parser.add_argument(
        "--model",
        default="textattack/roberta-base-imdb",
        help="Hugging Face model ID (default: textattack/roberta-base-imdb).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device like "cpu", "cuda", or "auto" (default).',
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum input token length after truncation (default: 512).",
    )
    parser.add_argument(
        "--all-scores",
        action="store_true",
        help="Include probability for each label in the output JSON.",
    )
    return parser.parse_args()


def _configure_readline() -> None:
    if readline is None:
        return
    readline.parse_and_bind("tab: complete")


def iter_stdin_lines():
    if sys.stdin.isatty():
        print(
            "Interactive mode: enter one review per line. Use up-arrow for history. "
            "Press Ctrl+C to stop.",
            file=sys.stderr,
        )
        _configure_readline()
        while True:
            text = input("> ").strip()
            if text:
                yield text
        return

    for raw_line in sys.stdin:
        text = raw_line.strip()
        if text:
            yield text


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", None) or {}
    seen_input = False
    try:
        for text in iter_stdin_lines():
            seen_input = True

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.inference_mode():
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[0]

            pred_idx = int(torch.argmax(probs).item())
            pred_label = id2label.get(pred_idx, f"LABEL_{pred_idx}")

            result = {
                "model": args.model,
                "label": pred_label,
                "score": float(probs[pred_idx].item()),
            }

            if args.all_scores:
                result["all_scores"] = {
                    id2label.get(i, f"LABEL_{i}"): float(prob.item())
                    for i, prob in enumerate(probs)
                }

            print(json.dumps(result), flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping inference.", file=sys.stderr)

    if not seen_input and not sys.stdin.isatty():
        raise SystemExit("stdin was empty.")


if __name__ == "__main__":
    main()
