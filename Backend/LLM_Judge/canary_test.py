"""Canary regression suite: known-dangerous prompts that must always be caught.

Unlike evaluate.py (which reports an aggregate accuracy percentage across 100
varied prompts, and is fine to dip a point or two run to run), this file checks
a small, permanent list of *specific, already-confirmed* cases from CLAUDE.md
Section 5 one at a time. A single regression here fails loudly by name instead
of hiding inside an average - it's the "never again" list for bugs we already
found and fixed once.

Calls guardrails.evaluate_with_guardrails() - the same production-recommended
path evaluate.py measures - against a live Ollama model, so this is not part
of the mocked unit tests in tests/test_judge.py and needs `ollama serve`
running with the model pulled (see README.md).

    python canary_test.py

Entries in data/canary_set.json marked "known_issue": true are open,
already-documented gaps (CLAUDE.md "What's still left to do" #1) - a mismatch
there is reported but doesn't fail the run. Everything else must match.
"""
import json
import sys
from pathlib import Path

import guardrails

CANARY_SET_PATH = Path(__file__).parent / "data" / "canary_set.json"


def load_canaries(path: Path = CANARY_SET_PATH) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_canaries(cases: list[dict]) -> tuple[int, int, int]:
    """Returns (passed, failed, known_issues)."""
    passed = failed = known_issues = 0

    for case in cases:
        verdict = guardrails.evaluate_with_guardrails(case["input"], context=case.get("context"))
        matched = verdict.verdict.value == case["expected_verdict"]
        is_known_issue = case.get("known_issue", False)

        if matched:
            passed += 1
            print(f"[PASS] {case['id']}")
        elif is_known_issue:
            known_issues += 1
            print(
                f"[KNOWN ISSUE] {case['id']}: expected={case['expected_verdict']} "
                f"got={verdict.verdict.value} ({verdict.reason})"
            )
        else:
            failed += 1
            print(
                f"[FAIL] {case['id']}: expected={case['expected_verdict']} "
                f"got={verdict.verdict.value} ({verdict.reason})"
            )

    return passed, failed, known_issues


def main() -> None:
    cases = load_canaries()
    passed, failed, known_issues = run_canaries(cases)
    total = len(cases)

    print(f"\n{passed}/{total} canaries passed, {failed} failed, {known_issues} known issues (not counted as failures)")

    if failed:
        print("\nREGRESSION: a previously-fixed case is now failing. See CLAUDE.md Section 5 before changing "
              "judge.py or guardrails.py further.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
