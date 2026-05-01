"""Evaluate the pipeline against sample_support_tickets.csv.

Compares predicted vs expected on status, request_type, product_area.
Prints per-row diffs and computes accuracy.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from indexer import build_index
from main import process_ticket
from schema import TicketInput

SAMPLE_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "support_tickets"
    / "sample_support_tickets.csv"
)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"


def load_sample_tickets() -> list[dict]:
    rows = []
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def run_eval():
    print("[eval] Loading index...")
    chunks_list, bm25, faiss_index, embed_model = build_index(DATA_DIR, INDEX_DIR)
    print(f"[eval] Index ready: {len(chunks_list)} chunks\n")

    rows = load_sample_tickets()
    print(f"[eval] {len(rows)} sample tickets loaded\n")

    total = 0
    correct_status = 0
    correct_request_type = 0
    correct_product_area = 0
    failures = []

    for i, row in enumerate(rows):
        ticket = TicketInput(
            index=i,
            issue=row.get("Issue", ""),
            subject=row.get("Subject", ""),
            company=row.get("Company", ""),
        )

        expected_status = row.get("Status", "").strip().lower()
        expected_request_type = row.get("Request Type", "").strip().lower()
        expected_product_area = row.get("Product Area", "").strip().lower()

        try:
            result = process_ticket(
                ticket, chunks_list, bm25, faiss_index, embed_model
            )
        except Exception as e:
            print(f"  ROW {i}: ERROR — {e}")
            failures.append(f"Row {i}: ERROR — {e}")
            total += 1
            continue

        pred_status = result.status.lower()
        pred_request_type = result.request_type.lower()
        pred_product_area = result.product_area.lower()

        total += 1
        s_match = pred_status == expected_status
        r_match = pred_request_type == expected_request_type
        p_match = (
            pred_product_area == expected_product_area
            or expected_product_area.startswith(pred_product_area)
            or pred_product_area.startswith(expected_product_area)
            or expected_product_area == ""
        )

        if s_match:
            correct_status += 1
        if r_match:
            correct_request_type += 1
        if p_match:
            correct_product_area += 1

        status_mark = "OK" if s_match else "MISS"
        rtype_mark = "OK" if r_match else "MISS"
        parea_mark = "OK" if p_match else "MISS"

        issue_preview = ticket.issue[:60].replace("\n", " ")
        print(f"  ROW {i}: '{issue_preview}...'")
        print(
            f"    status:  {pred_status:10} vs {expected_status:10} [{status_mark}]"
        )
        print(
            f"    req_type: {pred_request_type:15} vs {expected_request_type:15} [{rtype_mark}]"
        )
        print(
            f"    area:    {pred_product_area:25} vs {expected_product_area:25} [{parea_mark}]"
        )

        if not (s_match and r_match):
            fail_detail = (
                f"Row {i}: "
                f"status={pred_status}(exp:{expected_status}) "
                f"req_type={pred_request_type}(exp:{expected_request_type}) "
                f"area={pred_product_area}(exp:{expected_product_area})"
            )
            failures.append(fail_detail)
        print()

    print("=" * 60)
    print(f"Status accuracy:       {correct_status}/{total} ({correct_status/total:.0%})")
    print(
        f"Request type accuracy: {correct_request_type}/{total} ({correct_request_type/total:.0%})"
    )
    print(
        f"Product area accuracy: {correct_product_area}/{total} ({correct_product_area/total:.0%})"
    )
    print("=" * 60)

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")

        failures_path = Path(__file__).resolve().parent / "FAILURES.md"
        with open(failures_path, "w", encoding="utf-8") as fp:
            fp.write("# Evaluation Failures\n\n")
            for f in failures:
                fp.write(f"- {f}\n")
        print(f"\nFailures written to {failures_path}")
    else:
        print("\nAll rows passed!")


if __name__ == "__main__":
    run_eval()
