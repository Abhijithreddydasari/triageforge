"""TriageForge — support ticket triage agent.

CLI entry point. Reads tickets from CSV, runs the triage pipeline,
writes results to output CSV incrementally.

Usage:
    python main.py --input ../support_tickets/support_tickets.csv \
                   --output ../support_tickets/output.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from decide import should_force_escalate
from indexer import build_index
from llm import call_llm
from postprocess import postprocess
from preprocess import preprocess
from prompts import SYSTEM_PROMPT, build_user_prompt, format_chunks_for_prompt
from retriever import retrieve
from schema import TicketInput, TicketResult

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"

OUTPUT_FIELDS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


def load_tickets(csv_path: Path) -> list[TicketInput]:
    tickets = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tickets.append(
                TicketInput(
                    index=i,
                    issue=row.get("Issue", row.get("issue", "")),
                    subject=row.get("Subject", row.get("subject", "")),
                    company=row.get("Company", row.get("company", "")),
                )
            )
    return tickets


def process_ticket(
    ticket: TicketInput,
    chunks_list,
    bm25,
    faiss_index,
    embed_model,
) -> TicketResult:
    """Run the full pipeline for a single ticket."""
    prep = preprocess(ticket.issue, ticket.subject)

    top_chunks = retrieve(
        query=prep.query,
        chunks=chunks_list,
        bm25=bm25,
        faiss_index=faiss_index,
        embed_model=embed_model,
        company=ticket.company,
        k=6,
    )

    escalation_reason = should_force_escalate(top_chunks, prep.language)

    chunks_text = format_chunks_for_prompt(top_chunks)
    user_prompt = build_user_prompt(
        issue=prep.clean_issue,
        subject=prep.clean_subject,
        company=ticket.company,
        chunks_text=chunks_text,
        escalation_hint=escalation_reason,
    )

    llm_resp = call_llm(SYSTEM_PROMPT, user_prompt)

    result = postprocess(
        llm_resp,
        top_chunks,
        DATA_DIR,
        forced_escalation=escalation_reason is not None,
    )

    result.issue = ticket.issue
    result.subject = ticket.subject
    result.company = ticket.company or ""

    return result


def write_results(results: list[TicketResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "issue": r.issue,
                    "subject": r.subject,
                    "company": r.company,
                    "response": r.response,
                    "product_area": r.product_area,
                    "status": r.status,
                    "request_type": r.request_type,
                    "justification": r.justification,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="TriageForge support triage agent")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "support_tickets"
        / "support_tickets.csv",
        help="Input CSV with tickets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "support_tickets"
        / "output.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force rebuild of corpus index",
    )
    args = parser.parse_args()

    print("[main] Building / loading index...")
    chunks_list, bm25, faiss_index, embed_model = build_index(
        DATA_DIR, INDEX_DIR, force=args.force_reindex
    )
    print(f"[main] Index ready: {len(chunks_list)} chunks")

    tickets = load_tickets(args.input)
    print(f"[main] Loaded {len(tickets)} tickets from {args.input}")

    results: list[TicketResult] = []
    for ticket in tqdm(tickets, desc="Processing tickets"):
        try:
            result = process_ticket(
                ticket, chunks_list, bm25, faiss_index, embed_model
            )
            results.append(result)
        except Exception as e:
            print(f"\n[main] ERROR on ticket {ticket.index}: {e}")
            results.append(
                TicketResult(
                    issue=ticket.issue,
                    subject=ticket.subject,
                    company=ticket.company or "",
                    response="An error occurred processing this ticket. Escalating to human agent.",
                    product_area="general",
                    status="escalated",
                    request_type="product_issue",
                    justification=f"Processing error: {e}",
                )
            )

    write_results(results, args.output)
    print(f"\n[main] Done. {len(results)} results written to {args.output}")


if __name__ == "__main__":
    main()
