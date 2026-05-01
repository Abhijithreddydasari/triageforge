"""TriageForge — support ticket triage agent.

Terminal-based CLI. Supports batch processing, single-ticket mode, and health checks.

Usage:
    python main.py                                         # batch mode (default)
    python main.py --ticket "I lost my Visa card"          # single ticket
    python main.py --interactive                           # interactive REPL
    python main.py --status                                # health check
    python main.py --force-reindex                         # rebuild index
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decide import should_force_escalate
from indexer import build_index
from llm import call_llm
from postprocess import postprocess
from preprocess import preprocess
from prompts import SYSTEM_PROMPT, build_user_prompt, format_chunks_for_prompt
from retriever import retrieve
from schema import TicketInput, TicketResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"

OUTPUT_FIELDS = [
    "Issue",
    "Subject",
    "Company",
    "Status",
    "Product Area",
    "Request Type",
    "Response",
    "Justification",
]


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def validate_config() -> list[str]:
    """Check that all required configuration is present. Returns list of issues."""
    issues = []
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            issues.append(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and add it to .env"
            )
        elif not key.startswith("gsk_"):
            issues.append(
                "GROQ_API_KEY does not look valid (should start with 'gsk_'). "
                "Check your .env file."
            )
    elif provider == "huggingface":
        key = os.environ.get("HF_TOKEN", "").strip()
        if not key:
            issues.append(
                "HF_TOKEN is not set. Get a token at https://huggingface.co/settings/tokens "
                "and add it to .env"
            )
    else:
        issues.append(
            f"Unknown LLM_PROVIDER='{provider}'. Must be 'groq' or 'huggingface'."
        )

    if not DATA_DIR.exists():
        issues.append(f"Data directory not found at {DATA_DIR.relative_to(DATA_DIR.parent.parent)}")

    return issues


def run_status_check() -> None:
    """Run a health check: validate config, check index, test API connectivity."""
    print("=" * 50)
    print("TriageForge Health Check")
    print("=" * 50)

    # Config
    print("\n[1/4] Configuration...")
    issues = validate_config()
    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
    else:
        provider = os.environ.get("LLM_PROVIDER", "groq")
        model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
        print(f"  OK: Provider={provider}, Model={model}")

    # Data
    print("\n[2/4] Corpus data...")
    if DATA_DIR.exists():
        md_count = len(list(DATA_DIR.rglob("*.md")))
        print(f"  OK: {md_count} markdown files in data/")
    else:
        print("  ERROR: data/ directory not found")

    # Index
    print("\n[3/4] Index cache...")
    hash_file = INDEX_DIR / "corpus_hash.txt"
    if hash_file.exists() and (INDEX_DIR / "chunks.pkl").exists():
        print(f"  OK: Index cached at data/index/")
    else:
        print("  WARN: Index not built yet (will build on first run)")

    # API connectivity
    print("\n[4/4] API connectivity...")
    if issues:
        print("  SKIP: Cannot test API (config issues above)")
    else:
        try:
            from llm import call_llm_raw
            result = call_llm_raw(
                system="Reply with exactly: ok",
                user="ping",
            )
            if result and len(result.strip()) > 0:
                print(f"  OK: API responding (got: '{result.strip()[:20]}')")
            else:
                print("  ERROR: API returned empty response")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 50)


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
                    "Issue": r.issue,
                    "Subject": r.subject,
                    "Company": r.company,
                    "Status": r.status.capitalize(),
                    "Product Area": r.product_area,
                    "Request Type": r.request_type,
                    "Response": r.response,
                    "Justification": r.justification,
                }
            )


def run_single_ticket(issue: str, company: str = "") -> None:
    """Process a single ticket and print the result."""
    issues = validate_config()
    if issues:
        for i in issues:
            print(f"ERROR: {i}")
        sys.exit(1)

    print("[agent] Loading index...")
    chunks_list, bm25, faiss_index, embed_model = build_index(DATA_DIR, INDEX_DIR)

    ticket = TicketInput(index=0, issue=issue, subject="", company=company)
    print(f"[agent] Processing: '{issue[:60]}...'")
    result = process_ticket(ticket, chunks_list, bm25, faiss_index, embed_model)

    print("\n" + "=" * 50)
    print(f"Status:       {result.status}")
    print(f"Request Type: {result.request_type}")
    print(f"Product Area: {result.product_area}")
    print(f"Response:     {result.response[:200]}...")
    print(f"Justification:{result.justification}")
    print("=" * 50)


def run_interactive(chunks_list, bm25, faiss_index, embed_model) -> None:
    """Interactive REPL mode for demo purposes."""
    print("\n" + "=" * 50)
    print("TriageForge Interactive Mode")
    print("Type a support ticket issue. Type 'quit' to exit.")
    print("Prefix with [Company] to set company, e.g. [HackerRank] my issue")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("ticket> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        company = ""
        issue = user_input
        if user_input.startswith("["):
            bracket_end = user_input.find("]")
            if bracket_end > 0:
                company = user_input[1:bracket_end].strip()
                issue = user_input[bracket_end + 1:].strip()

        ticket = TicketInput(index=0, issue=issue, subject="", company=company)

        try:
            result = process_ticket(ticket, chunks_list, bm25, faiss_index, embed_model)
            print(f"\n  Status:       {result.status}")
            print(f"  Request Type: {result.request_type}")
            print(f"  Product Area: {result.product_area}")
            print(f"  Response:     {result.response}")
            print(f"  Justification:{result.justification}\n")
        except Exception as e:
            print(f"\n  ERROR: {e}\n")


def run_batch(input_path: Path, output_path: Path, force_reindex: bool) -> None:
    """Batch mode: process CSV and write output."""
    from tqdm import tqdm

    issues = validate_config()
    if issues:
        for i in issues:
            print(f"ERROR: {i}")
        sys.exit(1)

    print("[agent] Building / loading index...")
    chunks_list, bm25, faiss_index, embed_model = build_index(
        DATA_DIR, INDEX_DIR, force=force_reindex
    )
    print(f"[agent] Index ready: {len(chunks_list)} chunks")

    tickets = load_tickets(input_path)
    print(f"[agent] Loaded {len(tickets)} tickets from {input_path.name}")

    results: list[TicketResult] = []
    for i, ticket in enumerate(tqdm(tickets, desc="Processing tickets")):
        if i > 0:
            time.sleep(2)
        try:
            result = process_ticket(
                ticket, chunks_list, bm25, faiss_index, embed_model
            )
            results.append(result)
        except Exception as e:
            print(f"\n[agent] ERROR on ticket {ticket.index}: {e}")
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

    write_results(results, output_path)
    print(f"\n[agent] Done. {len(results)} results written to {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TriageForge — support ticket triage agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py                                    Batch mode (process CSV)
  python main.py --ticket "I lost my card"          Single ticket
  python main.py --interactive                      Interactive REPL
  python main.py --status                           Health check
  python main.py --force-reindex                    Rebuild corpus index
""",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "support_tickets"
        / "support_tickets.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "support_tickets"
        / "output.csv",
    )
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--ticket", type=str, help="Process a single ticket")
    parser.add_argument("--company", type=str, default="", help="Company for --ticket mode")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--status", action="store_true", help="Run health check")
    args = parser.parse_args()

    if args.status:
        run_status_check()
        return

    if args.ticket:
        run_single_ticket(args.ticket, args.company)
        return

    if args.interactive:
        issues = validate_config()
        if issues:
            for i in issues:
                print(f"ERROR: {i}")
            sys.exit(1)
        print("[agent] Loading index...")
        chunks_list, bm25, faiss_index, embed_model = build_index(
            DATA_DIR, INDEX_DIR, force=args.force_reindex
        )
        run_interactive(chunks_list, bm25, faiss_index, embed_model)
        return

    run_batch(args.input, args.output, args.force_reindex)


if __name__ == "__main__":
    main()
