"""Start a service component or manage its API tokens."""

from __future__ import annotations

import argparse
import getpass
import re

import uvicorn

from atc_service.api import create_app
from atc_service.config import Settings
from atc_service.logs import configure_logging
from atc_service.store import Store


def main() -> int:
    """Run an API, worker, or operator command.

    Returns
    -------
    int
        Command exit status.
    """
    parser = argparse.ArgumentParser(prog="atc-service")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("api", help="Start the HTTP API.")
    commands.add_parser("worker", help="Process the Docker job queue.")
    issue = commands.add_parser(
        "issue-token", help="Provision an account and issue a token."
    )
    issue.add_argument("username")
    commands.add_parser(
        "revoke-token", help="Revoke a token entered at a private prompt."
    )
    args = parser.parse_args()
    if args.command == "issue-token" and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.username
    ):
        parser.error(
            "Account name must use 1 to 64 letters, digits, dots, dashes, or underscores."
        )

    settings = Settings.from_env()
    store = Store(settings)
    store.migrate()
    if args.command == "issue-token":
        print(store.issue_token(args.username))
    elif args.command == "revoke-token":
        token = getpass.getpass("Token to revoke: ")
        if not store.revoke_token(token):
            parser.error("Token not found.")
    elif args.command == "api":
        configure_logging("api", settings.data_dir / "service-logs")
        uvicorn.run(create_app(settings), host="0.0.0.0", port=8080, access_log=False)
    else:
        from atc_service.worker import run_worker

        configure_logging("worker", settings.data_dir / "service-logs")
        run_worker(settings)
    return 0
