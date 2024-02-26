"""Nox sessions."""

import nox
from nox import session
import sys

package = "koinapy"
python_versions = ["3.8", "3.9", "3.10"]
nox.options.sessions = ("tests",)


@session(python=python_versions)
@nox.parametrize("trclient", ["2.23", "2.42"])
def tests(session, trclient) -> None:
    """Runtime type checking using Typeguard."""
    session.install(f"tritonclient[grpc]=={trclient}")
    session.install("pytest")
    session.install("requests")
    session.install(".")
    session.run("pytest", *session.posargs)
