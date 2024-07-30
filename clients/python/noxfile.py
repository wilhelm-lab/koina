"""Nox sessions."""

import nox
from nox import session
import sys

package = "koinapy"
python_versions = ["3.8", "3.9", "3.10", "3.11", "3.12"]
nox.options.sessions = ("tests",)


@session(python=python_versions)
def tests(session) -> None:
    """Runtime type checking using Typeguard."""
    session.install("pytest")
    session.install("requests")
    session.install(".")
    session.run("pytest", *session.posargs)
