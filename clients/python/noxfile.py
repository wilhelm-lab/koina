"""Nox sessions."""

import nox
from nox import session
import sys

package = "koinapy"
python_versions = ["3.8", "3.9", "3.10"]
nox.options.sessions = ("tests",)


@session(python=python_versions)
# @nox.parametrize("trclient", [f"2.{i}" for i in range(23, 43) if (i != 41 or i != 37)])
@nox.parametrize("trclient", [f"2.{i}" for i in range(23, 43)])
def tests(session, trclient) -> None:
    """Runtime type checking using Typeguard."""
    session.install(f"tritonclient[grpc]=={trclient}")
    session.install("pytest")
    session.install("requests")
    session.install(".")
    session.run("pytest", *session.posargs)
