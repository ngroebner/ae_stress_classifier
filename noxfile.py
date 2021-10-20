import nox

@nox.session(python=["3.9", "3.8", "3.7", "3.6"], venv_backend="conda", venv_params=["--use-local"])
def test(session):
    """Add tests
    """
    session.install()
    session.run("pytest")

@nox.session(python=["3.9", "3.8", "3.7", "3.6"])
def lint(session):
    """Lint the code with flake8.
    """
    session.install("flake8")
    session.run("flake8", "")
