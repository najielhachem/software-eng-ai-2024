# tasks.py
from invoke.context import Context
from invoke.tasks import task


@task
def test(ctx: Context) -> None:
    """Run all tests with pytest."""
    ctx.run("poetry run pytest tests/", pty=True)


@task
def lint(ctx: Context, fix: bool = False) -> None:
    """Run ruff to lint and format the code."""
    options = "--fix" if fix else ""
    ctx.run(f"poetry run ruff check src/ tasks.py tests/ {options}", pty=True)


@task
def format(ctx: Context) -> None:
    """Run ruff to lint and format the code."""
    ctx.run("poetry run ruff format src/ tasks.py tests/", pty=True)


@task
def type(ctx: Context) -> None:
    """Run ruff to lint and format the code."""
    ctx.run("poetry run mypy src/ tasks.py tests/", pty=True)


@task
def docs(ctx: Context) -> None:
    """Generate HTML documentation with pdoc."""
    ctx.run("poetry run pdoc src/ml_data_pipeline -o docs -d google", pty=True)


@task
def run(ctx: Context, config: str = "config/config_dev.yaml") -> None:
    """Run the data pipeline with the specified configuration file."""
    ctx.run(f"poetry run ml-data-pipeline --config {config}", pty=True)
