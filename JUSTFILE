default:
    just --list

test:
    cd py && poetry run pytest

lab:
    poetry run jupyter lab

lint:
    ruff check .

fmt:
    black .
