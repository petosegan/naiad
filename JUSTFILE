default:
    just --list

lab:
    poetry run jupyter lab

lint:
    ruff check .

fmt:
    black .
