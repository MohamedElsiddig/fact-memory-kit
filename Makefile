.PHONY: install test gate demo lint clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -e . pytest

test:
	.venv/bin/pytest -q

gate:
	.venv/bin/factkit gate

demo:
	.venv/bin/factkit --db /tmp/factkit-demo.db init --demo
	.venv/bin/factkit --db /tmp/factkit-demo.db golden-write
	.venv/bin/factkit --db /tmp/factkit-demo.db gate --no-vectors

clean:
	rm -rf .venv build dist *.egg-info __pycache__ .pytest_cache
