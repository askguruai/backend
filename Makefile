LINE_WIDTH=120
ISORT_FLAGS=--line-width=${LINE_WIDTH} --profile black
BLACK_FLAGS=--skip-string-normalization --line-length=${LINE_WIDTH}
PYTEST_FLAGS=-p no:warnings

install:
	pip install -r requirements.txt

install-format:
	pip install black isort

install-test:
	pip install pytest requests aiohttp

test:
	pytest tests -s

format:
	isort ${ISORT_FLAGS} --check-only --diff .
	black ${BLACK_FLAGS} --check --diff .

format-fix:
	isort ${ISORT_FLAGS} .
	black ${BLACK_FLAGS} .
