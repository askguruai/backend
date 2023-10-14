LINE_WIDTH=120
ISORT_FLAGS=--line-width=${LINE_WIDTH} --profile black
BLACK_FLAGS=--line-length=${LINE_WIDTH}
AUTOFLAKE_FLAGS=--remove-all-unused-imports --remove-unused-variables --recursive --exclude "**/__init__.py"
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
	autoflake ${AUTOFLAKE_FLAGS} .
	black ${BLACK_FLAGS} --check --diff .
	isort ${ISORT_FLAGS} --check-only --diff .

format-fix:
	autoflake ${AUTOFLAKE_FLAGS} --in-place .
	black ${BLACK_FLAGS} .
	isort ${ISORT_FLAGS} .
