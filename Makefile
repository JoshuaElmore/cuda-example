.PHONY: all cpp python julia venv clean distclean

# Override if the default python3 is too old for the wheels, e.g.
#   make PYTHON=python3.12 venv
PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
# Stamp lives inside the (gitignored) venv so deps reinstall when the
# requirements file changes, but not on every `make python`.
DEPS := $(VENV)/.deps-installed

all: cpp python julia

cpp:
	@echo "=== C++/CUDA ==="
	cd c++ && pixi run run

venv: $(DEPS)

$(DEPS): python/requirments.txt
	test -x $(PY) || $(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r python/requirments.txt
	touch $@

python: $(DEPS)
	@echo "=== Python ==="
	$(PY) python/cmy.py

julia:
	@echo "=== Julia ==="
	cd julia && pixi run run

clean:
	cd c++ && rm -f cmy_right

# Also throws away the venv; `make python` will rebuild it from scratch.
distclean: clean
	rm -rf $(VENV)
