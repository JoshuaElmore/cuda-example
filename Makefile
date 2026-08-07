.PHONY: all cpp python julia clean

all: cpp python julia

cpp:
	@echo "=== C++/CUDA ==="
	cd c++ && pixi run run

python:
	@echo "=== Python ==="
	.venv/bin/python python/cmy.py

julia:
	@echo "=== Julia ==="
	cd julia && pixi run run

clean:
	cd c++ && rm -f cmy_right
