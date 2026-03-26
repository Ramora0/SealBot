PYTHON ?= python3

.PHONY: help build rebuild play evaluate clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-12s %s\n", $$1, $$2}'

build: ## Build the C++ extension
	$(PYTHON) setup.py build_ext --inplace

rebuild: ## Clean and rebuild the C++ extension
	rm -rf build *.so *.pyd *.egg-info
	$(PYTHON) setup.py build_ext --inplace

play: build ## Play against SealBot interactively
	$(PYTHON) play.py

evaluate: build ## Run evaluation (SealBot vs random)
	$(PYTHON) evaluate.py -n $(or $(N),20) -t $(or $(T),0.1)

clean: ## Remove build artifacts
	rm -rf build __pycache__ *.so *.pyd *.egg-info positions/
