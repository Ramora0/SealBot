PYTHON ?= python3

.PHONY: help build rebuild play evaluate clean promote

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-12s %s\n", $$1, $$2}'

build: ## Build both current/ and best/ C++ extensions
	@for d in current best; do \
		echo "Building $$d ..."; \
		cd "$$d" && $(CURDIR)/.venv/bin/python setup.py build_ext --inplace && cd $(CURDIR); \
	done

rebuild: ## Clean and rebuild both C++ extensions
	@for d in current best; do \
		rm -rf "$$d"/build "$$d"/*.so "$$d"/*.pyd "$$d"/*.egg-info; \
		echo "Building $$d ..."; \
		cd "$$d" && $(CURDIR)/.venv/bin/python setup.py build_ext --inplace && cd $(CURDIR); \
	done

play: build ## Play against current SealBot interactively
	$(CURDIR)/.venv/bin/python play.py

evaluate: build ## Run evaluation (current vs best)
	$(CURDIR)/.venv/bin/python evaluate.py -n $(or $(N),20) -t $(or $(T),0.1)

promote: ## Copy current/ engine files to best/
	@for f in engine.h minimax_bot.cpp types.h pattern_data.h ankerl_unordered_dense.h setup.py; do \
		cp current/$$f best/$$f; \
	done
	@echo "Copied current -> best. Run 'make rebuild' to compile."

clean: ## Remove build artifacts
	@for d in current best; do \
		rm -rf "$$d"/build "$$d"/*.so "$$d"/*.pyd "$$d"/*.egg-info; \
	done
	rm -rf __pycache__ positions/
