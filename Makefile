# ═══════════════════════════════════════════════════════════════════════
#  Makefile for AMA WS25 Project
#  ═══════════════════════════════════════════════════════════════════════
#  Essential commands for development, testing, and documentation
#  Uses: uv, ruff, pytest, mypy, quarto
#  ═══════════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help
.PHONY: help

# Color codes
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m
RED := \033[0;31m

# Project directories
PKG_DIR := ama_tlbx
SRC_DIR := ama_tlbx
TEST_DIR := tests
DOCS_DIR := docs

PYTHON_INTERPRETER ?= /opt/homebrew/Caskroom/miniconda/base/envs/ama/bin/python
FORCE_ACTIV_CONDA_ENV ?= 1  # 1 to enforce the exact interpreter
CONDA_ENV_NAME ?= ama       # expected conda env name

#  ═══════════════════════════════════════════════════════════════════════
#  🔍 Code Quality
#  ═══════════════════════════════════════════════════════════════════════

lint: ## 🔍 Check code with ruff linter
	@echo "$(BLUE)Running ruff linter...$(NC)"
	cd $(PKG_DIR) && uv run ruff check $(SRC_DIR)

format: ## 🔍 Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	cd $(PKG_DIR) && uv run ruff format $(SRC_DIR)

fix: ## 🔍 Auto-fix code issues (format + lint)
	@echo "$(BLUE)Auto-fixing code issues...$(NC)"
	@$(MAKE) format
	@cd $(PKG_DIR) && uv run ruff check $(SRC_DIR) --fix
	@echo "$(GREEN)✓ Code fixed!$(NC)"

check: ## 🔍 Run all checks (lint + format + mypy)
	@echo "$(BLUE)Running all checks...$(NC)"
	@$(MAKE) lint
	@cd $(PKG_DIR) && uv run ruff format $(SRC_DIR) --check
	@cd $(PKG_DIR) && uv run mypy $(SRC_DIR)
	@echo "$(GREEN)✓ All checks passed!$(NC)"

#  ═══════════════════════════════════════════════════════════════════════
#  🧪 Testing
#  ═══════════════════════════════════════════════════════════════════════

test: ## 🧪 Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	cd $(PKG_DIR) && pytest -v

test-cov: ## 🧪 Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	cd $(PKG_DIR) && pytest -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

ci: ## 🧪 Run full continuous integration pipeline (checks + tests)
	@echo "$(BLUE)Running full CI pipeline (checks + tests)...$(NC)"
	@$(MAKE) check
	@$(MAKE) test-cov
	@echo "$(GREEN)✓ CI pipeline completed successfully!$(NC)"

#  ═══════════════════════════════════════════════════════════════════════
#  📚 Documentation
#  ═══════════════════════════════════════════════════════════════════════

docs-render: ## 📚 Build documentation (pdoc + quarto)
	@echo "$(BLUE)Generating API documentation...$(NC)"
	@cd $(DOCS_DIR) && bash scripts/generate_pdoc.sh
	@echo "$(BLUE)Rendering Quarto documentation...$(NC)"
	@cd $(DOCS_DIR) && quarto render

docs-preview: ## 📚 Preview documentation with live reload
	@echo "$(BLUE)Starting documentation preview...$(NC)"
	@cd $(DOCS_DIR) && bash scripts/generate_pdoc.sh
	@cd $(DOCS_DIR) && quarto preview --no-browser

docs-clean: ## 📚 Clean documentation output
	@echo "$(YELLOW)Cleaning documentation...$(NC)"
	@rm -rf $(DOCS_DIR)/_site $(DOCS_DIR)/.quarto
	@echo "$(GREEN)✓ Documentation cleaned$(NC)"


#  ═════════════════════════════════════════════════════════════════════=
#  Agent Context helpers
#  ═════════════════════════════════════════════════════════════════════=

.PHONY: _check_python
_check_python:
	@CURRENT=$$(which python); \
	if [ "$$CURRENT" != "$(PYTHON_INTERPRETER)" ]; then \
		echo "$(YELLOW)⚠️  Python interpreter mismatch.$(NC)"; \
		echo "  which python -> $$CURRENT"; \
		echo "  expected     -> $(PYTHON_INTERPRETER)"; \
		if [ "$(FORCE_ACTIV_CONDA_ENV)" = "1" ]; then \
			echo "$(RED)FORCE_ACTIV_CONDA_ENV=1 — aborting. Please run: conda activate $(CONDA_ENV_NAME)$(NC)"; \
			exit 1; \
		else \
			echo "$(YELLOW)Proceeding anyway. Set FORCE_ACTIV_CONDA_ENV=1 to enforce.$(NC)"; \
		fi; \
	fi

context-package: _check_python ## 🗺️ Summarize symbols per module (classes/functions/constants)
	@$(PYTHON_INTERPRETER) ama_tlbx/scripts/get_context.py packages --root ama_tlbx/ama_tlbx

context-classes: _check_python ## 🗺️ List classes with full docstrings
	echo "# Mermaid UML Diagram of the ama_tlbx:\n\`\`\`{mermaid}"
	@$(PYTHON_INTERPRETER) -m syrenka classdiagram ama_tlbx/ama_tlbx
	echo "\`\`\`\n---\n"
	@$(PYTHON_INTERPRETER) ama_tlbx/scripts/get_context.py classes --root ama_tlbx/ama_tlbx --full-doc

context-dir-tree: _check_python ## 🗺️ Print directory tree for `ama_tlbx/ama_tlbx/` (ignore __pycache__)
	@echo "Directory tree for ama_tlbx/ama_tlbx/:"
	@bash -lc 'tree ama_tlbx/ama_tlbx/ -I "__pycache__"'

#  ═══════════════════════════════════════════════════════════════════════
#  ℹ️  Help
#  ═══════════════════════════════════════════════════════════════════════

help: ## Show this help message
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)               AMA WS25 Project - Makefile Commands             $(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
		@echo "$(YELLOW)Usage:$(NC) make <target>"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "; section=""} \
		/^#  ═+$$/ {next} \
		/^#  [📦🔍🧪📚🔧🗺️]/ {if (section) print ""; section=$$0; gsub(/^#  /, "", section); print "$(YELLOW)" section "$(NC)"; next} \
		/^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-18s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
