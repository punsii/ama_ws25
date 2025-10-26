#  ═══════════════════════════════════════════════════════════════════════
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

# Project directories
PKG_DIR := ama_tlbx
SRC_DIR := ama_tlbx
TEST_DIR := tests
DOCS_DIR := docs

#  ═══════════════════════════════════════════════════════════════════════
#  📦 Installation
#  ═══════════════════════════════════════════════════════════════════════

install: ## 📦 Install package with minimal dependencies
	@echo "$(BLUE)Installing ama-tlbx...$(NC)"
	cd $(PKG_DIR) && uv sync

install-dev: ## 📦 Install with dev dependencies (tests, linting)
	@echo "$(BLUE)Installing with dev dependencies...$(NC)"
	cd $(PKG_DIR) && uv sync --extra dev

install-all: ## 📦 Install with all dependencies (dev + notebook + docs)
	@echo "$(BLUE)Installing with all dependencies...$(NC)"
	cd $(PKG_DIR) && uv sync --all-extras

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
	cd $(PKG_DIR) && uv run --no-sync pytest $(TEST_DIR) -v

test-cov: ## 🧪 Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	cd $(PKG_DIR) && uv run --no-sync pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

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

#  ═══════════════════════════════════════════════════════════════════════
#  🔧 Workflows
#  ═══════════════════════════════════════════════════════════════════════

ci: ## 🔧 Run full CI pipeline (install + check + test)
	@echo "$(BLUE)Running CI pipeline...$(NC)"
	@$(MAKE) install-dev
	@$(MAKE) check
	@$(MAKE) test
	@echo "$(GREEN)✓ CI pipeline passed!$(NC)"

clean: ## 🔧 Clean all caches and generated files
	@echo "$(YELLOW)Cleaning caches...$(NC)"
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf $(DOCS_DIR)/_site $(DOCS_DIR)/.quarto
	@echo "$(GREEN)✓ Workspace cleaned$(NC)"

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
		/^#  [📦🔍🧪📚🔧]/ {if (section) print ""; section=$$0; gsub(/^#  /, "", section); print "$(YELLOW)" section "$(NC)"; next} \
		/^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-18s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
