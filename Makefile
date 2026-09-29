.PHONY: help start stop install db-init db-reset clean test lint format \
	start-bill stop-bill status health install-units cutover \
	logs-vllm-llm logs-vllm-embed ps dev

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Braind Project - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

start: ## Start all services (WSL systemd units: vLLM GPU0+GPU1; Windows: app :8010)
	@echo "$(BLUE)🚀 Starting BILL stack...$(NC)"
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_bill.ps1

stop: ## Stop all services (systemd units + legacy holder + app)
	@echo "$(YELLOW)🛑 Stopping all services...$(NC)"
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_bill.ps1
	@echo "$(GREEN)✅ All services stopped$(NC)"

install-units: ## Install the vLLM systemd user units into WSL (idempotent, does not start)
	@echo "$(BLUE)🔧 Installing systemd user units...$(NC)"
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_units.ps1

cutover: ## Stop whatever holds vLLM now, then restart under systemd (durable)
	@echo "$(YELLOW)🔁 Cutting over to systemd-managed vLLM (reloads models)...$(NC)"
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_bill.ps1
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_bill.ps1

install: ## Install dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@python3 -m venv .venv
	@.venv/bin/pip install --upgrade pip
	@.venv/bin/pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

db-init: ## Initialize database (run migrations)
	@echo "$(BLUE)🗄️  Initializing database...$(NC)"
	@.venv/Scripts/python.exe -m alembic upgrade head
	@echo "$(GREEN)✅ Database initialized$(NC)"

db-reset: ## Reset database (WARNING: deletes all data)
	@echo "$(RED)⚠️  WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		PGPASSWORD='Bmf%40123' psql -U postgres -d braind -h 127.0.0.1 -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; \
		$(MAKE) db-init; \
		echo "$(GREEN)✅ Database reset complete$(NC)"; \
	else \
		echo "$(YELLOW)Aborted$(NC)"; \
	fi

clean: ## Clean cache files
	@echo "$(YELLOW)🧹 Cleaning cache files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cache cleaned$(NC)"

test: ## Run tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	@.venv/bin/pytest tests/ -v

lint: ## Run linters (ruff + mypy)
	@echo "$(BLUE)🔍 Running linters...$(NC)"
	@.venv/bin/ruff check .
	@.venv/bin/mypy .

format: ## Format code (ruff)
	@echo "$(BLUE)✨ Formatting code...$(NC)"
	@.venv/bin/ruff format .
	@.venv/bin/ruff check --fix .
	@echo "$(GREEN)✅ Code formatted$(NC)"

dev: ## Start in development mode (with auto-reload)
	@$(MAKE) start

logs-vllm-llm: ## Show vLLM chat logs (systemd journal, falls back to file)
	@wsl.exe -d Ubuntu -e journalctl --user -u braind-vllm-llm.service -f --no-pager 2>/dev/null \
		|| tail -f logs/vllm-chat.log 2>/dev/null || echo "$(RED)No logs found$(NC)"

logs-vllm-embed: ## Show vLLM embed logs (systemd journal, falls back to file)
	@wsl.exe -d Ubuntu -e journalctl --user -u braind-vllm-embed.service -f --no-pager 2>/dev/null \
		|| tail -f logs/vllm-embed.log 2>/dev/null || echo "$(RED)No logs found$(NC)"

ps: ## Show running services (systemd unit states + processes)
	@echo "$(BLUE)📊 vLLM systemd units:$(NC)"
	@wsl.exe -d Ubuntu -e bash -c 'systemctl --user is-active braind-vllm-embed.service braind-vllm-llm.service braind-vllm.target 2>/dev/null' || true
	@echo "$(BLUE)📊 vLLM processes (WSL):$(NC)"
	@wsl.exe -d Ubuntu -e bash -c "pgrep -af 'vllm.entrypoints' | grep -v pgrep" 2>/dev/null || echo "$(YELLOW)No vLLM processes$(NC)"
	@echo "$(BLUE)📊 App on :8010 (Windows):$(NC)"
	@powershell -NoProfile -Command "(Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue).OwningProcess" 2>/dev/null || echo "$(YELLOW)No app on :8010$(NC)"

health: ## Check service health (run from Windows — the app is on Windows; WSL cannot reach it back)
	@echo "$(BLUE)🏥 Checking service health...$(NC)"
	@curl -s http://localhost:8001/v1/models > /dev/null && echo "$(GREEN)✅ vLLM LLM: OK$(NC)" || echo "$(RED)❌ vLLM LLM: DOWN$(NC)"
	@curl -s http://localhost:8002/v1/models > /dev/null && echo "$(GREEN)✅ vLLM Embed: OK$(NC)" || echo "$(RED)❌ vLLM Embed: DOWN$(NC)"
	@curl -s http://localhost:8010/docs > /dev/null && echo "$(GREEN)✅ BILL API (8010): OK$(NC)" || echo "$(RED)❌ BILL API: DOWN$(NC) (run 'make health' from Windows, not WSL)"

start-bill: ## Start full stack via PowerShell orchestrator (idempotent)
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_bill.ps1

status: ## Show status of all three services + VRAM
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/status_bill.ps1

setup: ## Complete setup (install + db-init)
	@echo "$(BLUE)🚀 Running complete setup...$(NC)"
	@.venv/bin/python3 setup.py || python3 setup.py
	@echo "$(GREEN)✅ Setup complete! Run 'make start' to launch.$(NC)"
