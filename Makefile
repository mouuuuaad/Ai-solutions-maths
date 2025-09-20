# AI Math Solver - Development Commands

.PHONY: help install dev build clean docker-up docker-down

help: ## Show this help message
	@echo "AI Math Solver - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	@echo "Installing dependencies..."
	cd frontend && npm install
	cd backend && go mod tidy
	cd ai-service && python -m venv .venv && .venv/bin/pip install -r requirements.txt
	cd database && npm install

dev: ## Start all services in development mode
	@echo "Starting development servers..."
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8080"
	@echo "AI Service: http://localhost:8001"
	@echo "Database: localhost:5432"
	@echo ""
	@echo "Make sure PostgreSQL is running before starting the backend!"

dev-frontend: ## Start only frontend
	cd frontend && npm run dev

dev-backend: ## Start only backend
	cd backend && go run main.go

dev-ai: ## Start only AI service
	cd ai-service && source .venv/bin/activate && uvicorn app.main:app --reload --port 8001

build: ## Build all services
	@echo "Building all services..."
	cd frontend && npm run build
	cd backend && go build -o bin/server .
	cd ai-service && source .venv/bin/activate && pip install -r requirements.txt

docker-up: ## Start all services with Docker Compose
	@echo "Starting services with Docker Compose..."
	docker-compose up -d
	@echo "Services started! Frontend: http://localhost:3000"

docker-down: ## Stop all Docker services
	@echo "Stopping Docker services..."
	docker-compose down

docker-build: ## Build all Docker images
	@echo "Building Docker images..."
	docker-compose build

clean: ## Clean up build artifacts
	@echo "Cleaning up..."
	cd frontend && rm -rf .next node_modules
	cd backend && rm -rf bin
	cd ai-service && rm -rf .venv __pycache__
	docker-compose down -v

db-migrate: ## Run database migrations
	cd database && npx prisma migrate dev

db-studio: ## Open Prisma Studio
	cd database && npx prisma studio

test: ## Run tests
	@echo "Running tests..."
	cd frontend && npm test
	cd backend && go test ./...
	cd ai-service && source .venv/bin/activate && python -m pytest

logs: ## Show logs for all services
	docker-compose logs -f

status: ## Show status of all services
	docker-compose ps
