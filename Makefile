# PHProf - Makefile
# Assistant pédagogique IA local pour PHP et Symfony

.PHONY: help install download-model download-docs build-index \
        chat ask evaluate train export clean lint test all setup

# Variables
PYTHON := uv run python
MODEL_DIR := models
DOCS_DIR := rag/docs
INDEX_DIR := rag/index

# Couleurs
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

help: ## Affiche cette aide
	@echo "$(BLUE)PHProf$(RESET) - Assistant pédagogique PHP/Symfony"
	@echo ""
	@echo "$(GREEN)Commandes disponibles:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-18s$(RESET) %s\n", $$1, $$2}'

# ============================================================================
# Installation
# ============================================================================

install: ## Installe les dépendances Python
	@echo "$(BLUE)Installation des dépendances...$(RESET)"
	uv sync
	@echo "$(GREEN)-> Dépendances installées$(RESET)"

install-all: ## Installe toutes les dépendances (dev + train + eval)
	@echo "$(BLUE)Installation de toutes les dépendances...$(RESET)"
	uv sync --all-extras
	@echo "$(GREEN)-> Toutes les dépendances installées$(RESET)"

download-model: ## Télécharge le modèle de base
	@echo "$(BLUE)Téléchargement du modèle...$(RESET)"
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) -c "from huggingface_hub import hf_hub_download; \
		hf_hub_download('Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF', \
			'qwen2.5-coder-1.5b-instruct-q4_k_m.gguf', \
			local_dir='$(MODEL_DIR)')"
	@echo "$(GREEN)-> Modèle téléchargé dans $(MODEL_DIR)$(RESET)"

# ============================================================================
# Documentation RAG
# ============================================================================

download-docs: ## Télécharge la documentation PHP/Symfony
	@echo "$(BLUE)Téléchargement de la documentation...$(RESET)"
	$(PYTHON) scripts/download_docs.py
	@echo "$(GREEN)-> Documentation téléchargée$(RESET)"

build-index: ## Construit l'index RAG FAISS
	@echo "$(BLUE)Construction de l'index RAG...$(RESET)"
	$(PYTHON) scripts/build_index.py
	@echo "$(GREEN)-> Index construit$(RESET)"

# ============================================================================
# Utilisation
# ============================================================================

chat: ## Lance le chat interactif
	$(PYTHON) -m app.cli chat

chat-no-rag: ## Lance le chat sans RAG
	$(PYTHON) -m app.cli chat --no-rag

chat-finetuned: ## Lance le chat avec le modèle fine-tuné
	$(PYTHON) -m app.cli chat --finetuned

ask: ## Pose une question (usage: make ask Q="votre question")
ifndef Q
	@echo "$(YELLOW)Usage: make ask Q=\"votre question\"$(RESET)"
else
	$(PYTHON) -m app.cli ask "$(Q)"
endif

# ============================================================================
# Évaluation
# ============================================================================

evaluate: ## Lance l'évaluation complète
	@echo "$(BLUE)Évaluation en cours...$(RESET)"
	@mkdir -p reports
	$(PYTHON) scripts/evaluate.py
	@echo "$(GREEN)-> Évaluation terminée$(RESET)"

evaluate-quick: ## Évaluation rapide (10 prompts)
	@echo "$(BLUE)Évaluation rapide...$(RESET)"
	@mkdir -p reports
	$(PYTHON) scripts/evaluate.py --limit 10
	@echo "$(GREEN)-> Évaluation terminée$(RESET)"

syntax-check: ## Vérifie la syntaxe PHP des réponses
ifndef FILE
	@echo "$(YELLOW)Usage: make syntax-check FILE=reports/eval_xxx.json$(RESET)"
else
	$(PYTHON) scripts/syntax_check.py $(FILE) -v
endif

# ============================================================================
# Dataset et entraînement
# ============================================================================

build-dataset: ## Construit le dataset d'entraînement
	@echo "$(BLUE)Construction du dataset...$(RESET)"
	$(PYTHON) scripts/build_dataset.py
	@echo "$(GREEN)-> Dataset construit$(RESET)"

validate-dataset: ## Valide le dataset
	@echo "$(BLUE)Validation du dataset...$(RESET)"
	$(PYTHON) scripts/validate_dataset.py -v
	@echo "$(GREEN)-> Validation terminée$(RESET)"

train: ## Lance le fine-tuning LoRA
	@echo "$(BLUE)Fine-tuning LoRA...$(RESET)"
	$(PYTHON) scripts/train_lora.py
	@echo "$(GREEN)-> Fine-tuning terminé$(RESET)"

export: ## Exporte le modèle en GGUF
	@echo "$(BLUE)Export GGUF...$(RESET)"
	$(PYTHON) scripts/export_gguf.py
	@echo "$(GREEN)-> Export terminé$(RESET)"

# ============================================================================
# Développement
# ============================================================================

lint: ## Vérifie le code avec ruff
	@echo "$(BLUE)Vérification du code...$(RESET)"
	uv run ruff check app scripts
	uv run ruff format --check app scripts
	@echo "$(GREEN)-> Code validé$(RESET)"

lint-fix: ## Corrige automatiquement le code
	@echo "$(BLUE)Correction du code...$(RESET)"
	uv run ruff check --fix app scripts
	uv run ruff format app scripts
	@echo "$(GREEN)-> Code corrigé$(RESET)"

test: ## Lance les tests
	@echo "$(BLUE)Exécution des tests...$(RESET)"
	uv run pytest
	@echo "$(GREEN)-> Tests passés$(RESET)"

typecheck: ## Vérifie les types avec mypy
	@echo "$(BLUE)Vérification des types...$(RESET)"
	uv run mypy app scripts
	@echo "$(GREEN)-> Types validés$(RESET)"

# ============================================================================
# Nettoyage
# ============================================================================

clean: ## Nettoie les fichiers temporaires
	@echo "$(BLUE)Nettoyage...$(RESET)"
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info dist build
	@echo "$(GREEN)-> Nettoyé$(RESET)"

clean-all: clean ## Nettoie tout (y compris modèles et index)
	@echo "$(YELLOW)Suppression des modèles et index...$(RESET)"
	rm -rf $(MODEL_DIR)/*.gguf
	rm -rf $(INDEX_DIR)/*
	rm -rf $(DOCS_DIR)/*
	@echo "$(GREEN)-> Tout nettoyé$(RESET)"

# ============================================================================
# Raccourcis
# ============================================================================

setup: install download-model ## Installation complète (sans docs)
	@echo "$(GREEN)-> Installation complète$(RESET)"
	@echo ""
	@echo "Prochaines étapes:"
	@echo "  1. make download-docs  # Télécharge la documentation"
	@echo "  2. make build-index    # Construit l'index RAG"
	@echo "  3. make chat           # Lance le chat"

all: setup download-docs build-index ## Installation complète avec RAG
	@echo "$(GREEN)-> PHProf est prêt !$(RESET)"
	@echo ""
	@echo "Lancez: make chat"
