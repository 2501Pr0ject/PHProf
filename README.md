# PHProf

**Assistant pédagogique IA local pour PHP et Symfony**

PHProf est un assistant conversationnel qui aide les développeurs francophones à maîtriser PHP moderne (8.x) et le framework Symfony. Il fonctionne **100% en local**, sans connexion internet.

## Fonctionnalités

- **Chat interactif** - Interface CLI riche et intuitive
- **100% local** - Aucune donnée envoyée sur internet
- **RAG intégré** - Enrichi avec la documentation officielle Symfony et PHP
- **Français natif** - Réponses structurées en français
- **Pédagogique** - Format TL;DR / Problème / Solution / Explication / À retenir
- **Multi-plateforme** - macOS, Linux, Windows

## Cas d'usage

PHProf peut vous aider a :

- **Corriger du code PHP** - Identifier et corriger les erreurs de syntaxe ou de logique
- **Expliquer du code** - Comprendre un bout de code PHP ou Symfony
- **Ecrire du code** - Generer des controllers, entities, repositories, services...
- **Deboguer** - Trouver pourquoi votre code ne fonctionne pas
- **Apprendre** - Comprendre les concepts PHP 8.x (types, attributs, enums, match...)
- **Bonnes pratiques** - Appliquer les standards PSR, SOLID, design patterns
- **Symfony** - Routing, controllers, Doctrine, Forms, Security, Twig, Messenger...
- **Tests** - Ecrire des tests PHPUnit et des tests fonctionnels Symfony
- **Refactoring** - Moderniser du code legacy vers PHP 8.x

## Prérequis

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (gestionnaire de paquets Python)
- Git
- PHP 8.x (pour la vérification syntaxique)
- ~5 GB d'espace disque

### Plateformes supportées

| Plateforme | Chat/RAG | Fine-tuning | GPU |
|------------|----------|-------------|-----|
| macOS Apple Silicon | Oui | Oui (MLX) | Metal |
| macOS Intel | Oui | Non | CPU |
| Linux | Oui | Non* | CUDA / CPU |
| Windows | Oui | Non* | CUDA / CPU |

*Fine-tuning possible avec [unsloth](https://github.com/unslothai/unsloth) ou [peft](https://github.com/huggingface/peft)

## Installation

### macOS (Apple Silicon)

```bash
# 1. Cloner le projet
git clone https://github.com/2501Pr0ject/PHProf.git
cd PHProf

# 2. Installer les dépendances
uv sync

# 3. Télécharger le modèle
make download-model

# 4. Télécharger la documentation et construire l'index RAG
make download-docs
make build-index

# 5. Lancer le chat
make chat
```

### Linux / Windows

```bash
# 1. Cloner le projet
git clone https://github.com/2501Pr0ject/PHProf.git
cd PHProf

# 2. Installer les dépendances
uv sync

# 3. (Optionnel) Recompiler llama-cpp-python avec support GPU NVIDIA
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

# 4. Télécharger le modèle
make download-model

# 5. Télécharger la documentation et construire l'index RAG
make download-docs
make build-index

# 6. Lancer le chat
make chat
```

> **Note**: Sans GPU NVIDIA, le chat fonctionne en mode CPU (plus lent mais fonctionnel).

## Utilisation

### Chat interactif

```bash
# Avec RAG (recommandé)
make chat

# Sans RAG
make chat-no-rag

# Avec le modèle fine-tuné
make chat-finetuned
```

### Question unique

```bash
make ask Q="Comment créer un formulaire Symfony ?"
```

### Commandes dans le chat

- `quit` ou `exit` - Quitter
- `clear` - Effacer l'écran

## Structure du projet

```
phpprof/
├── app/                  # Application CLI
│   ├── cli.py           # Interface Typer
│   ├── chat.py          # Boucle de chat
│   ├── rag.py           # Système RAG
│   └── config.py        # Gestion configuration
│
├── scripts/              # Scripts d'automatisation
│   ├── build_index.py   # Construction index FAISS
│   ├── download_docs.py # Téléchargement documentation
│   ├── evaluate.py      # Évaluation automatique
│   ├── syntax_check.py  # Vérification syntaxe PHP
│   ├── build_dataset.py # Construction dataset
│   ├── train_lora.py    # Fine-tuning LoRA
│   └── export_gguf.py   # Export GGUF
│
├── configs/              # Configuration YAML
│   ├── model_config.yaml
│   ├── rag_config.yaml
│   ├── dataset_config.yaml
│   └── lora_config.yaml
│
├── eval/                 # Évaluation
│   ├── prompts_fr.jsonl # 50 prompts d'évaluation
│   └── rubric.json      # Critères d'évaluation
│
├── data/                 # Données d'entraînement
├── rag/                  # Index et documentation
├── models/               # Modèles GGUF
└── reports/              # Rapports d'évaluation
```

## Évaluation

PHProf inclut un système d'évaluation automatique :

```bash
# Évaluation complète (50 prompts)
make evaluate

# Évaluation rapide (10 prompts)
make evaluate-quick

# Vérification syntaxe PHP
make syntax-check FILE=reports/eval_xxx.json
```

### Critères d'évaluation

| Critère | Poids | Description |
|---------|-------|-------------|
| Langue française | 20% | Réponse en français |
| Blocs de code | 20% | Présence de code PHP |
| Couverture sujet | 20% | Concepts attendus |
| Longueur réponse | 20% | 200-800 mots |
| Qualité pédagogique | 20% | Clarté des explications |

## Fine-tuning (optionnel, macOS Apple Silicon)

Pour améliorer les réponses avec vos propres exemples :

```bash
# 0. Installer les dépendances MLX
make install-macos

# 1. Ajouter des exemples dans data/raw/
# 2. Construire le dataset
make build-dataset

# 3. Valider le dataset
make validate-dataset

# 4. Lancer le fine-tuning
make train

# 5. Exporter en GGUF
make export
```

> **Note**: Le fine-tuning avec MLX nécessite macOS avec Apple Silicon (M1/M2/M3).
> Pour Linux/Windows, utilisez [unsloth](https://github.com/unslothai/unsloth) ou [peft](https://github.com/huggingface/peft).

## Sources de documentation RAG

| Source | Description |
|--------|-------------|
| [PHP Manual](https://www.php.net/manual) | Documentation officielle PHP |
| [Symfony Docs](https://symfony.com/doc) | Documentation Symfony 7.x |
| [API Platform](https://api-platform.com/docs) | Framework API REST/GraphQL |
| [Doctrine ORM](https://doctrine-project.org) | ORM et DBAL |
| [Composer](https://getcomposer.org/doc) | Gestionnaire de dépendances |
| [PHPUnit](https://docs.phpunit.de) | Framework de tests |
| [Twig](https://twig.symfony.com/doc) | Moteur de templates |
| [PHP-FIG PSR](https://www.php-fig.org/psr) | Standards PHP |
| [Design Patterns PHP](https://designpatternsphp.readthedocs.io) | Patterns de conception |
| [PHP: The Right Way](https://phptherightway.com) | Bonnes pratiques PHP |

## Technologies

| Composant | Technologie |
|-----------|-------------|
| Inférence | llama-cpp-python (Metal / CUDA / CPU) |
| Modèle | Qwen2.5-Coder-1.5B-Instruct (GGUF Q4_K_M) |
| Embeddings | sentence-transformers |
| Index | FAISS |
| CLI | Typer + Rich |
| Fine-tuning | mlx-lm (LoRA, macOS uniquement) |

## Commandes Make

```bash
make help          # Affiche l'aide
make setup         # Installation complète
make all           # Setup + RAG complet
make chat          # Lance le chat
make evaluate      # Lance l'évaluation
make train         # Fine-tuning LoRA (macOS)
make install-macos # Installe MLX pour fine-tuning
make lint          # Vérifie le code
make test          # Lance les tests
make clean         # Nettoie les fichiers temporaires
```

## Licence

MIT

## Auteur

Abdel TOUATI
