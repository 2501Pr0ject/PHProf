#!/usr/bin/env python3
"""Fine-tuning LoRA avec mlx-lm pour PHProf."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_project_root, load_config


def main() -> None:
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning LoRA PHProf")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=get_project_root() / "configs" / "lora_config.yaml",
        help="Fichier de configuration LoRA",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Reprendre depuis un checkpoint",
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="Nombre d'itérations (override config)",
    )

    args = parser.parse_args()

    print("PHProf - Fine-tuning LoRA\n")

    config = load_config("lora_config")
    project_root = get_project_root()

    training_config = config["training"]
    lora_config = training_config["lora"]

    # Vérifie que le dataset existe
    data_dir = project_root / training_config["data_dir"]
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    if not train_file.exists():
        print(f"Dataset d'entraînement introuvable: {train_file}")
        print("   Exécutez 'make build-dataset' pour le créer.")
        sys.exit(1)

    # Prépare la commande mlx_lm.lora
    output_dir = project_root / training_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    iters = args.iters or training_config["iters"]

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", training_config["base_model"],
        "--data", str(data_dir),
        "--train",
        "--iters", str(iters),
        "--batch-size", str(training_config["batch_size"]),
        "--learning-rate", str(training_config["learning_rate"]),
        "--lora-rank", str(lora_config["rank"]),
        "--adapter-path", str(output_dir),
        "--val-batches", str(training_config["val_batches"]),
        "--steps-per-eval", str(training_config["steps_per_eval"]),
        "--steps-per-report", str(training_config["steps_per_report"]),
        "--save-every", str(training_config["save_every"]),
        "--seed", str(training_config["seed"]),
    ]

    if args.resume:
        cmd.extend(["--resume-adapter-file", str(args.resume)])

    print(f"Dataset: {data_dir}")
    print(f"Sortie: {output_dir}")
    print(f"Itérations: {iters}")
    print(f"LoRA rank: {lora_config['rank']}")
    print()

    try:
        subprocess.run(cmd, check=True)
        print("\nFine-tuning terminé!")
        print(f"   Adapters sauvegardés dans: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nErreur lors du fine-tuning: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nmlx-lm non installé. Installez-le avec:")
        print("   uv pip install mlx-lm")
        sys.exit(1)


if __name__ == "__main__":
    main()
