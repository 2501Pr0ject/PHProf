#!/usr/bin/env python3
"""Export du modèle fine-tuné en GGUF pour PHProf."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_project_root, load_config


def main() -> None:
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Export GGUF PHProf")
    parser.add_argument(
        "--adapters",
        type=Path,
        help="Chemin vers les adapters LoRA (override config)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Fichier de sortie GGUF (override config)",
    )
    parser.add_argument(
        "-q", "--quantization",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="Type de quantization (override config)",
    )

    args = parser.parse_args()

    print("PHProf - Export GGUF\n")

    config = load_config("lora_config")
    project_root = get_project_root()

    training_config = config["training"]
    export_config = config["export"]

    # Chemins
    adapters_path = args.adapters or (project_root / training_config["output_dir"])
    output_file = args.output or (project_root / export_config["output_file"])
    quantization = args.quantization or export_config["quantization"]

    if not adapters_path.exists():
        print(f"Adapters LoRA introuvables: {adapters_path}")
        print("   Exécutez 'make train' pour les créer.")
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Adapters: {adapters_path}")
    print(f"Sortie: {output_file}")
    print(f"Quantization: {quantization}")
    print()

    # Étape 1: Fuse les adapters avec le modèle de base
    print("Fusion des adapters avec le modèle de base...")

    fused_dir = project_root / "models" / "fused_model"
    fused_dir.mkdir(parents=True, exist_ok=True)

    fuse_cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", training_config["base_model"],
        "--adapter-path", str(adapters_path),
        "--save-path", str(fused_dir),
    ]

    try:
        subprocess.run(fuse_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la fusion: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("mlx-lm non installé")
        sys.exit(1)

    # Étape 2: Conversion en GGUF
    print("\nConversion en GGUF...")

    # Utilise llama.cpp convert script
    llama_cpp_path = project_root / "vendor" / "llama.cpp"
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"Script de conversion introuvable: {convert_script}")
        print("   Exécutez 'make setup-llama' pour configurer llama.cpp")
        sys.exit(1)

    gguf_unquantized = output_file.with_suffix(".f16.gguf")

    convert_cmd = [
        sys.executable, str(convert_script),
        str(fused_dir),
        "--outfile", str(gguf_unquantized),
        "--outtype", "f16",
    ]

    try:
        subprocess.run(convert_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion: {e}")
        sys.exit(1)

    # Étape 3: Quantization
    if quantization != "f16":
        print(f"\nQuantization {quantization}...")

        quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"

        if not quantize_bin.exists():
            print(f"llama-quantize introuvable: {quantize_bin}")
            sys.exit(1)

        quantize_cmd = [
            str(quantize_bin),
            str(gguf_unquantized),
            str(output_file),
            quantization.upper(),
        ]

        try:
            subprocess.run(quantize_cmd, check=True)
            # Supprime le fichier non quantifié
            gguf_unquantized.unlink(missing_ok=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de la quantization: {e}")
            sys.exit(1)
    else:
        # Pas de quantization, renomme simplement
        gguf_unquantized.rename(output_file)

    # Nettoyage
    import shutil
    shutil.rmtree(fused_dir, ignore_errors=True)

    print(f"\nExport terminé!")
    print(f"   Modèle: {output_file}")
    print(f"   Taille: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
