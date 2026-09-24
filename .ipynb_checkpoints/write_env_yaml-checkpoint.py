import argparse
from packaging.version import Version


CONDA_TORCH_MAX = Version("2.5.1")


def cuda_to_pip_tag(cuda_version: str) -> str:
    """
    Convert CUDA version like:
        12.8 -> cu128
        12.6 -> cu126
        11.8 -> cu118
    """
    major, minor = cuda_version.split(".")
    return f"cu{major}{minor}"


def build_environment(name: str, torch_version: str, cuda_version: str) -> str:
    torch_ver = Version(torch_version)

    lines = [
        f"name: {name}",
        "",
        "channels:",
        "  - pytorch",
        "  - nvidia",
        "  - conda-forge",
        "  - defaults",
        "  - bioconda",
        # "  - Ernest-NH-Ho",
        "",
        "dependencies:",
        "  - python",
        "  - numpy",
        "  - pandas",
        "  - pycifrw"
    ]

    if torch_ver <= CONDA_TORCH_MAX:
        # PyTorch <= 2.5.1: install through conda
        lines.extend([
            f"  - pytorch={torch_version}",
            f"  - pytorch-cuda={cuda_version}",
        ])

    lines.extend([
        "  - tqdm",
        "  - transformers",
        "  - tokenizers",
        "  - pip",
        "  - pip:",
    ])

    if torch_ver > CONDA_TORCH_MAX:
        # PyTorch > 2.5.1: install CUDA-specific PyTorch wheel through pip
        cuda_tag = cuda_to_pip_tag(cuda_version)

        lines.extend([
            f"      - --extra-index-url https://download.pytorch.org/whl/{cuda_tag}",
            f"      - torch=={torch_version}",
        ])

    lines.extend([
        "      - esm==3.2.1",
        "      - heiwa==0.1.2",
    ])

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate environment.yaml for HEIWA."
    )

    parser.add_argument(
        "--name",
        default="heiwa",
        help="Conda environment name (default: heiwa)",
    )

    parser.add_argument(
        "--torch",
        default="2.5.1",
        help="PyTorch version (default: 2.5.1)",
    )

    parser.add_argument(
        "--cuda",
        default="12.1",
        help="PyTorch CUDA version (default: 12.1)",
    )

    parser.add_argument(
        "--output",
        default="environment.yaml",
        help="Output YAML file (default: environment.yaml)",
    )

    args = parser.parse_args()

    content = build_environment(
        name=args.name,
        torch_version=args.torch,
        cuda_version=args.cuda,
    )

    with open(args.output, "w") as f:
        f.write(content)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()