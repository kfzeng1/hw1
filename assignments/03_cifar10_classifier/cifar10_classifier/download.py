import argparse

from .data import CIFAR10_MIRRORS, download_cifar10


def main():
    parser = argparse.ArgumentParser(description="Download CIFAR-10.")
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument(
        "--mirror",
        default="official",
        choices=tuple(CIFAR10_MIRRORS),
        type=str,
    )
    parser.add_argument("--download-url", default="", type=str)
    args = parser.parse_args()
    download_cifar10(args.data_dir, args.mirror, args.download_url)
    print(f"CIFAR-10 is ready in {args.data_dir}")


if __name__ == "__main__":
    main()
