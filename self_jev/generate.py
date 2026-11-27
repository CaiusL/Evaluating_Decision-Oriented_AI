"""Write the seeded readiness dataset."""
from __future__ import annotations

from self_jev.dataset import manifest, write_dataset


def main() -> None:
    path = write_dataset()
    from self_jev.dataset import load_dataset

    print(path)
    print(manifest(load_dataset()))


if __name__ == "__main__":
    main()
