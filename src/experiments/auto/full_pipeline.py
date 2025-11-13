from __future__ import annotations

from . import (
    bias_sweep,
    baseline,
    fairness_frontier,
    sanity_checks,
)


def main() -> None:
    baseline.main()
    sanity_checks.main()
    bias_sweep.main()
    fairness_frontier.main()


if __name__ == "__main__":
    main()
