"""Compatibility entry point for the v4 orchestrator."""

import sys

from orchestrator import hybrid_orchestrator

if __name__ == "__main__":
    hybrid_orchestrator.main()
else:
    sys.modules[__name__] = hybrid_orchestrator
