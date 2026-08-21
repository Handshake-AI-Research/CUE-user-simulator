"""Unified ``tau2-eval`` CLI: ``tau2-eval {prep, run} ...``."""

from __future__ import annotations

import sys


def _usage() -> str:
    return "usage: cue-tau2-eval {prep,run,personas,ppol_personas} ...\n"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_usage())
        return
    command, rest = argv[0], argv[1:]
    if command == "prep":
        from cue_training.evaluation.tau2_bench.prep import main as prep_main

        prep_main(rest)
    elif command == "run":
        from cue_training.evaluation.tau2_bench.runner import main as run_main

        run_main(rest)
    elif command == "personas":
        from cue_training.evaluation.tau2_bench.personas import main as personas_main

        personas_main(rest)
    elif command == "ppol_personas":
        from cue_training.evaluation.tau2_bench.ppol_personas import main as ppol_personas_main

        ppol_personas_main(rest)
    else:
        sys.stderr.write(f"tau2-eval: unknown command {command!r}\n{_usage()}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
