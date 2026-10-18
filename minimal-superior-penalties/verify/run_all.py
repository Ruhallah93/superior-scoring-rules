#!/usr/bin/env python3
"""Run the whole verification suite and emit report.md + report.json.

    python run_all.py

Exit code 0 iff every check passes.
"""
import json, platform, subprocess, sys, time
import numpy as np
import scipy

import check_endpoints, check_superiority, check_propriety, check_published

SEED = 20250815


def environment():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "seed": SEED,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    env = environment()
    print("=" * 78)
    print("Verification suite -- Minimal Superior Penalties")
    print("=" * 78)
    for k, v in env.items():
        print("  %-10s %s" % (k, v))

    results = []
    for name, mod in [("endpoints", check_endpoints),
                      ("superiority", check_superiority),
                      ("propriety", check_propriety),
                      ("published", check_published)]:
        print("\n" + "-" * 78)
        print("[%s]" % name)
        print("-" * 78)
        results += mod.run(np.random.default_rng(SEED))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("  %-8s %-6s %s" % ("ledger", "status", "description"))
    for r in results:
        print("  %-8s %-6s %s" % (r.ledger_id, "PASS" if r.passed else "FAIL",
                                  r.description[:58]))
    n_fail = sum(1 for r in results if not r.passed)
    print("\n  %d checks groups, %d failing" % (len(results), n_fail))

    payload = {"environment": env,
               "results": [r.to_dict() for r in results],
               "failing": n_fail}
    with open("report.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    with open("report.md", "w") as fh:
        fh.write("# Verification report\n\n")
        fh.write("| key | value |\n|---|---|\n")
        for k, v in env.items():
            fh.write("| %s | %s |\n" % (k, v))
        fh.write("\n## Results by ledger ID\n\n")
        fh.write("| ledger | status | description |\n|---|---|---|\n")
        for r in results:
            fh.write("| %s | %s | %s |\n" % (r.ledger_id,
                     "PASS" if r.passed else "**FAIL**", r.description))
        fh.write("\n## Detail\n\n")
        for r in results:
            fh.write("### %s\n\n%s\n\n" % (r.ledger_id, r.description))
            for ch in r.checks:
                fh.write("- %s %s — `%s`\n" % ("PASS" if ch["passed"] else "**FAIL**",
                                               ch["name"], ch["evidence"]))
            fh.write("\n")
    print("  wrote report.json and report.md")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
