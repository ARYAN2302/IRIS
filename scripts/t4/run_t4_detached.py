#!/usr/bin/env python3
"""
Detached launcher for T4 pipeline test.

Uses .spawn() instead of .remote() so the Modal function runs independently
of the local process. The local script can exit and the Modal app keeps running.

After spawning, polls the results volume until t4_pipeline_test.json appears.

Usage:
    python scripts/t4/run_t4_detached.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the modal app + function from test_pipeline_t4.py
from scripts.t4.test_pipeline_t4 import app, run_pipeline_test, RESULTS_VOL


@app.local_entrypoint()
def main(phase: str = "all"):
    """
    Run T4 pipeline test in detached mode.

    The function runs on Modal until completion. Results are saved to
    the iris-results volume as t4_pipeline_test.json.
    """
    print("=" * 60)
    print("IRIS T4 Pipeline Test — Detached Launcher")
    print("=" * 60)
    print(f"  Phase: {phase}")
    print(f"  Time:  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print()
    print("Spawning remote function (detached)...")
    print("The Modal app will run independently of this local process.")
    print("Results will be saved to: iris-results:/t4_pipeline_test.json")
    print()

    # Spawn the function — this returns immediately with a FunctionCall object
    function_call = run_pipeline_test.spawn(phases=phase)
    print(f"  [ok] spawned function call: {function_call.object_id}")
    print()
    print("Function is now running on Modal. You can:")
    print(f"  - Monitor: modal app logs <app-id>")
    print(f"  - Check status: modal call status {function_call.object_id}")
    print(f"  - Get results: modal volume get iris-results /t4_pipeline_test.json results/")
    print()
    print("This local script will now poll for completion...")

    # Poll for the function to complete
    # function_call.get() blocks until the function returns
    try:
        results = function_call.get()
        print("\n" + "=" * 60)
        print("T4 PIPELINE TEST COMPLETE")
        print("=" * 60)
        print(f"  Results: {results}")
        print()
        print("Results saved to Modal volume 'iris-results' as t4_pipeline_test.json")
        print("To download: modal volume get iris-results /t4_pipeline_test.json results/")
    except Exception as e:
        print(f"\n  [error] function failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
