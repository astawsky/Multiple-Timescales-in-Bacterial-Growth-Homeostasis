#!/usr/bin/env python3
"""
Run test_single_chain.py then evaluate.py in-process, printing output live
and saving it to analysis/pymc_marketing/run_output.log.

Run from the repo root:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc-marketing/bin/python \
        analysis/pymc_marketing/run.py
"""

import os
import sys
import runpy
from datetime import datetime

DIR = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(DIR, "run_output.log")

SCRIPTS = [
    os.path.join(DIR, "test_single_chain.py"),
    os.path.join(DIR, "evaluate.py"),
]


class Tee:
    """Write to both a file and the original stream simultaneously."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


with open(LOG, "w") as log_fh:
    tee = Tee(sys.__stdout__, log_fh)
    sys.stdout = tee
    sys.stderr = tee

    try:
        timestamp = f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(timestamp, end="")

        for script in SCRIPTS:
            header = f"\n{'='*60}\n{os.path.basename(script)}\n{'='*60}\n"
            print(header, end="")
            runpy.run_path(script, run_name="__main__")

        done = f"\nRun complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(done, end="")

    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

print(f"Output saved to {LOG}")
