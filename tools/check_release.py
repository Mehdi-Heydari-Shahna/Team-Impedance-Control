"""Verify the paper study and preserved numerical result files."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / 'results/manifest_sha256.json').read_text(encoding='utf-8'))
    failures = []
    for group in ('result_files', 'scientific_core'):
        for relative, expected in manifest[group].items():
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                failures.append(f'Changed or missing {relative}')
    for path in ROOT.rglob('*.py'):
        if '.git' not in path.parts:
            try:
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            except (SyntaxError, UnicodeError) as error:
                failures.append(str(error))
    metrics = json.loads((ROOT / 'results/mission/metrics.json').read_text(encoding='utf-8'))
    reference = metrics.get('al_n4_kz0', {})
    if abs(reference.get('mass', 0) - 63.06649681528663) > 1e-10 or not reference.get('completed'):
        failures.append('The supplied main result is not the completed 63.0665 kg aluminium study.')
    contact = json.loads((ROOT / 'results/mission/mujoco_refdisp_25.json').read_text(encoding='utf-8'))
    for kind, expected in (('psi', 0.19681161393790136), ('hand', 1.9408287616234503)):
        if abs(contact[kind]['max_dp_mm'] - expected) > 1e-10:
            failures.append(f'Unexpected stored contact result for {kind}.')
    if failures:
        for failure in failures:
            print('FAILED:', failure, file=sys.stderr)
        return 1
    print(f"Release checks passed: {len(manifest['result_files'])} supplied result files, "
          f"{len(manifest['scientific_core'])} unchanged scientific modules, "
          f"{len(metrics)} stored mission summaries, Python syntax, and aluminium-study reference values.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
