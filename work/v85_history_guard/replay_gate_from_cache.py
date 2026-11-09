"""Replay V85 historical retrieval from frozen CPU model caches and prepared data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PANELS = ('D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N',
          'D7_I405_S', 'D12_I5_N', 'D12_I5_S')


def replay(clean_data, release, output):
    clean_data, release, output = (path.resolve() for path in (clean_data, release, output))
    required = [clean_data / 'cleaning_report.json', release / 'config' / 'corridors.json']
    for panel in PANELS:
        required.extend(clean_data / panel / f'{part}.parquet' for part in (
            'links', 'train', 'validation', 'private'))
        required.extend(release / 'task2' / panel / split / 'window_index.csv' for split in (
            'validation', 'private'))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required prepared data files missing:\n' + '\n'.join(missing))

    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, V85_HANDOFF_CLEAN=str(clean_data), V85_HANDOFF_RELEASE=str(release),
               V85_HANDOFF_AUDIT=str(output), V85_HANDOFF_QUEUE=str(ROOT / 'inputs' / 'v84_queue.csv'),
               TFB_EXPERIMENT='v85_experiment',
               PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    script = ROOT / 'replay' / 'work' / 'analog_gate_official_audit.py'
    subprocess.run([sys.executable, str(script)], env=env, cwd=ROOT, check=True)

    files = ('official_gate_windows_audit.csv', 'official_gate_cells_audit.csv', 'official_gate_summary.json')
    comparisons = {}
    for name in files:
        actual = (output / name).read_bytes().replace(b'\r\n', b'\n')
        expected = (ROOT / 'audit' / name).read_bytes().replace(b'\r\n', b'\n')
        comparisons[name] = {
            'matches_original_ignoring_line_endings': actual == expected,
            'sha256_lf': hashlib.sha256(actual).hexdigest(),
        }
    report = {'all_audit_files_match': all(row['matches_original_ignoring_line_endings']
                                          for row in comparisons.values()), 'files': comparisons}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-data', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.clean_data, args.release, args.output)
