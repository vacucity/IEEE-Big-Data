"""Rebuild the exact V85 submission CSV using saved gate evidence and V84."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / 'inputs'
V84 = INPUTS / 'v84_observation_flow.zip'
V85 = INPUTS / 'V85_history_guard.zip'
SOURCE_HASHES = {
    V84: 'a429599ade6ee8663b28f850f86acc0819ba426d2b2f9cc0861b2831edc36090',
    V85: 'b594280f0bbe313d6e8bf45cffcb924099a41d95c647fad5f327ef5399c643a5',
}
HEADER = b'submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def queue_key(row, timestamp):
    return (row['split'], row['panel'], row['window_id'], row['link_id'], timestamp)


def reconstruct_selection(audit_dir):
    windows = list(read_rows(audit_dir / 'official_gate_windows_audit.csv'))
    by_window = {row['window_id']: row for row in windows}
    require(len(windows) == len(by_window) == 80, 'Expected 80 unique audited windows')
    require(Counter(row['split'] for row in windows) == {'validation': 40, 'private': 40},
            'Unexpected window split counts')

    candidates = {}
    per_window = Counter()
    gated_per_window = Counter()
    for row in read_rows(audit_dir / 'official_gate_cells_audit.csv'):
        window = by_window[row['window_id']]
        require(window['cls'] == 'constr' and all(
            row[key] == window[key] for key in ('panel', 'split', 'date', 'T')
        ), 'Candidate does not match its constrained window')
        require((row['before'], row['after']) == ('0', '1'), 'Expected only 0 to 1 changes')
        valid = int(row['valid_history'])
        history, cpu, gap = (float(row[key]) for key in (
            'history_probability', 'cpu_probability', 'probability_gap'))
        require(0 <= valid <= 30 and 0 <= history <= 1 and 0 <= cpu <= 1,
                'Invalid saved history evidence')
        require(abs(history - cpu - gap) <= 1e-7, 'Invalid probability gap')
        selected = valid >= 20 and gap >= 0.30
        require(row['gated'] == str(selected), 'Gate result differs from the original audit')
        require(row['baseline_replay_mismatch'] == 'False', 'Unexpected baseline mismatch')
        horizon = int(row['horizon_minutes'])
        require(horizon in (10, 15, 20, 25, 30), 'Unexpected candidate horizon')
        timestamp = datetime.fromisoformat(row['date'] + 'T00:00:00+00:00') + timedelta(
            minutes=5 * int(row['T']) + horizon)
        key = queue_key(row, timestamp)
        require(key not in candidates, 'Duplicate audited candidate')
        candidates[key] = selected
        per_window[row['window_id']] += 1
        gated_per_window[row['window_id']] += int(selected)

    require(len(candidates) == 12, 'Expected twelve pre-gate candidates')
    for window in windows:
        wid = window['window_id']
        require(per_window[wid] == int(window['ungated_official_changes']) and
                gated_per_window[wid] == int(window['gated_official_changes']),
                f'Counts disagree for {wid}')

    mapped = {}
    for row in read_rows(INPUTS / 'v84_queue.csv'):
        key = queue_key(row, datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')))
        if key in candidates:
            require(key not in mapped and row['queue_pred'] == '0', 'Invalid V84 queue mapping')
            mapped[key] = int(row['submission_id'])
    require(mapped.keys() == candidates.keys() and len(set(mapped.values())) == 12,
            'V84 queue mapping does not cover the twelve candidates')

    selected = {key: mapped[key] for key, apply in candidates.items() if apply}
    require(Counter(key[0] for key in selected) == {'validation': 8, 'private': 2},
            'Unexpected split counts for the ten selected cells')
    reference = {}
    for row in read_rows(INPUTS / 'V85_history_guard_changes.csv'):
        key = queue_key(row, datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')))
        require(key not in reference and (row['before'], row['after']) == ('0', '1'),
                'Invalid official change list')
        reference[key] = int(row['submission_id'])
    require(selected == reference, 'Recomputed mapping differs from the official change list')
    return set(mapped.values()), set(selected.values())


def patch_row(line, row_id, candidate_ids, patch_ids):
    if row_id not in candidate_ids:
        return line
    body = line.rstrip(b'\r\n')
    fields = body.split(b',')
    require(len(fields) == 6 and int(fields[0]) == row_id and fields[1] == b'queue'
            and fields[4] == b'0', f'Unexpected V84 record at {row_id}')
    if row_id not in patch_ids:
        return line
    fields[4] = b'1'
    return b','.join(fields) + line[len(body):]


def rebuild(audit_dir, output):
    for path, expected in SOURCE_HASHES.items():
        with path.open('rb') as stream:
            require(sha256_stream(stream) == expected, f'Unexpected original hash for {path.name}')
    candidate_ids, patch_ids = reconstruct_selection(audit_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), f'Output already exists: {output}')

    with tempfile.TemporaryDirectory(prefix='.v85-', dir=output.parent) as tmpdir:
        tmp = Path(tmpdir) / output.name
        digest = hashlib.sha256()
        with (ZipFile(V84) as baseline, ZipFile(V85) as reference,
              ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=6) as generated):
            members = [entry for entry in baseline.infolist() if not entry.is_dir()]
            require(len(members) == 1 and reference.namelist() == ['submission.csv'],
                    'Unexpected original ZIP structure')
            with (baseline.open(members[0]) as original, reference.open('submission.csv') as official,
                  generated.open('submission.csv', 'w') as result):
                header = original.readline()
                require(header.rstrip(b'\r\n') == HEADER and header == official.readline(),
                        'Submission CSV headers differ')
                buffer = bytearray(header)
                row_count = 0
                for row_count, line in enumerate(original, 1):
                    patched = patch_row(line, row_count, candidate_ids, patch_ids)
                    require(official.readline() == patched, f'Official CSV mismatch at row {row_count}')
                    buffer.extend(patched)
                    if len(buffer) >= 1024 * 1024:
                        result.write(buffer)
                        digest.update(buffer)
                        buffer.clear()
                require(row_count == 6_980_503 and official.read(1) == b'' and
                        max(candidate_ids) <= row_count, 'Unexpected submission row count')
                result.write(buffer)
                digest.update(buffer)

        with ZipFile(tmp) as generated, generated.open('submission.csv') as result:
            require(sha256_stream(result) == digest.hexdigest(), 'Rebuilt CSV hash or CRC mismatch')
        require(not output.exists(), f'Output already exists: {output}')
        tmp.rename(output)

    return {
        'status': 'PASS',
        'submission_rows': row_count,
        'pre_gate_candidates': len(candidate_ids),
        'changes': {'validation': 8, 'private': 2},
        'rebuilt_csv_byte_identical_to_official_v85': True,
        'submission_csv_sha256': digest.hexdigest(),
        'archive_crc': 'PASS',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'audit')
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced' / 'V85_history_guard_rebuilt.zip')
    args = parser.parse_args()
    print(json.dumps(rebuild(args.audit_dir, args.output), ensure_ascii=False, indent=2))
