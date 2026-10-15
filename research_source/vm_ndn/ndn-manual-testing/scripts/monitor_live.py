#!/usr/bin/env python3
"""Readable independent terminal monitor over the unchanged ML-7 helper."""
from collections import Counter, deque
import json
import os
from pathlib import Path
import signal
import subprocess as sp
import sys

LAB = Path('/home/ndn/research/ndn-manual-testing')
HELPER = LAB / 'scripts/attach_ml7_long.py'


def read_events(path):
    if not path or not path.exists():
        return []
    result = []
    for line in path.read_text(errors='replace').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def value(number, digits=1):
    return '-' if number is None else f'{number:.{digits}f}'


def interrupted(*_):
    raise KeyboardInterrupt


def main():
    if os.geteuid() != 0:
        sys.exit('Jalankan dengan sudo; monitor memakai namespace NFD')
    # nohup starts with SIGINT ignored. Restore it so a detached monitor can
    # still be stopped cleanly, and give the helper its normal Ctrl+C handler.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, interrupted)
    def restore_child_signals():
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    proc = sp.Popen(['/usr/bin/python3', str(HELPER)], stdout=sp.PIPE,
                    stderr=sp.STDOUT, text=True, bufsize=1,
                    start_new_session=True, preexec_fn=restore_child_signals)
    run_dir = None
    history = deque(maxlen=5)
    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith('[RUN] '):
                run_dir = LAB / 'runs' / line.split(' ', 1)[1]
                print(f'ML-7 run: {run_dir}', flush=True)
            elif line.startswith('{'):
                try:
                    decision = json.loads(line)
                except json.JSONDecodeError:
                    print(line, flush=True)
                    continue
                if decision.get('event') != 'decision':
                    continue
                status = ('UNKNOWN' if decision.get('status') != 'prediction'
                          else 'IFA' if decision.get('prediction') == 1 else 'NORMAL')
                probability = decision.get('probability_ifa') if status != 'UNKNOWN' else None
                frame = decision.get('sequence')
                history.append(f'{frame}:{status}:{value(probability,3)}')
                reports = read_events(run_dir / 'decisions.jsonl') if run_dir else []
                delivery = Counter(e.get('disposition') for e in reports if e.get('event') == 'report')
                wire = read_events(run_dir / 'collector_wire.jsonl') if run_dir else []
                received = sum(e.get('event') == 'telemetry_received' for e in wire)
                fetched = sum(e.get('event') == 'fetch_submitted' for e in wire)
                print(f'frame={frame:>3} {status:<7} P(IFA)={value(probability,4):>6} '
                      f'routers={decision.get("observed_router_count")}/10 '
                      f'missing={decision.get("missing_router_count")} '
                      f'age_max_ms={value(decision.get("current_report_age_max_ms")):>7} '
                      f'infer_ms={value(decision.get("inference_ms")):>5} '
                      f'publish_ms={value(decision.get("decision_to_publish_ms")):>5} '
                      f'wire={received}/{fetched} '
                      f'on_time={delivery["on_time"]} late={delivery["late"]} '
                      f'reason={decision.get("reason")} ', flush=True)
                print('  latest: ' + ' | '.join(history), flush=True)
            elif line:
                print(line, flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f'ML helper keluar dengan kode {proc.returncode}')
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except sp.TimeoutExpired:
                os.kill(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
        print(f'Monitor berhenti. Log: {run_dir}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
