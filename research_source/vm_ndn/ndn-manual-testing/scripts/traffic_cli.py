#!/usr/bin/env python3
"""Session-scoped, live Mini-NDN PoC11C traffic control. Run as root via CLI sh."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess as sp
import sys
import time

LAB = Path('/home/ndn/research/ndn-manual-testing')
SOURCE = Path('/home/ndn/research/experiments/poc11/scripts/poc11c')
APP = SOURCE / 'poc11c_apps_v5'
SINK = SOURCE / 'poc11c_silent_sink_v1'
STATE = LAB / 'configs/traffic_state.json'
LOCK = LAB / 'configs/traffic_cli.lock'
BASE_PREFIX = '/poc11c/attack/unavailable'
NODES = [f'r{i}' for i in range(1, 11)] + [f'c{i}' for i in range(1, 7)] + ['a1', 'a2', 'p1', 'p2', 'm']
METRICS = ('nPitEntries', 'nUnsatisfiedInterests', 'nInInterests', 'nOutInterests', 'nInNacks', 'nOutNacks')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def state_load():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save(state):
    write_json(STATE, state)
    write_json(Path(state['run_dir']) / 'session_state.json', state)


def event(state, kind, **details):
    with (Path(state['run_dir']) / 'actions.jsonl').open('a') as out:
        out.write(json.dumps({'event': kind, 'monotonic_ns': time.monotonic_ns(),
                              'utc': utc(), **details}) + '\n')


def command(args, timeout=8):
    done = sp.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    if done.returncode:
        raise RuntimeError(f'Perintah gagal ({done.returncode}): {args}: {done.stderr.strip()}')
    return done.stdout


def node_pid(node):
    if node not in NODES:
        raise ValueError(f'Node tidak valid: {node}')
    found = sp.run(['pgrep', '-f', f'^nfd --config /tmp/minindn/{node}/nfd.conf$'],
                   capture_output=True, text=True).stdout.split()
    if len(found) != 1:
        raise RuntimeError(f'NFD {node} harus tepat satu, ditemukan {found}')
    return int(found[0])


def proc_info(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        fields = path.read_text().rpartition(') ')[2].split()
        return {'state': fields[0], 'ticks': fields[19], 'pgrp': int(fields[2])}
    except (OSError, IndexError):
        return None


def alive(record):
    info = proc_info(record['pid'])
    return bool(info and info['state'] != 'Z' and info['ticks'] == record['ticks']
                and info['pgrp'] == record['pid'])


def ns(node, argv):
    return ['mnexec', '-a', str(node_pid(node)), 'env',
            f'NDN_CLIENT_TRANSPORT=unix:///run/nfd/{node}.sock', *map(str, argv)]


def nfd(node, *argv):
    return command(ns(node, ['nfdc', *argv]))


def status_metrics(node):
    raw = nfd(node, 'status', 'show')
    return {key: int(m.group(1)) for key in METRICS
            if (m := re.search(r'\b' + key + r'=(\d+)', raw))}


def check_topology(state=None):
    pids = {n: node_pid(n) for n in NODES}
    nlsr = sp.run(['pgrep', '-f', '^nlsr -f /tmp/minindn/'],
                  capture_output=True, text=True).stdout.split()
    if len(nlsr) != 21:
        raise RuntimeError(f'NLSR count {len(nlsr)}, perlu 21')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if state and (state['boot_id'] != boot or any(
            state['nfd'][n] != {'pid': pids[n], 'ticks': proc_info(pids[n])['ticks']}
            for n in NODES)):
        raise RuntimeError('Topologi/NFD berubah sejak sesi dibuat; hentikan dan periksa manual')
    if not state:
        return boot, {n: {'pid': pids[n], 'ticks': proc_info(pids[n])['ticks']} for n in NODES}


def conflicting_processes(state):
    owned = {p['pid'] for p in state['processes']} if state else set()
    conflicts = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in owned:
            continue
        try:
            cmdline = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        args = [x.decode(errors='replace') for x in cmdline if x]
        if not args:
            continue
        exe = Path(args[0]).name
        if exe == SINK.name or (exe.startswith('poc11c_apps_') and
                               len(args) > 1 and args[1] in ('producer', 'consumer')):
            conflicts.append({'pid': int(entry.name), 'args': args[:4]})
    return conflicts


def create_session():
    check_topology()
    conflicts = conflicting_processes(None)
    if conflicts:
        raise RuntimeError(f'Workload di luar sesi ini masih aktif: {conflicts}')
    boot, nfd_map = check_topology()
    run = 'interactive_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = LAB / 'runs' / run
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {'run_id': run, 'run_dir': str(run_dir), 'boot_id': boot,
             'nfd': nfd_map, 'processes': [], 'attack_cycle': 0, 'closed': False,
             'created_utc': utc()}
    save(state)
    event(state, 'session_created', nfd_count=21, nlsr_count=21)
    spawn(state, 'sampler', None, [sys.executable, __file__, '_sampler',
                                  str(run_dir), str(nfd_map['r1']['pid']),
                                  str(nfd_map['r2']['pid'])], 'pit_samples.jsonl')
    print(f'[SESSION] {run_dir}')
    return state


def current(state, role, node=None):
    return [p for p in state['processes'] if p['role'] == role and
            (node is None or p['node'] == node) and alive(p)]


def spawn(state, role, node, argv, filename):
    path = Path(state['run_dir']) / filename
    if path.exists():
        raise RuntimeError(f'Log akan tertimpa: {path}')
    with path.open('x') as out:
        proc = sp.Popen(ns(node, argv) if node else list(map(str, argv)),
                        stdout=out, stderr=sp.STDOUT, stdin=sp.DEVNULL,
                        start_new_session=True)
    for _ in range(50):
        info = proc_info(proc.pid)
        if info and info['pgrp'] == proc.pid:
            break
        time.sleep(.01)
    else:
        raise RuntimeError(f'Proses {role} gagal dilacak')
    record = {'role': role, 'node': node, 'pid': proc.pid, 'ticks': info['ticks'],
              'log': str(path), 'launched_ns': time.monotonic_ns(), 'argv': list(map(str, argv))}
    state['processes'].append(record)
    save(state)
    event(state, 'process_launched', role=role, node=node, pid=proc.pid, log=str(path))
    return record


def events(path):
    result = []
    if not Path(path).exists():
        return result
    for line in Path(path).read_text(errors='replace').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def wait_event(record, target, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if any(e.get('event') == target for e in events(record['log'])):
            return
        if not alive(record):
            raise RuntimeError(f'{record["role"]} berhenti sebelum {target}: {record["log"]}')
        time.sleep(.1)
    raise RuntimeError(f'{record["role"]} tidak mencapai {target}: {record["log"]}')


def stop_record(state, record):
    if not alive(record):
        return
    try:
        os.killpg(record['pid'], signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if not alive(record):
            break
        time.sleep(.1)
    if alive(record):
        try:
            os.killpg(record['pid'], signal.SIGKILL)
        except ProcessLookupError:
            pass
    event(state, 'process_stopped', role=record['role'], node=record['node'], pid=record['pid'])


def stop_role(state, role, node=None):
    found = current(state, role, node)
    for record in found:
        stop_record(state, record)
    print(f'[STOP] {role} {node or "all"}: {len(found)} process')


def session_for_start():
    state = state_load()
    if not state or state['closed']:
        return create_session()
    check_topology(state)
    conflicts = conflicting_processes(state)
    if conflicts:
        raise RuntimeError(f'Workload unmanaged aktif: {conflicts}')
    return state


def producer_start(state):
    if current(state, 'producer'):
        raise RuntimeError('Producer sudah aktif; gunakan status')
    started = []
    try:
        for node in ('p1', 'p2'):
            record = spawn(state, 'producer', node,
                           [APP, 'producer', node, state['run_id'],
                            time.monotonic_ns() + 2_000_000_000, 3600, 0,
                            f'/poc11c/legitimate/{node}'], f'producer_{node}.jsonl')
            started.append(record)
            wait_event(record, 'prefix_registered')
        print('[OK] Producer p1/p2 registered')
    except Exception:
        for record in started:
            stop_record(state, record)
        raise


def benign_start(state):
    if len(current(state, 'producer')) != 2:
        raise RuntimeError('Nyalakan kedua Producer dahulu')
    if current(state, 'benign'):
        raise RuntimeError('Consumer benign sudah aktif')
    for i in range(1, 7):
        node = f'c{i}'
        target = 'p1' if i <= 3 else 'p2'
        if f'/poc11c/legitimate/{target} ' not in nfd(node, 'fib', 'list'):
            raise RuntimeError(f'FIB {node} tidak punya route producer {target}')
    started = []
    start = time.monotonic_ns() + 3_000_000_000
    try:
        for i in range(1, 7):
            node = f'c{i}'
            target = 'p1' if i <= 3 else 'p2'
            record = spawn(state, 'benign', node,
                           [APP, 'consumer', node, state['run_id'],
                            start + (i-1)*20_000_000, 3600, 5,
                            f'/poc11c/legitimate/{target}'], f'benign_{node}.jsonl')
            started.append(record)
            wait_event(record, 'app_ready')
        event(state, 'benign_ready', planned_first_ns=start)
        print('[OK] Enam Consumer benign siap, 5 Interest/detik/node')
    except Exception:
        for record in started:
            stop_record(state, record)
        raise


def ensure_sink(state):
    found = current(state, 'sink')
    if found:
        return found[0]
    prefix = f'{BASE_PREFIX}/{state["run_id"]}'
    record = spawn(state, 'sink', 'p1',
                   [SINK, '--execute-after-pilot-approval', prefix], 'silent_sink.jsonl')
    try:
        wait_event(record, 'sink_prefix_registered')
        if prefix + ' ' not in nfd('p1', 'fib', 'list'):
            raise RuntimeError('PIT/FIB p1 belum mempunyai sink lokal')
        for node in ('a1', 'a2'):
            if BASE_PREFIX + ' ' not in nfd(node, 'fib', 'list'):
                raise RuntimeError(f'FIB {node} tidak punya route attack')
        event(state, 'sink_ready', prefix=prefix)
        return record
    except Exception:
        stop_record(state, record)
        raise


def attack_start(state, node, rate, duration, start_ns=None):
    if current(state, 'attack', node):
        raise RuntimeError(f'Attacker {node} sudah aktif')
    sink = ensure_sink(state)
    before = len([e for e in events(sink['log']) if e.get('event') == 'sink_interest_received'])
    state['attack_cycle'] += 1
    save(state)
    cycle = state['attack_cycle']
    prefix = f'{BASE_PREFIX}/{state["run_id"]}'
    run = f'{state["run_id"]}_attack_{cycle}'
    planned = start_ns if start_ns is not None else time.monotonic_ns() + 2_000_000_000
    if planned < time.monotonic_ns() + 500_000_000:
        raise RuntimeError('start-ns harus setidaknya 0.5 detik di masa depan')
    record = spawn(state, 'attack', node,
                   [APP, 'consumer', node, run, planned, duration, rate, prefix],
                   f'attack_{cycle}_{node}.jsonl')
    record['cycle'] = cycle
    save(state)
    try:
        wait_event(record, 'app_ready')
        wait_event(record, 'interest_submitted', 5)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rows = events(record['log'])
            if any(e.get('event') == 'nack' and e.get('reason') == 150 for e in rows):
                raise RuntimeError('NoRoute NACK muncul saat startup; cek route/sink')
            arrivals = [e for e in events(sink['log']) if e.get('event') == 'sink_interest_received']
            first = next(e for e in rows if e.get('event') == 'interest_submitted')
            own = next((e for e in arrivals[before:] if e.get('name') == first.get('name')), None)
            if own:
                event(state, 'attack_first_arrival', node=node, cycle=cycle,
                      submitted_ns=first['monotonic_ns'], sink_ns=own['monotonic_ns'])
                print(f'[OK] {node} aktif: sink menerima Interest; rate={rate}/s durasi={duration}s')
                return
            time.sleep(.1)
        raise RuntimeError('Sink belum menerima Interest pertama')
    except Exception:
        stop_record(state, record)
        raise


def summarize(state):
    print(f'Sesi: {state["run_id"]} | log: {state["run_dir"]}')
    for p in state['processes']:
        if p['role'] == 'sampler':
            continue
        rows = events(p['log'])
        counts = {k: sum(e.get('event') == k for e in rows) for k in
                  ('interest_submitted', 'data_received', 'data_sent', 'timeout', 'nack', 'sink_interest_received')}
        nr = sum(e.get('event') == 'nack' and e.get('reason') == 150 for e in rows)
        print(f'{p["role"]:8} {str(p["node"]):3} pid={p["pid"]:6} '
              f'{"RUN" if alive(p) else "STOP":4} '
              f'sent={counts["interest_submitted"]} data={counts["data_sent"] if p["role"] == "producer" else counts["data_received"]} '
              f'timeout={counts["timeout"]} nack={counts["nack"]} '
              f'NoRoute={nr} sink={counts["sink_interest_received"]}')
    print('PIT/counter r1:', status_metrics('r1'))
    print('PIT/counter r2:', status_metrics('r2'))


def sampler(run_dir, pid1, pid2):
    # Own isolated process; its stdout is append-only session evidence.
    last_alive = {}
    exit_logged = set()
    while True:
        state = state_load()
        if state and state.get('run_dir') == run_dir:
            for process in state['processes']:
                if process['pid'] == os.getpid() or process['pid'] in exit_logged:
                    continue
                if alive(process):
                    last_alive[process['pid']] = time.monotonic_ns()
                elif process['pid'] in last_alive:
                    event(state, 'process_exit_observed', role=process['role'],
                          node=process['node'], pid=process['pid'],
                          last_alive_ns=last_alive[process['pid']])
                    exit_logged.add(process['pid'])
        row = {'event': 'pit_sample', 'monotonic_ns': time.monotonic_ns(), 'utc': utc()}
        for node, pid in (('r1', pid1), ('r2', pid2)):
            if node_pid(node) != pid:
                raise RuntimeError(f'NFD {node} restarted; sampler stopped')
            row[node] = status_metrics(node)
        print(json.dumps(row), flush=True)
        time.sleep(.5)


def main():
    parser = argparse.ArgumentParser(description='PoC11C live manual traffic controller')
    sub = parser.add_subparsers(dest='group', required=True)
    sub.add_parser('status')
    sub.add_parser('pit')
    session = sub.add_parser('session'); session.add_argument('action', choices=['new'])
    producer = sub.add_parser('producer'); producer.add_argument('action', choices=['start', 'stop'])
    benign = sub.add_parser('benign'); benign.add_argument('action', choices=['start', 'stop'])
    attack = sub.add_parser('attack'); attack.add_argument('action', choices=['start', 'stop'])
    attack.add_argument('--node', choices=['a1', 'a2'], required=True)
    attack.add_argument('--rate', type=float, default=25)
    attack.add_argument('--duration', type=int, default=30)
    attack.add_argument('--start-ns', type=int,
                        help='waktu mulai monotonic VM; opsional untuk jadwal eksperimen')
    stop = sub.add_parser('stop'); stop.add_argument('what', choices=['all'])
    hidden = sub.add_parser('_sampler'); hidden.add_argument('run_dir'); hidden.add_argument('r1', type=int); hidden.add_argument('r2', type=int)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('Jalankan dengan sudo dari mini-ndn> sh atau terminal Ubuntu')
    if args.group == '_sampler':
        sampler(args.run_dir, args.r1, args.r2)
        return
    if args.group == 'attack' and args.action == 'start' and not (1 <= args.rate <= 50 and 1 <= args.duration <= 120):
        parser.error('Rate 1..50 Interest/detik dan durasi 1..120 detik')
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = state_load()
        if args.group == 'status':
            if state:
                check_topology(state); summarize(state)
            else:
                print('Belum ada sesi manual')
            return
        if args.group == 'pit':
            check_topology()
            for node in ('r1', 'r2'):
                print(node, status_metrics(node))
            return
        if args.group == 'session':
            if state and not state['closed'] and any(alive(p) for p in state['processes']):
                raise RuntimeError('Sesi aktif; gunakan stop all dahulu')
            create_session()
            return
        if args.group == 'stop' or args.action == 'stop':
            if not state:
                print('Tidak ada sesi aktif')
                return
            check_topology(state)
            if args.group == 'stop':
                for role in ('attack', 'benign', 'producer', 'sink', 'sampler'):
                    stop_role(state, role)
                state['closed'] = True
                save(state)
                event(state, 'session_closed')
            else:
                stop_role(state, args.group, args.node if args.group == 'attack' else None)
            return
        state = session_for_start()
        if args.group == 'producer':
            producer_start(state)
        elif args.group == 'benign':
            benign_start(state)
        elif args.group == 'attack':
            attack_start(state, args.node, args.rate, args.duration, args.start_ns)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
