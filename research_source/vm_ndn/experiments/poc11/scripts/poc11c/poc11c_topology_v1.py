#!/usr/bin/env python3
"""Stage 2 Mini-NDN orchestrator for the GNN4IFA pilot.

Starts NFD/NLSR, producer/consumer/attacker workloads, a collector, and
periodic per-node/per-interface telemetry. This is an experiment scaffold:
validate routes, offered load, and application behavior before using results.
"""
import argparse
import shlex
import shutil
import sys
import threading
import time
from pathlib import Path

from mininet.log import info, setLogLevel
from minindn.apps.app_manager import AppManager
from minindn.apps.nfd import Nfd
from minindn.apps.nlsr import Nlsr
from minindn.minindn import Minindn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", nargs="?", default="minindn_topology.conf")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--attack-rate", type=int, default=100)
    parser.add_argument("--log-dir", default="/tmp/gnn4ifa-stage2")
    parser.add_argument("--no-cli", action="store_true")
    return parser.parse_args()


def start_background(node, command, logfile):
    quoted = shlex.quote(command)
    output = shlex.quote(str(logfile))
    pid = node.cmd(
        "nohup sh -c %s > %s 2>&1 < /dev/null & echo $!"
        % (quoted, output)
    ).strip()
    return pid


def advertise(node, prefix, log_file):
    result = node.cmd("nlsrc advertise %s 2>&1" % shlex.quote(prefix))
    with log_file.open("a", encoding="utf-8") as stream:
        stream.write("$ nlsrc advertise %s\n%s\n" % (prefix, result))
    return result


def write_snapshot(node, log_file):
    commands = (
        "date -Is",
        "nfdc status report",
        "nfdc face list",
        "nfdc route list",
        "nlsrc status",
    )
    with log_file.open("a", encoding="utf-8") as stream:
        for command in commands:
            stream.write("$ %s\n" % command)
            stream.write(node.cmd(command))
            stream.write("\n")


def prepare_nlsr_configs(ndn):
    """Make the sample config visible to Nlsr.copyExistentFile on Ubuntu installs."""
    sample = Path("/usr/local/etc/ndn/nlsr.conf.sample")
    if not sample.exists():
        sample = Path("/etc/ndn/nlsr.conf.sample")
    if not sample.exists():
        raise FileNotFoundError("nlsr.conf.sample was not found")
    for node in ndn.net.hosts:
        home_dir = Path(node.params["params"]["homeDir"])
        home_dir.mkdir(parents=True, exist_ok=True)
        destination = home_dir / "nlsr.conf"
        if not destination.exists():
            shutil.copyfile(sample, destination)


def collect_interfaces(ndn, log_dir, stop_event):
    link_dir = log_dir / "links"
    link_dir.mkdir(parents=True, exist_ok=True)
    while not stop_event.is_set():
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        for node in ndn.net.hosts:
            for intf in node.intfList():
                name = intf.name
                if name == "lo":
                    continue
                log_file = link_dir / ("%s--%s.log" % (node.name, name))
                with log_file.open("a", encoding="utf-8") as stream:
                    stream.write("\n[%s] node=%s interface=%s\n" %
                                 (timestamp, node.name, name))
                    stream.write(node.cmd("ip -s link show dev %s" % name))
                    stream.write(node.cmd("tc -s qdisc show dev %s" % name))
        stop_event.wait(1.0)


def main():
    args = parse_args()
    setLogLevel("info")
    log_dir = Path(args.log_dir).absolute()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Minindn also parses sys.argv. Remove stage-2-only options before constructing it.
    sys.argv = [sys.argv[0], args.topology]
    Minindn.cleanUp()
    Minindn.verifyDependencies()
    ndn = Minindn(topoFile=args.topology, controller=None)
    processes = []
    stop_event = threading.Event()
    stats_thread = None

    try:
        ndn.start()
        info("Starting NFD on all nodes\n")
        AppManager(ndn, ndn.net.hosts, Nfd)
        # Give NFD sockets time to become available before NLSR creates faces.
        time.sleep(5)
        info("Starting NLSR on all nodes\n")
        prepare_nlsr_configs(ndn)
        AppManager(ndn, ndn.net.hosts, Nlsr)
        time.sleep(10)

        for node in ndn.net.hosts:
            write_snapshot(node, log_dir / ("%s.log" % node.name))

        # Mini-NDN normalizes names to lowercase; keep case-insensitive lookups
        # aligned with the names used in the topology documentation.
        nodes = {node.name.upper(): node for node in ndn.net.hosts}
        for node in ndn.net.hosts:
            if node.name.startswith('r'):
                advertise(node, '/monitor/'+node.name+'/telemetry', log_dir / (node.name+'.log'))
            if node.name in ('p1','p2'):
                advertise(node, '/poc11c/legitimate/'+node.name, log_dir / (node.name+'.log'))
            if node.name=='p1':
                advertise(node, '/poc11c/attack/unavailable', log_dir / (node.name+'.log'))
        # Workloads hanya diluncurkan runner setelah safety gate.

        info("Stage 2 workloads started for %d seconds\n" % args.duration)
        info("Logs: %s\n" % log_dir)
        if not args.no_cli:
            from minindn.util import MiniNDNCLI
            MiniNDNCLI(ndn.net)
        else:
            (log_dir / "topology_ready.json").write_text(
                __import__('json').dumps({'ready_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'attacker_enabled':False,'consumers':0,'source':'PoC11C topology only, no workloads'},indent=2))
            import signal
            halted=threading.Event()
            signal.signal(signal.SIGTERM,lambda *_:halted.set())
            signal.signal(signal.SIGINT,lambda *_:halted.set())
            halted.wait()
    finally:
        stop_event.set()
        if stats_thread is not None:
            stats_thread.join(timeout=3)
        for node, pid, _ in processes:
            if pid:
                node.cmd("kill %s 2>/dev/null || true" % pid)
        ndn.stop()


if __name__ == "__main__":
    main()

