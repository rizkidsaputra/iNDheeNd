#!/usr/bin/env python3
"""Start NFD on the pilot topology and stop cleanly after a short smoke run."""
import sys
import time

from mininet.log import setLogLevel, info
from minindn.minindn import Minindn
from minindn.apps.app_manager import AppManager
from minindn.apps.nfd import Nfd


def main():
    topology = sys.argv[1] if len(sys.argv) > 1 else "minindn_topology.conf"
    setLogLevel("info")
    Minindn.cleanUp()
    Minindn.verifyDependencies()
    ndn = Minindn(topoFile=topology, controller=None)
    try:
        ndn.start()
        AppManager(ndn, ndn.net.hosts, Nfd)
        time.sleep(5)
        info("Mini-NDN smoke test: %d emulated nodes started\n" % len(ndn.net.hosts))
        for node in ndn.net.hosts[:3]:
            info("%s: " % node.name)
            info(node.cmd("nfdc status | head -1"))
    finally:
        ndn.stop()


if __name__ == "__main__":
    main()
