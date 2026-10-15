import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ml7_online import (AsyncInference, CausalCollector, FEATURES, JsonlAudit,
                        ROUTERS, Schedule, load_runtime)
from ml7_service import Service


def report(schedule, seq=1, router="r1", value=10, instance="boot:1", boot="boot"):
    return {"run_id": "test", "boot_id": boot, "router_id": router, "sequence": seq,
            "created_ns": schedule.target(router, seq)+10_000_000,
            "nfd_instance_id": instance, "metrics": {k: value for k in FEATURES}}


def wire(p, received):
    return {"event": "telemetry_received", "monotonic_ns": received,
            "router_id": p["router_id"], "sequence": p["sequence"],
            "data_name": f"/monitor/{p['router_id']}/telemetry/poc11c/{p['run_id']}/{p['sequence']}",
            "raw_payload": json.dumps(p)}


def window():
    return {"x": np.ones((1,5,10,10), np.float32),
            "mask": np.ones((1,5,10), np.float32),
            "age": np.ones((1,5,10), np.float32),
            "age_valid": np.ones((1,5,10), np.float32),
            "feature_valid": np.ones((1,5,10,10), np.float32)}


class CausalTests(unittest.TestCase):
    def setUp(self):
        self.s = Schedule(1_000_000_000, 10)
        self.c = CausalCollector("test", "boot", self.s)

    def add(self, seq, **kwargs):
        p = report(self.s, seq, **kwargs)
        return self.c.receive(p, p["created_ns"]+100_000_000)

    def test_cold_start(self):
        f, w, _ = self.c.close(1)
        self.assertIsNone(w)
        for a in f.values(): self.assertFalse(a.any())

    def test_first_observation_has_gauges_and_no_rates(self):
        self.add(1)
        f, _, _ = self.c.close(1)
        np.testing.assert_array_equal(f["feature_valid"][0], [0]*8+[1,1])
        self.assertEqual(f["age_valid"][0], 1)

    def test_missing_features_zero_age_continues(self):
        self.add(1)
        f1, _, _ = self.c.close(1)
        f2, _, _ = self.c.close(2)
        self.assertEqual(f2["mask"][0], 0)
        self.assertFalse(f2["x"][0].any())
        self.assertAlmostEqual(float(f2["age"][0]-f1["age"][0]), 3, places=5)
        self.assertEqual(f2["age_valid"][0], 1)

    def test_late_before_global_decision_excluded(self):
        p = report(self.s)
        result = self.c.receive(p, self.s.deadline("r1",1)+1)
        self.assertEqual(result, "late_report_excluded")
        f, _, _ = self.c.close(1)
        self.assertEqual(f["mask"][0], 0)
        self.assertEqual(f["age_valid"][0], 0)

    def test_deadline_equality_included(self):
        self.c.receive(report(self.s), self.s.deadline("r1",1))
        self.assertEqual(self.c.close(1)[0]["mask"][0], 1)

    def test_late_does_not_become_rate_baseline(self):
        self.add(1, value=10); self.c.close(1)
        self.c.receive(report(self.s,2,value=100), self.s.deadline("r1",2)+1)
        self.c.close(2)
        self.add(3,value=70)
        f, _, _ = self.c.close(3)
        np.testing.assert_array_equal(f["x"][0,:8], np.full(8,10))

    def test_counter_gap_actual_elapsed(self):
        self.add(1,value=10); self.c.close(1); self.c.close(2)
        self.add(3,value=40)
        np.testing.assert_array_equal(self.c.close(3)[0]["x"][0,:8], np.full(8,5))

    def test_reset_recovers_after_next_observation(self):
        self.add(1,value=100); self.c.close(1)
        self.add(2,value=10); f,_,_ = self.c.close(2)
        self.assertFalse(f["feature_valid"][0,:8].any())
        self.add(3,value=40); f,_,_ = self.c.close(3)
        self.assertTrue(f["feature_valid"][0,:8].all())

    def test_instance_change_invalidates_rates(self):
        self.add(1); self.c.close(1)
        self.add(2,value=40,instance="boot:2")
        self.assertFalse(self.c.close(2)[0]["feature_valid"][0,:8].any())

    def test_future_reports_do_not_enter_previous_frames(self):
        self.add(2,value=100)
        f,_,_ = self.c.close(1)
        self.assertFalse(f["mask"].any())
        self.assertEqual(self.c.close(2)[0]["mask"][0], 1)

    def test_frames_are_immutable_and_closed_arrival_excluded(self):
        f,_,_ = self.c.close(1)
        with self.assertRaises(ValueError): f["mask"][0] = 1
        p = report(self.s)
        self.assertEqual(self.c.receive(p, p["created_ns"]), "closed_frame")
        self.assertFalse(f["mask"].any())

    def test_invalid_duplicate_run_boot_schema_and_time(self):
        self.add(1)
        with self.assertRaises(ValueError): self.add(1)
        for change in ({"boot_id":"wrong"}, {"run_id":"wrong"}, {"sequence": True},
                       {"created_ns": -1}, {"metrics": {"label": 1}}):
            p=report(self.s,2); p.update(change)
            with self.assertRaises(ValueError): self.c.receive(p, self.s.decision(2))
        p=report(self.s,2); p["metrics"][FEATURES[0]]=-1
        with self.assertRaises(ValueError): self.c.receive(p,self.s.decision(2))

    def test_window_order_and_no_cross_run_history(self):
        for seq in range(1,6):
            self.add(seq,value=seq*10); _,w,_=self.c.close(seq)
        self.assertEqual(w["x"].shape,(1,5,10,10))
        np.testing.assert_array_equal(w["x"][0,:,0,8], [10,20,30,40,50])
        self.assertIsNone(CausalCollector("new","boot",self.s).close(1)[1])

    def test_scenario_oracle_fields_do_not_change_tensors(self):
        other=CausalCollector("test","boot",self.s)
        p=report(self.s); p.update(label=1,source_oracle=[999],mode="A1",M="M1")
        other.receive(p,p["created_ns"])
        self.add(1)
        a,b=self.c.close(1)[0],other.close(1)[0]
        for k in a: np.testing.assert_array_equal(a[k],b[k])

    def test_exact_name_binding(self):
        class Worker:
            def poll(self): pass
        service=Service(self.c,Worker(),lambda x: None)
        event=wire(report(self.s), self.s.target("r1",1)+200_000_000)
        event["data_name"]+="/bad"
        with self.assertRaises(ValueError): service.ingest(event, self.s.decision(1))

    def test_cadence_changes_rejected(self):
        with self.assertRaises(ValueError): Schedule(0,interval_ns=1)

    def test_ipc_after_nominal_decision_cannot_backfill(self):
        p=report(self.s)
        self.assertEqual(self.c.receive(p,p["created_ns"],self.s.decision(1)+1),"ipc_after_decision_excluded")
        self.assertFalse(self.c.close(1)[0]["mask"].any())

    def test_invalid_availability_does_not_poison_report_key(self):
        p=report(self.s)
        with self.assertRaises(ValueError): self.c.receive(p,p["created_ns"],p["created_ns"]-1)
        self.assertEqual(self.c.receive(p,p["created_ns"],p["created_ns"]),"on_time")

    def test_non_object_transport_is_rejected(self):
        service=Service(self.c,None,lambda x: None)
        with self.assertRaises(ValueError): service.ingest([1,2])
        with self.assertRaises(ValueError): self.c.receive("[]",self.s.decision(1))


class AsyncTests(unittest.TestCase):
    def setUp(self):
        self.events=[]; self.now=[10_000_000_000]
        self.clock=lambda:self.now[0]

    def meta(self,seq=1): return {"decision_ns":self.now[0],"sequence":seq}

    def test_slow_worker_does_not_block_and_queue_is_bounded(self):
        entered=threading.Event(); release=threading.Event()
        class Slow:
            def predict(self,a):
                entered.set(); release.wait(2); return np.array([.7]),np.array([1])
        worker=AsyncInference(Slow(),self.events.append,"test",clock=self.clock)
        started=time.perf_counter(); worker.submit(window(),self.meta())
        self.assertLess(time.perf_counter()-started,.05)
        self.assertTrue(entered.wait(1))
        worker.submit(window(),self.meta(2)); self.assertFalse(worker.submit(window(),self.meta(3)))
        self.assertEqual(self.events[-1]["reason"],"queue_full")
        release.set(); worker.close()

    def test_timeout_discards_late_result(self):
        entered=threading.Event(); release=threading.Event()
        class Slow:
            def predict(self,a): entered.set(); release.wait(1); return [.7],[1]
        worker=AsyncInference(Slow(),self.events.append,"test",clock=self.clock,timeout_ms=20)
        worker.submit(window(),self.meta()); entered.wait(1)
        self.now[0]+=21_000_000; worker.poll()
        self.assertEqual(self.events[-1]["reason"],"inference_timeout")
        release.set(); time.sleep(.03); worker.poll(); worker.close()
        self.assertEqual(len(self.events),1)
        self.assertIsNone(self.events[0]["prediction"])

    def test_model_failure_has_unknown_output(self):
        class Broken:
            def predict(self,a): raise ValueError("bad")
        worker=AsyncInference(Broken(),self.events.append,"test",clock=self.clock)
        worker.submit(window(),self.meta())
        time.sleep(.03); worker.poll(); worker.close()
        self.assertEqual(self.events[0]["reason"],"runtime_error:ValueError")
        self.assertIsNone(self.events[0]["probability_ifa"])

    def test_nan_output_rejected(self):
        class Broken:
            def predict(self,a): return [float("nan")],[0]
        worker=AsyncInference(Broken(),self.events.append,"test",clock=self.clock)
        worker.submit(window(),self.meta()); time.sleep(.03); worker.poll(); worker.close()
        self.assertEqual(self.events[0]["status"],"unknown")

    def test_warmup_blackout_expired(self):
        worker=AsyncInference(None,self.events.append,"test",clock=self.clock)
        worker.submit(None,self.meta()); w=window(); w["mask"][0,-1]=0
        worker.submit(w,self.meta()); m=self.meta(); self.now[0]+=600_000_000
        worker.submit(window(),m); worker.close()
        self.assertEqual([e["reason"] for e in self.events], ["window_warmup","no_current_reports","decision_expired"])

    def test_window_copy_protects_against_caller_mutation(self):
        release=threading.Event(); values=[]
        class Runtime:
            def predict(self,a): release.wait(1); values.append(a["x"].sum()); return [.6],[1]
        worker=AsyncInference(Runtime(),self.events.append,"test",clock=self.clock)
        w=window(); worker.submit(w,self.meta()); w["x"][:]=0; release.set()
        time.sleep(.03); worker.poll(); worker.close()
        self.assertEqual(values,[500])
        self.assertEqual(self.events[0]["prediction"],1)


class TransportTests(unittest.TestCase):
    def test_audit_persists_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"audit.jsonl"; audit=JsonlAudit(p)
            for i in range(20): audit({"event":"test","i":i})
            audit.close()
            self.assertEqual(len(p.read_text().splitlines()),20)
            with self.assertRaises(FileExistsError): JsonlAudit(p)

    def test_audit_writer_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as d:
            audit=JsonlAudit(Path(d)/"audit.jsonl")
            audit({"bad":object()})
            audit.thread.join(timeout=1)
            with self.assertRaises(RuntimeError): audit.check()

    def test_cpp_nonblocking_bridge_and_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); endpoint=root/"ipc.sock"
            source=root/"test.cpp"
            source.write_text('#include "ml7_datagram_bridge.hpp"\nint main(int n,char**v){ Ml7DatagramBridge b(v[1]); return b.send("test-event") ? 0 : 9; }\n')
            subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-I",str(ROOT/"src"),str(source),"-o",str(root/"bridge")],check=True,capture_output=True)
            sock=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM); sock.bind(str(endpoint)); sock.settimeout(1)
            subprocess.run([str(root/"bridge"),str(endpoint)],check=True)
            self.assertEqual(sock.recv(1024),b"test-event"); sock.close(); endpoint.unlink()
            started=time.perf_counter()
            self.assertEqual(subprocess.run([str(root/"bridge"),str(endpoint)]).returncode,9)
            self.assertLess(time.perf_counter()-started,.2)

    def test_frozen_integrity_checks_work_under_optimized_python(self):
        package=ROOT/"dist/ml6_ndn_ifa_offline"
        if not package.exists(): package=ROOT/"vendor/ml6_ndn_ifa_offline"
        if not package.exists(): self.skipTest("Run this test in the ML repository")
        with tempfile.TemporaryDirectory() as d:
            clone=Path(d)/"package"; shutil.copytree(package,clone,ignore=shutil.ignore_patterns("__pycache__"))
            registry=json.loads((clone/"registry.json").read_text())
            model=next(m for m in registry["models"] if m["id"]=="ml5a_mask_age_s42")
            path=clone/model["encoder"]; path.write_text(path.read_text()+" ")
            code="import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from ml7_online import load_runtime; load_runtime(Path(sys.argv[2]),'ml5a_mask_age_s42')"
            r=subprocess.run([sys.executable,"-O","-c",code,str(ROOT/"src"),str(clone)],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn("integrity failure",r.stderr)

    def test_plain_model_rejected_at_online_boundary(self):
        package=ROOT/"dist/ml6_ndn_ifa_offline"
        if not package.exists(): package=ROOT/"vendor/ml6_ndn_ifa_offline"
        if not package.exists(): self.skipTest("Run this test in the ML repository")
        with self.assertRaises(ValueError): load_runtime(package,"ml5a_plain_s42")


if __name__ == "__main__": unittest.main(verbosity=2)
