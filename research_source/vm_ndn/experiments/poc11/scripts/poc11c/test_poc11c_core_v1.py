"""Pengujian temporal, reset, label dan pemisahan informasi tanpa jaringan."""
import json,unittest
from poc11c_core_v1 import *

class Checks(unittest.TestCase):
    def setUp(self):self.schedule=Schedule(1_000_000_000);self.ledger=CollectorLedger('unit','boot',self.schedule)
    def raw(self,seq=1,router='r1',value=10,instance='boot:pid:tick',extra=None):
        p={'run_id':'unit','boot_id':'boot','router_id':router,'sequence':seq,'created_ns':self.schedule.generated_target(router,seq)+10_000_000,'nfd_instance_id':instance,'metrics':{k:value for k in METRICS}}
        p.update(extra or {});return json.dumps(p).encode()
    def add(self,seq,value=10,lag_ns=100_000_000,**kw):return self.ledger.receive(self.raw(seq,value=value,**kw),self.schedule.generated_target('r1',seq)+lag_ns)
    def test_cold_has_no_oracle(self):
        x=self.ledger.snapshot(1)['routers']['r1'];self.assertIsNone(x['features']);self.assertEqual(model_vector(x),[0.0]*11+[1.0])
    def test_future_arrival_invisible(self):
        self.add(1,lag_ns=3_000_000_000);self.assertFalse(self.ledger.snapshot(1)['routers']['r1']['has_report'])
    def test_late_available_at_global_decision(self):
        self.add(1,lag_ns=2_000_000_000);x=self.ledger.snapshot(1)['routers']['r1'];self.assertEqual((x['mask'],x['on_time_mask']),(1,0))
    def test_stale_age_increases(self):
        self.add(1);a=self.ledger.snapshot(1)['routers']['r1'];b=self.ledger.snapshot(2)['routers']['r1'];self.assertEqual(b['mask'],0);self.assertAlmostEqual(b['age_seconds']-a['age_seconds'],3)
    def test_interval_rate_over_gap(self):
        self.add(1,value=10);self.add(3,value=70);x=self.ledger.snapshot(3)['routers']['r1'];self.assertEqual(x['features'][:8],[10.0]*8)
    def test_out_of_order_does_not_replace_newer(self):
        self.add(2,value=20);self.add(1,value=10,lag_ns=5_000_000_000);x=self.ledger.snapshot(2)['routers']['r1'];self.assertEqual(x['source_sequence'],2)
    def test_counter_reset_no_negative_rate(self):
        self.add(1,value=100);self.add(2,value=1);x=self.ledger.snapshot(2)['routers']['r1'];self.assertTrue(x['reset_detected']);self.assertIsNone(x['features']);self.assertRaises(ValueError,model_vector,x)
    def test_instance_reset(self):
        self.add(1);self.add(2,value=20,instance='boot:new:tick');self.assertTrue(self.ledger.snapshot(2)['routers']['r1']['reset_detected'])
    def test_old_epoch_late_packet_never_reverts_instance(self):
        self.add(2,value=20,instance='boot:new:tick');self.add(1,lag_ns=5_000_000_000);self.assertEqual(self.ledger.snapshot(2)['routers']['r1']['source_sequence'],2)
    def test_boot_change_rejected(self):self.assertRaises(ValueError,self.ledger.receive,self.raw(extra={'boot_id':'new'}),2_000_000_000)
    def test_duplicate_rejected(self):self.add(1);self.assertRaises(ValueError,self.add,1)
    def test_future_source_rejected(self):self.assertRaises(ValueError,self.ledger.receive,self.raw(extra={'created_ns':3_000_000_000}),2_000_000_000)
    def test_incomplete_metrics_rejected(self):self.assertRaises(ValueError,self.ledger.receive,self.raw(extra={'metrics':{'nPitEntries':1}}),2_000_000_000)
    def test_privileged_metadata_not_features(self):
        self.add(1);self.add(2,value=20,extra={'attack_active':True,'seed':11101,'workload':'A1'});vector=model_vector(self.ledger.snapshot(2)['routers']['r1']);self.assertEqual(len(vector),12);self.assertEqual(vector[:8],[10/3]*8)
    def test_numeric_router_order_and_no_overlap(self):
        self.assertEqual(ROUTERS[-1],'r10');self.assertEqual(self.schedule.decision(1),3_800_000_000);self.assertLess(self.schedule.decision(1),self.schedule.generated_target('r1',2))
    def test_label_interval_boundaries(self):
        events=[{'start_ns':10,'stop_ns':20},{'start_ns':12,'stop_ns':22}];self.assertFalse(attack_active(9,events));self.assertTrue(attack_active(20,events));self.assertFalse(attack_active(22,events))
    def test_order_full_factorial_and_seed(self):
        a=candidate_order();self.assertEqual(a,candidate_order());self.assertEqual(len(a),18)
        for b in [1,2,3]:self.assertEqual(len({(r['workload'],r['monitoring']) for r in a if r['block']==b}),6)
    def test_router_graph_from_source(self):
        edges=router_edges('[links]\nR1:R5 delay=5ms\nC1:R1 delay=5ms\nR5:R1 delay=5ms\nR5:R6 bw=5');self.assertEqual(edges,[('r1','r5'),('r5','r6')])
    def test_first_report_not_fabricated_rate(self):
        self.add(1);x=self.ledger.snapshot(1)['routers']['r1'];self.assertFalse(x['rate_valid']);self.assertRaises(ValueError,model_vector,x)

if __name__=='__main__':unittest.main(verbosity=2)
