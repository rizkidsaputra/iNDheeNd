"""Logika offline PoC11C; tidak membuka socket atau menjalankan workload."""
from dataclasses import dataclass
import hashlib, json, random

ROUTERS=tuple(f'r{i}' for i in range(1,11))
COUNTERS=('nInInterests','nOutInterests','nInData','nOutData','nInNacks','nOutNacks','nSatisfiedInterests','nUnsatisfiedInterests')
GAUGES=('nPitEntries','nCsEntries')
METRICS=COUNTERS+GAUGES

@dataclass(frozen=True)
class Schedule:
    start_ns:int
    count:int=50
    interval_ns:int=3_000_000_000
    stagger_ns:int=100_000_000
    offset_ns:int=300_000_000
    deadline_ns:int=1_900_000_000
    def generated_target(self,router,sequence):
        if router not in ROUTERS or not 1<=sequence<=self.count:raise ValueError('Identitas schedule tidak valid')
        return self.start_ns+(sequence-1)*self.interval_ns+ROUTERS.index(router)*self.stagger_ns
    def deadline(self,router,sequence):return self.generated_target(router,sequence)+self.deadline_ns
    def decision(self,sequence):return self.deadline(ROUTERS[-1],sequence)

@dataclass(frozen=True)
class Received:
    router_id:str
    sequence:int
    created_ns:int
    received_ns:int
    nfd_instance_id:str
    metrics:dict
    raw_sha256:str

class CollectorLedger:
    """Ledger hanya menerima payload collector; tidak menerima source oracle."""
    def __init__(self,run_id,boot_id,schedule):
        self.run_id=run_id;self.boot_id=boot_id;self.schedule=schedule;self.arrivals=[]
    def receive(self,raw,received_ns):
        p=json.loads(raw)
        if p.get('run_id')!=self.run_id or p.get('boot_id')!=self.boot_id:raise ValueError('Run/boot berbeda')
        router=p['router_id'];seq=p['sequence'];created=p['created_ns']
        if type(seq) is not int or type(created) is not int or type(received_ns) is not int:raise ValueError('Timestamp/sequence bukan integer')
        scheduled=self.schedule.generated_target(router,seq)
        if not scheduled<=created<=received_ns:raise ValueError('Urutan timestamp salah')
        metrics=p['metrics']
        if set(metrics)!=set(METRICS) or any(type(v) is not int or v<0 for v in metrics.values()):raise ValueError('Skema metrik tidak valid')
        instance=p['nfd_instance_id']
        if not isinstance(instance,str) or not instance:raise ValueError('NFD instance tidak tersedia')
        if any(r.router_id==router and r.sequence==seq for r in self.arrivals):raise ValueError('Payload sequence duplikat')
        record=Received(router,seq,created,received_ns,instance,metrics,hashlib.sha256(raw).hexdigest())
        self.arrivals.append(record)
        return record
    def snapshot(self,sequence,age_cap_seconds=30):
        decision=self.schedule.decision(sequence);output={}
        for router in ROUTERS:
            known=[r for r in self.arrivals if r.router_id==router and r.received_ns<=decision and r.sequence<=sequence]
            # Epoch mengikuti instance yang terakhir terlihat; tidak menjembatani reset.
            if not known:
                output[router]={'has_report':False,'mask':0,'on_time_mask':0,'age_seconds':None,'age_normalized':1.0,'features':None,'rate_valid':False,'reset_detected':False,'source_sequence':None}
                continue
            epoch=max(known,key=lambda r:r.created_ns).nfd_instance_id
            same=sorted([r for r in known if r.nfd_instance_id==epoch],key=lambda r:r.created_ns)
            latest=same[-1];previous=same[-2] if len(same)>1 else None
            changed_instance=any(r.nfd_instance_id!=epoch for r in known)
            reset=changed_instance or bool(previous and any(latest.metrics[k]<previous.metrics[k] for k in COUNTERS))
            valid=previous is not None and latest.created_ns>previous.created_ns and not reset
            rates=[(latest.metrics[k]-previous.metrics[k])/((latest.created_ns-previous.created_ns)/1e9) for k in COUNTERS] if valid else None
            age=(decision-latest.created_ns)/1e9
            output[router]={'has_report':True,'mask':int(latest.sequence==sequence),
                'on_time_mask':int(latest.sequence==sequence and latest.received_ns<=self.schedule.deadline(router,sequence)),
                'age_seconds':age,'age_normalized':min(age,age_cap_seconds)/age_cap_seconds,
                'features':rates+[float(latest.metrics[k]) for k in GAUGES] if valid else None,
                'rate_valid':valid,'reset_detected':reset,'source_sequence':latest.sequence,'created_ns':latest.created_ns,'received_ns':latest.received_ns}
        return {'sequence':sequence,'decision_ns':decision,'routers':output}

def model_vector(observation):
    if not observation['has_report']:return [0.0]*10+[0.0,1.0]
    if not observation['rate_valid']:raise ValueError('Rate belum valid: warm-up belum lengkap atau reset')
    return observation['features']+[float(observation['mask']),observation['age_normalized']]

def attack_active(decision_ns,actual_emission_intervals):
    """Ground truth terpisah: union interval emisi aktual [start, stop)."""
    return any(x['start_ns']<=decision_ns<x['stop_ns'] for x in actual_emission_intervals)

def router_edges(topology_text):
    edges=set();inlinks=False
    for line in topology_text.splitlines():
        line=line.split('#',1)[0].strip()
        if line.startswith('['):inlinks=line=='[links]';continue
        if inlinks and line:
            pair=line.split()[0].lower().split(':')
            if len(pair)==2 and all(n in ROUTERS for n in pair):
                a,b=pair
                if a==b:raise ValueError('Self-loop tidak ada pada topologi fisik')
                edges.add(tuple(sorted((a,b),key=ROUTERS.index)))
    return sorted(edges,key=lambda e:(ROUTERS.index(e[0]),ROUTERS.index(e[1])))

def candidate_order(seeds=(11101,11102,11103)):
    cells=[(w,m) for w in ['B0','B1','A1'] for m in ['M0','M1']];rows=[];last=None
    for block,seed in enumerate(seeds,1):
        rng=random.Random(seed)
        for _ in range(10000):
            order=rng.sample(cells,len(cells))
            if (last is None or order[0][0]!=last) and all(a[0]!=b[0] for a,b in zip(order,order[1:])):break
        else:raise RuntimeError('Tidak menemukan urutan blok')
        for index,(w,m) in enumerate(order,1):
            rows.append({'block':block,'seed':seed,'position':index,'workload':w,'monitoring':m,'onset_seconds':(45,60,75)[block-1],'stop_seconds':(90,105,120)[block-1],'status':'proposed_not_approved'})
        last=order[-1][0]
    return rows
