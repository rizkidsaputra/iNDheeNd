"""Pilot terikat jaringan lokal, evidence terpisah, tanpa restart otomatis."""
from pathlib import Path
import argparse,collections,datetime as dt,hashlib,json,os,re,signal,statistics,subprocess as sp,threading,time,traceback
from poc11c_core_v1 import CollectorLedger,Schedule,COUNTERS,METRICS
BASE=Path('/home/ndn/research/experiments/poc11')
SCRIPTS=BASE/'scripts/poc11c'
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writej(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def lines(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def command(args,timeout=10):
    r=sp.run(args,capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError({'command':args,'code':r.returncode,'stderr':r.stderr})
    return r.stdout
def pid(n):
    a=command(['pgrep','-f',f'^nfd --config /tmp/minindn/{n}/nfd.conf$']).split()
    if len(a)!=1:raise RuntimeError(f'{n}: NFD count {len(a)}')
    return int(a[0])
def tick(p):return Path(f'/proc/{p}/stat').read_text().rpartition(') ')[2].split()[19]
def ns(n,*a):return ['mnexec','-a',str(pid(n)),'env',f'NDN_CLIENT_TRANSPORT=unix:///run/nfd/{n}.sock',*map(str,a)]
def nfd(n,*a):return command(ns(n,'nfdc',*a),5)
def cpu():
    a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]));return sum(a),a[3]+a[4]
def memory():return {l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if len(l.split())>=2 and l.split()[1].isdigit()}
def counts():
    ps=command(['ps','-eo','args'])
    return sum(bool(re.fullmatch(r'nfd --config /tmp/minindn/\w+/nfd.conf',s)) for s in ps.splitlines()),sum(bool(re.fullmatch(r'nlsr -f /tmp/minindn/\w+/nlsr.conf',s)) for s in ps.splitlines())
def kernel():return command(['journalctl','-k','-b','--no-pager'],5)
def preflight(out,routers):
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert counts()==(21,21),'Topologi tidak lengkap'
    mem=memory();assert mem['MemAvailable']>=2*1024*1024,'RAM baseline <2GiB'
    assert os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize>=10*1024**3,'Disk <10GiB'
    k=kernel();assert 'soft lockup' not in k.lower(),'Soft lockup current boot'
    fib=nfd('m','fib','list')
    for r in routers:assert f'/monitor/{r}/telemetry ' in fib,f'Route collector hilang {r}'
    for n in ['c1','c2','c3','c4','c5','c6']:
        f=nfd(n,'fib','list');p='p1' if int(n[1:])<=3 else 'p2'
        assert f'/poc11c/legitimate/{p} ' in f,f'Route producer hilang {n}'
    for n in ['a1','a2']:assert '/poc11c/attack/unavailable ' in nfd(n,'fib','list'),f'Route IFA hilang {n}'
    nodes=[f'r{i}' for i in range(1,11)]+[f'c{i}' for i in range(1,7)]+['a1','a2','p1','p2','m']
    instances={n:{'pid':pid(n),'ticks':tick(pid(n)),'netns':os.readlink(f'/proc/{pid(n)}/ns/net')} for n in nodes}
    assert len({v['netns'] for v in instances.values()})==21
    (out/'kernel_before.log').write_text(k)
    effective=out/'effective_configs';effective.mkdir()
    for n in nodes:
        for kind in ['nfd','nlsr']:(effective/f'{n}_{kind}.conf').write_bytes(Path(f'/tmp/minindn/{n}/{kind}.conf').read_bytes())
    q={n:{'qdisc':command(ns(n,'tc','-j','-s','qdisc','show','dev',d)), 'class':command(ns(n,'tc','-s','class','show','dev',d))} for n,d in [('r5','r5-eth4'),('r6','r6-eth0')]}
    for v in q.values():assert 'rate 5Mbit ceil 5Mbit' in v['class'],'Bottleneck berubah'
    x={'utc':utc(),'boot_id':boot,'instances':instances,'memory_kb':mem,'counts':[21,21],'m_fib':fib,'bottleneck':q,'kernel':command(['uname','-a']),'source_sha256':{p.name:sha(p) for p in SCRIPTS.glob('*') if p.is_file()}}
    writej(out/'environment.json',x);return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['telemetry','B0','B1','A1'],required=True);p.add_argument('--routers',default=','.join(f'r{i}' for i in range(1,11)));p.add_argument('--duration',type=int,default=60);p.add_argument('--benign-rate',type=float,default=5);p.add_argument('--attack-rate',type=float,default=25);p.add_argument('--passive-seconds',type=int,default=15);p.add_argument('--run-id',required=True)
    a=p.parse_args();assert a.duration>=30 and a.duration%3==0;assert 0<a.benign_rate<=55 and 0<a.attack_rate<=150
    assert a.mode!='A1' or 30+2*a.attack_rate<=330
    routers=a.routers.split(',');assert set(routers)<=set(f'r{i}' for i in range(1,11))
    out=BASE/'results/poc11c/pilots'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    config=vars(a)|{'status':'authorized_limited_pilot','interval_s':3,'stagger_s':.1,'request_offset_s':.3,'deadline_s':1.9,'interest_lifetime_telemetry_ms':3600,'seed':11101,'data_payload_bytes':1024,'attack_lifetime_ms':4000,'benign_lifetime_ms':2000,'no_retransmission':True,'M':'M0','onset_s':15,'active_s':30,'source_generation_limit_ms':1000,'source_sampling_method':'persistent_nfd_controller','main_batch':False,'command_line':list(__import__('sys').argv),'created_utc':utc()}
    writej(out/'config.json',config)
    stop=threading.Event();causes=[];procs=[];streams=[];threads=[];lock=threading.Lock();health_rows=[]
    def abort(reason):
        with lock:
            if reason not in causes:causes.append(reason);print(json.dumps({'abort':reason,'utc':utc()}),flush=True)
        stop.set()
    def append(path,obj):
        with lock:
            with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    def launch(node,args,filename):
        path=out/filename;f=path.open('x');streams.append(f);cmd=ns(node,*args)
        if 'reporter' in [str(x) for x in args]:
            v=env['instances'][node]
            cmd=ns(node,'env',f"POC11C_NFD_INSTANCE={env['boot_id']}:{v['pid']}:{v['ticks']}",*args)
        append(out/'commands.jsonl',{'utc':utc(),'command':cmd,'node':node})
        process=sp.Popen(cmd,stdout=f,stderr=sp.STDOUT,start_new_session=True);procs.append((process,tick(process.pid),node))
        return process
    env=None;start=None
    try:
        env=preflight(out,routers);baseline_swap=memory()['SwapTotal']-memory()['SwapFree'];boot=env['boot_id']
        def health():
            oldcpu=cpu();previous=time.monotonic_ns();busy_since=None;i=0
            stop.wait(1)  # CPU pertama memakai interval nyata, bukan delta mikrodetik.
            while not stop.is_set():
                try:
                    stamp=time.monotonic_ns();c=cpu();mem=memory();busy=100*(1-(c[1]-oldcpu[1])/(c[0]-oldcpu[0])) if c[0]!=oldcpu[0] else 0;oldcpu=c
                    row={'utc':utc(),'monotonic_ns':stamp,'sample_gap_s':(stamp-previous)/1e9,'cpu_busy_percent':busy,'mem_available_kb':mem['MemAvailable'],'swap_used_kb':mem['SwapTotal']-mem['SwapFree'],'counts':counts()};previous=stamp
                    row['nfd_instances_unchanged']=all(Path(f"/proc/{v['pid']}").exists() and tick(v['pid'])==v['ticks'] for v in env['instances'].values())
                    health_rows.append(row);append(out/'health.jsonl',row)
                    if row['sample_gap_s']>5:abort('health_gap_gt5s')
                    if row['counts']!=(21,21):abort('nfd_nlsr_count_changed')
                    if not row['nfd_instances_unchanged']:abort('nfd_instance_changed')
                    if mem['MemAvailable']<1536*1024:abort('available_ram_lt1_5GiB')
                    if row['swap_used_kb']>baseline_swap+1024:abort('swap_growth')
                    if busy>85:
                        if busy_since is None:busy_since=stamp
                        elif stamp-busy_since>10e9:abort('cpu_busy_gt85pct_10s')
                    else:busy_since=None
                    if i%3==0:
                        if 'soft lockup' in kernel().lower():abort('kernel_soft_lockup')
                        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=boot:abort('boot_changed')
                    if i%3==0 and start is not None:
                        append(out/'routing.jsonl',{'monotonic_ns':stamp,'m_fib':nfd('m','fib','list'),'a1_fib':nfd('a1','fib','list'),'a2_fib':nfd('a2','fib','list')})
                    if i%2==0:
                        q={'monotonic_ns':stamp}
                        for n,d in [('r5','r5-eth4'),('r6','r6-eth0')]:
                            q[n]=json.loads(command(ns(n,'tc','-j','-s','qdisc','show','dev',d),3))
                        append(out/'qdisc.jsonl',q)
                    i+=1;stop.wait(1)
                except Exception as e:abort('health_exception:'+str(e))
        th=threading.Thread(target=health,daemon=True);th.start();threads.append(th)
        stop.wait(a.passive_seconds)
        if stop.is_set():raise RuntimeError('Passive safety gate failed')
        baseline=[v['cpu_busy_percent'] for v in health_rows]
        if statistics.mean(baseline)>50:raise RuntimeError('Passive CPU mean >50%')
        start=time.monotonic_ns()+5_000_000_000;config['start_ns']=start;config['boot_id']=boot;writej(out/'config.json',config)
        source=out/'source_payloads';source.mkdir()
        for r in routers:(source/r).mkdir()
        for r in routers:launch(r,[SCRIPTS/'poc11c_apps_v5','reporter',r,a.run_id,start,a.duration,0,source],f'reporter_{r}.jsonl')
        collector=launch('m',[SCRIPTS/'poc11c_apps_v5','collector','m',a.run_id,start,a.duration,0,a.routers],'collector_wire.jsonl')
        if a.mode!='telemetry':
            for n in ['p1','p2']:launch(n,[SCRIPTS/'poc11c_apps_v5','producer',n,a.run_id,start,a.duration,0,f'/poc11c/legitimate/{n}'],f'producer_{n}.jsonl')
            for i in range(1,7):
                n=f'c{i}';prefix='/poc11c/legitimate/'+('p1' if i<=3 else 'p2')
                phases=[(0,a.duration,5,'base')] if a.mode!='B1' else [(0,15,5,'pre'),(15,30,a.benign_rate,'high'),(45,a.duration-45,5,'post')]
                for offset,duration,rate,phase in phases:
                    if duration>0:launch(n,[SCRIPTS/'poc11c_apps_v5','consumer',n,a.run_id+'_'+phase,start+int(offset*1e9),duration,rate,prefix],f'workload_{n}_{phase}.jsonl')
            if a.mode=='A1':
                prefix='/poc11c/attack/unavailable/'+a.run_id
                launch('p1',[SCRIPTS/'poc11c_silent_sink_v1','--execute-after-pilot-approval',prefix],'silent_sink.jsonl')
                for n in ['a1','a2']:launch(n,[SCRIPTS/'poc11c_apps_v5','consumer',n,a.run_id,start+15_000_000_000,30,a.attack_rate,prefix],f'workload_{n}_attack.jsonl')
        time.sleep(.3)
        # Registrasi prefix lokal harus terjadi sebelum jadwal pertama.
        for r in routers:
            if not any(x.get('event')=='prefix_registered' for x in lines(out/f'reporter_{r}.jsonl')):raise RuntimeError('Reporter belum registered '+r)
        if a.mode!='telemetry':
            for n in ['p1','p2']:
                if not any(x.get('event')=='prefix_registered' for x in lines(out/f'producer_{n}.jsonl')):raise RuntimeError('Producer belum registered '+n)
        if a.mode=='A1' and not any(x.get('event')=='sink_prefix_registered' for x in lines(out/'silent_sink.jsonl')):raise RuntimeError('Silent sink belum registered')
        def sample(r):
            idx=int(r[1:])-1;instance=f"{boot}:{env['instances'][r]['pid']}:{env['instances'][r]['ticks']}"
            for seq in range(1,a.duration//3+1):
                target=start+(seq-1)*3_000_000_000+idx*100_000_000
                if stop.wait(max(0,(target-time.monotonic_ns())/1e9)):return
                try:
                    begin=time.monotonic_ns();raw=nfd(r,'status','show');end=time.monotonic_ns();general=raw.split('Channels:',1)[0]
                    metrics={k:int(re.search(rf'(?m)^\s*{k}=(\d+)\s*$',general)[1]) for k in METRICS}
                    payload={'run_id':a.run_id,'boot_id':boot,'router_id':r,'sequence':seq,'created_ns':end,'nfd_instance_id':instance,'metrics':metrics,'sample_begin_ns':begin,'sample_end_ns':end}
                    encoded=json.dumps(payload,separators=(',',':')).encode();tmp=source/r/f'{seq}.tmp';tmp.write_bytes(encoded);tmp.rename(source/r/f'{seq}.json')
                    append(out/'source_generated.jsonl',payload|{'target_ns':target,'generation_delay_ms':(end-target)/1e6,'raw_sha256':hashlib.sha256(encoded).hexdigest()})
                    if end-target>500_000_000:abort('telemetry_generation_delay_gt500ms:'+r)
                except Exception as e:abort('source_sampling_exception:'+r+':'+str(e));return
        # v4: sampling melalui Controller/Face persisten di reporter; bukan subprocess sampler.
        deadline=start+int((a.duration+6)*1e9)
        while time.monotonic_ns()<deadline and not stop.wait(.2):
            if collector.poll() is not None and time.monotonic_ns()<start+int((a.duration+4)*1e9):abort('collector_exited_early')
            if any(proc.poll() not in (None,0) for proc,_,_ in procs):abort('app_nonzero_exit')
            for r in routers:
                events=lines(out/f'reporter_{r}.jsonl')
                if any(x.get('event') in ('source_query_error','source_generation_safety_abort') for x in events):abort('persistent_source_sampling_failure:'+r)
            # Systematic failure is checked once five frames have had a chance to arrive.
            if time.monotonic_ns()>start+17_000_000_000:
                rec=[x for x in lines(out/'collector_wire.jsonl') if x.get('event')=='telemetry_received']
                if len(rec)<len(routers)*3:abort('systematic_telemetry_failure')
        stop.set()
    except Exception as e:abort('runner_exception:'+str(e));(out/'exception.txt').write_text(traceback.format_exc())
    finally:
        stop.set()
        for proc,started,node in procs:
            if proc.poll() is None:
                try:
                    if tick(proc.pid)==started:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for proc,started,node in procs:
            try:proc.wait(timeout=3)
            except sp.TimeoutExpired:
                if proc.poll() is None and tick(proc.pid)==started:os.killpg(proc.pid,signal.SIGKILL)
        for th in threads:th.join(timeout=6)
        for f in streams:f.close()
        if start is not None:
            for r in routers:
                for event in lines(out/f'reporter_{r}.jsonl'):
                    if event.get('event')=='source_generated':
                        payload=event['payload'];raw=(out/'source_payloads'/r/f"{payload['sequence']}.json").read_bytes()
                        append(out/'source_generated.jsonl',payload|{'target_ns':event['target_ns'],'generation_delay_ms':event['generation_delay_ms'],'raw_sha256':hashlib.sha256(raw).hexdigest()})
        (out/'kernel_after.log').write_text(kernel())
        writej(out/'manifest.json',{'run_id':a.run_id,'mode':a.mode,'status':'aborted' if causes else 'collected_pending_validation','abort_causes':causes,'start_ns':start,'ended_utc':utc(),'boot_id':env['boot_id'] if env else None,'main_batch':False,'pilot_authorized':True})
        validate(out)
        checksums=''.join(f'{sha(f)}  {f.relative_to(out).as_posix()}\n' for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS')
        (out/'SHA256SUMS').write_text(checksums)
        print(json.dumps({'out':str(out),'validation':json.loads((out/'validation.json').read_text())}),flush=True)

def validate(out):
    c=json.loads((out/'config.json').read_text());m=json.loads((out/'manifest.json').read_text());g=lines(out/'source_generated.jsonl');wire=lines(out/'collector_wire.jsonl');rec=[x for x in wire if x.get('event')=='telemetry_received'];health=lines(out/'health.jsonl')
    ledger=CollectorLedger(c['run_id'],m['boot_id'],Schedule(c.get('start_ns',0),count=c['duration']//3));errors=[];received=[]
    lookup={(x['router_id'],x['sequence']):x for x in g}
    for row in rec:
        try:
            raw=row['raw_payload'].encode();r=ledger.receive(raw,row['monotonic_ns']);matching=lookup.get((r.router_id,r.sequence),{}).get('raw_sha256')==r.raw_sha256
            if not matching:errors.append('source_wire_hash_mismatch')
            received.append({'router_id':r.router_id,'sequence':r.sequence,'created_ns':r.created_ns,'received_ns':r.received_ns,'raw_sha256':r.raw_sha256,'source_hash_match':matching,'on_time':r.received_ns<=ledger.schedule.deadline(r.router_id,r.sequence),'payload':json.loads(raw)})
        except Exception as e:errors.append('receive_validation:'+str(e))
    with (out/'collector_received.jsonl').open('w') as f:
        for row in received:f.write(json.dumps(row)+'\n')
    with (out/'observed_frames.jsonl').open('w') as f:
        for seq in range(1,c['duration']//3+1):f.write(json.dumps(ledger.snapshot(seq))+'\n')
    workloads={};labels=[]
    for path in out.glob('workload_*.jsonl'):
        data=lines(path);emits=[r for r in data if r.get('event')=='interest_submitted'];valid=[r for r in data if r.get('event')=='data_received' and r.get('valid_content') and r.get('exact_name')];nacks=[r for r in data if r.get('event')=='nack'];timeouts=[r for r in data if r.get('event')=='timeout']
        targets=[r['target_ns'] for r in emits];planned_active=30 if '_high' in path.stem or '_attack' in path.stem else (c['duration'] if '_base' in path.stem else 15)
        achieved=len(emits)/planned_active;actual_rate=(len(emits)-1)/((emits[-1]['monotonic_ns']-emits[0]['monotonic_ns'])/1e9) if len(emits)>1 else None
        workloads[path.stem]={'submitted':len(emits),'valid_data':len(valid),'nacks':len(nacks),'nack_reasons':dict(collections.Counter(str(r['reason']) for r in nacks)),'timeouts':len(timeouts),'offered_over_planned_interval_per_s':achieved,'actual_span_rate_per_s':actual_rate,'unique_names':len({r['name'] for r in emits}),'max_submit_lag_ms':max(((r['monotonic_ns']-r['target_ns'])/1e6 for r in emits),default=None)}
        if '_attack' in path.stem and emits:labels.append({'node':emits[0]['node'],'start_ns':emits[0]['monotonic_ns'],'last_submit_ns':emits[-1]['monotonic_ns'],'stop_scheduled_ns':c['start_ns']+45_000_000_000,'label_basis':'actual first submission and final scheduled active interval; inspect per-event gaps'})
    writej(out/'ground_truth.json',{'mode':c['mode'],'benign_labels':'Normal','attack_emission_intervals':labels,'label_source':'workload application events, not PIT/collector availability','ifa_confirmed':False,'confirmation_status':'requires independent validation below'})
    stats={}
    start=c.get('start_ns',0)
    for r in c['routers'].split(','):
        rows=sorted([x for x in g if x['router_id']==r],key=lambda x:x['sequence']);pre=[x['metrics']['nPitEntries'] for x in rows if x['created_ns']<start+15e9];active=[x['metrics']['nPitEntries'] for x in rows if start+15e9<=x['created_ns']<start+45e9];post=[x['metrics']['nPitEntries'] for x in rows if x['created_ns']>=start+45e9]
        stats[r]={'generated':len(rows),'received':sum(x['router_id']==r for x in received),'pit_pre_mean':statistics.mean(pre) if pre else None,'pit_active_max':max(active) if active else None,'pit_post_last':post[-1] if post else None,'counter_delta':{k:rows[-1]['metrics'][k]-rows[0]['metrics'][k] for k in COUNTERS} if len(rows)>1 else None,'max_generation_delay_ms':max((x['generation_delay_ms'] for x in rows),default=None)}
    sink=lines(out/'silent_sink.jsonl');sinkreceived=[x for x in sink if x.get('event')=='sink_interest_received']
    totalbenign=sum(x['submitted'] for name,x in workloads.items() if 'workload_c' in name);benignvalid=sum(x['valid_data'] for name,x in workloads.items() if 'workload_c' in name);attacksubmitted=sum(x['submitted'] for name,x in workloads.items() if 'workload_a' in name)
    report={'status':m['status'],'abort_causes':m['abort_causes'],'validation_errors':errors,'routers_requested':c['routers'].split(','),'routers_with_received':sorted({x['router_id'] for x in received},key=lambda r:int(r[1:])),'generated':len(g),'received':len(received),'on_time_received':sum(x['on_time'] for x in received),'expected_generated':len(c['routers'].split(','))*(c['duration']//3),'workloads':workloads,'benign_submitted':totalbenign,'benign_valid_data':benignvalid,'benign_satisfaction_fraction':benignvalid/totalbenign if totalbenign else None,'attack_submitted':attacksubmitted,'sink_received':len(sinkreceived),'sink_unique_names':len({r['name'] for r in sinkreceived}),'sink_data_generated':sum(bool(x.get('data_generated')) for x in sinkreceived),'source_router_metrics':stats,'cpu_mean_percent':statistics.mean(x['cpu_busy_percent'] for x in health) if health else None,'cpu_max_percent':max((x['cpu_busy_percent'] for x in health),default=None),'min_memory_available_kb':min((x['mem_available_kb'] for x in health),default=None),'max_health_gap_s':max((x['sample_gap_s'] for x in health),default=None),'all_health_counts_21':all(x['counts']==[21,21] for x in health),'nfd_instances_unchanged':all(x['nfd_instances_unchanged'] for x in health),'hash_provenance_match':not errors,'ifa_validated':False}
    writej(out/'validation.json',report)
if __name__=='__main__':main()





