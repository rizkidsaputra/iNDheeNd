"""Audit PoC11C: hanya membaca status, tanpa workload atau perubahan routing."""
from pathlib import Path
import datetime as dt, hashlib, json, os, re, subprocess as sp, time

NODES=[f'r{i}' for i in range(1,11)]+[f'c{i}' for i in range(1,7)]+['a1','a2','p1','p2','m']
ROUTERS=NODES[:10]
FIELDS=['nInInterests','nOutInterests','nInData','nOutData','nInNacks','nOutNacks','nSatisfiedInterests','nUnsatisfiedInterests','nPitEntries','nCsEntries']
def cmd(args):
    r=sp.run(args,capture_output=True,text=True,timeout=12)
    return {'command':args,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def must(args):
    r=cmd(args)
    if r['exit_code']:raise RuntimeError(r)
    return r['stdout']
def pid(node):
    values=must(['pgrep','-f',f'^nfd --config /tmp/minindn/{node}/nfd.conf$']).split()
    assert len(values)==1,(node,values)
    return int(values[0])
def ns(node,*args):return ['mnexec','-a',str(pid(node)),*args]
def nfd(node,*args):return ns(node,'env',f'NDN_CLIENT_TRANSPORT=unix:///run/nfd/{node}.sock','nfdc',*args)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cpustat():
    fields=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(fields),fields[3]+fields[4]
stamp=lambda: {'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'monotonic_ns':time.monotonic_ns(),'wall_ns':time.time_ns()}
result={'audit_started':stamp(),'hostname':must(['hostname']).strip(),'user':must(['id']).strip(),
        'kernel':must(['uname','-a']).strip(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'uptime':must(['uptime']).strip(),'cpu_count':os.cpu_count(),'memory':Path('/proc/meminfo').read_text(),
        'disk':must(['df','-B1','/','/tmp']),'processes':must(['ps','-eo','pid,ppid,pcpu,pmem,args']),
        'frozen_monitor_sha256':sha('/home/ndn/research/experiments/poc11/scripts/monitor11c_final.py'),
        'topology_sha256':sha('/home/ndn/research/experiments/iNDheeNd/minindn_topology.conf')}
t0,idle0=cpustat();time.sleep(5);t1,idle1=cpustat()
result['cpu_busy_percent_5s']=100*(1-(idle1-idle0)/(t1-t0))
result['nodes']={}
for node in NODES:
    p=pid(node); stat=Path(f'/proc/{p}/stat').read_text().rpartition(') ')[2].split(); ticks=int(stat[19])
    item={'nfd_pid':p,'nfd_start_ticks':ticks,'nfd_instance_id':f"{result['boot_id']}:{p}:{ticks}",
          'network_namespace':os.readlink(f'/proc/{p}/ns/net'),'socket_exists':Path(f'/run/nfd/{node}.sock').exists(),
          'interfaces':json.loads(must(ns(node,'ip','-j','address','show'))),
          'nfd_config_sha256':sha(f'/tmp/minindn/{node}/nfd.conf'),'nlsr_config_sha256':sha(f'/tmp/minindn/{node}/nlsr.conf')}
    if node in ROUTERS:
        status=must(nfd(node,'status','report'));general=status.split('Channels:',1)[0]
        item['status_report']=status
        item['metrics']={field:int(re.search(rf'(?m)^\s*{field}=(\d+)\s*$',general)[1]) for field in FIELDS}
    if node in ['m','p1','p2','r1','r2','r5','r6','r7','r10']:
        item['fib']=must(nfd(node,'fib','list'));item['faces']=must(nfd(node,'face','list'))
    result['nodes'][node]=item
result['unique_network_namespaces']=len({x['network_namespace'] for x in result['nodes'].values()})
result['nfd_process_count']=len(re.findall(r'(?m)^.*nfd --config /tmp/minindn/\w+/nfd.conf$',result['processes']))
result['nlsr_process_count']=len(re.findall(r'(?m)^.*nlsr -f /tmp/minindn/\w+/nlsr.conf$',result['processes']))
result['socket_count']=len(list(Path('/run/nfd').glob('*.sock')))
result['reporter_fib_available_at_m']={r:bool(re.search(rf'(?m)^\s*/monitor/{r}/telemetry ',result['nodes']['m']['fib'])) for r in ROUTERS}
result['bottleneck']={}
for node,dev in [('r5','r5-eth4'),('r6','r6-eth0')]:
    result['bottleneck'][node]={'qdisc':json.loads(must(ns(node,'tc','-j','-s','qdisc','show','dev',dev))),
        'class':must(ns(node,'tc','-s','class','show','dev',dev))}
result['kernel_log']=must(['journalctl','-k','-b','--no-pager'])
result['soft_lockup_lines']=[l for l in result['kernel_log'].splitlines() if 'soft lockup' in l.lower()]
result['workqueue_warning_count']=sum('hogged CPU' in l for l in result['kernel_log'].splitlines())
result['audit_ended']=stamp()
print(json.dumps(result,ensure_ascii=False))
