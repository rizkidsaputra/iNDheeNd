"""Read-only 1 Hz health monitoring with per-operation and wake-up timing."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
import poc11c_m1_calibration_runner_v12 as m

ROOT=m.BASE/'results/poc11c/m1_health_dryrun_v1'
RUN='health_only_v1_001'

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value)+'\n')
def timed(func):
    begin=time.monotonic_ns()
    value=func()
    return value,(time.monotonic_ns()-begin)/1e6

def main():
    if os.geteuid()!=0:raise RuntimeError('Root required for namespace health checks')
    out=ROOT/RUN;out.mkdir(parents=True,exist_ok=False)
    causes=[];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    write(out/'config.json',{'run_id':RUN,'kind':'health_only_no_experiment_traffic','duration_target_s':60,'health_gap_abort_s':5,'cpu_abort':'above 85 percent for >10 s','min_available_ram_mib':1536,'expected_nfd_nlsr':[21,21],'boot_id':boot,'script_sha256':m.sha(Path(__file__))})
    try:
        if subprocess.run(['pgrep','-x','iperf3'],capture_output=True).returncode==0:raise RuntimeError('iperf3 remains')
        if m.counts()!=(21,21):raise RuntimeError('Topology process count changed')
        nodes=[f'r{i}' for i in range(1,11)]+[f'c{i}' for i in range(1,7)]+['a1','a2','p1','p2','m']
        instances={n:(m.pid(n),m.tick(m.pid(n))) for n in nodes}
        write(out/'environment.json',{'boot_id':boot,'instances':instances,'start_utc':m.utc(),'kernel':m.command(['uname','-a'])})
        prior=m.cpu();previous=None;busy_since=None
        for index in range(60):
            begin=time.monotonic_ns();thread_cpu_begin=time.thread_time_ns();operations={}
            cp,operations['cpu_ms']=timed(m.cpu)
            mem,operations['memory_ms']=timed(m.memory)
            process_counts,operations['process_counts_ms']=timed(m.counts)
            same,operations['instance_check_ms']=timed(lambda:all(Path(f'/proc/{pid}').exists() and m.tick(pid)==ticks for pid,ticks in instances.values()))
            if index%3==0:
                log,operations['kernel_journal_ms']=timed(m.kernel)
                if 'soft lockup' in log.lower():causes.append('kernel_soft_lockup')
                _,operations['fib_routes_ms']=timed(lambda:[m.nfd(n,'fib','list') for n in ('m','a1','a2')])
            if index%2==0:
                _,operations['qdisc_ms']=timed(lambda:[m.command(m.ns(n,'tc','-j','-s','qdisc','show','dev',dev),3) for n,dev in [('r5','r5-eth4'),('r6','r6-eth0')]])
            current_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            busy=100*(1-(cp[1]-prior[1])/(cp[0]-prior[0])) if cp[0]!=prior[0] else 0
            prior=cp
            row={'index':index,'utc':m.utc(),'check_begin_ns':begin,'check_end_ns':time.monotonic_ns(),'thread_cpu_work_ns':time.thread_time_ns()-thread_cpu_begin,'intercheck_gap_s':(begin-previous)/1e9 if previous is not None else None,'cpu_busy_percent':busy,'loadavg':os.getloadavg(),'mem_available_kb':mem['MemAvailable'],'process_counts':process_counts,'nfd_instances_unchanged':same,'boot_unchanged':current_boot==boot,'operation_ms':operations,'schedstat':Path('/proc/self/schedstat').read_text().strip()}
            row['work_duration_ms']=(row['check_end_ns']-begin)/1e6
            previous=begin
            append(out/'health_detailed.jsonl',row)
            if row['intercheck_gap_s'] is not None and row['intercheck_gap_s']>5:causes.append('health_gap_gt5s')
            if process_counts!=(21,21) or not same:causes.append('nfd_nlsr_unstable')
            if current_boot!=boot:causes.append('boot_changed')
            if mem['MemAvailable']<1536*1024:causes.append('available_ram_lt1_5GiB')
            if busy>85:
                if busy_since is None:busy_since=begin
                elif begin-busy_since>10_000_000_000:causes.append('cpu_busy_gt85pct_10s')
            else:busy_since=None
            if causes:break
            wait_start=time.monotonic_ns()
            time.sleep(1)
            wake=time.monotonic_ns()
            append(out/'wake_timing.jsonl',{'index':index,'wait_start_ns':wait_start,'wake_ns':wake,'wait_overshoot_ms':(wake-wait_start-1_000_000_000)/1e6})
    except Exception as exc:
        causes.append('exception:'+repr(exc));(out/'exception.txt').write_text(traceback.format_exc())
    finally:
        write(out/'manifest.json',{'status':'aborted' if causes else 'completed','abort_causes':causes,'end_utc':m.utc(),'boot_id':boot,'no_udp_or_experiment_workload':True})
        (out/'SHA256SUMS').write_text(''.join(f'{m.sha(p)}  {p.relative_to(out).as_posix()}\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
        print(json.dumps({'out':str(out),'causes':causes,'checks':len(m.lines(out/'health_detailed.jsonl'))}),flush=True)

if __name__=='__main__':main()
