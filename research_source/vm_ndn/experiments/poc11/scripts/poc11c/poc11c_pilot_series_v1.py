"""Urutan pilot lokal terbatas; tidak meluncurkan batch utama."""
from pathlib import Path
import datetime,json,subprocess,time
BASE=Path('/home/ndn/research/experiments/poc11')
ROOT=BASE/'results/poc11c/pilots'
RUNNER=BASE/'scripts/poc11c/poc11c_pilot_runner_v3.py'
series_id=datetime.datetime.now(datetime.timezone.utc).strftime('series_%Y%m%dT%H%M%SZ')
summary=ROOT/(series_id+'.json')
results=[]
def store(status):summary.write_text(json.dumps({'series_id':series_id,'status':status,'main_batch_executed':False,'runs':results},indent=2)+'\n')
def assess(out):
    v=json.loads((out/'validation.json').read_text());c=json.loads((out/'config.json').read_text());issues=[]
    if v['status']!='collected_pending_validation' or v['abort_causes']:issues.append('safety_abort')
    if v['validation_errors'] or not v['hash_provenance_match']:issues.append('provenance_error')
    if v['generated']!=v['expected_generated']:issues.append('incomplete_generation')
    if v['received']<.98*v['expected_generated'] or v['on_time_received']<.98*v['expected_generated']:issues.append('M0_availability_below98pct')
    if not v['all_health_counts_21'] or not v['nfd_instances_unchanged']:issues.append('daemon_instability')
    if c['mode']!='telemetry':
        if v['benign_satisfaction_fraction'] is None or v['benign_satisfaction_fraction']<.98:issues.append('benign_satisfaction_below98pct')
        for name,w in v['workloads'].items():
            target=(c['attack_rate'] if 'workload_a' in name else c['benign_rate'] if name.endswith('_high') else 5)
            if abs(w['offered_over_planned_interval_per_s']-target)>.05*target:issues.append('offered_rate_deviation:'+name)
            if w['unique_names']!=w['submitted']:issues.append('duplicate_workload_names:'+name)
            if w['max_submit_lag_ms'] is None or w['max_submit_lag_ms']>250:issues.append('workload_scheduling_delay:'+name)
    if c['mode']=='A1':
        attacks=[w for n,w in v['workloads'].items() if 'workload_a' in n]
        if len(attacks)!=2:issues.append('two_attackers_missing')
        if v['attack_submitted']==0 or v['sink_unique_names']<.9*v['attack_submitted']:issues.append('insufficient_sink_arrival')
        if v['sink_data_generated']!=0:issues.append('sink_returned_data')
        if sum(w['timeouts'] for w in attacks)<.9*v['attack_submitted']:issues.append('insufficient_unfulfilled_attacker_interests')
        if sum(int(w['nack_reasons'].get('150',0)) for w in attacks)>.01*v['attack_submitted']:issues.append('NoRoute_not_valid_IFA')
        pressure=[]
        for r in ['r1','r2']:
            x=v['source_router_metrics'][r];baseline=x['pit_pre_mean'];peak=x['pit_active_max']
            pressure.append(baseline is not None and peak is not None and peak>=max(baseline+20,2*baseline))
        if not all(pressure):issues.append('insufficient_PIT_pressure_at_attack_edges')
        expired=sum(v['source_router_metrics'][r]['counter_delta']['nUnsatisfiedInterests'] for r in ['r1','r2'])
        if expired<.8*v['attack_submitted']:issues.append('unsatisfied_counter_evidence_insufficient')
    result={'run_id':out.name,'mode':c['mode'],'benign_rate':c['benign_rate'],'attack_rate':c['attack_rate'],'accepted_for_pilot':not issues,'issues':issues,'validation':v}
    # Evidence confirmation separated from primary raw files; no rehashing/overwriting raw manifests.
    confirmation=out.parent/(out.name+'_independent_review.json')
    confirmation.write_text(json.dumps({'issues':issues,'pilot_valid':not issues,'ifa_validated':c['mode']=='A1' and not issues,'criteria_version':'pilot_series_v1','source_validation_sha256':__import__('hashlib').sha256((out/'validation.json').read_bytes()).hexdigest()},indent=2)+'\n')
    return result
def main():
    initial=ROOT/'telemetry3_20261013T1312_v3'
    until=time.monotonic()+90
    while not (initial/'validation.json').exists() and time.monotonic()<until:time.sleep(2)
    if not (initial/'validation.json').exists():raise RuntimeError('Telemetry3 result not ready')
    first=assess(initial);results.append(first)
    if first['issues']:store('stopped_at_telemetry3');return
    stages=[('telemetry',0,0,30),('B0',5,0,60)]+[('B1',r,0,60) for r in [10,25,50,55]]+[('A1',5,r,60) for r in [25,50,100,150]]
    for i,(mode,br,ar,duration) in enumerate(stages,1):
        store('running');run_id=f'{series_id}_{i:02d}_{mode}_b{br}_a{ar}';out=ROOT/run_id
        args=['python3',str(RUNNER),'--mode',mode,'--run-id',run_id,'--duration',str(duration),'--passive-seconds','15','--benign-rate',str(br or 5),'--attack-rate',str(ar or 25)]
        (ROOT/(run_id+'_launch.json')).write_text(json.dumps({'command':args,'main_batch':False},indent=2)+'\n')
        with (ROOT/(run_id+'_runner.log')).open('x') as log:
            p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if p.returncode or not (out/'validation.json').exists():
            results.append({'run_id':run_id,'issues':['runner_failed'],'returncode':p.returncode});store('stopped_runner_failure');return
        result=assess(out);results.append(result);store('running')
        print(json.dumps({'run_id':run_id,'issues':result['issues'],'generated':result['validation']['generated'],'received':result['validation']['received'],'cpu_mean_percent':result['validation']['cpu_mean_percent']}),flush=True)
        if result['issues']:store('stopped_validation_gate');return
        time.sleep(10)
    store('limited_pilots_complete_main_pending_approval')
if __name__=='__main__':
    try:main()
    except Exception as e:
        results.append({'fatal_error':str(e)});store('series_error');raise

