#!/usr/bin/env python3
"""Retain actual host counterexample reports, including ASan's first-error stderr.
No GPU plugin is used. A PASS of the uninstrumented overread is deliberately shown
as a known observation boundary, not as memory safety.
"""
import argparse,json
from pathlib import Path
from radreplay.builder import Package
from radreplay.cli import main
from radreplay.format import require


def run(normal:Path,asan:Path,out:Path):
    require(not out.exists(),'fresh output required');out.mkdir(parents=True,mode=0o700)
    specs=[('overread-asan','audit_oob_read',True,0,None,False,1),
           ('missing-store','audit_no_store',False,0,None,False,1),
           ('scratch-before-init','audit_scratch_read',False,16,'kernel-initialized',False,1),
           ('masked-intermediate-write','mutate',False,0,None,True,1),
           ('correct-copy','copy',False,0,None,False,0),
           ('overread-without-instrumentation','audit_oob_read',False,0,None,False,0),
           ('explicit-zeroed-scratch','audit_scratch_read',False,16,'zeroed',False,0),
           ('correct-asan-copy','copy',True,0,None,False,0)]
    result={'schema':'radiance-host-blindspot-regressions/1','status':'PASS','gpu_execution':False,'cases':[]}
    for name,row,instrumented,scratch,policy,masked,expected_rc in specs:
        build=asan if instrumented else normal;plugin=build/'replay_test.so'
        p=Package(out/name,name,note='Deliberately faulty HOST mutation/control; never GPU code.')
        p.plugin('replay_test',plugin);p.stream();p.allocation('x',b'abcd',mutable=masked);p.allocation('y',b'abcd',mutable=True)
        for a in ('x','y'):p.view(a,a,nbytes=4,dtype=9,shape=[4])
        p.kernel('tested','p0',row,'replay_probe',['x','y'],[('x','r'),('y','w')],scratch_bytes=scratch,scratch_policy=policy)
        if masked:
            p.allocation('z',b'abcd');p.view('z','z',nbytes=4,dtype=9,shape=[4])
            p.kernel('later','p0','copy','replay_probe',['z','x'],[('z','r'),('x','w')]);p.check('x',b'abcd')
        p.check('y',b'abcd');path=p.finish()
        cmd=['run',str(path),'--out',str(out/(name+'-result')),'--worker',str(build/'rad-kbench-replay'),
             '--plugin','p0='+str(plugin),'--execute-trusted-code','--warmups','1','--reps','3']
        if instrumented:cmd+=['--memory-check','host-asan','--diagnostic-only']
        rc=main(cmd);r=json.loads((out/(name+'-result')/'report.json').read_text())
        result['cases'].append({'name':name,'expected_exit_code':expected_rc,'actual_exit_code':rc,'status':r['status'],
                                'report':name+'-result/report.json','error':r.get('error',r.get('runtime',{}).get('error'))})
        if rc!=expected_rc:result['status']='FAIL'
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    require(result['status']=='PASS','counterexample detector/control regression')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--normal-build',type=Path,required=True);p.add_argument('--asan-build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.normal_build.resolve(),a.asan_build.resolve(),a.out.resolve())
