"""Offline CLI. Only explicitly authorized CPU demo commands launch native host code."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .common import PlanError,read,write,new_dir,need,digest
from .model import catalog_check,workloads_check,profile
from .planner import plan,markdown,serial_path
from .calibrate import fit,dataset_check
from .radiance import extract,bind_resources
from .importer import import_replay,profile_from_replay
from .extended_cli import add_commands, execute as execute_extended, COMMANDS as EXTENDED_COMMANDS


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(prog='radplan',description='Offline candidate ranking; not GPU emulation or kernel certification.')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('catalog-radiance',help='Extract reviewed W4A8 arms from hash-pinned Radiance source')
    p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p=sub.add_parser('plan',help='Compute costs/ranking without loading executable code')
    p.add_argument('--catalog',required=True,type=Path);p.add_argument('--workloads',required=True,type=Path)
    p.add_argument('--profile',required=True,type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--require-calibrated',action='store_true')
    p=sub.add_parser('fit',help='Fit CPU OR GPU records; split whole shapes, not repetitions')
    p.add_argument('--measurements',required=True,type=Path);p.add_argument('--catalog',required=True,type=Path)
    p.add_argument('--profile',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--calibration-groups',type=int,default=10)
    p.add_argument('--test-groups',type=int,default=10);p.add_argument('--alpha',type=float,default=.1)
    p=sub.add_parser('import-replay',help='Validate checked single-kernel timing evidence and build records')
    p.add_argument('--catalog',required=True,type=Path);p.add_argument('--profile',required=True,type=Path)
    p.add_argument('--workloads',required=True,type=Path);p.add_argument('--index',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p=sub.add_parser('profile-from-replay',help='Pin observed device/worker/timing mode; not throughput calibration')
    p.add_argument('--base',required=True,type=Path);p.add_argument('--report',required=True,type=Path)
    p.add_argument('--clock-power',required=True);p.add_argument('--compiler-id',required=True);p.add_argument('--out',required=True,type=Path)
    p=sub.add_parser('bind-resources',help='Join actual compiled registers/LDS by explicit object+descriptor mapping')
    p.add_argument('--catalog',required=True,type=Path);p.add_argument('--isa-manifest',required=True,type=Path)
    p.add_argument('--isa-report',required=True,type=Path);p.add_argument('--mapping',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p=sub.add_parser('path',help='Choose a declared serial path with explicit layout-transition costs')
    p.add_argument('--plan',required=True,type=Path);p.add_argument('--spec',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p=sub.add_parser('cpu-demo',help='Run real Radiance HOST replay; no GPU timing or emulation')
    p.add_argument('--worker',required=True,type=Path);p.add_argument('--libref',required=True,type=Path)
    p.add_argument('--plugin',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--reps',type=int,default=15);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--execute-trusted-code',action='store_true')
    add_commands(sub)
    args=parser.parse_args(argv)
    try:
        if args.command in EXTENDED_COMMANDS:
            return execute_extended(args)
        if args.command=='catalog-radiance':
            r=extract(args.source);write(args.out,r);print(f"Extracted {len(r['candidates'])} source-bound arms (not all Radiance kernels)")
        elif args.command=='plan':
            r=plan(read(args.catalog),read(args.workloads),read(args.profile),read(args.calibration) if args.calibration else None)
            out=new_dir(args.out);write(out/'plan.json',r);(out/'plan.md').write_text(markdown(r))
            print('OFFLINE_PLAN_CREATED; GPU executed: false; deployment approved: false')
            if args.require_calibrated and any(x['status']!='CALIBRATED_SEPARATION' for x in r['workloads']):return 2
        elif args.command=='fit':
            r=fit(read(args.measurements),read(args.catalog),read(args.profile),seed=args.seed,ncal=args.calibration_groups,ntest=args.test_groups,alpha=args.alpha)
            write(args.out,r);print('CALIBRATION_CREATED; held-out validation:',r['validation'])
        elif args.command=='import-replay':
            cat=catalog_check(read(args.catalog));p=profile(read(args.profile));ws=workloads_check(read(args.workloads))
            table={c['id']:c for c in cat['candidates']};wt={w['id']:w for w in ws['workloads']}
            ix=read(args.index);need(isinstance(ix,list) and ix,'nonempty measurement index list required')
            records=[]
            for entry in ix:
                need(set(entry)=={'workload_id','candidate_id','report','fixture'},'index fields differ')
                need(entry['candidate_id'] in table and entry['workload_id'] in wt,'unknown workload/candidate')
                report_path=(args.index.parent/entry['report']).resolve();fixture=(args.index.parent/entry['fixture']).resolve()
                records.append(import_replay(report_path,fixture,wt[entry['workload_id']],table[entry['candidate_id']],p))
            r={'schema':'radiance-planner-measurements/1','profile_sha256':digest(p),'records':records}
            dataset_check(r,cat,p);write(args.out,r);print('Imported',len(records),'checked measurements')
        elif args.command=='profile-from-replay':
            write(args.out,profile_from_replay(read(args.base),args.report,args.clock_power,args.compiler_id))
            print('Observed environment pinned; throughput still uncalibrated')
        elif args.command=='bind-resources':
            r=bind_resources(read(args.catalog),args.isa_manifest.resolve(),args.isa_report.resolve(),read(args.mapping));write(args.out,r)
            print('Compiled resources bound; numerical correctness remains unproved')
        elif args.command=='path':
            r=serial_path(read(args.plan),read(args.spec));write(args.out,r);print('SERIAL_PATH_SCENARIO (not production latency)')
        elif args.command=='cpu-demo':
            need(args.execute_trusted_code,'cpu-demo loads native plugins; requires --execute-trusted-code')
            from .cpu_demo import run
            r=run(args.out,args.worker.resolve(),args.libref.resolve(),args.plugin.resolve(),seed=args.seed,reps=args.reps)
            print('CPU comparison:',r)
        return 0
    except (PlanError,ValueError,OSError,KeyError,TypeError,OverflowError,RuntimeError) as e:
        print('INCOMPLETE:',e,file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
