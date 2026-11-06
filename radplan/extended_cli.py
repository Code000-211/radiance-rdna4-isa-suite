"""v0.6 offline commands. They do not load plugins or launch GPU work."""
from pathlib import Path
from .common import read,write,new_dir,digest
from .operators import op_plan
from .source_ops import extended_catalog,model_workloads
from .graph import graph_plan
from .observations import compare_graph,measurement_batch

COMMANDS={'catalog-operators','plan-operators','plan-graph','model-workloads','compare-graph','measurement-batch','attention-cpu-demo','sensitivity'}

def add_commands(sub):
    p=sub.add_parser('sensitivity',help='Stress candidate ranking across an explicit finite set of profile assumptions')
    for x in ('catalog','workloads','systems','out'):p.add_argument('--'+x,type=Path,required=True)
    p=sub.add_parser('attention-cpu-demo',help='Execute genuine Radiance libref attention via host replay; not GPU calibration')
    for x in ('worker','libref','out'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--execute-trusted-code',action='store_true',required=True)
    p=sub.add_parser('catalog-operators',help='Enumerate reviewed source-family cost candidates, NOT native kernels')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('plan-operators',help='Attention/QSA/MoE/TP2 analytical scenarios')
    for x in ('catalog','workloads','system','out'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--require-calibrated',action='store_true')
    p=sub.add_parser('plan-graph',help='Search an explicit multi-device dependency/resource graph')
    for x in ('catalog','workloads','system','graph','out'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--require-calibrated',action='store_true')
    p=sub.add_parser('model-workloads',help='Expand supplied qwen4_exp geometry + explicit routing/selections; partial model')
    for x in ('source','config','run','out'):p.add_argument('--'+x,type=Path,required=True)
    p=sub.add_parser('compare-graph',help='Compare a graph scenario with identity-bound external GPU observations')
    for x in ('plan','observations','system','out'):p.add_argument('--'+x,type=Path,required=True)
    p=sub.add_parser('measurement-batch',help='Export NOT_EXECUTED workload/candidate requests')
    for x in ('plan','out'):p.add_argument('--'+x,type=Path,required=True)


def operator_md(r):
    lines=['# v0.6 operator scenarios','','No GPU was executed. Every new operator ranking needs target measurement.','',
           '| Workload | Candidate | Scenario µs | Decision |','|---|---|---:|---|']
    for w in r['workloads']:
        for x in w['ranked']:lines.append(f"| {w['workload_id']} | {x['candidate_id']} | {x['scenario_us']:.6g} | {x['status']} |")
        for x in w['rejected']:lines.append(f"\nRejected `{w['workload_id']}` / `{x['candidate_id']}`: {x['reason']}")
    return '\n'.join(lines)+'\n'


def graph_md(r):
    s=r['selected']
    lines=['# v0.6 graph scenario','','**Not measured; not a production throughput forecast.**','',
           f"Selected scenario: {s['makespan_us']:.6g} µs; serialized model: {r['serialized_scenario_us']:.6g} µs.",
           f"Search: {r['search']}. Evaluations: {r['evaluations']}.",'',
           '| Node | Candidate | Start µs | Finish µs |','|---|---|---:|---:|']
    for n,c in s['chosen'].items():
        start,end=s['node_times'][n];lines.append(f'| {n} | {c} | {start:.6g} | {end:.6g} |')
    lines+=['','## Limits','']+['- '+x for x in r['limitations']]
    return '\n'.join(lines)+'\n'


def execute(args):
    if args.command=='sensitivity':
        from .sensitivity import analyze
        write(args.out,analyze(read(args.catalog),read(args.workloads),read(args.systems)));print('ASSUMPTION_STRESS_TEST_ONLY; not confidence or measurements')
    elif args.command=='attention-cpu-demo':
        from .attention_demo import run
        run(args.out,args.worker,args.libref);print('HOST_ATTENTION_CHECKS_PASS; GPU execution: false')
    elif args.command=='catalog-operators':write(args.out,extended_catalog(args.source));print('SOURCE_FAMILY_COST_CATALOG_CREATED; no code generated')
    elif args.command=='plan-operators':
        r=op_plan(read(args.catalog),read(args.workloads),read(args.system));out=new_dir(args.out)
        write(out/'operator-plan.json',r);(out/'operator-plan.md').write_text(operator_md(r))
        write(out/'measurement-requests.json',measurement_batch(r));print('OPERATOR_SCENARIOS_CREATED; GPU executed: false')
        if args.require_calibrated:return 2
    elif args.command=='plan-graph':
        r=graph_plan(read(args.catalog),read(args.workloads),read(args.system),read(args.graph));out=new_dir(args.out)
        write(out/'graph-plan.json',r);(out/'graph-plan.md').write_text(graph_md(r));print('GRAPH_SCENARIO_CREATED; TP2 runtime executed: false')
        if args.require_calibrated:return 2
    elif args.command=='model-workloads':write(args.out,model_workloads(read(args.config),read(args.run),args.source));print('PARTIAL_MODEL_WORKLOADS_CREATED; no model TPS prediction')
    elif args.command=='compare-graph':write(args.out,compare_graph(read(args.plan),read(args.observations),read(args.system)));print('COMPARISON_ONLY; no automatic performance approval')
    elif args.command=='measurement-batch':write(args.out,measurement_batch(read(args.plan)));print('REQUESTS_CREATED; NOT_EXECUTED')
    return 0
