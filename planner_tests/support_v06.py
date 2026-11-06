import copy,json
from pathlib import Path
from radplan.source_ops import make_candidate
ROOT=Path(__file__).resolve().parents[1]
EX=ROOT/'planner/examples/v06'
def load(name):return json.loads((EX/name).read_text())
def system():return load('dual-r9700-unmeasured.json')
def catalog():return load('operator-catalog.json')
def workloads():return load('operator-workloads.json')
def wop(op,rank=0):return copy.deepcopy([w for w in workloads()['workloads'] if w['op']==op][rank])
def cop(op,**changes):
    c=copy.deepcopy(next(c for c in catalog()['candidates'] if c['op']==op))
    c['params'].update(changes);return c
def simple():
    s=system()
    for d in s['devices']:d['kernel_slots']=2;d['dram_bytes_per_us']=10;d['launch_us']=0
    cs=[make_candidate('a','explicit','x',{'service_scale':1},input_layout='x',output_layout='x')]
    ws=[{'id':n,'op':'explicit','contract':'x','device':d,'service':{'dram':100},'fixed_us':0,'workspace_bytes':0} for n,d in [('w0','gpu0'),('w1','gpu1')]]
    cat={'schema':'radiance-operator-catalog/1','candidates':cs};ww={'schema':'radiance-operator-workloads/1','workloads':ws}
    g={'schema':'radiance-planner-graph/1','nodes':[{'id':n,'workload_id':w,'allowed_candidates':['a'],'deps':[],'queue':q} for n,w,q in [('n0','w0','q0'),('n1','w1','q1')]],'transitions':[],'buffers':[],'resident_bytes':{},'search':{'max_combinations':16,'beam_width':2}}
    return cat,ww,s,g
