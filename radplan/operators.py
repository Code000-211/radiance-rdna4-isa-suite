"""Analytical attention, QSA indexer, routed MoE, dense GEMM and TP2 operators.

Counts follow DECLARED algorithms and data layouts. source bindings establish a
reviewed relationship to a template family, not instruction-level equivalence.
Rates, cache residency, pipeline overlap and contention still require GPU evidence.
"""
from __future__ import annotations
import math
from collections import defaultdict
from .common import fields, need, integer, number, digest, ceildiv, hash_string
from .flow import phase, resources, simulate, system_check, text
from .model import features as gemm_features, candidate as gemm_candidate, workload as gemm_workload

OPS={'attention','qsa_indexer','moe','collective','gemm','explicit'}


def base(w,c,s):
    system_check(s)
    if 'family' in w: text(w['family'],'operator family')
    need(isinstance(w,dict) and isinstance(c,dict),'operator workload/candidate must be objects')
    need(w.get('op') in OPS and c.get('op')==w['op'],'different/unsupported operation')
    for d in (w,c): text(d.get('id'),'id');text(d.get('contract'),'numerical/layout contract')
    need(w['contract']==c['contract'],'different numerical/layout contract')
    fields(c,{'id','op','contract','params','identity'}, {'input_layout','output_layout','source_binding','notes'},'operator candidate')
    fields(c['identity'],{'kind','sha256'}, {'symbol','plugin_sha256','object_sha256'},'operator identity')
    need(c['identity']['kind'] in ('hypothesis','source','code_object','plugin'),'identity kind')
    hash_string(c['identity']['sha256'],'identity')
    for k in ('plugin_sha256','object_sha256'):
        if k in c['identity']:hash_string(c['identity'][k],k)
    need(isinstance(c['params'],dict),'candidate params must be object')
    devs={d['id']:d for d in s['devices']}
    if w['op']!='collective':need(w.get('device') in devs,'unknown device')
    return devs


def id_check(items):
    need(isinstance(items,list) and 0<len(items)<=10000,'1..10000 items required')
    need(all(isinstance(x,dict) and isinstance(x.get('id'),str) for x in items),'object IDs required')
    need(len({x['id'] for x in items})==len(items),'duplicate IDs')

class Builder:
    def __init__(self,w,c,s):
        self.w,self.c,self.s=w,c,s;self.ps=[];self.devs={d['id']:d for d in s['devices']}
        self.counts={};self.warnings=[];self.workspace_bytes=0
    def kernel(self,label,demands,device=None,deps=None,launch=True):
        device=device or self.w['device'];d=self.devs[device]
        p=f'{len(self.ps)}:{label}';deps=list(deps if deps is not None else ([self.ps[-1]['id']] if self.ps else []))
        if launch:
            a=p+':start';self.ps.append(phase(a,deps,{},fixed_us=d['launch_us'],reservations={device+':kernels':1},label=label+' start-cost scenario'));deps=[a]
        demands=dict(demands)
        matrix_time=sum(v/d['matrix_ops_per_us'][k.split(':matrix:',1)[1]] for k,v in demands.items() if k.startswith(device+':matrix:'))
        if matrix_time:demands[device+':matrix_issue']=matrix_time
        self.ps.append(phase(p,deps,demands,reservations={device+':kernels':1},label=label));return p
    def transfer(self,label,src,dst,amount,access,deps):
        links=[l for l in self.s['links'] if (l['source'],l['target'],l['access'])==(src,dst,access)]
        need(len(links)==1,'exactly one declared directional/access link required')
        l=links[0];p=f'{len(self.ps)}:{label}'
        a=p+':latency';self.ps.append(phase(a,list(deps),{},fixed_us=l['latency_us'],label=label+' startup'))
        demand={'link:'+l['id']:amount,src+':dram':amount,dst+':dram':amount}
        for r in l['shared_resources']:demand['shared:'+r]=amount
        self.ps.append(phase(p,[a],demand,label=label));return p
    def finish(self):
        rates,slots=resources(self.s)
        result=simulate(self.ps,rates,slots)
        return {'candidate_id':self.c['id'],'workload_id':self.w['id'],'candidate_sha256':digest(self.c),
                'workload_sha256':digest(self.w),'system_sha256':digest(self.s),'op':self.w['op'],
                'counts':self.counts,'phases':self.ps,'scenario_us':result['makespan_us'],
                'workspace_bytes':self.workspace_bytes,'warnings':self.warnings,
                'status':'NEEDS_MEASUREMENT','interval_us':None,'gpu_executed':False,
                'numerical_equivalence_proved':False,'dispatch_mapping_proved':False,
                'input_layout':self.c.get('input_layout',self.c['contract']),
                'output_layout':self.c.get('output_layout',self.c['contract'])}

def dims(w,keys):
    for k in keys:integer(w[k],k,1,2**24)

def queries_check(w,selection=False):
    need(isinstance(w['query_lengths'],list) and 0<len(w['query_lengths'])<=65536,'query_lengths required, max 65536')
    for l in w['query_lengths']:integer(l,'visible causal length',1,2**24)
    need(isinstance(w['sequence_ids'],list) and len(w['sequence_ids'])==len(w['query_lengths']),'one sequence id per query')
    for sid in w['sequence_ids']:integer(sid,'sequence id',0,65535)
    positions=defaultdict(list)
    for sid,l in zip(w['sequence_ids'],w['query_lengths']):positions[sid].append(l)
    for ls in positions.values():need(all(a<b for a,b in zip(ls,ls[1:])),'queries per sequence must have increasing visible lengths')
    if selection:
        bs=w['block_size'];sel=w['selected_blocks']
        need(sel is None or (isinstance(sel,list) and len(sel)==len(w['query_lengths'])),'one block selection per query or null')
        if sel is not None:
            need(all(isinstance(x,list) for x in sel),'block selections must be lists')
            need(sum(len(x) for x in sel)<=1000000,'too many selected blocks')
            for l,ids in zip(w['query_lengths'],sel):
                need(isinstance(ids,list) and len(ids)>0 and len(ids)==len(set(ids)),'nonempty unique selected blocks required')
                for x in ids:integer(x,'block index',0,ceildiv(l,bs)-1)
    return positions

def attention(b:Builder):
    w,c=b.w,b.c;p=c['params'];d=b.devs[w['device']];dev=w['device']
    fields(w,{'id','op','contract','device','query_lengths','sequence_ids','selected_blocks','q_heads','kv_heads',
              'head_dim','block_size','q_bits','kv_bits','out_bits','kv_scale_bytes_per_block'}, {'note','family'}, 'attention workload')
    fields(p,{'query_tile','kv_tile','split_kv','gqa_reuse','compute_path','head_dim','gqa','block_size',
              'max_queries_per_sequence','force_merge','kv_bits','partial_bits'}, {'kv_fetch'},'attention candidate params')
    dims(w,['q_heads','kv_heads','head_dim','block_size']);dims(p,['query_tile','kv_tile','split_kv','head_dim','gqa','block_size'])
    integer(p['max_queries_per_sequence'],'max queries',1,65536)
    need(w['q_heads']%w['kv_heads']==0,'q_heads must divide into KV groups')
    gqa=w['q_heads']//w['kv_heads']
    need((p['head_dim'],p['gqa'],p['block_size'])==(w['head_dim'],gqa,w['block_size']),'candidate attention geometry differs')
    for k in ('q_bits','kv_bits','out_bits'):need(type(w[k]) is int and w[k] in (8,16,32),'unsupported precision')
    need(w['kv_bits']==p['kv_bits'],'KV format differs')
    need(type(p['partial_bits']) is int and p['partial_bits'] in (16,32),'partial precision required')
    need(w['q_bits']==16 and w['out_bits']==16,'current attention candidates require BF16 query/output')
    number(w['kv_scale_bytes_per_block'],'KV scale bytes')
    need(type(p['gqa_reuse']) is bool and type(p['force_merge']) is bool,'boolean attention flags required')
    need(p.get('kv_fetch','query') in ('query','sequence'),'kv_fetch must be query or sequence')
    need(p['compute_path'] in d['matrix_ops_per_us'],'missing actual arithmetic rate')
    positions=queries_check(w,True)
    need(max(map(len,positions.values()))<=p['max_queries_per_sequence'],'query width exceeds candidate contract')
    # Pairs are exact for the explicitly supplied causal prefix/selection; the
    # padded work model treats each query independently when masks differ.
    need(sum(ceildiv(l,w['block_size']) for l in w['query_lengths'])<=4000000,'attention enumeration resource cap')
    lengths=[];blocks=[];signatures=[]
    for i,l in enumerate(w['query_lengths']):
        ids=list(range(ceildiv(l,w['block_size']))) if w['selected_blocks'] is None else w['selected_blocks'][i]
        n=sum(min(w['block_size'],l-j*w['block_size']) for j in ids)
        lengths.append(n);blocks.append(len(ids));signatures.append((tuple(ids),l))
    need(sum(blocks)<=2**32,'attention block work too large')
    useful=4*w['head_dim']*w['q_heads']*sum(lengths)
    # KV fetch policy: a query tile reads each visible KV byte once in hardware, and
    # the R9700's Infinity Cache makes later queries of the same sequence reuse it.
    # 'sequence' charges the union once per sequence; 'query' charges every query.
    fetch_once=p.get('kv_fetch','query')=='sequence'
    executed=0;fetched=0;table_bytes=0;groups=0;live_splits=[0]*len(lengths);seq_fetch={};seq_table={}
    for sid in positions:
        ix=[i for i,s in enumerate(w['sequence_ids']) if s==sid]
        for z in range(0,len(ix),p['query_tile']):
            chunk=ix[z:z+p['query_tile']]
            # Each query may select a DIFFERENT block list. No unjustified reuse.
            bundles=[chunk] if w['selected_blocks'] is None or len({signatures[i] for i in chunk})==1 else [[i] for i in chunk]
            for bundle in bundles:
                i=max(bundle,key=lambda j:lengths[j]);n=lengths[i];nt=ceildiv(n,p['kv_tile']);parts=min(p['split_kv'],nt)
                
                for qi in bundle:live_splits[qi]=parts
                qt=len(bundle)*gqa if p['gqa_reuse'] else len(bundle)
                heads=w['kv_heads'] if p['gqa_reuse'] else w['q_heads']
                qpad=ceildiv(qt,16)*16; kpad=ceildiv(nt,parts)*parts*p['kv_tile']
                executed+=4*w['head_dim']*heads*qpad*kpad
                f=2*w['head_dim']*heads*n*w['kv_bits']/8+heads*blocks[i]*w['kv_scale_bytes_per_block']
                t=4*heads*blocks[i]
                if fetch_once:
                    seq_fetch[sid]=max(seq_fetch.get(sid,0),f);seq_table[sid]=max(seq_table.get(sid,0),t)
                else:
                    fetched+=f;table_bytes+=t
                groups+=parts*heads
    fetched+=sum(seq_fetch.values());table_bytes+=sum(seq_table.values())
    qn=len(lengths);out=qn*w['q_heads']*w['head_dim']*w['out_bits']/8
    qbytes=qn*w['q_heads']*w['head_dim']*w['q_bits']/8
    need(executed<=2**65,'attention work too large')
    merge=p['force_merge'] or max(live_splits)>1
    partial=sum(live_splits)*w['q_heads']*(w['head_dim']*p['partial_bits']/8+8) if merge else 0
    pair=w['q_heads']*sum(lengths)
    # A profile may carry a measured analytic model for this family: measured constants,
    # evaluated as a formula (no table). It replaces the generic resource simulation.
    am=d.get('attention_model')
    if am:
        seqs=len(positions)
        widest=max((max(ix) for ix in positions.values()), default=0)  # positions hold visible lengths
        rows_q=len(lengths)//max(1,seqs)   # queries per sequence; len(lengths) counts all sequences
        scan=am['us_per_token']*widest*seqs**am['scan_batch_exponent']      # KV scan, batch saturation
        work=am['us_per_query_ctx']*rows_q*seqs*widest                       # per-query work over the context
        t=am['a_us']+max(am['scan_floor_us'],scan,work)
        b.ps.append(phase('0:attention_measured_model',[],{},fixed_us=t,label='measured analytic attention model'))
        b.counts['model']='measured-analytic'
        b.counts['sequence_count']=seqs
        b.warnings.append('Scenario time comes from measured analytic constants for this family; it is a formula, not a compiled-kernel measurement.')
        return b.finish()
    b.kernel('attention', {dev+':matrix:'+p['compute_path']:executed,
            dev+':dram':fetched+table_bytes+qbytes+(partial if merge else out),
            dev+':vector':5*pair,dev+':exp':pair})
    if merge:
        b.kernel('splitkv_merge',{dev+':dram':partial+out,dev+':vector':sum(live_splits)*w['q_heads']*(2*w['head_dim']+5),
                                 dev+':exp':sum(live_splits)*w['q_heads']})
    b.workspace_bytes=int(partial)
    b.counts={'useful_matrix_ops':useful,'executed_matrix_ops_scenario':executed,'attention_pairs':pair,
              'selected_tokens_per_query':lengths,'selected_blocks_per_query':blocks,'kv_requested_bytes':fetched,
              'block_table_bytes':table_bytes,'split_partial_bytes':partial,'groups':groups,
              'effective_splits_per_query':live_splits,'quadratic_score_matrix_materialized':False}
    b.warnings=['BF16/FP8 storage does not determine compute throughput; compute_path is explicit.',
                'Dense query tiles use their causal union; differing sparse selections use separate queries (conservative scenario).',
                'WMMA tile work, pipeline overlap, dequantization and cache residency are not measured.',
                'This operator excludes QSA indexer/projection/KV-write unless separately present in graph.']

def qsa_indexer(b:Builder):
    w=b.w;p=b.c['params'];dev=w['device'];d=b.devs[dev]
    fields(w,{'id','op','contract','device','query_lengths','sequence_ids','block_size','heads','head_dim',
              'keep_blocks','updated_blocks','skip_score_when_dense'}, {'note'},'QSA workload')
    fields(p,{'score_tile','selection','passes','compute_path'},set(),'QSA candidate')
    dims(w,['block_size','heads','head_dim','keep_blocks']);integer(w['updated_blocks'],'updated blocks',0,2**24)
    dims(p,['score_tile','passes']);need(type(w['skip_score_when_dense']) is bool,'skip flag required')
    need(p['selection'] in ('scan_topk','radix_passes'),'explicit selection algorithm required')
    need(p['compute_path'] in d['matrix_ops_per_us'],'missing QSA compute path')
    queries_check(w)
    counts=[l//w['block_size'] for l in w['query_lengths']]
    scored=[n for n in counts if not(w['skip_score_when_dense'] and n<=w['keep_blocks'])]
    n=sum(scored);padded=sum(ceildiv(n0,p['score_tile'])*p['score_tile'] for n0 in scored)
    # Block-key update is separate and still exists while score/select are skipped.
    if w['updated_blocks']:
        u=w['updated_blocks']
        b.kernel('qsa_block_key',{dev+':vector':u*w['head_dim']*(w['block_size']+8),
                                 dev+':dram':2*u*w['head_dim']*(w['block_size']+1)})
    if scored:
        b.kernel('qsa_score',{dev+':matrix:'+p['compute_path']:2*padded*w['heads']*w['head_dim'],
                             dev+':vector':n*w['heads']*2,dev+':dram':n*w['head_dim']*2+4*n+len(scored)*w['heads']*w['head_dim']*2})
        comparison=n*w['keep_blocks'] if p['selection']=='scan_topk' else n*p['passes']
        b.kernel('qsa_select',{dev+':vector':comparison,dev+':dram':4*n*p['passes']+4*sum(min(x,w['keep_blocks'])+1 for x in scored)})
    if not b.ps:b.ps=[phase('0:qsa_noop',[],{},label='score/select skipped; no block update')]
    b.workspace_bytes=4*max(scored,default=0)*len(scored)
    b.counts={'history_blocks_per_query':counts,'scored_queries':len(scored),'score_matrix_elements':n,
              'score_matrix_bytes':4*n,'score_plane_capacity_bytes_scenario':b.workspace_bytes,
              'score_matrix_ops':2*n*w['heads']*w['head_dim'],
              'selected_complete_blocks_per_query':[min(v,w['keep_blocks']) for v in counts],
              'partial_block_present':[bool(l%w['block_size']) for l in w['query_lengths']],
              'updated_blocks':w['updated_blocks']}
    b.warnings=['QSA selects per query, not per batch; historical scoring is NOT limited to selected KV.',
                'Indexer projection/norm/RoPE and persistent tail state must be separate nodes.',
                'Selection-pass costs and pooled-key update operations are analytical hypotheses, not compiled instruction counts.']

def moe(b:Builder):
    w=b.w;p=b.c['params'];dev=w['device'];d=b.devs[dev]
    fields(w,{'id','op','contract','device','tokens','hidden','top_k','expert_tokens','expert_widths','shared_width',
              'weight_bits','activation_bits','scale_group','scale_bytes','cold_weight_fraction','router'}, {'note'},'MoE workload')
    fields(p,{'bm','bn','bk','compute_path','include_empty_experts','router_selection_passes'},set(),'MoE candidate')
    dims(w,['tokens','hidden','top_k','scale_group']);dims(p,['bm','bn','bk','router_selection_passes'])
    integer(w['shared_width'],'shared width',0,2**24)
    need(w['weight_bits'] in (4,8,16) and type(w['weight_bits']) is int,'weight format')
    need(w['activation_bits'] in (8,16) and type(w['activation_bits']) is int,'activation format')
    number(w['scale_bytes'],'scale bytes');number(w['cold_weight_fraction'],'cold fraction');need(w['cold_weight_fraction']<=1,'cold fraction >1')
    need(type(p['include_empty_experts']) is bool,'empty-expert policy required')
    need(p['compute_path'] in d['matrix_ops_per_us'],'MoE compute path missing')
    need(isinstance(w['expert_tokens'],list) and 0<len(w['expert_tokens'])<=65536,'expert histogram required')
    e=len(w['expert_tokens']);need(w['top_k']<=e,'top_k > experts')
    for n in w['expert_tokens']:integer(n,'tokens per expert',0,w['tokens'])
    need(sum(w['expert_tokens'])==w['tokens']*w['top_k'],'routing counts do not sum to tokens*top_k')
    need(isinstance(w['expert_widths'],list) and len(w['expert_widths'])==e,'one width per expert required')
    for k in w['expert_widths']:integer(k,'expert local width',1,2**24);need(k%w['scale_group']==0,'expert split crosses scale group')
    need(w['hidden']%w['scale_group']==0,'hidden crosses scale group')
    need(w['shared_width']%w['scale_group']==0,'shared split crosses scale group')
    need(w['router'] in ('provided','dense_topk'),'router contract required')
    dims_work=sum(n*k for n,k in zip(w['expert_tokens'],w['expert_widths']))
    useful=6*w['hidden']*dims_work;executed1=executed2=0;weights=tile_weights=0;groups=0;active=0
    for n,k in zip(w['expert_tokens'],w['expert_widths']):
        if not n and not p['include_empty_experts']:continue
        active+=int(n>0);gm=max(1,ceildiv(n,p['bm']))
        n1=ceildiv(2*k,p['bn'])*p['bn'];k1=ceildiv(w['hidden'],p['bk'])*p['bk']
        n2=ceildiv(w['hidden'],p['bn'])*p['bn'];k2=ceildiv(k,p['bk'])*p['bk']
        executed1+=2*gm*p['bm']*n1*k1;executed2+=2*gm*p['bm']*n2*k2
        payload=3*w['hidden']*k*w['weight_bits']/8
        scales=(2*k*ceildiv(w['hidden'],w['scale_group'])+w['hidden']*ceildiv(k,w['scale_group']))*w['scale_bytes']
        weights+=payload+scales;tile_weights+=gm*(payload+scales)
        groups+=gm*(ceildiv(2*k,p['bn'])+ceildiv(w['hidden'],p['bn']))
    slots=w['tokens']*w['top_k'];act=w['activation_bits']/8
    dispatch_bytes=slots*w['hidden']*act+slots*8
    inter_bytes=3*dims_work*2
    output_bytes=slots*w['hidden']*2
    if w['router']=='dense_topk':
        b.kernel('router_gemm',{dev+':matrix:wmma_f16':2*w['tokens']*w['hidden']*e,
                               dev+':dram':2*(w['tokens']*w['hidden']+e*w['hidden'])+4*w['tokens']*e})
        b.kernel('router_topk',{dev+':vector':w['tokens']*e*p['router_selection_passes'],dev+':dram':4*w['tokens']*e+8*slots})
    b.kernel('dispatch_scatter',{dev+':dram':2*dispatch_bytes,dev+':vector':slots*4})
    charged_weights=tile_weights*w['cold_weight_fraction']
    b.kernel('expert_gate_up',{dev+':matrix:'+p['compute_path']:executed1,
                              dev+':dram':charged_weights*2/3+dispatch_bytes+inter_bytes*2/3})
    b.kernel('activation_quant',{dev+':vector':dims_work*8,dev+':exp':dims_work,dev+':dram':inter_bytes})
    b.kernel('expert_down',{dev+':matrix:'+p['compute_path']:executed2,
                           dev+':dram':charged_weights/3+dims_work*act+output_bytes})
    b.kernel('gather_reduce',{dev+':vector':2*slots*w['hidden'],dev+':dram':output_bytes+2*w['tokens']*w['hidden']})
    shared_ops=0
    if w['shared_width']:
        shared_ops=6*w['tokens']*w['hidden']*w['shared_width']
        b.kernel('shared_expert_serial',{dev+':matrix:'+p['compute_path']:shared_ops,dev+':exp':w['tokens']*w['shared_width'],
                 dev+':vector':8*w['tokens']*w['shared_width'],dev+':dram':3*w['hidden']*w['shared_width']*w['weight_bits']/8+8*w['tokens']*w['shared_width']})
    b.workspace_bytes=int(dispatch_bytes+inter_bytes+output_bytes)
    b.counts={'active_experts':active,'routing_slots':slots,'weighted_expert_width':dims_work,
              'useful_matrix_ops':useful,'executed_matrix_ops_scenario':executed1+executed2,
              'expert_weight_unique_bytes':weights,'expert_weight_tile_bytes':tile_weights,
              'cold_weight_traffic_scenario_bytes':charged_weights,'groups':groups,'shared_expert_ops':shared_ops,
              'max_tokens_per_expert':max(w['expert_tokens']),'empty_experts':sum(n==0 for n in w['expert_tokens'])}
    b.warnings=['Histogram is explicit; uniform routing is never inferred from batch size.',
                'Each rank must provide its real per-expert widths; router and shared expert are separate costs.',
                'Packed-weight padding/relayout, cache hits, scatter and top-k algorithm are scenarios.',
                'Grouped kernel work is aggregated, not one launch per expert; histogram does not prove numerical correctness.',
                'Shared expert is serial here; explicit graph branches are required to explore overlap.']

def collective(b:Builder):
    w=b.w;p=b.c['params']
    fields(w,{'id','op','contract','ranks','bytes','element_bytes'}, {'note'},'collective workload')
    fields(p,{'algorithm','access','chunk_bytes','handshake_us'},set(),'collective candidate')
    need(isinstance(w['ranks'],list) and len(w['ranks'])==2 and len(set(w['ranks']))==2,'TP2 only; exactly two distinct ranks')
    need(set(w['ranks'])<=set(b.devs),'collective rank missing')
    need(all(b.devs[r]['arch']=='gfx1201' for r in w['ranks']),'GPU collective cannot use CPU timing profile')
    integer(w['bytes'],'collective bytes',1,2**48);integer(w['element_bytes'],'element bytes',1,8)
    need(w['element_bytes'] in (2,4) and w['bytes']%w['element_bytes']==0,'collective element alignment')
    need(p['algorithm'] in ('oneshot','ring2'),'unsupported collective algorithm')
    need(p['access'] in ('push','pull'),'explicit P2P access mode required')
    integer(p['chunk_bytes'],'chunk bytes',16,2**48);number(p['handshake_us'],'handshake')
    need(p['chunk_bytes']%16==0 and w['bytes']%16==0,'16-byte vector path required')
    a,z=w['ranks'];size=w['bytes'];elements=size//w['element_bytes'];deps=[];chunks=0;last_transfers={}
    starts={}
    for rank in (a,z):
        start='start:'+rank
        b.ps.append(phase(start,[],{},fixed_us=b.devs[rank]['launch_us'],reservations={rank+':kernels':1},label='collective launch-cost scenario'))
        starts[rank]=start
    if p['algorithm']=='oneshot':
        # Chunk-level model matches push -> handshake -> local reduce causality.
        # It does not emulate the real per-block flag protocol.
        for offset in range(0,size,p['chunk_bytes']):
            chunks+=1;need(chunks<=4096,'too many communication chunks')
            n=min(p['chunk_bytes'],size-offset);arr=[]
            for src,dst in ((a,z),(z,a)):
                # Stream on a directional link is FIFO. Both directions may overlap.
                td=[last_transfers[(src,dst)]] if (src,dst) in last_transfers else [starts[src]]
                t=b.transfer(f'exchange{chunks}:{src}>{dst}',src,dst,n,p['access'],td);last_transfers[(src,dst)]=t;arr.append(t)
            h=f'{len(b.ps)}:handshake{chunks}';b.ps.append(phase(h,arr,{},fixed_us=p['handshake_us'],label='modeled peer-visibility handshake'))
            for rank in (a,z):
                deps.append(b.kernel(f'local_reduce{chunks}:{rank}',{rank+':dram':3*n,rank+':vector':n/w['element_bytes']},device=rank,deps=[h],launch=False))
        b.workspace_bytes=4*size # two slots per rank, summed across two ranks
        wire=2*size
    else:
        need(size%32==0,'two-rank ring needs equal vector-aligned halves')
        half=size//2;need(p['chunk_bytes']>=half,'ring2 chunk subdivision not modeled; use chunk_bytes >= message/2')
        x=b.transfer('reduce_scatter:a>z',a,z,half,p['access'],[starts[a]])
        y=b.transfer('reduce_scatter:z>a',z,a,half,p['access'],[starts[z]])
        h='ring_handshake_1';b.ps.append(phase(h,[x,y],{},fixed_us=p['handshake_us'],label='reduce-scatter synchronization'))
        r1=b.kernel('reduce_local_a',{a+':dram':3*half,a+':vector':half/w['element_bytes']},device=a,deps=[h],launch=False)
        r2=b.kernel('reduce_local_z',{z+':dram':3*half,z+':vector':half/w['element_bytes']},device=z,deps=[h],launch=False)
        x=b.transfer('allgather:a>z',a,z,half,p['access'],[r1,r2])
        y=b.transfer('allgather:z>a',z,a,half,p['access'],[r1,r2])
        b.ps.append(phase('ring_handshake_2',[x,y],{},fixed_us=p['handshake_us'],label='allgather synchronization'))
        wire=2*size;chunks=2;b.workspace_bytes=2*size
    b.counts={'wire_payload_bytes_total':wire,'wire_payload_bytes_per_direction':size,'message_bytes_per_rank':size,
              'reduction_ops_total':2*elements if p['algorithm']=='oneshot' else elements,'chunks':chunks,
              'rank_count':2,'access':p['access'],'algorithm':p['algorithm']}
    b.warnings=['Directional push/pull rates, shared root bandwidth and protocol startup are explicit inputs.',
                'This is cost simulation, NOT a TP2 runtime backend or a fence/lockstep correctness test.',
                'Two launch costs per collective, chunk-transfer latency and modeled handshakes; residency/flags/checksums are not simulated.']

def dense(b:Builder):
    w=b.w;p=b.c['params'];dev=w['device'];d=b.devs[dev]
    fields(w,{'id','op','contract','device','gemm'}, {'note'},'dense wrapper')
    fields(p,{'candidate','profile'},set(),'dense params')
    gemm_workload(w['gemm']);gemm_candidate(p['candidate'])
    need(w['gemm']['contract']==w['contract']==p['candidate']['contract'],'GEMM wrapper contract mismatch')
    need(p['candidate']['arch']==d['arch'],'GEMM target mismatch')
    f=gemm_features(w['gemm'],p['candidate'],p['profile'])
    # Preserve v0.5 total scenario; fixed-duration tasks explicitly cannot model
    # additional bandwidth sharing without the original phase decomposition.
    b.ps=[phase('0:gemm_scenario',[],{},fixed_us=f['scenario_us'],reservations={dev+':kernels':d['kernel_slots']},label='v0.5 dense GEMM scenario')]
    b.counts=f;b.warnings=f['warnings']+['GEMM wrapper is a fixed-duration exclusive-device task; shared-resource overlap is not inferred.']

def explicit(b:Builder):
    w=b.w;p=b.c['params'];dev=w['device']
    fields(w,{'id','op','contract','device','service','fixed_us','workspace_bytes'}, {'note'},'explicit operator')
    fields(p,{'service_scale'},set(),'explicit candidate')
    number(p['service_scale'],'scale',positive=True);number(w['fixed_us'],'fixed latency');integer(w['workspace_bytes'],'workspace',0,2**48)
    need(isinstance(w['service'],dict),'service map required')
    dem={}
    rates,_=resources(b.s)
    for k,v in w['service'].items():
        name=dev+':'+k;need(name in rates,'unknown explicit service resource');number(v,k);dem[name]=v*p['service_scale']
    b.ps=[phase('0:explicit',[],dem,fixed_us=w['fixed_us'],reservations={dev+':kernels':1},label='declared opaque work scenario')]
    b.workspace_bytes=w['workspace_bytes'];b.counts={'service':dem,'fixed_us':w['fixed_us']}
    b.warnings=['Explicit cost supplied by caller; no semantic operation or hardware equivalence inferred.']

def resolve_system(w:dict,s:dict)->dict:
    """Apply a declared per-family rate override. An unknown family is an error, never
    a silent fallback to the device-wide rate: that would hide a measurement gap."""
    fam=w.get('family')
    table=s.get('operator_rates') or {}
    if fam is None: return s
    # A profile that declares no family at all is a device-wide assumption profile: it makes
    # no family claim, so the workload's family cannot be unmet. Once a profile does declare
    # families, every declared family must be covered or the gap would be silent.
    if not table: return s
    need(fam in table,f'no measured rates for operator family: {fam}')
    ov=table[fam];out=dict(s);out['devices']=[]
    for d in s['devices']:
        nd=dict(d)
        for k,v in ov.items():
            nd[k]={**d['matrix_ops_per_us'],**v} if k=='matrix_ops_per_us' else v
        out['devices'].append(nd)
    return out


def build(w:dict,c:dict,s:dict)->dict:
    s=resolve_system(w,s)
    base(w,c,s);b=Builder(w,c,s)
    {'attention':attention,'qsa_indexer':qsa_indexer,'moe':moe,'collective':collective,'gemm':dense,'explicit':explicit}[w['op']](b)
    return b.finish()

def op_plan(cat:dict,workloads:dict,s:dict)->dict:
    from .common import PlanError
    fields(cat,{'schema','candidates'}, {'source','notes'},'operator catalog')
    fields(workloads,{'schema','workloads'}, {'scope','notes'},'operator workloads')
    need(cat['schema']=='radiance-operator-catalog/1' and workloads['schema']=='radiance-operator-workloads/1','wrong operator schemas')
    id_check(cat['candidates']);id_check(workloads['workloads']);system_check(s)
    rows=[]
    for w in workloads['workloads']:
        ranked=[];reject=[]
        for c in cat['candidates']:
            need(c.get('op') in OPS and w.get('op') in OPS,'unknown or missing operator type')
            if c['op']!=w['op']:continue
            try:ranked.append(build(w,c,s))
            except (PlanError,KeyError,TypeError) as e:reject.append({'candidate_id':c['id'],'reason':str(e)})
        ranked.sort(key=lambda x:(x['scenario_us'],x['candidate_id']))
        rows.append({'workload_id':w['id'],'workload':w,'ranked':ranked,'rejected':reject,
                     'status':'NEEDS_MEASUREMENT' if ranked else 'NO_SUPPORTED_CANDIDATE',
                     'suggested_candidate':ranked[0]['candidate_id'] if ranked else None,
                     'measurement_queue':[r['candidate_id'] for r in ranked]})
    return {'schema':'radiance-operator-plan/1','system_sha256':digest(s),'catalog_sha256':digest(cat),
            'workloads_sha256':digest(workloads),'workloads':rows,'status':'OPERATOR_SCENARIOS_CREATED',
            'gpu_executed':False,'deployment_approved':False,'production_latency_proved':False,
            'scope':workloads.get('scope',{'complete_model':False})}
