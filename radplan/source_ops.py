"""Reviewed source geometry and bounded Qwen4-Exp shape expansion.

No checkpoint dimensions are guessed. This importer emits PARTIAL operator
workloads and a missing-component ledger, not an executable full-model graph.
"""
from __future__ import annotations
import copy
from pathlib import Path
from .common import fields,need,integer,digest,file_hash
from .flow import text

PINNED = {
    'arch/qwen4exp_fp8/qwen4exp_fp8.cpp': '046a79885406c0904d7ca31c369164a979f8950fb5e272492602e270d07ff0bf',
    'arch/common/rad_qsa.h': '8ba249047f321acde05e73b7889ac3cdf3c7910c7130d0e6f471cccb9ffa6d58',
    'arch/common/rad_block_moe_fp8.h': '310072e6bf1299f6c38ceb33b99fceb712ef79cc3403d1f84cb278a3c056c827',
    'docs/QSA.md': '01b7d2b3c6fc24a9c387c95664a54cff618bed4593a14294e97c1d94782d77c7',
    'libr4d/r4d_attn_paged_h256_gqa12.hip': 'a0078f0a42c8f6001329c51a6342c6d4365cefe49d97686b45542acd74c82f1c',
    'libr4d/r4d_attn_paged_h256_gqa6_bs4.hip': '7d9d0084a6c26102d2be4b821bceb92bbe01bbdfb9f08aafa2b0681b0f302ea5',
    'libr4d/r4d_attn_paged_gqa8.hip': 'dd5f504dcb7ef6c854cecd5c9b489d9b25e0b3baa03da89925bee87adee5713a',
    'libr4d/r4d_attn_decode_h256_gqa6.hip': 'f808b6eaa86459dc92f719bc241f59e926788126e8671728c643b6e37ea3dc6b',
    'libr4d/r4d_attn_prefill_h256_gqa6.hip': 'bf167b31d07c93d739baa4119fd4c37c9df0c77954a3fcfd96dc5b715a61a818',
    'libr4d/r4d_moe.hip': '7269dd7c51d29ba70a1a744edd54715cd6fbecc88574b45a06d28366d86edad2',
    'libr4d/r4d_ar_oneshot_2rank_exact.hip': '76cab43325c355f9427ffa0940c2f324940fca4481e08c07f4f74b780ceba8de',
    'libr4d/r4d_rows.cpp': 'a1b1e86a16a43904bb49af7ae97271e6603fa2057f44799fc6464ef6d81b3e14',
}


def verify_source(root:Path):
    out={}
    need(PINNED,'source pin table missing')
    for name,wanted in PINNED.items():
        p=(root/name).resolve();need(p.is_relative_to(root.resolve()) and p.is_file(),'source file missing: '+name)
        actual=file_hash(p);need(actual==wanted,'source changed; adapter review required: '+name);out[name]=actual
    return out


def make_candidate(cid,op,contract,params,*,binding=None,input_layout=None,output_layout=None):
    semantic={'id':cid,'op':op,'contract':contract,'params':params}
    r=dict(semantic,identity={'kind':'hypothesis','sha256':digest(semantic)})
    if binding:r['source_binding']=binding
    if input_layout:r['input_layout']=input_layout
    if output_layout:r['output_layout']=output_layout
    return r


def extended_catalog(root:Path):
    hashes=verify_source(root);binding={'source_files':hashes,'scope':'reviewed family geometry; costs remain hypotheses, not executable code or exact dispatch bindings'}
    cs=[]
    for gqa in (12,6):
        for bits in (8,16):
            for mode in ('decode','prefill'):
                for splits in (1,4,16):
                    params={'query_tile':1 if mode=='decode' else 384//gqa,
                            'kv_tile':16 if mode=='decode' else 48,'split_kv':splits,
                            'gqa_reuse':True,'compute_path':'wmma_f16','head_dim':256,'gqa':gqa,'block_size':4,
                            'max_queries_per_sequence':64//gqa if mode=='decode' else 65536,
                            'force_merge':mode=='decode','kv_bits':bits,'partial_bits':16}
                    cs.append(make_candidate(f'attn.g{gqa}.{bits}.{mode}.s{splits}','attention',f'bf16-q-fp{bits}-kv-selected-causal',params,
                                             binding=binding,input_layout='qkv_paged',output_layout='bf16_heads'))
    for tile in (32,64,128):
        cs.append(make_candidate(f'qsa.radix.t{tile}','qsa_indexer','qsa-block-score-select',
                    {'score_tile':tile,'selection':'radix_passes','passes':4,'compute_path':'wmma_f16'},binding=binding,
                    input_layout='index_keys',output_layout='selected_blocks'))
    for bm in (16,32):
        for bn in (64,128):
            cs.append(make_candidate(f'moe.w4a8.m{bm}.n{bn}','moe','routed-w4a8-gated-ffn',
                {'bm':bm,'bn':bn,'bk':128,'compute_path':'wmma_fp8','include_empty_experts':False,'router_selection_passes':10},
                binding=binding,input_layout='bf16_hidden',output_layout='bf16_hidden'))
    for access in ('push','pull'):
        for algo in ('oneshot','ring2'):
            for chunk in ((4096,65536) if algo=='oneshot' else (2**30,)):
                cs.append(make_candidate(f'tp2.{algo}.{access}.c{chunk}','collective','tp2-bf16-sum',
                  {'algorithm':algo,'access':access,'chunk_bytes':chunk,'handshake_us':1.},binding=binding,
                  input_layout='bf16_hidden',output_layout='bf16_hidden'))
    return {'schema':'radiance-operator-catalog/1','source':binding,'candidates':cs,
            'notes':'Finite algorithm/template search. Source macros justify geometry only. No native kernel code generated or dispatched.'}


def block_slices(width:int,world:int,expert:int,block:int=128):
    integer(width,'width');integer(world,'world',1,4);need(world in (1,2,4),'reviewed TP modes are 1,2,4');integer(expert,'expert',0,65535);integer(block,'block')
    need(width%block==0 and world<=width//block,'whole scale blocks required for every rank')
    nb=width//block
    if nb%world==0:return [(r*width//world,(r+1)*width//world) for r in range(world)]
    klass=expert%4 if world==4 else expert%2
    slices=[]
    for rank in range(world):
        place=(rank-klass+world)%world;base,extra=divmod(nb,world)
        lo=(place*base+min(place,extra))*block
        slices.append((lo,lo+(base+int(place<extra))*block))
    return slices


def shared_slices(shared_width:int,expert_width:int,world:int):
    integer(shared_width,'shared width',0);integer(expert_width,'expert width');integer(world,'world',1,4)
    need(world in (1,2,4),'reviewed TP modes are 1,2,4')
    if shared_width==0:return [(0,0)]*world
    need(shared_width%128==0 and world<=shared_width//128,'shared expert must split in whole blocks')
    nb=shared_width//128;first=(expert_width//64)%world;base,extra=divmod(nb,world)
    result=[]
    for r in range(world):
        place=(r-first+world)%world;lo=(place*base+min(place,extra))*128
        result.append((lo,lo+(base+int(place<extra))*128))
    return result


def model_workloads(config:dict,run:dict,source:Path)->dict:
    hashes=verify_source(source)
    # Text-config nesting is explicit and root architecture must agree.
    cfg=config.get('text_config',config)
    need(isinstance(cfg,dict),'text_config must be object')
    need(cfg.get('model_type',config.get('model_type'))=='qwen4_exp','adapter supports supplied qwen4_exp source only')
    keys=['hidden_size','num_attention_heads','num_key_value_heads','head_dim','moe_intermediate_size',
          'shared_expert_intermediate_size','num_experts','num_experts_per_tok','num_hidden_layers',
          'full_attention_interval','indexer_n_heads','indexer_head_dim','indexer_compress_ratio','indexer_budget']
    for k in keys:
        need(k in cfg,'missing explicit model field: '+k)
        integer(cfg[k],k,0 if k=='shared_expert_intermediate_size' else 1,2**24)
    need(cfg['full_attention_interval']>1,'hybrid interval invalid')
    fields(run,{'schema','world_size','devices','query_lengths','sequence_ids','routing_by_layer','selection_by_layer',
                'kv_bits','weight_bits','activation_bits','scale_group','scale_bytes','cold_weight_fraction','updated_blocks_by_layer'},
                {'note'},'model run')
    need(run['schema']=='radiance-model-run/1','wrong run schema')
    world=run['world_size'];integer(world,'world size',1,4);need(world in (1,2,4),'reviewed partition modes: 1,2,4')
    need(isinstance(run['devices'],list) and len(run['devices'])==world and len(set(run['devices']))==world,'one unique device per rank')
    h=cfg['hidden_size'];e=cfg['num_experts'];ff=cfg['moe_intermediate_size'];layers=cfg['num_hidden_layers']
    need(layers<=256 and e<=65536,'model adapter resource cap')
    need(cfg['num_attention_heads']%world==0,'query heads not divisible by ranks')
    kv=cfg['num_key_value_heads']
    need(kv%world==0 or world%kv==0,'unsupported KV replication')
    local_kv=max(1,kv//world);local_q=cfg['num_attention_heads']//world
    need(local_q%local_kv==0,'local GQA mismatch')
    need(run['kv_bits'] in (8,16),'only FP8/BF16 KV in this adapter')
    need(run['weight_bits']==4 and run['activation_bits']==8,'this adapter describes W4A8 routed experts only')
    need(run['scale_group'] in (32,64,128),'explicit scale group required')
    need(cfg['indexer_budget']%cfg['indexer_compress_ratio']==0,'indexer budget must divide block size')
    qs=len(run['query_lengths']);need(qs>0,'no queries')
    need(len(run['sequence_ids'])==qs,'query sequence lengths mismatch')
    # Validate sequence geometry even if no full attention layer occurs.
    from .operators import queries_check
    queries_check(run)
    full=[i for i in range(layers) if (i+1)%cfg['full_attention_interval']==0]
    if 'layer_types' in cfg:
        need(isinstance(cfg['layer_types'],list) and len(cfg['layer_types'])==layers,'layer schedule length')
        for i,t in enumerate(cfg['layer_types']):
            need((t in ('full_attention','qwen_sparse_attention'))==(i in full),'layer schedule differs from interval')
    for key in ('routing_by_layer','selection_by_layer','updated_blocks_by_layer'):need(isinstance(run[key],dict),'per-layer observed/scenario data required')
    need(set(run['routing_by_layer'])=={str(i) for i in range(layers)},'routing histogram required for EVERY layer; no uniform inference')
    need(set(run['selection_by_layer'])=={str(i) for i in full},'selection required for EVERY QSA layer')
    need(set(run['updated_blocks_by_layer'])=={str(i) for i in full},'block updates required for every QSA layer')
    shared=shared_slices(cfg['shared_expert_intermediate_size'],ff,world)
    slices=[block_slices(ff,world,x) for x in range(e)]
    out=[];dense_requests=[];parts=[]
    for rank,dev in enumerate(run['devices']):
        parts.append({'rank':rank,'device':dev,'q_heads':local_q,'kv_heads':local_kv,
                      'expert_widths':[hi-lo for sl in slices for lo,hi in [sl[rank]]],
                      'shared_width':shared[rank][1]-shared[rank][0]})
    for layer in range(layers):
        hist=run['routing_by_layer'][str(layer)]
        need(isinstance(hist,list) and len(hist)==e,'routing histogram length differs')
        for n in hist:integer(n,'expert tokens',0,qs)
        need(sum(hist)==qs*cfg['num_experts_per_tok'],'routing count differs from queries*topk')
        for part in parts:
            r=part['rank'];dev=part['device']
            if layer in full:
                sel=run['selection_by_layer'][str(layer)]
                # null means dense fallback; it is safe only below the bound for
                # every query. Beyond that the actual per-query selection is required.
                bs=cfg['indexer_compress_ratio'];keep=cfg['indexer_budget']//bs
                if sel is None:need(all(n<=cfg['indexer_budget']+bs-1 for n in run['query_lengths']), 'missing sparse selections beyond dense bound')
                else:
                    need(isinstance(sel,list) and len(sel)==qs,'one selection per query')
                    for length,ids in zip(run['query_lengths'],sel):
                        complete=length//bs
                        need(isinstance(ids,list) and len(ids)==len(set(ids)),'unique selection entries required')
                        need(all(type(j) is int for j in ids),'integer QSA block ids required')
                        need(len([j for j in ids if j<complete])==min(keep,complete),'wrong number of complete QSA blocks')
                        need((complete in ids)==bool(length%bs),'QSA partial tail must be included exactly when present')
                        need(all(type(j) is int and 0<=j<(length+bs-1)//bs for j in ids),'selection block outside visible prefix')
                out.append({'id':f'L{layer}.r{r}.qsa','op':'qsa_indexer','contract':'qsa-block-score-select','device':dev,
                            'query_lengths':run['query_lengths'],'sequence_ids':run['sequence_ids'],'block_size':bs,
                            'heads':cfg['indexer_n_heads'],'head_dim':cfg['indexer_head_dim'],'keep_blocks':keep,
                            'updated_blocks':run['updated_blocks_by_layer'][str(layer)],'skip_score_when_dense':sel is None})
                out.append({'id':f'L{layer}.r{r}.attn','op':'attention','contract':f'bf16-q-fp{run["kv_bits"]}-kv-selected-causal','device':dev,
                            'query_lengths':run['query_lengths'],'sequence_ids':run['sequence_ids'],'selected_blocks':sel,
                            'q_heads':local_q,'kv_heads':local_kv,'head_dim':cfg['head_dim'],'block_size':bs,
                            'q_bits':16,'kv_bits':run['kv_bits'],'out_bits':16,'kv_scale_bytes_per_block':0})
                for name,n,k in [('q_gate',2*local_q*cfg['head_dim'],h),('k',local_kv*cfg['head_dim'],h),
                                 ('v',local_kv*cfg['head_dim'],h),('o',h,local_q*cfg['head_dim']),
                                 ('index_qk',(cfg['indexer_n_heads']+1)*cfg['indexer_head_dim'],h)]:
                    dense_requests.append({'id':f'L{layer}.r{r}.{name}','M':qs,'N':n,'K':k,
                                           'status':'NEEDS_RECIPE_FORMAT_AND_MATCHING_GEMM_CATALOG'})
            out.append({'id':f'L{layer}.r{r}.moe','op':'moe','contract':'routed-w4a8-gated-ffn','device':dev,
                        'tokens':qs,'hidden':h,'top_k':cfg['num_experts_per_tok'],'expert_tokens':hist,
                        'expert_widths':part['expert_widths'],'shared_width':part['shared_width'],'weight_bits':run['weight_bits'],
                        'activation_bits':run['activation_bits'],'scale_group':run['scale_group'],'scale_bytes':run['scale_bytes'],
                        'cold_weight_fraction':run['cold_weight_fraction'],'router':'dense_topk'})
    # Unsupported components are NEVER assigned zero execution time in a full model.
    scope={'complete_model':False,'model_type':'qwen4_exp','layers':layers,'full_attention_layers':full,
           'unmodeled_recurrent_layers':[i for i in range(layers) if i not in full],
           'missing_components':['GDN/recurrent layers','HC/PLE/norm/RoPE/KV-write','model-specific packed byte layout',
                                 'MTP/drafter and logits','quantized dense projection dispatch','communication insertion/fusion'],
           'dense_shape_requests':dense_requests,'partition':parts,'source_sha256':digest(hashes),
           'config_sha256':digest(config),'run_sha256':digest(run),'tokens_per_second_predicted':False}
    return {'schema':'radiance-operator-workloads/1','workloads':out,'scope':scope,
            'notes':'Partial declared work only; selectors/histograms are supplied, not predicted from model name.'}
