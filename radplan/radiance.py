"""A deliberately narrow, hash-pinned adapter for the supplied Radiance snapshot.

Extracts source-declared instantiations, not an invented template search space. Does not compile
HIP, guess launcher defaults, rewrite kernels or bless the known scale-tail read.
"""
from __future__ import annotations
import ast
import copy
from pathlib import Path
import re
from .common import need, PlanError, file_hash, digest, read, hash_string
from .model import catalog_check

KNOWN_SOURCE = {'libr4d/r4d_gemm_w4a8_prefill.hip': '775590fdfd31b77fee636228ba820ff8b9943610b55bf0b1ee86d68370597a6e', 'libr4d/r4d_gemm_args.h': '1e3e58015de472dbf7bcc3b9fc77cefd7308b85525ea14f88da45904ce621e1e', 'libr4d/r4d_rows.cpp': '002478f4cafe4eac504c6df8b4cda00c6a35e402b1eeef32f6429afdcbf984c3'}  # Reviewed hashes of the supplied source snapshot.
CONTRACT='radiance-w4a8-symmetric-int8A-signed4W-f16scale128-bf16C'


def int_expr(s: str) -> int:
    node=ast.parse(s.strip(),mode='eval').body
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) is int and 0<=n.value<65536:return n.value
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.BitOr):return visit(n.left)|visit(n.right)
        raise PlanError('only integer/bitwise-OR template constants are accepted')
    return visit(node)


def extract(root: Path) -> dict:
    root=Path(root)
    hashes={name:file_hash(root/name) for name in KNOWN_SOURCE}
    need(hashes==KNOWN_SOURCE,'source differs from reviewed adapter snapshot; update/review adapter, do not reuse stale tile assumptions')
    text=(root/'libr4d/r4d_gemm_w4a8_prefill.hip').read_text()
    rows=(root/'libr4d/r4d_rows.cpp').read_text()
    ids=re.search(r'kPrefillArms\[\]\s*=\s*\{([^}]+)\}',rows)
    need(ids is not None,'prefill tunable table missing')
    expected={int(x) for x in ids[1].split(',')}
    arms={int(v):[int_expr(x) for x in values.split(',')] for v,values in re.findall(r'case\s+(\d+):\s+PF\(([^)]+)\)',text)}
    default=re.search(r'default:\s+PF\(([^)]+)\)',text)
    need(default is not None,'default arm missing');arms[0]=[int_expr(x) for x in default[1].split(',')]
    need(set(arms)==expected,'launcher/tunable coverage mismatch')
    mapped=re.search(r'static const Arm pf\[\]\s*=\s*\{(.*?)\};',rows,re.S)
    need(mapped is not None,'tile map missing')
    table={int(i):(int(m),int(n)) for i,m,n in re.findall(r'\{\s*(\d+),\s*(\d+),\s*(\d+)\s*\}',mapped[1])}
    candidates=[]
    for ident,(wm,wn,tm,tn,opt,gsw) in sorted(arms.items()):
        bm,bn=wm*tm*16,wn*tn*16
        need(table[ident]==(bm,bn),'source tile map contradicts template')
        # Exact static shared arrays: lds_a[2][BM*16] and lds_w[2][BN/16*128], uint32.
        lds=2*(bm*64+bn*32)
        c={'id':f'w4a8-prefill-tile{ident}','family':'radiance-w4a8-prefill','backend':'aql','arch':'gfx1201',
           'contract':CONTRACT,'compute_path':'wmma_i8','tile':[bm,bn,64],
           'wave_size':32,'waves_per_group':wm*wn,'stages':2,'lds_bytes':lds,
           'tail':{'M':True,'N':False,'K':False},'multiples':{'M':1,'N':256,'K':128},
           'work_policy':'full_tiles','identity':{'kind':'source','sha256':digest(hashes),'provenance':'hash-pinned source adapter; not compiled binary'},
           'launch':{'row':'gemm_w4a8_prefill','op':'gemm_nt_q','params':{'tile':ident,'dtype':'w4a8','group':128}},
           'format':{'a_bits':8,'b_bits':4,'out_bits':16,'weight_group':128,'weight_scale_bytes':2,'activation_scale_bytes':4},
           'unpack_ops_per_weight':0.375,'matrix_alu_overlap':False,
           'lds_bytes_per_kblock':bm*64+bn*32+bm*64*wn+bn*32*wm,
           'known_issues':[] if opt&128 else ['scalar_scale_tail'],
           'notes':f'WM={wm}, WN={wn}, TM={tm}, TN={tn}, OPT={opt}, GSW={gsw}. Registry additionally requires N%256, even for BN=128.',
           'unmodeled':['scale-fold/address instructions not fully counted','cache swizzle/hints and b64/b128 instruction scheduling not inferred','no GPU correctness/ISA result for this source arm']}
        candidates.append(c)
    return catalog_check({'schema':'radiance-planner-catalog/1','source':{'files':hashes,'adapter':'reviewed-w4a8-prefill/1'},
        'notes':'Only this family is extracted; not all Radiance kernels. Source-known scalar-scale tails excluded. No auto dispatch changes.',
        'candidates':candidates})


def bind_resources(cat: dict, manifest_path: Path, report_path: Path, mapping: dict) -> dict:
    """Explicit candidate->object+symbol map, cross-checked against actual evidence bytes."""
    catalog_check(cat);manifest=read(manifest_path);report=read(report_path)
    need(report.get('schema')=='radiance-rdna4-isa/2','unsupported ISA report')
    need(report.get('manifest_sha256')==file_hash(manifest_path),'ISA report/manifest hash mismatch')
    need(manifest.get('collection_complete') is True,'incomplete ISA collection')
    need(report.get('status')=='STATIC_CHECKS_PASS' and report.get('strict_review') is True,'need passing strict static evidence (not numerical proof)')
    need(isinstance(mapping,dict) and set(mapping)<= {c['id'] for c in cat['candidates']},'invalid candidate mapping')
    need(bool(mapping),'empty resource mapping')
    objects={o['sha256']:o for o in manifest['objects']}
    kernels={(k['object_sha256'],k['symbol']):k for k in report['kernels']}
    result=copy.deepcopy(cat)
    for c in result['candidates']:
        if c['id'] not in mapping:continue
        d=mapping[c['id']];need(set(d)=={'object_sha256','symbol','plugin_sha256'},'mapping requires plugin_sha256/object_sha256/symbol')
        h=hash_string(d['object_sha256'],'object');need(h in objects,'object absent from manifest')
        ph=hash_string(d['plugin_sha256'],'plugin')
        inputs=[x for x in manifest['inputs'] if x['sha256']==ph]
        need(len(inputs)==1,'plugin absent or ambiguous in manifest inputs')
        from radisa.binary import extract_code_objects
        plugin_path=Path(inputs[0]['path']);need(file_hash(plugin_path)==ph,'plugin input changed')
        extracted,_=extract_code_objects(plugin_path.read_bytes(),str(plugin_path))
        need(h in {o['sha256'] for o in extracted},'mapped object is not embedded in this plugin')
        obj=objects[h];p=(manifest_path.parent/obj['path']).resolve()
        need(p.is_relative_to(manifest_path.parent.resolve()),'object path escapes evidence')
        need(file_hash(p)==h,'code-object hash mismatch')
        k=kernels.get((h,d['symbol']));need(k is not None,'kernel absent from report')
        from radreplay.cli import isa_binding
        isa_binding(manifest_path,report_path,{ph},{ph:plugin_path})
        from radisa.binary import ELF,kernel_inventory
        actual,issues=kernel_inventory(ELF(p.read_bytes()))
        need(not issues,'actual object inventory has issues')
        live=[x for x in actual if x['symbol']==d['symbol']]
        need(len(live)==1 and not live[0]['issues'],'actual descriptor absent/invalid')
        originals=[x for x in obj['kernels'] if x['symbol']==d['symbol']];need(len(originals)==1,'ambiguous metadata')
        md=live[0]['metadata'];r=k['resources'];need(isinstance(r,dict),'resource metadata missing')
        need(md==originals[0]['metadata'],'stored metadata differs from object')
        need(r['logical_vgprs']==md['.vgpr_count'] and r['descriptor_vgprs']==live[0]['descriptor']['encoded_vgprs'],'register report differs from binary')
        need(md['.wavefront_size']==c['wave_size'],'wave size disagrees with candidate')
        need(not any(f['severity'] in ('error','incomplete','review') for f in k['findings']),'ISA candidate has unresolved findings')
        c['resources']={'vgprs':r['logical_vgprs'],'descriptor_vgprs':r['descriptor_vgprs'],'scratch_bytes':k['scratch_bytes_per_workitem'],
            'origin':'compiler_metadata','object_sha256':h,'symbol':d['symbol'],'report_sha256':file_hash(report_path)}
        c['lds_bytes']=md['.group_segment_fixed_size']
        c['identity'].update({'object_sha256':h,'plugin_sha256':ph,'symbol':d['symbol']})
    return catalog_check(result)
