import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from radreplay.builder import Package
from radreplay.format import ReplayError, validate, read_json, checked_span

class FormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.plugin=self.root/'dummy.so';self.plugin.write_bytes(b'not native code; validator-only fixture')
        p=Package(self.root/'case','test');p.plugin('test',self.plugin);p.stream()
        p.allocation('x',bytes(range(16)));p.allocation('y',bytes(16),mutable=True)
        p.view('x','x',nbytes=16,dtype=9,shape=[16]);p.view('y','y',nbytes=16,dtype=9,shape=[16])
        p.kernel('k0','p0','copy','copy',['x','y'],[('x','r'),('y','w')]);p.check('y',bytes(range(16)))
        self.path=p.finish();self.m=read_json(self.path)
    def tearDown(self):self.tmp.cleanup()
    def save(self):self.path.write_text(json.dumps(self.m))
    def invalid(self,change):
        change(self.m);self.save()
        with self.assertRaises((ReplayError,KeyError,TypeError,OSError)):validate(self.path)
    def test_valid_inventory(self):
        i=validate(self.path);self.assertEqual(i['kernel_nodes'],['k0']);self.assertFalse(i['runtime_executed'])
    def test_duplicate_json(self):
        self.path.write_text('{"schema":1,"schema":2}')
        with self.assertRaises(ReplayError):read_json(self.path)
    def test_nan_json(self):
        self.path.write_text('{"x":NaN}')
        with self.assertRaises(ReplayError):read_json(self.path)
    def test_blob_hash(self):
        b=self.m['allocations'][0]['initial'];(self.path.parent/b['path']).write_bytes(bytes(16))
        with self.assertRaises(ReplayError):validate(self.path)
    def test_symlink_blob(self):
        b=self.m['allocations'][0]['initial'];p=self.path.parent/b['path'];q=self.root/'escape';q.write_bytes(p.read_bytes());p.unlink();p.symlink_to(q)
        with self.assertRaises(ReplayError):validate(self.path)
    def test_budget(self):
        with self.assertRaises(ReplayError):validate(self.path,budget=16)
    def test_partial_output_not_sufficient(self):
        self.m['views'].append(dict(self.m['views'][1],id='part',bytes=8,shape=[8],stride=[1]))
        c=self.m['checks'][0];c['view']='part'
        p=Package(self.root/'helper','helper');c['expected']=p.blob(bytes(range(8)))
        # copy this helper's blob into this case.
        src=p.root/c['expected']['path'];dst=self.path.parent/c['expected']['path'];dst.write_bytes(src.read_bytes())
        self.save()
        with self.assertRaises(ReplayError):validate(self.path)
    def test_explicit_relocation(self):
        p=Package(self.root/'pointer','pointer');p.plugin('test',self.plugin);p.stream()
        p.allocation('x',b'abcd');p.allocation('y',bytes(4),mutable=True)
        p.allocation('ptr',b'address!',relocations=[{'offset':0,'target':'x','addend':0}])
        p.view('x','x',nbytes=4,dtype=9,shape=[4]);p.view('y','y',nbytes=4,dtype=9,shape=[4]);p.view('ptr','ptr',nbytes=8,dtype=12,shape=[1])
        p.kernel('k','p0','indirect','probe',['ptr','y'],[('ptr','r'),('x','r'),('y','w')]);p.check('y',b'abcd')
        path=p.finish();self.assertEqual(validate(path)['node_count'],1)
        m=read_json(path);m['nodes'][0]['accesses'].pop(1);path.write_text(json.dumps(m))
        with self.assertRaises(ReplayError):validate(path)
    def test_pointer_addresses_must_be_zero(self):
        self.m['allocations'][0]['relocations']=[{'offset':0,'target':'y','addend':0}];self.save()
        with self.assertRaises(ReplayError):validate(self.path)
    def test_alias_views(self):
        self.m['views'].append(dict(self.m['views'][1],id='alias'))
        self.m['nodes'][0]['tensors'][1]='alias';self.m['nodes'][0]['accesses'][1]['view']='alias'
        self.save();validate(self.path)
    def gpu(self):
        self.m['target'].update(backend='aql',arch='gfx1201')
        for a in self.m['allocations']:a['kind']='device'
        self.m['streams'].append({'id':'q1','priority':0})
        k=copy.deepcopy(self.m['nodes'][0]);k.update(id='k1',stream='q1');return k
    def test_unordered_conflict(self):
        self.m['nodes'].append(self.gpu());self.save()
        with self.assertRaises(ReplayError):validate(self.path)
    def test_events_make_cross_stream_order(self):
        k=self.gpu();self.m['nodes'] += [dict(id='record',kind='record',stream='q0',phase='measure',event='e'),dict(id='wait',kind='wait',stream='q1',phase='measure',event='e'),k]
        self.save();self.assertEqual(validate(self.path)['stream_count'],2)
    def test_packed_subbyte_row_span(self):
        self.assertEqual(checked_span(dict(dtype=23,shape=[2,5],stride=[5,1],bytes=4)),4)
    def test_fresh_destination(self):
        with self.assertRaises(ReplayError):Package(self.path.parent,'test')

# Each case exercises a distinct rejected contract rather than one omnibus smoke test.
mutations={
'unknown_field':lambda m:m.update(surprise=1),
'unsupported_tp2':lambda m:m['target'].update(world_size=2),
'backend_fallback':lambda m:m['target'].update(arch='gfx1201'),
'raw_tape':lambda m:m['contract'].update(execution='raw-aql'),
'hidden_state':lambda m:m['contract'].update(hidden_state='unknown'),
'incomplete_pointers':lambda m:m['contract'].update(pointer_fields_complete=False),
'wrong_abi':lambda m:m['plugins'][0].update(abi=29),
'duplicate_allocation':lambda m:m['allocations'].append(copy.deepcopy(m['allocations'][0])),
'bad_alignment':lambda m:m['allocations'][0].update(alignment=24),
'unknown_memory':lambda m:m['allocations'][0].update(kind='managed'),
'view_oob':lambda m:m['views'][0].update(offset=1),
'stride_oob':lambda m:m['views'][0].update(stride=[2]),
'negative_stride':lambda m:m['views'][0].update(stride=[-1]),
'overflow_shape':lambda m:m['views'][0].update(shape=[2**63]),
'unknown_dtype':lambda m:m['views'][0].update(dtype=15),
'missing_tensor':lambda m:m['nodes'][0]['tensors'].__setitem__(0,'absent'),
'missing_access':lambda m:m['nodes'][0]['accesses'].pop(0),
'write_immutable':lambda m:m['allocations'][1].update(mutable=False),
'no_output_checks':lambda m:m.update(checks=[]),
'byte_tolerance':lambda m:m['checks'][0].update(atol=1),
'unknown_compare':lambda m:m['checks'][0].update(mode='silently-pass'),
'nan_tolerance':lambda m:m['checks'][0].update(rtol=float('nan')),
'float_wrong_dtype':lambda m:m['checks'][0].update(mode='float'),
'prefix_only':lambda m:m['nodes'][0].update(phase='prefix'),
'chain_mode':lambda m:m['nodes'][0]['params'].append({'key':'chain','kind':'int','value':1}),
'unresolved_wait':lambda m:m['nodes'].insert(0,dict(id='wait',kind='wait',stream='q0',phase='prefix',event='absent')),
'host_two_streams':lambda m:m['streams'].append({'id':'q1','priority':0}),
'path_escape':lambda m:m['allocations'][0]['initial'].update(path='../escape'),
'empty_kernel_inventory':lambda m:m.update(nodes=[]),
'unused_plugin':lambda m:m['plugins'].append(dict(m['plugins'][0],id='extra')),
'bad_plugin_digest':lambda m:m['plugins'][0].update(sha256='0'),
'mutable_pointer_table':lambda m:m['allocations'][1].update(relocations=[{'offset':0,'target':'x','addend':0}]),
'pointer_oob':lambda m:m['allocations'][1].update(relocations=[{'offset':16,'target':'x','addend':0}]),
'pointer_target_missing':lambda m:m['allocations'][1].update(relocations=[{'offset':0,'target':'absent','addend':0}]),
'pointer_addend_oob':lambda m:m['allocations'][1].update(relocations=[{'offset':0,'target':'x','addend':16}]),
'overlapping_copy':lambda m:m['nodes'].append(dict(id='c',kind='copy',stream='q0',phase='measure',src='y',dst='y',bytes=4)),
}
for name, change in mutations.items():
    def method(self,change=change):self.invalid(change)
    setattr(FormatTests,'test_reject_'+name,method)
