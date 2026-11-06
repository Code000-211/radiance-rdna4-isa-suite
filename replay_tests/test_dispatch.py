import copy,json,tempfile,unittest,subprocess,shutil,os
from pathlib import Path
from radreplay.dispatch import bind,allowed_identities
from radreplay.format import ReplayError

class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.path=self.root/'trace.json'
        self.h='a'*64;self.identity='b'*64+':foo.kd'
        self.fixture={'nodes':[{'id':'k','kind':'kernel'}]}
        self.isa={'objects':[{'sha256':'b'*64,'kernels':[{'symbol':'foo.kd'}]}]}
        e={'identity':self.identity,'node':'k','phase':'diagnostic-node','pattern':0,'device':0,'queue':1,
           'grid':[32,1,1],'group':[32,1,1],'group_bytes':0,'private_bytes':0}
        self.t={'schema':'radiance-dispatch-trace/1','completed':True,'error':'','fixture_sha256':self.h,'entries':[dict(e,pattern=i) for i in range(3)]}
    def tearDown(self):self.tmp.cleanup()
    def call(self):self.path.write_text(json.dumps(self.t));return bind(self.path,self.fixture,self.h,self.isa,0)
    def test_success_scoped(self):r=self.call();self.assertEqual(r['status'],'DISPATCH_BINDING_PASS');self.assertFalse(r['timed_dispatches_observed'])
    def test_empty_blocked(self):
        self.t['entries']=[]
        with self.assertRaises(ReplayError):self.call()
    def test_missing_pass_blocked(self):
        self.t['entries'].pop()
        with self.assertRaises(ReplayError):self.call()
    def test_stale_blocked(self):
        self.t['fixture_sha256']='f'*64
        with self.assertRaises(ReplayError):self.call()
    def test_unapproved_symbol_blocked(self):
        self.t['entries'][0]['identity']='b'*64+':helper.kd'
        with self.assertRaises(ReplayError):self.call()
    def test_other_device_blocked(self):
        self.t['entries'][0]['device']=1
        with self.assertRaises(ReplayError):self.call()
    def test_unmodeled_phase_blocked(self):
        self.t['entries'][0]['phase']='raw-tape'
        with self.assertRaises(ReplayError):self.call()
    def test_unknown_node_blocked(self):
        self.t['entries'][0]['node']='missing'
        with self.assertRaises(ReplayError):self.call()
    def test_partial_run_blocked(self):
        self.t['completed']=False
        with self.assertRaises(ReplayError):self.call()
    def test_zero_geometry_blocked(self):
        self.t['entries'][0]['grid']=[0,1,1]
        with self.assertRaises(ReplayError):self.call()
    def test_large_group_blocked(self):
        self.t['entries'][0]['group']=[1024,2,1]
        with self.assertRaises(ReplayError):self.call()
    def test_helper_counted(self):
        self.t['entries'].append(dict(self.t['entries'][0],phase='diagnostic-support',node=''))
        self.assertEqual(self.call()['support_identities'],[self.identity])
    def test_unapproved_helper_blocked(self):
        self.t['entries'].append(dict(self.t['entries'][0],identity='c'*64+':helper.kd',phase='diagnostic-support',node=''))
        with self.assertRaises(ReplayError):self.call()
    def test_collector_compiled(self):
        cxx=os.environ.get('CXX','c++');self.assertIsNotNone(shutil.which(cxx),'C++ compiler required, not skipped')
        root=Path(__file__).resolve().parents[1]
        src=self.root/'test.cpp';src.write_text(r'''
#include "replay/dispatch_trace.h"
#include <cassert>
#include <thread>
int main(){
    const std::string h(64,'b'),id=h+":foo.kd";
    RadReplayDispatchV1 e{1,h.c_str(),"foo.kd",0,1,{32,1,1},{32,1,1},0,0};
    using namespace rad_replay_trace;
    begin({id},3);context("k","diagnostic-node",0);
    assert(rad_replay_trace_enabled_v1()==1 && observe(&e)==0 && state().entries.size()==1);
    e.descriptor_symbol="unknown.kd";assert(observe(&e)==1);assert(!state().error.empty());
    e.descriptor_symbol="foo.kd";assert(observe(&e)==1); // sticky failure
    begin({id},1);assert(observe(&e)==0);assert(observe(&e)==1); // cap not truncation
    begin({id});std::thread t([&](){assert(observe(&e)==1);});t.join();
    begin({id});e.group[0]=2048;assert(observe(&e)==1);
    end();assert(observe(nullptr)==0 && rad_replay_trace_enabled_v1()==0);
}
''')
        exe=self.root/'test';subprocess.run([cxx,'-std=c++17','-pthread','-I'+str(root),str(src),'-o',str(exe)],check=True,capture_output=True,timeout=45)
        subprocess.run([str(exe)],check=True,capture_output=True,timeout=10)
