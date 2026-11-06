import unittest,copy
from radplan.common import PlanError,digest
from radplan.graph import graph_plan
from radplan.observations import compare_graph,measurement_batch
from radplan.operators import op_plan
from support_v06 import *

class GraphTests(unittest.TestCase):
    def test_two_ranks_overlap(self):
        c,w,s,g=simple();r=graph_plan(c,w,s,g)
        self.assertEqual(r['selected']['makespan_us'],10);self.assertEqual(r['serialized_scenario_us'],20)
    def test_control_dependency_serializes(self):
        c,w,s,g=simple();g['nodes'][1]['deps']=[{'node':'n0','kind':'control'}]
        self.assertEqual(graph_plan(c,w,s,g)['selected']['makespan_us'],20)
    def test_queue_order_without_explicit_dependency(self):
        c,w,s,g=simple();g['nodes'][1]['queue']='q0';self.assertEqual(graph_plan(c,w,s,g)['selected']['makespan_us'],20)
    def test_same_device_shared_bandwidth(self):
        c,w,s,g=simple();w['workloads'][1]['device']='gpu0';self.assertEqual(graph_plan(c,w,s,g)['selected']['makespan_us'],20)
    def test_cross_gpu_data_edge_rejected(self):
        c,w,s,g=simple();g['nodes'][1]['deps']=[{'node':'n0','kind':'data'}]
        with self.assertRaisesRegex(PlanError,'cross-device'):graph_plan(c,w,s,g)
    def test_missing_layout_transition_rejected(self):
        c,w,s,g=simple();w['workloads'][1]['device']='gpu0';c['candidates'].append(copy.deepcopy(c['candidates'][0]));c['candidates'][1].update(id='b',input_layout='other')
        g['nodes'][1]['allowed_candidates']=['b'];g['nodes'][1]['deps']=[{'node':'n0','kind':'data'}]
        with self.assertRaisesRegex(PlanError,'no feasible'):graph_plan(c,w,s,g)
    def test_layout_transition_charged(self):
        c,w,s,g=simple();w['workloads'][1]['device']='gpu0';c['candidates'].append(copy.deepcopy(c['candidates'][0]));c['candidates'][1].update(id='b',input_layout='other')
        g['nodes'][1]['allowed_candidates']=['b'];g['nodes'][1]['deps']=[{'node':'n0','kind':'data'}]
        g['transitions']=[{'from':'x','to':'other','device':'gpu0','bytes':50,'fixed_us':0,'provenance':'assumed'}]
        self.assertEqual(graph_plan(c,w,s,g)['selected']['makespan_us'],30)
    def test_graph_cycle(self):
        c,w,s,g=simple();g['nodes'][0]['deps']=[{'node':'n1','kind':'control'}];g['nodes'][1]['deps']=[{'node':'n0','kind':'control'}]
        with self.assertRaisesRegex(PlanError,'cycle'):graph_plan(c,w,s,g)
    def test_node_not_in_catalog(self):
        c,w,s,g=simple();g['nodes'][0]['allowed_candidates']=['missing']
        with self.assertRaises(PlanError):graph_plan(c,w,s,g)
    def test_buffer_requires_dependency_path(self):
        c,w,s,g=simple();g['buffers']=[{'id':'b','device':'gpu0','bytes':100,'producer':'n0','consumers':['n1']}]
        with self.assertRaisesRegex(PlanError,'dependency path'):graph_plan(c,w,s,g)
    def test_memory_budget_rejects_plan(self):
        c,w,s,g=simple();g['resident_bytes']['gpu0']=2**40
        with self.assertRaisesRegex(PlanError,'no feasible'):graph_plan(c,w,s,g)
    def test_workspace_in_memory_budget(self):
        c,w,s,g=simple();w['workloads'][0]['workspace_bytes']=1024
        r=graph_plan(c,w,s,g);self.assertEqual(r['selected']['memory']['gpu0']['peak_bytes'],1024)
    def test_exhaustive_finds_best_cost_hypothesis(self):
        c,w,s,g=simple();new=copy.deepcopy(c['candidates'][0]);new['id']='twice';new['params']['service_scale']=2;c['candidates'].append(new)
        for n in g['nodes']:n['allowed_candidates']=['a','twice']
        r=graph_plan(c,w,s,g);self.assertEqual(r['candidate_combinations'],4);self.assertEqual(r['selected']['chosen'],{'n0':'a','n1':'a'});self.assertTrue(r['search_complete'])
    def test_beam_never_claims_complete_search(self):
        c,w,s,g=simple();new=copy.deepcopy(c['candidates'][0]);new['id']='twice';new['params']['service_scale']=2;c['candidates'].append(new)
        for n in g['nodes']:n['allowed_candidates']=['a','twice']
        g['search']['max_combinations']=1;r=graph_plan(c,w,s,g);self.assertFalse(r['search_complete']);self.assertFalse(r['global_hardware_optimum_proved'])
    def test_unsupported_candidate_no_silent_fallback(self):
        c,w,s,g=simple();c['candidates'][0]['contract']='not-x'
        with self.assertRaises(PlanError):graph_plan(c,w,s,g)
    def test_measurement_requests_are_not_execution(self):
        c,w,s,g=simple();r=measurement_batch(op_plan(c,w,s));self.assertFalse(r['gpu_executed']);self.assertTrue(all(x['state']=='NOT_EXECUTED' for x in r['requests']))
    def test_full_tp2_example_has_waiting_collective(self):
        r=graph_plan(catalog(),workloads(),system(),load('tp2-moe-graph.json'))
        self.assertEqual(r['candidate_combinations'],96);self.assertFalse(r['gpu_executed']);self.assertFalse(r['tp2_runtime_executed'])
        self.assertEqual(r['status'],'NEEDS_MEASUREMENT')
        ts=r['selected']['node_times'];last=max(ts[k][1] for k in ('rank0','rank1'))
        comm=next(v for k,v in ts.items() if 'reduce' in k or 'ar' in k)
        self.assertGreaterEqual(comm[0]+1e-8,last)

class ObservationTests(unittest.TestCase):
    def setUp(self):
        c,w,self.s,g=simple();self.r=graph_plan(c,w,self.s,g)
        self.obs={'schema':'radiance-graph-observations/1','graph_plan_sha256':digest(self.r),'environment_sha256':digest(self.s['environment']),'backend':'aql','devices':[{'id':d['id'],'arch':d['arch']} for d in self.s['devices']],'samples_us':[10,20,30],'correctness':'PASS','instrumented':False,'evidence_sha256':'a'*64}
    def test_comparison_only_mock_protocol(self):
        r=compare_graph(self.r,self.obs,self.s);self.assertEqual(r['observed_median_us'],20);self.assertEqual(r['absolute_relative_error'],.5);self.assertFalse(r['deployment_approved'])
    def test_cpu_cannot_calibrate_gpu(self):
        self.obs['backend']='host'
        with self.assertRaises(PlanError):compare_graph(self.r,self.obs,self.s)
    def test_other_plan_rejected(self):
        self.obs['graph_plan_sha256']='b'*64
        with self.assertRaises(PlanError):compare_graph(self.r,self.obs,self.s)
    def test_changed_clock_profile_rejected(self):
        self.s['environment']['clock']='changed'
        with self.assertRaises(PlanError):compare_graph(self.r,self.obs,self.s)
    def test_instrumented_rejected(self):
        self.obs['instrumented']=True
        with self.assertRaises(PlanError):compare_graph(self.r,self.obs,self.s)
    def test_failed_correctness_rejected(self):
        self.obs['correctness']='FAIL'
        with self.assertRaises(PlanError):compare_graph(self.r,self.obs,self.s)
    def test_nan_sample_rejected(self):
        self.obs['samples_us'][0]=float('nan')
        with self.assertRaises(PlanError):compare_graph(self.r,self.obs,self.s)
    def test_insufficient_samples_rejected(self):
        self.obs['samples_us']=[1]
        with self.assertRaises(PlanError):compare_graph(self.r,self.obs,self.s)
