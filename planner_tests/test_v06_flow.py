import unittest,random,copy,math
from radplan.common import PlanError
from radplan.flow import phase,simulate,system_check,resources,memory_peak
from support_v06 import system

class FlowTests(unittest.TestCase):
    def test_simple_service(self):self.assertEqual(simulate([phase('x',[],{'r':100})],{'r':10},{})['makespan_us'],10)
    def test_sequential_dependency(self):
        ps=[phase('a',[],{'r':100}),phase('b',['a'],{'r':100})]
        self.assertEqual(simulate(ps,{'r':10},{})['makespan_us'],20)
    def test_independent_resources_overlap(self):
        ps=[phase('a',[],{'a':100}),phase('b',[],{'b':100})]
        self.assertEqual(simulate(ps,{'a':10,'b':10},{})['makespan_us'],10)
    def test_shared_bandwidth_not_doubled(self):
        ps=[phase('a',[],{'r':100}),phase('b',[],{'r':100})]
        self.assertEqual(simulate(ps,{'r':10},{})['makespan_us'],20)
    def test_finish_releases_bandwidth(self):
        ps=[phase('a',[],{'r':100}),phase('b',[],{'r':10})]
        self.assertAlmostEqual(simulate(ps,{'r':10},{})['makespan_us'],11)
    def test_service_units_conserved(self):
        ps=[phase('a',[],{'r':100}),phase('b',[],{'r':10})]
        r=simulate(ps,{'r':10},{},retain_segments=True)
        self.assertAlmostEqual(sum((x['end_us']-x['start_us'])*x['rates']['r'] for x in r['segments']),110)
    def test_slot_limit(self):
        ps=[phase(i,[],{'r'+i:100},reservations={'slot':1}) for i in ['a','b']]
        self.assertEqual(simulate(ps,{'ra':10,'rb':10},{'slot':1})['makespan_us'],20)
    def test_two_slots(self):
        ps=[phase(i,[],{'r'+i:100},reservations={'slot':1}) for i in ['a','b']]
        self.assertEqual(simulate(ps,{'ra':10,'rb':10},{'slot':2})['makespan_us'],10)
    def test_zero_phase(self):
        self.assertEqual(simulate([phase('a',[],{}),phase('b',['a'],{},fixed_us=3)],{}, {})['makespan_us'],3)
    def test_fixed_latency_not_added_to_parallel_service(self):
        self.assertEqual(simulate([phase('a',[],{'r':100},fixed_us=3)],{'r':10},{})['makespan_us'],10)
    def test_serial_baseline(self):
        ps=[phase('a',[],{'a':100}),phase('b',[],{'b':100})]
        self.assertEqual(simulate(ps,{'a':10,'b':10},{},mode='serial')['makespan_us'],20)
    def test_cycle(self):
        with self.assertRaisesRegex(PlanError,'cycle'):simulate([phase('a',['b'],{}),phase('b',['a'],{})],{}, {})
    def test_missing_dependency(self):
        with self.assertRaises(PlanError):simulate([phase('a',['missing'],{})],{}, {})
    def test_duplicate_id(self):
        with self.assertRaises(PlanError):simulate([phase('a',[],{}),phase('a',[],{})],{}, {})
    def test_missing_resource(self):
        with self.assertRaises(PlanError):simulate([phase('a',[],{'missing':1})],{}, {})
    def test_oversized_reservation(self):
        with self.assertRaises(PlanError):simulate([phase('a',[],{},reservations={'s':2})],{}, {'s':1})
    def test_nan_rate(self):
        with self.assertRaises(PlanError):simulate([phase('a',[],{'r':1})],{'r':float('nan')},{})
    def test_resource_floor(self):
        for seed in range(20):
            rng=random.Random(seed);ps=[phase(str(i),[str(i-1)] if i%3==0 and i else [],{'r':rng.randrange(1,100),'s':rng.randrange(1,100)}) for i in range(10)]
            r=simulate(ps,{'r':5,'s':7},{},retain_segments=True)
            self.assertGreaterEqual(r['makespan_us']+1e-8,r['model_floor_us'])
            for seg in r['segments']:
                for key,v in seg['rates'].items():self.assertLessEqual(v,{'r':5,'s':7}[key]+1e-8)
    def test_reproducible_order(self):
        ps=[phase(i,[],{'r':100},reservations={'s':1}) for i in ('a','b')]
        self.assertEqual(simulate(ps,{'r':10},{'s':1}),simulate(ps,{'r':10},{'s':1}))
    def test_system_rejects_same_endpoint(self):
        s=system();s['links'][0]['target']=s['links'][0]['source']
        with self.assertRaises(PlanError):system_check(s)
    def test_system_shared_root_required(self):
        s=system();s['shared_resources']={}
        with self.assertRaises(PlanError):system_check(s)
    def test_directional_paths_are_distinct(self):
        s=system();rs,_=resources(s);self.assertEqual(len([x for x in rs if x.startswith('link:')]),4)
    def test_memory_lifetime_peak(self):
        ds={'d':{'memory_bytes':250}};ts={'a':(0,1),'b':(1,2)}
        bs=[{'id':'x','device':'d','bytes':100,'producer':'a','consumers':['b']}]
        r=memory_peak(bs,ts,ds,resident={'d':50});self.assertEqual(r['d']['peak_bytes'],150);self.assertTrue(r['d']['fits'])
    def test_memory_oom_not_spilled(self):
        ds={'d':{'memory_bytes':99}};bs=[{'id':'x','device':'d','bytes':100,'producer':None,'consumers':['a']}]
        self.assertFalse(memory_peak(bs,{'a':(0,1)},ds)['d']['fits'])
    def test_memory_missing_dependency(self):
        b={'id':'x','device':'d','bytes':10,'producer':'a','consumers':['b']}
        with self.assertRaises(PlanError):memory_peak([b],{'a':(0,2),'b':(1,2)},{'d':{'memory_bytes':100}})
    def test_memory_alias_declared_once(self):
        b={'id':'x','device':'d','bytes':10,'producer':None,'consumers':['a']}
        with self.assertRaises(PlanError):memory_peak([b,b],{'a':(0,1)},{'d':{'memory_bytes':100}})
