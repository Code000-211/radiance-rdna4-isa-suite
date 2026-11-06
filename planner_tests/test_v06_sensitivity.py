import unittest,copy
from radplan.common import PlanError
from radplan.sensitivity import analyze
from support_v06 import *

class SensitivityTests(unittest.TestCase):
    def inputs(self):
        c,w,s,g=simple();s2=copy.deepcopy(s);s2['id']='second';s2['devices'][0]['dram_bytes_per_us']=20
        return c,w,{'schema':'radiance-system-scenarios/1','systems':[s,s2]}
    def test_same_winner_not_certification(self):
        c,w,p=self.inputs();r=analyze(c,w,p);self.assertTrue(r['workloads'][0]['same_winner_in_all_scenarios']);self.assertFalse(r['deployment_approved']);self.assertFalse(r['confidence_interval'])
    def test_times_change_with_rates(self):
        c,w,p=self.inputs();r=analyze(c,w,p);a=r['workloads'][0]['candidates'][0];self.assertEqual(a['scenario_times_us'],[10,5])
    def test_single_profile_not_sensitivity(self):
        c,w,p=self.inputs();p['systems'].pop()
        with self.assertRaises(PlanError):analyze(c,w,p)
    def test_duplicate_scenario_names(self):
        c,w,p=self.inputs();p['systems'][1]['id']=p['systems'][0]['id']
        with self.assertRaises(PlanError):analyze(c,w,p)
    def test_no_cpu_gpu_profile_mix(self):
        c,w,p=self.inputs();p['systems'][1]['devices'][0]['arch']='host'
        with self.assertRaises(PlanError):analyze(c,w,p)
    def test_no_topology_mix(self):
        c,w,p=self.inputs();p['systems'][1]['links'].pop()
        with self.assertRaises(PlanError):analyze(c,w,p)
    def test_unsupported_panel_rejected(self):
        c,w,p=self.inputs();c['candidates'][0]['contract']='different'
        with self.assertRaises(PlanError):analyze(c,w,p)
    def test_worst_case_regret_explicit(self):
        c,w,p=self.inputs();slower=copy.deepcopy(c['candidates'][0]);slower['id']='twice';slower['params']['service_scale']=2;c['candidates'].append(slower)
        r=analyze(c,w,p);a=next(x for x in r['workloads'][0]['candidates'] if x['candidate_id']=='twice');self.assertEqual(a['max_relative_regret_within_profile_set'],1)
    def test_zero_noop(self):
        c,w,p=self.inputs()
        for x in w['workloads']:x['service']={}
        r=analyze(c,w,p);self.assertEqual(r['workloads'][0]['candidates'][0]['max_scenario_us'],0)
