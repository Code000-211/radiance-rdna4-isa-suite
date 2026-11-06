import copy
import unittest
from radplan.common import PlanError,digest
from radplan.calibrate import fit,dataset_check,calibrated_prediction,nnls,split_groups
from radplan.model import features,FEATURES
from support_planner import *

class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.cat,cls.data=synthetic_data();cls.cal=fit(cls.data,cls.cat,cls.p,seed=3,ncal=10,ntest=10)
    def data_copy(self):return copy.deepcopy(self.data)
    def test_disjoint_shape_sets(self):
        a,b,c=map(set,[self.cal['split'][k] for k in ('train','calibration','test')]);self.assertFalse(a&b or a&c or b&c);self.assertEqual(len(a|b|c),40)
    def test_split_deterministic(self):self.assertEqual(split_groups(list('abcdefghijklmnopqrstu'),seed=1,ncal=3,ntest=3),split_groups(list('abcdefghijklmnopqrstu'),seed=1,ncal=3,ntest=3))
    def test_split_too_small(self):
        with self.assertRaises(PlanError):split_groups(['a','b'],seed=1,ncal=1,ntest=1)
    def test_keeps_test_errors(self):self.assertEqual(len(self.cal['held_out_predictions']),20)
    def test_surrogate_accuracy_synthetic(self):self.assertLess(self.cal['validation']['median_relative_latency_error'],.01)
    def test_nonnegative_coefficients(self):self.assertTrue(all(v>=0 for m in self.cal['models'].values() for v in m['coefficients']))
    def test_small_calibration_unbounded(self):
        r=fit(self.data,self.cat,self.p,ncal=2,ntest=10);self.assertIsNone(r['envelope']['log_radius'])
    def test_all_candidate_errors_enter_envelope(self):self.assertEqual(len(self.cal['envelope']['group_log_error_scores']),10)
    def test_no_gpu_calibration_from_cpu(self):self.assertFalse(self.cal['gpu_calibration'])
    def test_profile_changed_rejected(self):
        p=copy.deepcopy(self.p);p['id']='changed'
        with self.assertRaises(PlanError):dataset_check(self.data,self.cat,p)
    def test_backend_contamination(self):
        d=self.data_copy();d['records'][0]['backend']='aql'
        with self.assertRaisesRegex(PlanError,'contamination'):dataset_check(d,self.cat,self.p)
    def test_environment_changed(self):
        d=self.data_copy();d['records'][0]['environment_sha256']='f'*64
        with self.assertRaises(PlanError):dataset_check(d,self.cat,self.p)
    def test_binary_changed(self):
        d=self.data_copy();d['records'][0]['evidence']['plugin_sha256']='f'*64
        with self.assertRaises(PlanError):dataset_check(d,self.cat,self.p)
    def test_candidate_changed(self):
        cat=copy.deepcopy(self.cat);cat['candidates'][0]['tile']=[2,4,1]
        with self.assertRaises(PlanError):dataset_check(self.data,cat,self.p)
    def test_duplicate_shape_candidate(self):
        d=self.data_copy();d['records'].append(copy.deepcopy(d['records'][0]))
        with self.assertRaises(PlanError):dataset_check(d,self.cat,self.p)
    def test_failed_correctness_rejected(self):
        d=self.data_copy();d['records'][0]['correctness_status']='FAIL'
        with self.assertRaises(PlanError):dataset_check(d,self.cat,self.p)
    def test_missing_rival_rejected(self):
        d=self.data_copy();d['records'].pop()
        with self.assertRaisesRegex(PlanError,'panel'):fit(d,self.cat,self.p)
    def test_zero_sample_rejected(self):
        d=self.data_copy();d['records'][0]['samples_us']=[1,0,1]
        with self.assertRaises(PlanError):dataset_check(d,self.cat,self.p)
    def test_metric_mismatch(self):
        d=self.data_copy();d['records'][0]['metric']='individual_dispatch_us'
        with self.assertRaises(PlanError):dataset_check(d,self.cat,self.p)
    def test_unverified_origin_rejected(self):
        d=self.data_copy();d['records'][0]['evidence']['origin']='theoretical'
        with self.assertRaises(PlanError):dataset_check(d,self.cat,self.p)
    def test_empty_dataset(self):
        d=self.data_copy();d['records']=[]
        with self.assertRaises(PlanError):dataset_check(d,self.cat,self.p)
    def test_extrapolation_requests_measurement(self):
        w=cpu_work(10000);c=self.cat['candidates'][0];f=features(w,c,self.p)
        r=calibrated_prediction(self.cal,w,c,self.p,f);self.assertIsNone(r['interval_us']);self.assertTrue(r['reasons'])
    def test_new_candidate_not_blessed(self):
        c=copy.deepcopy(self.cat['candidates'][0]);c['tile']=[1,16,1];w=cpu_work()
        r=calibrated_prediction(self.cal,w,c,self.p,features(w,c,self.p));self.assertEqual(r['status'],'NEW_OR_CHANGED_CANDIDATE')
    def test_coefficient_order_versioned(self):
        cal=copy.deepcopy(self.cal);cal['feature_order'].reverse();w=cpu_work();c=self.cat['candidates'][0]
        with self.assertRaises(PlanError):calibrated_prediction(cal,w,c,self.p,features(w,c,self.p))
    def test_alpha_out_of_range(self):
        with self.assertRaises(PlanError):fit(self.data,self.cat,self.p,alpha=1)
    def test_no_auto_deploy(self):self.assertFalse(self.cal['deployment_approved']);self.assertFalse(self.cal['envelope']['production_coverage_guaranteed'])
    def test_nnls_not_enough_rows(self):
        with self.assertRaises(PlanError):nnls([[1]*len(FEATURES)],[1])
