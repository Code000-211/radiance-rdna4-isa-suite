import unittest,tempfile,json,contextlib,io
from pathlib import Path
from radplan.cli import main
from support_v06 import simple,ROOT

class ExtendedCliTests(unittest.TestCase):
    def run_case(self,command,*,calibrated=False,bad=False):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cat,w,s,g=simple()
            if bad:g['nodes'][0]['allowed_candidates']=['missing']
            for name,obj in [('catalog',cat),('workloads',w),('system',s),('graph',g)]: (p/(name+'.json')).write_text(json.dumps(obj))
            args=[command,'--catalog',str(p/'catalog.json'),'--workloads',str(p/'workloads.json'),'--system',str(p/'system.json'),'--out',str(p/'out')]
            if command=='plan-graph':args+=['--graph',str(p/'graph.json')]
            if calibrated:args+=['--require-calibrated']
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):rc=main(args)
            files=list((p/'out').glob('*.json')) if (p/'out').exists() else []
            return rc,[json.loads(x.read_text()) for x in files]
    def test_operator_command(self):rc,reports=self.run_case('plan-operators');self.assertEqual(rc,0);self.assertTrue(reports)
    def test_graph_command(self):rc,reports=self.run_case('plan-graph');self.assertEqual(rc,0);self.assertEqual(reports[0]['status'],'NEEDS_MEASUREMENT')
    def test_operator_strict_calibration(self):self.assertEqual(self.run_case('plan-operators',calibrated=True)[0],2)
    def test_graph_strict_calibration(self):self.assertEqual(self.run_case('plan-graph',calibrated=True)[0],2)
    def test_bad_graph_nonzero(self):self.assertEqual(self.run_case('plan-graph',bad=True)[0],2)
    def test_fresh_output_required(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cat,w,s,g=simple()
            for n,o in [('c',cat),('w',w),('s',s)]: (p/(n+'.json')).write_text(json.dumps(o))
            args=['plan-operators','--catalog',str(p/'c.json'),'--workloads',str(p/'w.json'),'--system',str(p/'s.json'),'--out',str(p)]
            with contextlib.redirect_stderr(io.StringIO()):rc=main(args)
            self.assertEqual(rc,2)
