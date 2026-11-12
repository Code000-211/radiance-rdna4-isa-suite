"""SARIF 2.1.0 and GitHub workflow-command renderings of `radisa check` reports (radisa/formats.py).

The SARIF checks are structural (the required properties and value domains of the OASIS SARIF 2.1.0
schema that the output uses); no schema download, no network."""
import copy
import json
from pathlib import Path
import re
import unittest

from radisa import check, formats

SARIF_LEVELS = {'none', 'note', 'warning', 'error'}


def finding(rule, severity='error', line=None, pc=0x10, file='src/k.hip', message='m', **kw):
    f = {'rule': rule, 'severity': severity, 'pc': pc, 'message': message, 'source': f'cite {rule}', **kw}
    if line is not None:
        f['loc'] = {'file': file, 'line': line, 'column': 3, 'inline_asm': False}
    return f


def report(inp='src/k.hip', kernel_findings=(), findings=(), status='FAIL', name='kern'):
    r = {'schema': 'radisa-check-v1', 'input': inp, 'status': status, 'exit_code': 1, 'isa_table': None,
         'decoder': None, 'counts': {}, 'operand_model': {'known': 1, 'unknown': 0}, 'incremental': None,
         'findings': list(findings), 'not_checked': [],
         'kernels': [{'name': name, 'symbol': name + '.kd', 'object': inp, 'instructions': [1],
                      'operand_model': {'operand_model_known': 1, 'operand_model_unknown': 0},
                      'findings': list(kernel_findings)}]}
    return r


def validate_sarif(test: unittest.TestCase, log: dict) -> None:
    """Required properties / domains of SARIF 2.1.0 used by radisa (sarif-schema-2.1.0.json)."""
    test.assertEqual(log['version'], '2.1.0')
    test.assertIsInstance(log['runs'], list)
    json.dumps(log)  # serialisable
    for run in log['runs']:
        drv = run['tool']['driver']
        test.assertIsInstance(drv['name'], str)
        ids = []
        for rule in drv.get('rules', []):
            test.assertIsInstance(rule['id'], str)
            ids.append(rule['id'])
            for k in ('shortDescription', 'fullDescription'):
                if k in rule:
                    test.assertIsInstance(rule[k]['text'], str)
                    test.assertTrue(rule[k]['text'])
            test.assertNotIn('helpUri', rule)
            if 'defaultConfiguration' in rule:
                test.assertIn(rule['defaultConfiguration']['level'], SARIF_LEVELS)
        test.assertEqual(len(ids), len(set(ids)))
        arts = run.get('artifacts', [])
        for a in arts:
            test.assertIsInstance(a['location']['uri'], str)
        for inv in run.get('invocations', []):
            test.assertIsInstance(inv['executionSuccessful'], bool)
        for res in run['results']:
            test.assertIsInstance(res['message']['text'], str)
            test.assertIn(res['level'], SARIF_LEVELS)
            if 'ruleIndex' in res:
                test.assertEqual(drv['rules'][res['ruleIndex']]['id'], res['ruleId'])
            for loc in res.get('locations', []):
                pl = loc.get('physicalLocation')
                if pl:
                    al = pl['artifactLocation']
                    test.assertIsInstance(al['uri'], str)
                    if 'index' in al:
                        test.assertEqual(arts[al['index']]['location']['uri'], al['uri'])
                    reg = pl.get('region')
                    if reg:
                        test.assertGreaterEqual(reg['startLine'], 1)
                        test.assertGreaterEqual(reg.get('startColumn', 1), 1)
                for ll in loc.get('logicalLocations', []):
                    test.assertIsInstance(ll['name'], str)
            for k, v in res.get('partialFingerprints', {}).items():
                test.assertIsInstance(k, str)
                test.assertIsInstance(v, str)


class SarifTest(unittest.TestCase):
    def test_structure_levels_rules_locations(self):
        rs = [report(kernel_findings=[finding('MISSING_WAIT', line=14, instruction='v_add_f32 v1, v1, v1'),
                                      finding('SGPR_PAIR_GUARD', 'review', pc=0x20),
                                      finding('CFG_FALLTHROUGH', 'info', line=20)],
                     findings=[finding('HIP_NO_LAUNCH_BOUNDS', 'info', line=3, pc=None)]),
              report('b.s', kernel_findings=[finding('DECODE', 'incomplete', pc=4)], status='INCOMPLETE',
                     name='<snippet>')]
        log = formats.to_sarif(rs, base=Path.cwd())
        validate_sarif(self, log)
        run = log['runs'][0]
        self.assertEqual([r['id'] for r in run['tool']['driver']['rules']],
                         sorted(['MISSING_WAIT', 'SGPR_PAIR_GUARD', 'CFG_FALLTHROUGH', 'HIP_NO_LAUNCH_BOUNDS', 'DECODE']))
        mw = next(r for r in run['tool']['driver']['rules'] if r['id'] == 'MISSING_WAIT')
        self.assertEqual(mw['fullDescription']['text'], 'cite MISSING_WAIT')
        lv = {r['ruleId']: r['level'] for r in run['results']}
        self.assertEqual(lv, {'MISSING_WAIT': 'error', 'SGPR_PAIR_GUARD': 'warning', 'CFG_FALLTHROUGH': 'note',
                              'HIP_NO_LAUNCH_BOUNDS': 'note', 'DECODE': 'note'})
        res = {r['ruleId']: r for r in run['results']}
        pl = res['MISSING_WAIT']['locations'][0]['physicalLocation']
        self.assertEqual(pl['artifactLocation']['uri'], 'src/k.hip')
        self.assertEqual(pl['region']['startLine'], 14)
        self.assertEqual(pl['region']['startColumn'], 3)
        self.assertEqual(res['MISSING_WAIT']['locations'][0]['logicalLocations'][0]['name'], 'kern')
        # No source line: logical location kernel + pc, artifact = the input without a region.
        sg = res['SGPR_PAIR_GUARD']['locations'][0]
        self.assertEqual(sg['logicalLocations'][0]['fullyQualifiedName'], 'kern+0x20')
        self.assertNotIn('region', sg['physicalLocation'])
        self.assertEqual(res['DECODE']['locations'][0]['logicalLocations'][0]['fullyQualifiedName'], '<snippet>+0x4')
        self.assertFalse(run['invocations'][0]['executionSuccessful'])  # one INCOMPLETE file

    def test_fingerprints_stable_across_line_moves(self):
        def one(line, pc, other_line):
            r = report(kernel_findings=[finding('MISSING_WAIT', line=line, pc=pc, instruction='v_add_f32 v1, v1, v1',
                                                message=f'reads v1 while global_load_b32 (k.hip:{other_line}) may be '
                                                        f'outstanding at {pc:#x}')])
            return formats.to_sarif([r])['runs'][0]['results'][0]['partialFingerprints']
        self.assertEqual(one(14, 0x10, 12), one(40, 0x80, 38))
        r = report(kernel_findings=[finding('MISSING_WAIT', line=14, instruction='v_add_f32 v2, v2, v2')])
        self.assertNotEqual(one(14, 0x10, 12), formats.to_sarif([r])['runs'][0]['results'][0]['partialFingerprints'])

    def test_identical_findings_get_distinct_fingerprints(self):
        f = finding('MISSING_WAIT', line=5)
        r = report(kernel_findings=[f, copy.deepcopy(f)])
        fps = [x['partialFingerprints'][formats.FINGERPRINT_KEY] for x in formats.to_sarif([r])['runs'][0]['results']]
        self.assertEqual(len(set(fps)), 2)

    def test_same_basename_in_different_directories(self):
        """Regression: v1 keyed the input by its basename, so a/k.s and c/k.s collided."""
        f = finding('MISSING_WAIT', line=5, instruction='v_add_f32 v1, v1, v1')
        log = formats.to_sarif([report('a/k.s', kernel_findings=[f]), report('c/k.s', kernel_findings=[copy.deepcopy(f)])],
                               base=Path.cwd())
        fps = [x['partialFingerprints'][formats.FINGERPRINT_KEY] for x in log['runs'][0]['results']]
        self.assertEqual(len(set(fps)), 2)
        self.assertEqual(formats.FINGERPRINT_KEY, 'radisaFinding/v2')

    def test_sets_and_paths_serialise(self):
        r = report(kernel_findings=[finding('MISSING_WAIT', registers={('v', 1), ('v', 2)}, producer=Path('x'))])
        validate_sarif(self, json.loads(json.dumps(formats.to_sarif([r]))))

    def test_empty(self):
        log = formats.to_sarif([report(status='PASS')])
        validate_sarif(self, log)
        self.assertEqual(log['runs'][0]['results'], [])
        self.assertTrue(log['runs'][0]['invocations'][0]['executionSuccessful'])

    def test_absolute_path_outside_base_is_file_uri(self):
        r = report(kernel_findings=[finding('MISSING_WAIT', line=2, file='/elsewhere/x.hip')])
        log = formats.to_sarif([r], base='/home/nobody')
        uri = log['runs'][0]['results'][0]['locations'][0]['physicalLocation']['artifactLocation']['uri']
        self.assertEqual(uri, 'file:///elsewhere/x.hip')


class GithubTest(unittest.TestCase):
    def test_commands_and_escaping(self):
        r = report(kernel_findings=[finding('MISSING_WAIT', line=7, file='a,b:c.hip', message='50% done\nnext\r',
                                            instruction='v_mov_b32 v0, v1'),
                                    finding('SGPR_PAIR_GUARD', 'review'), finding('CFG_FALLTHROUGH', 'info')])
        out = formats.to_github([r]).splitlines()
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0].startswith('::error file=a%2Cb%3Ac.hip,line=7,col=3,title=MISSING_WAIT::'))
        self.assertIn('50%25 done%0Anext%0D', out[0])
        self.assertIn('%0A    v_mov_b32 v0, v1', out[0])
        self.assertNotIn('\n', ''.join(out))
        self.assertTrue(out[1].startswith('::warning file=src/k.hip,title=SGPR_PAIR_GUARD::[kern+0x10] '))
        self.assertTrue(out[2].startswith('::notice '))
        for line in out:
            self.assertRegex(line, r'^::(error|warning|notice) [^:]*::')

    def test_empty(self):
        self.assertEqual(formats.to_github([report(status='PASS')]), '')

    def test_file_outside_base_is_a_path(self):
        """Regression: a file outside `base` was written as file%3A///... (a URI, not a path)."""
        r = report('/elsewhere/x.hip', kernel_findings=[finding('MISSING_WAIT', line=2, file='/elsewhere/x.hip'),
                                                        finding('SGPR_PAIR_GUARD', 'review')])
        out = formats.to_github([r], base='/home/nobody').splitlines()
        self.assertTrue(out[0].startswith('::error file=/elsewhere/x.hip,line=2,'), out[0])
        self.assertTrue(out[1].startswith('::warning file=/elsewhere/x.hip,title='), out[1])
        self.assertNotIn('file%3A', ''.join(out))


class EmitTest(unittest.TestCase):
    def test_all_formats(self):
        r = report(kernel_findings=[finding('MISSING_WAIT', line=7)])
        check.finalize(r, False)
        self.assertIn('[MISSING_WAIT]', formats.emit([r], 'text'))
        self.assertIn('-> FAIL', formats.emit([r], 'text'))
        self.assertEqual(json.loads(formats.emit([r], 'json'))['status'], 'FAIL')
        self.assertIsInstance(json.loads(formats.emit([r, r], 'json')), list)
        # one report: exactly check.render; several: each block names its file (regression)
        self.assertEqual(formats.emit([r], 'text'), check.render(r, False) + '\n')
        two = formats.emit([r, report('b.s', status='PASS')], 'text')
        self.assertIn('==> src/k.hip: FAIL <==\n', two)
        self.assertIn('==> b.s: PASS <==\n', two)
        validate_sarif(self, json.loads(formats.emit([r], 'sarif')))
        self.assertTrue(formats.emit([r], 'github').startswith('::error '))
        with self.assertRaises(ValueError):
            formats.emit([r], 'xml')


if __name__ == '__main__':
    unittest.main()
