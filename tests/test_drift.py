"""radisa drift: extractors, normalisation, drift / missing-anchor detection, --update, shipped pins.

All source trees are synthetic (written to a temporary directory); nothing depends on an LLVM
checkout. Set RADISA_LLVM_SRC to an llvm-project checkout to also compare the shipped pins
against it (test_shipped_pins_match_llvm_src)."""
import argparse
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import re
import tempfile
import unittest

from radisa import drift
from support import ROOT

HEADER = r'''
// GCNSubtarget.h stand-in
class Sub {
  bool hasFoo() const { return Gen >= 11; }
  bool hasBar() const {
    // a brace in a comment: {
    return Gen == 12; /* and } here */
  }
  bool hasBaz() const;   // declaration only
  const char *name() const { return "}{"; }
  char brace() const { return '{'; }
};
'''
CPP = r'''
#include "Sub.h"
static const char *Raw = R"x(not a } brace)x";
bool Rec::fixThing(MachineInstr *MI) {
  if (!ST.hasFoo()) return false;
  auto L = [&](int X) { return X > 1; };
  return L(2);
}
void Rec::fixAll(MachineInstr *MI) {
  fixThing(MI);
  other.fixThing(MI);
}
auto Rec::trailing(int A) const -> int { return A; }
void Rec::run() {
  if (!Opt && Dyn) {
    emit(1);
  } else if (Gen >= 11) {
    if (Flag) { emit(2); }
  } else {
    emit(3);
  }
  after();
}
'''
TD = r'''
def FeatureX12 : FeatureSet<
  [FeatureA, // trailing comment
   FeatureB]>;
def FeatureX12_50 : FeatureSet<[FeatureA]>;
def ProcY : Model<"y", [{ code { with braces }]>;
def WithBody : Base {
  let Field = 1;
}
'''
FACTS = [
    ('feat', 'features', 'F.td', 'FeatureX12', 'td_def', ['RULE_FEAT']),
    ('proc', 'features', 'F.td', 'ProcY', 'td_def', ['*']),
    ('body', 'features', 'F.td', 'WithBody', 'td_def', []),
    ('foo', 'pred', 'Sub.h', 'hasFoo', 'cxx_function', ['RULE_FOO']),
    ('bar', 'pred', 'Sub.h', 'hasBar', 'cxx_function', ['RULE_BAR']),
    ('thing', 'haz', 'Rec.cpp', 'Rec::fixThing', 'cxx_function', ['RULE_THING']),
    ('all', 'haz', 'Rec.cpp', 'Rec::fixAll', 'cxx_function', ['RULE_ALL']),
    ('trail', 'haz', 'Rec.cpp', 'Rec::trailing', 'cxx_function', []),
    ('block', 'wait', 'Rec.cpp', 'if (!Opt&&Dyn)', 'statement', ['RULE_BLOCK']),
    ('whole', 'wait', 'Sub.h', 'Sub.h', 'file', ['RULE_WHOLE']),
]


def pins_for(facts=FACTS) -> dict:
    return {'schema': drift.SCHEMA, 'provenance': {'revision': 'r0'}, 'facts': [
        {'id': i, 'family': fam, 'file': f, 'anchor': a, 'extractor': e, 'sha256': '', 'rules': r, 'note': ''}
        for i, fam, f, a, e, r in facts]}


class Tree:
    """A synthetic llvm-project checkout: <tmp>/llvm/lib/Target/AMDGPU/{Sub.h,Rec.cpp,F.td}."""
    def __init__(self, d: str):
        self.top = Path(d)
        self.amdgpu = self.top / drift.SUBDIR
        self.amdgpu.mkdir(parents=True)
        for name, text in (('Sub.h', HEADER), ('Rec.cpp', CPP), ('F.td', TD)):
            (self.amdgpu / name).write_text(text)

    def edit(self, name: str, old: str, new: str):
        p = self.amdgpu / name
        text = p.read_text()
        assert old in text, old
        p.write_text(text.replace(old, new, 1))


def cli(*argv) -> tuple[int, str]:
    p = argparse.ArgumentParser()
    drift.add_parser(p.add_subparsers(dest='command'))
    args = p.parse_args(['drift', *map(str, argv)])
    out = io.StringIO()
    with redirect_stdout(out):
        code = args.fn(args)
    return code, out.getvalue()


class Extractors(unittest.TestCase):
    def tok(self, text, **kw):
        return [t for t, _ in drift.tokenize(text, **kw)]

    def test_literals_and_comments_hide_braces(self):
        self.assertEqual(self.tok('a /* { */ "}" \'{\' // }\n b'), ['a', '"}"', "'{'", 'b'])
        self.assertEqual(self.tok('R"x(})x" c'), ['R"x(})x"', 'c'])
        self.assertEqual(self.tok("1'000 x"), ["1'000", 'x'])

    def test_unterminated_comment_or_literal(self):
        for text in ('a /* never closed', 'a "never closed\n'):
            with self.assertRaises(drift.AnchorError):
                drift.tokenize(text)

    def test_tablegen_code_blocks(self):
        # like TableGen's lexer, the first `}]` ends the block; braces inside it never count
        self.assertEqual(self.tok('[{ x  { y; }\n // c\n }] z', tablegen=True), ['[{x{y;}}]', 'z'])
        self.assertEqual(self.tok('[{ x }]'), ['[', '{', 'x', '}', ']'])

    def test_normalize_ignores_layout_only(self):
        a = drift.normalize(drift.tokenize('bool f() const {\n  return a  &&  b; // c\n}'))
        b = drift.normalize(drift.tokenize('bool f() const { return a&&b; }'))
        self.assertEqual(a, b)
        self.assertEqual(a, 'bool f()const{return a&&b;}')
        self.assertNotEqual(a, drift.normalize(drift.tokenize('bool f() const { return a||b; }')))

    def test_function_definition_not_call_or_declaration(self):
        toks = drift.tokenize(CPP)
        a, b = drift.extract_cxx_function(toks, 'Rec::fixThing')
        text = drift.normalize(toks[a:b + 1])
        self.assertTrue(text.startswith('bool Rec::fixThing(MachineInstr*MI){'), text)  # return type kept
        self.assertTrue(text.endswith('return L(2);}'), text)       # lambda braces matched
        a, b = drift.extract_cxx_function(toks, 'fixThing')          # unqualified: still the one definition
        self.assertIn('return L(2);}', drift.normalize(toks[a:b + 1]))
        a, b = drift.extract_cxx_function(toks, 'Rec::trailing')
        self.assertEqual(drift.normalize(toks[a:b + 1]), 'auto Rec::trailing(int A)const->int{return A;}')
        with self.assertRaisesRegex(drift.AnchorError, 'no definition'):
            drift.extract_cxx_function(drift.tokenize(HEADER), 'hasBaz')

    def test_td_def_exact_name_and_body(self):
        toks = drift.tokenize(TD, tablegen=True)
        a, b = drift.extract_td_def(toks, 'FeatureX12')
        self.assertEqual(drift.normalize(toks[a:b + 1]), 'def FeatureX12:FeatureSet<[FeatureA,FeatureB]>;')
        a, b = drift.extract_td_def(toks, 'ProcY')
        self.assertTrue(drift.normalize(toks[a:b + 1]).endswith('}]>;'))
        a, b = drift.extract_td_def(toks, 'WithBody')
        self.assertEqual(drift.normalize(toks[a:b + 1]), 'def WithBody:Base{let Field=1;}')

    def test_statement_includes_else_chain(self):
        toks = drift.tokenize(CPP)
        a, b = drift.extract_statement(toks, 'if (!Opt && Dyn)')
        text = drift.normalize(toks[a:b + 1])
        self.assertTrue(text.endswith('else{emit(3);}'), text)
        self.assertNotIn('after', text)

    def stmt(self, src, anchor):
        toks = drift.tokenize(src)
        a, b = drift.extract_statement(toks, anchor)
        return drift.normalize(toks[a:b + 1])

    def test_statement_unbraced_do_while_lambda(self):
        """Review issue 4: the statement extractor stopped at the first ';' or '}'."""
        self.assertEqual(self.stmt('void g(){ if (A) x(); else y(); z(); }', 'if (A)'), 'if(A)x();else y();')
        self.assertEqual(self.stmt('void g(){ if (A) x(); else if (B) { y(); } else w(); z(); }', 'if (A)'),
                         'if(A)x();else if(B){y();}else w();')
        self.assertEqual(self.stmt('void g(){ do { x(); } while (A); z(); }', 'do'), 'do{x();}while(A);')
        self.assertEqual(self.stmt('void g(){ if (c) v = [&]{ return 1; }(); z(); }', 'if (c)'),
                         'if(c)v=[&]{return 1;}();')
        self.assertEqual(self.stmt('void g(){ for (int i = 0; i < 3; ++i) if (i) x(); z(); }', 'for'),
                         'for(int i=0;i<3;++i)if(i)x();')
        self.assertEqual(self.stmt('void g(){ while (A) { x(); } z(); }', 'while (A)'), 'while(A){x();}')
        with self.assertRaises(drift.AnchorError):
            self.stmt('void g(){ do { x(); } z(); }', 'do')

    def test_normalize_is_token_faithful(self):
        """Review issue 5b: multi-character operators are tokens; distinct token sequences never
        normalise to the same text."""
        n = lambda s: drift.normalize(drift.tokenize(s))
        self.assertEqual(self.tok('a->b::c && d-- <<= e'), ['a', '->', 'b', '::', 'c', '&&', 'd', '--', '<<=', 'e'])
        for a, b in (('a - -b', 'a--b'), ('a & &b', 'a&&b'), ('a > > b', 'a>>b'), ('x - > y', 'x->y'),
                     ('a . . . b', 'a...b'), ('a + +b', 'a++b'), ('a < = b', 'a<=b')):
            self.assertNotEqual(n(a), n(b), (a, b))
            self.assertEqual(self.tok(n(a)), self.tok(a))       # the text re-lexes to the same tokens
        self.assertEqual(n('x ->  y'), 'x->y')

    def test_function_text_includes_specifiers_and_return_type(self):
        """Review issue 5a: the return type and specifiers are part of the hashed definition."""
        def text(src, anchor='hasX'):
            toks = drift.tokenize(src)
            a, b = drift.extract_cxx_function(toks, anchor)
            return drift.normalize(toks[a:b + 1])
        self.assertEqual(text('bool hasX() const {return A;}'), 'bool hasX()const{return A;}')
        self.assertNotEqual(text('bool hasX() const {return A;}'), text('int hasX() const {return A;}'))
        self.assertNotEqual(text('static bool hasX() {return A;}'), text('bool hasX() {return A;}'))
        self.assertEqual(text('class C { public: [[nodiscard]] int hasX() const {return A;} };'),
                         '[[nodiscard]]int hasX()const{return A;}')
        self.assertEqual(text('#include "a"\ntemplate <typename T> std::vector<T> C::f(T a) { return {}; }', 'C::f'),
                         'template<typename T>std::vector<T>C::f(T a){return{};}')
        self.assertEqual(text('void g() { p->f(1); } int f(int a) { return a; }', 'f'), 'int f(int a){return a;}')

    def test_apostrophe_in_preprocessor_directive(self):
        """Review issue 5c: `#error don't` is one directive token, not an unterminated literal."""
        self.assertEqual(self.tok("#error don't\n  #  include \"x\" // c\nint y;"),
                         ["#error don't", '#include"x"', 'int', 'y', ';'])
        self.assertEqual(self.tok('#define F(a) \\\n  (a + 1)\nint y;'), ['#define F(a)(a+1)', 'int', 'y', ';'])
        self.assertEqual(self.tok('x # y', tablegen=True), ['x', '#', 'y'])    # TableGen paste operator
        with self.assertRaises(drift.AnchorError):
            drift.tokenize("int c = 'x;")

    def test_ambiguous_anchor(self):
        toks = drift.tokenize('int f() { return 1; }\nint f() { return 2; }')
        with self.assertRaisesRegex(drift.AnchorError, '2 definitions'):
            drift.extract_cxx_function(toks, 'f')

    def test_unbalanced(self):
        with self.assertRaises(drift.AnchorError):
            drift.extract_cxx_function(drift.tokenize('void f() { if (x) { }'), 'f')


class DriftCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tree = Tree(self.tmp.name)
        self.pins_path = Path(self.tmp.name, 'pins.json')
        drift.write_pins(pins_for(), self.pins_path)
        code, out = cli('--llvm-src', self.tree.top, '--pins', self.pins_path, '--update', '--revision', 'rev-A')
        self.assertEqual(code, 0, out)

    def compare(self):
        return drift.compare(drift.load_pins(self.pins_path), self.tree.top)

    def test_self_consistent(self):
        pins = drift.load_pins(self.pins_path)
        self.assertEqual(pins['provenance']['revision'], 'rev-A')
        self.assertTrue(all(re.fullmatch('[0-9a-f]{64}', f['sha256']) for f in pins['facts']))
        r = self.compare()
        self.assertEqual((r['status'], r['exit_code']), ('PASS', 0))
        self.assertEqual({f['state'] for f in r['facts']}, {'same'})
        # every accepted layout of --llvm-src: checkout, its llvm/ directory, the AMDGPU directory
        for src in (self.tree.top / 'llvm', self.tree.amdgpu):
            self.assertEqual(drift.compare(drift.load_pins(self.pins_path), src)['status'], 'PASS')
        code, out = cli('--llvm-src', self.tree.top, '--pins', self.pins_path)
        self.assertEqual(code, 0, out)
        self.assertIn('10 of 10 facts unchanged', out)

    def test_reformatting_is_not_drift(self):
        self.tree.edit('Sub.h', 'bool hasFoo() const { return Gen >= 11; }',
                       'bool hasFoo() const {\n    // reviewed\n    return Gen >=  11;\n  }')
        self.tree.edit('F.td', '[FeatureA, // trailing comment\n   FeatureB]', '[FeatureA,FeatureB]')
        r = self.compare()
        self.assertEqual(r['status'], 'PASS', [f for f in r['facts'] if f['state'] != 'same'])

    def test_body_change_is_drift(self):
        self.tree.edit('Sub.h', 'return Gen == 12;', 'return Gen >= 12;')
        r = self.compare()
        self.assertEqual((r['status'], r['exit_code']), ('DRIFT', 1))
        changed = {f['id'] for f in r['facts'] if f['state'] == 'changed'}
        self.assertEqual(changed, {'bar', 'whole'})               # the predicate and the whole file
        code, out = cli('--llvm-src', self.tree.top, '--pins', self.pins_path)
        self.assertEqual(code, 1)
        self.assertIn('CHANGED  bar', out)
        self.assertIn('re-review: RULE_BAR', out)
        self.assertIn('RULE_WHOLE (whole)', out)
        self.assertNotIn('RULE_FOO', out)

    def test_feature_list_and_statement_drift(self):
        self.tree.edit('F.td', 'FeatureB]>;', 'FeatureB, FeatureNewHazard]>;')
        self.tree.edit('Rec.cpp', 'if (Flag) { emit(2); }', 'emit(2);')
        r = self.compare()
        self.assertEqual({f['id'] for f in r['facts'] if f['state'] == 'changed'}, {'feat', 'block'})

    def test_missing_anchor_and_file(self):
        self.tree.edit('Rec.cpp', 'void Rec::fixAll(', 'void Rec::fixEverything(')
        (self.tree.amdgpu / 'F.td').unlink()
        r = self.compare()
        self.assertEqual((r['status'], r['exit_code']), ('INCOMPLETE', 3))
        missing = {f['id']: f['reason'] for f in r['facts'] if f['state'] == 'missing'}
        self.assertEqual(set(missing), {'all', 'feat', 'proc', 'body'})
        self.assertIn('no definition of Rec::fixAll', missing['all'])
        self.assertIn('F.td not found', missing['feat'])
        code, out = cli('--llvm-src', self.tree.top, '--pins', self.pins_path)
        self.assertEqual(code, 3)
        self.assertIn('MISSING  all', out)

    def test_missing_beats_drift(self):
        self.tree.edit('Sub.h', 'return Gen == 12;', 'return Gen >= 12;')
        self.tree.edit('Rec.cpp', 'void Rec::fixAll(', 'void Rec::fixEverything(')
        self.assertEqual(self.compare()['exit_code'], 3)

    def test_update_records_revision_and_accepts_change(self):
        self.tree.edit('Sub.h', 'return Gen == 12;', 'return Gen >= 12;')
        code, out = cli('--llvm-src', self.tree.top, '--pins', self.pins_path, '--update', '--revision', 'rev-B')
        self.assertEqual(code, 0, out)
        pins = drift.load_pins(self.pins_path)
        self.assertEqual(pins['provenance']['revision'], 'rev-B')
        self.assertEqual(self.compare()['status'], 'PASS')

    def test_update_replaces_provenance_note(self):
        """Review issue 3: --update must not keep a note that describes the old hashes."""
        pins = drift.load_pins(self.pins_path)
        pins['provenance']['note'] = 'these pins are NOT from rev-B'
        drift.write_pins(pins, self.pins_path)
        self.tree.edit('Sub.h', 'return Gen == 12;', 'return Gen >= 12;')
        code, out = cli('--llvm-src', self.tree.top, '--pins', self.pins_path, '--update', '--revision', 'rev-B')
        self.assertEqual(code, 0, out)
        prov = drift.load_pins(self.pins_path)['provenance']
        self.assertNotIn('NOT from', prov['note'])
        self.assertIn('rev-B', prov['note'])
        self.assertIn('bar', prov['note'])                         # names the facts that changed
        code, out = cli('--llvm-src', self.tree.top, '--pins', self.pins_path, '--update', '--revision', 'rev-C',
                        '--note', 'reviewed by hand')
        self.assertEqual((code, drift.load_pins(self.pins_path)['provenance']['note']), (0, 'reviewed by hand'))

    def test_update_refuses_with_missing_anchor(self):
        before = self.pins_path.read_text()
        self.tree.edit('Rec.cpp', 'void Rec::fixAll(', 'void Rec::fixEverything(')
        code, out = cli('--llvm-src', self.tree.top, '--pins', self.pins_path, '--update', '--revision', 'rev-C')
        self.assertEqual(code, 3)
        self.assertIn('NOT rewritten', out)
        self.assertEqual(self.pins_path.read_text(), before)

    def test_json_output(self):
        self.tree.edit('Sub.h', 'return Gen == 12;', 'return Gen >= 12;')
        j = Path(self.tmp.name, 'r.json')
        code, _ = cli('--llvm-src', self.tree.top, '--pins', self.pins_path, '--json', j)
        r = json.loads(j.read_text())
        self.assertEqual((code, r['status'], r['pinned_revision']), (1, 'DRIFT', 'rev-A'))
        bar = next(f for f in r['facts'] if f['id'] == 'bar')
        self.assertEqual((bar['state'], bar['rules'], bar['line']), ('changed', ['RULE_BAR'], 5))

    def test_helper_function_drift(self):
        """Review issue 2: a static helper the pinned caller uses is its own fact."""
        self.tree.edit('Rec.cpp', 'void Rec::run() {', 'static bool isPerm(int O) {\n  return O == 1 || O == 2;\n}\nvoid Rec::run() {')
        pins = drift.load_pins(self.pins_path)
        pins['facts'].append({'id': 'helper', 'family': 'haz', 'file': 'Rec.cpp', 'anchor': 'isPerm',
                              'extractor': 'cxx_function', 'sha256': '', 'rules': ['RULE_HELPER'], 'note': ''})
        drift.write_pins(pins, self.pins_path)
        self.assertEqual(cli('--llvm-src', self.tree.top, '--pins', self.pins_path, '--update')[0], 0)
        self.tree.edit('Rec.cpp', 'return O == 1 || O == 2;', 'return O == 3 || O == 2;')
        r = self.compare()
        self.assertEqual({f['id'] for f in r['facts'] if f['state'] == 'changed'}, {'helper'})

    def test_alt_files_fallback(self):
        """Review issue 1: a TableGen def that moved between files (GCNFeatures.td upstream,
        AMDGPU.td in ROCm d6f6cb69) is found through alt_files, with the same hash."""
        pins = drift.load_pins(self.pins_path)
        feat = next(f for f in pins['facts'] if f['id'] == 'feat')
        feat['file'], feat['alt_files'] = 'GCNFeatures.td', ['F.td']
        drift.write_pins(pins, self.pins_path)
        r = self.compare()                                   # GCNFeatures.td does not exist: F.td is used
        self.assertEqual(r['status'], 'PASS', drift.render(r))
        self.assertEqual(next(f for f in r['facts'] if f['id'] == 'feat')['found_in'], 'F.td')
        (self.tree.amdgpu / 'GCNFeatures.td').write_text('404: Not Found')     # present but without the def
        self.assertEqual(self.compare()['status'], 'PASS')
        (self.tree.amdgpu / 'GCNFeatures.td').write_text('def FeatureX12 : FeatureSet<[FeatureA]>;')
        r = self.compare()                                   # in both files: ambiguous, not silently one
        feat = next(f for f in r['facts'] if f['id'] == 'feat')
        self.assertEqual((r['status'], feat['state']), ('INCOMPLETE', 'missing'))
        self.assertIn('found in 2 files', feat['reason'])
        (self.tree.amdgpu / 'GCNFeatures.td').unlink(); (self.tree.amdgpu / 'F.td').unlink()
        feat = next(f for f in self.compare()['facts'] if f['id'] == 'feat')
        self.assertIn('GCNFeatures.td not found', feat['reason'])
        self.assertIn('F.td not found', feat['reason'])

    def test_bad_alt_files(self):
        bad = pins_for()
        bad['facts'][0]['alt_files'] = 'F.td'
        drift.write_pins(bad, self.pins_path)
        with self.assertRaisesRegex(ValueError, 'alt_files'):
            drift.load_pins(self.pins_path)

    def test_bad_pin_file(self):
        bad = pins_for()
        del bad['facts'][0]['rules']
        drift.write_pins(bad, self.pins_path)
        with self.assertRaisesRegex(ValueError, 'lacks rules'):
            drift.load_pins(self.pins_path)


REQUIRED = {  # (file, anchor) the task requires pinned
    ('AMDGPU.td', 'FeatureISAVersion12'),
    ('GCNSubtarget.h', 'hasVALUReadSGPRHazard'), ('GCNSubtarget.h', 'requiresNopBeforeDeallocVGPRs'),
    ('GCNSubtarget.h', 'requiresWaitIdleBeforeGetReg'), ('GCNSubtarget.h', 'setRegModeNeedsVNOPs'),
    ('GCNSubtarget.h', 'hasVALUPartialForwardingHazard'), ('GCNSubtarget.h', 'hasVALUMaskWriteHazard'),
    ('GCNSubtarget.h', 'hasLdsDirect'),
    ('GCNHazardRecognizer.cpp', 'GCNHazardRecognizer::fixHazards'),
    ('GCNHazardRecognizer.cpp', 'GCNHazardRecognizer::fixWMMAHazards'),
    ('GCNHazardRecognizer.cpp', 'GCNHazardRecognizer::fixVcmpxPermlaneHazards'),
    ('AMDGPUWaitSGPRHazards.cpp', 'AMDGPUWaitSGPRHazards.cpp'),
    ('SIInsertWaitcnts.cpp', 'WaitcntBrackets::determineVGPR16Dependency'),
    ('SIInsertHardClauses.cpp', 'SIInsertHardClauses.cpp'),
    # helpers whose content the rules restate (review issue 2)
    ('GCNHazardRecognizer.cpp', 'isPermlane'), ('GCNHazardRecognizer.cpp', 'isVCmpXWritesExec'),
    ('SIMemoryLegalizer.cpp', 'SIGfx12CacheControl::insertWaitsBeforeSystemScopeStore'),
}


class ShippedPins(unittest.TestCase):
    def setUp(self):
        self.pins = drift.load_pins()

    def test_required_facts_present(self):
        have = {(f['file'], f['anchor']) for f in self.pins['facts']}
        self.assertFalse(REQUIRED - have, REQUIRED - have)
        self.assertTrue(any(f['file'] == 'SIInsertWaitcnts.cpp' and f['extractor'] == 'statement'
                            and 'DEALLOC' in ''.join(f['rules']) for f in self.pins['facts']))
        self.assertTrue(all(re.fullmatch('[0-9a-f]{64}', f['sha256']) for f in self.pins['facts']))

    def test_provenance_is_honest(self):
        """The shipped pins are from the target compiler, ROCm llvm-project d6f6cb69 (full hash
        recorded), and the note says that upstream main differs in 9 reviewed facts."""
        prov = self.pins['provenance']
        self.assertIn('d6f6cb691863e9258034ab48c1dfce21689d5606', prov['revision'])
        self.assertIn('ROCm', prov['revision'])
        self.assertIn('upstream', prov['note'])
        self.assertIn('9 facts', prov['note'])

    def test_feature_defs_found_in_both_trees(self):
        """ROCm d6f6cb69 (the pinned tree) defines FeatureISAVersion12 and FeatureGFX12 in
        AMDGPU.td; upstream main moved them to GCNFeatures.td. Each feature fact searches both."""
        feats = [f for f in self.pins['facts'] if f['extractor'] == 'td_def' and f['anchor'].startswith('Feature')]
        self.assertEqual(len(feats), 2)
        for f in feats:
            self.assertEqual({f['file'], *f.get('alt_files', [])}, {'AMDGPU.td', 'GCNFeatures.td'}, f['id'])

    def test_rule_names_exist(self):
        """Each pinned rule name is a rule radisa can report ('*' = all; 'X_*' = a family)."""
        code = '\n'.join(p.read_text() for p in (ROOT / 'radisa').glob('*.py'))
        for f in self.pins['facts']:
            for rule in f['rules']:
                if rule == '*':
                    continue
                pat = re.escape(rule[:-1]) + r'[A-Z0-9_]+' if rule.endswith('*') else re.escape(rule)
                self.assertRegex(code, "'" + pat + "'", f'{f["id"]}: {rule}')

    @unittest.skipUnless(os.environ.get('RADISA_LLVM_SRC'), 'set RADISA_LLVM_SRC to an llvm-project checkout')
    def test_shipped_pins_match_llvm_src(self):
        r = drift.compare(self.pins, Path(os.environ['RADISA_LLVM_SRC']))
        self.assertEqual(r['status'], 'PASS', drift.render(r))


if __name__ == '__main__':
    unittest.main()
