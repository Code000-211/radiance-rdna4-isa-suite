"""Linter hygiene for `radisa check`: rule registry, configuration, inline suppression, baseline.

Everything here runs on the finished report (`check.REPORT_FILTERS` entry `report_filter`) and
changes only which findings are shown and at which severity; no analysis depends on it. The kernel
passes see the enable/disable sets through `options_from(args)` (ctx['options']).

Order inside `report_filter`: configuration (severity, disable/off), then inline suppressions, then
the baseline. Findings that say the analysis did not happen (severity 'incomplete', and the rules in
PROTECTED: DECODE, BYTE_COVERAGE, CHECK_FAILED, ASSEMBLER, SOURCE_MAP, COMPILE, NO_KERNELS) are never
re-graded, suppressed or baselined: hiding them would turn "not analysed" into "clean".

Configuration (`--config FILE`, default the nearest `.radisa.json` in the input's directory or above)::

    {"severity": {"RULE": "error|warning|note|off"}, "enable": ["SYS_STORE_WAIT"],
     "disable": ["HIP_NO_LAUNCH_BOUNDS"], "baseline": "radisa-baseline.json"}

warning is the internal 'review', note is 'info'; 'off' drops the finding (like "disable"). A
relative "baseline" is relative to the config file. Command-line `--severity RULE=LEVEL`,
`--enable RULE`, `--disable RULE` and `--baseline FILE` override the file: a later --enable undoes a
configured disable and the reverse. `enable` switches on opt-in rules (SYS_STORE_WAIT); it reaches
the passes as options['enable'].

Inline suppression, in a comment (`//`, `/* */` in HIP/C++; `;`, `//`, `/* */` in assembly)::

    // radisa: ignore[MISSING_WAIT,MISSING_ALU_WAIT] the asm block above waits for it
    ; radisa: ignore-file[HIP_NO_LAUNCH_BOUNDS] launch bounds come from the host wrapper

`ignore[...]` in a trailing comment (code before it on the same line) covers findings on that line
only. On a comment-only line L it covers line L and the next line with code, reached over
comment-only lines (further `radisa:` lines, the rest of a block comment) but not over a blank
line. A /* */ directive may continue its reason on the next lines of the same block comment.
`ignore-file[...]` covers the whole file, and, in the checked input file itself, also findings that
carry no source location (REGISTER_SPILL, ...). Line-scoped suppression needs finding['loc']: a
finding without one (machine code without line tables) cannot be suppressed by line, and the
UNUSED_SUPPRESSION message then says so. The reason after `]` is mandatory: without it the comment
does nothing and SUPPRESSION_NO_REASON (warning) is reported. Every listed rule of a directive in
the checked input file that suppressed nothing is reported as UNUSED_SUPPRESSION (warning), except
for rules that are disabled and when `--kernel` limits the run; directives in other files (headers)
are not, since another translation unit may need them. Only files that the input is or that a
finding points into are read for comments.

Suppressed findings move to report['suppressed'] (with 'suppression': kind, reason, file, line, and
'kernel'), baselined ones to report['baselined'] (`hidden_findings`, `sarif_suppression`). Every
error-severity finding that is not shown as an error, whether suppressed, baselined, disabled or
re-graded by the configuration, is named in the summary line (`summary_suffix`, "hidden errors"):
hiding a memory-safety error is the user's call, but it stays visible.

Baseline fingerprint (`--write-baseline FILE`, `--baseline FILE`): sha256 over the JSON list

    [rule, kernel name ('' for report-level findings), normalised instruction text,
     normalised producer instruction text, normalised source line text, occurrence index]

* instruction text: finding['instruction'], lower-case, comments removed, whitespace collapsed,
  no blanks after commas;
* producer instruction: finding['producer_instruction'] when check.py provides it, else '';
* source line text: only for findings without an instruction (HIP lexical findings): the text of
  the line finding['loc'] points at, normalised the same way; '' otherwise;
* occurrence index: the position among findings with the same five components, ordered by pc,
  then line.

No pc, address or line number goes in, so edits elsewhere in the file do not invalidate the
baseline; an edit of the flagged instruction (or, for lexical findings, of the flagged line) does.

A baseline file holds the entries of any number of input files: each entry has 'input', the input
file relative to the baseline's directory (`input_key`); an older single-file baseline without it
belongs to its top-level 'input'. `--baseline` applies only the entries of the checked file (one
file's baseline never hides findings in another, also with a repository-wide "baseline" in
.radisa.json), moves matching findings to report['baselined'] and adds BASELINE_STALE (note) for
this file's entries that no longer occur (not with --kernel). `--write-baseline FILE` replaces this
input's entries in FILE (with --kernel: those of the checked kernels) and keeps every other file's;
the file is rewritten atomically under a lock on its directory (POSIX), so concurrent runs into one
FILE all land. It records all findings left after configuration and inline suppression (not this
module's own hygiene findings) and then applies FILE as the baseline of the same run.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from . import __version__

SCHEMA = 'radisa-baseline-v1'
CONFIG_NAME = '.radisa.json'
LEVELS = {'error': 'error', 'warning': 'review', 'note': 'info', 'off': 'off',
          'review': 'review', 'info': 'info'}
# "The analysis did not happen" findings: never re-graded, suppressed or baselined.
PROTECTED = ('DECODE', 'BYTE_COVERAGE', 'CHECK_FAILED', 'ASSEMBLER', 'SOURCE_MAP', 'COMPILE', 'NO_KERNELS')
HYGIENE = ('SUPPRESSION_NO_REASON', 'UNUSED_SUPPRESSION', 'BASELINE_STALE', 'CONFIG_UNKNOWN_RULE')
ASM_SUFFIXES = ('.s', '.S', '.asm')
TEXT_SUFFIXES = ASM_SUFFIXES + ('.hip', '.cpp', '.cc', '.cxx', '.cu', '.c', '.h', '.hpp', '.hh', '.cuh', '.inc')
DIRECTIVE = re.compile(r'radisa:\s*ignore(?P<file>-file)?\[(?P<rules>[^\]]*)\](?P<reason>.*)')
SOURCE = 'radisa/suppress.py (radisa check configuration, inline suppression and baseline)'
UNRELEASED = 'unreleased'
V = '0.2.0'
LLVM = 'LLVM llvm/lib/Target/AMDGPU/'


def _r(severity: str, summary: str, basis: str, since: str = V, **extra) -> dict:
    return {'severity': severity, 'summary': summary, 'basis': basis, 'since': since, **extra}


# Every rule id the radisa package can emit. tests/test_suppress.py greps the package for rule
# literals and fails for any id missing here. 'tool': 'audit' marks `radisa audit` (cli.py) rules;
# 'opt_in': off unless enabled; 'memory_safety': a wrong program may read or write wrong data.
RULES: dict[str, dict] = {
    # --- counters and waits (waits.py, sgpr.py)
    'MISSING_WAIT': _r('error', 'a register is read or overwritten while a memory instruction writing it may '
                       'still be outstanding on some CFG path', 'RDNA4 ISA §5.7.1 memory dependency counters',
                       memory_safety=True),
    'LDS_STORE_BEFORE_BARRIER': _r('error', 'an LDS write may still be outstanding (DScnt) at a workgroup barrier',
                                   'LLVM AMDGPU User Guide, memory model GFX12', memory_safety=True),
    'MISSING_ALU_WAIT': _r('error', 'an SGPR read by vector instructions is read before its last write is '
                           'committed, with no s_wait_alu in between', LLVM + 'AMDGPUWaitSGPRHazards.cpp',
                           memory_safety=True),
    'ALU_WAIT_TTMP': _r('review', 'the MISSING_ALU_WAIT hazard where LLVM itself emits no wait (ttmp read in between)',
                        LLVM + 'AMDGPUWaitSGPRHazards.cpp (ttmp registers numbered as SGPR 0)'),
    'CALLEE_RETURN_OUTSTANDING': _r('review', 'a called function returns while memory operations other than '
                                    'stores may still be outstanding', LLVM + 'SIInsertWaitcnts.cpp (waits before '
                                    'a return); RDNA4 ISA §5.7.1', UNRELEASED),
    'LOCAL_MEMORY_HAZARD': _r('review', 'older local-memory wait heuristic (radisa audit; superseded by '
                              'MISSING_WAIT in radisa check)', 'RDNA4 ISA §5.7.1', tool='audit'),
    'SGPR_PAIR_GUARD': _r('review', 'VALU-read / SALU-write / read SGPR-pair pattern without s_wait_alu',
                          'RDNA4 ISA §5.8'),
    'SYS_STORE_WAIT': _r('review', 'system-scope store while loads/stores may be outstanding (opt-in, '
                         'unconfirmed)', 'LLVM SIMemoryLegalizer.cpp SIGfx12CacheControl::finalizeStore; '
                         'GCNFeatures.td FeatureWaitsBeforeSystemScopeStores', UNRELEASED, opt_in=True),
    # --- control flow and program end (rules.py)
    'CFG_FALLTHROUGH': _r('error', 'a path runs past the last instruction without s_endpgm (note for snippets)',
                          'RDNA4 ISA §5.5'),
    'CFG_INDIRECT': _r('incomplete', 'call or indirect branch; its target is not followed', 'RDNA4 ISA §5.5'),
    'CFG_TARGET': _r('incomplete', 'a branch target is not an instruction boundary', 'RDNA4 ISA §5.5'),
    'CFG_ENCODING': _r('incomplete', 'unexpected branch encoding size', 'RDNA4 ISA §5.5'),
    'CALLS': _r('info', 'calls or indirect branches; callee bodies are assumed to drain the counters',
                'LLVM SIInsertWaitcnts.cpp callee prologue'),
    'REACHABLE_CODE_END': _r('error', 'reachable s_code_end padding', 'RDNA4 ISA §2.4'),
    # --- registers and resources
    'VGPR_OUT_OF_RANGE': _r('error', 'an instruction names a VGPR the kernel descriptor does not allocate',
                            'RDNA4 ISA §3.3.2', memory_safety=True),
    'UNINITIALIZED_READ': _r('review', 'a register is read on some path before anything defines it',
                             'LLVM AMDGPU User Guide, initial kernel execution state; SIISelLowering.cpp',
                             UNRELEASED),
    'REGISTER_SPILL': _r('review', 'VGPR/SGPR spills to scratch', 'AMDGPU code object metadata'),
    'SCRATCH_USE': _r('review', 'scratch memory without spills (private array not kept in registers)',
                      'AMDGPU code object metadata'),
    'OPERAND_MODEL_UNKNOWN': _r('review', 'instructions outside the AMD operand table, assumed to read and write '
                                'every register they name', 'AMD RDNA4 ISA XML; radisa/isadb.py', UNRELEASED),
    # --- hazards (hazards.py)
    'WMMA_D_TO_AB': _r('error', 'WMMA/SWMMAC D feeds the A/B (or SWMMAC index) of the next one without a VALU '
                       'in between', LLVM + 'GCNHazardRecognizer.cpp fixWMMAHazards; RDNA4 ISA §7.12.1',
                       memory_safety=True),
    'WMMA_OPERANDS': _r('incomplete', 'WMMA operands not understood', 'RDNA4 ISA §7.12.1'),
    'PERMLANE_AFTER_CMPX': _r('error', 'V_PERMLANE* after a compare that writes EXEC without a VALU in between',
                              LLVM + 'GCNHazardRecognizer.cpp fixVcmpxPermlaneHazards; RDNA4 ISA §7.2.8'),
    'DEALLOC_NO_NOP': _r('review', 's_sendmsg MSG_DEALLOC_VGPRS not directly preceded by s_nop',
                         LLVM + 'SIInsertWaitcnts.cpp; GCNSubtarget.h requiresNopBeforeDeallocVGPRs', UNRELEASED),
    'CLAUSE_TOO_LONG': _r('review', 's_clause longer than LLVM ever forms for gfx1201 (32)',
                          LLVM + 'SIInsertHardClauses.cpp; FeatureMaxHardClauseLength32', UNRELEASED),
    'CLAUSE_MIXED': _r('review', 's_clause over memory instructions of different clause types or non-clause '
                       'instructions', LLVM + 'SIInsertHardClauses.cpp getHardClauseType', UNRELEASED),
    'CLAUSE_BRANCH': _r('review', 'a branch targets the inside of an s_clause body',
                        LLVM + 'SIInsertHardClauses.cpp (clauses are formed per basic block)', UNRELEASED),
    # --- dynamic VGPRs / deallocation (rules.py)
    'USE_AFTER_DEALLOC': _r('error', 'vector or memory operation reachable after MSG_DEALLOC_VGPRS',
                            'RDNA4 ISA §3.3.3', memory_safety=True),
    'DEALLOC_EPILOGUE': _r('review', 'nontrivial epilogue after VGPR release', 'RDNA4 ISA §3.3.3'),
    'DEALLOC_DYNAMIC': _r('review', 's_alloc_vgpr and full deallocation in one kernel', 'RDNA4 ISA §3.3.3'),
    'DYNAMIC_WAVE64': _r('error', 's_alloc_vgpr in wave64', 'RDNA4 ISA §3.3.3'),
    'DYNAMIC_VGPR_MODE': _r('incomplete', 'dynamic VGPR mode is not modelled', 'RDNA4 ISA §3.3.3'),
    'LAST_USE_LIFETIME': _r('review', 'last-use (TH_LU) access needs a whole-buffer lifetime proof',
                            'RDNA4 ISA §4.1.1'),
    'WATERFALL_CANDIDATE': _r('info', 'EXEC loop around relative VGPR indexing', 'RDNA4 ISA §7.11'),
    # --- VOPD (vopd.py via rules.py)
    'VOPD_WAVE32': _r('error', 'VOPD in wave64', 'RDNA4 ISA §7.8'),
    'VOPD_OPCODE': _r('error', 'reserved VOPD X/Y opcode', 'RDNA4 ISA §7.8, §16.11'),
    'VOPD_DPP': _r('error', 'DPP source in a VOPD component', 'RDNA4 ISA §7.8'),
    'VOPD_SOURCE': _r('error', 'reserved source encoding in a VOPD component', 'RDNA4 ISA §7.8, §16.11'),
    'VOPD_SOURCE_HI': _r('review', 'VCC_HI/EXEC_HI sourced by a wave32 VOPD component', 'RDNA4 ISA §7.8'),
    'VOPD_REGISTER_RANGE': _r('review', 'VOPD names a VGPR outside the descriptor allocation', 'RDNA4 ISA §3.3.2'),
    'VOPD_BANK': _r('error', 'VOPD SRC0/SRC1 bank conflict', 'RDNA4 ISA §7.8'),
    'VOPD_SRC2': _r('error', 'VOPD SRC2 operands of equal parity', 'RDNA4 ISA §7.8'),
    'VOPD_RAW_DPMACC': _r('error', 'VOPD X (DPMACC) writes a register Y reads', 'RDNA4 ISA §7.8; '
                          'LLVM SIInstrInfo::isVOPDAntidependencyAllowed'),
    'VOPD_ANTIDEP': _r('review', 'VOPD X writes a register Y reads (LLVM emits it for non-DPMACC)',
                       'RDNA4 ISA §7.8; LLVM SIInstrInfo::isVOPDAntidependencyAllowed'),
    'VOPD_SCALAR_BUS': _r('error', 'VOPD pair with more than two scalar/literal sources', 'RDNA4 ISA §7.8'),
    'VOPD_IDENTITY': _r('incomplete', 'LLVM VOPD text does not match the raw format decoder', 'RDNA4 ISA §7.8'),
    # --- HIP source (hipsrc.py, asmclobber.py)
    'HIP_INLINE_ASM_MEMORY': _r('review', 'inline asm with a memory instruction and no wait in it',
                                'lexical; RDNA4 ISA §5.7.1'),
    'HIP_WAVE64_WIDTH': _r('review', 'shuffle width 64 on a wave32 target', 'lexical'),
    'HIP_LEGACY_WAITCNT': _r('review', '__builtin_amdgcn_s_waitcnt (S_WAITCNT = S_WAIT_IDLE on RDNA4)',
                             'lexical; RDNA4 ISA §16.5'),
    'HIP_NO_LAUNCH_BOUNDS': _r('info', 'kernel without __launch_bounds__', 'lexical'),
    'ASM_UNDECLARED_CLOBBER': _r('review', 'inline asm writes a register that is neither an output nor clobbered',
                                 'LLVM LangRef inline assembler constraints; clang AMDGPU GCCRegNames', UNRELEASED),
    # --- toolchain, evidence, coverage (check.py, rules.py, incremental.py)
    'COMPILE': _r('error', 'the gfx1201 device compile failed', 'toolchain'),
    'ASSEMBLER': _r('error', 'llvm-mc rejected the input (incomplete for compiler output)', 'toolchain'),
    'COMPILER_DIAGNOSTICS': _r('info', 'compiler warnings', 'toolchain'),
    'SOURCE_MAP': _r('incomplete', 'machine code could not be mapped to source lines', 'radisa/hipsrc.py'),
    'CHECK_FAILED': _r('incomplete', 'the input could not be read or analysed', 'radisa/check.py'),
    'NO_KERNELS': _r('info', 'the input has no gfx1201 kernel', 'AMDGPU code object metadata'),
    'CACHE': _r('info', 'the incremental result cache was not written', 'radisa/incremental.py'),
    'ASM_CLOBBER_SKIPPED': _r('info', 'the device IR for ASM_UNDECLARED_CLOBBER could not be produced '
                              '(--no-incremental)', 'radisa/check.py check_hip; radisa/hipsrc.py compile_ir',
                              UNRELEASED),
    # --- radisa check-all (batch.py): one input file was not analysed
    'CHECK_CRASHED': _r('incomplete', 'the checker failed on this file (exception or worker exit)',
                        'radisa check-all', UNRELEASED),
    'CHECK_TIMEOUT': _r('incomplete', 'the file took longer than --timeout-per-file', 'radisa check-all', UNRELEASED),
    'SKIPPED_TOO_MANY_KERNELS': _r('incomplete', 'the file has more kernels than --max-kernels-per-file',
                                   'radisa check-all', UNRELEASED),
    'DECODE': _r('incomplete', 'bytes the disassembler could not decode', 'radisa/decode.py'),
    'BYTE_COVERAGE': _r('incomplete', 'not every kernel byte was decoded', 'radisa/rules.py'),
    'KERNEL_EVIDENCE': _r('incomplete', 'kernel descriptor or metadata problem', 'AMDGPU code object'),
    'OBJECT_EVIDENCE': _r('incomplete', 'code object problem', 'AMDGPU code object'),
    # --- radisa audit (cli.py)
    'COLLECTION': _r('incomplete', 'evidence collection error', 'radisa collect', tool='audit'),
    'AUDIT_OBJECT': _r('incomplete', 'a code object could not be audited', 'radisa audit', tool='audit'),
    'OBJECT_SET_MISMATCH': _r('incomplete', 'evidence directory and manifest differ', 'radisa audit', tool='audit'),
    'EMPTY_COVERAGE': _r('incomplete', 'zero expected kernels', 'radisa audit', tool='audit'),
    'MANIFEST_COUNT': _r('incomplete', 'inconsistent manifest kernel total', 'radisa audit', tool='audit'),
    'MISSING_KERNELS': _r('incomplete', 'not every kernel reached the checker', 'radisa audit', tool='audit'),
    'BASELINE_MISMATCH': _r('incomplete', 'audited inventory differs from the approved inventory baseline',
                            'radisa audit --expect', tool='audit'),
    'RUNTIME_COVERAGE': _r('incomplete', 'capture log problem', 'radisa audit --capture-log', tool='audit'),
    'RUNTIME_NOT_CAPTURED': _r('info', 'runtime-resolved kernels not reconciled', 'radisa audit', tool='audit'),
    # --- this module
    'SUPPRESSION_NO_REASON': _r('review', 'a radisa: ignore comment without a reason (it suppresses nothing)',
                                SOURCE, UNRELEASED),
    'UNUSED_SUPPRESSION': _r('review', 'a radisa: ignore comment lists a rule that reported nothing there',
                             SOURCE, UNRELEASED),
    'BASELINE_STALE': _r('info', 'baseline entries that no longer occur', SOURCE, UNRELEASED),
    'CONFIG_UNKNOWN_RULE': _r('review', 'configuration names a rule radisa does not know', SOURCE, UNRELEASED),
}


class RuleSet(frozenset):
    """A frozenset of rule ids whose str()/repr() is sorted, so json.dumps(..., default=str) of the
    pass options (check.analysis_identity, the incremental cache salt) is the same in every run."""

    def __repr__(self) -> str:
        return 'RuleSet(' + repr(sorted(self)) + ')'

    __str__ = __repr__


# ------------------------------------------------------------------ arguments and configuration

def add_arguments(parser) -> None:
    """The `radisa check` options of this module (call from check.add_parser)."""
    g = parser.add_argument_group('rule selection and suppression')
    g.add_argument('--config', help=f'JSON configuration (default: nearest {CONFIG_NAME} upward from the input)')
    g.add_argument('--no-config', action='store_true', help=f'do not look for {CONFIG_NAME}')
    g.add_argument('--severity', action='append', default=[], metavar='RULE=LEVEL',
                   help='override a rule severity: error, warning, note or off (repeatable)')
    g.add_argument('--enable', action='append', default=[], metavar='RULE', help='enable a rule (repeatable)')
    g.add_argument('--disable', action='append', default=[], metavar='RULE', help='disable a rule (repeatable)')
    g.add_argument('--baseline', help='hide findings recorded in this baseline file; report only new ones')
    g.add_argument('--write-baseline', metavar='FILE', help='record the current findings as the baseline')


def _names(values) -> list[str]:
    out = []
    for v in values or []:
        out += [x.strip().upper() for x in re.split(r'[,\s]+', str(v)) if x.strip()]
    return out


def _level(rule: str, value) -> str:
    level = LEVELS.get(str(value).strip().lower())
    if level is None:
        raise ValueError(f'severity for {rule}: {value!r} is not one of error, warning, note, off')
    return level


def find_config(start: Path) -> Path | None:
    """The nearest .radisa.json in `start` (a file's directory) or a parent directory."""
    d = start.resolve()
    if not d.is_dir():
        d = d.parent
    for p in (d, *d.parents):
        if (p / CONFIG_NAME).is_file():
            return p / CONFIG_NAME
    return None


def load_config(args) -> tuple[dict, Path | None]:
    """(config dict, its path); ({}, None) without one. ValueError on a malformed file."""
    if getattr(args, 'config', None):
        path = Path(args.config)
    elif getattr(args, 'no_config', False) or not getattr(args, 'input', None):
        return {}, None
    else:
        path = find_config(Path(args.input))
        if path is None:
            return {}, None
    try:
        cfg = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as ex:
        raise ValueError(f'radisa config {path}: {ex}') from ex
    if not isinstance(cfg, dict):
        raise ValueError(f'radisa config {path}: the top level must be an object')
    extra = set(cfg) - {'severity', 'enable', 'disable', 'baseline'}
    if extra:
        raise ValueError(f'radisa config {path}: unknown keys {sorted(extra)}')
    if not isinstance(cfg.get('severity', {}), dict):
        raise ValueError(f'radisa config {path}: "severity" must be an object')
    for k in ('enable', 'disable'):
        if not isinstance(cfg.get(k, []), list):
            raise ValueError(f'radisa config {path}: "{k}" must be a list')
    return cfg, path


def settings(args) -> dict:
    """The merged selection: {'severity': {rule: internal level or 'off'}, 'enable': set,
    'disable': set (includes 'off'), 'baseline': Path | None, 'config': Path | None,
    'named': set of every rule id the configuration names}. ValueError on bad input."""
    cached = getattr(args, '_radisa_settings', None)
    if cached is not None:
        return cached
    cfg, path = load_config(args)
    severity = {str(r).upper(): _level(r, v) for r, v in (cfg.get('severity') or {}).items()}
    enable, disable = set(_names(cfg.get('enable'))), set(_names(cfg.get('disable')))
    for item in getattr(args, 'severity', None) or []:
        rule, sep, value = str(item).partition('=')
        if not sep or not rule.strip():
            raise ValueError(f'--severity {item!r}: expected RULE=LEVEL')
        rule = rule.strip().upper()
        severity[rule] = _level(rule, value)
        if severity[rule] != 'off':
            disable.discard(rule)
    for rule in _names(getattr(args, 'enable', None)):
        enable.add(rule)
        disable.discard(rule)
        if severity.get(rule) == 'off':
            del severity[rule]
    for rule in _names(getattr(args, 'disable', None)):
        disable.add(rule)
        enable.discard(rule)
    disable |= {r for r, v in severity.items() if v == 'off'}
    baseline = None
    if getattr(args, 'baseline', None):
        baseline = Path(args.baseline)
    elif cfg.get('baseline'):
        baseline = Path(cfg['baseline'])
        if not baseline.is_absolute():
            baseline = path.parent / baseline
    s = {'severity': severity, 'enable': enable, 'disable': disable, 'baseline': baseline, 'config': path,
         'named': set(severity) | enable | disable}
    try:
        args._radisa_settings = s
    except AttributeError:
        pass
    return s


def options_from(args) -> dict:
    """Pass options from the configuration: {'enable': RuleSet, 'disable': RuleSet}.

    Merge into check.pass_options(args) (ctx['options']); memorder.enabled() reads 'enable'."""
    s = settings(args)
    return {'enable': RuleSet(s['enable']), 'disable': RuleSet(s['disable'])}


def validate(args) -> None:
    """Raise ValueError now for a malformed config, --severity or a missing --baseline file."""
    s = settings(args)
    b = s['baseline']
    if b is not None and not b.is_file() and not getattr(args, 'write_baseline', None):
        raise ValueError(f'baseline file not found: {b}')


# ------------------------------------------------------------------ helpers

def _all_lists(report: dict):
    """(kernel name, findings list) for the report-level list and every kernel's."""
    yield '', report['findings']
    for k in report.get('kernels') or []:
        yield k.get('name') or '', k['findings']


def _protected(f: dict) -> bool:
    return f.get('severity') == 'incomplete' or f.get('rule') in PROTECTED


def _norm(text: str) -> str:
    t = re.split(r'\s(?:;|//)', ' ' + (text or ''), maxsplit=1)[0]
    t = re.sub(r'\s+', ' ', t.strip().lower())
    return re.sub(r'\s*,\s*', ',', t)


class _Files:
    """Source files findings point into, read once, matched by resolved path."""

    def __init__(self, input_path: str | None):
        self.input = Path(input_path).resolve() if input_path else None
        self.text: dict[str, list[str] | None] = {}
        self.display: dict[str, str] = {}   # resolved path -> the name findings use for it
        if self.input:
            self.display[str(self.input)] = input_path

    def resolve(self, name: str | None) -> str | None:
        if not name:
            return None
        p = Path(name)
        base = self.input.parent if self.input else Path.cwd()
        cands = [p] if p.is_absolute() else [base / p, Path.cwd() / p]
        if self.input and p.name == self.input.name:
            cands.append(self.input)
        cands.append(base / p.name)
        for c in cands:
            try:
                if c.is_file():
                    key = str(c.resolve())
                    self.display.setdefault(key, name)
                    return key
            except OSError:
                continue
        return None

    def lines(self, key: str | None) -> list[str] | None:
        if key is None:
            return None
        if key not in self.text:
            try:
                self.text[key] = Path(key).read_text(encoding='utf-8', errors='replace').splitlines()
            except OSError:
                self.text[key] = None
        return self.text[key]


def scan_comments(lines: list[str], asm: bool) -> list[dict]:
    """Per line: {'comments': [(start, end, block_continues)], 'code': bool}.

    Comments are // to the end of the line, /* ... */ (also over several lines) and, in assembly,
    ; to the end of the line; text in "..." / '...' literals is code. 'code' says whether the line
    has anything outside comments. block_continues: the segment is a /* comment still open at the
    end of the line."""
    out, block = [], False
    for text in lines:
        segs, code, i, n = [], False, 0, len(text)
        while i < n:
            if block:
                j = text.find('*/', i)
                if j < 0:
                    segs.append((i, n, True))
                    break
                segs.append((i, j, False))
                block, i = False, j + 2
                continue
            if text.startswith('/*', i):
                block, i = True, i + 2
                continue
            if text.startswith('//', i) or (asm and text[i] == ';'):
                segs.append((i + (1 if text[i] == ';' else 2), n, False))
                break
            if text[i] in '"\'':
                q, j = text[i], i + 1
                while j < n and text[j] != q:
                    j += 2 if text[j] == '\\' else 1
                code, i = True, j + 1
                continue
            code = code or not text[i].isspace()
            i += 1
        out.append({'comments': segs, 'code': code})
    return out


def _clean_reason(text: str) -> str:
    return text.strip().lstrip('*').strip().lstrip(':-–— ').strip()


def parse_directives(lines: list[str], path: str) -> list[dict]:
    """The `radisa: ignore[...]` / `ignore-file[...]` comments of one file.

    Each: {'file', 'line' (1-based), 'scope' 'line'|'file', 'rules' [ids], 'reason' str,
    'comment_only' bool, 'covers' set of lines, 'target' the code line a comment-only directive
    applies to (or None)}. A directive counts only inside a comment (`scan_comments`): //, /* */
    and, in assembly, ;. The reason is the rest of that comment; a /* */ directive whose reason is
    empty takes it from the following lines of the same block comment.

    Coverage: a directive on a line that also has code (a trailing comment) covers that line only.
    A directive on a comment-only line covers its own line and the next line with code, reached
    over comment-only lines (more directives, the rest of a block comment) but not over a blank
    line.
    """
    asm = Path(path).suffix in ASM_SUFFIXES
    info = scan_comments(lines, asm)
    found = []
    for n, (text, li) in enumerate(zip(lines, info), 1):
        for start, end, cont in li['comments']:
            m = DIRECTIVE.search(text[start:end])
            if not m:
                continue
            reason = _clean_reason(m['reason'])
            j = n
            while cont and not re.search(r'\w', reason) and j < len(lines):
                nxt = info[j]['comments']   # the block comment's continuation on line j + 1
                if not nxt or nxt[0][0] != 0:
                    break
                s0, e0, cont = nxt[0]
                reason = _clean_reason(lines[j][s0:e0])
                j += 1
            found.append({'file': path, 'line': n, 'scope': 'file' if m['file'] else 'line',
                          'rules': _names([m['rules']]), 'reason': reason if re.search(r'\w', reason) else '',
                          'comment_only': not li['code'], 'used': set()})
    for d in found:
        d['covers'], d['target'] = {d['line']}, None
        if d['comment_only']:
            j = d['line'] + 1
            while j <= len(lines) and not info[j - 1]['code'] and info[j - 1]['comments']:
                j += 1
            if j <= len(lines) and info[j - 1]['code']:
                d['covers'].add(j)
                d['target'] = j
    return found


def _hygiene(rule: str, message: str, d: dict | None, s: dict, **extra) -> dict | None:
    sev = RULES[rule]['severity']
    if rule in s['disable']:
        return None
    if rule in s['severity']:
        sev = s['severity'][rule]
    f = {'rule': rule, 'severity': sev, 'pc': None, 'message': message, 'source': SOURCE, **extra}
    if d is not None:
        f['loc'] = {'file': d['file'], 'line': d['line'], 'column': 1}
    return f


# ------------------------------------------------------------------ fingerprints and baseline

def fingerprints(report: dict, files: _Files | None = None) -> list[tuple[str, dict, list]]:
    """[(fingerprint, finding, findings list it is in)] for every baselinable finding.

    See the module docstring for what goes into a fingerprint."""
    files = files or _Files(report.get('input'))
    groups: dict[tuple, list] = {}
    for kernel, lst in _all_lists(report):
        for f in lst:
            if _protected(f) or f['rule'] in HYGIENE:
                continue
            src = ''
            if not f.get('instruction') and f.get('loc'):
                text = files.lines(files.resolve(f['loc'].get('file')))
                ln = f['loc'].get('line') or 0
                if text and 0 < ln <= len(text):
                    src = _norm(text[ln - 1])
            key = (f['rule'], kernel, _norm(f.get('instruction') or ''),
                   _norm(f.get('producer_instruction') or ''), src)
            groups.setdefault(key, []).append((f, lst))
    out = []
    for key, items in groups.items():
        items.sort(key=lambda x: (x[0].get('pc') is None, x[0].get('pc') or 0, (x[0].get('loc') or {}).get('line') or 0))
        for idx, (f, lst) in enumerate(items):
            fp = hashlib.sha256(json.dumps([*key, idx], ensure_ascii=True).encode()).hexdigest()
            out.append((fp, f, lst))
    return out


def input_key(input_path: str | Path, baseline: str | Path) -> str:
    """The per-entry 'input' of a baseline: the input file relative to the baseline's directory
    (POSIX). The same file gets the same key from any working directory."""
    return Path(os.path.relpath(Path(input_path).resolve(), Path(baseline).resolve().parent)).as_posix()


def _load_doc(path: Path, missing_ok: bool = False) -> dict:
    """The baseline document; ValueError when it is not one (an empty or, with missing_ok, an
    absent file is an empty baseline)."""
    try:
        text = Path(path).read_text(encoding='utf-8')
    except FileNotFoundError:
        if missing_ok:
            return {'schema': SCHEMA, 'entries': []}
        raise ValueError(f'baseline file not found: {path}') from None
    except OSError as ex:
        raise ValueError(f'baseline {path}: {ex}') from ex
    if missing_ok and not text.strip():
        return {'schema': SCHEMA, 'entries': []}
    try:
        data = json.loads(text)
    except ValueError as ex:
        raise ValueError(f'baseline {path}: {ex}') from ex
    if not isinstance(data, dict) or data.get('schema') != SCHEMA or not isinstance(data.get('entries'), list):
        raise ValueError(f'baseline {path}: not a {SCHEMA} file')
    return data


def _owner(e: dict, doc: dict, path: Path, here: Path) -> str | None:
    """Which input file baseline entry `e` belongs to, as a resolved path string, or None.

    Per-entry 'input' (written since 0.2.0, and by `radisa check-all`) is relative to the baseline's
    directory. An entry without one belongs to the document's top-level 'input' (a single-file
    baseline, as given on that command line: tried against the working directory, the baseline's
    directory and `here`'s directory); when that names no existing file but has `here`'s basename
    (the checkout moved), it is taken as `here`."""
    bdir = Path(path).resolve().parent
    if e.get('input'):
        return str((bdir / e['input']).resolve())
    top = doc.get('input')
    if not top:
        return None
    for c in (Path(top), Path.cwd() / top, bdir / top, here.parent / Path(top).name):
        try:
            if c.is_file():
                return str(c.resolve())
        except OSError:
            continue
    return str(here) if Path(top).name == here.name else None


def read_baseline(path: Path, input_path: str | Path | None = None) -> dict[str, dict]:
    """{fingerprint: entry} of baseline `path` that apply to `input_path` (every entry when None).

    A baseline holds entries of any number of input files (see `_owner`): fingerprints carry no
    file name, so an entry recorded for one file must never hide a finding in another."""
    doc = _load_doc(Path(path))
    here = Path(input_path).resolve() if input_path else None
    out = {}
    for e in doc['entries']:
        if not isinstance(e, dict) or 'fingerprint' not in e:
            continue
        if here is None or _owner(e, doc, Path(path), here) == str(here):
            out[e['fingerprint']] = e
    return out


@contextlib.contextmanager
def _locked(directory: Path):
    """An exclusive flock on `directory` (POSIX), so concurrent `--write-baseline` runs into one
    file merge instead of losing each other's entries. Without fcntl (Windows) no lock is taken."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)   # releases the lock


def write_baseline(path: Path, entries: list[tuple[str, dict, list]], report: dict,
                   kernels: set[str] | None = None) -> int:
    """Record `entries` (from `fingerprints`) as the baseline of report['input'] in `path`.

    Merges: entries of other input files already in `path` are kept, this input's are replaced
    (with `kernels`, a --kernel run: only those of the kernels in `kernels`). The file is replaced
    atomically under a directory lock, so readers never see half a file and concurrent writers
    (one per input) all land. ValueError when `path` exists and is not a baseline (it is not
    overwritten). Returns the number of entries in the file."""
    path = Path(path)
    bdir = path.resolve().parent
    here = Path(report['input']).resolve() if report.get('input') else None
    key = input_key(here, path) if here else None
    rows = [{'fingerprint': fp, 'rule': f['rule'], 'severity': f['severity'], 'kernel': f.get('_kernel', ''),
             'instruction': f.get('instruction', ''), 'message': f['message'][:200],
             **({'input': key} if key else {})} for fp, f, _ in entries]
    with _locked(bdir):
        doc = _load_doc(path, missing_ok=True)
        for e in doc['entries']:
            if not isinstance(e, dict) or 'fingerprint' not in e:
                continue
            owner = _owner(e, doc, path, here) if here else None
            if here is not None and owner == str(here):
                if kernels is None or e.get('kernel', '') in kernels or e.get('kernel', '') == '':
                    continue   # replaced by this run
            if owner is not None and not e.get('input'):
                e = {**e, 'input': Path(os.path.relpath(owner, bdir)).as_posix()}
            if owner is None and not e.get('input'):
                continue       # belongs to no file that can be named any more
            rows.append(e)
        rows.sort(key=lambda r: (r.get('input', ''), r['rule'], r.get('kernel', ''), r['fingerprint']))
        inputs = sorted({r['input'] for r in rows if r.get('input')})
        out = {'schema': SCHEMA, 'radisa': __version__,
               'input': report.get('input') if inputs == ([key] if key else []) else None,
               'inputs': inputs, 'entries': rows}
        fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=bdir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps(out, indent=2, sort_keys=True) + '\n')
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    return len(rows)


# ------------------------------------------------------------------ the report filter

def report_filter(report: dict, args) -> None:
    """check.REPORT_FILTERS entry: configuration, inline suppression, baseline (module docstring).

    Adds report['suppressed'], report['baselined'] (lists of findings, with 'kernel' and
    'suppression') and report['suppression'] (summary counts); hygiene findings go to
    report['findings']."""
    s = settings(args)
    partial = bool(getattr(args, 'kernel', None))
    summary = {'config': str(s['config']) if s['config'] else None,
               'baseline': str(s['baseline']) if s['baseline'] else None,
               'disabled': Counter(), 'regraded': Counter(), 'suppressed': 0, 'baselined': 0,
               'suppressed_errors': Counter(), 'config_errors': Counter()}
    suppressed, baselined = report.setdefault('suppressed', []), report.setdefault('baselined', [])
    new: list[dict] = []

    # 1. configuration
    seen = set()
    for _, lst in _all_lists(report):
        keep = []
        for f in lst:
            seen.add(f['rule'])
            if not _protected(f):
                if f['rule'] in s['disable']:
                    summary['disabled'][f['rule']] += 1
                    if f['severity'] == 'error':
                        summary['config_errors'][f['rule']] += 1
                    continue
                level = s['severity'].get(f['rule'])
                if level and level != f['severity']:
                    f.setdefault('default_severity', f['severity'])
                    f['severity'] = level
                    summary['regraded'][f['rule']] += 1
            keep.append(f)
        lst[:] = keep
    unknown = sorted(r for r in s['named'] if r not in RULES and r not in seen)
    if unknown:
        new.append(_hygiene('CONFIG_UNKNOWN_RULE', 'the configuration names rule(s) radisa does not know: '
                            + ', '.join(unknown) + ' (see `radisa rules`)', None, s, rules=unknown))

    # 2. inline suppression
    files = _Files(report.get('input'))
    directives: dict[str, list[dict]] = {}

    def scan(key):
        if key is not None and key not in directives:
            text = files.lines(key)
            directives[key] = parse_directives(text, files.display.get(key, key)) if text is not None else []
        return directives.get(key, [])

    input_file = None
    if files.input and files.input.suffix in TEXT_SUFFIXES and files.input.is_file():
        input_file = str(files.input)
        scan(input_file)
    for kernel, lst in _all_lists(report):
        keep = []
        for f in lst:
            hit = None
            if not _protected(f) and f['rule'] not in HYGIENE:
                key = files.resolve((f.get('loc') or {}).get('file')) if f.get('loc') else input_file
                line = (f.get('loc') or {}).get('line')
                for d in scan(key):
                    if not d['reason'] or f['rule'] not in d['rules']:
                        continue
                    if d['scope'] == 'file' or (line is not None and line in d['covers']):
                        hit = d
                        break
            if hit is None:
                keep.append(f)
                continue
            hit['used'].add(f['rule'])
            suppressed.append({**f, 'kernel': kernel, 'suppression': {
                'kind': 'inline' if hit['scope'] == 'line' else 'file', 'reason': hit['reason'],
                'file': hit['file'], 'line': hit['line']}})
            summary['suppressed'] += 1
            if f.get('default_severity', f['severity']) == 'error':
                summary['suppressed_errors'][f['rule']] += 1
        lst[:] = keep
    # Findings left in the input file that carry no line: a line-scoped directive cannot match them.
    unlocated = Counter(f['rule'] for _, lst in _all_lists(report) for f in lst
                        if not f.get('loc') and not _protected(f) and f['rule'] not in HYGIENE)
    for key, ds in directives.items():
        for d in ds:
            where = 'ignore-file' if d['scope'] == 'file' else 'ignore'
            if not d['reason']:
                new.append(_hygiene('SUPPRESSION_NO_REASON', f"radisa: {where}[{','.join(d['rules'])}] has no "
                                    'reason; it suppresses nothing. Write why the finding does not apply after '
                                    'the closing bracket', d, s, rules=d['rules']))
                continue
            # Only the checked file's own directives: one in a header may be used by another
            # translation unit or instantiation, which this run does not see.
            if partial or key != input_file:
                continue
            for rule in d['rules']:
                if rule in d['used'] or rule in s['disable']:
                    continue
                if d['scope'] == 'file':
                    scope = 'in this file'
                elif not d['comment_only']:
                    scope = f"on line {d['line']}"
                elif d['target'] is not None:
                    scope = f"on line {d['target']} (the code line this comment applies to)"
                else:
                    scope = f"on line {d['line']} (no code line follows this comment directly)"
                if rule not in RULES:
                    text = (f'radisa: {where}[{rule}] suppresses nothing: {rule} is not a rule radisa knows '
                            '(see `radisa rules`); remove or correct it')
                elif d['scope'] == 'line' and unlocated[rule]:
                    text = (f'radisa: {where}[{rule}] matched nothing {scope}, but {unlocated[rule]} {rule} '
                            'finding(s) of this file carry no source line (machine code that could not be mapped '
                            'to lines of this file), so a line-scoped ignore cannot match them. Use '
                            f'ignore-file[{rule}] <reason> if they are all accepted')
                else:
                    text = f'radisa: {where}[{rule}] suppresses nothing: no {rule} finding {scope}; remove it'
                new.append(_hygiene('UNUSED_SUPPRESSION', text, d, s, suppression_rule=rule))
    report['findings'] += [f for f in new if f]

    # 3. baseline
    for kernel, lst in _all_lists(report):
        for f in lst:
            f['_kernel'] = kernel
    try:
        entries = fingerprints(report, files)
        target = s['baseline']
        if getattr(args, 'write_baseline', None):
            write_baseline(Path(args.write_baseline), entries, report,
                           {k.get('name') or '' for k in report.get('kernels') or []} if partial else None)
            target = Path(args.write_baseline)
        if target is not None:
            known = read_baseline(target, report.get('input') or '')
            summary['baseline'] = str(target)
            hidden = set()
            matched = set()
            for fp, f, lst in entries:
                if fp in known:
                    matched.add(fp)
                    hidden.add(id(f))
                    baselined.append({**{k: v for k, v in f.items() if k != '_kernel'}, 'kernel': f['_kernel'],
                                      'suppression': {'kind': 'baseline', 'file': str(target), 'fingerprint': fp}})
                    if f.get('default_severity', f['severity']) == 'error':
                        summary['suppressed_errors'][f['rule']] += 1
            for _, lst in _all_lists(report):
                lst[:] = [f for f in lst if id(f) not in hidden]
            summary['baselined'] = len(hidden)
            stale = [known[fp] for fp in known if fp not in matched]
            if stale and not partial:
                rules = Counter(e.get('rule', '?') for e in stale)
                f = _hygiene('BASELINE_STALE', f'{len(stale)} baseline entr{"y" if len(stale) == 1 else "ies"} no '
                             'longer occur (' + ', '.join(f'{r} x{n}' for r, n in sorted(rules.items())) +
                             f'); fixed or changed. Regenerate with --write-baseline {target}', None, s,
                             stale=sorted(e['fingerprint'] for e in stale))
                if f:
                    report['findings'].append(f)
    finally:
        for _, lst in _all_lists(report):
            for f in lst:
                f.pop('_kernel', None)
    # Errors that stay in the report re-graded below error by the configuration are hidden too.
    for _, lst in _all_lists(report):
        for f in lst:
            if f.get('default_severity') == 'error' and f['severity'] != 'error':
                summary['config_errors'][f['rule']] += 1
    report['suppression'] = {**summary, 'disabled': dict(summary['disabled']), 'regraded': dict(summary['regraded']),
                             'suppressed_errors': dict(summary['suppressed_errors']),
                             'config_errors': dict(summary['config_errors'])}


def summary_suffix(report: dict) -> str:
    """Text to append to the `radisa check:` summary line ('' when nothing was hidden or re-graded).

    Every error-severity finding that is not shown as an error, whatever hid it (inline or file
    suppression, the baseline, --disable / "off", a lower --severity), is named in
    "(hidden errors: RULE xN, ...)"."""
    s = report.get('suppression') or {}
    parts = []
    if s.get('suppressed'):
        parts.append(f"{s['suppressed']} suppressed inline")
    if s.get('baselined'):
        parts.append(f"{s['baselined']} in baseline")
    if s.get('disabled'):
        parts.append(f"{sum(s['disabled'].values())} disabled by configuration")
    if s.get('regraded'):
        parts.append(f"{sum(s['regraded'].values())} re-graded by configuration")
    if not parts:
        return ''
    text = '; ' + ', '.join(parts)
    errs = Counter(s.get('suppressed_errors') or {}) + Counter(s.get('config_errors') or {})
    if errs:
        text += ' (hidden errors: ' + ', '.join(f'{r} x{n}' for r, n in sorted(errs.items())) + ')'
    return text


def hidden_findings(report: dict) -> list[dict]:
    """The findings this filter took out of the report by suppression or baseline (not the
    disabled ones, which are only counted), each with 'kernel' and 'suppression'
    {'kind': 'inline'|'file'|'baseline', 'reason' or 'fingerprint', 'file', ...}. For output
    formats that list suppressed results (SARIF `suppressions`, see `sarif_suppression`)."""
    return list(report.get('suppressed') or []) + list(report.get('baselined') or [])


def sarif_suppression(f: dict) -> dict:
    """The SARIF 2.1.0 `suppression` object (§3.35) of a finding from `hidden_findings`:
    kind 'inSource' for a radisa: ignore comment, 'external' for the baseline file."""
    sup = f.get('suppression') or {}
    if sup.get('kind') == 'baseline':
        return {'kind': 'external', 'status': 'accepted',
                'justification': f"recorded in baseline {sup.get('file')}",
                'properties': {'fingerprint': sup.get('fingerprint')}}
    out = {'kind': 'inSource', 'status': 'accepted', 'justification': sup.get('reason') or ''}
    if sup.get('file') and sup.get('line'):
        out['location'] = {'physicalLocation': {'artifactLocation': {'uri': str(sup['file'])},
                                                'region': {'startLine': int(sup['line'])}}}
    return out


# ------------------------------------------------------------------ radisa rules

LABEL = {'error': 'error', 'review': 'warning', 'info': 'note', 'incomplete': 'incomplete'}


def add_rules_parser(sub) -> None:
    """`radisa rules [--explain RULE] [--json]`."""
    r = sub.add_parser('rules', help='list every rule radisa check can report')
    r.add_argument('--explain', metavar='RULE', help='show one rule in full')
    r.add_argument('--json', action='store_true', help='print the registry as JSON')
    r.set_defaults(fn=rules_command)


def explain(rule: str) -> str:
    e = RULES[rule]
    flags = [x for x in ('opt_in', 'memory_safety') if e.get(x)]
    lines = [f'{rule}', f"  default severity: {LABEL.get(e['severity'], e['severity'])}"
             + (' (off unless enabled)' if e.get('opt_in') else ''),
             f"  summary: {e['summary']}", f"  basis: {e['basis']}", f"  since: {e['since']}"]
    if e.get('tool'):
        lines.append(f"  reported by: radisa {e['tool']}")
    if flags:
        lines.append('  flags: ' + ', '.join(flags))
    if rule in PROTECTED or e['severity'] == 'incomplete':
        lines.append('  configure: cannot be re-graded, disabled, suppressed or baselined (it means part of '
                     'the input was not analysed)')
    elif rule in HYGIENE:
        lines.append(f'  configure: --severity {rule}=error|warning|note|off, --disable {rule}')
    else:
        lines.append(f'  configure: --severity {rule}=error|warning|note|off, --disable {rule}; '
                     f'suppress: radisa: ignore[{rule}] <reason>')
    return '\n'.join(lines)


def rules_command(args) -> int:
    rule = (getattr(args, 'explain', None) or '').strip().upper()
    if rule:
        if rule not in RULES:
            close = [r for r in RULES if rule in r]
            print(f'unknown rule {rule}' + (f"; did you mean {', '.join(close)}" if close else ''))
            return 2
        if getattr(args, 'json', False):
            print(json.dumps({rule: RULES[rule]}, indent=2, sort_keys=True))
        else:
            print(explain(rule))
        return 0
    if getattr(args, 'json', False):
        print(json.dumps(RULES, indent=2, sort_keys=True))
        return 0
    width = max(map(len, RULES))
    for r in sorted(RULES):
        e = RULES[r]
        sev = LABEL.get(e['severity'], e['severity']) + ('*' if e.get('opt_in') else '')
        tool = ' [audit]' if e.get('tool') == 'audit' else ''
        print(f'{r:<{width}}  {sev:<10}  {e["summary"]}{tool}')
    print(f'{len(RULES)} rules; * = off unless enabled. radisa rules --explain RULE for the basis.')
    return 0
