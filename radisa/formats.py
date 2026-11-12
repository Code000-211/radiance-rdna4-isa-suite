"""Machine-readable renderings of `radisa check` reports: SARIF 2.1.0, GitHub workflow commands.

Normative basis
  SARIF   OASIS "Static Analysis Results Interchange Format (SARIF) Version 2.1.0" (Errata 01):
          sarifLog requires `version` and `runs`; run requires `tool.driver.name`; result requires
          `message`; reportingDescriptor requires `id`; region lines/columns are 1-based.
  GitHub  "Workflow commands for GitHub Actions": `::error file=F,line=L,col=C,title=T::MSG`;
          data escapes % CR LF, property values additionally `:` and `,`.

Input: report dicts as returned by `check.analyze` (one per checked file).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import re

from . import __version__

SARIF_SCHEMA = 'https://json.schemastore.org/sarif-2.1.0.json'
SARIF_LEVEL = {'error': 'error', 'review': 'warning', 'incomplete': 'note', 'info': 'note'}
GITHUB_LEVEL = {'error': 'error', 'review': 'warning', 'incomplete': 'notice', 'info': 'notice'}
FORMATS = ('text', 'json', 'sarif', 'github')
FINGERPRINT_KEY = 'radisaFinding/v1'


def iter_findings(report: dict):
    """(kernel dict or None, finding) for the report-level findings, then every kernel's."""
    for f in report.get('findings', []):
        yield None, f
    for k in report.get('kernels', []):
        for f in k.get('findings', []):
            yield k, f


def _uri(file: str, base: Path | None) -> str:
    """Repository-relative POSIX path when `file` lies under `base`, else a file:// URI (absolute)
    or the path as given (relative)."""
    p = Path(file)
    if base is not None:
        try:
            return (p if p.is_absolute() else (Path.cwd() / p)).resolve().relative_to(base.resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return p.resolve().as_uri() if p.is_absolute() else p.as_posix()


_LINE_REF = re.compile(r'(:\d+)+\b|\b0x[0-9a-f]+\b|\bpc \d+\b', re.I)


def fingerprint(report: dict, kernel: dict | None, f: dict) -> str:
    """Stable across line moves: rule, input name, kernel, instruction text and the message with
    line numbers / pcs removed. pc and loc.line are deliberately left out."""
    msg = _LINE_REF.sub('', f.get('message', ''))
    key = json.dumps([f.get('rule'), Path(str(report.get('input', ''))).name,
                      (kernel or {}).get('name'), f.get('instruction', ''), msg])
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _rule_text(rule: str, findings: list[dict]) -> str:
    srcs = sorted({f['source'] for f in findings if f.get('source')})
    if srcs:
        return '; '.join(srcs[:4]) + (f' (+{len(srcs) - 4} more)' if len(srcs) > 4 else '')
    return f'radisa check rule {rule}'


def to_sarif(reports: list[dict], base: str | Path | None = None) -> dict:
    """One SARIF 2.1.0 log with one run covering all `reports`.

    `base` (default: current directory) is the root that artifact URIs are made relative to,
    so GitHub code scanning can map them onto the repository."""
    base = Path(base) if base is not None else Path.cwd()
    by_rule: dict[str, list[dict]] = {}
    for r in reports:
        for _, f in iter_findings(r):
            by_rule.setdefault(f['rule'], []).append(f)
    rule_ids = sorted(by_rule)
    rules = []
    for rid in rule_ids:
        sev = Counter(f['severity'] for f in by_rule[rid]).most_common(1)[0][0]
        text = _rule_text(rid, by_rule[rid])
        rules.append({'id': rid, 'name': rid.title().replace('_', ''),
                      'shortDescription': {'text': rid},
                      'fullDescription': {'text': text},
                      'defaultConfiguration': {'level': SARIF_LEVEL.get(sev, 'note')}})
    index = {rid: i for i, rid in enumerate(rule_ids)}
    results, artifacts, art_index = [], [], {}

    def artifact(uri: str) -> int:
        if uri not in art_index:
            art_index[uri] = len(artifacts)
            artifacts.append({'location': {'uri': uri}})
        return art_index[uri]

    for r in reports:
        seen: Counter = Counter()
        input_uri = _uri(str(r.get('input', '')), base)
        for k, f in iter_findings(r):
            res = {'ruleId': f['rule'], 'ruleIndex': index[f['rule']],
                   'level': SARIF_LEVEL.get(f['severity'], 'note'),
                   'message': {'text': f.get('message') or f['rule']}}
            loc = f.get('loc')
            location: dict = {}
            if loc and loc.get('file') and loc.get('line'):
                uri = _uri(loc['file'], base)
                region = {'startLine': max(1, int(loc['line']))}
                if loc.get('column'):
                    region['startColumn'] = max(1, int(loc['column']))
                if f.get('instruction'):
                    region['snippet'] = {'text': f['instruction']}
                location['physicalLocation'] = {'artifactLocation': {'uri': uri, 'index': artifact(uri)},
                                                'region': region}
            elif input_uri:
                location['physicalLocation'] = {'artifactLocation': {'uri': input_uri,
                                                                     'index': artifact(input_uri)}}
            if k is not None:
                name = k.get('name') or k.get('symbol') or '?'
                full = name if f.get('pc') is None else f"{name}+{f['pc']:#x}"
                location['logicalLocations'] = [{'name': name, 'fullyQualifiedName': full, 'kind': 'function'}]
            if location:
                res['locations'] = [location]
            fp = fingerprint(r, k, f)
            seen[fp] += 1
            res['partialFingerprints'] = {FINGERPRINT_KEY: f'{fp}:{seen[fp]}'}
            props = {'severity': f['severity']}
            for key in ('pc', 'instruction', 'source', 'registers', 'producer', 'operand_model'):
                if f.get(key) is not None:
                    props[key] = f[key]
            res['properties'] = json.loads(json.dumps(props, default=_plain))
            results.append(res)
    statuses = [r.get('status') for r in reports]
    run = {'tool': {'driver': {'name': 'radisa check', 'version': str(__version__),
                               'informationUri': 'https://llvm.org/docs/AMDGPUUsage.html',
                               'rules': rules}},
           'artifacts': artifacts,
           'results': results,
           'invocations': [{'executionSuccessful': all(s in ('PASS', 'FAIL', 'NO_KERNELS') for s in statuses),
                            'properties': {'statuses': dict(Counter(statuses))}}],
           'columnKind': 'unicodeCodePoints',
           'properties': {'files': [{'input': r.get('input'), 'status': r.get('status'),
                                     'exit_code': r.get('exit_code')} for r in reports]}}
    if not artifacts:
        del run['artifacts']
    return {'$schema': SARIF_SCHEMA, 'version': '2.1.0', 'runs': [run]}


def _plain(x):
    if isinstance(x, (set, frozenset)):
        return sorted(x, key=str)
    return str(x)


def gh_data(s: str) -> str:
    """Escape a workflow-command message."""
    return s.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')


def gh_prop(s: str) -> str:
    """Escape a workflow-command property value."""
    return gh_data(s).replace(':', '%3A').replace(',', '%2C')


def to_github(reports: list[dict], base: str | Path | None = None) -> str:
    """One `::error` / `::warning` / `::notice` workflow command per finding."""
    base = Path(base) if base is not None else Path.cwd()
    out = []
    for r in reports:
        for k, f in iter_findings(r):
            level = GITHUB_LEVEL.get(f['severity'], 'notice')
            loc = f.get('loc') or {}
            props = []
            if loc.get('file'):
                props.append(f"file={gh_prop(_uri(loc['file'], base))}")
                if loc.get('line'):
                    props.append(f"line={int(loc['line'])}")
                if loc.get('column'):
                    props.append(f"col={int(loc['column'])}")
            elif r.get('input'):
                props.append(f"file={gh_prop(_uri(str(r['input']), base))}")
            props.append(f"title={gh_prop(f['rule'])}")
            msg = f.get('message') or f['rule']
            if k is not None:
                where = k.get('name') or '?'
                if f.get('pc') is not None and not loc:
                    where += f"+{f['pc']:#x}"
                msg = f'[{where}] {msg}'
            if f.get('instruction'):
                msg += f"\n    {f['instruction']}"
            if f.get('source'):
                msg += f"\n    see {f['source']}"
            out.append(f"::{level} {','.join(props)}::{gh_data(msg)}")
    return '\n'.join(out) + ('\n' if out else '')


def emit(reports: list[dict], fmt: str, strict: bool = False, base: str | Path | None = None) -> str:
    """`reports` rendered as `fmt` (one of FORMATS). text: check.render per report; json: the
    report itself for one report, else a list."""
    if fmt == 'text':
        from .check import render
        return '\n'.join(render(r, strict) for r in reports) + '\n'
    if fmt == 'json':
        value = reports[0] if len(reports) == 1 else reports
        return json.dumps(value, indent=2, sort_keys=True, default=_plain) + '\n'
    if fmt == 'sarif':
        return json.dumps(to_sarif(reports, base), indent=2) + '\n'
    if fmt == 'github':
        return to_github(reports, base)
    raise ValueError(f'unknown format {fmt!r} (choose from {", ".join(FORMATS)})')
