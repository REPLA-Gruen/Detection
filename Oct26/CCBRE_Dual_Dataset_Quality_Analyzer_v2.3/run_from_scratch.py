"""Run the standalone July+October audit and create human-oriented problem evidence.

All raw inputs are read-only. Never silently accept an old result: by default the
technical core is executed anew and the new evidence is constructed from its CSVs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import json
import hashlib

ROOT=Path(__file__).resolve().parent
CORE=ROOT/'legacy_auditor'/'analyze_oct26.py'
EVIDENCE=ROOT/'quality_evidence.py'

def run(command):
    print('\n> '+' '.join(str(i) for i in command),flush=True)
    subprocess.run(command,check=True,cwd=str(ROOT))


def inputs(p):
    for attr in ('july_images','july_labels','october_root'):
        v=Path(getattr(p,attr)).resolve()
        if not v.is_dir():raise SystemExit(f'Expected directory missing: --{attr.replace("_","-")} {v}')
    v=Path(p.july_yaml)
    if not v.is_file():raise SystemExit(f'July class YAML missing: {v}')
    for context in ('biowaste-collection-images','biowaste-floorspread-images'):
        c=Path(p.october_root)/context
        if not c.is_dir():raise SystemExit(f'October source folder missing: {c}')
        for child in ('images','labels','labels-detect','data.yaml','report.md','.sync-state.json'):
            if not (c/child).exists():raise SystemExit(f'October required source item missing: {c/child}')
    if Path(p.october_root).resolve().is_relative_to(Path(p.output).resolve()):
        raise SystemExit('Analysis output cannot be an ancestor of the October source directory.')
    if Path(p.july_images).resolve().is_relative_to(Path(p.output).resolve()):
        raise SystemExit('Analysis output cannot be an ancestor of the July source images.')
    if Path(p.output).resolve().is_relative_to(Path(p.october_root).resolve()):
        raise SystemExit('Do not put analysis output inside original October Raw source directory.')
    if Path(p.output).resolve().is_relative_to(Path(p.july_images).resolve()):
        raise SystemExit('Do not put analysis output inside original July photographs directory.')


def main():
    parser=argparse.ArgumentParser(description='CCBRE dual-dataset QA v2.2; run original July fingerprinting, full October+July label reconciliation, and human-readable problem evidence.')
    parser.add_argument('--july-images',required=True)
    parser.add_argument('--july-labels',required=True)
    parser.add_argument('--july-yaml',required=True)
    parser.add_argument('--october-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference-class-map',default='',help='Optional explicit two-column class equivalence CSV. Leave Diverses unresolved until partner decision.')
    parser.add_argument('--samples-per-problem',type=int,default=20)
    parser.add_argument('--all-duplicate-sheets',action='store_true')
    parser.add_argument('--evidence-only',action='store_true',help='For re-rendering after a successful v2.2 run. Uses THIS RUN technical outputs, not an external historic audit.')
    a=parser.parse_args();inputs(a)
    out=Path(a.output).resolve()
    if a.samples_per_problem<1 or a.samples_per_problem>200:raise SystemExit('samples-per-problem must be 1..200')
    ref=out/'99_Technical_Verification'/'July_Fingerprints'
    core=out/'99_Technical_Verification'/'Full_Dataset_Audit_v171'
    if not a.evidence_only:
        if (ref/'31_fingerprint_manifest.csv').exists() or (core/'01_overview.json').exists():
            raise SystemExit('Output contains previous results. Use a NEW output folder for a from-scratch run; or --evidence-only for rendering on unchanged inputs.')
        ref.mkdir(parents=True,exist_ok=True)
        run([sys.executable,str(CORE),'--fingerprint-only','--images-root',str(Path(a.july_images).resolve()),
             '--output',str(ref),'--recursive'])
        # Do not assume that the July image folder contains exactly 3359 originals:
        # warn in evidence, and refuse obviously derived multi-copy inputs.
        import csv
        with (ref/'31_fingerprint_manifest.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        if not rows:raise SystemExit('No July image fingerprints created')
        stems=[Path(r['image']).stem.casefold() for r in rows]
        if len(stems)!=len(set(stems)):
            raise SystemExit('July image folder contains repeated stems; choose authoritative original image folder without training/derived copies.')
        run_args=[sys.executable,str(CORE),'--dataset',str(Path(a.october_root).resolve()),
                  '--output',str(core),'--reference-manifest',str(ref/'31_fingerprint_manifest.csv'),
                  '--reference-labels',str(Path(a.july_labels).resolve()),
                  '--reference-yaml',str(Path(a.july_yaml).resolve()),
                  '--dhash-threshold','3']
        if a.reference_class_map:run_args+=['--reference-class-map',str(Path(a.reference_class_map).resolve())]
        run(run_args)
    else:
        for p in (ref/'31_fingerprint_manifest.csv',core/'01_overview.json',core/'39_reference_label_diff_summary.json'):
            if not p.exists():raise SystemExit(f'Evidence-only requires earlier outputs from this standalone analyzer: {p}')
    argv=[sys.executable,str(EVIDENCE),'--july-images',str(Path(a.july_images).resolve()),
        '--july-labels',str(Path(a.july_labels).resolve()),'--july-yaml',str(Path(a.july_yaml).resolve()),
        '--october-root',str(Path(a.october_root).resolve()),'--audit-dir',str(core),'--output',str(out),
        '--samples-per-problem',str(a.samples_per_problem)]
    if a.all_duplicate_sheets:argv.append('--all-duplicate-sheets')
    run(argv)
    run([sys.executable,str(ROOT/'general_statistics_v22.py'),'--analysis',str(out)])
    run([sys.executable,str(ROOT/'html_report.py'),'--analysis',str(out)])
    print('\nDONE. Start with:',out/'OPEN_FIRST.html',flush=True)
    print('Partner review decisions:',out/'09_Provider_Corrective_Actions'/'Corrective_Action_Register.csv',flush=True)

if __name__=='__main__':main()
