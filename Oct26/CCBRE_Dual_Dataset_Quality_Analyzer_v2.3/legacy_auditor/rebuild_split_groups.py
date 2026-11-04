#!/usr/bin/env python3
"""Recompute STRICT/EXTENDED October groups from v1.6 manifests with a
post-hoc dHash threshold (0..12), without re-decoding any source images.
Does NOT alter the existing analysis results or Raw directory.
"""
import argparse
import csv
import json
from pathlib import Path
from analyze_oct26 import make_strict_groups, export_split_groups, DHASH_REVIEW_MAX_DISTANCE


def read(path):
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser(description='Rebuild v1.6 split groups from saved dHash candidates')
    ap.add_argument('--analysis',required=True,type=Path,help='Completed Analysis_v1.6 directory')
    ap.add_argument('--dhash-threshold',required=True,type=int,help='0..12 inclusive')
    ap.add_argument('--output',type=Path,help='Fresh output dir (default Analysis/SplitGroups_dhashNN)')
    args=ap.parse_args()
    if args.dhash_threshold<0 or args.dhash_threshold>DHASH_REVIEW_MAX_DISTANCE:
        ap.error('threshold must lie between 0 and 12')
    analysis=args.analysis.resolve()
    output=(args.output or (analysis/f'SplitGroups_dhash{args.dhash_threshold:02d}')).resolve()
    if output==analysis:ap.error('output must differ from completed analysis')
    if output.exists() and any(output.iterdir()):ap.error('refusing to overwrite nonempty output')
    overview=json.loads((analysis/'01_overview.json').read_text(encoding='utf-8'))
    if str(overview.get('analyzer_version','')).split('.')[0:2]!=['1','6'] and str(overview.get('analyzer_version','')).split('.')[0:2]!=['1','7']:
        ap.error('Only Analysis_v1.6 or v1.7 outputs are supported')
    if overview.get('dhash_candidate_scan',{}).get('review_rows_truncated'):
        ap.error('Candidate CSV was truncated: post-hoc extended grouping would be incomplete')
    images=read(analysis/'02_image_manifest.csv')
    candidates=read(analysis/'23_dhash_candidates.csv')
    group,indices=make_strict_groups(images)
    extended=group.clone()
    selected=0
    for row in candidates:
        if int(row['dhash_hamming'])>args.dhash_threshold:continue
        if row['image_a'] not in indices or row['image_b'] not in indices:
            raise SystemExit('dHash CSV images missing from image manifest')
        extended.union(indices[row['image_a']],indices[row['image_b']]);selected+=1
    output.mkdir(parents=True,exist_ok=True)
    summary=export_split_groups(images,group,extended,output)
    summary.update(threshold=args.dhash_threshold,selected_representative_pairs=selected,
        source_analysis=str(analysis),n_images=len(images),
        status='REVIEW_ONLY_EXTENDED_GROUPS',
        warning='Extended dHash links must not be treated as verified matches or automatic split boundaries.')
    (output/'rebuild_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(f'Post-hoc groups at cutoff {args.dhash_threshold}: {summary["strict_group_count"]} strict / {summary["extended_group_count"]} extended. Output: {output}')

if __name__=='__main__':main()
