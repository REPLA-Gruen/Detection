"""Read-only independent CSV/JSON reconciliation for CCBRE Oct-26 auditor v1.7.

Does not reread raw images. Does not use original analyzer functions.
Requires only the Python standard library, Python 3.10+.

Usage: py -3 verify_v17_results.py --analysis "C:\\...\\Analysis_v1.7"
Exit 0 means every specified consistency check passed. It does not imply
labels are visually correct or acquisition groups are independently certified.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def rows(path):
    with path.open('r', newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def js(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def number(val):
    return int(val) if str(val).strip() else 0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis',type=Path,required=True)
    ap.add_argument('--expected-snapshot',action='store_true',help='Enforce the verified Oct 2026 snapshot values; omit for a changed S3 snapshot')
    args=ap.parse_args()
    base=args.analysis
    required=['01_overview.json','02_image_manifest.csv','03_class_statistics.csv','04_source_context_statistics.csv',
              '07_quality_issues.csv','08_exact_duplicates.csv','11_object_manifest.csv',
              '20_internal_consistency_checks.json','22_duplicate_conflict_tiers.csv',
              '34_reference_overlap_summary.json','38_day_level_group_summary.json',
              '39_reference_label_diff_summary.json','40_reference_label_diff_per_image.csv',
              '41_reference_class_mapping.csv','42_reference_label_object_matches.csv',
              '43_reference_label_diff_issues.csv','44_reference_label_comparison_by_oct_image.csv',
              '45_reference_class_diff_counts.csv']
    missing=[name for name in required if not (base/name).is_file()]
    if missing:
        print('MISSING: '+', '.join(missing),file=sys.stderr)
        return 2
    o=js(base/'01_overview.json'); ims=rows(base/'02_image_manifest.csv');objs=rows(base/'11_object_manifest.csv')
    cl=rows(base/'03_class_statistics.csv');src=rows(base/'04_source_context_statistics.csv')
    dup=rows(base/'08_exact_duplicates.csv');issues=rows(base/'07_quality_issues.csv')
    tiers=rows(base/'22_duplicate_conflict_tiers.csv'); overlap=js(base/'34_reference_overlap_summary.json')
    day=js(base/'38_day_level_group_summary.json'); lab=js(base/'39_reference_label_diff_summary.json')
    july=rows(base/'40_reference_label_diff_per_image.csv'); cand=rows(base/'44_reference_label_comparison_by_oct_image.csv')
    matched=rows(base/'42_reference_label_object_matches.csv');diff_issues=rows(base/'43_reference_label_diff_issues.csv')
    mapping=rows(base/'41_reference_class_mapping.csv');classdiff=rows(base/'45_reference_class_diff_counts.csv')
    checks=[]
    def chk(label,found,expected):
        status=found==expected
        checks.append((label,status,found,expected))
    chk('overview analyzer version',o['analyzer_version'] in ('1.7.0','1.7.1'),True)
    chk('overview internal checks',o['internal_consistency_passed'],True)
    chk('overview metadata reconciliation',o['metadata_reconciliation_passed'],True)
    chk('image records',len(ims),o['image_count'])
    chk('unique image identifiers',len(set(r['image'] for r in ims)),len(ims))
    chk('object manifest rows',len(objs),o['detection_object_count'])
    chk('positive images',sum(number(r['detection_instances'])>0 for r in ims),o['positive_images'])
    chk('explicit empty images',sum(r['box_label_state']=='empty' for r in ims),o['explicitly_empty_images'])
    chk('object per-image count',sum(number(r['detection_instances']) for r in ims),len(objs))
    chk('class total (all)',sum(number(r['instances']) for r in cl if r['context']=='all'),len(objs))
    chk('class total (by contexts)',sum(number(r['instances']) for r in cl if r['context']!='all'),len(objs))
    chk('context totals',sum(number(r['image_count']) for r in o['source_contexts']),len(ims))
    chk('all image hashes present',sum(bool(r['sha256']) for r in ims),len(ims))
    chk('all decoded hashes present',sum(bool(r['display_pixel_sha256']) for r in ims),len(ims))
    group_sizes=Counter(r['sha256'] for r in ims)
    chk('exact duplicate groups',sum(v>1 for v in group_sizes.values()),o['duplicate_hash_groups'])
    chk('distinct byte contents',len(group_sizes),len(ims)-sum(v-1 for v in group_sizes.values()))
    chk('duplicate CSV groups',len(dup),o['duplicate_hash_groups'])
    chk('conflicting duplicate CSV groups',sum(r['annotation_conflict'].lower()=='true' for r in dup),o['duplicate_groups_with_conflicting_labels'])
    chk('tier rows',len(tiers),len(dup))
    chk('reference July images',len(july),lab['reference_images'])
    chk('valid July rows',sum(r['label_state']=='ok' for r in july),lab['reference_images_with_valid_labels'])
    for col,key in [('july_rows','july_observed_box_or_polygon_rows'),('july_context','july_context_rows'),
                    ('july_mapped','july_mapped_material_rows'),('july_unmapped','july_unmapped_rows'),
                    ('oct_box_min','observed_oct_box_count_min'),('oct_box_max','observed_oct_box_count_max'),
                    ('same_class_iou05_matched_min','class_aware_iou05_match_min'),
                    ('same_class_iou05_matched_max','class_aware_iou05_match_max')]:
        chk(col,sum(number(r[col]) for r in july),lab[key])
    chk('July context + mapped + unmapped',lab['july_context_rows']+lab['july_mapped_material_rows']+lab['july_unmapped_rows'],lab['july_observed_box_or_polygon_rows'])
    chk('July foreign material counted',lab['july_mapped_material_rows']+lab['july_unmapped_rows'],lab['july_observed_foreign_material_rows'])
    chk('reference identity links',len(cand),overlap['oct_images_with_strong_overlap'])
    chk('July images with versions',sum(number(r['oct_candidate_versions'])>0 for r in july),lab['july_images_with_october_comparison'])
    chk('July multi-version count',sum(number(r['oct_candidate_versions'])>1 for r in july),lab['july_images_with_multiple_october_versions'])
    chk('label issues',len(diff_issues),lab['issue_count'])
    chk('matched object record lower bound',len(matched)>=lab['class_aware_iou05_match_max'],True)
    chk('each match IoU threshold',all(float(r['iou'])>=0.5 for r in matched),True)
    chk('days groups sum',sum(number(k)*number(v) for k,v in day['distribution'].items()),len(ims))
    chk('day group count',sum(number(v) for v in day['distribution'].values()),day['group_count'])
    chk('day max size',max(map(number,day['distribution'].keys())),day['largest_group'])
    chk('floorspread days counted',sum(day['capture_days'].values()),sum(r['context']=='floorspread' for r in ims))
    chk('class mapping rows',len(mapping),len(classdiff))
    chk('class mapping count',sum(r['category']=='mapped' for r in mapping),lab['mapped_reference_class_count'])
    if args.expected_snapshot:
        for label,found,expected in [
            ('verified images',len(ims),9628),('verified objects',len(objs),14087),
            ('verified unique byte images',len(group_sizes),9407),
            ('verified July total annotation rows',lab['july_observed_box_or_polygon_rows'],28168),
            ('verified July foreign object rows',lab['july_observed_foreign_material_rows'],5640),
            ('verified July context boxes',lab['july_context_rows'],22528),
            ('verified July mapped boxes',lab['july_mapped_material_rows'],5427),
            ('verified July unmapped boxes',lab['july_unmapped_rows'],213),
            ('verified July matches low',lab['class_aware_iou05_match_min'],3565),
            ('verified July matches high',lab['class_aware_iou05_match_max'],3708),
            ('verified days groups',day['group_count'],3236),
            ('verified largest day group',day['largest_group'],920),
            ('verified exact duplicate groups',len(dup),221)]:
            chk(label,found,expected)
    problems=[r for r in checks if not r[1]]
    for label,ok,found,expected in checks:
        print(('PASS' if ok else 'FAIL')+f' {label}: {found}'+('' if ok else f' (expected {expected})'))
    print(f'\n{len(checks)-len(problems)}/{len(checks)} checks passed; quality-issue rows: {len(issues)}; July-diff issues: {len(diff_issues)}')
    print('NOTE: Visual annotation quality, site identity, and canonical labels require human/partner confirmation.')
    print('NOTE: Per-class min/max IoU matches need not sum to the global min/max because versions are selected per image.')
    return 1 if problems else 0

if __name__=='__main__':sys.exit(main())
