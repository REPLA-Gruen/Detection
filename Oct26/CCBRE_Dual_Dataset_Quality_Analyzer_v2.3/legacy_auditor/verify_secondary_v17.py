"""Independent read-only secondary checks of v1.7 July/October tables.

Reads only CSV outputs 02, 32, 37, 40, 42 and 44. No raw images,
July labels, or original analyzer imports are required. The selection
rules are specified so interpretation is reproducible:
 - earliest/latest: annotation_updated_utc, with oct_image tie-break
 - most-boxes: maximum oct_boxes, with timestamp/name tie-break
 - fewer/equal/more comparisons: ONLY July images with july_mapped > 0
 - per-image image-state checks: ALL alternative October annotations
 - group novelty: matching July content by file or decoded-pixel identity
 - pooled IoU>=0.9: all pairs in file 42, including repeated variants.
"""
import csv, json, argparse, collections, pathlib, sys

def load(path):
    with open(path, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def n(v):return int(v)

def analyze(root):
    july=load(root/'40_reference_label_diff_per_image.csv')
    variants=load(root/'44_reference_label_comparison_by_oct_image.csv')
    matches=load(root/'42_reference_label_object_matches.csv')
    groups=load(root/'37_day_level_split_groups.csv')
    strong=load(root/'32_reference_strong_matches.csv')
    images=load(root/'02_image_manifest.csv')
    by=collections.defaultdict(list)
    for row in variants:by[row['ref_image']].append(row)
    assert set(by)==set(r['ref_image'] for r in july), 'Reference image coverage differs'

    def selection(mode):
        result=[]
        for ref, options in by.items():
            if mode=='earliest': row=min(options,key=lambda r:(r['annotation_updated_utc'] or '9999',r['oct_image']))
            elif mode=='latest':row=max(options,key=lambda r:(r['annotation_updated_utc'] or '',r['oct_image']))
            elif mode=='most': row=max(options,key=lambda r:(n(r['oct_boxes']),r['annotation_updated_utc'] or '',r['oct_image']))
            elif mode=='least':row=min(options,key=lambda r:(n(r['oct_boxes']),r['annotation_updated_utc'] or '',r['oct_image']))
            result.append(row)
        selected_positive=[r for r in result if n(r['july_mapped'])>0]
        fewer=[r for r in selected_positive if n(r['oct_boxes'])<n(r['july_mapped'])]
        return dict(boxes=sum(n(r['oct_boxes']) for r in result),
                    class_aware_iou05_matches=sum(n(r['same_class_iou05_matched']) for r in result),
                    july_positive_fewer=sum(n(r['oct_boxes'])<n(r['july_mapped']) for r in selected_positive),
                    july_positive_equal=sum(n(r['oct_boxes'])==n(r['july_mapped']) for r in selected_positive),
                    july_positive_more=sum(n(r['oct_boxes'])>n(r['july_mapped']) for r in selected_positive),
                    july_positive_box_share_in_fewer=sum(n(r['july_mapped']) for r in fewer)/sum(n(r['july_mapped']) for r in selected_positive))
    sel={mode:selection(mode) for mode in ('earliest','latest','most','least')}

    unmapped_only=[r for r in july if n(r['july_mapped'])==0 and n(r['july_unmapped'])>0]
    original_empty=[r for r in july if n(r['july_mapped'])==0 and n(r['july_unmapped'])==0]
    july_mapped_pos=[r for r in july if n(r['july_mapped'])>0]
    iu=dict(july_mapped_positive_images=len(july_mapped_pos),
        july_mapped_positive_all_oct_positive=sum(all(n(v['oct_boxes'])>0 for v in by[r['ref_image']]) for r in july_mapped_pos),
        july_mapped_positive_any_oct_empty=sum(any(n(v['oct_boxes'])==0 for v in by[r['ref_image']]) for r in july_mapped_pos),
        july_empty_images=len(original_empty),
        july_empty_any_oct_positive=sum(any(n(v['oct_boxes'])>0 for v in by[r['ref_image']]) for r in original_empty),
        july_empty_all_oct_positive=sum(all(n(v['oct_boxes'])>0 for v in by[r['ref_image']]) for r in original_empty),
        july_empty_oct_positive_only_one_version=sum(any(n(v['oct_boxes'])>0 for v in by[r['ref_image']]) and any(n(v['oct_boxes'])==0 for v in by[r['ref_image']]) for r in original_empty),
        july_unmapped_only_images=len(unmapped_only),
        july_unmapped_only_all_oct_empty=sum(all(n(v['oct_boxes'])==0 for v in by[r['ref_image']]) for r in unmapped_only),
        pool_matches_iou_ge_09=sum(float(r['iou'])>=0.9 for r in matches),
        pool_matches_all=len(matches))

    july_match_images=set(r['oct_image'] for r in strong)
    groupmembers=collections.defaultdict(list)
    for r in groups:groupmembers[r['split_group_day']].append(r)
    group_class_count=collections.Counter()
    group_records=collections.Counter()
    new_only_images=[]
    for members in groupmembers.values():
        if members[0]['context']!='collection':continue
        matches_here=sum(r['image'] in july_match_images for r in members)
        unmatched=len(members)-matches_here
        state='mixed' if matches_here and unmatched else 'july_only' if matches_here else 'new_only'
        group_class_count[state]+=1
        group_records[state]+=unmatched
        if state=='new_only':new_only_images.extend(r['image'] for r in members)
    image_by_name={r['image']:r for r in images}
    gro=dict(collection_group_counts=dict(group_class_count),unmatched_collection_records_by_group=dict(group_records),
             unique_new_only_collection_content=len({image_by_name[x]['sha256'] for x in new_only_images}),
             unmatched_collection_image_records=len([r for r in images if r['context']=='collection' and r['image'] not in july_match_images]))
    return dict(selection_rules=sel,image_status=iu,group_novelty=gro)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',required=True,type=pathlib.Path)
    parser.add_argument('--output-json',type=pathlib.Path,help='Optional independent verification artifact (outside source evidence)')
    parser.add_argument('--expected-snapshot',action='store_true')
    a=parser.parse_args()
    r=analyze(a.analysis)
    check=[]
    def c(label, actual, expected):check.append(dict(name=label,pass_=actual==expected,observed=actual,expected=expected))
    sel=r['selection_rules'];is_=r['image_status'];gr=r['group_novelty']
    c('earliest box total',sel['earliest']['boxes'],3893)
    c('latest box total',sel['latest']['boxes'],3948)
    c('most-boxes total',sel['most']['boxes'],3981)
    c('least-boxes total',sel['least']['boxes'],3860)
    for typ, counts, share in [('earliest',(550,691,9),.724),('most',(536,705,9),.715)]:
        s=sel[typ]
        for k,v in zip(('fewer','equal','more'),counts):c(f'{typ} July-positive {k}',s[f'july_positive_{k}'],v)
        c(f'{typ} fraction of July mapped boxes in fewer-images rounded to 0.1%',round(s['july_positive_box_share_in_fewer']*100,1),round(share*100,1))
    for k,v in {'july_mapped_positive_images':1250,'july_mapped_positive_all_oct_positive':1243,
                'july_mapped_positive_any_oct_empty':7,'july_empty_images':2087,
                'july_empty_any_oct_positive':7,'july_empty_all_oct_positive':1,
                'july_empty_oct_positive_only_one_version':6,
                'july_unmapped_only_images':22,'july_unmapped_only_all_oct_empty':21,
                'pool_matches_iou_ge_09':3602,'pool_matches_all':4152}.items():c(k,is_[k],v)
    for k,v in {'july_only':1508,'new_only':1720,'mixed':1}.items():c('collection groups '+k,gr['collection_group_counts'].get(k,0),v)
    for k,v in {'new_only':3883,'mixed':1}.items():c('unmatched records in '+k,gr['unmatched_collection_records_by_group'].get(k,0),v)
    c('new-only collection distinct contents',gr['unique_new_only_collection_content'],3882)
    c('unmatched collection records',gr['unmatched_collection_image_records'],3884)
    bad=[x for x in check if not x['pass_']]
    summary={'checks':len(check),'passed':len(check)-len(bad),'failed':len(bad),'checks_detail':check,'derived':r}
    for t in check:print(('PASS' if t['pass_'] else 'FAIL'),t['name'],t['observed'],'expected',t['expected'])
    print(f'Secondary verification: {summary["passed"]}/{summary["checks"]} checks passed')
    if a.output_json:
        a.output_json.parent.mkdir(parents=True,exist_ok=True)
        a.output_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return 1 if bad else 0

if __name__=='__main__':sys.exit(main())
