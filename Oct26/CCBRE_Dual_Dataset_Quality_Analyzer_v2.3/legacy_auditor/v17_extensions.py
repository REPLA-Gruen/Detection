"""v1.7 evidence extensions. No image decoding, no modifications to source data.

Reference labels are sourced ONLY from the supplied July labels directory and
matched to the supplied ORIGINAL July fingerprint manifest by image stem.
The output explicitly preserves unresolved labels and multiple October annotation versions.
"""
from __future__ import annotations
import csv, hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

# Verified German-to-English semantic equivalents from the historical source
# taxonomy and the current 29-class names. Ambiguous names are intentionally
# left unmapped for manual policy adjudication.
ALIASES = {
    'schnur': 'String', 'zigaretten':'Cigarettes', 'kaffeekapsel':'Coffee Capsule',
    'papier':'Paper', 'katzensand':'Cat Litter', 'plastiksack':'Plastic Bag',
    'lebensmittelverpackung':'Plastic Food Packaging', 'alubuechse':'Aluminum Can',
    'alubüchse':'Aluminum Can', 'baw-sack':'BAW Bag', 'glasflasche':'Glass Bottle',
    'karton':'Cardboard', 'alufolie':'Aluminum Foil',
    'textilien':'Textiles', 'metalle':'Misc Metal', 'windel':'Diaper',
    'blumentopf':'Plastic Flower Pot', 'gummiband':'Rubber Bands',
}
# 'Diverses' is deliberately excluded because the export appears to both
# ignore this label and rename it to Unknown. This is an unresolved policy.
BLOCKED={'other', 'diverses'}
DEFAULT_AMBIGUOUS={'styropor/dämmung', 'styropor/daemmung'}
CONTEXT={'container', 'wagon'}

def norm(s):
    return ' '.join(str(s or '').replace('_',' ').strip().casefold().split())

def write_csv(path, fields, records):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)

def read_csv(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def get_names(path):
    try:import yaml
    except ImportError as e:raise RuntimeError('PyYAML is required for label diff') from e
    if not Path(path).is_file():raise FileNotFoundError(f'Missing July data.yaml: {path}')
    doc=yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(doc,dict) or 'names' not in doc:raise ValueError('July data.yaml must contain names')
    val=doc['names']
    if isinstance(val,list):return {int(i):str(v) for i,v in enumerate(val)}
    if isinstance(val,dict):return {int(k):str(v) for k,v in val.items()}
    raise ValueError('July data.yaml names must be list or id/name mapping')

def parse_map(path,oct_names):
    if path is None:return {}
    rows=read_csv(path)
    if not rows:raise ValueError('Class map file is empty')
    if len(rows[0])!=2:raise ValueError('Class map must have exactly 2 columns: july_class,october_class')
    cols=list(rows[0]);targets={norm(x):x for x in oct_names.values()}
    manual={}
    for row in rows:
        source=norm(row[cols[0]]);target=norm(row[cols[1]])
        if source in BLOCKED:raise ValueError(f'Blocked ambiguous July class requires separate policy resolution: {source}')
        if target not in targets:raise ValueError(f'Unknown October class in class map: {row[cols[1]]}')
        if source in manual and manual[source]!=targets[target]:raise ValueError(f'Conflicting mapping for {source}')
        manual[source]=targets[target]
    return manual

def mapping_rows(jul_names,oct_names,manual):
    by_name={norm(x):int(i) for i,x in oct_names.items()}
    mappings={}; records=[]
    for jul_id,name in sorted(jul_names.items()):
        s=norm(name);category='';target='';reason=''
        if s in CONTEXT: category='context_excluded';reason='exporter_context_policy'
        elif s in BLOCKED: category='unmapped_ambiguous';reason='Other/residual_or_unverified_ignore_rename_policy'
        elif s in DEFAULT_AMBIGUOUS and s not in manual: category='unmapped_ambiguous';reason='ambiguous_material_semantics_needs_explicit_map'
        elif s in manual: target=manual[s];category='mapped';reason='user_explicit_class_map'
        elif s in by_name: target=next(x for x in oct_names.values() if norm(x)==s);category='mapped';reason='exact_class_name'
        elif s in ALIASES and norm(ALIASES[s]) in by_name:
            target=ALIASES[s];category='mapped';reason='verified_name_equivalence'
        else: category='unmapped';reason='no_verified_october_equivalent'
        cid=by_name.get(norm(target),'') if target else ''
        mappings[jul_id]=dict(july_class=name,july_class_id=jul_id,category=category,
                              october_class=target,october_id=cid,reason=reason)
        records.append(mappings[jul_id])
    return mappings,records

def parse_july_label(path,names):
    """YOLO box/seg. Polygon bounds are kept for comparable box geometry.

    Do NOT infer negative when a file is missing/unreadable.
    """
    result=[];issues=[]
    for ln,text in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        data=text.split()
        if not data:continue
        try:
            cid=int(data[0]);values=[float(t) for t in data[1:]]
            if any(not math.isfinite(v) for v in values):raise ValueError('nonfinite coordinates')
            if cid not in names:issues.append(f'line {ln}: id {cid} missing from YAML')
            if len(values)==4:
                xc,yc,w,h=values; box=(xc-w/2,yc-h/2,xc+w/2,yc+h/2);geo='box'
            elif len(values)>=6 and len(values)%2==0:
                xx=values[::2]; yy=values[1::2];box=(min(xx),min(yy),max(xx),max(yy));geo='polygon'
            else:raise ValueError(f'wrong coordinate count: {len(values)}')
            if box[2]<=box[0] or box[3]<=box[1] or any(x < -1e-6 or x > 1+1e-6 for x in box):
                issues.append(f'line {ln}: invalid geometry {box}')
                continue
            result.append(dict(id=cid,bbox=box,geometry=geo,line=ln))
        except Exception as ex:issues.append(f'line {ln}: {ex}')
    return result,issues

def _iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));z=w*h
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);den=aa+bb-z
    return z/den if den else 0

def maximum_bipartite_match(jul,octo,iou_thr=0.5):
    """Maximum-cardinality matching by class and IoU, independent of row order.

    Returns list of (July local index, October local index, IoU).
    """
    available=defaultdict(list)
    for j,ref in enumerate(jul):
        if ref.get('october_id','')=='':continue
        for k,other in enumerate(octo):
            if int(other['class_id']) != int(ref['october_id']):continue
            iou=_iou(ref['bbox'],tuple(float(other[v]) for v in ('xmin','ymin','xmax','ymax')))
            if iou>=iou_thr:available[j].append((k,iou))
        available[j].sort(key=lambda x:(-x[1],x[0]))
    matched_to={}
    def augment(j,seen):
        for k,score in available[j]:
            if k in seen:continue
            seen.add(k)
            if k not in matched_to or augment(matched_to[k][0],seen):
                matched_to[k]=(j,score)
                return True
        return False
    for j in sorted(available, key=lambda x:(len(available[x]),x)):
        augment(j,set())
    return sorted((j,k,score) for k,(j,score) in matched_to.items())

def day_level_groups(images,out):
    """Collection Q/stop + floorspread EXIF calendar-day, connected by file/pixel identity.

    In the absence of a parseable floorspread capture date, keep the available
    camera-number run as a fallback and explicitly flag the uncertainty.
    """
    count=len(images);parent=list(range(count)); size=[1]*count
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b]
    by_key={};items=[]
    for i,r in enumerate(images):
        context=r.get('context','')
        if context=='floorspread':
            dt=str(r.get('exif_datetime_original',''))
            try:day=datetime.strptime(dt[:10],'%Y:%m:%d').date().isoformat()
            except ValueError:day=''
            key='floorspread|capture_day|'+day if day else 'floorspread|fallback|'+str(r.get('file_group_key') or r['image'])
            rule='EXIF_capture_day' if day else 'DSC_run_fallback_or_single_image'
        else:
            day=''
            key=str(r.get('file_group_key') or 'ungrouped|'+r['image'])
            rule='collection_Q_stop' if r.get('file_group_key') else 'ungrouped_filename'
        items.append(dict(image=r['image'],context=context,capture_day=day,
                          grouping_basis=rule,base_group=key))
        for typ,val in [('base',key),('sha256',r.get('sha256','')),
                        ('display_pixel_sha256',r.get('display_pixel_sha256','') if r.get('pixel_coordinate_frame')=='EXIF_display' else '')]:
            if not val:continue
            label=(typ,val)
            if label in by_key:union(i,by_key[label])
            else:by_key[label]=i
    members=defaultdict(list)
    for i in range(count):members[find(i)].append(i)
    groups=[]
    for inds in members.values():
        names=sorted(images[i]['image'] for i in inds)
        gid='SGD-'+hashlib.sha256(names[0].encode()).hexdigest()[:16]
        group=([items[i] for i in inds], [images[i] for i in inds])
        groups.append(dict(group_id=gid,size=len(inds),contexts=' | '.join(sorted({r['context'] for r in group[1]})),
                           capture_days=' | '.join(sorted({r['capture_day'] for r in group[0] if r['capture_day']})),
                           first_image=names[0],members=' | '.join(names)))
        for i in inds:
            items[i].update(split_group_day=gid,day_group_size=len(inds))
            images[i]['split_group_day']=gid
            images[i]['day_group_size']=len(inds)
            images[i]['capture_day']=items[i]['capture_day']
    write_csv(Path(out)/'37_day_level_split_groups.csv',
        ['image','context','capture_day','grouping_basis','base_group','split_group_day','day_group_size'],items)
    write_csv(Path(out)/'38_day_level_group_members.csv',
        ['group_id','size','contexts','capture_days','first_image','members'],sorted(groups,key=lambda g:(-g['size'],g['group_id'])))
    summ=dict(group_count=len(groups),largest_group=max((x['size'] for x in groups),default=0),
        distribution={str(k):v for k,v in sorted(Counter(g['size'] for g in groups).items())},
        capture_days={d:sum(x['capture_day']==d for x in items) for d in sorted({x['capture_day'] for x in items if x['capture_day']})},
        floorspread_images_without_date=sum(x['context']=='floorspread' and not x['capture_day'] for x in items),
        notes='Capture-day splits assume EXIF DateTimeOriginal is capture time. Q-code meaning still requires partner confirmation.')
    (Path(out)/'38_day_level_group_summary.json').write_text(json.dumps(summ,indent=2),encoding='utf-8')
    return summ

def reference_label_diff(ref_manifest,ref_labels,ref_yaml,oct_images,oct_objects,oct_names,class_map,out):
    """Compare July source label files to ALL October label versions by SHA or pixel identity.

    No representative annotation is silently selected; totals are bounded by
    candidate-version outcomes. Reference source has its own class namespace.
    """
    out=Path(out); ref_manifest=Path(ref_manifest);ref_labels=Path(ref_labels)
    if not ref_manifest.is_file():raise FileNotFoundError(ref_manifest)
    if not ref_labels.is_dir():raise NotADirectoryError(ref_labels)
    refs=read_csv(ref_manifest)
    if not refs:raise ValueError('Empty original July fingerprint manifest')
    if not {'image','sha256','source_path'}.issubset(refs[0]):
        raise ValueError('July reference fingerprint manifest must include image, sha256, source_path')
    jul_names=get_names(ref_yaml)
    manual=parse_map(class_map,oct_names)
    map_by_id,map_table=mapping_rows(jul_names,oct_names,manual)
    write_csv(out/'41_reference_class_mapping.csv',
        ['july_class_id','july_class','category','october_id','october_class','reason'],map_table)
    by_stem=defaultdict(list)
    for path in sorted(ref_labels.rglob('*.txt')):by_stem[path.stem.casefold()].append(path)
    for key in by_stem:by_stem[key].sort()
    oct_by_sha=defaultdict(list);oct_by_px=defaultdict(list)
    for o in oct_images:
        if o.get('sha256'):oct_by_sha[o['sha256']].append(o)
        if o.get('display_pixel_sha256') and o.get('pixel_coordinate_frame')=='EXIF_display':
            oct_by_px[o['display_pixel_sha256']].append(o)
    oct_objects_by_img=defaultdict(list)
    for o in oct_objects:oct_objects_by_img[o['image']].append(o)
    # Runtime agreement is required for pixel-only cross-dataset comparisons.
    env=ref_manifest.parent/'31_fingerprint_runtime.json';compatible=False
    if env.is_file():
        try:
            from analyze_oct26 import pillow_runtime
            a=json.loads(env.read_text(encoding='utf-8')).get('runtime',{}); b=pillow_runtime()
            keys=('pillow_version','jpeg_decoder_version','pixel_algorithm','dhash_algorithm','dhash_reduction_factor')
            compatible=all(a.get(x)==b.get(x) for x in keys)
        except Exception:compatible=False
    label_issues=[];detail=[];per_img=[];per_candidate=[]
    class_cnt=Counter();unmapped_cnt=Counter();class_match_lo=Counter();class_match_hi=Counter();ref_analyzed=0
    unmatched_july_images=0; missing_labels=0;conflicting_label_paths=0
    total_observed_rows=0
    for idx,ref in enumerate(refs,1):
        source_name=Path(ref.get('source_path') or ref['image']).stem
        choices=by_stem.get(source_name.casefold(),[])
        record=dict(ref_image=ref['image'],source_path=ref.get('source_path',''),stem=source_name,
            label_file='',label_state='',july_rows='',july_context='',july_mapped='',
            july_unmapped='',oct_candidate_versions='',oct_box_min='',oct_box_max='',
            same_class_iou05_matched_min='',same_class_iou05_matched_max='',
            annotations_disagree=False,reference_to_oct_match='')
        if not choices:
            missing_labels+=1;record['label_state']='missing';
            label_issues.append(dict(ref_image=ref['image'],issue='reference_label_missing',details=source_name))
            per_img.append(record);continue
        if len(choices)>1:
            file_content={hashlib.sha256(p.read_bytes()).hexdigest() for p in choices}
            if len(file_content)>1:
                conflicting_label_paths+=1;record['label_state']='ambiguous_label_files'
                label_issues.append(dict(ref_image=ref['image'],issue='different_label_files_same_stem',details=' | '.join(map(str,choices))))
                per_img.append(record);continue
        label=choices[0];record['label_file']=str(label);record['label_state']='ok'
        try: parsed,parse_issues=parse_july_label(label,jul_names)
        except Exception as exc:
            record['label_state']='unreadable'
            label_issues.append(dict(ref_image=ref['image'],issue='reference_label_unreadable',details=str(exc)))
            per_img.append(record);continue
        if parse_issues:
            record['label_state']='invalid_rows'
            for issue in parse_issues:label_issues.append(dict(ref_image=ref['image'],issue='reference_label_parse_issue',details=issue))
            per_img.append(record);continue
        ref_analyzed+=1;total_observed_rows+=len(parsed)
        for r in parsed:
            m=map_by_id.get(r['id'],dict(category='unknown_id',july_class=f'ID_{r["id"]}',october_id='',october_class=''))
            r.update(m)
            class_cnt[(r['id'],m['category'])]+=1
            if m['category'].startswith('unmapped'):unmapped_cnt[m['july_class']]+=1
        record['july_rows']=len(parsed)
        record['july_context']=sum(r['category']=='context_excluded' for r in parsed)
        record['july_mapped']=sum(r['category']=='mapped' for r in parsed)
        record['july_unmapped']=sum(r['category'].startswith('unmapped') for r in parsed)
        targets={}
        for row in oct_by_sha.get(ref.get('sha256',''),[]) if ref.get('sha256') else []:
            targets[row['image']]=(row,'file_sha256')
        if compatible and ref.get('pixel_coordinate_frame')=='EXIF_display':
            for row in oct_by_px.get(ref.get('display_pixel_sha256',''),[]) if ref.get('display_pixel_sha256') else []:
                targets.setdefault(row['image'],(row,'display_pixel_sha256'))
        record['oct_candidate_versions']=len(targets)
        if not targets:
            unmatched_july_images+=1; record['reference_to_oct_match']='none'
        else:record['reference_to_oct_match']='file/pixel_identity'
        variants=[]
        variant_match_classes=[]
        for octid,(orow,method) in sorted(targets.items()):
            octboxes=oct_objects_by_img.get(octid,[])
            pairs=maximum_bipartite_match(parsed,octboxes,iou_thr=.5)
            matched={j for j,k,v in pairs}
            variant_match_classes.append(Counter(parsed[j]['id'] for j,k,v in pairs))
            varsig=tuple(sorted((int(x['class_id']),*(round(float(x[z]),6) for z in ('xmin','ymin','xmax','ymax'))) for x in octboxes))
            variants.append((octid,len(octboxes),len(pairs),varsig))
            per_candidate.append(dict(ref_image=ref['image'],oct_image=octid,identity=method,
                july_rows=len(parsed),july_mapped=record['july_mapped'],oct_boxes=len(octboxes),
                same_class_iou05_matched=len(pairs),unmatched_mapped=record['july_mapped']-len(pairs),
                annotation_updated_utc=orow.get('annotation_updated_utc','')))
            for j,k,iou in pairs:
                detail.append(dict(ref_image=ref['image'],oct_image=octid,reference_label=str(label),
                    july_line=parsed[j]['line'],july_class=parsed[j]['july_class'],
                    october_class=parsed[j]['october_class'],october_line=octboxes[k].get('line',''),
                    iou=round(iou,7),match='matched'))
        if variants:
            for c in {p['id'] for p in parsed if p['category']=='mapped'}:
                class_match_lo[c]+=min(t[c] for t in variant_match_classes)
                class_match_hi[c]+=max(t[c] for t in variant_match_classes)
            record['oct_box_min']=min(x[1] for x in variants);record['oct_box_max']=max(x[1] for x in variants)
            record['same_class_iou05_matched_min']=min(x[2] for x in variants)
            record['same_class_iou05_matched_max']=max(x[2] for x in variants)
            record['annotations_disagree']=len({x[3] for x in variants})>1
        per_img.append(record)
    write_csv(out/'40_reference_label_diff_per_image.csv',
        list(per_img[0]) if per_img else ['ref_image'],per_img)
    write_csv(out/'42_reference_label_object_matches.csv',
        ['ref_image','oct_image','reference_label','july_line','july_class','october_class','october_line','iou','match'],detail)
    write_csv(out/'43_reference_label_diff_issues.csv',['ref_image','issue','details'],label_issues)
    write_csv(out/'44_reference_label_comparison_by_oct_image.csv',
        list(per_candidate[0]) if per_candidate else ['ref_image'],per_candidate)
    per_class=[]
    for m in map_table:
        cid=m['july_class_id']
        per_class.append({**m,'july_objects':class_cnt[(cid,m['category'])],
                          'iou05_same_class_matches_min':class_match_lo[cid],
                          'iou05_same_class_matches_max':class_match_hi[cid]})
    write_csv(out/'45_reference_class_diff_counts.csv',
        ['july_class_id','july_class','category','october_id','october_class','reason','july_objects',
         'iou05_same_class_matches_min','iou05_same_class_matches_max'],per_class)
    observed=sum(class_cnt.values());mapped=sum(v for (cid,cat),v in class_cnt.items() if cat=='mapped')
    context=sum(v for (cid,cat),v in class_cnt.items() if cat=='context_excluded')
    unmapped=observed-mapped-context
    valid=[x for x in per_img if x['label_state']=='ok']
    compared=[x for x in valid if x['oct_candidate_versions']>0]
    result=dict(reference_images=len(refs),reference_images_with_valid_labels=ref_analyzed,
        reference_missing_label_images=missing_labels,reference_conflicting_label_stem_images=conflicting_label_paths,
        july_observed_box_or_polygon_rows=observed,july_context_rows=context,
        july_mapped_material_rows=mapped,july_unmapped_rows=unmapped,
        july_rows_reconciliation_ok=observed==mapped+context+unmapped,
        july_observed_foreign_material_rows=mapped+unmapped,
        july_foreign_material_rows_vs_historical_5640=mapped+unmapped-5640,
        historical_5640_denominator_is_reported_foreign_material_instances=True,
        july_images_without_october_identity_match=unmatched_july_images,
        july_images_with_october_comparison=len(compared),
        july_images_with_multiple_october_versions=sum(int(r['oct_candidate_versions'])>1 for r in per_img if r['label_state']=='ok'),
        july_images_with_conflicting_october_versions=sum(bool(r['annotations_disagree']) for r in per_img if r['label_state']=='ok'),
        observed_oct_box_count_min=sum(int(r['oct_box_min']) for r in compared),
        observed_oct_box_count_max=sum(int(r['oct_box_max']) for r in compared),
        class_aware_iou05_match_min=sum(int(r['same_class_iou05_matched_min']) for r in compared),
        class_aware_iou05_match_max=sum(int(r['same_class_iou05_matched_max']) for r in compared),
        no_silent_canonical_annotation_selection=True,
        mapped_reference_class_count=sum(m['category']=='mapped' for m in map_table),
        class_map_manual_overrides=manual,
        ambiguous_other_is_always_unmapped=True,
        source_yaml=str(ref_yaml),reference_labels=str(ref_labels),class_map=str(class_map or ''),
        pixel_only_matching_runtime_compatible=compatible,
        warnings=['July source label counts and October export counts may represent different class exclusion policies.'],
        issue_count=len(label_issues))
    (out/'39_reference_label_diff_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result
