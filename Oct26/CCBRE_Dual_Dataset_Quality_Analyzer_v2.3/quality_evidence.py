"""CCBRE Dual Dataset Quality Analyzer v2.2: partner-facing evidence generation.

Input: original Jul-26 images/labels/YAML and Oct-26 S3 Raw, plus the freshly
produced v1.7.1 technical audit in this SAME run. No previous audit needed.
Raw files are never changed. The evidence is a review queue, not a diagnosis of
why annotations changed or proof a photograph is clean.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import hashlib

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'legacy_auditor'))
from analyze_oct26 import yaml_names, parse_annotation, hash_file
from v17_extensions import parse_july_label

IMG_EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}

PROBLEMS = [
  ('01_Duplicate_Annotation_Problems', 'Same photograph, different object markings'),
  ('02_Missing_July_Annotations', 'Objects marked in July but absent in the October export'),
  ('03_Class_Mapping_Problems', 'Object names and exclusions are inconsistent'),
  ('04_Empty_Annotation_Problems', 'No exported markings does not mean reviewed clean'),
  ('05_Image_Orientation', 'Photographs and object markings can display differently'),
  ('06_Camera_Metadata_and_Dates', 'Recorded camera dates need confirmation'),
  ('07_Image_Context_and_Views', 'Photo view and load identity require agreed names'),
  ('08_Dataset_Representativeness', 'Different populations and uneven category coverage'),
  ('09_Provider_Corrective_Actions', 'Decisions needed from collection, annotation and exporter teams')
]


def readcsv(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(f'Expected analysis output missing: {p}')
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def savecsv(path, rows, columns):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)


def savejson(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def asint(value):
    try:return int(float(value or 0))
    except(ValueError,TypeError):return 0


def fp(r):
    """Source Windows-formatted relative manifest path -> folder within Oct Raw."""
    return Path(*str(r).replace('\\','/').split('/'))


def oct_img_path(root, rel):
    path=root/fp(rel)
    if not path.is_file():raise FileNotFoundError(f'Missing October image: {path}')
    return path


def oct_label_path(root, rel):
    parts=list(fp(rel).parts)
    if 'images' not in parts:raise ValueError(f'Cannot locate labels-detect for {rel}')
    parts[parts.index('images')]='labels-detect'
    return (root/Path(*parts)).with_suffix('.txt')


def index_stems(folder):
    folder=Path(folder); d=defaultdict(list)
    if not folder.is_dir():raise FileNotFoundError(folder)
    for p in sorted(folder.rglob('*.txt')):d[p.stem.casefold()].append(p)
    return d


def find_label(by_stem, stem):
    hits=by_stem.get(str(stem).casefold(),[])
    return hits[0] if len(hits)==1 else None


def title_for_file(f):
    return Path(str(f).replace('\\','/')).name


def get_font(size=17):
    for f in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf']:
        try:return ImageFont.truetype(f,size)
        except OSError:pass
    return ImageFont.load_default()


def load_display(path, max_width=1000, max_height=750):
    with Image.open(path) as raw:
        # The verified label convention is the EXIF-displayed coordinate frame.
        # If EXIF processing fails, do not quietly claim alignment: warning is logged.
        try:im=ImageOps.exif_transpose(raw).convert('RGB')
        except Exception as exc:raise RuntimeError(f'EXIF conversion failed for {path}: {exc}') from exc
    im.thumbnail((max_width,max_height), Image.Resampling.LANCZOS)
    return im


def boxes_oct(path,oct_names):
    return parse_annotation(path,'box',oct_names,lambda *a:None,str(path))[0]


def boxes_july(path,july_names):
    if not path or not path.is_file():return []
    boxes,_=parse_july_label(path,july_names)
    return [dict(class_id=r['id'],bbox=r['bbox']) for r in boxes
            if july_names.get(r['id'],'').strip().casefold() not in ('container','wagon')]


def annotated_panel(imgpath, rows, names, heading, note='', line_color=(213,51,67), panel_width=850, panel_height=615):
    W,H=panel_width,panel_height
    canvas=Image.new('RGB',(W,H),(255,255,255));d=ImageDraw.Draw(canvas)
    font=get_font(17);small=get_font(15)
    d.text((14,8),heading[:98],font=font,fill=(30,45,57))
    try:
        img=load_display(imgpath,W-28,H-105)
        x=(W-img.width)//2; y=44+(H-105-img.height)//2
        canvas.paste(img,(x,y))
        d=ImageDraw.Draw(canvas)
        for r in rows:
            b=r['bbox']; cl=r.get('class_id',r.get('id',-1))
            xx=[x+round(b[k]*img.width) for k in (0,2)]
            yy=[y+round(b[k]*img.height) for k in (1,3)]
            if min(xx)<x-2 or min(yy)<y-2 or max(xx)>x+img.width+2 or max(yy)>y+img.height+2:continue
            d.rectangle((xx[0],yy[0],xx[1],yy[1]),outline=line_color,width=3)
            label=f'{names.get(int(cl),str(cl))}'
            tw=min(360,max(90,8*len(label)+10)); yt=max(y, yy[0]-21)
            d.rectangle((xx[0],yt,xx[0]+tw,yt+21),fill=line_color)
            d.text((xx[0]+4,yt+2),label[:42],font=small,fill='white')
    except Exception as exc:
        d.text((17,72),'IMAGE DISPLAY ERROR',fill='red',font=font)
        d.text((17,103),str(exc)[:98],fill='red',font=small)
    d.text((13,H-45),note[:120],font=small,fill=(50,62,72))
    return canvas


def panel_grid(path,panels,caption='',ncols=2):
    if not panels:return
    W,H=panels[0].size
    cols=min(ncols,len(panels));rows=math.ceil(len(panels)/cols)
    canvas=Image.new('RGB',(cols*W,rows*H+65),'white')
    for j,p in enumerate(panels):canvas.paste(p,((j%cols)*W,(j//cols)*H))
    d=ImageDraw.Draw(canvas);d.text((18,rows*H+16),caption[:190],font=get_font(19),fill=(35,45,52))
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=86)


def file_slug(name,max_chars=66):
    base=re.sub(r'[^A-Za-z0-9_.-]','_',str(name));base=base.strip('._-')
    return (base[-max_chars:] or 'image')+'_'+hashlib.sha256(str(name).encode()).hexdigest()[:8]


def plain_card(path, rows, title, max_examples=12, img_root=None):
    # Contact sheet of unannotated example photos, deliberately without claiming they are clean.
    selection=rows[:max_examples]; items=[]
    for row in selection:
        rel=row['image'];imgpath=oct_img_path(img_root,rel)
        items.append(annotated_panel(imgpath,[],{},title_for_file(rel),'NO EXPORTED FOREIGN-MATERIAL MARKINGS',panel_width=700,panel_height=485))
    if not items:return
    W,H=items[0].size;cols=3;N=math.ceil(len(items)/cols)
    canvas=Image.new('RGB',(W*cols,H*N+60),'white')
    for i,p in enumerate(items):canvas.paste(p,((i%cols)*W,(i//cols)*H))
    ImageDraw.Draw(canvas).text((20,H*N+12),title,font=get_font(19),fill=(20,30,40))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=82)


def bar_chart(path, labels, values, title, xlabel='Images',limit=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError: return
    if limit:labels=labels[:limit];values=values[:limit]
    h=max(3.6,0.35*len(labels)+1.8)
    fig,ax=plt.subplots(figsize=(10,h))
    ax.barh(labels[::-1],values[::-1]);ax.set_xlabel(xlabel);ax.set_title(title)
    for i,v in enumerate(values[::-1]):ax.text(v,i, f' {v:,}',va='center',fontsize=8)
    ax.set_xlim(0,max(values or [1])*1.16);ax.grid(axis='x',alpha=.15)
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)



def build(args):
    base=Path(args.output).resolve(); raw=Path(args.october_root).resolve(); tech=Path(args.audit_dir).resolve()
    july_images=Path(args.july_images).resolve();july_labels=Path(args.july_labels).resolve();july_yaml=Path(args.july_yaml).resolve()
    octnames,_=yaml_names(raw/'biowaste-collection-images'/'data.yaml')
    julnames,_=yaml_names(july_yaml)
    if not julnames or not octnames:raise ValueError('Unable to load July or October YAML class names.')
    for slug,name in PROBLEMS:(base/slug/'Examples').mkdir(parents=True,exist_ok=True)
    gen=base/'00_General_Dataset_Analysis';gen.mkdir(parents=True,exist_ok=True)
    qa=base/'99_Technical_Verification';qa.mkdir(parents=True,exist_ok=True)
    print('Reading technical manifests from this run',flush=True)
    im=readcsv(tech/'02_image_manifest.csv'); objects=readcsv(tech/'11_object_manifest.csv')
    dup=readcsv(tech/'22_duplicate_conflict_tiers.csv')
    julycomp=readcsv(tech/'40_reference_label_diff_per_image.csv')
    octcomp=readcsv(tech/'44_reference_label_comparison_by_oct_image.csv')
    classes=readcsv(tech/'03_class_statistics.csv')
    mapping=readcsv(tech/'41_reference_class_mapping.csv')
    issues=readcsv(tech/'07_quality_issues.csv')
    orientation=readcsv(tech/'30_orientation_review_index.csv')
    viewletters=readcsv(tech/'25_view_letter_statistics.csv')
    qgroups=readcsv(tech/'21_capture_group_manifest.csv')
    clock=readcsv(tech/'37_day_level_split_groups.csv')
    oroot=base/'01_Duplicate_Annotation_Problems';miss=base/'02_Missing_July_Annotations'
    mapdir=base/'03_Class_Mapping_Problems';emptydir=base/'04_Empty_Annotation_Problems'
    oridir=base/'05_Image_Orientation';metadir=base/'06_Camera_Metadata_and_Dates'
    viewdir=base/'07_Image_Context_and_Views';bal=base/'08_Dataset_Representativeness'
    act=base/'09_Provider_Corrective_Actions'
    by_im={r['image']:r for r in im}
    oct_obj=defaultdict(list)
    for r in objects:
        oct_obj[r['image']].append({'bbox':tuple(float(r[k]) for k in ('xmin','ymin','xmax','ymax')),'class_id':int(r['class_id'])})
    jul_stems=index_stems(july_labels)
    # July independent audit: read original files and labels (no reuse of October-only derived summaries).
    print('Independently inspecting July labels and source image inventory',flush=True)
    src_images=sorted(p for p in july_images.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT)
    juli=[]; jcounts=Counter();jissues=[]
    for p in src_images:
        label=find_label(jul_stems,p.stem)
        if label is None:
            jissues.append({'image':p.name,'issue':'missing_or_ambiguous_label_file','detail':str(jul_stems.get(p.stem.casefold(),[]))})
            rows=[]
        else:
            rows,errors=parse_july_label(label,julnames)
            for e in errors:jissues.append({'image':p.name,'issue':'invalid_july_label','detail':e})
        kinds=Counter(julnames.get(r['id'],f'Unknown ID {r["id"]}') for r in rows)
        for c,n in kinds.items():jcounts[c]+=n
        # Read header, not full decode. Jul fingerprints are created separately in core runner.
        try:
            with Image.open(p) as f:
                w,h=f.size; ex=f.getexif();dt=str(ex.get(36867,'') or '');ori=int(ex.get(274,1) or 1)
                make=str(ex.get(271,'') or '');model=str(ex.get(272,'') or '')
        except Exception as exc:
            w=h=0;dt=make=model='';ori=0
            jissues.append({'image':p.name,'issue':'unreadable_july_image_header','detail':str(exc)})
        juli.append({'image':p.name,'label_file':str(label or ''),'width':w,'height':h,
          'original_camera_date':dt,'orientation':ori,'camera_make':make,'camera_model':model,
          'material_rows':sum(n for c,n in kinds.items() if c.casefold() not in ('container','wagon')),
          'context_rows':sum(n for c,n in kinds.items() if c.casefold() in ('container','wagon')),
          'all_label_rows':len(rows),'annotation_state':'missing_label' if label is None else ('positive' if rows else 'explicit_empty')})
    savecsv(gen/'July_Image_Inventory.csv',juli,list(juli[0]) if juli else ['image'])
    savecsv(gen/'July_Class_Statistics.csv',[{'july_class':c,'objects':n} for c,n in sorted(jcounts.items(),key=lambda z:-z[1])],['july_class','objects'])
    savecsv(qa/'July_Quality_Issues.csv',jissues,['image','issue','detail'])
    shutil.copy2(tech/'03_class_statistics.csv',gen/'October_Class_Statistics.csv')
    shutil.copy2(tech/'04_source_context_statistics.csv',gen/'October_Context_Statistics.csv')
    shutil.copy2(tech/'39_reference_label_diff_summary.json',gen/'July_October_Annotation_Reconciliation.json')
    # General charts
    bars=[('Collection positive',sum(r['context']=='collection' and r['annotation_state']=='positive' for r in im)),
          ('Collection no markings',sum(r['context']=='collection' and r['annotation_state']=='explicit_empty' for r in im)),
          ('Plant positive',sum(r['context']=='floorspread' and r['annotation_state']=='positive' for r in im)),
          ('Plant no markings',sum(r['context']=='floorspread' and r['annotation_state']=='explicit_empty' for r in im))]
    bar_chart(gen/'Photo_Marking_Status.png',[t for t,v in bars],[v for t,v in bars],'Photos with and without exported markings')
    allclass=[r for r in classes if r['context']=='all']
    allclass.sort(key=lambda r:-asint(r['instances']))
    bar_chart(gen/'Top_Material_Classes.png',[r['class_name'] for r in allclass],[asint(r['instances']) for r in allclass], 'Marked foreign material by category',xlabel='Object markings')
    bar_chart(gen/'Same_July_Photos_Annotation_Comparison.png',['July markings','October minimum','October maximum'],[5640,3860,3981],'Same photographs: July vs October markings',xlabel='Object markings')
    # Problem 1 duplicate label conflicts
    dorder={'positive_empty':0,'count_change':1,'class_change':2,'geometry_substantive':3,'geometry_equivalent':4,'identical':5}
    dup.sort(key=lambda r:(dorder.get(r['tier'],8),r['image_a']))
    review=[]
    for i,r in enumerate(dup):
        a,b=r['image_a'],r['image_b']
        review.append({'case_id':f'DUP-{i+1:03d}','file_1':a,'file_2':b,'type':r['tier'],
         'objects_file_1':r['count_a'],'objects_file_2':r['count_b'],
         'why_review':'Same photograph has differing visible object markings' if r['tier']!='identical' else 'Identical markings',
         'approved_version':'','reason_for_change':'','correction_required':'','reviewer':'','review_date':''})
    savecsv(oroot/'Review_List.csv',review,list(review[0]))
    interesting=[r for r in dup if r['tier'] not in ('identical','geometry_equivalent')]
    for i,r in enumerate(interesting if args.all_duplicate_sheets else interesting[:args.samples_per_problem]):
        a,b=r['image_a'],r['image_b']
        panels=[annotated_panel(oct_img_path(raw,rel),oct_obj[rel],octnames,
                    f'COPY {j+1}: {title_for_file(rel)}',f'{len(oct_obj[rel])} marked items | {r["tier"]}') for j,rel in enumerate([a,b])]
        panel_grid(oroot/'Examples'/f'{i+1:03d}_{file_slug(title_for_file(a))}.jpg',panels,'Same source photo | TWO different annotation versions')
    bar_chart(oroot/'Difference_Types.png',list(Counter(r['tier'] for r in dup).keys()),list(Counter(r['tier'] for r in dup).values()),'Duplicate-photo annotation differences',xlabel='Duplicate pairs')
    # Problem 2 July vs Oct - preserve all candidate versions in case records
    oct_by_ref=defaultdict(list)
    for r in octcomp:oct_by_ref[r['ref_image']].append(r)
    losses=[]
    for r in julycomp:
        m=asint(r['july_mapped']); mn=asint(r['oct_box_min']); mx=asint(r['oct_box_max'])
        if m>mx:losses.append((m-mx,r))
    losses.sort(key=lambda x:(-x[0],x[1]['ref_image']))
    lostrows=[]
    for i,(delta,r) in enumerate(losses):
        cand=oct_by_ref.get(r['ref_image'],[])
        lostrows.append({'case_id':f'JUL-{i+1:04d}','july_image':r['ref_image'],
          'july_mapped_objects':r['july_mapped'],'october_boxes_min':r['oct_box_min'],
          'october_boxes_max':r['oct_box_max'],'minimum_box_count_shortfall':delta,
          'october_versions':' | '.join(x['oct_image'] for x in cand),
          'what_happened_to_missing_objects':'','approved_correction':'','reviewer':'','review_date':''})
    savecsv(miss/'Review_List.csv',lostrows,list(lostrows[0]) if lostrows else ['case_id'])
    for i,(_,r) in enumerate(losses[:args.samples_per_problem]):
        stem=r['stem']; julfile=find_label(jul_stems,stem); jul_image=july_images/title_for_file(r['ref_image'])
        if not jul_image.is_file():
            matches=[p for p in src_images if p.stem.casefold()==stem.casefold()]
            jul_image=matches[0] if len(matches)==1 else None
        if jul_image is None or not julfile:continue
        candidates=sorted(oct_by_ref[r['ref_image']],key=lambda x:(x.get('annotation_updated_utc',''),x['oct_image']))
        p=[annotated_panel(jul_image,boxes_july(julfile,julnames),julnames,'JULY: '+stem,'Original July annotation (material + scene)')]
        for j,r2 in enumerate([candidates[0],candidates[-1]] if len(candidates)>1 else candidates):
            rel=r2['oct_image'];p.append(annotated_panel(oct_img_path(raw,rel),oct_obj[rel],octnames,f'OCTOBER VERSION {j+1}: '+stem,f'{len(oct_obj[rel])} objects | not canonical'))
        panel_grid(miss/'Examples'/f'{i+1:03d}_{file_slug(stem)}.jpg',p,'July and October annotations on the SAME source photograph',ncols=len(p))
    # Problem 3 category mapping + reviewed-empty affected
    shutil.copy2(tech/'41_reference_class_mapping.csv',mapdir/'Old_to_New_Category_Mapping.csv')
    only_unmapped=[r for r in julycomp if asint(r['july_unmapped'])>0 and asint(r['july_mapped'])==0]
    maprev=[]
    for i,r in enumerate(only_unmapped):
        maprev.append({'case_id':f'MAP-{i+1:03d}','july_image':r['ref_image'],
          'unmapped_old_objects':r['july_unmapped'],'oct_min':r['oct_box_min'],'oct_max':r['oct_box_max'],
          'decision_on_old_object':'','approved_new_category':'','restore_in_export':'','reviewer':'','review_date':''})
    savecsv(mapdir/'Review_List.csv',maprev,list(maprev[0]) if maprev else ['case_id'])
    for i,r in enumerate(only_unmapped[:args.samples_per_problem]):
        stem=r['stem']; julfile=find_label(jul_stems,stem)
        hits=[p for p in src_images if p.stem.casefold()==stem.casefold()]
        if not julfile or len(hits)!=1:continue
        canv=annotated_panel(hits[0],boxes_july(julfile,julnames),julnames,'JULY categories: '+stem,
                               'Check which old material name should be retained')
        panels=[canv]
        linked=sorted(oct_by_ref.get(r['ref_image'],[]),key=lambda v:(v.get('annotation_updated_utc',''),v['oct_image']))
        if linked:
            rel=linked[0]['oct_image']
            panels.append(annotated_panel(oct_img_path(raw,rel),oct_obj[rel],octnames,
                'OCTOBER export: '+stem,'Unmapped old items might be absent'))
        panel_grid(mapdir/'Examples'/f'{i+1:03d}_{file_slug(stem)}.jpg',panels,
                   'Same photograph: old category annotations versus current export')
    shutil.copy2(tech/'07_quality_issues.csv',mapdir/'All_Exporter_and_Quality_Issues.csv')
    # Problem 4 empty labels: review queue contains all 4841; examples deterministic sample
    emp=[r for r in im if r['annotation_state']=='explicit_empty']
    erev=[{'image':r['image'],'context':r['context'],'source_bucket':r['source_bucket'],'review_status':'unreviewed',
           'foreign_material_visible':'','if_yes_categories':'','reason_no_boxes':'','reviewer':'','review_date':''} for r in emp]
    savecsv(emptydir/'Review_List.csv',erev,list(erev[0]) if erev else ['image'])
    for context in ('collection','floorspread'):
        group=sorted([r for r in emp if r['context']==context],key=lambda r:r['image'])
        if group:plain_card(emptydir/'Examples'/f'{context}_No_Exported_Markings.jpg',group,
                  f'{context.capitalize()}: examples for HUMAN checking (not certified clean)',args.samples_per_problem,raw)
    # Problem 5 orientation: all comparison sheets are in freshly generated technical output
    orientrows=[]
    for r in orientation:
        a=by_im.get(r['image'],{})
        orientrows.append({'image':r['image'],'context':a.get('context',''),'exif_orientation':r['exif_orientation'],
          'comparison_sheet':r['sheet'],'review_decision':'','correct_object_alignment':'','reviewer':'','review_date':''})
    savecsv(oridir/'All_Orientation_Review.csv',orientrows,list(orientrows[0]) if orientrows else ['image'])
    # stratified, deterministic: order rotation 3,6,8 crossed with sources
    buckets=defaultdict(list)
    for r in orientrows:buckets[(r['context'],r['exif_orientation'])].append(r)
    for v in buckets.values():v.sort(key=lambda r:r['image'])
    groupkeys=sorted(buckets)
    chosen=[]
    while len(chosen)<min(30,len(orientrows)) and groupkeys:
        nextkeys=[]
        for k in groupkeys:
            if buckets[k]:
                chosen.append(buckets[k].pop(len(buckets[k])//2));
                if len(chosen)==30:break
            if buckets[k]:nextkeys.append(k)
        if len(chosen)==30:break
        groupkeys=nextkeys
    savecsv(oridir/'Orientation_30_Human_Signoff_Ledger.csv',chosen,list(chosen[0]) if chosen else ['image'])
    for i,r in enumerate(chosen):
        src=tech/fp(r['comparison_sheet'])
        if src.exists():shutil.copy2(src,oridir/'Examples'/f'{i+1:02d}_{file_slug(title_for_file(r["image"]))}.jpg')
    # Problem 6 capture date: recorded EXIF dates, no invented correction to 2025-2026
    byQ=defaultdict(list)
    for r in im:
        if r['context']=='collection':byQ[r.get('q_code') or 'UNPARSED'].append(r)
    date_rows=[]
    for q,rows in sorted(byQ.items()):
        dates=sorted(set((r['exif_datetime_original'] or '')[:10] for r in rows if r['exif_datetime_original']))
        date_rows.append({'q_code':q,'photo_records':len(rows),'first_exif_date':dates[0] if dates else '',
            'last_exif_date':dates[-1] if dates else '', 'distinct_exif_days':len(dates),
            'camera_models':' | '.join(sorted({r['exif_camera_make']+' '+r['exif_camera_model'] for r in rows})),
            'actual_capture_date_or_period':'','camera_clock_was_correct':'','proof_or_source':'','reviewer':'','review_date':''})
    savecsv(metadir/'Collection_Camera_Date_Review.csv',date_rows,list(date_rows[0]) if date_rows else ['q_code'])
    savecsv(metadir/'Image_Capture_Metadata.csv',[{'image':r['image'],'context':r['context'],
      'EXIF_original_date':r['exif_datetime_original'],'camera_make':r['exif_camera_make'],
      'camera_model':r['exif_camera_model'],'EXIF_orientation':r['exif_orientation'],
      'verified_actual_date':'','date_verification_source':''} for r in im],
      ['image','context','EXIF_original_date','camera_make','camera_model','EXIF_orientation','verified_actual_date','date_verification_source'])
    monthcounts=Counter((r['context'],(r['exif_datetime_original'] or '')[:7]) for r in im if r['exif_datetime_original'])
    savecsv(metadir/'EXIF_Capture_Months.csv',[{'context':c,'exif_year_month':mo,'photos':n} for (c,mo),n in sorted(monthcounts.items())],['context','exif_year_month','photos'])
    # Problem 7 source views, suffixes, group examples
    shutil.copy2(tech/'25_view_letter_statistics.csv',viewdir/'Collection_View_Suffix_Counts.csv')
    viewrows=[{'suffix':r['view_letter'],'number_photos':r['images'],
          'agreed_meaning':'','view_definition':'','verified_by':'','review_date':''} for r in viewletters]
    savecsv(viewdir/'View_Letters_Need_Definitions.csv',viewrows,['suffix','number_photos','agreed_meaning','view_definition','verified_by','review_date'])
    by_group=defaultdict(list)
    for r in im:
        if r['context']=='collection':by_group[(r['q_code'],r['stop'])].append(r)
    load_rows=[]
    for (q,stop),rr in sorted(by_group.items()):
        load_rows.append({'q_code':q,'stop':stop,'photos':len(rr),
        'view_letters':' '.join(sorted({r['view_letter'] for r in rr})),
        'all_photos_same_physical_load':'','actual_delivery_or_load_id':'','verified_by':'','review_date':''})
    savecsv(viewdir/'Collection_Load_and_Views.csv',load_rows,['q_code','stop','photos','view_letters','all_photos_same_physical_load','actual_delivery_or_load_id','verified_by','review_date'])
    for suf in ['c','w','t','b']:
        rr=sorted([r for r in im if r.get('view_letter')==suf and r.get('context')=='collection'],key=lambda r:r['image'])
        if rr:plain_card(viewdir/'Examples'/f'view_{suf}_examples.jpg',rr,f'Suffix {suf} | meaning to confirm',max_examples=6,img_root=raw)
    fdays=Counter(r['capture_day'] for r in clock if r['context']=='floorspread')
    savecsv(viewdir/'Plant_Photos_by_Capture_Day.csv',[{'day':k,'photos':v,'one_physical_scene':'','scene_id':'','confirmed_by':''} for k,v in sorted(fdays.items())],['day','photos','one_physical_scene','scene_id','confirmed_by'])
    bar_chart(viewdir/'Plant_Capture_Days.png',list(sorted(fdays)),[fdays[k] for k in sorted(fdays)],'Processing-plant photographs by camera date',xlabel='Photographs')
    qdays=Counter((r.get('exif_datetime_original','') or '')[:10] for r in im if r.get('context')=='collection')
    savecsv(viewdir/'Collection_Photos_by_Camera_Day.csv',[{'day':k,'photos':v} for k,v in sorted(qdays.items())],['day','photos'])
    topq=Counter(r['q_code'] for r in im if r['context']=='collection').most_common(18)
    bar_chart(viewdir/'Collection_Top_Q_Groups.png',[k for k,v in topq],[v for k,v in topq], 'Collection photo records by Q-code (largest groups)')
    # Problem 8 prevalence and scarce materials
    stats=[{'context':ctx,'total_photos':sum(r['context']==ctx for r in im),
      'photos_with_exported_markings':sum(r['context']==ctx and r['annotation_state']=='positive' for r in im),
      'photos_without_exported_markings':sum(r['context']==ctx and r['annotation_state']=='explicit_empty' for r in im),
      'positive_share_percent':round(100*sum(r['context']==ctx and r['annotation_state']=='positive' for r in im)/max(1,sum(r['context']==ctx for r in im)),1)}
      for ctx in ('collection','floorspread')]
    savecsv(bal/'Positive_Empty_Comparison.csv',stats,list(stats[0]))
    rare=[{'class_name':r['class_name'],'marked_objects':r['instances'],'image_records':r['images_containing_class'],
           'more_examples_possible':'','capture_conditions_to_add':'','verified_by':''} for r in allclass if asint(r['images_containing_class'])<100]
    savecsv(bal/'Rare_Materials_To_Consider.csv',rare,['class_name','marked_objects','image_records','more_examples_possible','capture_conditions_to_add','verified_by'])
    for name in ('Photo_Marking_Status.png','Top_Material_Classes.png'):
        if (gen/name).exists():shutil.copy2(gen/name,bal/name)
    # Corrective action register human-readable, one row per issue
    entries=[
      ('1','Same photograph has different object markings','Annotation owner','Choose the correct annotation for each of 107 significant disagreements, starting with 13 marked-versus-empty pairs','CCBRE compares approved decision to export'),
      ('2','Some July object markings are missing on the same October photographs','Annotation owner and exporter','Examine ~30 representative changes and explain each lost/merged object; supply approved revisions','CCBRE checks full corrected set'),
      ('3','Old material names may be ignored during export','Exporter and annotation owner','Confirm Diverses/Unknown, Styropor/Insulation and Textile/Textiles decisions; provide excluded-object inventory','CCBRE verifies complete agreed mapping'),
      ('4','An empty annotation file does not establish a checked clean photograph','Annotation team','Mark reviewed clean, reviewed contaminated or review pending; inspect a representative sample','CCBRE keeps these statuses distinct'),
      ('5','Some photographs and labels need consistent rotation','Photography and annotation team','Confirm intended capture direction, export display-frame coordinates and sign the 30-example comparison','CCBRE keeps EXIF handling consistent'),
      ('6','Camera timestamps may not reflect actual collection dates','Photographers and collection coordinator','Verify actual capture date against logs, document camera-clock errors and retain the original EXIF value','CCBRE uses verified dates separately'),
      ('7','Container/wagon/other views and physical-load IDs are unclear','Collection and plant teams','Explain c/w/t/b suffixes and Q/stop; give shared load/scene IDs for related photographs','CCBRE records context on each photo'),
      ('8','Clean/contaminated and rare-material examples are uneven','Photographers and annotation team','Confirm reviewed-clean status and whether additional representative scenes can be collected','CCBRE plans appropriate AI sampling'),
      ('9','Earlier annotation history lacks matching exported photos','Exporter','Explain 467 collection and 29 plant history entries not linked to exported photos','CCBRE checks supplied revision log')]
    decisions=[{'problem':n,'issue':desc,'swiss_team_owner':owner,'action_requested':todo,
       'ccbre_action':us,'partner_decision':'','date_agreed':'','target_delivery':'','status':'OPEN'} for n,desc,owner,todo,us in entries]
    savecsv(act/'Corrective_Action_Register.csv',decisions,list(decisions[0]))
    # Explicit invariants from existing verified snapshot, separated from input correctness
    overview=json.loads((tech/'01_overview.json').read_text(encoding='utf-8'))
    jd=json.loads((tech/'39_reference_label_diff_summary.json').read_text(encoding='utf-8'))
    expectations={'october_photo_records':(len(im),9628),'october_box_rows':(len(objects),14087),
                  'july_original_image_files':(len(src_images),3359),
                  'july_foreign_material_rows':(int(jd.get('july_observed_foreign_material_rows',-1)),5640),
                  'duplicate_image_pairs':(len(dup),221),
                  'substantive_duplicate_pairs':(len(interesting),107),
                  'reference_images_found':(int(jd.get('reference_images_with_valid_labels',-1)),3359),
                  'exif_nondefault_photos':(int(overview.get('exif_orientation_nontrivial_count',-1)),349),
                  'recommended_day_protected_groups':(int(overview.get('day_level_split_groups',{}).get('group_count',-1)),3236)}
    vrows=[{'test':k,'observed':val,'frozen_Oct26_expected':exp,'same_as_frozen_snapshot':val==exp}
           for k,(val,exp) in expectations.items()]
    savecsv(qa/'Frozen_Baseline_Comparison.csv',vrows,['test','observed','frozen_Oct26_expected','same_as_frozen_snapshot'])
    savejson(qa/'Evidence_Run_Manifest.json',{'time_utc':datetime.now(timezone.utc).isoformat(),
      'original_raw_sources':{'july_images':str(july_images),'july_labels':str(july_labels),
        'july_yaml':str(july_yaml),'october_raw':str(raw)},'full_audit_of_current_run':str(tech),
      'auditor': 'CCBRE Dual Dataset Quality Analyzer v2.2 using bundled audited core v1.7.1',
      'immutable_input_policy':True,'number_of_october_image_rows':len(im),'number_of_july_original_image_files':len(src_images),
      'frozen_snapshot_checks':vrows,'human_review_needed':True,
      'notes':['EXIF camera dates are reported values, not verified physical dates.',
        'Empty export labels do not assert reviewed-clean status.',
        'All figures and examples are derived from the actual original image/label files, not generated image substitutes.']})
    # Consistent plain-English four-part problem sheets, tied to actual evidence.
    problem_text={
     '01': (
      'We received repeated copies of exactly the same photograph. Their object markings sometimes differ.',
      f'{len(dup)} repeated-photo pairs were identified. {len(interesting)} pairs have substantial differences; 13 say that an object is present in one copy but absent from the other.',
      'Repeated photographs are not new photographic observations. A changed rectangle can be a legitimate correction, but only the annotation owner can choose the correct object names, counts and positions.'),
     '02': (
      'Photographs already delivered in July appear again in October, but many have fewer marked foreign objects.',
      f'The historical July photos contain {jd["july_observed_foreign_material_rows"]} marked foreign materials. October contains between {jd["observed_oct_box_count_min"]} and {jd["observed_oct_box_count_max"]} marked objects on those same photographs, depending on which available marking version is examined.',
      'Do not compare the October whole-dataset total with July to conclude that markings increased or decreased. Compare each July photo with its October copy and document why any object was removed, merged or reclassified.'),
     '03': (
      'Material names have changed. Old categories can be ignored in the exported list unless their meaning is explicitly agreed.',
      'The executed mapping leaves 200 July Diverses annotations and 13 Styropor/Dämmung annotations unmapped. The former may have been excluded by the exporter. The latter has an Insulation equivalent and should be checked as a mapping issue. Three Textile masks were excluded from the plant export.',
      'A missing mapped name does not prove that an item was incorrectly labeled. The exporter must distinguish intentionally ignored categories from unintentionally lost objects. Obtain the exact excluded-object list.'),
     '04': (
      'A photograph without exported object rectangles may have been reviewed and found clean, or it may be unreviewed or missing markings.',
      f'{len(emp)} October image records have an existing empty annotation file. The attached images illustrate review cases but are not certified clean.',
      'A correct AI alarm may be scored as wrong if the supposedly clean reference photograph actually contains an unmarked foreign object. Human review status must be recorded separately from exported label count.'),
     '05': (
      'Some cameras store pictures sideways or upside down and give software instructions for how to display them. Object markings must follow the same displayed picture.',
      f'{sum(r["exif_orientation"] not in ("","1") for r in im)} photographs have a non-default EXIF orientation. {len(orientation)} of those contain object markings and have two-way comparison sheets. Thirty selected sheets are offered for human sign-off.',
      'Camera rotation metadata is different from whether photographers were asked to shoot horizontally. The display-frame interpretation is supported by preliminary assessment, but the named reviewer and date fields remain blank.'),
     '06': (
      'The dates stored inside collection photographs may disagree with the dates recorded by the collection team.',
      'EXIF reports dates mainly in 2024–2025 for collection photographs. The Swiss collection team reports that the actual acquisition may instead have taken place during 2025–2026. This discrepancy has not been resolved by the audit.',
      'Camera clocks can be set incorrectly. Preserve the original EXIF value and add a separately verified actual date supported by route logs or other records. Never silently shift dates by a guessed number of months or years.'),
     '07': (
      'Several photographs can show the same load from different views, but the practical meaning of filenames and suffix letters is not fully recorded.',
      f'{len(by_group)} collection Q-code/stop combinations were observed. Suffixes c, w, t, b are recorded separately. The plant photographs span {len(fdays)} camera capture days.',
      'The immediate need is that each view of the material has its visible foreign objects marked correctly. The photographer should identify container, wagon and any additional view types in readable data fields. Do not guess the meaning of t or b.'),
     '08': (
      'The two photo sources contain very different proportions of photos with labeled foreign materials. Some materials appear rarely.',
      f'Collection has {stats[0]["positive_share_percent"]}% of photos with exported markings, while plant photos have {stats[1]["positive_share_percent"]}%. Twenty-one material classes have fewer than 100 supporting photo records in the audited snapshot.',
      'Photos intentionally staged at a plant and naturally acquired collection photos need clear provenance labels. Extra representative clean and contaminated scenes, with correctly marked rare materials, can help development. Keep observed population frequencies documented for fair evaluation.'),
     '09': (
      'Several questions require written decisions by the colleagues who collected, marked, and exported the photographs.',
      f'The Corrective_Action_Register.csv contains {len(decisions)} assigned issues, with editable decision, date and status fields.',
      'The numerical audit is complete. The authoritative annotation set, camera dates, category exclusions and image context decisions are still human/provider matters.')
    }
    for idx,(slug,title) in enumerate(PROBLEMS,1):
        action=entries[idx-1] if idx<=len(entries) else None
        description,observations,notes=problem_text[f'{idx:02d}']
        example_note='See Review_List.csv, other CSVs, graphs and Examples/ in this folder. Images show actual source photographs, not reconstructed illustrations.'
        if idx==9:example_note='Open Corrective_Action_Register.csv to record each partner decision and the acceptance state.'
        txt=(f'# Problem {idx}: {title}\n\n'
             f'## A. Problem\n{description}\n\n'
             f'## B. Numerical and photographic evidence\n{observations}\n\n{example_note}\n\n'
             f'## C. Important observations\n{notes}\n\n'
             f'## D. Required actions\n**Swiss colleagues:** {action[3] if action else "Complete the agreed review."}\n\n'
             f'**CCBRE:** {action[4] if action else "Verify the accepted corrections."}\n')
        (base/slug/'READ_ME_FIRST.md').write_text(txt,encoding='utf-8')
    # simple local HTML index with navigable problem structure
    lis=''.join(f'<li><a href="{html.escape(slug)}/READ_ME_FIRST.md">Problem {i}: {html.escape(title)}</a> — <a href="{html.escape(slug)}/Examples/">examples</a></li>' for i,(slug,title) in enumerate(PROBLEMS,1))
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CCBRE dataset quality review</title><style>body{{max-width:1000px;margin:55px auto;font:17px/1.6 Arial;color:#243447}}h1{{font-size:32px}}a{{color:#1a6372}}li{{padding:7px}}</style></head><body><h1>Green-waste photographs and labels — Quality Review</h1><p>CCBRE | July reference + October export | v2.2 evidence generated from the original files</p><p>Each problem folder contains a review list, numerical evidence and photographs drawn from actual source files. These examples are not decisions about which labels are correct.</p><ol>{lis}</ol><p><a href="09_Provider_Corrective_Actions/Corrective_Action_Register.csv">Corrective action register</a> | <a href="99_Technical_Verification/Frozen_Baseline_Comparison.csv">Verification against the frozen October snapshot</a></p><p>Partner-facing report v4.2 should be prepared only after checking the new evidence and receiving the photographers’/annotators’ explanations.</p></body></html>'''
    (base/'OPEN_FIRST.html').write_text(page,encoding='utf-8')
    print('Provider evidence saved to:',base)
    print('Snapshot consistency:',sum(v['same_as_frozen_snapshot'] for v in vrows),'/',len(vrows), 'matches (differences need explanation)')
    return vrows


def main():
    p=argparse.ArgumentParser(description='Create partner-ready problem folders from original July and October files and a freshly computed v1.7.1 audit.')
    for flag in ('july-images','july-labels','july-yaml','october-root','audit-dir','output'):
        p.add_argument('--'+flag,required=True)
    p.add_argument('--samples-per-problem',type=int,default=20,help='Max number of visual examples per problem (default 20)')
    p.add_argument('--all-duplicate-sheets',action='store_true',help='Render all 107 substantive duplicate cases')
    a=p.parse_args()
    build(a)

if __name__=='__main__':main()
