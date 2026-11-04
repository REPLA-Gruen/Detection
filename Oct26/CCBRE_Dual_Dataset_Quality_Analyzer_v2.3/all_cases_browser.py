"""Offline, searchable all-case browsers for the standalone July--October report.

Every row in the review lists is accessible, not just example images. The primary
and supporting tables remain separate so counts retain their original meanings.
Visual links are provided only where a photographic evidence file actually exists.
This generator does not modify any original July or October data or audit CSVs.
"""
from __future__ import annotations
import csv
import html
import json
from pathlib import Path
from urllib.parse import quote


def rows_csv(path:Path):
    if not path.is_file(): return []
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path:Path, rows:list[dict]):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def link_from(root:Path, source:Path):
    resolved=source.resolve()
    if not resolved.is_relative_to(root.resolve()): return None
    return '/'.join(quote(p,safe='-_.()') for p in resolved.relative_to(root.resolve()).parts)


def linked_examples(root,slug):
    return sorted(p for p in (root/slug/'Examples').glob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))


def attach_example(rows,examples, root, case_index_name='case_id'):
    # By construction, examples 001_*, 002_*, ... correspond to the first
    # items in the source review list. Do not guess a link for other cases.
    lookup={int(p.name.split('_',1)[0]):p for p in examples if p.name[:3].isdigit()}
    for row in rows:
        ref=str(row.get(case_index_name,'')).rsplit('-',1)[-1]
        i=int(ref) if ref.isdigit() else 0
        p=lookup.get(i)
        row['photo_evidence'] = link_from(root,p) if p else ''
    return rows


def build_sets(root:Path):
    t=root/'99_Technical_Verification'/'Full_Dataset_Audit_v171'
    output=[]
    # 1) Same source photo with different markings; all 135 differing pairs.
    d=root/'01_Duplicate_Annotation_Problems'
    original=rows_csv(d/'Review_List.csv')
    differing=[dict(r) for r in original if r.get('type')!='identical']
    attach_example(differing,linked_examples(root,d.name),root)
    allpairs=[dict(r) for r in original]
    attach_example(allpairs,linked_examples(root,d.name),root)
    output.append(dict(i=1,slug=d.name,name='Duplicate photographs with different markings',
      desc=f'{len(differing):,} same-photo pairs have different annotation signatures. The complete list of {len(original):,} duplicate pairs, including {len(original)-len(differing)} with identical markings, is available separately.',
      primary=('All differing duplicate pairs',differing),support=[('All 221 exact-duplicate pairs',allpairs)],
      columns=['case_id','file_1','file_2','type','objects_file_1','objects_file_2','approved_version','reviewer'],
      caution='A different box signature is a review candidate, not proof of an error. The 107 substantive differences have higher review priority than the 28 small geometry changes.'))
    # 2) Distinguish any version losing boxes vs every available version losing boxes.
    d=root/'02_Missing_July_Annotations'
    original=rows_csv(d/'Review_List.csv')
    allref=rows_csv(t/'40_reference_label_diff_per_image.csv')
    uncertain=[]
    by_stem={r.get('july_image','').replace('\\','/').split('/')[-1].lower():r for r in original}
    lookup={r.get('case_id',''):p for r,p in []}
    preview_lookup={int(p.name.split('_',1)[0]):p for p in linked_examples(root,d.name) if p.name[:3].isdigit()}
    for r in allref:
        j=int(float(r.get('july_mapped') or 0));mi=int(float(r.get('oct_box_min') or 0));ma=int(float(r.get('oct_box_max') or 0))
        if j <= mi: continue
        n=len(uncertain)+1
        row={'case_id':f'JUL-CAND-{n:04d}','july_image':r.get('ref_image',''),
             'july_mapped_objects':j,'october_boxes_min':mi,'october_boxes_max':ma,
             'reduction_in_all_versions':'YES' if j>ma else 'ONLY_SOME_VERSIONS',
             'missing_at_least':max(0,j-ma),'missing_at_most':max(0,j-mi),
             'review_decision':'','reviewer':'','review_date':'','photo_evidence':''}
        known=by_stem.get(row['july_image'].replace('\\','/').split('/')[-1].lower())
        if known:
            ref=str(known.get('case_id','')).rsplit('-',1)[-1]
            p=preview_lookup.get(int(ref)) if ref.isdigit() else None
            if p: row['photo_evidence']=link_from(root,p)
        uncertain.append(row)
    confirmed=attach_example([dict(r) for r in original],linked_examples(root,d.name),root)
    output.append(dict(i=2,slug=d.name,name='July markings reduced in October',
      desc=f'{len(uncertain):,} photographs have fewer boxes in at least one October annotation version; {sum(r["reduction_in_all_versions"]=="YES" for r in uncertain):,} have fewer boxes under every available version. The latter are the original focused review queue.',
      primary=('Any-version reduction candidates',uncertain),support=[('Reduction under all versions',confirmed)],
      columns=['case_id','july_image','july_mapped_objects','october_boxes_min','october_boxes_max','reduction_in_all_versions','missing_at_least','missing_at_most'],
      caution='Changes in box counts can reflect deletion, merging, category policy, or deliberate corrections. This list does not establish the cause.'))
    # 3) All images with unmapped-only July classes and all export issues.
    d=root/'03_Class_Mapping_Problems'
    names=[('July images with only unmapped material classes',rows_csv(d/'Review_List.csv')),
           ('All recorded exporter/quality issues',rows_csv(d/'All_Exporter_and_Quality_Issues.csv')),
           ('Old to new category map',rows_csv(d/'Old_to_New_Category_Mapping.csv'))]
    primary=[dict(r) for r in names[0][1]]
    attach_example(primary,linked_examples(root,d.name),root)
    output.append(dict(i=3,slug=d.name,name='Class mappings and excluded markings',
      desc=f'{len(primary):,} July photographs have only unmapped foreign-material categories. The full export issue list and the category mapping are available as separate tables.',
      primary=('July photos with only unmapped categories',primary),support=names[1:],
      columns=['case_id','july_image','unmapped_old_objects','oct_min','oct_max','decision_on_old_object','approved_new_category','reviewer'],
      caution='An unmapped category in the comparison is not automatically a missing object in the October export.'))
    # 4) All explicitly empty annotation records.
    d=root/'04_Empty_Annotation_Problems'; original=rows_csv(d/'Review_List.csv')
    output.append(dict(i=4,slug=d.name,name='Photographs with no exported object markings',
      desc=f'All {len(original):,} exported image records with explicitly empty object markings are listed. These were not independently certified clean.',
      primary=('All explicitly empty October records',[dict(r) for r in original]),support=[],
      columns=['image','context','source_bucket','review_status','foreign_material_visible','reviewer','review_date'],
      caution='A blank annotation file does not prove a photograph was checked and found free of foreign material. The visual contact sheets are samples.'))
    # 5) All rotated images (positive and empty), link sheets for the annotated ones.
    d=root/'05_Image_Orientation'; selected=rows_csv(d/'All_Orientation_Review.csv')
    by_name={r['image']:r for r in selected}
    raw=rows_csv(t/'02_image_manifest.csv')
    positive=[]
    for r in raw:
        ori=r.get('exif_orientation','1').strip()
        if ori in ('','1'):continue
        source=by_name.get(r.get('image',''),{})
        sheet=str(source.get('comparison_sheet','')).replace('\\','/')
        path=t/sheet if sheet else None
        positive.append({'image':r.get('image',''),'context':r.get('context',''),
             'exif_orientation':ori,'has_foreign_object_markings':r.get('annotation_state','')=='positive',
             'annotation_count':r.get('detection_instances','0'),'camera_model':r.get('exif_camera_model',''),
             'comparison_sheet':sheet,'review_decision':'','reviewer':'','review_date':'',
             'photo_evidence':link_from(root,path) if path and path.is_file() else ''})
    output.append(dict(i=5,slug=d.name,name='Camera rotation and label alignment',
      desc=f'All {len(positive):,} non-default camera-orientation image records are listed, including {len(selected):,} labeled photos with two-way comparison sheets. The 30 selected sign-off sheets remain separately available.',
      primary=('All rotated image records',positive),support=[('All 211 orientation sheets',selected),
          ('30-image human sign-off ledger',rows_csv(d/'Orientation_30_Human_Signoff_Ledger.csv'))],
      columns=['image','context','exif_orientation','has_foreign_object_markings','annotation_count','photo_evidence','reviewer'],
      caution='A portrait photograph is not necessarily incorrectly taken. The question is whether label rectangles align with the intended display orientation. Human review remains pending.'))
    # 6) 63 Q-code groups + all metadata for individual photos.
    d=root/'06_Camera_Metadata_and_Dates';review=rows_csv(d/'Collection_Camera_Date_Review.csv')
    output.append(dict(i=6,slug=d.name,name='Camera dates to verify',
      desc=f'All {len(review):,} collection Q-code groups are listed for photography-date confirmation. All {len(rows_csv(d/"Image_Capture_Metadata.csv")):,} individual image metadata rows are separately searchable.',
      primary=('Collection groups needing date verification',review),
      support=[('All image camera metadata',rows_csv(d/'Image_Capture_Metadata.csv'))],
      columns=['q_code','photo_records','first_exif_date','last_exif_date','distinct_exif_days','camera_models','actual_capture_date_or_period','reviewer'],
      caution='A discrepancy in recorded camera dates is a request for verification, not proof that the camera date is wrong.'))
    # 7) All grouped collection loads and view types; 7 plant days.
    d=root/'07_Image_Context_and_Views'; grouped=rows_csv(d/'Collection_Load_and_Views.csv')
    output.append(dict(i=7,slug=d.name,name='Photo views and collection loads',
      desc=f'All {len(grouped):,} collection (Q-code, stop) groups are available for confirmation. The view-letter definitions and seven plant capture days are included.',
      primary=('All collection stop groups',grouped),
      support=[('View letter meanings',rows_csv(d/'View_Letters_Need_Definitions.csv')),
               ('Plant capture day grouping',rows_csv(d/'Plant_Photos_by_Capture_Day.csv')),
               ('Collection views by suffix',rows_csv(d/'Collection_View_Suffix_Counts.csv'))],
      columns=['q_code','stop','photos','view_letters','all_photos_same_physical_load','actual_delivery_or_load_id','verified_by'],
      caution='Filename-based groups are candidates for partner confirmation. The important practical request is to mark visible objects correctly in every view.'))
    # 8) Long-tail material classes and the two source populations.
    d=root/'08_Dataset_Representativeness';rare=rows_csv(d/'Rare_Materials_To_Consider.csv')
    output.append(dict(i=8,slug=d.name,name='Rare materials and unbalanced photo types',
      desc=f'All {len(rare):,} low-support material categories appear in the review table, with the complete 29-category summary and source proportions as supporting tables.',
      primary=('Rare material categories needing review',rare),
      support=[('All October 29 material categories',rows_csv(root/'00_General_Dataset_Analysis'/'General_Statistics'/'04_October_29_Classes.csv')),
               ('Photo-source positive/empty proportions',rows_csv(d/'Positive_Empty_Comparison.csv'))],
      columns=['class_name','marked_objects','image_records','more_examples_possible','capture_conditions_to_add','verified_by'],
      caution='Few labeled instances do not necessarily mean the material is absent from real green waste. The class mix reflects what was photographed and marked.'))
    # 9) Complete action register.
    d=root/'09_Provider_Corrective_Actions';actions=rows_csv(d/'Corrective_Action_Register.csv')
    output.append(dict(i=9,slug=d.name,name='Partner decisions and corrections',
      desc=f'All {len(actions):,} corrective-action topics are available for review, assignment and resolution.',
      primary=('Complete corrective-action register',actions),support=[],
      columns=['problem','issue','swiss_team_owner','action_requested','ccbre_action','partner_decision','target_delivery','status'],
      caution='A blank decision is a pending review, not an accepted correction. Keep the original CSV for recording signed decisions.'))
    return output


def write_case_browser(root:Path, spec:dict, title:str, entries:list[dict], suffix:str='', supporting:list[str]|None=None):
    from html_report import doc_header,tail,esc,relative
    num=spec['i']; key=f'ALL_CASES_{num:02d}{suffix}'
    fp=root/f'{key}.html'
    export_path=root/f'{key}.csv'
    # The exported case indexes are derived views; the original CSV evidence is untouched.
    save_csv(export_path, entries)
    columns=list(dict.fromkeys(spec.get('columns',[])+list(entries[0]) if entries else spec.get('columns',[])))
    if 'photo_evidence' in columns:columns.remove('photo_evidence')
    columns=columns[:18] # full column set remains in downloadable CSV
    html_cols=[c for c in columns if c not in ('why_review','correction_required','reason_for_change')]
    if len(html_cols)>12:html_cols=html_cols[:12]
    def badge_link(name,url):
        return f'<a class="pill" href="{esc(url)}">{esc(name)}</a>'
    body=doc_header(title,root)
    body+=f'<h1>{esc(spec["name"])}</h1><p class="sub">Problem {num} | {esc(title)} | {len(entries):,} records</p>'
    body+=f'<p>{esc(spec["desc"])}</p>'
    body+=f'<p class="note"><strong>Important:</strong> {esc(spec["caution"])}</p>'
    body+=f'<div class="linkrow">{badge_link("← Return to problem",f"PROBLEM_{num:02d}.html")}{badge_link("Download all records (CSV)",export_path.name)}'
    for label,link in (supporting or []):body+=badge_link(label,link)
    body+='</div>'
    body+='''<div class="case-controls"><label for="case-search">Search filename, category or any column</label>
      <input type="search" id="case-search" placeholder="Type to filter all cases..." spellcheck="false" autocomplete="off">
      <label for="page-size">Rows per page</label><select id="page-size"><option>50</option><option selected>100</option><option>250</option></select>
      <button id="reset-search" type="button">Clear search</button></div>
      <div class="case-meta"><strong id="match-count">Loading cases...</strong><span id="page-info"></span></div>
      <div class="scroll"><table class="case-table"><thead id="case-header"></thead><tbody id="case-body"></tbody></table></div>
      <div class="pagination"><button type="button" id="page-prev">← Previous</button><span id="page-counter"></span><button type="button" id="page-next">Next →</button></div>
      <p class="small">All matching cases can be browsed here, with the full columns available in the CSV. An Open photo link appears only when a visual comparison sheet was generated. Other cases are still included in the table and can be located by their filename.</p>'''
    # Prevent closing the script element if a filename contains HTML-looking data.
    data=json.dumps(entries,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    cdata=json.dumps(html_cols,ensure_ascii=False)
    body+=f'''<script>
"use strict";
const DATA={data};
const COLS={cdata};
let matching=DATA, page=0;
const searchBox=document.getElementById('case-search');
const bodyNode=document.getElementById('case-body');
const pageSize=document.getElementById('page-size');
const header=document.getElementById('case-header');
const tr=document.createElement('tr');
for(const key of ['Evidence',...COLS]){{ const th=document.createElement('th'); th.textContent=key==='Evidence'?key:key.replaceAll('_',' ');tr.appendChild(th); }}
header.appendChild(tr);
function textCell(value){{const td=document.createElement('td');td.textContent=String(value===null||value===undefined?'':value);return td;}}
function show(){{
  const chunk=Number(pageSize.value)||100;
  const pages=Math.max(1,Math.ceil(matching.length/chunk));
  page=Math.min(Math.max(0,page),pages-1);
  const frag=document.createDocumentFragment();
  for(const row of matching.slice(page*chunk,(page+1)*chunk)){{
    const tr=document.createElement('tr');const imgCell=document.createElement('td');
    if(row.photo_evidence){{const a=document.createElement('a');a.href=row.photo_evidence;a.target='_blank';a.rel='noopener';a.textContent='Open photo';imgCell.appendChild(a);}}
    else {{imgCell.textContent='—';imgCell.className='small';}}
    tr.appendChild(imgCell);
    for(const key of COLS)tr.appendChild(textCell(row[key]));frag.appendChild(tr);
  }}
  bodyNode.replaceChildren(frag);
  document.getElementById('match-count').textContent=matching.length.toLocaleString()+' of '+DATA.length.toLocaleString()+' cases';
  document.getElementById('page-info').textContent='Search all records, not just the current page.';
  document.getElementById('page-counter').textContent='Page '+(page+1)+' / '+pages;
  document.getElementById('page-prev').disabled=page===0;
  document.getElementById('page-next').disabled=page>=pages-1;
}}
function filter(){{let term=searchBox.value.toLocaleLowerCase().trim();
  matching=!term?DATA:DATA.filter(row=>Object.values(row).some(value=>String(value||'').toLocaleLowerCase().includes(term)));
  page=0;show();
}}
searchBox.addEventListener('input',filter);pageSize.addEventListener('change',show);
document.getElementById('reset-search').addEventListener('click',()=>{{searchBox.value='';filter();searchBox.focus();}});
document.getElementById('page-prev').addEventListener('click',()=>{{page--;show();window.scrollTo(0,0);}});
document.getElementById('page-next').addEventListener('click',()=>{{page++;show();window.scrollTo(0,0);}});
show();
</script>'''
    fp.write_text(body+tail(),encoding='utf-8')
    return fp


def generate(root:Path):
    root=Path(root).resolve()
    specs=build_sets(root)
    outline={}
    for spec in specs:
        num=spec['i'];main=spec['primary']; supporting=[]
        # Secondary source lists have their own searchable pages and CSV files.
        for j,(name,data) in enumerate(spec['support'],1):
            sub=f'_SUPPORT_{j:02d}'
            out=write_case_browser(root,spec,name,[dict(r) for r in data],suffix=sub)
            supporting.append((f'{name} ({len(data):,})',out.name))
        main_page=write_case_browser(root,spec,main[0],main[1],supporting=supporting)
        outline[spec['slug']]={'problem_number':num,'description':spec['desc'],'primary_count':len(main[1]),'primary_title':main[0],
            'main_page':main_page.name,'supporting_tables':[{'label':lab,'page':f'ALL_CASES_{num:02d}_SUPPORT_{j:02d}.html','rows':len(data)} for j,(lab,data) in enumerate(spec['support'],1)]}
    (root/'ALL_CASES_MANIFEST.json').write_text(json.dumps({'version':'2.3','generated_from_existing_evidence':True,
      'description':'Full searchable row-level case listings per partner problem. Photographic preview links are limited to existing rendered examples.',
      'cases':outline},indent=2,ensure_ascii=False),encoding='utf-8')
    return outline
