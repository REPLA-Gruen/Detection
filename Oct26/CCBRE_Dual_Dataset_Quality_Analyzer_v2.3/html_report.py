"""Generate a complete OFFLINE HTML review from a finished standalone dual-dataset run.

The HTML links to original analysis-derived charts and photo comparisons located inside
that output folder. No network, server, or external JavaScript libraries are required.
This reporter never changes the source datasets or numerical CSV/JSON evidence.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
from collections import Counter
from urllib.parse import quote

PROBLEMS = [
 ('01_Duplicate_Annotation_Problems','Same photograph, different object markings',
  'Some photographs were exported more than once, with different rectangles or category names.',
  'Two versions of the same photograph can give contradictory instructions to a reviewer or AI system.',
  'Annotation team: approve one marking version per photograph, beginning with marked-versus-empty conflicts; record why each material was added, removed, merged or renamed.',
  'CCBRE: verify that each approved version is represented once and that the corrected export agrees with the decision log.'),
 ('02_Missing_July_Annotations','Objects marked in July but missing in October',
  'The original July photographs appear in the October delivery, but many have fewer foreign-material markings.',
  'A photograph may look unchanged while its object marking list becomes incomplete. Differences may be deliberate corrections or export omissions.',
  'Annotation owner: compare representative before/after photographs, explain changed objects and approve a complete corrected annotation set.',
  'CCBRE: confirm each correction in the July–October comparison and report any unresolved differences.'),
 ('03_Class_Mapping_Problems','Object category names and excluded markings',
  'Names have changed between exports, and some older categories do not appear in the new markings.',
  'An object can disappear from the exported marking list because its category name was ignored rather than translated.',
  'Export and annotation teams: provide an approved old-to-new category list and a per-photo record of intentionally omitted objects, including Diverses, Textile and Styropor/Dämmung.',
  'CCBRE: verify mappings and repeat the comparison without guessing the meaning of ambiguous old categories.'),
 ('04_Empty_Annotation_Problems','Photographs without marked foreign materials',
  'Some photographs have no exported object markings, but this does not tell us whether they were reviewed.',
  'A photograph marked as empty may truly be clean, may contain missed material, or may have lost a category during export.',
  'Annotation team: clearly distinguish reviewed clean, reviewed contaminated, and pending review. Inspect representative examples from each source.',
  'CCBRE: keep the three review states separate when preparing datasets.'),
 ('05_Image_Orientation','Camera rotation and alignment of object markings',
  'Some photographs have camera rotation instructions and can display vertically or upside down unless software uses those instructions.',
  'The marks around bags or packaging must remain aligned with the correct photograph orientation. A vertical photo is not automatically unusable.',
  'Photography and annotation teams: confirm any horizontal-photo rule, the display convention and the correctness of all 30 human-review examples. Sign and date the review ledger.',
  'CCBRE: use the same image orientation for checking photographs and for the AI input.'),
 ('06_Camera_Metadata_and_Dates','Photograph dates may not match real collection dates',
  'Camera-recorded dates for collection photographs may disagree with physical collection records.',
  'Incorrect camera settings can give a misleading acquisition history and affect interpretation of which materials were photographed when.',
  'Collection/photography team: verify actual dates against original records. Record original camera date and corrected date separately, with the source and reviewer.',
  'CCBRE: use verified dates without overwriting original photo metadata.'),
 ('07_Image_Context_and_Views','Container, wagon, other views and repeated loads',
  'Filenames indicate different photo types, but some view letters and shared-load identifiers lack an agreed meaning.',
  'Several photographs may show the same load. Each view should have complete and accurate object markings.',
  'Photographers: confirm the meaning of c, w, t and b, and provide a stable identifier for each physical load or plant arrangement.',
  'CCBRE: record the confirmed photo context and group information for subsequent analysis.'),
 ('08_Dataset_Representativeness','Clean and contaminated examples; rare materials',
  'Collection and processing-plant photographs have very different shares of images containing marked foreign objects, and rare categories have few examples.',
  'Unrepresentative photographs can make model development less reliable, especially for ordinary clean loads and uncommon objects.',
  'Photo and annotation teams: confirm clean status and staged-versus-field provenance. Where feasible, collect realistic examples of less common materials and both clean and contaminated scenes.',
  'CCBRE: document real source frequencies separately from any deliberate training balance.'),
 ('09_Provider_Corrective_Actions','Decisions and corrected delivery requested',
  'Several issues require the judgment of the people who photographed, marked and exported the data.',
  'A numerical check confirms the file contents but cannot decide which annotation or acquisition date is true.',
  'Swiss partners: record each decision, owner, review date, category policy and corrected-export version in the provided register.',
  'CCBRE: check the final correction package before undertaking model evaluation.'),
]

STYLES = '''
:root{color-scheme:light}*{box-sizing:border-box}html{scroll-behavior:smooth}
body{font-family:Arial,Helvetica,sans-serif;color:#23303a;margin:0;background:#fff;line-height:1.55}
a{color:#126185;text-decoration:none}a:hover{text-decoration:underline}
nav{position:sticky;top:0;background:white;border-bottom:1px solid #d9e0e6;padding:11px max(14px,calc((100vw - 1160px)/2));z-index:10;display:flex;gap:18px;flex-wrap:wrap;font-size:14px}
.wrap{max-width:1160px;padding:20px 26px 70px;margin:auto}h1{font-size:30px;line-height:1.2;margin:15px 0}h2{font-size:24px;border-bottom:1px solid #d5dee4;padding-bottom:8px;margin-top:37px}h3{font-size:18px;margin:19px 0 7px}h4{font-size:15px;margin:15px 0 4px}
p{margin:8px 0 12px}.sub{color:#586874;font-size:15px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:23px 0}
.metric{border:1px solid #d9e0e6;border-radius:8px;padding:15px 16px}.metric strong{font-size:29px;display:block}.metric span{color:#51616d;font-size:13px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:17px}.plain{border:1px solid #d9e0e6;border-radius:8px;padding:12px 15px;margin:12px 0}
.problem{padding-top:6px;margin-top:18px}.subblock{padding:0 0 3px 0}.small{font-size:13px;color:#5b6872}
.figure{margin:15px 0 20px;padding:10px 0;border-bottom:1px solid #edf0f2}.figure img{max-width:100%;max-height:550px;object-fit:contain;border:1px solid #e3e9ee;border-radius:5px}
.figure figcaption{font-size:13px;color:#596772}.thumbs{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin:12px 0}
.thumb{padding:6px 6px 11px;border:1px solid #e2e7eb;border-radius:6px}.thumb img{width:100%;height:200px;object-fit:contain}.thumb span{display:block;font-size:12px;color:#53616e;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.35;margin:12px 0 8px;table-layout:auto}th,td{padding:8px 9px;border-bottom:1px solid #dfe5e9;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#f4f7f8}tbody tr:nth-child(2n){background:#fbfcfd}
.scroll{overflow:auto;max-width:100%}.linkrow{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 18px}.linkrow a{border:1px solid #dbe4e9;border-radius:5px;padding:5px 9px;font-size:13px}
li{margin:6px 0}.note{border-left:3px solid #9ab5c2;padding-left:12px;margin:17px 0;color:#485966}
.case-button{display:inline-block;background:#16577c;color:#fff;border-radius:7px;padding:11px 18px;margin:8px 0 15px;font-weight:700;font-size:15px;text-decoration:none;border:1px solid #16577c}
.case-button:hover,.case-button:focus{background:#0b405d;color:#fff;text-decoration:none}
.case-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:18px 0}
.case-controls input{padding:10px 12px;font-size:15px;border:1px solid #a6b6c1;border-radius:5px;min-width:280px;flex:1}
.case-controls select,.case-controls button,.pagination button{padding:9px 12px;font-size:14px;border:1px solid #aebdc8;border-radius:5px;background:white;cursor:pointer}
.pagination{display:flex;gap:16px;align-items:center;margin:20px 0}.pagination button:disabled{opacity:0.4;cursor:default}
.case-meta{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:10px 0}
.case-table td{font-size:12px;max-width:330px}.pill{display:inline-block;padding:6px 10px;border:1px solid #d6e0e7;border-radius:6px}

footer{border-top:1px solid #dce3e8;font-size:13px;color:#52616b;margin-top:40px;padding-top:15px}
@media print{nav{display:none} .wrap{max-width:100%;padding:5px}h2{break-before:page}tr,img{break-inside:avoid}.thumbs{grid-template-columns:repeat(2,1fr)}}
'''

def esc(x): return html.escape(str(x if x is not None else ''),quote=True)

def number(x):
    try:return f'{int(float(x)):,}'
    except (TypeError, ValueError):return str(x)

def csvrows(path):
    if not path.is_file():return []
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def readjson(path):
    if not path.is_file():return {}
    return json.loads(path.read_text(encoding='utf-8'))

def inside(root,child):
    child=Path(child).resolve();root=Path(root).resolve()
    if not child.is_relative_to(root):raise ValueError(f'Not under output folder: {child}')
    return child

def relative(root,file):
    p=inside(root,file).relative_to(root)
    return '/'.join(quote(part,safe='-_.()') for part in p.parts)

def href(root,file,text=None):
    p=Path(file)
    if not p.exists():return ''
    return f'<a href="{esc(relative(root,p))}">{esc(text or p.name)}</a>'

def previewtable(rows,max_rows=15):
    if not rows:return '<p class="small">No rows in this file.</p>'
    keys=list(rows[0]);
    if len(keys)>11:keys=keys[:11]
    parts=['<div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(k.replace("_"," "))}</th>' for k in keys)+'</tr></thead><tbody>']
    for row in rows[:max_rows]:
        parts.append('<tr>'+''.join(f'<td>{esc(str(row.get(k,""))[:160])}</td>' for k in keys)+'</tr>')
    parts.append('</tbody></table></div>')
    if len(rows)>max_rows:parts.append(f'<p class="small">Showing {max_rows} of {len(rows):,} rows. Download the CSV for the complete list.</p>')
    return ''.join(parts)

def doc_header(title,root):
    tabs=[('COMPLETE_DATASET_QUALITY_REPORT.html','Complete report'),('GENERAL_STATISTICS.html','All statistics'),('CORRECTIVE_ACTIONS.html','Action register'),('EVIDENCE_INDEX.html','Evidence files')]
    nav='<nav>'+''.join(href(root,root/file,label) for file,label in tabs)+'</nav>'
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{esc(title)}</title><style>{STYLES}</style></head><body>'+nav+'<main class="wrap">')

def tail():
    return '<footer>CCBRE | July–October photograph and marking review | Evidence is for human/provider confirmation. No model accuracy is measured by this report.</footer></main></body></html>'

def fig(root,p,caption=''):
    p=Path(p)
    if not p.is_file():return ''
    link=relative(root,p)
    return f'<figure class="figure"><a href="{esc(link)}"><img loading="lazy" src="{esc(link)}" alt="{esc(caption or p.stem)}"></a><figcaption>{esc(caption or p.stem.replace("_"," "))}</figcaption></figure>'

def samples(root,paths,limit=6):
    paths=list(paths)
    if not paths:return '<p class="small">No visual examples are available in this folder for the current run.</p>'
    cards=[]
    for p in paths[:limit]:
        link=relative(root,p)
        cards.append(f'<a class="thumb" href="{esc(link)}"><img loading="lazy" src="{esc(link)}" alt="{esc(p.name)}"><span>{esc(p.name)}</span></a>')
    return f'<div class="thumbs">{"".join(cards)}</div><p class="small">{len(paths):,} photographs or comparison sheets are available in the evidence folder. Select a preview to open the full image.</p>'

def problem_evidence(root,slug,compact=False):
    directory=root/slug
    pictures=sorted((directory/'Examples').glob('*.jpg'))+sorted((directory/'Examples').glob('*.png'))
    charts=[p for p in sorted(directory.glob('*.png')) if p.is_file()]
    csvs=sorted(directory.glob('*.csv'))
    links='<div class="linkrow">'+''.join(href(root,p,p.name) for p in csvs)+'</div>'
    # Partner-relevant figure and examples, with all files linked.
    figs=''.join(fig(root,p,p.stem.replace('_',' ')) for p in charts[:2])
    evidence=figs+samples(root,pictures,3 if compact else 10)
    if csvs:
        main=next((p for p in csvs if p.name.lower() in ('review_list.csv','corrective_action_register.csv')),csvs[0])
        evidence+=f'<h4>Review list: {esc(main.name)}</h4>'+previewtable(csvrows(main),7 if compact else 30)
    return evidence+links

def general(root):
    general_path=root/'00_General_Dataset_Analysis'/'General_Statistics'
    gen=root/'00_General_Dataset_Analysis'
    title='General July–October dataset statistics'
    page=doc_header(title,root)+f'<h1>{esc(title)}</h1><p class="sub">Numbers describe photos and exported object markings. July photographs already recur within October collection, so the two periods must not be added as independent photographic populations.</p>'
    keytables=['01_Dataset_Overview.csv','04_October_29_Classes.csv','06_Collection_View_Suffixes.csv',
      '10_Image_Display_Resolutions.csv','12_Objects_Per_Image_Distribution.csv','13_Collection_Q_Code_Counts.csv',
      '14_Processing_Plant_Capture_Days.csv','15_Object_Size_1536px.csv','16_Same_July_Images_Markings.csv']
    for f in keytables:
        p=general_path/f
        if p.exists():
            page+=f'<h2>{esc(p.stem.replace("_"," "))}</h2>'+previewtable(csvrows(p),29 if 'Classes' in f else 18)+f'<p>{href(root,p,"Download complete CSV")}</p>'
    chartpaths=sorted(general_path.glob('*.png'))
    legacy=[root/'99_Technical_Verification'/'Full_Dataset_Audit_v171'/'figures'/i for i in ('01_class_image_support.png','02_source_images.png','03_box_area.png')]
    chartpaths.extend(x for x in legacy if x.exists())
    page+='<h2>Complete chart collection</h2><p>All available charts are included below, with links to their original full-resolution images.</p>'
    page+=''.join(fig(root,p,p.stem.replace('_',' ')) for p in chartpaths)
    allcsv=sorted(general_path.glob('*.csv'))+sorted(gen.glob('*.csv'))
    page+='<h2>Download all general-statistics tables</h2><div class="linkrow">'+''.join(href(root,p,p.name) for p in allcsv)+'</div>'
    page+='<div class="note">Camera-recorded dates have not been independently confirmed against the Swiss collection logs. Photographs without exported markings are not certified clean photographs. The relative-area histogram uses fractions per equal log10-area bin.</div>'
    (root/'GENERAL_STATISTICS.html').write_text(page+tail(),encoding='utf-8')

def evidenceindex(root):
    title='Complete evidence index'
    page=doc_header(title,root)+f'<h1>{esc(title)}</h1><p class="sub">Index of all CSV, JSON, PNG and JPG outputs. Original July and October source image files are not copied into this report.</p>'
    roots=[root/'00_General_Dataset_Analysis']+[root/s for s,*_ in PROBLEMS]+[root/'99_Technical_Verification']
    for folder in roots:
        if not folder.exists():continue
        files=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.csv','.json','.html','.md') and p.name!='EVIDENCE_INDEX.html')
        page+=f'<h2>{esc(folder.name.replace("_"," "))} — {len(files):,} files</h2><details><summary>View complete file list</summary><ul>'
        page+=''.join('<li>'+href(root,p,str(p.relative_to(folder)))+'</li>' for p in files)
        page+='</ul></details>'
    (root/'EVIDENCE_INDEX.html').write_text(page+tail(),encoding='utf-8')

def all_case_button(spec):
    if not spec:return ''
    count=spec['primary_count']
    label=f'View all {count:,} cases'
    return (f'<p><a class="case-button" href="{esc(spec["main_page"])}">'
            f'{esc(label)} &rarr;</a><br><span class="small">Search and browse the full case list, not only the example photographs. Download the full CSV from that page.</span></p>')


def mainreport(root, case_index=None):
    gen=root/'00_General_Dataset_Analysis'; t=root/'99_Technical_Verification'/'Full_Dataset_Audit_v171'
    jul=csvrows(gen/'July_Image_Inventory.csv');octim=csvrows(t/'02_image_manifest.csv')
    overview=readjson(t/'01_overview.json');label=readjson(t/'39_reference_label_diff_summary.json')
    objects=overview.get('detection_object_count',sum(int(float(r.get('detection_instances') or 0)) for r in octim))
    collection=sum(r.get('context')=='collection' for r in octim);plant=sum(r.get('context')=='floorspread' for r in octim)
    reps=csvrows(t/'22_duplicate_conflict_tiers.csv');confl=sum(r.get('tier') not in ('identical','geometry_equivalent') for r in reps)
    distinct= len(octim)-len(reps) # exact-file representative count when every duplicate group size is two; for varied groups prefer hash set
    unique_hashes=len({r['sha256'] for r in octim if r.get('sha256')})
    if unique_hashes:distinct=unique_hashes
    summary=[('July original photos',len(jul)),('October photo records',len(octim)),('Different October file contents',distinct),('October material markings',objects),
            ('Collection photo records',collection),('Processing-plant records',plant),('Repeated-image groups',len(reps)),('Substantive annotation disagreements',confl)]
    page=doc_header('CCBRE complete July–October photograph and annotation quality review',root)
    page+='<h1>Photographs and object markings: complete July–October quality review</h1><p class="sub">CCBRE | Working evidence for Swiss collection, photography, annotation and export teams</p>'
    page+='<p>This report explains what was received, shows the photographs and figures supporting each observation, and identifies the actions that need a decision before the data can be used as the agreed reference for detection work.</p>'
    page+='<section class="kpis">'+''.join(f'<div class="metric"><strong>{number(v)}</strong><span>{esc(name)}</span></div>' for name,v in summary)+'</section>'
    compare=csvrows(gen/'General_Statistics'/'01_Dataset_Overview.csv')
    if compare:
        page+='<h2 id="overview">1. What was received?</h2>'+previewtable(compare,10)
    page+=fig(root,gen/'General_Statistics'/'01_Photos_With_And_Without_Markings.png','July, October collection and processing plant: photographs with and without marked objects')
    support=gen/'General_Statistics'/'04_October_29_Classes.csv'
    if support.exists():
        page+='<details><summary>Show the complete 29-category statistics table</summary>' + previewtable(csvrows(support),50) + '</details>'
    views=gen/'General_Statistics'/'06_Collection_View_Suffixes.csv'
    if views.exists():
        page+='<h3>Collection photograph view types (definitions to confirm)</h3>'+previewtable(csvrows(views),8)
    page+='<p>'+href(root,root/'GENERAL_STATISTICS.html','Open all charts and statistical tables')+'</p>' 
    page+='<h2 id="problems">2. Problems requiring joint review</h2><p>Every issue below follows the same four sections: the problem, evidence, important observations and required actions.</p><ol>'
    page+=''.join(f'<li><a href="#p{i}">{esc(title)}</a></li>' for i,(_,title,*_) in enumerate(PROBLEMS,1))+'</ol>'
    for i,(slug,title,desc,notes,swiss,ccbre) in enumerate(PROBLEMS,1):
        page+=f'<section class="problem" id="p{i}"><h2>{i}. {esc(title)}</h2>'
        page+=f'<div class="subblock"><h3>A. The problem</h3><p>{esc(desc)}</p>' + all_case_button((case_index or {}).get(slug)) + '</div>'
        page+='<div class="subblock"><h3>B. Photographs, tables and numerical evidence</h3>'
        page+=problem_evidence(root,slug,compact=True)+'</div>'
        page+=f'<div class="subblock"><h3>C. Important observations</h3><p>{esc(notes)}</p></div>'
        page+=f'<div class="subblock"><h3>D. Required actions</h3><p><b>Swiss colleagues:</b> {esc(swiss)}</p><p><b>CCBRE:</b> {esc(ccbre)}</p></div>'
        page+=f'<p class="small">{href(root,root/("PROBLEM_%02d.html"%i),"View all available evidence for this problem")}</p></section>'
    page+='<h2>3. Decisions for the meeting</h2><p>Record the correction, responsible person, delivery date and confirmation for each issue. The current action register is a blank decision log, not evidence that problems have been fixed.</p>'
    actions=root/'09_Provider_Corrective_Actions'/'Corrective_Action_Register.csv'
    page+=previewtable(csvrows(actions),12)+f'<p>{href(root,actions,"Download the editable corrective-action register")}</p>'
    page+='<h2>4. When is the data ready?</h2><p>The file inventory is technically consistent. We still need an approved version of the markings, documented category decisions, verified camera dates and orientation practice, and confirmed view/load identifiers. Only after those decisions and a checked export should model performance be measured.</p>'
    basecomp=root/'99_Technical_Verification'/'Frozen_Baseline_Comparison.csv';checks=csvrows(basecomp)
    if checks:
        passed=sum(str(r.get('same_as_frozen_snapshot')).lower()=='true' for r in checks)
        page+=f'<p class="small">Historic-reference consistency: {passed}/{len(checks)} checks match the previous October snapshot. Differences require investigation, not automatic correction.</p>'
    page+=f'<p class="small">Analysis source: standalone dual-dataset run, with no dependence on historical v1.6/v1.7 output folders. Input fingerprints and software versions are in {href(root,root/"99_Technical_Verification"/"Evidence_Run_Manifest.json","the run manifest")}.</p>'
    charts=sorted((gen/'General_Statistics').glob('*.png'))
    if charts:
        page+=f'<h2>5. Complete general-statistics chart gallery ({len(charts)} figures)</h2>'
        page+='<p>The charts below cover image-source counts, category support, camera records, view types, object sizes and the July–October comparison.</p><details><summary>Show all general-statistics figures</summary>'
        page+=''.join(fig(root,p,p.stem.replace('_',' ')) for p in charts)
        page+='</details>'
    (root/'COMPLETE_DATASET_QUALITY_REPORT.html').write_text(page+tail(),encoding='utf-8')
    # Problem pages with ALL generated review examples (lazy-loaded), complete CSV evidence links.
    for i,(slug,title,desc,notes,swiss,ccbre) in enumerate(PROBLEMS,1):
        p=doc_header(title,root)+f'<h1>Problem {i}: {esc(title)}</h1>'
        p+=f'<h2>A. The problem</h2><p>{esc(desc)}</p>' + all_case_button((case_index or {}).get(slug)) + '<h2>B. Full photographic and numerical evidence</h2>'
        p+=problem_evidence(root,slug,compact=False)
        pictures=sorted((root/slug/'Examples').glob('*.jpg'))+sorted((root/slug/'Examples').glob('*.png'))
        if len(pictures)>10:
            p+='<h3>All remaining comparison photographs</h3>'+samples(root,pictures[10:],len(pictures)-10)
        p+=f'<h2>C. Important observations</h2><p>{esc(notes)}</p>'
        p+=f'<h2>D. Required actions</h2><p><b>Swiss colleagues:</b> {esc(swiss)}</p><p><b>CCBRE:</b> {esc(ccbre)}</p>'
        (root/f'PROBLEM_{i:02d}.html').write_text(p+tail(),encoding='utf-8')

def actions(root):
    title='Provider corrective actions and meeting decisions'
    p=doc_header(title,root)+f'<h1>{esc(title)}</h1><p>Use the existing CSV to record who will do what, the agreed correction and its completion date.</p>'
    f=root/'09_Provider_Corrective_Actions'/'Corrective_Action_Register.csv'
    p+=previewtable(csvrows(f),100)+f'<p>{href(root,f,"Download corrective-action CSV")}</p>'
    p+='<h2>Review sheets for signing</h2>'
    for d,name in [('01_Duplicate_Annotation_Problems','Review_List.csv'),('02_Missing_July_Annotations','Review_List.csv'),('03_Class_Mapping_Problems','Review_List.csv'),('04_Empty_Annotation_Problems','Review_List.csv'),('05_Image_Orientation','Orientation_30_Human_Signoff_Ledger.csv'),('06_Camera_Metadata_and_Dates','Collection_Camera_Date_Review.csv'),('07_Image_Context_and_Views','View_Letters_Need_Definitions.csv')]:
        path=root/d/name
        if path.exists():p+='<p>'+href(root,path,d.replace('_',' ')+' — '+name)+'</p>'
    (root/'CORRECTIVE_ACTIONS.html').write_text(p+tail(),encoding='utf-8')

def main(root):
    root=Path(root).resolve()
    required=[root/'00_General_Dataset_Analysis'/'July_Image_Inventory.csv',root/'99_Technical_Verification'/'Full_Dataset_Audit_v171'/'01_overview.json',root/'09_Provider_Corrective_Actions'/'Corrective_Action_Register.csv']
    for path in required:
        if not path.exists():raise FileNotFoundError(f'Incomplete analysis: {path}')
    from all_cases_browser import generate as generate_all_case_browsers
    case_index=generate_all_case_browsers(root)
    general(root);mainreport(root,case_index);actions(root);evidenceindex(root)
    # Make OPEN_FIRST an independent, complete report rather than a link to .md files.
    (root/'OPEN_FIRST.html').write_text((root/'COMPLETE_DATASET_QUALITY_REPORT.html').read_text(encoding='utf-8'),encoding='utf-8')
    overview={'report_generator_version':'2.3','generation_time_utc':datetime.now(timezone.utc).isoformat(),
      'main_html':'OPEN_FIRST.html','complete_html':'COMPLETE_DATASET_QUALITY_REPORT.html',
      'general_statistics_html':'GENERAL_STATISTICS.html','actions_html':'CORRECTIVE_ACTIONS.html',
      'evidence_index':'EVIDENCE_INDEX.html','problem_html_count':len(PROBLEMS),'case_browser_count':len(PROBLEMS),
      'all_cases_manifest':'ALL_CASES_MANIFEST.json',
      'no_external_network_dependencies':True,'read_only_to_source_data':True,
      'caveat':'Human decisions and actual capture dates are not certified by the computation.'}
    (root/'HTML_REPORT_MANIFEST.json').write_text(json.dumps(overview,indent=2),encoding='utf-8')
    print('HTML report ready:',root/'OPEN_FIRST.html')
    print('HTML sections:',len(PROBLEMS),'| Statistics and charts:',root/'GENERAL_STATISTICS.html')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Generate offline HTML partner report from a completed July–October dual-dataset quality audit.')
    parser.add_argument('--analysis',required=True,help='Analysis directory generated by RUN_ALL.ps1')
    args=parser.parse_args()
    main(Path(args.analysis))
