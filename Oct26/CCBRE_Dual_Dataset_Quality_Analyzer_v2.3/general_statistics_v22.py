"""CCBRE dual-dataset QA 2.2: generate additional GENERAL statistics, tables and plots.

Reads only the saved output of the standalone v2.0 audit, not historical v1.6/1.7 runs.
No original input files are modified. Values describe annotation records, not independently
verified clean/contaminated material. This module can be rerun without image decoding.
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def num(v, default=0):
    try: return int(float(v))
    except (TypeError, ValueError): return default

def nfloat(v, default=0):
    try: return float(v)
    except (TypeError, ValueError): return default

def out_csv(path, data, cols):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(data)

def bar(figpath, names, nums, title, xlabel='Image records', horizontal=True, percent=False):
    plt.figure(figsize=(10.8,max(3.4,.31*len(names)+1.4) if horizontal else 5.2))
    if horizontal:
        ix=np.arange(len(names)); plt.barh(ix, nums);plt.yticks(ix,names,fontsize=9); plt.gca().invert_yaxis(); plt.xlabel(xlabel)
        lim=max(nums or [1]);plt.xlim(0,max(lim*1.21,1));
        for i,v in enumerate(nums):plt.text(v+max(lim,1)*.008,i,f'{v:,.1f}%' if percent else f'{v:,}',va='center',fontsize=8)
    else:
        ix=np.arange(len(names)); plt.bar(ix,nums);plt.xticks(ix,names,rotation=35,ha='right',fontsize=9);plt.ylabel(xlabel)
    plt.title(title,loc='left',pad=13);plt.grid(axis='x' if horizontal else 'y',alpha=.18);plt.tight_layout();plt.savefig(figpath,dpi=175,bbox_inches='tight');plt.close()

def grouped_bar(path,categories,series,title,ylabel='Images',percent=False):
    vals=list(series.items());ix=np.arange(len(categories));w=.8/len(vals)
    plt.figure(figsize=(max(9,0.68*len(categories)+3),5.5))
    for j,(name,data) in enumerate(vals):plt.bar(ix-.4+j*w+w/2,data,width=w,label=name)
    plt.xticks(ix,categories,rotation=30,ha='right');plt.ylabel(ylabel);plt.title(title,loc='left',pad=12)
    plt.legend(frameon=False);plt.grid(axis='y',alpha=.18);plt.tight_layout();plt.savefig(path,dpi=175,bbox_inches='tight');plt.close()

def note_plot(ax,title,x,y):
    ax.set_title(title);ax.grid(alpha=.18)

def main(root:Path):
    root=root.resolve();general=root/'00_General_Dataset_Analysis';base=root/'99_Technical_Verification'/'Full_Dataset_Audit_v171'
    dest=general/'General_Statistics';dest.mkdir(exist_ok=True,parents=True)
    jul=rows(general/'July_Image_Inventory.csv');jcls=rows(general/'July_Class_Statistics.csv');ocl=rows(general/'October_Class_Statistics.csv')
    ctx=rows(general/'October_Context_Statistics.csv');octim=rows(base/'02_image_manifest.csv');obj=rows(base/'11_object_manifest.csv')
    vr=rows(root/'07_Image_Context_and_Views'/'Collection_View_Suffix_Counts.csv')
    capture=rows(root/'06_Camera_Metadata_and_Dates'/'EXIF_Capture_Months.csv')
    src=rows(root/'07_Image_Context_and_Views'/'Plant_Photos_by_Capture_Day.csv')
    ref=json.loads((general/'July_October_Annotation_Reconciliation.json').read_text(encoding='utf-8'))
    july_pos=sum(num(r['material_rows'])>0 for r in jul);july_material=sum(num(r['material_rows']) for r in jul)
    oct_pos=sum(num(r['positive_images']) for r in ctx);oct_im=sum(num(r['image_count']) for r in ctx)
    if not jul or not octim:
        raise RuntimeError('July or October dataset inventory is empty; cannot produce statistics.')
    if sum(num(r['detection_instances']) for r in octim) != len(obj):
        raise RuntimeError('October image-level and object-level detection totals disagree.')
    # Historic snapshot expectations are checked in Frozen_Baseline_Comparison.csv, not assumed for new exports.
    # Primary source table (data populations have different denominators)
    overview=[
      {'population':'July source photographs (foreign-material annotations only)','image_records':len(jul),'with_markings':july_pos,'no_foreign_markings':len(jul)-july_pos,'objects':july_material,'positive_share_pct':round(100*july_pos/max(1,len(jul)),1),'note':f'Original July labels include {sum(num(r["context_rows"]) for r in jul):,} scene/context boxes; excluded from foreign-material object tally.'},
    ]
    for c in ctx:
        n=num(c['image_count']);p=num(c['positive_images']); overview.append({'population':'October '+c['context'],'image_records':n,'with_markings':p,'no_foreign_markings':num(c['explicit_empty_images']),'objects':num(c['detection_objects']),'positive_share_pct':round(100*p/n,1),'note':'Includes repeated file records where present; not a verified contamination prevalence.'})
    overview.append({'population':'October export total','image_records':oct_im,'with_markings':oct_pos,'no_foreign_markings':oct_im-oct_pos,'objects':len(obj),'positive_share_pct':round(100*oct_pos/max(1,oct_im),1),'note':'July photographs are already included in October collection.'})
    out_csv(dest/'01_Dataset_Overview.csv',overview,list(overview[0]))
    grouped_bar(dest/'01_Photos_With_And_Without_Markings.png',[r['population'].replace(' (foreign-material annotations only)','') for r in overview[:3]],{'Marked foreign material':[r['with_markings'] for r in overview[:3]],'No marked foreign material':[r['no_foreign_markings'] for r in overview[:3]]},'Photographs with and without exported foreign-material markings')
    # Counts as a percentage for denominator-safe comparison
    bar(dest/'02_Share_With_Markings.png',[r['population'].replace(' (foreign-material annotations only)','') for r in overview[:3]],[r['positive_share_pct'] for r in overview[:3]],'Share of photographs with at least one material marking',xlabel='Percent of image records',percent=True)
    # July class counts: non-context only, and October counts with separate sets
    jul_classes=[{'class_name':r['july_class'],'july_objects':num(r['objects'])} for r in jcls if r['july_class'].casefold() not in ('container','wagon')]
    jul_classes=sorted(jul_classes,key=lambda r:-r['july_objects'])
    out_csv(dest/'03_July_Foreign_Material_Classes.csv',jul_classes,['class_name','july_objects'])
    bar(dest/'03_July_Material_Counts.png',[r['class_name'] for r in jul_classes],[r['july_objects'] for r in jul_classes],'July foreign-material objects by original category',xlabel='Object markings')
    by_cls=defaultdict(dict)
    for r in ocl:by_cls[r['class_name']][r['context']]=r
    classes=[]
    for name,parts in by_cls.items():
        n1=num(parts.get('collection',{}).get('instances'));n2=num(parts.get('floorspread',{}).get('instances'))
        classes.append({'category':name,'collection_objects':n1,'floorspread_objects':n2,'total_objects':n1+n2,'collection_images_with_category':num(parts.get('collection',{}).get('images_containing_class')),'floorspread_images_with_category':num(parts.get('floorspread',{}).get('images_containing_class'))})
    classes.sort(key=lambda d:-d['total_objects'])
    out_csv(dest/'04_October_29_Classes.csv',classes,list(classes[0]))
    # Side-by-side FULL 29 class counts by context, horizontal grouped
    names=[c['category'] for c in classes];inds=np.arange(len(names));fig,ax=plt.subplots(figsize=(11,11.8))
    ax.barh(inds-.2,[c['collection_objects'] for c in classes],.38,label='Collection')
    ax.barh(inds+.2,[c['floorspread_objects'] for c in classes],.38,label='Processing plant')
    ax.set_yticks(inds,names);ax.invert_yaxis();ax.set_xlabel('Foreign-material markings (exported image records)');ax.set_title('All 29 October material categories by source',loc='left');ax.legend(frameon=False);ax.grid(axis='x',alpha=.18);fig.tight_layout();fig.savefig(dest/'04_October_29_Classes_By_Context.png',dpi=175,bbox_inches='tight');plt.close(fig)
    # Category support ratio chart: top 12
    supp=classes[:12]
    grouped_bar(dest/'05_Top12_Category_Image_Support.png',[c['category'] for c in supp],{'Collection':[c['collection_images_with_category'] for c in supp], 'Processing plant':[c['floorspread_images_with_category'] for c in supp]},'How many photographs contain each common category',ylabel='Photographs with category')
    # Views from suffix: record percentages and counts
    view=[]
    for r in vr:
        n=num(r['images']);p=num(r['positive']);view.append({'view_suffix':r['view_letter'],'photos':n,'with_markings':p,'no_markings':num(r['explicit_empty']),'share_with_markings_pct':round(p/max(1,n)*100,1),'objects':num(r['objects']),'meaning':'To be confirmed by photography team'})
    out_csv(dest/'06_Collection_View_Suffixes.csv',view,list(view[0]))
    grouped_bar(dest/'06_Collection_View_Markings.png',[r['view_suffix'] for r in view],{'With markings':[r['with_markings'] for r in view], 'No markings':[r['no_markings'] for r in view]},'Collection filename view letters (definitions require partner confirmation)')
    bar(dest/'07_Collection_View_Marking_Share.png',[r['view_suffix'] for r in view],[r['share_with_markings_pct'] for r in view],'Share of collection photographs with markings by view suffix',xlabel='Percent of image records',percent=True)
    # Cameras, source and months, all metadata unverified actual dates
    make=Counter((r['context'],r.get('exif_camera_make','?'),r.get('exif_camera_model','?')) for r in octim)
    cameras=[{'context':k[0],'camera_make':k[1],'camera_model':k[2],'images':v} for k,v in sorted(make.items())]
    out_csv(dest/'08_Camera_Models.csv',cameras,['context','camera_make','camera_model','images'])
    bar(dest/'08_Camera_Models.png',[f"{r['context']} | {r['camera_make']} {r['camera_model']}" for r in cameras],[r['images'] for r in cameras],'Cameras recorded in original image metadata')
    # month rows in EXIF; no claim that actual capture year is right
    months=defaultdict(dict)
    for r in capture:months[r['exif_year_month']][r['context']]=num(r['photos'])
    month_keys=sorted(months)
    out_csv(dest/'09_Recorded_Camera_Months.csv',[{'exif_month':k,'collection':months[k].get('collection',0),'floorspread':months[k].get('floorspread',0)} for k in month_keys],['exif_month','collection','floorspread'])
    plt.figure(figsize=(12,4.6));plt.plot(month_keys,[months[k].get('collection',0) for k in month_keys],marker='o',label='Collection');plt.plot(month_keys,[months[k].get('floorspread',0) for k in month_keys],marker='s',label='Processing plant');plt.xticks(rotation=60,ha='right');plt.ylabel('Image records');plt.title('Months recorded by camera (actual dates require verification)',loc='left');plt.legend(frameon=False);plt.grid(alpha=.18);plt.tight_layout();plt.savefig(dest/'09_Camera_Recorded_Months.png',dpi=175,bbox_inches='tight');plt.close()
    # Orientation displayed physical dimensions
    dims=Counter((r['context'],num(r.get('display_width') or r.get('width')),num(r.get('display_height') or r.get('height'))) for r in octim)
    dimrows=[{'context':k[0],'display_dimensions':f'{k[1]} × {k[2]}','display_width':k[1],'display_height':k[2],'photos':n,'direction':'landscape' if k[1]>k[2] else ('portrait' if k[1]<k[2] else 'square')} for k,n in sorted(dims.items(),key=lambda x:(x[0][0],-x[1]))]
    out_csv(dest/'10_Image_Display_Resolutions.csv',dimrows,list(dimrows[0]))
    bar(dest/'10_Image_Display_Resolutions.png',[f"{r['context']}: {r['display_dimensions']}" for r in dimrows],[r['photos'] for r in dimrows],'Actual displayed image resolutions',xlabel='Photographs')
    orient=Counter((r['context'],str(r.get('exif_orientation','')).strip() or '?') for r in octim)
    orows=[{'context':c,'exif_orientation':o,'photos':v} for (c,o),v in sorted(orient.items())]
    out_csv(dest/'11_EXIF_Orientation_Codes.csv',orows,['context','exif_orientation','photos'])
    grouped_bar(dest/'11_Orientation_Code_Counts.png',sorted({o for _,o in orient}),{c:[orient[(c,o)] for o in sorted({o for _,o in orient})] for c in ['collection','floorspread']},'Camera orientation codes (different from portrait/landscape)')
    # Distribution markings per picture, do NOT count context labels, and separate 0,1,2,3,4,5,6-10,11+
    def bucket(n): return str(n) if n<6 else ('6–10' if n<=10 else '11+')
    bins=['0','1','2','3','4','5','6–10','11+'];nper=defaultdict(Counter)
    for r in jul:nper['July'][bucket(num(r['material_rows']))]+=1
    for r in octim:nper[r['context']][bucket(num(r['detection_instances']))]+=1
    out_csv(dest/'12_Objects_Per_Image_Distribution.csv',[{'dataset':c,'foreign_material_objects_bucket':b,'photos':nper[c][b]} for c in ['July','collection','floorspread'] for b in bins],['dataset','foreign_material_objects_bucket','photos'])
    grouped_bar(dest/'12_Objects_Per_Photo.png',bins,{c:[nper[c][b] for b in bins] for c in ['July','collection','floorspread']},'Number of marked foreign objects per photograph')
    # Q-code group summaries (even if Q semantic unknown)
    qc=Counter((r['q_code'] or 'UNPARSED') for r in octim if r['context']=='collection')
    qpos=Counter((r['q_code'] or 'UNPARSED') for r in octim if r['context']=='collection' and num(r['detection_instances'])>0)
    qrows=[{'q_code':k,'photo_records':v,'with_markings':qpos[k],'share_marked_pct':round(qpos[k]/v*100,1)} for k,v in qc.most_common()]
    out_csv(dest/'13_Collection_Q_Code_Counts.csv',qrows,list(qrows[0]))
    bar(dest/'13_Largest_Collection_Q_Codes.png',[r['q_code'] for r in qrows[:15]],[r['photo_records'] for r in qrows[:15]],'Largest collection filename groups (meaning to confirm)')
    # Plant capture day structure as list (already saved)
    out_csv(dest/'14_Processing_Plant_Capture_Days.csv',[{'recorded_day':r['day'],'photos':num(r['photos'])} for r in src],['recorded_day','photos'])
    bar(dest/'14_Plant_Camera_Days.png',[r['day'] for r in src],[num(r['photos']) for r in src],'Processing-plant photographs by recorded capture day')
    # Foreign object shorter-side at 1536; bins
    scales=['<16','16–<32','32–<64','64–<128','128–<256','≥256']; cnt=defaultdict(Counter)
    for o in obj:
        v=nfloat(o.get('short_side_1536_px'))
        b=scales[0] if v<16 else scales[1] if v<32 else scales[2] if v<64 else scales[3] if v<128 else scales[4] if v<256 else scales[5]
        cnt[o['context']][b]+=1
    out_csv(dest/'15_Object_Size_1536px.csv',[{'context':c,'shorter_side_pixels':b,'marked_objects':cnt[c][b]} for c in ['collection','floorspread'] for b in scales],['context','shorter_side_pixels','marked_objects'])
    grouped_bar(dest/'15_Object_Size_1536px.png',scales,{c:[cnt[c][b] for b in scales] for c in ['collection','floorspread']},'How large marked objects appear after resizing photographs',ylabel='Objects')
    # Box count differences reference range: no misleading composite from mixed rules
    selected=[{'source':'July original foreign-material labels','objects':july_material},{'source':'October boxes on same July images: fewest-version choice','objects':num(ref['observed_oct_box_count_min'])},{'source':'October boxes on same July images: most-version choice','objects':num(ref['observed_oct_box_count_max'])}]
    out_csv(dest/'16_Same_July_Images_Markings.csv',selected,['source','objects'])
    bar(dest/'16_Markings_On_Same_July_Images.png',[x['source'] for x in selected],[x['objects'] for x in selected],'Comparison on the same 3,359 July photographs',xlabel='Foreign-material markings')
    # Requested supplementary figures: all 29 category-image support, by-context source image counts,
    # and a correctly normalised histogram with equal-width bins in log10(area).
    allrows=[r for r in ocl if r['context']=='all']
    if allrows:
        allrows.sort(key=lambda r:-num(r['images_containing_class']))
        out_csv(dest/'17_All_Class_Image_Support.csv',
                [{'category':r['class_name'],'image_records_with_class':num(r['images_containing_class'])}
                 for r in allrows],['category','image_records_with_class'])
        bar(dest/'17_All_Class_Image_Support.png',[r['class_name'] for r in allrows],
            [num(r['images_containing_class']) for r in allrows],
            'Number of image records containing each October material category',xlabel='Image records containing category')
    source_counts={r['context']:num(r['image_count']) for r in ctx}
    out_csv(dest/'18_Image_Records_By_Source.csv',
            [{'context':k,'image_records':v} for k,v in sorted(source_counts.items())],
            ['context','image_records'])
    bar(dest/'18_Image_Records_By_Source.png',list(source_counts),list(source_counts.values()),
        'October image records by source',xlabel='Image records')
    areas=defaultdict(list)
    for o in obj:
        a=nfloat(o.get('bbox_area_frac'),0)*100
        if a>0 and math.isfinite(a):areas[o['context']].append(math.log10(a))
    if any(areas.values()):
        allvals=[v for values in areas.values() for v in values]
        low,high=min(allvals),max(allvals)
        if low==high:high=low+1
        edges=np.linspace(low,high,65)
        fig,ax=plt.subplots(figsize=(11.4,5.5))
        records=[]
        for ctx_name in ('collection','floorspread'):
            vals=areas[ctx_name]
            if not vals:continue
            counts,_=np.histogram(vals,bins=edges)
            fraction=counts/len(vals)
            mid=(edges[:-1]+edges[1:])/2
            ax.step(mid,fraction,where='mid',label=f'{ctx_name} (n={len(vals):,})',linewidth=1.8)
            records.extend({'context':ctx_name,'log10_area_bin_left':round(float(edges[i]),6),
                'log10_area_bin_right':round(float(edges[i+1]),6),'objects':int(counts[i]),
                'fraction_within_context':round(float(fraction[i]),8)} for i in range(len(counts)))
        out_csv(dest/'19_Box_Area_Log_Bin_Fractions.csv',records,
                ['context','log10_area_bin_left','log10_area_bin_right','objects','fraction_within_context'])
        ax.set_title('Relative size of marked objects (equal-width logarithmic area bins)',loc='left')
        ax.set_xlabel('log10(box area as percent of full image)');ax.set_ylabel('Fraction of objects in each bin')
        ax.grid(alpha=.2);ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(dest/'19_Box_Area_Log_Bin_Fractions.png',dpi=175,bbox_inches='tight');plt.close(fig)
    # Summary sanity checks - independent regenerated outputs
    summary={'analyzer_extension':'2.2','derived_from':'Standalone current run; no prior run required','july_original_image_records':len(jul),'july_foreign_material_objects':july_material,'july_images_with_foreign_markings':july_pos,'october_image_records':len(octim),'october_box_rows':len(obj),'october_collection_records':sum(r['context']=='collection' for r in octim),'october_floorspread_records':sum(r['context']=='floorspread' for r in octim),'exif_months_are_verified_actual_capture_dates':False,'with_markings_means_verified_contamination':False,'class_name_mapping_july_october_automatically_merged':False,'charts_and_tables':len(list(dest.glob('*.png')))+len(list(dest.glob('*.csv')))}
    (dest/'00_Supplement_Overview.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    readme='''# General figures and tables (v2.2)\n\nGenerated from the completed standalone v2.0 output, without re-decoding images.\nThe numbered CSV tables and paired charts are additional evidence; none changes the original auditing results.\n\n**Please interpret carefully:** A file containing no foreign-material markings is not necessarily a clean photo.\nCollection and processing-plant counts are photo *records*, including repeated file records.\nJuly photos are already present within the October collection export and MUST NOT be added to October counts to claim total new photographs.\nDo not compare raw July category IDs with October IDs. July's 22,528 Container/Wagon context boxes are excluded from July material figures.\nThe EXIF months describe camera-clock entries, not verified collection dates.\nFilename view suffixes c/w/t/b and Q-code meaning await Swiss partner confirmation.\nObjects per image counts refer to foreign-material boxes, not July context labels.\nAll percentages use their respective context/period denominator.\n\nGeneral charts include 01–19. Chart 19 uses fractions per equal log10-area bin (not misleading linear-width density normalisation).\nDetailed category tables: 03 and 04. All numeric tables are CSV for review and further use.\n\nRun: py -3 general_statistics_v22.py --analysis "<your Dual_Dataset_Quality_v2.0 directory>"\n'''
    (dest/'READ_ME_FIRST.md').write_text(readme,encoding='utf-8')
    entry=root/'OPEN_FIRST.html'
    if entry.exists():
        txt=entry.read_text(encoding='utf-8')
        link='<p><strong>Additional July–October general figures and tables:</strong> <a href="00_General_Dataset_Analysis/General_Statistics/READ_ME_FIRST.md">Open the v2.1 supplement</a>.</p>'
        if 'General_Statistics/READ_ME_FIRST.md' not in txt:
            txt=txt.replace('</body>',link+'</body>')
            entry.write_text(txt,encoding='utf-8')
    print('Additional tables:',len(list(dest.glob('*.csv'))),'charts:',len(list(dest.glob('*.png'))),'folder:',dest)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',required=True,help='Existing Dual_Dataset_Quality_v2.0 output folder')
    args=parser.parse_args();main(Path(args.analysis))
