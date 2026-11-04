"""Test July labels, class policies, two-version IoU bounds, day groups, mapping second pass."""
import csv, json, sys, tempfile, subprocess
from pathlib import Path
from v17_extensions import reference_label_diff, day_level_groups

def csvwrite(path, fields, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def boxline(cid,xc,yc,w,h):return f'{cid} {xc} {yc} {w} {h}\n'
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);labels=root/'labels';(labels/'train'/'labels').mkdir(parents=True)
    ref=root/'july_fingerprints';ref.mkdir()
    # July has: a Container+Paper+Other, b Wagon+BAW-Sack+Insulation, c missing label
    (labels/'train'/'labels'/'Q1000-1c.txt').write_text(boxline(0,.5,.5,.8,.8)+boxline(2,.5,.5,.2,.2)+boxline(4,.1,.1,.1,.1))
    (labels/'train'/'labels'/'Q1000-2w.txt').write_text(boxline(1,.5,.5,.8,.8)+boxline(3,.4,.4,.2,.2)+boxline(5,.6,.6,.1,.1))
    (root/'data.yaml').write_text('names: [Container, Wagon, Papier, BAW-Sack, Other, Styropor/Dämmung]\n',encoding='utf8')
    csvwrite(ref/'31_fingerprint_manifest.csv',['image','source_path','sha256','display_pixel_sha256','pixel_coordinate_frame'],[
        dict(image='reference01/Q1000-1c.jpg',source_path='C:/July/Q1000-1c.jpg',sha256='sha-a',display_pixel_sha256='',pixel_coordinate_frame='EXIF_display'),
        dict(image='reference01/Q1000-2w.jpg',source_path='C:/July/Q1000-2w.jpg',sha256='sha-b',display_pixel_sha256='',pixel_coordinate_frame='EXIF_display'),
        dict(image='reference01/Q1000-3w.jpg',source_path='C:/July/Q1000-3w.jpg',sha256='sha-c',display_pixel_sha256='',pixel_coordinate_frame='EXIF_display')])
    octimgs=[]
    def octrow(image,sha,group,context='collection',dt=''):
        return dict(image=image,context=context,sha256=sha,display_pixel_sha256='',pixel_coordinate_frame='EXIF_display',
                    file_group_key=group,exif_datetime_original=dt)
    octimgs += [octrow('collection/a1.jpg','sha-a','collection|Q1000|1'),octrow('collection/a2.jpg','sha-a','collection|Q1000|1'),
                octrow('collection/b.jpg','sha-b','collection|Q1000|2'),
                octrow('floorspread/x.jpg','sha-x','floorspread|run1','floorspread','2026:06:16 12:00:00'),
                octrow('floorspread/y.jpg','sha-y','floorspread|run2','floorspread','2026:06:16 13:00:00'),
                octrow('floorspread/z.jpg','sha-z','floorspread|run3','floorspread','2026:07:01 10:00:00')]
    octboxes=[]
    def add(image,cid,x=.4,y=.4,X=.6,Y=.6):
        octboxes.append(dict(image=image,class_id=str(cid),xmin=x,ymin=y,xmax=X,ymax=Y,line=1))
    add('collection/a1.jpg',13)
    # Version 2 of the identical photo has no boxes; must report bounds 0..1.
    add('collection/b.jpg',28,.3,.3,.5,.5)
    dest=root/'results';dest.mkdir()
    gm=day_level_groups(octimgs,dest)
    assert gm['group_count']==4 and gm['largest_group']==2,(gm)
    result=reference_label_diff(ref/'31_fingerprint_manifest.csv',labels,root/'data.yaml',octimgs,octboxes,
                                {13:'Paper',28:'BAW Bag',22:'Insulation'},None,dest)
    assert result['reference_images']==3,result
    assert result['reference_images_with_valid_labels']==2,result
    assert result['july_observed_box_or_polygon_rows']==6,result
    assert result['july_context_rows']==2,result
    assert result['july_mapped_material_rows']==2,result
    assert result['july_unmapped_rows']==2,result
    assert result['july_images_with_conflicting_october_versions']==1,result
    assert result['class_aware_iou05_match_min']==1 and result['class_aware_iou05_match_max']==2,result
    assert result['reference_missing_label_images']==1,result
    mapcsv=root/'mapping.csv'
    csvwrite(mapcsv,['july_class','october_class'],[dict(july_class='Styropor/Dämmung',october_class='Insulation')])
    alt=root/'remap';alt.mkdir()
    result2=reference_label_diff(ref/'31_fingerprint_manifest.csv',labels,root/'data.yaml',octimgs,octboxes,
                                {13:'Paper',28:'BAW Bag',22:'Insulation'},mapcsv,alt)
    assert result2['july_mapped_material_rows']==3 and result2['july_unmapped_rows']==1,result2
    # Other remains unmapped even when no manual mapping is specified.
    classes=list(csv.DictReader((alt/'45_reference_class_diff_counts.csv').open(encoding='utf-8-sig')))
    assert next(x for x in classes if x['july_class']=='Other')['category']=='unmapped_ambiguous'
    # CLI fast second pass reads saved evidence; does not need or decode raw images.
    inp=root/'saved_analysis'; inp.mkdir()
    csvwrite(inp/'02_image_manifest.csv',list(octimgs[0]),octimgs)
    csvwrite(inp/'11_object_manifest.csv',list(octboxes[0]),octboxes)
    (inp/'13_taxonomy_mappings.json').write_text(json.dumps({'collection':{'13':'Paper','28':'BAW Bag','22':'Insulation'}}))
    cli_dest=root/'cli_derived'
    proc=subprocess.run([sys.executable,str(Path(__file__).parent/'analyze_oct26.py'),
        '--label-diff-only','--analysis-input',str(inp),'--output',str(cli_dest),
        '--reference-manifest',str(ref/'31_fingerprint_manifest.csv'),
        '--reference-labels',str(labels),'--reference-yaml',str(root/'data.yaml')],
        capture_output=True,text=True)
    assert proc.returncode==0,(proc.stdout,proc.stderr)
    assert (cli_dest/'39_reference_label_diff_summary.json').is_file()
    assert (cli_dest/'38_day_level_group_summary.json').is_file()
    assert not (inp/'39_reference_label_diff_summary.json').exists(), 'Read-only input changed'
    print('PASS v1.7: day groups, July classes/contexts, unmapped Other, multi-version bounds, IoU matching, custom map.')
