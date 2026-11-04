"""v1.5 integration regression: pixel hashes, cross-group dHash, truncated JPEG,
EXIF orientation, unparsed filename, class/read-only reconciliation.
Uses a temporary synthetic export; no actual partner images or credentials.
"""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image,ImageDraw

from analyze_oct26 import EXPECTED as NAMES


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def make_image():
    im=Image.new('RGB',(320,220),'#c0d39f')
    d=ImageDraw.Draw(im)
    d.rectangle((20,30,190,180),fill='#112240')
    d.ellipse((150,60,305,200),fill='#e1c04a')
    d.line((10,15,309,203), fill='#d93325',width=12)
    ex=im.getexif();ex[274]=6;ex[271]='TEST-MAKE';ex[272]='TEST-MODEL';ex[36867]='2026:10:01 12:00:00'
    return im,ex


def build_source(root, context, files):
    folder=root/('biowaste-collection-images' if context=='collection' else 'biowaste-floorspread-images')
    for sub in ['images','labels','labels-detect']:(folder/sub).mkdir(parents=True,exist_ok=True)
    state={'objects':{},'annotations':{},'images':{},'classes':NAMES,
           'ignore':['wagon'], 'rename':{'Diverses':'Unknown'}}
    count=0; positives=0
    bucket_counts={}
    for idx, (filename, data, positive) in enumerate(files):
        (folder/'images'/filename).write_bytes(data)
        stem=filename.rsplit('.',1)[0]
        (folder/'labels-detect'/(stem+'.txt')).write_text('28 0.5 0.5 0.2 0.3\n' if positive else '')
        (folder/'labels'/(stem+'.txt')).write_text('28 0.4 0.35 0.6 0.35 0.6 0.65 0.4 0.65\n' if positive else '')
        key=filename.replace('__','/',1)
        bid=key.split('/')[0]
        bucket_counts[bid]=bucket_counts.get(bid,0)+1
        anno=f'ann/{idx}#{idx}'
        state['images'][key]=[anno,f'2026-10-28T10:00:{idx:02d}Z',{'28':1} if positive else {},{}]
        state['annotations'][anno]={'object':f'ann/{idx}', 'image':key,
            'updated':f'2026-10-28T10:00:{idx:02d}Z','cancelled':False}
        state['objects'][f'ann/{idx}']='etag'
        count+=1;positives+=bool(positive)
    (folder/'.sync-state.json').write_text(json.dumps(state),encoding='utf-8')
    (folder/'data.yaml').write_text('names:\n'+''.join(f'  {i}: "{v}"\n' for i,v in enumerate(NAMES)))
    source_table='\n'.join('| '+k+' | '+str(v)+' |' for k,v in bucket_counts.items())
    (folder/'report.md').write_text('\n'.join([
       f'# {folder.name}', 'Generated 2026-10-28 10:00 UTC','',
       f'- images: {count} ({count-positives} without labels)',
       f'- labels (polygons): {positives}, mean 1, max 1 per image',
       '- classes: 1 of 29 with labels', '- dropped (label not in class list): 0','',
       '| source bucket | images |','|---|---:|', source_table,'',
       '| id | class | labels | % labels | images | % images |',
       '|---:|---|---:|---:|---:|---:|',
       f'| 28 | BAW Bag | {positives} | 100.0 | {positives} | 50.0 |','']))


with tempfile.TemporaryDirectory(prefix='ccbre_v15_') as tmp:
    root=Path(tmp)/'Raw'; output=Path(tmp)/'results'
    im,ex=make_image()
    original=Path(tmp)/'original.jpg';im.save(original,'JPEG',quality=94,exif=ex)
    base=original.read_bytes()
    comment=b'same pixels, different JPEG COM segment'
    metadata_copy=base[:2]+b'\xff\xfe'+(len(comment)+2).to_bytes(2,'big')+comment+base[2:]
    recode=Path(tmp)/'recode.jpg'; im.save(recode,'JPEG',quality=72,exif=ex)
    recopied=recode.read_bytes()
    corrupt=base[:-280]
    f_collection=[
        ('src__Q1000-1c.jpg',base,True),
        ('src__Q1000-2w.jpg',metadata_copy,True),
        ('src__Q1000-3c.jpg',recopied,True),
        ('src__Q1000-4b.jpg',corrupt,True),
        ('src__weirdname.jpg',base,False),
    ]
    f_floor=[('plant__DSC03316.jpg',recopied,True),
             ('plant__DSC03320.jpg',base,False)]
    build_source(root,'collection',f_collection)
    build_source(root,'floorspread',f_floor)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    proc=subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_oct26.py')),
        '--dataset',str(root),'--output',str(output)],capture_output=True,text=True,timeout=120)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    summary=json.loads((output/'01_overview.json').read_text())
    images=read_csv(output/'02_image_manifest.csv')
    pixel=read_csv(output/'28_pixel_content_duplicates.csv')
    near=read_csv(output/'23_dhash_candidates.csv')
    issues=read_csv(output/'07_quality_issues.csv')
    resolutions=read_csv(output/'05_resolution_statistics.csv')
    objects=read_csv(output/'11_object_manifest.csv')
    oriented=read_csv(output/'30_orientation_review_index.csv')
    assert summary['image_count']==7
    assert summary['metadata_reconciliation_passed'], summary['issues_by_type']
    assert summary['runtime']['pillow_version']
    assert summary['runtime']['dhash_reduction_factor']==16
    assert summary['decoded_pixel_hash_available_count']==6,summary
    assert summary['image_headers_with_dimensions_count']==7,summary
    assert any(int(r['distinct_file_sha256'])>=2 and r['crosses_filename_groups']=='True'
               for r in pixel),pixel
    assert any(r['cross_context']=='True' for r in near),near
    assert any(r['cross_filename_group']=='True' for r in near),near
    assert any(r['issue']=='filename_pattern_unparsed' for r in issues)
    assert any(r['issue']=='image_decode_or_hash_failed' and '4b' in r['image'] for r in issues),issues
    broken=next(r for r in images if '4b.jpg' in r['filename'])
    assert (int(broken['width']),int(broken['height']))==(320,220),broken
    assert not broken['display_pixel_sha256']
    assert (int(broken['display_width']),int(broken['display_height']))==(220,320)
    assert all(int(r['width'])==220 and int(r['height'])==320 for r in resolutions),resolutions
    obj=next(r for r in objects if 'Q1000-1c' in r['image'])
    assert obj['coordinate_frame_assumption']=='display_untransformed'
    assert abs(float(obj['short_side_native_px'])-44.)<.01,obj
    assert abs(float(obj['short_side_native_encoded_px'])-64.)<.01,obj
    assert len(oriented)==5,oriented  # positive orientation-6, truncated positive included but sheet fails
    assert sum(r['status']=='generated' for r in oriented)>=4
    assert all((output/r['sheet']).exists() for r in oriented if r['status']=='generated')
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    assert before==after,'Raw dataset mutated by audit!'
    print('PASS v1.5: decoded-pixel grouping across groups, global dHash across contexts,',
          'header retention for corrupt JPEG, EXIF display sizes, dual overlays, read-only.')
    print('Decoded-pixel groups:',len(pixel),'global dHash candidates:',len(near),
          'orientation sheet records:',len(oriented))
