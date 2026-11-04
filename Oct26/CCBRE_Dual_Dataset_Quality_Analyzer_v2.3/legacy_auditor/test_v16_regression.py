"""v1.6 integration tests: unique-pixel dHash, histogram, camera/time runs,
July reference overlap and strict/extended group exports. No real data used."""
import csv, json, hashlib, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import analyze_oct26 as a


def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def make_im(ori=1,make='SONY',model='A7',date='2026:10:01 10:00:00'):
    im=Image.new('RGB',(200,150),'white');d=ImageDraw.Draw(im)
    d.ellipse((20,30,115,120), fill='#215543');d.rectangle((95,15,175,130),fill='#483e25')
    ex=im.getexif();ex[274]=ori;ex[271]=make;ex[272]=model
    ex[36867]=date
    return im,ex


def save_jpg(where,im,ex,quality=93):
    im.save(where,quality=quality,exif=ex)
    return where.read_bytes()


def build_export(root,context,records):
    folder=root/('biowaste-collection-images' if context=='collection' else 'biowaste-floorspread-images')
    for sub in ('images','labels','labels-detect'):(folder/sub).mkdir(parents=True,exist_ok=True)
    sync=dict(objects={},annotations={},images={},classes=a.EXPECTED,ignore=[],rename={})
    sources={}
    for i,(name,payload) in enumerate(records):
        (folder/'images'/name).write_bytes(payload)
        stem=Path(name).stem
        (folder/'labels-detect'/(stem+'.txt')).write_text('28 0.5 0.5 0.2 0.3\n')
        (folder/'labels'/(stem+'.txt')).write_text('28 0.4 0.35 0.6 0.35 0.6 0.65 0.4 0.65\n')
        key=name.replace('__','/',1)
        sync['images'][key]=[f'ann/{i}#1','2026-10-28T10:00:00Z',{'28':1},{}]
        sync['objects'][f'ann/{i}']='etag'
        sync['annotations'][f'ann/{i}#1']={'object':f'ann/{i}','image':key,'updated':'2026-10-28T10:00:00Z','cancelled':False}
        source=name.split('__')[0]; sources[source]=sources.get(source,0)+1
    (folder/'data.yaml').write_text('names:\n'+''.join(f'  {i}: "{n}"\n' for i,n in enumerate(a.EXPECTED)))
    (folder/'.sync-state.json').write_text(json.dumps(sync))
    n=len(records)
    (folder/'report.md').write_text('\n'.join(['# EXPORT','Generated 2026-10-28 10:00 UTC',
        f'- images: {n} (0 without labels)', f'- labels (polygons): {n}, mean 1, max 1 per image',
        '- classes: 1 of 29 with labels','- dropped (label not in class list): 0','',
        '| source bucket | images |','|---|---:|',
        *[f'| {s} | {v} |' for s,v in sources.items()], '',
        '| id | class | labels | % labels | images | % images |',
        '|---:|---|---:|---:|---:|---:|',
        f'| 28 | BAW Bag | {n} | 100.0 | {n} | 100.0 |']))


with tempfile.TemporaryDirectory() as tmp:
    t=Path(tmp);raw=t/'Raw';out=t/'Analysis';refout=t/'Reference';refer=t/'JulySource';refer.mkdir()
    im,ex=make_im();original=save_jpg(t/'original.jpg',im,ex)
    comment=b'EXIF comment changed only'
    metadata_copy=original[:2]+b'\xff\xfe'+(len(comment)+2).to_bytes(2,'big')+comment+original[2:]
    reencoded=save_jpg(t/'recode.jpg',im,ex,quality=73)
    im2,ex2=make_im(make='iPhone',model='Model2')
    ImageDraw.Draw(im2).rectangle((10,10,60,60),fill='#bd0033')
    second_camera=save_jpg(t/'second.jpg',im2,ex2)
    im3,ex3=make_im(make='SONY',model='A7',date='2026:10:01 10:26:00')
    same_camera_later=save_jpg(t/'later.jpg',im3,ex3)
    build_export(raw,'collection',[
       ('source__Q1000-1c.jpg',original),('source__Q1000-1w.jpg',metadata_copy),
       ('source__Q1000-2c.jpg',reencoded)])
    build_export(raw,'floorspread',[
       ('plant__DSC03316.jpg',original),('plant__DSC03317.jpg',same_camera_later),
       ('plant__DSC03318.jpg',second_camera)])
    (refer/'July_original.jpg').write_bytes(original)
    (refer/'July_other.jpg').write_bytes(second_camera)
    before={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in raw.rglob('*') if x.is_file()}
    cmd=[sys.executable,str(Path(a.__file__))]
    r=subprocess.run(cmd+['--fingerprint-only','--images-root',str(refer),'--output',str(refout)],
                     capture_output=True,text=True,timeout=90)
    assert r.returncode==0,(r.stdout,r.stderr)
    r=subprocess.run(cmd+['--dataset',str(raw),'--output',str(out),
            '--reference-manifest',str(refout/'31_fingerprint_manifest.csv')],
            capture_output=True,text=True,timeout=120)
    assert r.returncode==0,(r.stdout,r.stderr)
    images=load_csv(out/'02_image_manifest.csv'); assert len(images)==6
    assert json.loads((out/'01_overview.json').read_text())['metadata_reconciliation_passed']
    candidates=load_csv(out/'23_dhash_candidates.csv'); distances=load_csv(out/'29_dhash_distance_distribution.csv')
    overview=json.loads((out/'01_overview.json').read_text())
    scan=overview['dhash_candidate_scan']
    assert scan['unique_decoded_pixel_representatives']==3,scan
    assert scan['removed_redundant_image_rows']==3,scan
    assert scan['pairs_considered']==3,scan
    assert sum(int(row['pairs']) for row in distances)==3,distances
    assert any(int(row['pixel_group_size_a'])>1 or int(row['pixel_group_size_b'])>1 for row in candidates)
    assert any('Q1000-1w' in row['pixel_group_members_a']+row['pixel_group_members_b'] for row in candidates)
    g=load_csv(out/'31_split_group_manifest.csv');assert len(g)==6
    assert all(x['split_group_strict'].startswith('SGS-') for x in g)
    assert any(x['split_group_extended']!=x['split_group_strict'] for x in g)
    assert any(int(x['strict_group_size'])>=4 for x in g),g
    assert max(int(x['extended_group_size']) for x in g)>=max(int(x['strict_group_size']) for x in g)
    plant=[x for x in images if x['context']=='floorspread']
    assert len({x['camera_identity_key'] for x in plant})==2,plant
    sony=[x for x in plant if 'sony' in x['camera_identity_key']]
    assert len(sony)==2 and len({x['file_group_key'] for x in sony})==1,sony
    assert len({x['time_gap_group_key'] for x in sony})==2,sony
    assert overview['inferred_contiguous_DSC_runs']==2,overview
    strong=load_csv(out/'32_reference_strong_matches.csv')
    assert any(x['method']=='exact_file_sha256' for x in strong),strong
    assert any(x['method']=='identical_decoded_pixels' for x in strong),strong
    refsummary=json.loads((out/'34_reference_overlap_summary.json').read_text())
    assert refsummary['decoded_comparison_permitted'],refsummary
    assert refsummary['oct_images_with_strong_overlap']>=3,refsummary
    assert refsummary['oct_images_without_strong_overlap']>0,refsummary
    assert len(load_csv(out/'35_split_group_size_distribution.csv'))>0
    rebuilt=t/'Rebuilt'
    post=subprocess.run([sys.executable,str(Path(a.__file__).with_name('rebuild_split_groups.py')),
        '--analysis',str(out),'--dhash-threshold','6','--output',str(rebuilt)],
        capture_output=True,text=True,timeout=60)
    assert post.returncode==0,(post.stdout,post.stderr)
    orig={r['image']:(r['split_group_strict'],r['split_group_extended']) for r in g}
    regen={r['image']:(r['split_group_strict'],r['split_group_extended']) for r in load_csv(rebuilt/'31_split_group_manifest.csv')}
    assert orig==regen,(orig,regen)
    after={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in raw.rglob('*') if x.is_file()}
    assert before==after,'Raw modified'
    # Header EXIF failure should not suppress pixel hashes; force orientation failure.
    original_exif=a.ImageOps.exif_transpose
    try:
        def fail(*args,**kwargs):raise ValueError('synthetic EXIF failure')
        a.ImageOps.exif_transpose=fail
        findings=[]
        w,h,values=a.inspect_image(t/'original.jpg',lambda *x:findings.append(x),'orig')
        assert w==200 and h==150 and values['display_pixel_sha256'] and values['dhash64']
        assert values['pixel_coordinate_frame']=='encoded_fallback'
        assert any(x[1]=='exif_transpose_failed_encoded_fallback' for x in findings)
    finally:a.ImageOps.exif_transpose=original_exif
    print('PASS v1.6: de-duplicated dHash, Hamming distribution, per-camera numeric/time sessions,')
    print('strict/extended group IDs, July fingerprint/reference match, EXIF fallback, read-only.')
