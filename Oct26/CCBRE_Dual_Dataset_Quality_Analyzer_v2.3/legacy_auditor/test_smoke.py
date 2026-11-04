"""Integration smoke test for v1.3. Generates tiny input, does not touch real Raw."""
from pathlib import Path
import tempfile,subprocess,sys,json,csv
from PIL import Image

NAMES=['Plastic Bag','Coffee Capsule','Plastic Flower Pot','PET 0.5l','PET 1.5l','Glass Bottle',
       'Aluminum Can','Cat Litter','Diaper','Garbage Bag 17l','Garbage Bag 35l','Garbage Bag 65l',
       'Plastic Tarp','Paper','Cardboard','Cigarettes','Textiles','Plastic Food Packaging',
       'Tin Can','Aluminum Foil','String','Soil Bags','Insulation','Rubber Bands','Misc Metal',
       'Misc Ceramic','Misc Plastic','Unknown','BAW Bag']

def create_dataset(base, label_mismatch=False):
    for ctx,folder in [('collection','biowaste-collection-images'),('floorspread','biowaste-floorspread-images')]:
        src=base/folder
        for child in ['images','labels','labels-detect']:(src/child).mkdir(parents=True,exist_ok=True)
        names=[('site__a_c.png', 'site/a_c.png',0),('site__b_w.png','site/b_w.png',1)] if ctx=='collection' else [('floor__x.png','floor/x.png',1)]
        state={'objects':{},'annotations':{},'images':{},'classes':NAMES, 'rename':{'Papier':'Paper'},'ignore':['wagon']}
        class_counts={0:0,13:0,28:0}; pos=0
        for idx,(filename,source_key,has_obj) in enumerate(names):
            # identical image contents in 2 collection photos to exercise duplicate detection
            im=Image.new('RGB',(120,80),(150,100,75))
            im.save(src/'images'/filename)
            txt_name=filename.removesuffix('.png')+'.txt'
            classes={'28':1} if has_obj else {}
            if has_obj:
                (src/'labels-detect'/txt_name).write_text('28 0.5 0.5 0.2 0.3\n')
                (src/'labels'/txt_name).write_text('28 0.4 0.35 0.6 0.35 0.6 0.65 0.4 0.65\n')
                pos+=1
                class_counts[28]+=1
            else:
                (src/'labels-detect'/txt_name).write_text('')
                (src/'labels'/txt_name).write_text('')
            ref=f'annot/{idx}#{idx}'
            state['images'][source_key]=[ref,f'2026-10-20T10:00:{idx:02d}Z',classes,{}]
            state['annotations'][ref]={'object':f'annot/{idx}','image':source_key,
                                       'updated':f'2026-10-20T10:00:{idx:02d}Z','cancelled':False}
            state['objects'][f'annot/{idx}']='example-etag'
        if label_mismatch and ctx=='collection':
            state['images']['site/b_w.png'][2]={'13':1}
        (src/'.sync-state.json').write_text(json.dumps(state),encoding='utf8')
        (src/'data.yaml').write_text('names:\n'+''.join(f'  {i}: "{v}"\n' for i,v in enumerate(NAMES)),encoding='utf8')
        bucket='site' if ctx=='collection' else 'floor'
        ct=class_counts[28]
        report='\n'.join([f'# {folder}','Generated 2026-10-28 10:00 UTC', '',f'- images: {len(names)} ({len(names)-pos} without labels)',
                          f'- labels (polygons): {ct}, mean 1, max 1 per image','- classes: 1 of 29 with labels',
                          '- dropped (label not in class list): 0','',
                          '| source bucket | images |','|---|---:|',f'| {bucket} | {len(names)} |','',
                          '| id | class | labels | % labels | images | % images |',
                          '|---:|---|---:|---:|---:|---:|',
                          f'| 28 | BAW Bag | {ct} | 100.0 | {pos} | 50.0 |',''])
        (src/'report.md').write_text(report,encoding='utf8')

def run(base):
    output=base.parent/'analysis'
    subprocess.run([sys.executable,str(Path(__file__).parent/'analyze_oct26.py'),
                    '--dataset',str(base),'--output',str(output)],check=True,timeout=60)
    return output,json.loads((output/'01_overview.json').read_text())

with tempfile.TemporaryDirectory(prefix='ccbre_oct26_test_') as t:
    base=Path(t)/'Raw';create_dataset(base)
    o,s=run(base)
    assert s['image_count']==3,s
    assert s['positive_images']==2,s
    assert s['explicitly_empty_images']==1,s
    assert s['detection_object_count']==2,s
    assert s['polygon_object_count']==2,s
    assert s['metadata_reconciliation_passed'],s['issues_by_type']
    assert s['duplicate_hash_groups']==1,s
    assert s['cross_source_duplicate_hash_groups']==1,s # same pixel content across the two source datasets
    assert s['validation_status']=='REQUIRES_REVIEW',s
    assert s['issues_by_type'].get('duplicate_image_conflicting_box_labels') == 1, s
    assert len(list(csv.DictReader((o/'18_class_reconciliation.csv').open(encoding='utf-8-sig'))))==58
with tempfile.TemporaryDirectory(prefix='ccbre_oct26_test_mismatch_') as t:
    base=Path(t)/'Raw';create_dataset(base,True)
    o,s=run(base)
    assert s['issues_by_type'].get('per_image_sync_class_count_mismatch',0)==1,s
    assert s['metadata_reconciliation_passed'] is False
print('PASS: clean metadata, geometry, context counts, exact duplicates, and deliberate metadata-mismatch detection.')
