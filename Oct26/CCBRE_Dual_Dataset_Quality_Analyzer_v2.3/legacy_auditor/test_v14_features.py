"""Regression tests of the additional v1.4 logic (does not access user dataset)."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image, TiffImagePlugin
import analyze_oct26 as a

# Source filename hierarchy is parsed but Q-code meaning is NOT assumed.
x=a.parse_source_filename('simpla-gruen-source__Q8184-64.2c.jpg','collection')
assert (x['q_code'],x['stop'],x['sub_index'],x['view_letter'])==('Q8184','64','2','c')
assert x['file_group_key']=='collection|Q8184|64'
y=a.parse_source_filename('repla-synthetic-images-2__DSC04954.JPG','floorspread')
assert y['dsc_number']==4954

# Capture runs separated at DSC gaps (provisional, not necessarily separate shoots).
rows=[dict(context='floorspread',dsc_number=n,file_group_key='',group_method='',run_id='')
      for n in (3316,3317,3320)]
assert a.summarize_group_keys(rows)[0]==2
assert rows[0]['file_group_key']==rows[1]['file_group_key']
assert rows[0]['file_group_key']!=rows[2]['file_group_key']

box=lambda c,b:dict(class_id=c,bbox=b)
p=[box(0,(.1,.1,.4,.4))]
assert a.category_duplicate_pair(p,p)['tier']=='geometry_equivalent'
assert a.category_duplicate_pair(p,[box(0,(.101,.101,.401,.401))])['equivalent']
assert a.category_duplicate_pair(p,[box(28,(.1,.1,.4,.4))])['tier']=='class_change'
assert a.category_duplicate_pair(p,[])['tier']=='positive_empty'
assert a.category_duplicate_pair(p,p+[box(0,(.5,.5,.6,.6))])['tier']=='count_change'
assert all(abs(v-e)<1e-10 for v,e in zip(a.orient_box((.2,.3,.5,.7),6),(.3,.2,.7,.5)))

with TemporaryDirectory() as t:
    f=Path(t)/'small.jpg'
    im=Image.new('RGB',(120,80),'purple')
    ex=im.getexif();ex[274]=6;ex[272]='SmokeTestCamera'
    im.save(f,exif=ex)
    errors=[]
    w,h,x=a.inspect_image(f,lambda *args:errors.append(args),'small')
    assert (w,h)==(120,80) and x['exif_orientation']==6
    assert len(x['display_pixel_sha256'])==64 and len(x['dhash64'])==16
    assert not errors
print('PASS: v1.4 filenames, grouping, duplicate tiers, EXIF and fingerprints')
