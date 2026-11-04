"""Focused regression for EXIF-display overlay labels in v1.7.1."""
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from analyze_oct26 import make_preview_sheets

def test():
    with TemporaryDirectory() as td:
        root=Path(td); out=root/'audit';(out/'figures').mkdir(parents=True)
        raw=root/'biowaste-collection-images'/'images';raw.mkdir(parents=True)
        im=Image.new('RGB',(180,120),(90,110,90))
        exif=Image.Exif();exif[274]=6
        im.save(raw/'Q0001-1c.jpg',exif=exif,quality=98)
        img='biowaste-collection-images/images/Q0001-1c.jpg'
        images=[dict(image=img,context='collection',annotation_state='positive',filename='Q0001-1c.jpg')]
        objects=[dict(image=img,xmin=.1,ymin=.2,xmax=.35,ymax=.45,class_id=28,class_name='BAW Bag')]
        created=make_preview_sheets(root,out,images,objects,limit=1)
        assert created==['04_collection_box_overlay_samples.jpg'],created
        with Image.open(out/'figures'/created[0]) as preview:
            # Displayed EXIF-6 image is 120 x 180, at x=235, y=30.
            # First box corner must be around x=247, y=66, not transformed coordinates.
            def red_near(x,y):
                return any((lambda z: z[0]>180 and z[1]<125 and z[2]<90)(preview.getpixel((xx,yy)))
                           for xx in range(x-3,x+4) for yy in range(y-3,y+4))
            assert red_near(247,66),'Display-frame box corner not drawn at expected location'
        print('PASS display-frame overlay regression (EXIF orientation=6)')

if __name__=='__main__':test()
