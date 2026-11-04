"""Refresh ONLY legacy 04_* contact sheets using EXIF display-frame labels.

Requires original October Raw image files and an existing full Analysis_v1.7
CSV directory. Rewrites only figures/04_*jpg in that output; copies previous
sheets to *_LEGACY_ENCODED_FRAME.jpg first. The HTML report uses relative
links and will display refreshed figures automatically. No annotation or
numerical CSV/JSON analysis is modified. Run on a COPY of the frozen analysis.
"""
from __future__ import annotations
import argparse,csv,pathlib,shutil
from analyze_oct26 import make_preview_sheets

def read(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,type=pathlib.Path)
    ap.add_argument('--analysis',required=True,type=pathlib.Path)
    args=ap.parse_args()
    analysis=args.analysis.resolve(); raw=args.dataset.resolve()
    if not (raw/'biowaste-collection-images'/'images').is_dir():raise SystemExit('Missing raw collection images')
    if not (analysis/'02_image_manifest.csv').is_file():raise SystemExit('Missing analysis image manifest')
    ims=read(analysis/'02_image_manifest.csv');obs=read(analysis/'11_object_manifest.csv')
    # CSV serialization stores bounding-box coordinates as strings.
    for obj in obs:
        for key in ('xmin','ymin','xmax','ymax'):
            obj[key]=float(obj[key])
    fig=analysis/'figures'
    for source in ['collection','floorspread']:
        original=fig/f'04_{source}_box_overlay_samples.jpg'
        legacy=fig/f'04_{source}_box_overlay_samples_LEGACY_ENCODED_FRAME.jpg'
        if original.exists() and not legacy.exists():shutil.copy2(original,legacy)
    saved=make_preview_sheets(raw,analysis,ims,obs)
    print('Refreshed display-frame overlay sheets:',saved)
    print('The relative links in Dataset_Audit_Report.html now refer to the corrected figures.')
    print('The original numerical CSV and JSON audit files are unchanged.')

if __name__=='__main__':main()
