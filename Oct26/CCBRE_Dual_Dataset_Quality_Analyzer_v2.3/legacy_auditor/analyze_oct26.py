#!/usr/bin/env python3
"""CCBRE Oct-26 read-only raw-data audit v1.7, adapted from raw analyzer v1.0.

Strictly scans exactly the two original S3 dataset products and explicitly pairs
images with labels-detect and polygon labels independently. Never edits Raw.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
import platform
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None
try:
    import PIL
    from PIL import Image, ImageDraw, ImageOps, ExifTags, features
except ImportError:
    Image = None
    ImageDraw = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

VERSION = '1.7.1'
DHASH_REDUCTION_FACTOR = 16
DHASH_DISTANCE_THRESHOLD = 6
DHASH_ALGORITHM = 'EXIF-display RGB-to-L then reduce(16) then LANCZOS 9x8; 64-bit horizontal dHash v1.5'
PIXEL_ALGORITHM = 'SHA256(width x height x RGB NUL + EXIF-display decoded RGB bytes); fallback=encoded pixels'
DHASH_REVIEW_MAX_DISTANCE = 12
DSC_TIME_GAP_SECONDS = 600

EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
SOURCES = {
    'collection': 'biowaste-collection-images',
    'floorspread': 'biowaste-floorspread-images',
}
EXPECTED = [
    'Plastic Bag','Coffee Capsule','Plastic Flower Pot','PET 0.5l','PET 1.5l',
    'Glass Bottle','Aluminum Can','Cat Litter','Diaper','Garbage Bag 17l',
    'Garbage Bag 35l','Garbage Bag 65l','Plastic Tarp','Paper','Cardboard',
    'Cigarettes','Textiles','Plastic Food Packaging','Tin Can',
    'Aluminum Foil','String','Soil Bags','Insulation','Rubber Bands',
    'Misc Metal','Misc Ceramic','Misc Plastic','Unknown','BAW Bag',
]


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def yaml_names(path):
    if not path.exists():
        return {}, 'missing data.yaml'
    if yaml is None:
        return {}, 'PyYAML not installed; class names unavailable'
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        names = data.get('names', {}) if isinstance(data, dict) else {}
        if isinstance(names, list):
            return {int(i): str(n) for i, n in enumerate(names)}, 'ok'
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}, 'ok'
        return {}, 'names missing or malformed'
    except Exception as exc:
        return {}, 'failed to parse: ' + str(exc)


def normalized(name):
    return re.sub(r'\s+', ' ', name.strip().lower())


def polygon_area(pts):
    return abs(sum(pts[i][0] * pts[(i+1) % len(pts)][1] -
                   pts[(i+1) % len(pts)][0] * pts[i][1]
                   for i in range(len(pts)))) / 2


def parse_annotation(path, mode, known_classes, issue, image_rel):
    """Return annotations, missing/empty/nonempty status; invalid rows logged."""
    if not path.is_file():
        return [], 'missing'
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except Exception as exc:
        issue(image_rel, 'unreadable_' + mode, str(path), str(exc))
        return [], 'unreadable'
    result = []
    any_text = False
    for ln, line in enumerate(lines, 1):
        tokens = line.split()
        if not tokens:
            continue
        any_text = True
        if not re.fullmatch(r'[+-]?\d+', tokens[0]):
            issue(image_rel, 'noninteger_class_id', mode, f'line={ln}: {tokens[0]}')
            continue
        cls = int(tokens[0])
        try:
            nums = [float(v) for v in tokens[1:]]
        except ValueError:
            issue(image_rel, 'nonnumeric_coordinates', mode, f'line={ln}')
            continue
        if not all(math.isfinite(x) for x in nums):
            issue(image_rel, 'nonfinite_coordinates', mode, f'line={ln}')
            continue
        if cls not in known_classes:
            issue(image_rel, 'unknown_class_id', mode, f'line={ln}, class={cls}')
        if mode == 'box':
            if len(nums) != 4:
                issue(image_rel, 'invalid_box_field_count', mode, f'line={ln}, n={len(nums)}')
                continue
            xc, yc, bw, bh = nums
            bounds = (xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2)
            geom_area = bw * bh
            valid = bw > 0 and bh > 0 and min(bounds) >= -1e-6 and max(bounds) <= 1 + 1e-6
        else:
            if len(nums) < 6 or len(nums) % 2:
                issue(image_rel, 'invalid_polygon_field_count', mode, f'line={ln}, n={len(nums)}')
                continue
            pts = list(zip(nums[::2], nums[1::2]))
            bounds = (min(x for x,y in pts), min(y for x,y in pts),
                      max(x for x,y in pts), max(y for x,y in pts))
            geom_area = polygon_area(pts)
            valid = geom_area > 1e-12 and min(bounds) >= -1e-6 and max(bounds) <= 1 + 1e-6
        if not valid:
            issue(image_rel, 'invalid_' + mode + '_geometry', mode, f'line={ln}, bounds={bounds}')
        result.append({'class_id': cls, 'bbox': bounds, 'area': geom_area,
                       'valid': valid, 'line': ln})
    return result, 'nonempty' if any_text else 'empty'


def box_iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    ba = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / (aa + ba - inter) if aa + ba > inter else 0.


def compare_geometry(boxes, polygons, image_rel, issue):
    """Match polygon enclosing bounds to detection boxes by class and maximal IoU."""
    if not boxes or not polygons:
        return None
    if Counter(x['class_id'] for x in boxes) != Counter(x['class_id'] for x in polygons):
        issue(image_rel, 'box_polygon_class_counts_differ', 'geometry',
              f'box={dict(Counter(x["class_id"] for x in boxes))}, polygon={dict(Counter(x["class_id"] for x in polygons))}')
    candidates = sorted(((box_iou(b['bbox'], p['bbox']), bi, pi)
                         for bi,b in enumerate(boxes) for pi,p in enumerate(polygons)
                         if b['class_id'] == p['class_id']), reverse=True)
    used_b, used_p, matches = set(), set(), []
    for iou, bi, pi in candidates:
        if bi not in used_b and pi not in used_p:
            matches.append(iou)
            used_b.add(bi); used_p.add(pi)
    if len(matches) != len(boxes) or len(matches) != len(polygons):
        issue(image_rel, 'unmatched_box_polygon_objects', 'geometry',
              f'matched={len(matches)}; boxes={len(boxes)}; polygons={len(polygons)}')
    if matches and min(matches) < 0.98:
        issue(image_rel, 'box_polygon_enclosures_disagree', 'geometry',
              f'min matched enclosing-box IoU={min(matches):.6f}')
    return min(matches) if matches else 0


def inspect_image(path, issue, rel):
    """Preserve header measurements even if EXIF/decode/pixel-hash fails.

    Full-resolution RGB bytes produce a pixel-content SHA-256 after EXIF
    transpose; dHash is an approximate image-similarity candidate signature.
    The two hashes are implementation-dependent and comparable only within
    the same decoder environment / pinned algorithm and Pillow versions.
    """
    extra = dict(exif_orientation='', exif_datetime_original='',
                 exif_camera_model='', exif_camera_make='',
                 display_width='', display_height='',
                 display_pixel_sha256='', dhash64='', pixel_coordinate_frame='EXIF_display')
    try:
        im = Image.open(path)
    except Exception as exc:
        issue(rel, 'image_header_unreadable', 'image', str(exc))
        return None, None, extra
    with im:
        try:
            width, height = im.size
        except Exception as exc:
            issue(rel, 'image_header_unreadable', 'image', str(exc))
            return None, None, extra
        orientation=1
        try:
            exif=im.getexif()
            orientation = int(exif.get(274, 1) or 1)
            if orientation not in range(1,9):
                issue(rel, 'invalid_exif_orientation', 'exif', str(orientation))
                orientation=1
            exif_ifd=exif.get_ifd(0x8769) if hasattr(exif,'get_ifd') else {}
            extra.update(exif_orientation=orientation,
                exif_datetime_original=str(exif_ifd.get(36867, '') or exif.get(36867, '')
                    or exif_ifd.get(36868, '') or exif.get(36868, '') or ''),
                exif_camera_model=str(exif.get(272,'') or ''),
                exif_camera_make=str(exif.get(271,'') or ''))
        except Exception as exc:
            issue(rel, 'image_exif_read_failed', 'exif', str(exc))
            extra['exif_orientation']=orientation
        # Dimensions are header-derived, even if the compressed pixels are corrupt.
        dw,dh=(height,width) if orientation in (5,6,7,8) else (width,height)
        extra.update(display_width=dw,display_height=dh)
        try:
            try:
                display=ImageOps.exif_transpose(im)
            except Exception as transpose_exc:
                issue(rel, 'exif_transpose_failed_encoded_fallback', 'exif', str(transpose_exc))
                display=im.copy()
                extra['pixel_coordinate_frame']='encoded_fallback'
            rgb=display.convert('RGB')
            ph=hashlib.sha256()
            ph.update(f'{rgb.width}x{rgb.height}xRGB\0'.encode('ascii'))
            ph.update(rgb.tobytes())
            extra['display_pixel_sha256']=ph.hexdigest()
            grey=rgb.convert('L')
            # Reduce before the costly thumbnail resample. The versioned method
            # intentionally changes dHash values relative to v1.4.
            factor=min(DHASH_REDUCTION_FACTOR,
                       max(1,min(grey.width//9,grey.height//8)))
            if factor>1: grey=grey.reduce(factor)
            grey=grey.resize((9,8), Image.Resampling.LANCZOS)
            pixels=list(grey.tobytes())
            bits=0
            for y in range(8):
                for x in range(8):
                    bits=(bits<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
            extra['dhash64']=f'{bits:016x}'
        except Exception as exc:
            issue(rel, 'image_decode_or_hash_failed', 'image', str(exc))
        return width,height,extra


def pillow_runtime():
    try: codec=features.version_codec('jpg')
    except Exception: codec=None
    return dict(python_version=platform.python_version(),
                pillow_version=PIL.__version__, jpeg_decoder_version=codec,
                dhash_reduction_factor=DHASH_REDUCTION_FACTOR,
                dhash_max_hamming=DHASH_DISTANCE_THRESHOLD,
                dhash_algorithm=DHASH_ALGORITHM, pixel_algorithm=PIXEL_ALGORITHM,
                dhash_review_max_distance=DHASH_REVIEW_MAX_DISTANCE,
                dsc_time_gap_seconds=DSC_TIME_GAP_SECONDS)


def decoded_pixel_duplicate_groups(images):
    """Pixel-identical decoded EXIF-display arrays with DIFFERENT file hashes.

    Reencoded JPEGs usually have altered pixels: these require dHash candidates.
    """
    by_pixel=defaultdict(list)
    for row in images:
        if row['display_pixel_sha256'] and row.get('pixel_coordinate_frame')=='EXIF_display':
            by_pixel[row['display_pixel_sha256']].append(row)
    groups=[]
    for pixel_hash, members in sorted(by_pixel.items()):
        sha={r['sha256'] for r in members}
        if len(members)<2 or len(sha)<2:continue
        groups.append(dict(display_pixel_sha256=pixel_hash,
            images=len(members),distinct_file_sha256=len(sha),
            contexts='|'.join(sorted({r['context'] for r in members})),
            group_keys=' | '.join(sorted({r['file_group_key'] or 'UNPARSED' for r in members})),
            crosses_contexts=len({r['context'] for r in members})>1,
            crosses_filename_groups=len({r['file_group_key'] for r in members})>1,
            paths=' | '.join(sorted(r['image'] for r in members))))
    return groups


class UnionFind:
    def __init__(self, count):
        self.parent=list(range(count))
        self.size=[1]*count
    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]]
            i=self.parent[i]
        return i
    def union(self,i,j):
        x,y=self.find(i),self.find(j)
        if x==y:return False
        if self.size[x]<self.size[y]:x,y=y,x
        self.parent[y]=x;self.size[x]+=self.size[y]
        return True
    def clone(self):
        cp=UnionFind(len(self.parent));cp.parent=self.parent.copy();cp.size=self.size.copy()
        return cp


def make_strict_groups(images):
    """Conservative union: filename acquisition groups + exact files + equal decoded pixels."""
    uf=UnionFind(len(images))
    ix_by_image={r['image']:i for i,r in enumerate(images)}
    for key_field in ('file_group_key','sha256','display_pixel_sha256'):
        first={}
        for i,r in enumerate(images):
            k=r.get(key_field)
            if not k:continue
            if key_field=='display_pixel_sha256' and r.get('pixel_coordinate_frame')!='EXIF_display':
                # Fallback pixels are from the encoded frame, not the same display convention.
                continue
            if k in first:uf.union(first[k],i)
            else:first[k]=i
    return uf,ix_by_image


def dhash_representatives(images):
    """One representative for every decoded-display pixel array (or file SHA if decode fails)."""
    groups=defaultdict(list)
    for i,r in enumerate(images):
        if not r.get('dhash64') or r.get('pixel_coordinate_frame')!='EXIF_display':continue
        if r.get('display_pixel_sha256') and r.get('pixel_coordinate_frame')=='EXIF_display':
            key=('pixel',r['display_pixel_sha256'])
        elif r.get('sha256'):
            key=('file',r['sha256'])
        else:key=('image',r['image'])
        groups[key].append((i,r))
    reps=[]
    for members in groups.values():
        members.sort(key=lambda v:v[1]['image'])
        a=members[0][1]
        reps.append(dict(index=members[0][0],image=a['image'],hash=int(a['dhash64'],16),
             members=[r for _,r in members],member_indices=[i for i,_ in members]))
    reps.sort(key=lambda x:x['image'])
    return reps


def global_dhash_scan(images,dest,extended_uf,indices,threshold=DHASH_DISTANCE_THRESHOLD,
                      review_max=DHASH_REVIEW_MAX_DISTANCE,max_rows=2000000):
    """Globally scan UNIQUE decoded pixel arrays, track 0..12 distance histogram.

    dHash edges are *candidates*, not verified duplicates. Extended union is advisory.
    Rows <= review_max are retained so an alternative future cutoff can be examined.
    """
    reps=dhash_representatives(images)
    columns=['image_a','image_b','context_a','context_b','group_key_a','group_key_b',
             'cross_context','cross_filename_group','dhash_hamming','selected_for_extended',
             'pixel_group_size_a','pixel_group_size_b','pixel_group_members_a',
             'pixel_group_members_b','method']
    hist=Counter(); total_selected=0;total_review=0; written=0;cross_group=0;cross_context=0
    union_edges=0
    with dest.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for idx,a in enumerate(reps):
            for b in reps[idx+1:]:
                d=(a['hash']^b['hash']).bit_count()
                hist[d]+=1
                if d>review_max:continue
                total_review+=1
                cross_ctx=any(x['context']!=y['context'] for x in a['members'] for y in b['members'])
                cross_group_any=any(x['file_group_key']!=y['file_group_key'] for x in a['members'] for y in b['members'])
                selected=d<=threshold
                if selected:
                    total_selected+=1;cross_group+=cross_group_any;cross_context+=cross_ctx
                    if extended_uf.union(a['index'],b['index']):union_edges+=1
                if written<max_rows:
                    aa,bb=a['members'][0],b['members'][0]
                    writer.writerow(dict(image_a=aa['image'],image_b=bb['image'],
                         context_a=aa['context'],context_b=bb['context'],
                         group_key_a=aa['file_group_key'],group_key_b=bb['file_group_key'],
                         cross_context=cross_ctx,cross_filename_group=cross_group_any,
                         dhash_hamming=d,selected_for_extended=selected,
                         pixel_group_size_a=len(a['members']),pixel_group_size_b=len(b['members']),
                         pixel_group_members_a=' | '.join(x['image'] for x in a['members']),
                         pixel_group_members_b=' | '.join(x['image'] for x in b['members']),
                         method='global_unique_pixel_dhash64_candidate_only'))
                    written+=1
    write_csv(dest.with_name('29_dhash_distance_distribution.csv'),
        ['distance','pairs','selected_for_extended','retained_for_review'],
        [dict(distance=d,pairs=hist[d],selected_for_extended=d<=threshold,
              retained_for_review=d<=review_max) for d in range(65)])
    return dict(total=total_selected,candidate_pairs_up_to_review_max=total_review,
         rows_written=written,review_rows_truncated=total_review>written,
         cross_filename_group=cross_group,cross_context=cross_context,
         images_with_dhash=sum(bool(r['dhash64']) and r.get('pixel_coordinate_frame')=='EXIF_display' for r in images),
         excluded_encoded_fallback_images=sum(bool(r['dhash64']) and r.get('pixel_coordinate_frame')!='EXIF_display' for r in images),
         unique_decoded_pixel_representatives=len(reps),
         removed_redundant_image_rows=sum(bool(r['dhash64']) and r.get('pixel_coordinate_frame')=='EXIF_display' for r in images)-len(reps),
         pairs_considered=len(reps)*(len(reps)-1)//2,threshold_for_extended=threshold,
         review_distance_limit=review_max,extended_union_edges=union_edges,
         hamming_distribution_0_to_12={str(d):hist[d] for d in range(13)},
         hamming_distance_over_12=sum(hist[d] for d in range(13,65)))


def export_split_groups(images,strict,extended,out):
    """Emit one strict/extended group ID per October image and size distributions."""
    ids={}; sizes={}; summaries=[]; distributions=[]
    for label,uf,tag in [('strict',strict,'SGS'),('extended',extended,'SGE')]:
        members=defaultdict(list)
        for i,r in enumerate(images):members[uf.find(i)].append(r)
        by_root={}
        for root,group in members.items():
            token=hashlib.sha256(min(x['image'] for x in group).encode('utf-8')).hexdigest()[:16]
            group_id=f'{tag}-{token}'
            by_root[root]=group_id
            summaries.append(dict(level=label,split_group_id=group_id,images=len(group),
                contexts=' | '.join(sorted({x['context'] for x in group})),
                source_filename_groups=len({x['file_group_key'] for x in group if x['file_group_key']}),
                exact_image_hashes=len({x['sha256'] for x in group if x['sha256']}),
                first_image=min(x['image'] for x in group),
                members=' | '.join(sorted(x['image'] for x in group))))
        ids[label]={i:by_root[uf.find(i)] for i in range(len(images))}
        sizes[label]={i:len(members[uf.find(i)]) for i in range(len(images))}
        dist=Counter(len(g) for g in members.values())
        distributions.extend(dict(level=label,images_per_group=n,groups=count,
                                  images_covered=n*count) for n,count in sorted(dist.items()))
    write_csv(out/'31_split_group_manifest.csv',
       ['image','context','file_group_key','time_gap_group_key',
        'split_group_strict','split_group_extended','strict_group_size','extended_group_size'],
       [dict(image=r['image'],context=r['context'],file_group_key=r['file_group_key'],
            time_gap_group_key=r.get('time_gap_group_key',''),
            split_group_strict=ids['strict'][i],split_group_extended=ids['extended'][i],
            strict_group_size=sizes['strict'][i],extended_group_size=sizes['extended'][i])
         for i,r in enumerate(images)])
    write_csv(out/'35_split_group_size_distribution.csv',
        ['level','images_per_group','groups','images_covered'],distributions)
    write_csv(out/'36_split_group_summary.csv',
        ['level','split_group_id','images','contexts','source_filename_groups',
         'exact_image_hashes','first_image','members'],summaries)
    for i,r in enumerate(images):
        r.update(split_group_strict=ids['strict'][i],split_group_extended=ids['extended'][i],
                 strict_group_size=sizes['strict'][i],extended_group_size=sizes['extended'][i])
    return dict(strict_group_count=len({v for v in ids['strict'].values()}),
                extended_group_count=len({v for v in ids['extended'].values()}),
                strict_max_group_size=max(sizes['strict'].values(),default=0),
                extended_max_group_size=max(sizes['extended'].values(),default=0),
                extended_is_candidate_only=True,
                note='Extended dHash links can chain unrelated green-waste images and MUST be visually adjudicated before split construction.')


def parse_exif_datetime(value):
    if not value:return None
    for fmt in ('%Y:%m:%d %H:%M:%S','%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S'):
        try:return datetime.strptime(str(value).strip()[:19],fmt)
        except ValueError:pass
    return None


def parse_source_filename(filename, context):
    original = Path(filename.split('__', 1)[-1]).stem
    # Strip only Windows copy suffix at stem end; retain original for audit.
    cleaned = re.sub(r'\s*\(\d+\)$', '', original)
    row = dict(q_code='', stop='', sub_index='', view_letter='', dsc_number='', run_id='',
               file_group_key='', group_method='', filename_parse_ok=False)
    if context == 'collection':
        m = re.fullmatch(r'(Q\d+)-(\d+)(?:\.(\d+))?([cwbt])', cleaned, flags=re.I)
        if m:
            qcode, stop, sub, view = m.groups()
            row.update(q_code=qcode.upper(), stop=str(int(stop)), sub_index=sub or '',
                  view_letter=view.lower(), file_group_key=f'collection|{qcode.upper()}|{int(stop)}',
                  group_method='Q-and-stop_filename_candidate',filename_parse_ok=True)
    else:
        m = re.fullmatch(r'DSC0*(\d+)', cleaned, flags=re.I)
        if m:
            row.update(dsc_number=int(m.group(1)),filename_parse_ok=True)
    return row


def orient_box(box, orientation):
    """Transform normalized *encoded-frame* box into EXIF-transposed display frame.

    NOTE: annotation tools may store boxes already in display frame; the exported
    geometry convention must be visually confirmed before interpreting overlays.
    """
    x1,y1,x2,y2 = box
    pts = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
    def transform(x,y):
        return {1:(x,y),2:(1-x,y),3:(1-x,1-y),4:(x,1-y),
                5:(y,x),6:(1-y,x),7:(1-y,1-x),8:(y,1-x)}.get(orientation,(x,y))
    p=[transform(x,y) for x,y in pts]
    return min(x for x,y in p),min(y for x,y in p),max(x for x,y in p),max(y for x,y in p)


def category_duplicate_pair(a, b, min_iou=0.95):
    """One-to-one class-aware IoU matching, conservative when classes/count differ."""
    ca,cb=Counter(x['class_id'] for x in a),Counter(x['class_id'] for x in b)
    same_count=len(a)==len(b)
    same_classes=ca==cb
    positive_empty=(bool(a)!=bool(b))
    if not same_count:
        tier='positive_empty' if positive_empty else 'count_change'
        return dict(tier=tier,min_matched_iou='',equivalent=False,
                    positive_empty=positive_empty,count_change=True,
                    class_change=not same_classes)
    if not same_classes:
        return dict(tier='class_change',min_matched_iou='',equivalent=False,
                    positive_empty=False,count_change=False,class_change=True)
    if not a:
        return dict(tier='identical',min_matched_iou=1.0,equivalent=True,
                    positive_empty=False,count_change=False,class_change=False)
    pairs=sorted(((box_iou(x['bbox'],y['bbox']),i,j)
                  for i,x in enumerate(a) for j,y in enumerate(b)
                  if x['class_id']==y['class_id']),reverse=True)
    seen_a,seen_b,matched=set(),set(),[]
    for score,i,j in pairs:
        if i not in seen_a and j not in seen_b:
            seen_a.add(i);seen_b.add(j);matched.append(score)
    minimum=min(matched) if len(matched)==len(a) else 0.0
    # Exact rounded signature equality is tested by caller, not here.
    return dict(tier='geometry_equivalent' if minimum>=min_iou else 'geometry_substantive',
                min_matched_iou=minimum,equivalent=(minimum>=min_iou),
                positive_empty=False,count_change=False,class_change=False)


def summarize_group_keys(images):
    """Build camera-aware numeric runs plus timestamp-aware provisional sessions.

    Missing camera identity is a separate UNKNOWN bucket; a model is not a body
    serial number. Numeric runs are conservatively kept; time sessions are an
    additional diagnostic, never silently a replacement for file grouping.
    """
    camera_rows=defaultdict(list)
    for r in images:
        if r['context']!='floorspread':continue
        mk=' '.join(str(r.get('exif_camera_make','')).lower().split())
        model=' '.join(str(r.get('exif_camera_model','')).lower().split())
        key=f'{mk}|{model}' if mk or model else 'UNKNOWN_CAMERA'
        slug=re.sub(r'[^a-z0-9]+','_',key.lower()).strip('_')[:30] or 'unknown'
        camid=slug+'_'+hashlib.sha256(key.encode()).hexdigest()[:7]
        r['camera_identity_key']=key
        r['camera_id']=camid
        r['time_gap_group_key']=''
        if r['dsc_number']!='':camera_rows[camid].append(r)
        else:r['file_group_key']=''
    numeric_count=time_count=0
    for camera,rows in sorted(camera_rows.items()):
        numbers=sorted({int(r['dsc_number']) for r in rows})
        number_runs={};old=None;local_num=0
        for n in numbers:
            if old is None or n!=old+1:local_num+=1;numeric_count+=1
            number_runs[n]=f'floorspread|camera_{camera}|DSC_{local_num:05d}'
            old=n
        for row in rows:
            rkey=number_runs[int(row['dsc_number'])]
            row.update(file_group_key=rkey,run_id=rkey.rsplit('_',1)[-1],
                       group_method='camera_plus_contiguous_DSC_numbers_provisional')
        ordered=sorted(rows,key=lambda r:(int(r['dsc_number']),
                          r.get('exif_datetime_original',''),r.get('image','')))
        prev_num=prev_time=None;time_id=0
        for row in ordered:
            n=int(row['dsc_number']);tm=parse_exif_datetime(row.get('exif_datetime_original'))
            # A missing EXIF timestamp never establishes a time-based edge.
            if tm is None:
                prev_num=prev_time=None
                continue
            gap=(tm-prev_time).total_seconds() if prev_time is not None else None
            if prev_time is None or n>prev_num+1 or n<prev_num or gap<0 or gap>DSC_TIME_GAP_SECONDS:
                time_id+=1;time_count+=1
            row['time_gap_group_key']=f'floorspread|camera_{camera}|TIME_{time_id:05d}'
            prev_num=n;prev_time=tm
    for r in images:
        if r['context']=='collection':
            r['camera_identity_key']=' '.join((str(r.get('exif_camera_make','')),str(r.get('exif_camera_model','')))).strip()
            r['camera_id']=''
            r['time_gap_group_key']=''
    return numeric_count,time_count


def report_table(rows, keys):
    if not rows:
        return '<p>No records.</p>'
    th = ''.join(f'<th>{html.escape(k)}</th>' for k in keys)
    trs = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(row.get(k,"")))}</td>' for k in keys) + '</tr>' for row in rows)
    return f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def make_plots(out, class_rows, context_rows, sizes):
    if plt is None:
        return []
    plots = []
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)
    named = sorted((r for r in class_rows if r['context'] == 'all'), key=lambda r:r['instances'])
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh([r['class_name'] for r in named], [r['images_containing_class'] for r in named])
    ax.set_xlabel('Distinct images containing class')
    ax.set_title('October raw dataset — class image support')
    fig.tight_layout()
    name='01_class_image_support.png'; fig.savefig(fig_dir/name, dpi=165); plt.close(fig); plots.append(name)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r['context'] for r in context_rows], [r['image_count'] for r in context_rows])
    ax.set_ylabel('Image files'); ax.set_title('Images by source context')
    fig.tight_layout(); name='02_source_images.png'; fig.savefig(fig_dir/name,dpi=165); plt.close(fig); plots.append(name)
    if sizes:
        fig, ax = plt.subplots(figsize=(8,4))
        logsize=[math.log10(100*x) for x in sizes if x>0]
        ax.hist(logsize,bins=55,weights=[1/len(logsize)]*len(logsize))
        ax.set_xlabel('log10(bounding-box area as % of image)')
        ax.set_ylabel('Fraction per equal log10-area bin')
        ax.set_title('Normalized object sizes — equal-width log bins')
        fig.tight_layout(); name='03_box_area.png'; fig.savefig(fig_dir/name,dpi=165); plt.close(fig); plots.append(name)
    return plots


def make_preview_sheets(root, out, images, objects, limit=8):
    """Deterministic contact sheets of DETECTION overlays for human visual review."""
    if Image is None or ImageDraw is None:
        return []
    obj_index = defaultdict(list)
    for obj in objects:
        obj_index[obj['image']].append(obj)
    saved=[]
    width, height = 590, 430
    for source in ('collection','floorspread'):
        positives = sorted((r for r in images if r['context']==source and
                            r['annotation_state']=='positive'),key=lambda r:r['image'])
        if not positives:
            continue
        # Evenly spaced selection avoids choosing only sequential frames.
        count=min(limit,len(positives))
        chosen=[positives[round((len(positives)-1)*i/(count-1))] if count>1 else positives[0]
                for i in range(count)]
        sheet=Image.new('RGB',(width*2,height*((count+1)//2)),(245,248,245))
        draw_sheet=ImageDraw.Draw(sheet)
        for idx, item in enumerate(chosen):
            image_path = root/Path(item['image'])
            try:
                with Image.open(image_path) as im:
                    natural_w,natural_h=im.size
                    ori=int(im.getexif().get(274,1) or 1)
                    display=ImageOps.exif_transpose(im).convert('RGB')
                    display.thumbnail((width-18,height-64))
                    img=display.copy()
            except Exception:
                continue
            xd=(idx % 2)*width; yd=(idx // 2)*height
            x0=xd+(width-img.width)//2
            y0=yd+30
            sheet.paste(img,(x0,y0))
            d=ImageDraw.Draw(sheet)
            d.text((xd+8,yd+7),item['filename'][:70],fill=(10,55,22))
            for o in obj_index[item['image']]:
                # Detection labels use the EXIF-displayed frame.  Never rotate boxes here.
                bx=(o['xmin'],o['ymin'],o['xmax'],o['ymax'])
                xx1=x0+int(bx[0]*img.width); yy1=y0+int(bx[1]*img.height)
                xx2=x0+int(bx[2]*img.width); yy2=y0+int(bx[3]*img.height)
                if xx2 <= xx1 or yy2 <= yy1:
                    continue
                d.rectangle([xx1,yy1,xx2,yy2],outline=(255,70,10),width=2)
                d.text((xx1,min(yy1+2,yd+height-25)),f"{o['class_id']} {o['class_name']}",
                       fill=(255,35,10),stroke_width=1,stroke_fill='white')
            d.text((xd+9,yd+height-24),f'{len(obj_index[item["image"]])} display-frame labels | EXIF orientation {ori}',fill=(10,55,22))
        name=f'04_{source}_box_overlay_samples.jpg'
        (out/'figures').mkdir(exist_ok=True)
        sheet.save(out/'figures'/name,quality=86)
        saved.append(name)
    return saved


def make_orientation_comparison_sheets(root,out,images,objects):
    """For EVERY positive nontrivial-EXIF image, draw both YOLO frame hypotheses.

    Left: labels encoded-frame (transformed with EXIF orientation).
    Right: labels already display-frame (no box transform).
    This is REVIEW material, not automated truth selection.
    """
    if Image is None or ImageDraw is None:return []
    index=defaultdict(list)
    for obj in objects:index[obj['image']].append(obj)
    folder=out/'figures'/'orientation_comparison'
    folder.mkdir(parents=True,exist_ok=True)
    records=[]
    candidates=[r for r in images if r['annotation_state']=='positive'
                and r['exif_orientation'] not in ('',1)]
    pw,ph=780,580
    for row in candidates:
        rel=row['image']; ori=row['exif_orientation']
        try:
            with Image.open(root/Path(rel)) as im:
                try:disp=ImageOps.exif_transpose(im).convert('RGB')
                except Exception:disp=im.convert('RGB')
                disp.thumbnail((pw-25,ph-110))
            sheet=Image.new('RGB',(2*pw,ph),'white')
            d=ImageDraw.Draw(sheet)
            for k,label_frame in enumerate(('encoded','display')):
                x0=k*pw+(pw-disp.width)//2; y0=55
                sheet.paste(disp,(x0,y0))
                d.text((k*pw+12,12),f'{label_frame.upper()}-FRAME LABEL HYPOTHESIS',fill='#152f21')
                d.text((k*pw+12,30),f'EXIF orientation {ori} | {rel[-76:]}',fill='#152f21')
                for o in index[rel]:
                    bbox=(o['xmin'],o['ymin'],o['xmax'],o['ymax'])
                    if label_frame=='encoded':bbox=orient_box(bbox,ori)
                    x1=x0+round(bbox[0]*disp.width);y1=y0+round(bbox[1]*disp.height)
                    x2=x0+round(bbox[2]*disp.width);y2=y0+round(bbox[3]*disp.height)
                    if x2<=x1 or y2<=y1:continue
                    d.rectangle((x1,y1,x2,y2),outline='#ff471f',width=3)
                    d.text((x1,max(y1-12,y0)),f"{o['class_id']} {o['class_name']}",fill='#ff471f',stroke_width=1,stroke_fill='white')
            d.text((15,ph-40),'Compare by eye. No automatic inference of the annotation coordinate convention.',fill='#152f21')
            token=hashlib.sha256(rel.encode('utf8')).hexdigest()[:10]
            short=re.sub(r'[^A-Za-z0-9_.-]+','_',Path(rel).stem)[:55].strip('_')
            name=f'{short}_{token}.jpg'
            sheet.save(folder/name,quality=88)
            records.append(dict(image=rel,exif_orientation=ori,
                     coordinate_convention='UNRESOLVED_REVIEW_BOTH',
                     sheet=f'figures/orientation_comparison/{name}',status='generated'))
        except Exception as exc:
            records.append(dict(image=rel,exif_orientation=ori,
                coordinate_convention='UNRESOLVED_REVIEW_BOTH',sheet='',status=f'failed: {exc}'))
    write_csv(out/'30_orientation_review_index.csv',
        ['image','exif_orientation','coordinate_convention','sheet','status'],records)
    return records


def parse_export_report(path):
    """Parse partner-generated Markdown headline counts (informational)."""
    if not path.is_file():
        return {}
    content=path.read_text(encoding='utf-8-sig')
    result={'generated_at':None,'images':None,'without_labels':None,'objects':None,
            'classes_with_labels':None,'dropped_label_masks':0}
    m=re.search(r'Generated\s+([^\n]+)',content)
    if m:result['generated_at']=m.group(1).strip()
    m=re.search(r'^- images: (\d+) \((\d+) without labels\)',content,re.M)
    if m:result['images'],result['without_labels']=map(int,m.groups())
    m=re.search(r'^- labels \(polygons\): (\d+)',content,re.M)
    if m:result['objects']=int(m.group(1))
    m=re.search(r'^- classes: (\d+) of (\d+) with labels',content,re.M)
    if m:result['classes_with_labels']=int(m.group(1))
    m=re.search(r'^- dropped \(label not in class list\): (\d+)',content,re.M)
    if m:result['dropped_label_masks']=int(m.group(1))
    return result


def parse_sync_state(path):
    """Parse all relevant synchronization fields, including image identity and dropped masks.

    NB: image record [1] is the label-update timestamp, not necessarily capture time.
    """
    if not path.is_file(): return {'present':False}
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    image_record=data.get('images',{})
    if not isinstance(image_record,dict):
        raise ValueError(f'Invalid .sync-state.json: images must be a mapping: {path}')
    counts=Counter(); dropped=Counter(); annotated=0
    for key,record in image_record.items():
        if not (isinstance(record,list) and len(record)>=3 and isinstance(record[2],dict)):
            raise ValueError(f'Invalid .sync-state image record: {key}')
        if record[2]: annotated += 1
        for k,v in record[2].items(): counts[int(k)] += int(v)
        if len(record)>3 and isinstance(record[3],dict):
            for k,v in record[3].items(): dropped[k] += int(v)
    return {'present':True,'images':len(image_record),'objects':sum(counts.values()),
            'class_instances':dict(counts),'classes':data.get('classes',[]),
            'renames':data.get('rename',{}),'ignore':data.get('ignore',[]),
            'dropped_labels':dict(dropped), 'annotated_images':annotated,
            'image_records':image_record,
            'source_objects_records':len(data.get('objects',{})),
            'source_annotations_records':len(data.get('annotations',{})),
            'referenced_annotation_records':len(set(rec[0] for rec in image_record.values())),
            'unreferenced_annotation_history_records':len(set(data.get('annotations',{})) -
                 set(rec[0] for rec in image_record.values())),
            'cancelled_annotation_history_records':sum(bool(rec.get('cancelled'))
                 for rec in data.get('annotations',{}).values()),
            'missing_referenced_annotations':sum(rec[0] not in data.get('annotations',{})
                 for rec in image_record.values()),
            'source_image_reference_mismatches':sum(
                 rec[0] in data.get('annotations',{}) and
                 data['annotations'][rec[0]].get('image') != key
                 for key,rec in image_record.items()),
            'missing_referenced_source_objects':sum(
                 str(rec[0]).split('#')[0] not in data.get('objects',{})
                 for rec in image_record.values())}



def parse_report_tables(path):
    """Read partner Markdown tables by header, keeping zero-count classes.

    All counts here describe the *exporter's* declared snapshot, not our local audit.
    """
    if not path.is_file():
        return {'source_buckets':{}, 'classes':{}, 'dropped_labels':{}}
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    tables={'source_buckets':{},'classes':{},'dropped_labels':{}}
    section=None
    for line in lines:
        if line.startswith('| source bucket | images |'):
            section='source_buckets';continue
        if line.startswith('| id | class | labels |'):
            section='classes';continue
        if line.startswith('| label | masks |'):
            section='dropped_labels';continue
        if not line.strip():
            section=None; continue
        if section and line.startswith('|'):
            cells=[s.strip() for s in line.strip().strip('|').split('|')]
            if all(c.startswith('-') or c=='--:' for c in cells):
                continue
            try:
                if section=='source_buckets' and len(cells)==2:
                    tables[section][cells[0]]=int(cells[1].replace(',',''))
                elif section=='classes' and len(cells)>=5:
                    cid=int(cells[0]); tables[section][cid]={
                        'name':cells[1], 'instances':int(cells[2].replace(',','')),
                        'images':int(cells[4].replace(',',''))}
                elif section=='dropped_labels' and len(cells)==2:
                    tables[section][cells[0]]=int(cells[1].replace(',',''))
            except ValueError:
                pass
    return tables


def normalized_sync_key(file_relative):
    """S3 exporter encodes source-bucket/original-name as source-bucket__name."""
    fname=file_relative.name
    if '__' not in fname:
        return None
    prefix, original=fname.split('__',1)
    return prefix+'/'+original


def source_id_from_sync_key(key):
    return key.split('/',1)[0] if '/' in key else 'UNKNOWN'


def infer_collection_context(filename):
    """Legacy site filename convention; name-derived, NOT a verified site label."""
    stem=Path(filename).stem
    stem=stem.rsplit('__',1)[-1]
    stem=re.sub(r'\s*\(\d+\)$','',stem)
    letter=stem[-1:].lower()
    if letter in ('c','b','t'):return 'container_inferred'
    if letter=='w':return 'wagon_inferred'
    return 'unknown'


def bbox_signature(rows):
    return tuple(sorted((r['class_id'], *[round(float(v),6) for v in r['bbox']]) for r in rows))


def checked_requirements():
    missing=[]
    if yaml is None: missing.append('PyYAML')
    if Image is None: missing.append('Pillow')
    if plt is None: missing.append('matplotlib')
    if missing:
        raise SystemExit('Missing REQUIRED package(s): '+', '.join(missing)
              +'. Install using: py -3 -m pip install -r requirements.txt. '
              +'No partial report has been generated.')


def fingerprint_only_mode(image_roots,out,recursive=False):
    """Create a reference signature manifest without interpreting training derivatives."""
    out.mkdir(parents=True,exist_ok=True)
    records=[];issues=[];sources=[]
    for rid,source in enumerate(image_roots,1):
        folder=source.resolve()
        if not folder.is_dir():raise SystemExit(f'Image source directory does not exist: {folder}')
        originals=sorted(folder.rglob('*') if recursive else folder.iterdir())
        targets=[x for x in originals if x.is_file() and x.suffix.lower() in EXT]
        if not targets:raise SystemExit(f'No images found in explicit source: {folder}')
        sources.append(dict(root=str(folder),images=len(targets),recursive=recursive))
        for n,im_path in enumerate(targets,1):
            rel=f'reference{rid:02d}/'+im_path.relative_to(folder).as_posix()
            def issue(path,kind,component,detail):
                issues.append(dict(image=path,issue=kind,component=component,detail=detail))
            w,h,extra=inspect_image(im_path,issue,rel)
            try:sha=hash_file(im_path)
            except Exception as exc:sha='';issue(rel,'sha256_failed','image',str(exc))
            records.append(dict(image=rel,source_root=str(folder),source_path=str(im_path),
                width=w or '',height=h or '',sha256=sha,**extra))
            if n%500==0:print(f'Reference root {rid}: {n}/{len(targets)}',flush=True)
    write_csv(out/'31_fingerprint_manifest.csv',list(records[0].keys()),records)
    write_csv(out/'31_fingerprint_quality_issues.csv',['image','issue','component','detail'],issues)
    state=dict(analyzer_version=VERSION,fingerprint_only=True,generated_at=datetime.now(timezone.utc).isoformat(),
        runtime=pillow_runtime(),image_roots=sources,images=len(records),
        file_sha_count=sum(bool(r['sha256']) for r in records),
        pixel_sha_count=sum(bool(r['display_pixel_sha256']) for r in records),
        dhash_count=sum(bool(r['dhash64']) for r in records),
        warnings=['The supplied sources are assumed to be ORIGINAL July image folders. The code does not establish source-vs-derived status.'],
        quality_issue_count=len(issues))
    (out/'31_fingerprint_runtime.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf8')
    print('Fingerprint-only manifests:',len(records),'images in',out)


def reference_comparison(images,reference_path,out,threshold):
    """Evidence tier: file identity > within-decoder pixel identity > dHash candidate.

    Only compare pixel/dHash when runtime metadata and decoder are compatible.
    All matches are links, NOT proof of a new independent acquisition population.
    """
    ref=reference_path.resolve()
    if not ref.is_file():raise SystemExit(f'Reference manifest not found: {ref}')
    with ref.open(encoding='utf-8-sig',newline='') as f:refs=list(csv.DictReader(f))
    required={'image','sha256','display_pixel_sha256','dhash64'}
    if not refs or not required.issubset(set(refs[0])):
        raise SystemExit('Reference manifest must contain image, sha256, display_pixel_sha256 and dhash64')
    fingerprint_runtime=ref.parent/'31_fingerprint_runtime.json'
    comparable=False;env_note='Reference runtime unavailable; pixel and dHash comparisons skipped.'
    if fingerprint_runtime.is_file():
        refenv=json.loads(fingerprint_runtime.read_text(encoding='utf8')).get('runtime',{})
        env=pillow_runtime()
        keys=('pillow_version','jpeg_decoder_version','dhash_algorithm','pixel_algorithm',
              'dhash_reduction_factor')
        comparable=all(refenv.get(k)==env.get(k) for k in keys)
        env_note='Compatible decoding and hashing methods.' if comparable else 'Decoder or hash method mismatch: ONLY FILE SHA256 compared.'
    by_sha=defaultdict(list);by_px=defaultdict(list)
    for r in refs:
        if r['sha256']:by_sha[r['sha256']].append(r)
        if comparable and r['display_pixel_sha256'] and r.get('pixel_coordinate_frame','EXIF_display')=='EXIF_display':
            by_px[r['display_pixel_sha256']].append(r)
    strong=[];matched_oct=set();file_oct=set();pixel_oct=set()
    for row in images:
        matched={}
        for r in by_sha.get(row['sha256'],[]) if row['sha256'] else []:
            matched[r['image']]=('exact_file_sha256',r)
        if comparable and row['pixel_coordinate_frame']=='EXIF_display':
            for r in by_px.get(row['display_pixel_sha256'],[]) if row['display_pixel_sha256'] else []:
                matched.setdefault(r['image'],('identical_decoded_pixels',r))
        for _,(method,r) in sorted(matched.items()):
            strong.append(dict(oct_image=row['image'],reference_image=r['image'],
                method=method,oct_sha256=row['sha256'],reference_sha256=r['sha256'],
                oct_source_group=row['file_group_key'],reference_source_root=r.get('source_root','')))
            matched_oct.add(row['image'])
            if method=='exact_file_sha256':file_oct.add(row['image'])
            else:pixel_oct.add(row['image'])
    write_csv(out/'32_reference_strong_matches.csv',
        ['oct_image','reference_image','method','oct_sha256','reference_sha256',
         'oct_source_group','reference_source_root'],strong)
    distribution=Counter();write_count=0;candidate_oct=set();candidate_pairs=0
    ref_reps=dhash_representatives([{**r,'context':'reference','file_group_key':'',
         'pixel_coordinate_frame':r.get('pixel_coordinate_frame','EXIF_display')} for r in refs]) if comparable else []
    oct_reps=dhash_representatives(images) if comparable else []
    with (out/'33_reference_dhash_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['oct_image','reference_image','hamming','selected_for_review','oct_group_key',
                'oct_group_members','reference_group_members','method']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for a in oct_reps:
            if all(r['image'] in matched_oct for r in a['members']):continue
            for b in ref_reps:
                d=(a['hash']^b['hash']).bit_count()
                distribution[d]+=1
                if d>DHASH_REVIEW_MAX_DISTANCE:continue
                candidate_pairs+=1
                if d<=threshold:
                    candidate_oct.update(r['image'] for r in a['members'])
                if write_count<2000000:
                    w.writerow(dict(oct_image=a['image'],reference_image=b['image'],hamming=d,
                       selected_for_review=d<=threshold,oct_group_key=a['members'][0]['file_group_key'],
                       oct_group_members=' | '.join(r['image'] for r in a['members']),
                       reference_group_members=' | '.join(r['image'] for r in b['members']),
                       method='unverified_cross_dataset_dhash_candidate'))
                    write_count+=1
    result=dict(reference_images=len(refs),october_images=len(images),
        oct_images_with_same_file_bytes=len(file_oct),
        oct_images_with_same_decoded_pixels_only=len(pixel_oct-file_oct),
        oct_images_with_strong_overlap=len(matched_oct),
        oct_images_with_unverified_dhash_hit=len(candidate_oct-matched_oct),
        oct_images_without_strong_overlap=len(images)-len(matched_oct),
        october_reference_dhash_candidate_representative_pairs=candidate_pairs,
        dhash_rows_written=write_count,dhash_rows_truncated=candidate_pairs>write_count,
        decoded_comparison_permitted=comparable,runtime_compatibility=env_note,
        cutoff=threshold,reference_manifest=str(ref),
        limitations='No exact/pixel match does not establish independence; dHash candidates need visual review. These are record counts, not unique acquisitions.')
    (out/'34_reference_overlap_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def main():
    checked_requirements()
    ap = argparse.ArgumentParser(description='CCBRE Oct-26 read-only data audit, two separate S3 contexts')
    ap.add_argument('--dataset', type=Path, help='Path to Oct-26\\Raw (contains both S3 folders)')
    ap.add_argument('--output', required=True, type=Path, help='Path to Oct-26\\Analysis')
    ap.add_argument('--no-sha256', action='store_true', help='Skip file hashing (not for evidence runs)')
    ap.add_argument('--fingerprint-only', action='store_true', help='Build manifest from explicit image root(s), e.g. July source images')
    ap.add_argument('--images-root', action='append',type=Path, help='One or more image directories for --fingerprint-only; only direct children by default')
    ap.add_argument('--recursive', action='store_true', help='Allow recursive image search in fingerprint-only mode (beware derived copies)')
    ap.add_argument('--reference-manifest', type=Path, help='CSV created with --fingerprint-only, used to compare October against July')
    ap.add_argument('--reference-labels', type=Path, help='July YOLO label root, searched recursively by image stem')
    ap.add_argument('--reference-yaml', type=Path, help='July class definition data.yaml')
    ap.add_argument('--reference-class-map', type=Path, help='Optional two-column July class name -> October class name CSV')
    ap.add_argument('--analysis-input', type=Path, help='Read saved October 02/11/13 manifests for --label-diff-only')
    ap.add_argument('--label-diff-only', action='store_true', help='Use existing October 02/11 manifests, no image decoding')
    ap.add_argument('--dhash-threshold', type=int, default=DHASH_DISTANCE_THRESHOLD, help='0..12 candidate link cutoff for extended grouping (default 6)')
    args = ap.parse_args()
    out = args.output.resolve()
    if not 0<=args.dhash_threshold<=DHASH_REVIEW_MAX_DISTANCE:
        ap.error('--dhash-threshold must be between 0 and 12')
    if args.fingerprint_only:
        if not args.images_root:ap.error('--fingerprint-only requires at least one --images-root')
        fingerprint_only_mode(args.images_root,out,args.recursive)
        return
    if (args.reference_labels is None) != (args.reference_yaml is None):
        ap.error('--reference-labels and --reference-yaml must be supplied together')
    if args.reference_labels and not args.reference_manifest:
        ap.error('--reference-labels and --reference-yaml require --reference-manifest')
    if args.label_diff_only:
        if not (args.reference_manifest and args.reference_labels and args.reference_yaml):
            ap.error('--label-diff-only requires --reference-manifest, --reference-labels, --reference-yaml')
        from v17_extensions import reference_label_diff, read_csv
        inp=(args.analysis_input or out).resolve()
        images_path=inp/'02_image_manifest.csv'
        objs_path=inp/'11_object_manifest.csv'
        names_path=inp/'13_taxonomy_mappings.json'
        if not (images_path.is_file() and objs_path.is_file() and names_path.is_file()):
            ap.error('label-diff-only needs an existing analysis directory with 02, 11, and 13 outputs')
        out.mkdir(parents=True,exist_ok=True)
        names_doc=json.loads(names_path.read_text(encoding='utf-8'))
        oct_names={int(k):v for k,v in names_doc.get('collection',names_doc.get('floorspread',{})).items()}
        from v17_extensions import day_level_groups
        loaded_images=read_csv(images_path)
        day_info=day_level_groups(loaded_images,out)
        result=reference_label_diff(args.reference_manifest,args.reference_labels,args.reference_yaml,
            loaded_images,read_csv(objs_path),oct_names,args.reference_class_map,out)
        print(json.dumps({'day_level_groups':day_info,'reference_label_diff':result},indent=2))
        return
    if args.dataset is None:ap.error('--dataset is required for the October audit')
    root = args.dataset.resolve()
    if not root.is_dir():
        ap.error(f'dataset directory not found: {root}')
    if out == root or root in out.parents:
        ap.error('output must be outside the raw directory')
    out.mkdir(parents=True, exist_ok=True)
    issues, images, objects, cooccurrence = [], [], [], Counter()
    names_by_source, name_status = {}, {}
    duplicates, resolutions, class_counts, class_image_support = defaultdict(list), Counter(), Counter(), Counter()
    size_by_context = defaultdict(list)
    counts_by_ctx = defaultdict(Counter)
    file_inventory = []
    sync_reconciliation = []
    export_buckets = []
    export_class_support = []
    provenance = []
    all_metadata = {}
    image_label_signatures = {}
    image_boxes = {}
    export_records = []
    context_inference = Counter()
    for context, folder in SOURCES.items():
        folder_root = root / folder
        if not folder_root.is_dir():
            issues.append(dict(image='', issue='source_folder_missing', component=context, detail=str(folder_root)))
            continue
        meta_hash_before={name:hash_file(folder_root/name) if (folder_root/name).is_file() else None
                           for name in ('data.yaml','report.md','.sync-state.json')}
        known, status = yaml_names(folder_root/'data.yaml')
        export_report = parse_export_report(folder_root/'report.md')
        if not (folder_root/'report.md').is_file():
            issues.append(dict(image='',issue='missing_partner_report',component=context,detail='report.md'))
        if not (folder_root/'.sync-state.json').is_file():
            issues.append(dict(image='',issue='missing_sync_state',component=context,detail='.sync-state.json'))
        sync_state = parse_sync_state(folder_root/'.sync-state.json')
        report_tables = parse_report_tables(folder_root/'report.md')
        all_metadata[context] = {
            'yaml_class_names':known,
            'report':export_report,
            'report_tables':report_tables,
            'sync':{k:v for k,v in sync_state.items() if k!='image_records'},
        }
        names_by_source[context] = known
        name_status[context] = status
        if status != 'ok':
            issues.append(dict(image='', issue='class_map_problem', component=context, detail=status))
        for integrity_key in ('missing_referenced_annotations','source_image_reference_mismatches',
                              'missing_referenced_source_objects'):
            if sync_state.get(integrity_key,0):
                issues.append(dict(image='',issue=integrity_key,component=context,
                                   detail=str(sync_state[integrity_key])))
        if sync_state.get('present') and [known.get(i) for i in range(len(EXPECTED))] != sync_state.get('classes'):
            issues.append(dict(image='',issue='yaml_vs_sync_class_mismatch',component=context, detail='data.yaml and sync-state.json differ'))
        if export_report.get('dropped_label_masks',0):
            issues.append(dict(image='',issue='source_masks_excluded_before_export',component=context,detail=f"{export_report['dropped_label_masks']} source masks excluded by exporter; inspect report.md dropped-name table"))
        for cid in range(len(EXPECTED)):
            if cid not in known:
                issues.append(dict(image='', issue='expected_class_absent_from_yaml', component=context, detail=f'expected id={cid}, name={EXPECTED[cid]}'))
            elif normalized(known[cid]) != normalized(EXPECTED[cid]):
                issues.append(dict(image='', issue='class_name_differs_from_email', component=context,
                                   detail=f'id={cid}, yaml={known[cid]}, email={EXPECTED[cid]}'))
        reported_dropped = export_report.get('dropped_label_masks',0)
        table_dropped = sum(report_tables['dropped_labels'].values())
        sync_dropped_total = sum(sync_state.get('dropped_labels',{}).values())
        if reported_dropped != table_dropped or reported_dropped != sync_dropped_total:
            issues.append(dict(image='',issue='excluded_source_mask_count_mismatch',
                               component=context, detail=f'report={reported_dropped}; '
                               f'report table={table_dropped}; sync state={sync_dropped_total}'))
        for sub in ('images','labels','labels-detect'):
            if not (folder_root / sub).is_dir():
                issues.append(dict(image='', issue='annotation_or_image_folder_missing', component=context, detail=sub))
        image_root = folder_root / 'images'
        poly_root = folder_root / 'labels'
        box_root = folder_root / 'labels-detect'
        image_files = sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in EXT) if image_root.is_dir() else []
        def issue(relative, kind, component, detail):
            issues.append(dict(image=relative, issue=kind, component=component, detail=detail))
        image_relative_stems = Counter(str(p.relative_to(image_root).with_suffix('')).lower() for p in image_files)
        actual_sync_names = []
        local_source_buckets = Counter()
        source_class_image_support = Counter()
        for stem, c in image_relative_stems.items():
            if c > 1: issue(stem, 'duplicate_image_stem', 'images', f'{c} image files with same stem')
        print(f'Analyzing {context}: {len(image_files):,} images',flush=True)
        for image_number,img in enumerate(image_files,1):
            if image_number % 500 == 0:
                print(f'  {context}: processed {image_number:,} / {len(image_files):,}',flush=True)
            subpath = img.relative_to(image_root)
            rel = str(Path(folder) / 'images' / subpath)
            label_rel = subpath.with_suffix('.txt')
            sync_key = normalized_sync_key(subpath)
            if sync_key:
                actual_sync_names.append(sync_key)
                source_bucket = source_id_from_sync_key(sync_key)
                local_source_buckets[source_bucket] += 1
            else:
                source_bucket = 'UNKNOWN'
                issue(rel,'filename_not_sync_encoded','provenance','Expected source_bucket__original_filename format')
            sync_rec = sync_state.get('image_records',{}).get(sync_key)
            if sync_rec is None:
                issue(rel,'local_image_absent_from_sync','provenance', str(sync_key))
                sync_obj_counts = {}
                sync_dropped = {}
                source_annotation_id=''
                annotation_updated_utc=''
            else:
                source_annotation_id = str(sync_rec[0]) if len(sync_rec)>0 else ''
                annotation_updated_utc = str(sync_rec[1]) if len(sync_rec)>1 else ''
                sync_obj_counts = {int(k):int(v) for k,v in sync_rec[2].items()}
                sync_dropped = sync_rec[3] if len(sync_rec)>3 and isinstance(sync_rec[3],dict) else {}
            box_path = box_root/label_rel
            poly_path = poly_root/label_rel
            bb, box_state = parse_annotation(box_path, 'box', known, issue, rel)
            poly, poly_state = parse_annotation(poly_path, 'polygon', known, issue, rel)
            box_file_sha256 = hash_file(box_path) if box_path.is_file() else ''
            polygon_file_sha256 = hash_file(poly_path) if poly_path.is_file() else ''
            if box_state in ('missing','unreadable'):
                issue(rel, 'missing_box_label' if box_state=='missing' else 'unreadable_box_label', 'box',str(box_path))
            if poly_state in ('missing','unreadable'):
                issue(rel, 'missing_polygon_label' if poly_state=='missing' else 'unreadable_polygon_label', 'polygon',str(poly_path))
            if box_state == 'empty' and poly_state == 'nonempty' or box_state == 'nonempty' and poly_state == 'empty':
                issue(rel,'annotation_empty_nonempty_mismatch','geometry', f'box={box_state}, polygon={poly_state}')
            if box_state == 'nonempty' and poly_state == 'nonempty':
                compare_geometry(bb,poly,rel,issue)
            observed_classes = dict(Counter(x['class_id'] for x in bb))
            if sync_rec is not None and observed_classes != sync_obj_counts:
                issue(rel,'per_image_sync_class_count_mismatch','provenance',
                      f'local={observed_classes}; sync={sync_obj_counts}')
            if len(bb) != len(poly) and box_state not in ('missing','unreadable') and poly_state not in ('missing','unreadable'):
                issue(rel,'box_polygon_count_mismatch','geometry',f'{len(bb)} boxes, {len(poly)} polygons')
            width,height, image_extras = inspect_image(img,issue,rel)
            if args.no_sha256:
                sha = ''
            else:
                try: sha = hash_file(img)
                except Exception as exc:
                    sha = ''; issue(rel,'sha256_failed','image',str(exc))
            if sha: duplicates[sha].append(rel)
            image_label_signatures[rel] = bbox_signature(bb)
            image_boxes[rel] = bb
            has_pair = box_state not in ('missing','unreadable') and poly_state not in ('missing','unreadable')
            if box_state == 'missing': state='missing_annotation'
            elif box_state == 'unreadable': state='unreadable_annotation'
            elif box_state == 'empty': state='explicit_empty' if poly_state=='empty' else 'needs_review'
            else: state='positive' if bb else 'needs_review'
            names_present=set()
            for a in bb:
                cls = a['class_id']; name = known.get(cls,f'UNKNOWN_CLASS_{cls}')
                names_present.add(cls)
                class_counts[(context,cls)] += 1
                area = max(0., a['bbox'][2]-a['bbox'][0])*max(0.,a['bbox'][3]-a['bbox'][1])
                size_by_context[context].append(area)
                # Primary measurement assumes label coordinates in the EXIF-display
                # frame. Also export encoded-frame hypothesis for undecided EXIF.
                xy=a['bbox']
                dw=image_extras.get('display_width') or 0
                dh=image_extras.get('display_height') or 0
                ew,eh=width or 0,height or 0
                native_display=min((xy[2]-xy[0])*dw,(xy[3]-xy[1])*dh) if dw and dh else None
                native_encoded=min((xy[2]-xy[0])*ew,(xy[3]-xy[1])*eh) if ew and eh else None
                objects.append(dict(image=rel,context=context,class_id=cls,class_name=name,
                    xmin=xy[0],ymin=xy[1],xmax=xy[2],ymax=xy[3],
                    bbox_area_frac=round(area,9),valid=a['valid'],line=a['line'],
                    coordinate_frame_assumption='display_untransformed',
                    exif_orientation=image_extras['exif_orientation'],
                    short_side_native_px=round(native_display,3) if native_display is not None else '',
                    short_side_1536_px=round(native_display*(1536/max(dw,dh)),3)
                        if native_display is not None else '',
                    short_side_native_encoded_px=round(native_encoded,3) if native_encoded is not None else '',
                    short_side_1536_encoded_px=round(native_encoded*(1536/max(ew,eh)),3)
                        if native_encoded is not None else ''))
            for cls in names_present: class_image_support[(context,cls)] += 1
            for a in sorted(names_present):
                for b in sorted(x for x in names_present if x>=a):cooccurrence[(context,a,b)]+=1
            if width and height:
                resolutions[(context, image_extras['display_width'], image_extras['display_height'], width, height)] += 1
            filename_fields = parse_source_filename(img.name,context)
            if not filename_fields['filename_parse_ok']:
                issue(rel, 'filename_pattern_unparsed', 'provenance', img.name)
            source_hint = 'source_name_contains_synthetic_unverified' if 'synthetic' in img.name.lower() else 'unspecified'
            # Historical Jul-26 suffix rule documented in master report. Filename-inferred only.
            acquisition = 'floorspread' if context=='floorspread' else infer_collection_context(img.name)
            context_inference[(context,acquisition)] += 1
            image_row=dict(image=rel,context=context,acquisition_context=acquisition,
                           filename=img.name,filename_prefix=img.stem.split('__')[0] if '__' in img.stem else img.stem.split('_')[0],
                           source_bucket=source_bucket,source_object_key=sync_key or '',
                           source_annotation_id=source_annotation_id,annotation_updated_utc=annotation_updated_utc,
                           excluded_source_masks=sum(int(v) for v in sync_dropped.values()),
                           provenance_hint=source_hint,width=width or '',height=height or '',
                           bytes=img.stat().st_size,sha256=sha,
                           box_label_sha256=box_file_sha256,
                           polygon_label_sha256=polygon_file_sha256,
                           box_label_state=box_state,
                           polygon_label_state=poly_state,annotation_state=state,
                           detection_instances=len(bb),polygon_instances=len(poly),
                           classes='|'.join(known.get(c,str(c)) for c in sorted(names_present)),
                           **image_extras,**filename_fields)
            images.append(image_row)
            if sync_dropped:
                issue(rel, 'source_masks_excluded_before_export_image', 'provenance',json.dumps(sync_dropped))
            counts_by_ctx[context]['images'] += 1
            counts_by_ctx[context]['bytes'] += image_row['bytes']
            counts_by_ctx[context]['objects'] += len(bb)
            counts_by_ctx[context]['polygons'] += len(poly)
            counts_by_ctx[context][state] += 1
            if not all(a['valid'] for a in bb): counts_by_ctx[context]['images_with_invalid_boxes'] += 1
        # Compare downloaded local files to the exact image keys in .sync-state.
        if sync_state.get('present'):
            for key in sorted(set(sync_state['image_records'])-set(actual_sync_names)):
                issue('', 'sync_image_missing_locally', context, key)
            if len(actual_sync_names)!=len(set(actual_sync_names)):
                issue('', 'source_sync_key_collision', context,'Multiple local filenames resolve to one sync-state source key')
        for bucket in sorted(set(local_source_buckets)|set(report_tables['source_buckets'])):
            local_count=local_source_buckets[bucket]
            declared=report_tables['source_buckets'].get(bucket)
            export_buckets.append(dict(context=context,source_bucket=bucket,local_images=local_count,
                                       reported_images=declared,match=local_count==declared))
            if declared!=local_count:
                issue('','source_bucket_images_mismatch',context,
                      f'{bucket}: local={local_count}, report={declared}')
        # Orphan label files are not to be mistaken for negative images.
        for mode, base in (('box',box_root),('polygon',poly_root)):
            if not base.is_dir(): continue
            for txt in sorted(base.rglob('*.txt')):
                stem = str(txt.relative_to(base).with_suffix('')).lower()
                file_inventory.append(dict(context=context,kind=mode,label=str(txt.relative_to(root))))
                if stem not in image_relative_stems:
                    issue(str(txt.relative_to(root)),f'orphan_{mode}_label',mode,'No matching image stem')
        for name,old_sha in meta_hash_before.items():
            path=folder_root/name
            new_sha=hash_file(path) if path.is_file() else None
            if old_sha != new_sha:
                issue('','metadata_changed_during_analysis',context,
                      f'{name} changed during this audit; rerun after synchronization is finished')
        # Reconcile THREE independent evidence layers: downloaded data, export report, sync-state.
        actual=counts_by_ctx[context]
        er=dict(context=context,report_generated=export_report.get('generated_at'),
                raw_images=actual['images'],raw_positive=actual['positive'],
                report_images=export_report.get('images'),
                report_positive=(export_report.get('images')-export_report.get('without_labels'))
                    if export_report.get('images') is not None and export_report.get('without_labels') is not None else None,
                sync_positive=sync_state.get('annotated_images'),
                sync_images=sync_state.get('images'),raw_empty=actual['explicit_empty'],
                report_without_labels=export_report.get('without_labels'),
                raw_objects=actual['objects'],report_objects=export_report.get('objects'),
                sync_objects=sync_state.get('objects'),
                dropped_before_export=export_report.get('dropped_label_masks',0),
                export_images_match=actual['images']==export_report.get('images')==sync_state.get('images'),
                export_objects_match=actual['objects']==export_report.get('objects')==sync_state.get('objects'))
        export_records.append(er)
        for label,match in [('image_total',er['export_images_match']),('object_total',er['export_objects_match'])]:
            if export_report.get('images') is not None and sync_state.get('present') and not match:
                issue('', 'export_'+label+'_mismatch',context,json.dumps(er))
        if sync_state.get('present') and actual['positive'] != sync_state.get('annotated_images'):
            issue('', 'sync_positive_image_mismatch',context,
                  f'actual={actual["positive"]}; metadata={sync_state.get("annotated_images")}')
        if export_report.get('without_labels') is not None and export_report.get('without_labels') != actual['explicit_empty']:
            issue('', 'export_empty_image_mismatch',context,
                  f"reported={export_report.get('without_labels')} actual={actual['explicit_empty']}")
        if export_report.get('classes_with_labels') is not None:
            actually_present=sum(class_counts[(context,cls)]>0 for cls in range(len(EXPECTED)))
            if actually_present != export_report['classes_with_labels']:
                issue('', 'export_nonzero_class_count_mismatch', context,
                      f'actual={actually_present}; report={export_report["classes_with_labels"]}')
        for cls in range(len(EXPECTED)):
            reportrow=report_tables['classes'].get(cls,{})
            obs_instances=class_counts[(context,cls)]
            obs_support=class_image_support[(context,cls)]
            sync_instances=sync_state.get('class_instances',{}).get(cls,0)
            export_class_support.append(dict(context=context,class_id=cls,class_name=known.get(cls,''),
                local_instances=obs_instances,local_images=obs_support,
                report_instances=reportrow.get('instances',''),report_images=reportrow.get('images',''),
                sync_instances=sync_instances,
                local_vs_report_instances=(obs_instances==reportrow.get('instances')),
                local_vs_report_images=(obs_support==reportrow.get('images')),
                local_vs_sync_instances=(obs_instances==sync_instances)))
            if reportrow:
                if normalized(str(reportrow.get('name',''))) != normalized(known.get(cls,'')):
                    issue('', 'report_class_name_mismatch', context,f'id={cls}; report={reportrow.get("name")}; yaml={known.get(cls)}')
                if obs_instances != reportrow.get('instances') or obs_support!=reportrow.get('images'):
                    issue('', 'report_class_stats_mismatch',context,
                          f'id={cls}; objects local={obs_instances} report={reportrow.get("instances")}; '
                          f'images local={obs_support} report={reportrow.get("images")}')
        for cls in range(len(EXPECTED)):
            observed=class_counts[(context,cls)]
            expected=sync_state.get('class_instances',{}).get(cls,0)
            if sync_state.get('present') and observed != expected:
                issue('', 'sync_class_count_mismatch',context,f'id={cls}; observed={observed}, state={expected}')
    number_dsc_runs,number_time_runs = summarize_group_keys(images)
    # DSC stems can wrap/recur across devices; detect same number from named cameras.
    floor_dsc=defaultdict(list)
    for row in images:
        if row['context']=='floorspread' and row['dsc_number']!='':
            floor_dsc[int(row['dsc_number'])].append(row)
    dsc_repeats={n:rows for n,rows in floor_dsc.items() if len(rows)>1}
    cross_camera_dsc=[]
    for n,rows in sorted(dsc_repeats.items()):
        identities={(r.get('exif_camera_make',''),r.get('exif_camera_model','')) for r in rows
                    if r.get('exif_camera_make','') or r.get('exif_camera_model','')}
        if len(identities)>1:
            cross_camera_dsc.append(n)
            issues.append(dict(image=' | '.join(r['image'] for r in rows),
                issue='dsc_number_repeated_across_camera_models',component='provenance',
                detail=f'DSC{n}: camera identities={sorted(identities)}; do not treat DSC number as a unique acquisition identity'))
    for c in ('collection','floorspread'):
        policy=all_metadata.get(c,{}).get('sync',{})
        if 'diverses' in [str(t).lower() for t in policy.get('ignore',[])]:
            if any(str(k).lower()=='diverses' and str(v).lower()=='unknown'
                   for k,v in policy.get('renames',{}).items()):
                issues.append(dict(image='',issue='ignore_rename_order_ambiguous',component=c,
                    detail='Exporter ignore includes diverses and rename Diverses->Unknown; ask exporter for evaluation order and ignored-label counts.'))
    # Different YAML mappings between exports would silently corrupt cross-context statistics.
    if len(names_by_source)==2 and names_by_source['collection'] != names_by_source['floorspread']:
        issues.append(dict(image='',issue='class_maps_differ_between_contexts',component='taxonomy',
                           detail='STOP before harmonizing or combining the exports'))
    if len(all_metadata)==2:
        c=all_metadata['collection']['sync']
        f=all_metadata['floorspread']['sync']
        for policy in ('renames','ignore'):
            if c.get(policy)!=f.get(policy):
                issues.append(dict(image='',issue='export_policies_differ',component='taxonomy',
                                   detail=f'{policy}: collection and floorspread differ'))
    # SHA-256 exact duplicates only; additionally, decoded-pixel and dHash candidates
    # are produced separately and NEVER automatically treated as duplicates.
    dup_rows=[]
    duplicate_pair_details=[]
    for h,paths in duplicates.items():
        if len(paths)<=1:continue
        sigs=[image_label_signatures[p] for p in paths]
        contexts_in_group={p.replace('\\','/').split('/')[0] for p in paths}
        for p1,p2 in combinations(paths,2):
            comp=category_duplicate_pair(image_boxes[p1],image_boxes[p2])
            identical=image_label_signatures[p1]==image_label_signatures[p2]
            if identical: comp.update(tier='identical',equivalent=True,min_matched_iou=1.0)
            duplicate_pair_details.append(dict(sha256=h, image_a=p1,image_b=p2,
                 count_a=len(image_boxes[p1]),count_b=len(image_boxes[p2]),
                 exact_label_signature=identical,**comp))
        dup_rows.append(dict(sha256=h,count=len(paths),paths=' | '.join(paths),
                      spans_contexts=len(contexts_in_group)>1,
                      annotation_conflict=len(set(sigs))>1,
                      object_count_conflict=len(set(len(s) for s in sigs))>1,
                      annotation_counts=' | '.join(str(len(s)) for s in sigs)))
    dup_rows.sort(key=lambda r:r['sha256'])
    duplicate_pair_details.sort(key=lambda r:(r['sha256'],r['image_a']))
    for d in dup_rows:
        if d['annotation_conflict']:
            issues.append(dict(image='',issue='duplicate_image_conflicting_box_labels',
                               component='duplicate',detail=f'{d["sha256"]}: {d["paths"]}'))
        if d['spans_contexts']:
            issues.append(dict(image='',issue='duplicate_image_spans_contexts',
                               component='duplicate',detail=f'{d["sha256"]}: {d["paths"]}'))
    # All-pairs dHash scan, including across source contexts and filename groups.
    # Exact decoded pixel matches (different file hashes) are independently listed.
    pixel_duplicate_rows=decoded_pixel_duplicate_groups(images)
    write_csv(out/'28_pixel_content_duplicates.csv',
        ['display_pixel_sha256','images','distinct_file_sha256','contexts',
         'group_keys','crosses_contexts','crosses_filename_groups','paths'],pixel_duplicate_rows)
    strict_uf,indices=make_strict_groups(images)
    extended_uf=strict_uf.clone()
    dhash_metrics=global_dhash_scan(images,out/'23_dhash_candidates.csv',extended_uf,indices,threshold=args.dhash_threshold)
    split_metrics=export_split_groups(images,strict_uf,extended_uf,out)
    from v17_extensions import day_level_groups
    day_metrics=day_level_groups(images,out)
    # Keep explicit alias filename to avoid obscuring the changed scan semantics.
    # All candidates are review leads, not confirmed duplicates.
    img_total = len(images)
    contexts = [dict(context=c,image_count=d['images'],positive_images=d['positive'],
                     explicit_empty_images=d['explicit_empty'],unlabelled_images=d['missing_annotation']+d['unreadable_annotation'],
                     needs_review_images=d['needs_review'],detection_objects=d['objects'],
                     polygon_objects=d['polygons'],bytes=d['bytes'])
                for c,d in sorted(counts_by_ctx.items())]
    # Positive count already accumulated once per image; do not double-count.
    class_rows=[]
    for context in ('collection','floorspread','all'):
        keys = list(range(len(EXPECTED)))  # report all 29, including zero-support classes
        for cls in keys:
            n = sum(v for (c,k),v in class_counts.items() if k==cls and (c==context or context=='all'))
            support = sum(v for (c,k),v in class_image_support.items() if k==cls and (c==context or context=='all'))
            cname=next((names_by_source[c].get(cls) for c in ('collection','floorspread') if cls in names_by_source.get(c,{})),f'UNKNOWN_{cls}')
            class_rows.append(dict(context=context,class_id=cls,class_name=cname,instances=n,
                                   images_containing_class=support,
                                   images_for_target=counts_by_ctx[context]['images'] if context!='all' else img_total,
                                   instances_per_image=round(n/support,4) if support else 0,
                                   share_of_all_images_pct=round(100*support/(counts_by_ctx[context]['images'] if context!='all' else img_total),3) if img_total else 0))
    size_rows=[]
    for c, arr in sorted(size_by_context.items()):
        ordered=sorted(arr)
        def quantile(q):
            if not ordered:return None
            ix=(len(ordered)-1)*q
            lo=math.floor(ix); hi=math.ceil(ix)
            return ordered[lo]*(hi-ix)+ordered[hi]*(ix-lo) if lo!=hi else ordered[lo]
        size_rows.append(dict(context=c,objects=len(arr),mean_area_pct=round(100*sum(arr)/len(arr),5) if arr else 0,
                              median_area_pct=round(100*quantile(.5),5) if arr else 0,
                              p10_area_pct=round(100*quantile(.1),5) if arr else 0,
                              p90_area_pct=round(100*quantile(.9),5) if arr else 0,
                              under_0_1pct=sum(v<.001 for v in arr),under_1pct=sum(v<.01 for v in arr)))
    prefix_counts=Counter((r['context'],r['filename_prefix']) for r in images)
    write_csv(out/'21_capture_group_manifest.csv',
        ['image','context','q_code','stop','sub_index','view_letter','dsc_number','run_id',
         'file_group_key','time_gap_group_key','camera_id','camera_identity_key',
         'group_method','filename_parse_ok','exif_datetime_original',
         'exif_camera_make','exif_camera_model','exif_orientation','sha256','display_pixel_sha256','dhash64',
         'split_group_strict','split_group_extended'], images)
    write_csv(out/'22_duplicate_conflict_tiers.csv',
        ['sha256','image_a','image_b','count_a','count_b','exact_label_signature',
         'tier','min_matched_iou','equivalent','positive_empty','count_change','class_change'],
        duplicate_pair_details)
    group_rows=[]
    grouped=defaultdict(list)
    for row in images:
        if row['file_group_key']:
            grouped[row['file_group_key']].append(row)
    for g, members in sorted(grouped.items()):
        flags=Counter(r['view_letter'] for r in members if r['view_letter'])
        exif_times=sorted(r['exif_datetime_original'] for r in members if r['exif_datetime_original'])
        group_rows.append(dict(group_key=g,context=members[0]['context'],
             images=len(members),distinct_sha256=len({r['sha256'] for r in members}),
             c=flags['c'],w=flags['w'],b=flags['b'],t=flags['t'],
             min_dsc=min((int(r['dsc_number']) for r in members if r['dsc_number']!=''),default=''),
             max_dsc=max((int(r['dsc_number']) for r in members if r['dsc_number']!=''),default=''),
             exif_camera_makes=' | '.join(sorted({r.get('exif_camera_make','') for r in members if r.get('exif_camera_make','')})),
             exif_camera_models=' | '.join(sorted({r.get('exif_camera_model','') for r in members if r.get('exif_camera_model','')})),
             timestamp_session_group_keys=len({r['time_gap_group_key'] for r in members if r.get('time_gap_group_key')}),
             exif_timestamp_coverage=len(exif_times),
             min_exif_timestamp=exif_times[0] if exif_times else '',
             max_exif_timestamp=exif_times[-1] if exif_times else ''))
    write_csv(out/'24_capture_group_summary.csv',
        ['group_key','context','images','distinct_sha256','c','w','b','t','min_dsc','max_dsc',
         'exif_camera_makes','exif_camera_models','timestamp_session_group_keys','exif_timestamp_coverage',
         'min_exif_timestamp','max_exif_timestamp'],group_rows)
    policy_rows=[]
    for c in ('collection','floorspread'):
        pol=all_metadata.get(c,{}).get('sync',{})
        policy_rows.append(dict(context=c, ignored_source_labels=' | '.join(pol.get('ignore',[])),
             rename_diverses_to_unknown=pol.get('renames',{}).get('Diverses',''),
             exported_unknown_objects=class_counts[(c,27)],
             source_objects_records=pol.get('source_objects_records',0),
             source_annotation_records=pol.get('source_annotations_records',0),
             unreferenced_annotation_history_records=pol.get('unreferenced_annotation_history_records',0),
             source_masks_excluded=sum(pol.get('dropped_labels',{}).values())))
    write_csv(out/'27_export_policy_history_summary.csv',
        ['context','ignored_source_labels','rename_diverses_to_unknown','exported_unknown_objects',
         'source_objects_records','source_annotation_records',
         'unreferenced_annotation_history_records','source_masks_excluded'],policy_rows)
    write_csv(out/'25_view_letter_statistics.csv',
        ['view_letter','images','positive','explicit_empty','objects'],
        [dict(view_letter=c,images=sum(x['view_letter']==c for x in images),
              positive=sum(x['view_letter']==c and x['annotation_state']=='positive' for x in images),
              explicit_empty=sum(x['view_letter']==c and x['annotation_state']=='explicit_empty' for x in images),
              objects=sum(x['detection_instances'] for x in images if x['view_letter']==c))
         for c in ('c','w','t','b')])
    inference_size=[]
    for c in ('collection','floorspread'):
        values=sorted(float(o['short_side_1536_px']) for o in objects if o['context']==c and o.get('short_side_1536_px')!='')
        def pct(v,q):
            if not v:return ''
            idx=(len(v)-1)*q; low=int(idx); high=math.ceil(idx)
            return v[low]*(high-idx)+v[high]*(idx-low) if low!=high else v[low]
        inference_size.append(dict(context=c,objects=len(values),
             median_short_px=round(pct(values,.5),2) if values else '',
             p10_short_px=round(pct(values,.1),2) if values else '',
             under_32px=sum(v<32 for v in values),under_16px=sum(v<16 for v in values)))
    write_csv(out/'26_inference_short_side_statistics.csv',
        ['context','objects','median_short_px','p10_short_px','under_32px','under_16px'],inference_size)
    write_csv(out/'02_image_manifest.csv', list(images[0].keys()) if images else ['image'], images)
    write_csv(out/'03_class_statistics.csv', ['context','class_id','class_name','instances','images_containing_class','images_for_target','instances_per_image','share_of_all_images_pct'],class_rows)
    write_csv(out/'04_source_context_statistics.csv', ['context','image_count','positive_images','explicit_empty_images','unlabelled_images','needs_review_images','detection_objects','polygon_objects','bytes'],contexts)
    write_csv(out/'05_resolution_statistics.csv',
        ['context','width','height','encoded_width','encoded_height','images','reference'],
        [dict(context=c,width=w,height=h,encoded_width=ew,encoded_height=eh,images=n,reference='EXIF_display_frame')
         for (c,w,h,ew,eh),n in sorted(resolutions.items())])
    write_csv(out/'06_object_size_statistics.csv',['context','objects','mean_area_pct','median_area_pct','p10_area_pct','p90_area_pct','under_0_1pct','under_1pct'],size_rows)
    write_csv(out/'07_quality_issues.csv',['image','issue','component','detail'],issues)
    write_csv(out/'08_exact_duplicates.csv',['sha256','count','paths','spans_contexts','annotation_conflict','object_count_conflict','annotation_counts'],dup_rows)
    write_csv(out/'09_filename_prefixes.csv',['context','prefix','images'],[dict(context=c,prefix=p,images=n) for (c,p),n in sorted(prefix_counts.items(),key=lambda x:(x[0][0],-x[1],x[0][1]))])
    write_csv(out/'10_cooccurrence.csv',['context','class_a','class_b','images'],[dict(context=c,class_a=a,class_b=b,images=n) for (c,a,b),n in sorted(cooccurrence.items())])
    write_csv(out/'11_object_manifest.csv',['image','context','class_id','class_name','xmin','ymin','xmax','ymax','bbox_area_frac','valid','line','coordinate_frame_assumption','exif_orientation',
        'short_side_native_px','short_side_1536_px','short_side_native_encoded_px','short_side_1536_encoded_px'],objects)
    write_csv(out/'12_source_label_inventory.csv',['context','kind','label'],file_inventory)
    write_csv(out/'14_export_reconciliation.csv',
        ['context','report_generated','raw_images','raw_positive','report_images','report_positive',
         'sync_positive','sync_images','raw_empty','report_without_labels','raw_objects',
         'report_objects','sync_objects','dropped_before_export','export_images_match','export_objects_match'],export_records)
    write_csv(out/'15_collection_context_inference.csv',['source','context_inferred','images'],[dict(source=c,context_inferred=a,images=n) for (c,a),n in sorted(context_inference.items())])
    write_csv(out/'17_source_bucket_reconciliation.csv',['context','source_bucket','local_images','reported_images','match'],export_buckets)
    write_csv(out/'18_class_reconciliation.csv',['context','class_id','class_name','local_instances','local_images','report_instances','report_images','sync_instances','local_vs_report_instances','local_vs_report_images','local_vs_sync_instances'],export_class_support)
    (out/'19_export_metadata.json').write_text(json.dumps(all_metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(out/'16_duplicate_annotation_conflicts.csv',['sha256','count','paths','annotation_conflict','object_count_conflict','annotation_counts'],[r for r in dup_rows if r['annotation_conflict']])
    source_metadata_hashes={}
    for context, folder in SOURCES.items():
        source_metadata_hashes[context]={}
        for filename in ('data.yaml','report.md','.sync-state.json'):
            path=root/folder/filename
            source_metadata_hashes[context][filename]=hash_file(path) if path.is_file() else None
    frozen_inventory_sha256=hashlib.sha256()
    for row in sorted(images,key=lambda r:r['image']):
        frozen_inventory_sha256.update(json.dumps(
            [row['image'],row['sha256'],row['box_label_sha256'],row['polygon_label_sha256']],
            separators=(',',':')).encode('utf-8'))
        frozen_inventory_sha256.update(b'\n')
    for context in sorted(source_metadata_hashes):
        for filename in sorted(source_metadata_hashes[context]):
            frozen_inventory_sha256.update(f'{context}/{filename}:'.encode('utf-8'))
            frozen_inventory_sha256.update(str(source_metadata_hashes[context][filename]).encode('utf-8'))
            frozen_inventory_sha256.update(b'\n')
    summary = dict(analyzer_version=VERSION,generated_at=datetime.now(timezone.utc).isoformat(),
                   inventory_fingerprint_sha256=frozen_inventory_sha256.hexdigest(),
                   inventory_fingerprint_includes_image_contents=not args.no_sha256,
                   metadata_sha256=source_metadata_hashes,
                   dataset_root=str(root),image_count=img_total,
                   last_run_does_not_confirm_remote_snapshot_is_final=True,
                   detection_object_count=len(objects), polygon_object_count=sum(r['polygon_instances'] for r in images),
                   positive_images=sum(r['annotation_state']=='positive' for r in images),
                   explicitly_empty_images=sum(r['annotation_state']=='explicit_empty' for r in images),
                   missing_or_unreadable_detection_label_images=sum(r['annotation_state'] in ('missing_annotation','unreadable_annotation') for r in images),
                   ambiguous_annotation_images=sum(r['annotation_state']=='needs_review' for r in images),
                   source_contexts=contexts,
                   issues_by_type=dict(Counter(x['issue'] for x in issues)),
                   duplicate_hash_groups=len(dup_rows) if not args.no_sha256 else None,
                   duplicate_check_status='skipped' if args.no_sha256 else 'completed',
                   metadata_reconciliation_passed=not any(
                      i['issue'] in ('per_image_sync_class_count_mismatch','source_bucket_images_mismatch',
                      'export_image_total_mismatch','export_object_total_mismatch',
                      'report_class_stats_mismatch','sync_image_missing_locally',
                      'local_image_absent_from_sync','class_maps_differ_between_contexts',
                      'missing_partner_report','missing_sync_state','class_map_problem',
                      'metadata_changed_during_analysis',
                      'export_empty_image_mismatch','sync_class_count_mismatch',
                      'expected_class_absent_from_yaml','class_name_differs_from_email',
                      'report_class_name_mismatch','export_policies_differ',
                      'excluded_source_mask_count_mismatch',
                      'export_nonzero_class_count_mismatch','sync_positive_image_mismatch',
                      'missing_referenced_annotations','source_image_reference_mismatches',
                      'missing_referenced_source_objects',
                      'yaml_vs_sync_class_mismatch') for i in issues),
                   cross_source_duplicate_hash_groups=sum(bool(d['spans_contexts']) for d in dup_rows),
                   duplicate_groups_with_conflicting_labels=sum(bool(d['annotation_conflict']) for d in dup_rows),
                   export_reconciliation=export_records,
                   inferred_context_counts={k[1]:v for k,v in context_inference.items() if k[0]=='collection'},
                   sha256_enabled=not args.no_sha256,
                   dimensions_available=Image is not None,
                   warning='Collection subcontexts are filename-inferred from the established Jul-26 c/b/t versus w convention; not independently confirmed for October. Floorspread is explicit by folder.',
                   caveat='Filename synthetic prefixes do not prove AI-generated images. Check export provenance.',
                   class_map_status=name_status,
                   validation_status='REQUIRES_REVIEW' if issues else 'STRUCTURAL_CHECKS_PASS',
                   exif_datetime_original_record_count=sum(bool(x['exif_datetime_original']) for x in images),
                   exif_orientation_nontrivial_count=sum(x['exif_orientation'] not in (1,'') for x in images),
                   image_capture_time_available=any(bool(x['exif_datetime_original']) for x in images),
                   decoded_pixel_hash_available_count=sum(bool(x['display_pixel_sha256']) for x in images),
                   decoded_pixel_hash_missing_count=sum(not x['display_pixel_sha256'] for x in images),
                   dhash_available_count=sum(bool(x['dhash64']) for x in images),
                   decoded_pixel_hash_groups_different_file_hashes=len(pixel_duplicate_rows),
                   decoded_pixel_hash_groups_crossing_filename_groups=sum(bool(r['crosses_filename_groups']) for r in pixel_duplicate_rows),
                   image_headers_with_dimensions_count=sum(bool(x['width'] and x['height']) for x in images),
                   exif_make_record_count=sum(bool(x['exif_camera_make']) for x in images),
                   exif_model_record_count=sum(bool(x['exif_camera_model']) for x in images),
                   runtime=pillow_runtime(),
                   inferred_contiguous_DSC_runs=number_dsc_runs,
                   inferred_time_gap_sessions=number_time_runs,
                   exif_timestamp_parseable_count=sum(parse_exif_datetime(x['exif_datetime_original']) is not None for x in images),
                   repeated_DSC_numbers=len(dsc_repeats),
                   DSC_numbers_with_multiple_camera_identities=len(cross_camera_dsc),
                   inferred_filename_group_count=len(grouped),
                   duplicate_conflict_tiers=dict(Counter(r['tier'] for r in duplicate_pair_details)),
                   dhash_near_candidate_pairs=dhash_metrics['total'],
                   dhash_candidate_scan=dhash_metrics,split_groups=split_metrics,
                   day_level_split_groups=day_metrics)

    checks = {
        'sources_present':all((root/f).is_dir() for f in SOURCES.values()),
        'images_match_context_sum':img_total==sum(r['image_count'] for r in contexts),
        'positive_empty_review_accounting':img_total == sum(
            r['positive_images']+r['explicit_empty_images']+r['unlabelled_images']+
            r['needs_review_images'] for r in contexts),
        'object_count_matches_context_sum':len(objects)==sum(r['detection_objects'] for r in contexts),
        'polygon_count_matches_context_sum':summary['polygon_object_count']==sum(
            r['polygon_objects'] for r in contexts),
        'object_count_matches_class_sum':len(objects)==sum(r['instances'] for r in class_rows if r['context']=='all'),
        'source_classes_are_complete':len([r for r in class_rows if r['context']=='all'])==29,
        'class_maps_identical':len(names_by_source)==2 and
            names_by_source.get('collection')==names_by_source.get('floorspread'),
        'image_source_keys_unique':len({(r['context'],r['source_object_key']) for r in images})==img_total,
    }
    summary['internal_consistency_checks'] = checks
    summary['internal_consistency_passed']=all(checks.values())
    if args.reference_manifest:
        summary['reference_overlap']=reference_comparison(images,args.reference_manifest,out,args.dhash_threshold)
        if args.reference_labels and args.reference_yaml:
            from v17_extensions import reference_label_diff
            oct_names=names_by_source.get('collection') or names_by_source.get('floorspread') or {}
            summary['reference_label_diff']=reference_label_diff(
                args.reference_manifest,args.reference_labels,args.reference_yaml,
                images,objects,oct_names,args.reference_class_map,out)
    (out/'20_internal_consistency_checks.json').write_text(
        json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'01_overview.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'13_taxonomy_mappings.json').write_text(json.dumps({c:{str(k):v for k,v in m.items()} for c,m in names_by_source.items()},ensure_ascii=False,indent=2),encoding='utf-8')
    all_sizes=[v for xs in size_by_context.values() for v in xs]
    plots=make_plots(out,class_rows,contexts,all_sizes)
    previews=make_preview_sheets(root,out,images,objects)
    orientation_records=make_orientation_comparison_sheets(root,out,images,objects)
    summary['orientation_review_sheets_generated']=sum(x['status']=='generated' for x in orientation_records)
    summary['orientation_review_sheets_failed']=sum(x['status']!='generated' for x in orientation_records)
    (out/'01_overview.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    warn_list=[f'{k}: {v}' for k,v in summary['issues_by_type'].items()]
    plot_markup=''.join(f'<figure><img src="figures/{html.escape(p)}" alt="{html.escape(p)}"></figure>' for p in plots)
    preview_markup=''.join(f'<figure><img src="figures/{html.escape(p)}" alt="{html.escape(p)}"></figure>' for p in previews)
    css='''<style>body{font:15px/1.6 Arial,sans-serif;color:#243040;max-width:1150px;margin:40px auto;padding:0 18px}h1,h2{color:#16462b}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0 26px}th,td{border:1px solid #dae2e0;padding:7px;text-align:left}th{background:#d7e9dd}tr:nth-child(even){background:#f5f8f5}.warn{background:#fff4db;border-left:5px solid #cc9500;padding:12px}figure{margin:20px 0}figure img{max-width:100%}code{background:#f4f4f4;padding:2px 5px}.cards{display:flex;gap:10px;flex-wrap:wrap}.cards div{padding:16px;background:#e7f0e9;min-width:140px;border-radius:7px}.cards strong{font-size:23px;display:block}</style>'''
    cards=[('Images',img_total),('Objects (boxes)',len(objects)),('Positive images',summary['positive_images']),('Empty annotation pairs',summary['explicitly_empty_images']),('Issue records',len(issues))]
    card_markup='<div class="cards">'+''.join(f'<div><strong>{v:,}</strong>{html.escape(k)}</div>' for k,v in cards)+'</div>'
    class_summary=[r for r in class_rows if r['context']=='all']
    html_report=f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>CCBRE October 2026 — Raw Data Audit</title>{css}</head><body>
    <h1>CCBRE | WP Detection | October 2026 raw dataset audit</h1>
    <p>Prepared {html.escape(summary['generated_at'])} | Auditor {VERSION} | Read-only, two independent S3 source folders.</p>
    {card_markup}<h2>1. Scope and evidence</h2><p>Path: <code>{html.escape(str(root))}</code></p>
    <p>Decoder: Pillow {html.escape(str(summary['runtime']['pillow_version']))}; JPEG codec {html.escape(str(summary['runtime']['jpeg_decoder_version']))}; dHash reduce(16) before LANCZOS (changes hashes from v1.4).</p><p>Two export products are analyzed separately: <strong>collection</strong> and <strong>floorspread</strong>.
    Detection object counts use only <code>labels-detect</code>; polygon files are analyzed separately for correspondence.</p>
    <p class="warn"><strong>Important:</strong> Collection context is inferred from the documented historical name suffix, not confirmed independently. An empty annotation file is distinguished from a missing file; zero annotated objects does not guarantee a visually clean image. Files with “synthetic” in their names are provenance candidates, NOT proven computer-generated.</p>
    <h2>2. Image populations and annotation coverage</h2>
    {report_table(contexts,['context','image_count','positive_images','explicit_empty_images','unlabelled_images','needs_review_images','detection_objects','polygon_objects'])}
    <h2>2a. Partner export reconciliation (report.md and .sync-state.json)</h2>
    {report_table(export_records,['context','report_generated','raw_images','report_images','sync_images','raw_empty','report_without_labels','raw_objects','report_objects','sync_objects','dropped_before_export','export_images_match','export_objects_match'])}
    <h2>2b. Collection filename-derived context (not independently verified)</h2>
    {report_table([dict(context=k[1], images=v) for k,v in sorted(context_inference.items()) if k[0]=='collection'], ['context','images'])}
    <h2>3. All 29 source classes: image support and object count</h2>{plot_markup}
    {report_table(class_summary,['class_id','class_name','instances','images_containing_class','instances_per_image','share_of_all_images_pct'])}
    <p>Detailed per-source values: 03_class_statistics.csv. Counts are descriptive; no T=400 or T=1000 grouping is applied.</p>
    <h2>4. Image resolutions (EXIF-display frame)</h2>{report_table([dict(context=c,width=w,height=h,encoded_width=ew,encoded_height=eh,images=n) for (c,w,h,ew,eh),n in sorted(resolutions.items())],['context','width','height','encoded_width','encoded_height','images'])}
    <h2>5. Object sizes</h2>{report_table(size_rows,['context','objects','median_area_pct','p10_area_pct','p90_area_pct','under_0_1pct','under_1pct'])}
    <h2>6. Data-quality findings</h2><p>{len(issues):,} records across {len(summary['issues_by_type'])} issue types.</p>
    {report_table([{'issue':k,'count':v} for k,v in sorted(summary['issues_by_type'].items(),key=lambda x:-x[1])],['issue','count'])}
    <p>Decoded pixel duplicate groups with different file hashes: {len(pixel_duplicate_rows):,}. Global dHash screening uses {dhash_metrics['unique_decoded_pixel_representatives']:,} unique representatives, reducing duplicates by {dhash_metrics['removed_redundant_image_rows']:,} images; <= {dhash_metrics['threshold_for_extended']} bits: {dhash_metrics['total']:,} representative-pair leads. Histogram 0..12 in 29_dhash_distance_distribution.csv. dHash links are review candidates, not proof of a shared acquisition. Strict filename/exact/pixel groups: {split_metrics['strict_group_count']:,}; extended dHash-linked groups: {split_metrics['extended_group_count']:,}; largest extended group: {split_metrics['extended_max_group_size']:,} images (potential chain effect).</p><p>Exact image duplicate SHA-256 groups: {len(dup_rows):,}; with differing box annotations: {summary['duplicate_groups_with_conflicting_labels']:,}; across both source contexts: {summary['cross_source_duplicate_hash_groups']:,}. Image perceptual similarity, label semantics and full acquisition independence are not established.</p>
    <h2>6a. Capture grouping and conflict tiers</h2>
    <p>Q-code semantics (tour/municipality/site) and b/t view letters require partner confirmation. DSC runs are grouped per EXIF camera make/model when available, and time-gap sessions are independently recorded using a 10-minute cutoff. Camera model does not uniquely identify a physical body; unreferenced or missing capture timestamps limit independence claims. The extended dHash-linked group is for review, NOT an automatic hard training split constraint.</p>
    {report_table(inference_size,['context','objects','median_short_px','p10_short_px','under_32px','under_16px'])}
    {report_table([{'tier':k,'pairs':v} for k,v in Counter(r['tier'] for r in duplicate_pair_details).items()],['tier','pairs'])}
    <p>For all positive images with nontrivial EXIF orientation, compare encoded-frame and display-frame label hypotheses via 30_orientation_review_index.csv. Neither is assumed correct without inspection. Primary shorter-side measurements assume display-frame labels; encoded-frame alternatives are also exported.</p>
    <p>EXIF-nontrivial image count: {summary['exif_orientation_nontrivial_count']:,}; comparison sheets generated: {summary['orientation_review_sheets_generated']:,}. Files are indexed in 30_orientation_review_index.csv. DSC number repeats: {summary['repeated_DSC_numbers']:,}; differing known camera identities: {summary['DSC_numbers_with_multiple_camera_identities']:,}. Capture groups remain provisional.</p>
    <h2>7. Box annotation overlay samples (display-frame convention, pending sign-off)</h2><p>Red boxes with numeric class IDs and names are drawn from labels-detect in the EXIF-displayed coordinate frame. Visually inspect the original images and the boxes before treating the export as validated. This sample is illustrative, not a formal stratified correctness audit.</p>{preview_markup}
    <h2>8. Provenance and taxonomy</h2><p>Per-source <code>data.yaml</code> mappings are stored verbatim in 13_taxonomy_mappings.json. Actual mappings are compared to partner-provided 29-class definitions; any mismatch is reported as a QC finding.</p>
    <p>Partner report.md records dropped source masks before export; they are not invalid objects in the downloaded boxes. New S3 snapshot may include legacy images; do not call all files “new October acquisitions” without matching against the frozen July source manifest.</p>
    <h2>9. Next evidence gates</h2><ol><li>Verify export completion and freeze dataset fingerprint.</li><li>Resolve critical issues and inspect a stratified sample of box overlays against actual images.</li><li>Determine collection Container/Wagon and origin/acquisition group metadata.</li><li>Compare SHA-256 and perceptual duplicates with July source images before any merge or external-test claim.</li><li>Use a clean unseen split for Stage-A and end-to-end generalization tests, followed by November building.</li></ol>
    <p>Evidence files are CSV and JSON beside this HTML. This is a raw dataset characterization, not an end-to-end model-performance report.</p></body></html>'''
    (out/'Dataset_Audit_Report.html').write_text(html_report,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('image_count','detection_object_count','positive_images','explicitly_empty_images','missing_or_unreadable_detection_label_images','duplicate_hash_groups','issues_by_type')},indent=2))
    print('Analysis saved to:',out)
    print('Structural status:',summary['validation_status'],
          '| Metadata reconciled:',summary['metadata_reconciliation_passed'],
          '| Internal checks:',summary['internal_consistency_passed'])
    if not summary['internal_consistency_passed']:
        print('ERROR: Internal consistency check FAILED; see 20_internal_consistency_checks.json',file=sys.stderr)
        raise SystemExit(2)
    print('NOTE: dataset counts do not prove label correctness or model performance.')

if __name__ == '__main__':
    main()
