"""Tiny from-scratch synthetic smoke test. Never substitutes for the real 9,628-image run."""
from pathlib import Path
import tempfile, subprocess, sys, json, csv
from PIL import Image
from legacy_auditor.analyze_oct26 import EXPECTED

base=Path(__file__).parent
with tempfile.TemporaryDirectory(prefix='ccbre_v22_test_') as td:
    root=Path(td);jul=root/'Jul-26';octo=root/'Oct-26'/'Raw'; out=root/'DualV22'
    (jul/'images').mkdir(parents=True);(jul/'labels').mkdir()
    # historical July context+material IDs: Context and plastic bag
    (jul/'dataset.yaml').write_text('names: [Container, Wagon, Plastiksack, Diverses]\n',encoding='utf8')
    def jpg(p,color):
        p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(180,130),color).save(p,quality=95)
    jpg(jul/'images'/'Q1000-1c.jpg',(80,140,70))
    jpg(jul/'images'/'Q1000-2w.jpg',(100,120,80))
    (jul/'labels'/'Q1000-1c.txt').write_text('0 .5 .5 .9 .9\n2 .55 .55 .2 .2\n',encoding='utf8')
    (jul/'labels'/'Q1000-2w.txt').write_text('1 .5 .5 .9 .9\n',encoding='utf8')
    for bucket in ['biowaste-collection-images','biowaste-floorspread-images']:
        p=octo/bucket
        for sub in ['images','labels','labels-detect']:(p/sub).mkdir(parents=True)
        (p/'data.yaml').write_text('names:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(EXPECTED)),encoding='utf8')
        (p/'report.md').write_text('## example synthetic test export\n',encoding='utf8')
        (p/'.sync-state.json').write_text(json.dumps({'classes':EXPECTED,'images':{},'objects':{},'annotations':{}}),encoding='utf8')
    c=octo/'biowaste-collection-images';f=octo/'biowaste-floorspread-images'
    def writeoct(bucket,name,photo,rows):
        (bucket/'images'/name).write_bytes(photo.read_bytes())
        (bucket/'labels-detect'/Path(name).with_suffix('.txt')).write_text(''.join(rows),encoding='utf8')
        (bucket/'labels'/Path(name).with_suffix('.txt')).write_text(''.join('0 .4 .4 .6 .4 .6 .6 .4 .6\n' for _ in rows),encoding='utf8')
    # duplicate same physical photo with class-consistent object count difference
    writeoct(c,'simpla-gruen-source__Q1000-1c.jpg',jul/'images'/'Q1000-1c.jpg',['0 .55 .55 .2 .2\n'])
    writeoct(c,'simpla-gruen-source-2026__Q1000-1c.jpg',jul/'images'/'Q1000-1c.jpg',[])
    writeoct(c,'simpla-gruen-source__Q1000-2w.jpg',jul/'images'/'Q1000-2w.jpg',[])
    jpg(c/'images'/'simpla-gruen-source-2026__Q2000-1b.jpg',(25,40,70))
    (c/'labels-detect'/'simpla-gruen-source-2026__Q2000-1b.txt').write_text('')
    (c/'labels'/'simpla-gruen-source-2026__Q2000-1b.txt').write_text('')
    jpg(f/'images'/'repla-synthetic-images__DSC03316.jpg',(240,220,100))
    (f/'labels-detect'/'repla-synthetic-images__DSC03316.txt').write_text('28 .6 .6 .3 .3\n')
    (f/'labels'/'repla-synthetic-images__DSC03316.txt').write_text('28 .45 .45 .75 .45 .75 .75 .45 .75\n')
    cmd=[sys.executable,str(base/'run_from_scratch.py'),'--july-images',str(jul/'images'),
         '--july-labels',str(jul/'labels'),'--july-yaml',str(jul/'dataset.yaml'),
         '--october-root',str(octo),'--output',str(out),'--samples-per-problem','2']
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=240)
    if p.returncode:
        print('STDOUT:',p.stdout[-8000:]);print('STDERR:',p.stderr[-8000:]); raise AssertionError(f'v2.2 complete test failed with code {p.returncode}')
    assert (out/'OPEN_FIRST.html').exists()
    assert (out/'OPEN_FIRST.html').read_text(encoding='utf-8').count('class="case-button"') == 9
    assert (out/'ALL_CASES_MANIFEST.json').exists()
    assert (out/'ALL_CASES_01.html').exists()
    assert (out/'ALL_CASES_04.html').exists()
    assert (out/'COMPLETE_DATASET_QUALITY_REPORT.html').exists()
    assert (out/'GENERAL_STATISTICS.html').exists()
    assert (out/'GENERAL_STATISTICS.html').read_text(encoding='utf-8').count('<img') >= 3
    assert (out/'CORRECTIVE_ACTIONS.html').exists()
    assert (out/'00_General_Dataset_Analysis'/'General_Statistics'/'19_Box_Area_Log_Bin_Fractions.csv').exists()
    assert (out/'01_Duplicate_Annotation_Problems'/'Review_List.csv').exists()
    assert (out/'02_Missing_July_Annotations'/'Review_List.csv').exists()
    assert (out/'99_Technical_Verification'/'Full_Dataset_Audit_v171'/'39_reference_label_diff_summary.json').exists()
    assert (out/'09_Provider_Corrective_Actions'/'Corrective_Action_Register.csv').exists()
    assert len(list((out/'01_Duplicate_Annotation_Problems'/'Examples').glob('*.jpg')))==1
    print('PASS v2.2 end-to-end synthetic: original July+October, duplicate and missing-label photos, class tables, readable problem folders')
