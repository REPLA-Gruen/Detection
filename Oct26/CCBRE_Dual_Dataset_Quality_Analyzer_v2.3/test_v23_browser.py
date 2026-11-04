"""Exercise the offline all-cases HTML generator and protect against unsafe filenames."""
import csv
from pathlib import Path
import tempfile
from all_cases_browser import write_case_browser

with tempfile.TemporaryDirectory(prefix='ccbre_case_browser_') as temp:
    root=Path(temp)
    # Make a minimal results root; linked chart/photo paths are optional.
    spec={'i':1,'name':'Test duplicate views','desc':'Two candidate photographs','caution':'Human decision required.',
          'columns':['case_id','file_1','type']}
    evil='Q0001-1c.jpg </script><script>evil()</script>'
    records=[{'case_id':'DUP-001','file_1':evil,'type':'class_change','photo_evidence':''},
             {'case_id':'DUP-002','file_1':'Q0001-2w.jpg','type':'positive_empty','photo_evidence':''}]
    page=write_case_browser(root,spec,'All different label cases',records)
    assert page.is_file()
    s=page.read_text(encoding='utf-8')
    assert "const DATA=" in s and "Loading cases" in s
    assert '</script><script>evil()' not in s, 'Raw filename escaped neither as JSON nor HTML'
    assert '\\u003c/script\\u003e' in s
    csvfile=root/'ALL_CASES_01.csv'
    assert csvfile.is_file()
    with csvfile.open(encoding='utf-8-sig',newline='') as f:
        cases=list(csv.DictReader(f))
    assert len(cases)==2 and cases[0]['file_1']==evil
print('PASS v2.3 offline all-case browser: safe JSON, searchable records and complete CSV')
