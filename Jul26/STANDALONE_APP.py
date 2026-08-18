#!/usr/bin/env python3
"""CCBRE AI-LAB unified Stage-0 + A/B/C visual application.

The main window deliberately contains no permanent model/threshold control rows.
Advanced controls live in menus/dialogs so image and analysis results receive
nearly the entire screen.
"""
from __future__ import annotations
import argparse,json,os,sys,threading,traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog,messagebox,ttk
import yaml
from PIL import Image,ImageTk
from context_inference import ContextClassifier
from phase4_inference import Pipeline,norm
from phase4_rendering import render_unlabeled,render_with_gt
from combined_rendering import prepend_context_banner
from evaluate_TCoreOther_comprehensive import parse_ground_truth,same_class_match
ROOT=Path(__file__).resolve().parent

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def bundled(rel):return Path(getattr(sys,'_MEIPASS',ROOT))/rel if hasattr(sys,'_MEIPASS') else ROOT/rel
def default_config():
    if getattr(sys,'frozen',False):
        p=Path(sys.executable).resolve().parent/'config.json'
        if p.is_file():return p
    return ROOT/'config.json'
def category_counts(gt_count,verified,same_tp,exact_count):
    uncertain=max(0,verified-exact_count);fp=max(0,exact_count-same_tp)+uncertain;fn=max(0,gt_count-same_tp)
    if gt_count==0 and verified==0:return 'correct_negative',fp,fn,uncertain
    if fp==0 and fn==0:return 'perfect_detection',fp,fn,uncertain
    if fp>0 and fn==0:return 'false_positive_only',fp,fn,uncertain
    if fn>0 and fp==0:return 'false_negative_only',fp,fn,uncertain
    return 'mixed_error',fp,fn,uncertain
def find_yaml(start):
    cur=start.resolve()
    for _ in range(6):
        for n in ('dataset.yaml','data.yaml','dataset.yml','data.yml'):
            p=cur/n
            if p.is_file():return p
        if cur.parent==cur:break
        cur=cur.parent
    return None
def names_from_yaml(p):
    if not p:return None
    raw=yaml.safe_load(p.read_text(encoding='utf-8-sig')) or {};n=raw.get('names')
    if isinstance(n,list):return {i:str(v) for i,v in enumerate(n)}
    if isinstance(n,dict):return {int(k):str(v) for k,v in n.items()}
def sibling_labels(image):
    out=[image.with_suffix('.txt')];parts=list(image.parts)
    for i,part in enumerate(parts):
        if part.casefold()=='images':q=parts.copy();q[i]='labels';out.append(Path(*q).with_suffix('.txt'))
    return list(dict.fromkeys(out))
def context_gt_from_folder(image):
    parts=[p.casefold() for p in image.parts]
    for cls in ('container','wagon'):
        if cls in parts:return cls
    return None
def context_gt_from_suffix(image):
    s=image.stem.casefold()
    if s.endswith('w'):return 'wagon'
    if s.endswith(('c','b','t')):return 'container'
    return None

class App:
    def __init__(self,root,config_path):
        self.root=root;self.config_path=Path(config_path).resolve();self.c=load(self.config_path);self.T=int(self.c['threshold']);self.runs=fmt(self.c['runs_root'],self.T);self.final=self.runs/'FINAL_MODELS';self.manifest=load(self.final/'final_model_manifest.json')
        self.context_path=Path(self.manifest['models']['context']);self.a_path=Path(self.manifest['models']['stage_a']);self.b_path=Path(self.manifest['models']['stage_b']);self.c_path=Path(self.manifest['models']['stage_c']);self.op=tk.StringVar(value=self.c['default_operating_point']);self.show_analysis=tk.BooleanVar(value=bool(self.c['app']['show_analysis_panel_by_default']));self.show_full_render=tk.BooleanVar(value=bool(self.c['app']['show_saved_render_panel_in_viewer_by_default']));self.context_suffix_gt=tk.BooleanVar(value=bool(self.c['app']['context_gt_from_filename_default']));self.fullscreen=False;self.context_model=None;self.pipe=None;self.pipeline_key=None;self.image_path=None;self.original=None;self.display_image=None;self.full_render=None;self.photo=None;self.result=None
        root.title('CCBRE AI-LAB — Context + Foreign Material Analysis');root.geometry('1580x980');root.minsize(1000,700)
        self.make_menu();self.make_header();self.make_body();self.make_footer();self.bind_keys();self.status.set('Ready — select an image (Ctrl+O).')
    def make_menu(self):
        m=tk.Menu(self.root);self.root.config(menu=m)
        f=tk.Menu(m,tearoff=0);m.add_cascade(label='File',menu=f);f.add_command(label='Open image and analyze…',accelerator='Ctrl+O',command=self.open_image);f.add_command(label='Re-analyze current image',accelerator='Ctrl+R',command=self.start);f.add_command(label='Save rendered result as…',accelerator='Ctrl+S',command=self.save_as);f.add_command(label='Open output folder',command=self.open_output);f.add_separator();f.add_command(label='Exit',command=self.root.destroy)
        models=tk.Menu(m,tearoff=0);m.add_cascade(label='Models',menu=models);models.add_command(label='Use FINAL_MODELS (recommended)',command=self.reset_models);models.add_command(label='Use optional D1 YOLOv8m context reference',command=self.use_context_reference);custom=tk.Menu(models,tearoff=0);models.add_cascade(label='Custom checkpoints',menu=custom);custom.add_command(label='Stage 0 context classifier…',command=lambda:self.browse_model('context'));custom.add_command(label='Stage A Any-FM detector…',command=lambda:self.browse_model('a'));custom.add_command(label='Stage B binary verifier…',command=lambda:self.browse_model('b'));custom.add_command(label='Stage C exact classifier…',command=lambda:self.browse_model('c'));models.add_separator();models.add_command(label='Show selected model paths',command=self.show_models);models.add_command(label='Validate selected models',command=self.validate_models)
        opt=tk.Menu(m,tearoff=0);m.add_cascade(label='Options',menu=opt);ops=tk.Menu(opt,tearoff=0);opt.add_cascade(label='Operating point',menu=ops)
        for name in self.c['operating_points']:ops.add_radiobutton(label=name,variable=self.op,value=name,command=self.invalidate)
        opt.add_checkbutton(label='Use filename suffix as context GT (project audit only)',variable=self.context_suffix_gt)
        view=tk.Menu(m,tearoff=0);m.add_cascade(label='View',menu=view);view.add_checkbutton(label='Show analysis panel',accelerator='F9',variable=self.show_analysis,command=self.toggle_analysis);view.add_checkbutton(label='Show saved-render information panel in viewer',variable=self.show_full_render,command=self.refresh_image);view.add_command(label='Fullscreen',accelerator='F11',command=self.toggle_fullscreen)
        tools=tk.Menu(m,tearoff=0);m.add_cascade(label='Tools',menu=tools);tools.add_command(label='Validate selected models',command=self.validate_models);tools.add_command(label='Open output folder',command=self.open_output)
        h=tk.Menu(m,tearoff=0);m.add_cascade(label='Help',menu=h);h.add_command(label='About',command=lambda:messagebox.showinfo('About','CCBRE AI-LAB\nStage 0: Container/Wagon context\nStages A/B/C: foreign-material detection/verification/classification'))
    def make_header(self):
        h=ttk.Frame(self.root,padding=(8,4));h.pack(fill='x');logo=bundled(Path('assets')/'CCBRE_AI_LAB_Logo.png')
        if logo.is_file():
            im=Image.open(logo).convert('RGBA');im.thumbnail((62,48),Image.Resampling.LANCZOS);self.logo=ImageTk.PhotoImage(im);ttk.Label(h,image=self.logo).pack(side='left',padx=(0,8))
        text=ttk.Frame(h);text.pack(side='left',fill='x',expand=True);ttk.Label(text,text='AI-LAB CCBRE.EU — Context + Foreign Material Analysis',font=('Segoe UI',14,'bold')).pack(anchor='w');self.headline=tk.StringVar(value='Stage 0: context | Stages A/B/C: foreign material');ttk.Label(text,textvariable=self.headline,font=('Segoe UI',9)).pack(anchor='w')
    def make_body(self):
        self.pane=ttk.Panedwindow(self.root,orient='horizontal');self.pane.pack(fill='both',expand=True,padx=4,pady=2);self.image_frame=ttk.Frame(self.pane);self.analysis_frame=ttk.Frame(self.pane);self.pane.add(self.image_frame,weight=4);self.pane.add(self.analysis_frame,weight=2)
        self.canvas=tk.Canvas(self.image_frame,bg='#171717',highlightthickness=0);self.canvas.pack(fill='both',expand=True);self.canvas.bind('<Configure>',lambda e:self.refresh_image())
        self.text=tk.Text(self.analysis_frame,wrap='word',font=('Consolas',10),borderwidth=0);self.text.pack(fill='both',expand=True);self.text.configure(state='disabled')
        if not self.show_analysis.get():self.pane.forget(self.analysis_frame)
    def make_footer(self):
        self.status=tk.StringVar();ttk.Label(self.root,textvariable=self.status,anchor='w',padding=(6,3)).pack(fill='x')
    def bind_keys(self):
        self.root.bind('<Control-o>',lambda e:self.open_image());self.root.bind('<Control-r>',lambda e:self.start());self.root.bind('<Control-s>',lambda e:self.save_as());self.root.bind('<F9>',lambda e:(self.show_analysis.set(not self.show_analysis.get()),self.toggle_analysis()));self.root.bind('<F11>',lambda e:self.toggle_fullscreen())
    def toggle_fullscreen(self):self.fullscreen=not self.fullscreen;self.root.attributes('-fullscreen',self.fullscreen)
    def toggle_analysis(self):
        shown=str(self.analysis_frame) in self.pane.panes()
        if self.show_analysis.get() and not shown:self.pane.add(self.analysis_frame,weight=2)
        elif not self.show_analysis.get() and shown:self.pane.forget(self.analysis_frame)
    def reset_models(self):
        self.context_path=Path(self.manifest['models']['context']);self.a_path=Path(self.manifest['models']['stage_a']);self.b_path=Path(self.manifest['models']['stage_b']);self.c_path=Path(self.manifest['models']['stage_c']);self.invalidate();self.status.set('Reset to FINAL_MODELS.')
    def use_context_reference(self):
        ref=self.manifest.get('optional_context_reference_yolov8m')
        if not ref or not Path(ref).is_file():
            messagebox.showwarning('D1 reference unavailable','No optional YOLOv8m context reference is present in FINAL_MODELS.')
            return
        self.context_path=Path(ref);self.invalidate();self.status.set('Using optional D1 YOLOv8m context reference.')
    def browse_model(self,key):
        p=filedialog.askopenfilename(title='Select checkpoint',filetypes=[('PyTorch checkpoint','*.pt'),('All files','*.*')]);
        if not p:return
        if key=='context':self.context_path=Path(p)
        elif key=='a':self.a_path=Path(p)
        elif key=='b':self.b_path=Path(p)
        else:self.c_path=Path(p)
        self.invalidate();self.status.set(f'Custom {key} model selected.')
    def show_models(self):messagebox.showinfo('Selected models',f'Stage 0:\n{self.context_path}\n\nStage A:\n{self.a_path}\n\nStage B:\n{self.b_path}\n\nStage C:\n{self.c_path}\n\nOperating point: {self.op.get()}')
    def validate_models(self,quiet=False):
        missing=[str(p) for p in (self.context_path,self.a_path,self.b_path,self.c_path) if not Path(p).is_file()]
        if missing:
            if not quiet:messagebox.showerror('Missing models','\n'.join(missing))
            raise FileNotFoundError('\n'.join(missing))
        if not quiet:messagebox.showinfo('Models valid','All four selected checkpoints exist.')
    def invalidate(self):self.context_model=None;self.pipe=None;self.pipeline_key=None
    def load_models(self):
        self.validate_models(True);key=(str(self.context_path),str(self.a_path),str(self.b_path),str(self.c_path),self.op.get())
        if self.pipeline_key==key:return
        self.status.set('Loading selected models…');self.root.update_idletasks();self.context_model=ContextClassifier(self.context_path,self.c['device'],self.c['context']['app']['imgsz']);self.pipe=Pipeline(self.a_path,self.b_path,self.c_path,self.c['operating_points'][self.op.get()],self.c['device']);self.pipeline_key=key
    def open_image(self):
        p=filedialog.askopenfilename(title='Select image',filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp'),('All files','*.*')]);
        if not p:return
        self.image_path=Path(p);self.original=Image.open(self.image_path).convert('RGB');self.display_image=self.original;self.full_render=None;self.result=None;self.headline.set(self.image_path.name);self.status.set('Image selected — analysis starts automatically.');self.set_text('Image selected.\n\nAnalysis is starting automatically. Advanced settings are under the menus.');self.refresh_image();self.start()
    def start(self):
        if not self.image_path:return
        if getattr(self,'busy',False):return
        self.busy=True;self.status.set('Analyzing Stage 0 + A/B/C…');threading.Thread(target=self.worker,daemon=True).start()
    def worker(self):
        try:
            self.load_models();ctx=self.context_model.analyze(self.image_path);image,fm=self.pipe.analyze(self.image_path);gt=None;names=None;label=None
            for cand in sibling_labels(self.image_path):
                if cand.is_file():label=cand;names=names_from_yaml(find_yaml(cand.parent)) or {0:'BAW-Sack',1:'Papier',2:'Plastiksack',3:'Other'};gt=parse_ground_truth(cand,image.size,names);break
            matches=[];exact=[];category=None;fp=fn=uncertain=0
            if gt is not None:
                target_id={norm(n):i for i,n in names.items()}
                for d in fm['detections']:
                    if d['class_name'] is None:continue
                    mapped=target_id.get(norm(d['class_name']))
                    if mapped is not None:exact.append({'class_id':int(mapped),'bbox':d['bbox'],'confidence':d['detector_confidence']*d['verifier_fo_probability']*d['exact_confidence']})
                matches,_,_=same_class_match(gt,exact);category,fp,fn,uncertain=category_counts(len(gt),len(fm['detections']),len(matches),len(exact));render=render_with_gt(image,fm,gt,exact,matches,names,category,fp,fn)
            else:render=render_unlabeled(image,fm);uncertain=sum(d['class_name'] is None for d in fm['detections'])
            context_gt=context_gt_from_folder(self.image_path)
            if context_gt is None and self.context_suffix_gt.get():context_gt=context_gt_from_suffix(self.image_path)
            full=prepend_context_banner(render,ctx,context_gt);image_only=full.crop((0,full.height-image.height,full.width,full.height))
            outroot=fmt(self.c['app']['output_root'],self.T);sub='GT_COMPARISON/'+category if category else 'UNLABELED_INFERENCE';folder=outroot/sub;folder.mkdir(parents=True,exist_ok=True);stamp=datetime.now().strftime('%Y%m%d_%H%M%S');jpg=folder/f'{self.image_path.stem}__context_hybrid_{stamp}.jpg';js=jpg.with_suffix('.json');full.save(jpg,quality=95)
            record={'image':str(self.image_path),'context':ctx,'context_ground_truth':context_gt,'foreign_material':fm,'foreign_gt_label':str(label) if label else None,'diagnostic_category':category,'tp':len(matches),'fp':fp,'fn':fn,'class_uncertain_verified_objects':uncertain,'models_used':{'context':str(self.context_path),'stage_a':str(self.a_path),'stage_b':str(self.b_path),'stage_c':str(self.c_path),'operating_point':self.op.get()},'rendered_image':str(jpg)};js.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8');self.root.after(0,self.done,record,full,image_only,jpg,js)
        except Exception:self.root.after(0,self.failed,traceback.format_exc())
    def done(self,r,full,image_only,jpg,js):
        self.result=r;self.full_render=full;self.image_only=image_only;self.display_image=full if self.show_full_render.get() else image_only;self.busy=False;ctx=r['context'];fm=r['foreign_material'];fo='FOREIGN MATERIAL DETECTED' if fm['foreign_material_present'] else 'NO FOREIGN MATERIAL DETECTED';cat=f" | {r['diagnostic_category']}" if r['diagnostic_category'] else '';self.headline.set(f"Context: {ctx['class_name'].upper()} ({ctx['confidence']:.1%}) | {fo} | verified={len(fm['detections'])}{cat}");lines=[fo,f"Context: {ctx['class_name']} | confidence={ctx['confidence']:.4f}",f"Context GT: {r['context_ground_truth'] or 'not available'}",'',f"Mode: {'GT comparison' if r['foreign_gt_label'] else 'unlabeled inference'}",f"Category: {r['diagnostic_category'] or 'n/a'}",f"TP / FP / FN: {r['tp']} / {r['fp']} / {r['fn']}",f"Verified FO objects: {len(fm['detections'])}",f"Class-uncertain: {r['class_uncertain_verified_objects']}",'']
        for i,d in enumerate(fm['detections'],1):lines += [f"Object {i}: {d['class_name'] or 'ForeignMaterial (class uncertain)'}",f"  Det={d['detector_confidence']:.3f}  Ver={d['verifier_fo_probability']:.3f}  Cls={d['exact_confidence']:.3f}"]
        lines += ['', 'Models:',f"  Stage 0: {r['models_used']['context']}",f"  Stage A: {r['models_used']['stage_a']}",f"  Stage B: {r['models_used']['stage_b']}",f"  Stage C: {r['models_used']['stage_c']}",f"  OP: {r['models_used']['operating_point']}",'','Saved:',str(jpg),str(js)];self.set_text('\n'.join(lines));self.status.set('Analysis complete.');self.refresh_image()
    def failed(self,e):self.busy=False;self.status.set('Analysis failed.');self.set_text(e);messagebox.showerror('Analysis failed',e)
    def refresh_image(self):
        if self.result is not None and self.full_render is not None:self.display_image=self.full_render if self.show_full_render.get() else self.image_only
        if self.display_image is None:return
        w=max(100,self.canvas.winfo_width()-12);h=max(100,self.canvas.winfo_height()-12);im=self.display_image.copy();im.thumbnail((w,h),Image.Resampling.LANCZOS);self.photo=ImageTk.PhotoImage(im);self.canvas.delete('all');self.canvas.create_image(self.canvas.winfo_width()//2,self.canvas.winfo_height()//2,image=self.photo,anchor='center')
    def set_text(self,s):self.text.config(state='normal');self.text.delete('1.0','end');self.text.insert('1.0',s);self.text.config(state='disabled')
    def save_as(self):
        if self.full_render is None:return
        p=filedialog.asksaveasfilename(defaultextension='.jpg',filetypes=[('JPEG','*.jpg'),('PNG','*.png')]);
        if p:self.full_render.save(p,quality=95);self.status.set(f'Saved: {p}')
    def open_output(self):
        p=fmt(self.c['app']['output_root'],self.T);p.mkdir(parents=True,exist_ok=True)
        try:os.startfile(p)
        except Exception as exc:messagebox.showerror('Open output folder',str(exc))
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=default_config());a=p.parse_args();root=tk.Tk();ico=bundled(Path('assets')/'CCBRE_AI_LAB_Logo.ico')
    if ico.is_file():
        try:root.iconbitmap(default=str(ico))
        except Exception:pass
    App(root,a.config);root.mainloop();return 0
if __name__=='__main__':raise SystemExit(main())
