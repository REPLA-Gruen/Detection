#!/usr/bin/env python3
from __future__ import annotations
from PIL import Image,ImageDraw
from phase4_rendering import get_font

def prepend_context_banner(rendered:Image.Image,context:dict,context_gt:str|None=None)->Image.Image:
    width=rendered.width;font=get_font(max(15,round(width/90)),bold=True);sub=get_font(max(13,round(width/105)))
    h=max(44,round(width*0.035));canvas=Image.new('RGB',(width,h+rendered.height),(20,20,20));canvas.paste(rendered,(0,h));d=ImageDraw.Draw(canvas)
    pred=f"Context: {str(context.get('class_name','unknown')).upper()} | confidence={float(context.get('confidence',0.0)):.3f}"
    if context_gt:pred+=f" | GT={context_gt.upper()} | {'MATCH' if context_gt.casefold()==str(context.get('class_name','')).casefold() else 'MISMATCH'}"
    d.text((10,5),pred,font=font,fill=(245,245,245));box=d.textbbox((10,5),pred,font=font);d.text((10,box[3]+2),'Stage 0 = acquisition context; Stages A/B/C = foreign-material analysis.',font=sub,fill=(190,190,190));return canvas
