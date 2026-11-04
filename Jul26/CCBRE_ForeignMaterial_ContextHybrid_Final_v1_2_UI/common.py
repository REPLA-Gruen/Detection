from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
def load_config(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS)
def transfer(source,target,mode="hardlink"):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():target.unlink()
    if mode=="hardlink":
        try:target.hardlink_to(source);return
        except OSError:pass
    shutil.copy2(source,target)
def md5(path,block=1024*1024):
    digest=hashlib.md5()
    with Path(path).open("rb") as stream:
        while True:
            chunk=stream.read(block)
            if not chunk:break
            digest.update(chunk)
    return digest.hexdigest()
