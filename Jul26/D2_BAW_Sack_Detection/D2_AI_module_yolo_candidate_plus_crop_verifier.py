
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw
import csv

# ============================================================
# D2 AI MODULE — YOLO CANDIDATE + CROP VERIFIER
# ============================================================
#
# image -> YOLO candidate detector -> crop -> classifier verifier -> final detections
# The cross/X-pattern information is learned by the verifier classifier.
# ============================================================

DATASET_DIR = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Focused_Datasets_With_Limited_Negatives_Optimized_80_10_10\D2_BAW_Sack_binary_neg1to1")
SPLIT = "test"

# Paste paths before running.
DETECTOR_MODEL = Path(r"PASTE_D2_DETECTOR_BEST_PT_HERE")
VERIFIER_MODEL = Path(r"PASTE_D2_CROP_VERIFIER_BEST_PT_HERE")

OUTPUT_ROOT = Path(r"C:\Hadi\Foreign_Material\Runs\D2_BAW_Sack_AI_Module\D2_final_candidate_plus_verifier_eval")

CLASS_ID = 0
IMG_SIZE_DET = 1536
IMG_SIZE_CLS = 384
DEVICE = 0
DETECTOR_CONF = 0.03
DETECTOR_IOU_NMS = 0.70
MATCH_IOU = 0.10
VERIFIER_THRESHOLDS = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]
IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}


def norm_model_path(p):
    p = Path(p)
    if p.is_dir():
        p = p / "best.pt"
    return p


def images():
    d = DATASET_DIR / "images" / SPLIT
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_gt(stem):
    p = DATASET_DIR / "labels" / SPLIT / f"{stem}.txt"
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0])); x,y,w,h = map(float, parts[1:5])
        except ValueError:
            continue
        if cls == CLASS_ID:
            out.append([x-w/2, y-h/2, x+w/2, y+h/2])
    return out


def area(b):
    return max(0,b[2]-b[0]) * max(0,b[3]-b[1])


def iou(a,b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1) * max(0,y2-y1)
    union = area(a) + area(b) - inter
    return inter/union if union > 0 else 0


def crop_from_norm(img, box):
    W,H = img.size
    x1,y1,x2,y2 = box
    px = (
        max(0,min(W-1,int(round(x1*W)))),
        max(0,min(H-1,int(round(y1*H)))),
        max(1,min(W,int(round(x2*W)))),
        max(1,min(H,int(round(y2*H)))),
    )
    crop = img.crop(px).resize((IMG_SIZE_CLS, IMG_SIZE_CLS))
    return crop, px


def baw_score(cls_model, crop):
    r = cls_model.predict(source=crop, imgsz=IMG_SIZE_CLS, device=DEVICE, verbose=False)[0]
    names = r.names
    baw_idx = None
    for idx, name in names.items():
        if str(name).lower().replace("-","_") == "baw_sack":
            baw_idx = int(idx)
            break
    if baw_idx is None:
        baw_idx = 0
    return float(r.probs.data.cpu().numpy()[baw_idx])


def run_candidates(det_model, cls_model, img_list):
    pred_by_img, candidate_rows = {}, []
    for ip in img_list:
        with Image.open(ip) as im:
            im = im.convert("RGB")
            det = det_model.predict(
                source=str(ip), imgsz=IMG_SIZE_DET,
                conf=DETECTOR_CONF, iou=DETECTOR_IOU_NMS,
                device=DEVICE, verbose=False
            )[0]
            cands = []
            if det.boxes is not None and len(det.boxes):
                boxes = det.boxes.xyxyn.cpu().numpy()
                confs = det.boxes.conf.cpu().numpy()
                clss = det.boxes.cls.cpu().numpy()
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                    if int(cls) != CLASS_ID:
                        continue
                    b = [float(v) for v in box.tolist()]
                    crop, _ = crop_from_norm(im, b)
                    score = baw_score(cls_model, crop)
                    cands.append({"box": b, "det_conf": float(conf), "ver_score": score})
                    candidate_rows.append([str(ip), ip.stem, i, float(conf), score, *b])
            pred_by_img[ip.stem] = cands
    return pred_by_img, candidate_rows


def eval_thr(img_list, pred_by_img, thr):
    tp=fp=fn=0
    img_tp=img_fp=img_tn=img_fn=0
    details=[]
    for ip in img_list:
        gt = read_gt(ip.stem)
        preds = [p for p in pred_by_img.get(ip.stem, []) if p["ver_score"] >= thr]
        matched_g, matched_p = set(), set()
        matches = []
        for pi,p in enumerate(preds):
            for gi,g in enumerate(gt):
                v = iou(p["box"], g)
                if v >= MATCH_IOU:
                    matches.append((v,pi,gi))
        matches.sort(reverse=True)
        for v,pi,gi in matches:
            if pi in matched_p or gi in matched_g:
                continue
            matched_p.add(pi); matched_g.add(gi)
        itp = len(matched_g)
        ifp = len(preds) - len(matched_p)
        ifn = len(gt) - len(matched_g)
        tp += itp; fp += ifp; fn += ifn
        has_gt, has_pred = len(gt)>0, len(preds)>0
        if has_gt and has_pred: img_tp += 1
        elif (not has_gt) and has_pred: img_fp += 1
        elif (not has_gt) and (not has_pred): img_tn += 1
        else: img_fn += 1
        details.append([str(ip), ip.stem, len(gt), len(preds), itp, ifp, ifn, int(has_gt), int(has_pred)])
    prec = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    iprec = img_tp/(img_tp+img_fp) if img_tp+img_fp else 0
    irec = img_tp/(img_tp+img_fn) if img_tp+img_fn else 0
    if1 = 2*iprec*irec/(iprec+irec) if iprec+irec else 0
    return {"thr":thr,"tp":tp,"fp":fp,"fn":fn,"p":prec,"r":rec,"f1":f1,
            "itp":img_tp,"ifp":img_fp,"itn":img_tn,"ifn":img_fn,"ip":iprec,"ir":irec,"if1":if1,"details":details}


def render_best(img_list, pred_by_img, thr, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ip in img_list:
        with Image.open(ip) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            W,H = im.size
            for p in pred_by_img.get(ip.stem, []):
                if p["ver_score"] < thr:
                    continue
                x1,y1,x2,y2 = p["box"]
                px = [int(x1*W), int(y1*H), int(x2*W), int(y2*H)]
                draw.rectangle(px, outline=(255,0,0), width=4)
                draw.text((px[0], max(0,px[1]-15)), f"BAW det={p['det_conf']:.2f} ver={p['ver_score']:.2f}", fill=(255,0,0))
            im.save(out_dir / ip.name, quality=95)


def main():
    det_path = norm_model_path(DETECTOR_MODEL)
    ver_path = norm_model_path(VERIFIER_MODEL)
    if "PASTE_" in str(det_path):
        raise RuntimeError("Paste DETECTOR_MODEL best.pt path first.")
    if "PASTE_" in str(ver_path):
        raise RuntimeError("Paste VERIFIER_MODEL best.pt path first.")
    if not det_path.exists():
        raise FileNotFoundError(det_path)
    if not ver_path.exists():
        raise FileNotFoundError(ver_path)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    det_model = YOLO(str(det_path))
    cls_model = YOLO(str(ver_path))
    img_list = images()

    print("="*100)
    print("D2 AI module: YOLO candidate + crop verifier")
    print(f"Images: {len(img_list)}")
    print("="*100)

    pred_by_img, candidate_rows = run_candidates(det_model, cls_model, img_list)

    with (OUTPUT_ROOT / "D2_module_candidates_with_verifier_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image","stem","candidate_index","detector_conf","verifier_score","x1","y1","x2","y2"])
        w.writerows(candidate_rows)

    rows = []; best = None
    for thr in VERIFIER_THRESHOLDS:
        res = eval_thr(img_list, pred_by_img, thr)
        rows.append([thr,res["tp"],res["fp"],res["fn"],res["p"],res["r"],res["f1"],res["itp"],res["ifp"],res["itn"],res["ifn"],res["ip"],res["ir"],res["if1"]])
        if best is None or res["f1"] > best["f1"]:
            best = res

    with (OUTPUT_ROOT / "D2_module_threshold_sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["verifier_threshold","object_tp","object_fp","object_fn","object_precision","object_recall","object_f1","image_tp","image_fp","image_tn","image_fn","image_precision","image_recall","image_f1"])
        w.writerows(rows)

    with (OUTPUT_ROOT / f"D2_module_details_best_threshold_{best['thr']:.3f}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image","stem","gt_count","final_pred_count","object_tp","object_fp","object_fn","image_has_gt","image_has_pred"])
        w.writerows(best["details"])

    render_dir = OUTPUT_ROOT / f"rendered_final_predictions_threshold_{best['thr']:.3f}"
    render_best(img_list, pred_by_img, best["thr"], render_dir)

    print("="*100)
    print("D2 AI MODULE RESULT")
    print(f"Best verifier threshold: {best['thr']:.3f}")
    print(f"Object Precision={best['p']:.4f}, Recall={best['r']:.4f}, F1={best['f1']:.4f}")
    print(f"Image  Precision={best['ip']:.4f}, Recall={best['ir']:.4f}, F1={best['if1']:.4f}")
    print(f"Output: {OUTPUT_ROOT}")
    print("="*100)


if __name__ == "__main__":
    main()
