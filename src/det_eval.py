"""Detection evaluation helpers built on pycocotools."""
import contextlib
import copy
import io

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from src.data.common import CLASSES
from src.data.datasets import detection_collate

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def load_gt(gt_json, agnostic=False) -> COCO:
    """Load ground truth. agnostic=True maps every box to one 'animal' category (id 1)."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
    if not agnostic:
        return gt
    d = copy.deepcopy(gt.dataset)
    for a in d["annotations"]:
        a["category_id"] = 1
    d["categories"] = [{"id": 1, "name": "animal", "supercategory": "animal"}]
    agn = COCO()
    agn.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        agn.createIndex()
    return agn


@torch.no_grad()
def run_detector(model, dataset, device, batch_size=4, num_workers=4, score_thr=0.05, amp=True):
    """Run a torchvision detector over a DetectionDataset; return COCO-format results (xywh boxes)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=detection_collate)
    results = []
    for imgs, _, ids in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            outs = model([im.to(device) for im in imgs])
        for out, img_id in zip(outs, ids):
            for (x1, y1, x2, y2), s, c in zip(out["boxes"].float().tolist(), out["scores"].float().tolist(),
                                              out["labels"].tolist()):
                if s >= score_thr:
                    results.append({"image_id": int(img_id), "category_id": int(c),
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return results


def coco_eval(gt: COCO, results, verbose=False):
    """Returns (stats dict, COCOeval object or None)."""
    if not results:
        return {k: 0.0 for k in STAT_NAMES}, None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dt = gt.loadRes(results)
        E = COCOeval(gt, dt, "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    if verbose:
        print(buf.getvalue())
    return dict(zip(STAT_NAMES, (float(v) for v in E.stats))), E


def per_class_ap50(E: COCOeval) -> dict:
    """AP at IoU 0.5 for each category (area=all, maxDets=100)."""
    prec = E.eval["precision"][0, :, :, 0, 2]  # [recall, category]
    out = {}
    for k, cat_id in enumerate(E.params.catIds):
        p = prec[:, k]
        p = p[p > -1]
        name = CLASSES[cat_id - 1] if len(E.params.catIds) == len(CLASSES) else "animal"
        out[f"AP50_{name}"] = float(p.mean()) if p.size else 0.0
    return out


def _iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def multi_animal_stats(gt: COCO, results, count_thr=0.5, iou_thr=0.5) -> dict:
    """E4: on images with >= 2 animals (and >= 2 species), class-agnostic recall at IoU 0.5
    and count accuracy (number of predictions with score > count_thr == number of animals)."""
    preds = {}
    for r in results:
        preds.setdefault(r["image_id"], []).append(r)
    out = {}
    for name, cond in [("multi", lambda anns: len(anns) >= 2),
                       ("multispecies", lambda anns: len({a["category_id"] for a in anns}) >= 2)]:
        n_img = n_gt = n_hit = n_count_ok = 0
        for img_id in gt.getImgIds():
            anns = gt.loadAnns(gt.getAnnIds(imgIds=img_id))
            if not cond(anns):
                continue
            n_img += 1
            n_gt += len(anns)
            dets = sorted(preds.get(img_id, []), key=lambda r: -r["score"])
            used = set()
            for a in anns:  # greedy match each GT box to its best unused detection
                best, best_j = 0.0, None
                for j, d in enumerate(dets):
                    if j not in used:
                        iou = _iou(a["bbox"], d["bbox"])
                        if iou > best:
                            best, best_j = iou, j
                if best >= iou_thr:
                    used.add(best_j)
                    n_hit += 1
            n_count_ok += int(sum(d["score"] > count_thr for d in dets) == len(anns))
        if n_img:
            out[f"{name}_images"] = n_img
            out[f"{name}_recall50"] = n_hit / n_gt
            out[f"{name}_count_accuracy"] = n_count_ok / n_img
    return out
