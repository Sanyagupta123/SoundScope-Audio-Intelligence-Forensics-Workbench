"""Geometry-grounded scene analysis helpers."""
from __future__ import annotations
import io
import math
from typing import Any
import networkx as nx
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

def load_image(data: bytes):
    if not data:
        raise ValueError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                raise ValueError("Upload a JPG, PNG, or WEBP image.")
            if probe.width > 8192 or probe.height > 8192 or probe.width * probe.height > 30_000_000:
                raise ValueError("Image dimensions are too large (maximum 8192 px per side and 30 megapixels).")
            probe.verify()
        image = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

def iou(a, b):
    x1,y1,x2,y2=a; X1,Y1,X2,Y2=b
    inter=max(0,min(x2,X2)-max(x1,X1))*max(0,min(y2,Y2)-max(y1,Y1))
    area_a=max(0,x2-x1)*max(0,y2-y1); area_b=max(0,X2-X1)*max(0,Y2-Y1)
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0

def features(obj, width, height):
    x1,y1,x2,y2=obj["bbox"]; w=x2-x1; h=y2-y1
    return {"center_x":(x1+x2)/2,"center_y":(y1+y2)/2,"width":w,"height":h,"area":w*h,
            "normalized_center":[(x1+x2)/(2*width),(y1+y2)/(2*height)],"normalized_bbox":[x1/width,y1/height,x2/width,y2/height],
            "aspect_ratio":w/h if h else 0}

def infer_relations(objects, width, height):
    edges=[]
    for i,a in enumerate(objects):
        ax1,ay1,ax2,ay2=a["bbox"]; af=features(a,width,height)
        for b in objects[i+1:]:
            bx1,by1,bx2,by2=b["bbox"]; bf=features(b,width,height)
            dx=bf["center_x"]-af["center_x"]; dy=bf["center_y"]-af["center_y"]
            dist=math.hypot(dx/width,dy/height)
            overlap=iou(a["bbox"],b["bbox"])
            def add(subject,obj,kind,score,evidence):
                edges.append({"subject_id":subject["id"],"object_id":obj["id"],"type":kind,"confidence":round(max(0,min(1,score)),3),"method":"geometry_heuristic","evidence":evidence})
            if overlap >= .08:
                add(a,b,"overlapping",min(.99,.65+overlap*.34),{"iou":round(overlap,3)})
            # A contained box is often a useful visual relation; geometry alone does not imply semantic containment.
            if ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2 and overlap > .5:
                add(b,a,"inside",min(.98,.7+overlap*.28),{"inner_box_inside_outer_box":True,"iou":round(overlap,3)})
            vertical_gap=max(0,max(ay1,by1)-min(ay2,by2))/height
            horizontal_gap=max(0,max(ax1,bx1)-min(ax2,bx2))/width
            if abs(dy)/height > .045 and overlap < .18:
                if dy > 0: add(a,b,"above",.62+min(.32,abs(dy)/height*.4),{"normalized_center_delta_y":round(-dy/height,3),"iou":round(overlap,3)})
                else: add(b,a,"above",.62+min(.32,abs(dy)/height*.4),{"normalized_center_delta_y":round(dy/height,3),"iou":round(overlap,3)})
            if abs(dx)/width > .045 and overlap < .18:
                if dx > 0: add(a,b,"left_of",.62+min(.32,abs(dx)/width*.4),{"normalized_center_delta_x":round(-dx/width,3),"iou":round(overlap,3)})
                else: add(b,a,"left_of",.62+min(.32,abs(dx)/width*.4),{"normalized_center_delta_x":round(dx/width,3),"iou":round(overlap,3)})
            if dist < .18 and overlap < .08:
                add(a,b,"beside",.55+max(0,.18-dist)*1.8,{"normalized_center_distance":round(dist,3),"iou":round(overlap,3)})
    # Keep at most the strongest instance of each directed relation.
    unique={}
    for e in edges:
        key=(e["subject_id"],e["object_id"],e["type"])
        if key not in unique or e["confidence"]>unique[key]["confidence"]: unique[key]=e
    return list(unique.values())

def make_graph(objects, relations, width, height):
    graph=nx.DiGraph()
    for obj in objects: graph.add_node(obj["id"], **{**obj,"features":features(obj,width,height)})
    for rel in relations: graph.add_edge(rel["subject_id"],rel["object_id"],**rel)
    return graph

def serialize_graph(graph):
    return {"nodes":[dict(data) for _,data in graph.nodes(data=True)],"edges":[dict(data) for _,_,data in graph.edges(data=True)]}

def describe(objects, relations):
    if not objects: return "No objects were detected above the selected confidence threshold."
    labels={o["id"]:o["label"] for o in objects}
    names=", ".join(f"{label} ({sum(o['label']==label for o in objects)})" for label in sorted(set(labels.values())))
    sentence=f"Observed objects: {names}."
    if relations:
        rels=[f"{labels[r['subject_id']]} {r['type'].replace('_',' ')} {labels[r['object_id']]}" for r in relations[:6]]
        sentence+=" Geometry suggests: "+"; ".join(rels)+". These relationships are heuristic and may be uncertain."
    else: sentence+=" No reliable spatial relationships were inferred."
    return sentence

def answer_question(question, objects, relations):
    q=question.lower().strip().rstrip("?.! ")
    if not q: return "Please enter a question."
    if any(s in q for s in ("what objects","which objects","objects in","in the scene","what is in")):
        return "Detected: "+(", ".join(f"{label} ({sum(o['label']==label for o in objects)})" for label in sorted({o['label'] for o in objects})) if objects else "no objects.")
    counts={o["label"]:sum(x["label"]==o["label"] for x in objects) for o in objects}
    for label,n in counts.items():
        if label in q and any(x in q for x in ("how many","count","number of")): return f"{n} {label}{'s' if n!=1 else ''} detected."
    rel_types={"above":"above","over":"above","below":"below","under":"below","left of":"left_of","right of":"right_of","beside":"beside","next to":"beside","inside":"inside","overlapping":"overlapping"}
    rel=next(((phrase,typ) for phrase,typ in rel_types.items() if phrase in q),None)
    if rel:
        phrase,typ=rel
        # Queries often ask 'what is above the table'; the named anchor is the target node.
        anchor=next((o for o in objects if o["label"] in q),None)
        hits=[]
        if anchor:
            inverse={"above":"below","below":"above","left_of":"right_of","right_of":"left_of"}.get(typ,typ)
            for r in relations:
                if r["object_id"]==anchor["id"] and r["type"]==typ: hits += [r["subject_id"]]
                elif r["subject_id"]==anchor["id"] and r["type"]==inverse: hits += [r["object_id"]]
        else:
            for r in relations:
                if r["type"]==typ: hits.append(r["subject_id"])
        if hits:
            lookup={o["id"]:o["label"] for o in objects}
            return f"Geometry suggests {', '.join(lookup[x] for x in hits)} {phrase} {anchor['label'] if anchor else 'another detected object'}."
        return "That relationship is not available in the scene graph."
    for obj in objects:
        if obj["label"] in q:
            neighbors=[r for r in relations if r["subject_id"]==obj["id"] or r["object_id"]==obj["id"]]
            if neighbors:
                lookup={o["id"]:o["label"] for o in objects}
                return "Connected by inferred geometry to: "+", ".join(f"{lookup[r['subject_id']]} {r['type'].replace('_',' ')} {lookup[r['object_id']]}" for r in neighbors)+"."
            return f"{obj['label']} is detected, but no spatial relationships were inferred for it."
    return "That information is not available from the detected objects and geometric scene graph."

def compare_scenes(a,b):
    # Greedy per-label nearest normalized centers; matches are approximate.
    remaining=list(b["objects"]); matches=[]; removed=[]
    for old in a["objects"]:
        candidates=[(math.hypot(old["features"]["normalized_center"][0]-n["features"]["normalized_center"][0],old["features"]["normalized_center"][1]-n["features"]["normalized_center"][1]),n) for n in remaining if n["label"]==old["label"]]
        if not candidates: removed.append(old["label"]); continue
        distance,new=min(candidates,key=lambda x:x[0]); remaining.remove(new)
        if distance>.06: matches.append({"label":old["label"],"distance":round(distance,3),"from":old["features"]["normalized_center"],"to":new["features"]["normalized_center"]})
    oldrels={(r["type"], aobj_label(a,r["subject_id"]),aobj_label(a,r["object_id"])) for r in a["relations"]}
    newrels={(r["type"], aobj_label(b,r["subject_id"]),aobj_label(b,r["object_id"])) for r in b["relations"]}
    return {"added_objects":[n["label"] for n in remaining],"removed_objects":removed,"moved_objects":matches,
            "added_relationships":[list(x) for x in sorted(newrels-oldrels)],"removed_relationships":[list(x) for x in sorted(oldrels-newrels)],"matching":"Approximate matching by class label and normalized center position."}

def aobj_label(scene, oid): return next((o["label"] for o in scene["objects"] if o["id"]==oid),"")
