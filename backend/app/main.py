from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .config import CORS_ORIGINS, MAX_FILE_BYTES
from . import scene
from .detector import detect

app=FastAPI(title="SceneMind API",version="1.0.0",description="Explainable visual scene understanding with YOLO and geometric scene graphs.")
app.add_middleware(CORSMiddleware,allow_origins=CORS_ORIGINS,allow_methods=["*"],allow_headers=["*"])

class AskRequest(BaseModel):
    question:str=Field(min_length=1,max_length=500)
    scene:dict

async def analyze_upload(file:UploadFile,threshold:float):
    if file.size and file.size>MAX_FILE_BYTES: raise HTTPException(413,"Image must be 5 MB or smaller.")
    data=await file.read(MAX_FILE_BYTES+1)
    if len(data)>MAX_FILE_BYTES: raise HTTPException(413,"Image must be 5 MB or smaller.")
    try: image=scene.load_image(data)
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    try: objects=detect(image,threshold)
    except Exception as exc: raise HTTPException(503,f"{exc}. Check model installation and weight download access.") from exc
    w,h=image.size; relations=scene.infer_relations(objects,w,h); graph=scene.make_graph(objects,relations,w,h)
    enriched=scene.serialize_graph(graph)
    for obj in objects: obj["features"]=next(n["features"] for n in enriched["nodes"] if n["id"]==obj["id"])
    return {"width":w,"height":h,"objects":objects,"relations":relations,"graph":enriched,"description":scene.describe(objects,relations)}

@app.get("/")
def root(): return {"name":"SceneMind API","status":"ok","docs":"/docs"}
@app.get("/api/health")
def health(): return {"status":"ok","detector":"YOLO","model":"lazy-loaded"}
@app.post("/api/analyze")
async def analyze(file:UploadFile=File(...),confidence:float=Form(.25)):
    if not 0<=confidence<=1: raise HTTPException(422,"Confidence must be between 0 and 1.")
    return await analyze_upload(file,confidence)
@app.post("/api/ask")
def ask(req:AskRequest):
    return {"answer":scene.answer_question(req.question,req.scene.get("objects",[]),req.scene.get("relations",[]))}
@app.post("/api/compare")
async def compare(image_a:UploadFile=File(...),image_b:UploadFile=File(...),confidence:float=Form(.25)):
    if not 0<=confidence<=1: raise HTTPException(422,"Confidence must be between 0 and 1.")
    a=await analyze_upload(image_a,confidence); b=await analyze_upload(image_b,confidence)
    return {"summary":scene.compare_scenes(a,b),"scene_a":a,"scene_b":b}
