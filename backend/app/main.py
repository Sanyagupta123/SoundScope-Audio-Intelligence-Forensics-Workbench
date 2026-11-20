"""SoundScope audio analysis API."""
import io, json, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from .config import CORS_ORIGINS, MAX_FILE_BYTES, STORAGE_DIR
from .services import analyze_signal, make_spectrogram, detect_silence, compare_features, classify_chunks
from .database.session import init_db, connect

app=FastAPI(title="SoundScope API",version="1.0.0",description="Audio intelligence and signal analysis workbench.")
app.add_middleware(CORSMiddleware,allow_origins=CORS_ORIGINS,allow_methods=["*"],allow_headers=["*"])

class CompareRequest(BaseModel):
    audio_a: str
    audio_b: str

@app.on_event("startup")
def startup():
    STORAGE_DIR.mkdir(parents=True,exist_ok=True); init_db()

def read_audio(data: bytes):
    try: samples,sr=sf.read(io.BytesIO(data),dtype="float32",always_2d=True)
    except Exception:
        try:
            import librosa
            mono,sr=librosa.load(io.BytesIO(data),sr=None,mono=False)
            samples=mono.T if mono.ndim>1 else mono[:,None]
        except Exception as exc: raise HTTPException(400,"Could not decode audio. Upload a valid WAV, FLAC or MP3 recording.") from exc
    if samples.size == 0 or not np.isfinite(samples).all(): raise HTTPException(400,"Audio is empty or contains invalid samples.")
    mono=samples.mean(axis=1)
    if len(mono)/sr > 120: raise HTTPException(400,"Audio duration must be 120 seconds or less.")
    return mono,sr

def fetch(audio_id):
    with connect() as db: row=db.execute("select * from analyses where id=?",(audio_id,)).fetchone()
    if not row: raise HTTPException(404,"Analysis not found.")
    return dict(row)

@app.get("/")
def root(): return {"name":"SoundScope API","status":"ok","docs":"/docs"}
@app.get("/api/health")
def health(): return {"status":"ok","classifier":"lazy optional AST","model":"MIT/ast-finetuned-audioset-10-10-0.4593"}

@app.post("/api/audio/upload")
async def upload(file:UploadFile=File(...)):
    name=Path(file.filename or "recording").name
    if Path(name).suffix.lower() not in {".wav",".flac",".mp3"}: raise HTTPException(415,"Supported formats: WAV, FLAC and MP3.")
    data=await file.read(MAX_FILE_BYTES+1)
    if len(data)>MAX_FILE_BYTES: raise HTTPException(413,"Audio must be 25 MB or smaller.")
    mono,sr=read_audio(data); audio_id=str(uuid.uuid4()); path=STORAGE_DIR/f"{audio_id}.audio"
    path.write_bytes(data); duration=len(mono)/sr
    with connect() as db: db.execute("insert into analyses(id,filename,path,sample_rate,duration,created_at,status) values(?,?,?,?,?,?,?)",(audio_id,name,str(path),sr,duration,datetime.now(timezone.utc).isoformat(),"uploaded"))
    return {"audio_id":audio_id,"filename":name,"duration":duration,"sample_rate":sr,"channels":1,"size":len(data),"status":"uploaded"}

@app.post("/api/audio/{audio_id}/analyze")
def analyze(audio_id:str):
    row=fetch(audio_id); samples,sr=read_audio(Path(row["path"]).read_bytes())
    try:
        result=analyze_signal(samples,sr); silence=detect_silence(samples,sr); spec=make_spectrogram(samples,sr)
        try: predictions=classify_chunks(samples,sr); classifier_error=None
        except Exception as exc: predictions=[]; classifier_error=str(exc)
        result.update({"audio_id":audio_id,"filename":row["filename"],"duration":len(samples)/sr,"sample_rate":sr,"silence":silence,"predictions":predictions,"spectrogram":spec,"classifier_error":classifier_error,"status":"completed"})
        with connect() as db: db.execute("update analyses set status='completed',result=? where id=?",(json.dumps(result),audio_id))
        return result
    except HTTPException: raise
    except Exception as exc:
        with connect() as db: db.execute("update analyses set status='failed' where id=?",(audio_id,))
        raise HTTPException(500,f"Audio analysis failed: {exc}") from exc

@app.get("/api/audio/{audio_id}/results")
def results(audio_id:str):
    row=fetch(audio_id)
    if not row["result"]: raise HTTPException(409,"This recording has not been analyzed yet.")
    return json.loads(row["result"])

@app.get("/api/audio/{audio_id}/stream")
def stream(audio_id:str,range_header:str|None=Header(default=None,alias="Range")):
    row=fetch(audio_id); data=Path(row["path"]).read_bytes(); ext=Path(row["filename"]).suffix.lower(); mime={".wav":"audio/wav",".mp3":"audio/mpeg",".flac":"audio/flac"}[ext]
    headers={"Accept-Ranges":"bytes"}
    if range_header:
        try:
            start_s,end_s=range_header.removeprefix("bytes=").split("-",1); start=int(start_s or 0); end=int(end_s) if end_s else len(data)-1
            if start<0 or end<start or start>=len(data): raise ValueError
            end=min(end,len(data)-1); chunk=data[start:end+1]
            headers.update({"Content-Range":f"bytes {start}-{end}/{len(data)}","Content-Length":str(len(chunk))})
            return Response(chunk,status_code=206,media_type=mime,headers=headers)
        except ValueError: raise HTTPException(416,"Invalid byte range.")
    headers["Content-Length"]=str(len(data)); return Response(data,media_type=mime,headers=headers)

@app.get("/api/audio/{audio_id}/spectrogram")
def spectrogram(audio_id:str):
    row=fetch(audio_id); samples,sr=read_audio(Path(row["path"]).read_bytes()); return make_spectrogram(samples,sr)

@app.post("/api/compare")
def compare(req:CompareRequest):
    a,b=fetch(req.audio_a),fetch(req.audio_b)
    if not a["result"] or not b["result"]: raise HTTPException(409,"Analyze both recordings before comparing.")
    return compare_features(json.loads(a["result"]),json.loads(b["result"]))

@app.get("/api/history")
def history():
    with connect() as db: rows=db.execute("select id,filename,sample_rate,duration,created_at,status from analyses order by created_at desc").fetchall()
    return [dict(r) for r in rows]
@app.get("/api/history/{analysis_id}")
def history_detail(analysis_id:str):
    row=fetch(analysis_id); return {**{k:row[k] for k in ("id","filename","sample_rate","duration","created_at","status")},"result":json.loads(row["result"]) if row["result"] else None}
@app.delete("/api/history/{analysis_id}")
def delete(analysis_id:str):
    row=fetch(analysis_id)
    try: Path(row["path"]).unlink(missing_ok=True)
    finally:
        with connect() as db: db.execute("delete from analyses where id=?",(analysis_id,))
    return {"deleted":True}
