"""Independent, deterministic signal processing helpers."""
import numpy as np
from scipy.signal import resample_poly
from scipy.fft import dct
from .config import MODEL_ID

def preprocess(samples, sample_rate, target_rate=16000):
    x=np.asarray(samples,dtype=np.float32)
    if x.ndim==2: x=x.mean(axis=1)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all(): raise ValueError("Audio must be a non-empty finite mono or multichannel signal.")
    if sample_rate<=0: raise ValueError("Sample rate must be positive.")
    if sample_rate!=target_rate:
        from math import gcd
        g=gcd(int(sample_rate),int(target_rate)); x=resample_poly(x,target_rate//g,sample_rate//g).astype(np.float32)
    return x, target_rate

def frames(x,size=1024,hop=512):
    if len(x)<size: x=np.pad(x,(0,size-len(x)))
    count=1+(len(x)-size)//hop
    return np.stack([x[i*hop:i*hop+size] for i in range(count)])

def analyze_signal(samples,sr):
    x,_=preprocess(samples,sr); f=frames(x); win=np.hanning(f.shape[1]); spec=np.abs(np.fft.rfft(f*win,axis=1)); freqs=np.fft.rfftfreq(f.shape[1],1/16000); power=spec**2; denom=power.sum(axis=1)+1e-12
    rms=np.sqrt(np.mean(f*f,axis=1)); zcr=np.mean(np.abs(np.diff(np.signbit(f),axis=1)),axis=1)
    centroid=(spec*freqs).sum(axis=1)/(spec.sum(axis=1)+1e-12); bandwidth=np.sqrt((power*(freqs-centroid[:,None])**2).sum(axis=1)/denom)
    cumulative=np.cumsum(power,axis=1); roll=np.array([freqs[min(np.searchsorted(row,row[-1]*.85),len(freqs)-1)] for row in cumulative])
    # Compact mel filterbank (26 bands), then 13 log-energy DCT coefficients.
    mel=lambda hz:2595*np.log10(1+hz/700); hz=lambda m:700*(10**(m/2595)-1)
    edges=hz(np.linspace(mel(0),mel(8000),28)); filters=np.zeros((26,len(freqs)))
    for m in range(26):
        left,center,right=edges[m:m+3]; filters[m]=np.maximum(0,np.minimum((freqs-left)/(center-left+1e-12),(right-freqs)/(right-center+1e-12)))
    mfcc=dct(np.log(np.maximum(power@filters.T,1e-10)),type=2,norm='ortho',axis=1)[:,:13]
    summary=lambda a:{"mean":float(np.mean(a)),"std":float(np.std(a)),"min":float(np.min(a)),"max":float(np.max(a))}
    return {"waveform":x[::max(1,len(x)//1200)].astype(float).tolist(),"features":{"rms":{"unit":"amplitude","summary":summary(rms),"values":rms[::max(1,len(rms)//300)].astype(float).tolist()},"zero_crossing_rate":{"unit":"crossings/frame","summary":summary(zcr),"values":zcr[::max(1,len(zcr)//300)].astype(float).tolist()},"spectral_centroid":{"unit":"Hz","summary":summary(centroid),"values":centroid[::max(1,len(centroid)//300)].astype(float).tolist()},"spectral_bandwidth":{"unit":"Hz","summary":summary(bandwidth),"values":bandwidth[::max(1,len(bandwidth)//300)].astype(float).tolist()},"spectral_rolloff":{"unit":"Hz (85%)","summary":summary(roll),"values":roll[::max(1,len(roll)//300)].astype(float).tolist()},"mfcc":{"unit":"dB-scaled coefficient","summary":summary(mfcc.mean(axis=0)),"coefficients":mfcc.mean(axis=0).astype(float).tolist()}},"background_noise_rms":float(np.percentile(rms,20)),"analysis_sample_rate":16000}

def make_spectrogram(samples,sr):
    x,_=preprocess(samples,sr); f=frames(x,1024,512); spec=20*np.log10(np.maximum(np.abs(np.fft.rfft(f*np.hanning(1024),axis=1)),1e-6)); spec=spec[::max(1,len(spec)//160),::2]
    return {"values":np.round(spec,1).astype(float).tolist(),"time_step":512/16000,"frequency_step":16000/1024*2,"min_db":float(spec.min()),"max_db":float(spec.max())}

def detect_silence(samples,sr,threshold_db=-42,min_duration=.35):
    x,_=preprocess(samples,sr); size=5120; hop=1600; f=frames(x,size,hop); rms=np.sqrt(np.mean(f*f,axis=1)); db=20*np.log10(np.maximum(rms,1e-8)); silent=db<threshold_db; events=[]; start=None
    for i,flag in enumerate(silent):
        if flag and start is None: start=i*hop/16000
        if start is not None and (not flag or i==len(silent)-1):
            end=(i*hop/16000) if not flag else len(x)/16000
            if end-start>=min_duration: events.append({"start":round(start,3),"end":round(end,3)})
            start=None
    return events

def compare_features(a,b):
    keys=["rms","zero_crossing_rate","spectral_centroid","spectral_bandwidth","spectral_rolloff"]
    av=np.array([a["features"][k]["summary"]["mean"] for k in keys]); bv=np.array([b["features"][k]["summary"]["mean"] for k in keys]); av=av/(np.linalg.norm(av)+1e-12); bv=bv/(np.linalg.norm(bv)+1e-12)
    return {"duration_a":a["duration"],"duration_b":b["duration"],"features":{k:{"a":a["features"][k]["summary"]["mean"],"b":b["features"][k]["summary"]["mean"],"unit":a["features"][k]["unit"]} for k in keys},"cosine_distance":float(1-np.dot(av,bv)),"method":"Cosine distance on normalized mean signal-feature vector; similarity does not establish a shared source or event."}

_pipeline=None
def classify_chunks(samples,sr,chunk_seconds=10):
    global _pipeline
    try:
        from transformers import pipeline
        if _pipeline is None: _pipeline=pipeline("audio-classification",model=MODEL_ID,device=-1)
    except Exception as exc: raise RuntimeError(f"AST classifier unavailable: {exc}") from exc
    x,_=preprocess(samples,sr); events=[]
    for start in range(0,len(x),chunk_seconds*16000):
        chunk=x[start:start+chunk_seconds*16000]
        if not len(chunk): continue
        preds=_pipeline({"raw":chunk,"sampling_rate":16000},top_k=3)
        for p in sorted(preds,key=lambda z:z["score"],reverse=True): events.append({"start":start/16000,"end":min((start+len(chunk))/16000,len(x)/16000),"label":p["label"],"score":float(p["score"])})
    return events
