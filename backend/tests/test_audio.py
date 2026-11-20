import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from app.main import app
from app.services import analyze_signal, detect_silence, make_spectrogram, preprocess, compare_features

def tone(freq=440,duration=1,sr=22050):
    x=np.sin(2*np.pi*freq*np.arange(int(sr*duration))/sr).astype('float32')
    b=io.BytesIO(); sf.write(b,x,sr,format='WAV'); return x,b.getvalue()

def test_preprocess_mono_and_resample():
    x=np.zeros((2205,2),dtype='float32'); y,sr=preprocess(x,22050)
    assert sr==16000 and y.shape==(1600,)

def test_analysis_and_spectrogram_are_real():
    x,_=tone(); r=analyze_signal(x,22050); s=make_spectrogram(x,22050)
    assert r['features']['rms']['summary']['mean']>0.5
    assert len(r['features']['mfcc']['coefficients'])==13 and s['values']

def test_silence_intervals():
    x=np.concatenate([np.ones(16000,dtype='float32')*.1,np.zeros(16000,dtype='float32'),np.ones(16000,dtype='float32')*.1])
    result=detect_silence(x,16000)
    assert len(result)==1 and .8<result[0]['start']<1.2 and result[0]['end']>1.8

def test_identical_comparison_distance_zero():
    x,_=tone(); a=analyze_signal(x,22050); a['duration']=1
    assert compare_features(a,a)['cosine_distance']<1e-8

def test_health_and_invalid_upload():
    with TestClient(app) as client:
        assert client.get('/api/health').json()['status']=='ok'
        assert client.post('/api/audio/upload',files={'file':('sample.txt',b'no','text/plain')}).status_code==415

def test_upload_and_history_delete(tmp_path,monkeypatch):
    from app import config
    from app import main
    monkeypatch.setattr(config,'STORAGE_DIR',tmp_path/'uploads'); monkeypatch.setattr(config,'DATABASE_PATH',tmp_path/'db.sqlite')
    monkeypatch.setattr(main,'STORAGE_DIR',tmp_path/'uploads')
    from app.database.session import init_db
    init_db()
    _,data=tone()
    with TestClient(app) as client:
        created=client.post('/api/audio/upload',files={'file':('tone.wav',data,'audio/wav')})
        assert created.status_code==200
        audio_id=created.json()['audio_id']; assert client.get('/api/history').json()[0]['id']==audio_id
        ranged=client.get('/api/audio/'+audio_id+'/stream',headers={'Range':'bytes=0-15'})
        assert ranged.status_code==206 and len(ranged.content)==16
        assert client.delete('/api/history/'+audio_id).json()['deleted']
