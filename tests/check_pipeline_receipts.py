from pathlib import Path
import tempfile,sys,importlib.util,json
root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('pipeline',root/'run_full_pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as d:
    work=Path(d);(work/'validation').mkdir();out=work/'result.txt'
    cmd=[sys.executable,'-c',"from pathlib import Path;Path(r'"+str(out)+"').write_text('verified')"]
    m.stage('probe',cmd,[out],work,None,resume=False)
    m.stage('probe',cmd,[out],work,None,resume=True)
    out.write_text('tampered')
    try: m.stage('probe',cmd,[out],work,None,resume=True)
    except ValueError: pass
    else: raise AssertionError('tampered cache accepted')
    try: m.stage('failure',[sys.executable,'-c','raise SystemExit(3)'],[out],work,None)
    except Exception: pass
    assert json.loads((work/'validation/failure.json').read_text())['exit_code']==3
print('pipeline stage receipt checks passed')

with tempfile.TemporaryDirectory() as d:
    base=Path(d);cache=base/'producer';work=base/'consumer';scene=work/'reconstruction/image'
    raw=cache/'reconstruction/image/stage2/scene_raw';raw.mkdir(parents=True)
    (cache/'validation').mkdir();(work/'validation').mkdir(parents=True)
    for stage in (cache/'reconstruction/image/stage1',scene/'stage1'):
        stage.mkdir(parents=True);(stage/'mask').write_bytes(b'mask')
    asset=raw/'native.ply';asset.write_bytes(b'original')
    identity={'image_sha256':'image','runtime_sha256':'runtime'}
    (cache/'run.json').write_text(json.dumps({'identity':identity}))
    receipt={'status':'passed','outputs':{str(asset):m.sha256(asset)}}
    (cache/'validation/reconstruction.json').write_text(json.dumps(receipt))
    m.copy_reconstruction_cache(cache,scene,work,identity)
    assert (scene/'stage2/scene_raw/native.ply').read_bytes()==b'original'
    assert json.loads((work/'validation/reconstruction.json').read_text())['status']=='verified_cache'
    asset.write_bytes(b'tampered')
    try:m.copy_reconstruction_cache(cache,scene,work,identity)
    except ValueError:pass
    else:raise AssertionError('tampered reconstruction accepted')
print('verified reconstruction cache checks passed')
