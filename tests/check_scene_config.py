"""Run the actual JSON decoder without importing unrelated GPU boundary kernels."""
import ast
import json
from pathlib import Path
import tempfile
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'PhysGaussian/utils/decode_param.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decode_param_json')
namespace={'json':json}
exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual decoder>','exec'),namespace)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'config.json'
    for config in ({},{'particle_filling':None}):
        path.write_text(json.dumps(config));assert namespace['decode_param_json'](path)[3]['particle_filling'] is None
    path.write_text(json.dumps({'particle_filling':{}}))
    result=namespace['decode_param_json'](path)
    assert result[3]['particle_filling']['n_grid']==4*result[0]['n_grid']
    path.write_text(json.dumps({'particle_filling':[]}))
    try:namespace['decode_param_json'](path)
    except ValueError:pass
    else:raise AssertionError('invalid fill config accepted')
print('PASS: missing/null filling disabled, empty object defaults, invalid type rejected')
