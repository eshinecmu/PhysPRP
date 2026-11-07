"""Run without API/GPU: python tests/check_rest3d_runtime.py."""
import ast
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import sys, tempfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    spec=importlib.util.spec_from_file_location("rest3d_vlm",ROOT/"REST3D/rest3d/utils/vlm.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with patch.dict(os.environ,{"LAB_LLM_API_KEY":"test-placeholder","LAB_LLM_BASE_URL":"https://test.invalid/v1/","LAB_LLM_MODEL":"configured-model"}),patch("requests.post") as post:
        post.return_value.json.return_value={"choices":[{"message":{"content":"ok"}}]}
        assert module.generate_vlm_response_gpt4o([{"role":"user","content":[{"type":"text","text":"test"}]}])=="ok"
        assert post.call_args.args[0]=="https://test.invalid/v1/chat/completions"
        assert post.call_args.kwargs["json"]["model"]=="configured-model"
    client=MagicMock()
    client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='toy, fabric, 20-40, Shore A'))])
    with patch.dict(sys.modules, {'openai':SimpleNamespace(OpenAI=MagicMock(return_value=client))}), patch.dict(os.environ, {'LAB_LLM_API_KEY':'test-placeholder'}), tempfile.TemporaryDirectory() as temporary:
        spec=importlib.util.spec_from_file_location('gp_vlm',ROOT/'GaussianProperty/utils/vlm_utils.py')
        gp=importlib.util.module_from_spec(spec);spec.loader.exec_module(gp)
        image=Path(temporary)/'image.png';image.write_bytes(b'test')
        assert gp.GPT4V(image,'test') == gp.Qwen(image,'test') == 'toy, fabric, 20-40, Shore A'
    tree=ast.parse((ROOT/"REST3D/scripts/infer_scenetree.py").read_text(encoding="utf-8"))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="call_sam_service"]
    assert calls and all(not c.args and {"image_path","text_prompt"}.issubset({k.arg for k in c.keywords}) for c in calls)
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="prompt_safe_syn" for t in n.targets))
    assert eval(compile(ast.Expression(assignment.value),"<path>","eval"),{"synonym":"hardwood floor"})=="hardwood floor"
    print("PASS: VLM environment routing, bound SAM service keywords, synonym cache path")

if __name__=="__main__":main()
