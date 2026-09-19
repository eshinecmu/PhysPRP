from pathlib import Path
p=Path('/mnt/workspace/eshine/PhysPRP/train_relation_lora_valid.py')
s=p.read_text(); s=s.replace("M=RelationMPM(2,32,.08,.08,2.).to(D)\nfor q", "M=RelationMPM(2,32,.08,.08,2.).to(D)\nfor p in M.parameters(): p.requires_grad_(False)\nfor q")
p.write_text(s)
p=Path('/mnt/workspace/eshine/PhysPRP/PhysGaussian/gs_simulation.py'); s=p.read_text()
old='''                relation_module = getattr(relation, name)\n                setattr(relation, name, LoRA(relation_module, {"A": ckpt[name+".A"], "B": ckpt[name+".B"]}).to(device))'''
new='''                relation_module = getattr(relation, name)\n                base_state = {"weight": ckpt[name+".base.weight"], "bias": ckpt[name+".base.bias"]}\n                relation_module.load_state_dict(base_state)\n                setattr(relation, name, LoRA(relation_module, {"A": ckpt[name+".A"], "B": ckpt[name+".B"]}).to(device))'''
if old not in s: raise SystemExit('loader pattern not found')
p.write_text(s.replace(old,new))
