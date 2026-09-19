from pathlib import Path
p=Path('/mnt/workspace/eshine/PhysPRP/PhysGaussian/gs_simulation.py'); s=p.read_text()
s=s.replace('parser.add_argument("--relation_checkpoint", type=str, default=None)', 'parser.add_argument("--relation_checkpoint", type=str, default=None)\n    parser.add_argument("--relation_full_checkpoint", type=str, default=None)')
needle='''        if args.relation_checkpoint:
            class LoRA'''
insert='''        if args.relation_full_checkpoint:
            full = torch.load(args.relation_full_checkpoint, map_location=device)["model"]
            current = relation.state_dict()
            usable = {k:v for k,v in full.items() if k in current and current[k].shape == v.shape}
            relation.load_state_dict(usable, strict=False)
            print("loaded full relation Transformer", args.relation_full_checkpoint, "tensors", len(usable))
        if args.relation_checkpoint:
            class LoRA'''
if needle not in s: raise SystemExit('needle missing')
p.write_text(s.replace(needle,insert))
