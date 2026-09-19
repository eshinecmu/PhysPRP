from pathlib import Path
p=Path('/mnt/workspace/eshine/PhysPRP/PhysGaussian/gs_simulation.py')
s=p.read_text()
s=s.replace('parser.add_argument("--property_path", type=str, default=None)', 'parser.add_argument("--property_path", type=str, default=None)\n    parser.add_argument("--relation_checkpoint", type=str, default=None)')
s=s.replace('        if os.getenv("RELATION_ANALYTIC_PRIOR") == "1":', '''        if args.relation_checkpoint:
            class LoRA(torch.nn.Module):
                def __init__(self, base, state):
                    super().__init__(); self.base = base
                    for p in base.parameters(): p.requires_grad_(False)
                    rank = state["A"].shape[0]
                    self.A = torch.nn.Parameter(state["A"].to(device)); self.B = torch.nn.Parameter(state["B"].to(device)); self.scale = 2.0
                def forward(self, x): return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale
            ckpt = torch.load(args.relation_checkpoint, map_location=device)["model"]
            for name in ("query", "key", "value", "force_output"):
                relation_module = getattr(relation, name)
                setattr(relation, name, LoRA(relation_module, {"A": ckpt[name+".A"], "B": ckpt[name+".B"]}).to(device))
            print("loaded relation LoRA", args.relation_checkpoint)
        if os.getenv("RELATION_ANALYTIC_PRIOR") == "1":''')
p.write_text(s)
