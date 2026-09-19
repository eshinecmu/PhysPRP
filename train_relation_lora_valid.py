import torch,sys
sys.path.insert(0,'/mnt/workspace/eshine/PhysPRP/PhysGaussian')
from relation_mpm import RelationMPM,build_relation_edges
class LoRA(torch.nn.Module):
 def __init__(self,b,r=4):
  super().__init__(); self.base=b
  for p in b.parameters(): p.requires_grad_(False)
  self.A=torch.nn.Parameter(torch.randn(r,b.in_features,device=b.weight.device)*.02)
  self.B=torch.nn.Parameter(torch.zeros(b.out_features,r,device=b.weight.device))
 def forward(self,x): return self.base(x)+x@self.A.t()@self.B.t()*2
D='cuda'; torch.manual_seed(3); n=64
x=torch.cat((torch.randn(32,3,device=D)*.003,torch.randn(32,3,device=D)*.003+torch.tensor([.025,0,0],device=D)),0).float()
v=torch.randn(n,3,device=D)*.01; oid=torch.cat((torch.zeros(32,dtype=torch.long),torch.ones(32,dtype=torch.long))).to(D); mid=oid.clone()
mass=torch.ones(n,device=D); E=torch.full((n,),2e6,device=D); nu=torch.full((n,),.3,device=D); rho=torch.full((n,),500.,device=D)
pairs,types=build_relation_edges(x,oid,mid,.08,.08); cross=types==2; i,j=pairs[cross].unbind(1); d=x[j]-x[i]; dist=d.norm(dim=1,keepdim=True).clamp_min(1e-6); normal=d/dist
force=(.8*(.03-dist)-.15*((v[j]-v[i])*normal).sum(1,keepdim=True))*normal; target=torch.zeros_like(v); target.index_add_(0,i,force); target.index_add_(0,j,-force); target*=.01
M=RelationMPM(2,32,.08,.08,2.).to(D)
for p in M.parameters(): p.requires_grad_(False)
for q in ['query','key','value','force_output']: setattr(M,q,LoRA(getattr(M,q)))
trainable=[p for p in M.parameters() if p.requires_grad]; opt=torch.optim.AdamW(trainable,lr=.004)
with torch.no_grad(): init=M(x,v,oid,mid,mass,.01,E,nu,rho); print('cross_edges',int(cross.sum()),'target_norm',float(target.norm()),'trainable',sum(p.numel() for p in trainable),'initial_loss',float((init-target).square().mean()))
for step in range(201):
 pred=M(x,v,oid,mid,mass,.01,E,nu,rho); loss=(pred-target).square().mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.); opt.step()
 if step%25==0: print('step',step,'loss',float(loss))
out='/mnt/workspace/eshine/PhysPRP/PhysGaussian/checkpoints/relation_lora_synth.pt'; torch.save({'model':M.state_dict()},out); print('saved',out)
