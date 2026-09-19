import torch,sys
sys.path.insert(0,'/mnt/workspace/eshine/PhysPRP/PhysGaussian')
from relation_mpm import RelationMPM,build_relation_edges
class L(torch.nn.Module):
 def __init__(self,b):
  super().__init__();self.b=b
  for p in b.parameters():p.requires_grad=False
  self.a=torch.nn.Parameter(torch.randn(4,b.in_features,device=b.weight.device)*.01);self.z=torch.nn.Parameter(torch.zeros(b.out_features,4,device=b.weight.device))
 def forward(self,x):return self.b(x)+x@self.a.t()@self.z.t()*2
torch.manual_seed(1);d='cuda';n=128
x=torch.cat((torch.randn(n//2,3)*.006,torch.randn(n//2,3)*.006+torch.tensor([.025,0,0])),0).to(d).float();v=torch.randn(n,3,device=d)*.02;o=torch.cat((torch.zeros(n//2,dtype=torch.long),torch.ones(n//2,dtype=torch.long))).to(d);m=o.clone();mass=torch.ones(n,device=d);E=torch.full((n,),2e6,device=d);nu=torch.full((n,),.3,device=d);rho=torch.full((n,),500.,device=d)
M=RelationMPM(2,32,.08,.08,2.).to(d)
for q in ['query','key','value','force_output']:setattr(M,q,L(getattr(M,q)))
opt=torch.optim.AdamW([p for p in M.parameters() if p.requires_grad],lr=.003)
P,_=build_relation_edges(x,o,m,.055,.055);y=torch.zeros_like(v);i,j=P.unbind(1);q=x[j]-x[i];r=q.norm(1,keepdim=True).clamp_min(1e-6);u=q/r;c=(o[i]!=o[j])&(r[:,0]<.05);f=(c.float()[:,None]*(.8*(.04-r)-.15*((v[j]-v[i])*u).sum(1,keepdim=True))).clamp(-.08,.08)*u;y.index_add_(0,i,f);y.index_add_(0,j,-f);y*=.01
for t in range(101):
 z=M(x,v,o,m,mass,.01,E,nu,rho);loss=(z-y).square().mean();opt.zero_grad();loss.backward();opt.step()
 if t%25==0:print(t,float(loss))
torch.save({'model':M.state_dict()},'/mnt/workspace/eshine/PhysPRP/PhysGaussian/checkpoints/relation_lora_synth.pt');print('saved')