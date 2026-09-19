import torch,sys,random
sys.path.insert(0,'/mnt/workspace/eshine/PhysPRP/PhysGaussian')
from relation_mpm import RelationMPM,build_relation_edges
D='cuda'; torch.manual_seed(10); random.seed(10); S=64; n=64
samples=[]
for k in range(S):
 gap=random.uniform(.012,.045); spread=random.uniform(.002,.008); speed=random.uniform(.0,.03); stiff=random.uniform(.3,1.2); damp=random.uniform(.05,.3)
 x=torch.cat((torch.randn(32,3)*spread,torch.randn(32,3)*spread+torch.tensor([gap,0,0])),0).to(D).float(); v=torch.zeros(n,3,device=D); v[:32,0]=speed; v[32:,0]=-speed
 oid=torch.cat((torch.zeros(32,dtype=torch.long),torch.ones(32,dtype=torch.long))).to(D); mid=oid.clone(); mass=torch.ones(n,device=D); E=torch.full((n,),2e6,device=D); nu=torch.full((n,),.3,device=D); rho=torch.full((n,),500.,device=D)
 P,T=build_relation_edges(x,oid,mid,.08,.08); mask=T==2
 if not mask.any(): continue
 i,j=P[mask].unbind(1); d=x[j]-x[i]; dist=d.norm(dim=1,keepdim=True).clamp_min(1e-6); normal=d/dist
 f=(stiff*(.03-dist)-damp*((v[j]-v[i])*normal).sum(1,keepdim=True))*normal; y=torch.zeros_like(v); y.index_add_(0,i,f); y.index_add_(0,j,-f); y*=.01
 samples.append((x,v,oid,mid,mass,E,nu,rho,y))
print('samples',len(samples),'edges',sum(int((build_relation_edges(a[0],a[2],a[3],.08,.08)[1]==2).sum()) for a in samples))
M=RelationMPM(16,32,.08,.08,2.).to(D); opt=torch.optim.AdamW(M.parameters(),lr=.002)
for ep in range(31):
 total=0.; opt.zero_grad()
 for x,v,o,m,mass,E,nu,rho,y in samples:
  total=total+((M(x,v,o,m,mass,.01,E,nu,rho)-y)**2).mean()
 total=total/len(samples); total.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(),1.); opt.step()
 if ep%5==0: print('epoch',ep,'mean_loss',float(total))
out='/mnt/workspace/eshine/PhysPRP/PhysGaussian/checkpoints/relation_full_multi.pt'; torch.save({'model':M.state_dict(),'samples':len(samples)},out); print('saved',out)
