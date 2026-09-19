import torch,sys
sys.path.insert(0,'/mnt/workspace/eshine/PhysPRP/PhysGaussian')
from relation_mpm import RelationMPM,build_relation_edges
D='cuda'; torch.manual_seed(3); n=64
x=torch.cat((torch.randn(32,3,device=D)*.003,torch.randn(32,3,device=D)*.003+torch.tensor([.025,0,0],device=D)),0).float()
v=torch.randn(n,3,device=D)*.01; oid=torch.cat((torch.zeros(32,dtype=torch.long),torch.ones(32,dtype=torch.long))).to(D); mid=oid.clone()
mass=torch.ones(n,device=D); E=torch.full((n,),2e6,device=D); nu=torch.full((n,),.3,device=D); rho=torch.full((n,),500.,device=D)
pairs,types=build_relation_edges(x,oid,mid,.08,.08); cross=types==2; i,j=pairs[cross].unbind(1); d=x[j]-x[i]; dist=d.norm(dim=1,keepdim=True).clamp_min(1e-6); normal=d/dist
force=(.8*(.03-dist)-.15*((v[j]-v[i])*normal).sum(1,keepdim=True))*normal; target=torch.zeros_like(v); target.index_add_(0,i,force); target.index_add_(0,j,-force); target*=.01
model=RelationMPM(2,32,.08,.08,2.).to(D); opt=torch.optim.AdamW(model.parameters(),lr=.002)
with torch.no_grad(): init=model(x,v,oid,mid,mass,.01,E,nu,rho); print('cross_edges',int(cross.sum()),'target_norm',float(target.norm()),'trainable',sum(p.numel() for p in model.parameters()),'initial_loss',float((init-target).square().mean()))
for step in range(301):
 pred=model(x,v,oid,mid,mass,.01,E,nu,rho); loss=(pred-target).square().mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
 if step%25==0: print('step',step,'loss',float(loss))
out='/mnt/workspace/eshine/PhysPRP/PhysGaussian/checkpoints/relation_full_synth.pt'; torch.save({'model':model.state_dict(),'config':{'n_materials':2,'hidden_dim':32,'local_radius':.08,'contact_radius':.08}},out); print('saved',out)
