"""WorldParticle material-prior refinement and MPM grid-velocity correction.

MaterialTransformer and WorldParticlePhysCoreCorrector reuse the original core.
The older GridCorrector below remains a separate numerical prototype.
"""
import math
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import torch
from torch import nn
from PhysGaussian.gaussian_property_adapter import MATERIAL_LIMITS

def refine_material(raw, prior, material_range="prior"):
    """Decode material corrections without reading material labels."""
    low_nu, high_nu = MATERIAL_LIMITS["nu"]
    prior_log_E = prior["E"].log()
    if material_range == "augmented":
        # ponytail: augmentation only needs softer materials than the GP prior;
        # retain its upper bound so the explicit solver needs no stiffer domain.
        log_E = (prior_log_E + (prior_log_E - math.log(100.)) * raw[:, 0].tanh()).clamp_max(prior_log_E)
        nu = (prior["nu"] + (high_nu-low_nu) * raw[:, 1].tanh()).clamp(low_nu, high_nu)
    elif material_range == "prior":
        low_E, high_E = MATERIAL_LIMITS["E"]
        log_E = (prior_log_E + math.log(10.) * raw[:, 0].tanh()).clamp(math.log(low_E), math.log(high_E))
        nu = (prior["nu"] + .05 * raw[:, 1].tanh()).clamp(low_nu, high_nu)
    else:
        raise ValueError("Unknown material range")
    return {"log_E": log_E[None], "nu": nu[None]}


class MaterialTransformer(nn.Module):
    def __init__(self, backbone, motion_dim, material_range="prior"):
        super().__init__()
        self.backbone = backbone
        if material_range not in ("prior", "augmented"):
            raise ValueError("Unknown material range")
        self.material_range = material_range
        self.input = nn.Linear(motion_dim + 3, 32)
        head = nn.Linear(backbone.output_network[-1].in_features, 2)
        self.backbone.output_network[-1] = head
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.backbone.output_scale = 1.0

    def forward(self, x0, v0, motion, prior, boundary, normals):
        prior_features = torch.stack((
            (prior["E"] / 2000.).log(),
            prior["nu"],
            (prior["density"] / 1000.).log(),
        ), dim=-1)
        features = self.input(torch.cat((motion, prior_features), dim=-1))
        raw, _ = self.backbone.compute_correction(
            x0, v0, features, boundary, normals, k_partite=2
        )
        return refine_material(raw, prior, self.material_range)

def support(x, grid, domain):
    dx=domain/grid
    gp=x/dx
    base=torch.floor(gp-.5).long()
    fx=gp-base
    off=torch.cartesian_prod(*[torch.arange(3,device=x.device)]*3)
    nodes=base[:,None]+off
    if bool(((nodes<0)|(nodes>=grid)).any()):
        raise ValueError("Prototype requires the entire 27-node support inside the grid")
    w=torch.stack((.5*(1.5-fx)**2,.75-(fx-1)**2,.5*(fx-.5)**2),2)
    dw=torch.stack((fx-1.5,-2*(fx-1),fx-.5),2)/dx
    weight=w[:,0,off[:,0]]*w[:,1,off[:,1]]*w[:,2,off[:,2]]
    grad=torch.stack((dw[:,0,off[:,0]]*w[:,1,off[:,1]]*w[:,2,off[:,2]],
                      w[:,0,off[:,0]]*dw[:,1,off[:,1]]*w[:,2,off[:,2]],
                      w[:,0,off[:,0]]*w[:,1,off[:,1]]*dw[:,2,off[:,2]]),2)
    flat=(nodes[:,:,0]*grid+nodes[:,:,1])*grid+nodes[:,:,2]
    ids,inv=torch.unique(flat.flatten(),sorted=True,return_inverse=True)
    pos=torch.stack((ids//(grid*grid),(ids//grid)%grid,ids%grid),1).to(x.dtype)*dx
    return dict(ids=ids,inv=inv.reshape(len(x),27),weight=weight,grad=grad,
                dpos=off[None]-fx[:,None],pos=pos,dx=dx)

def snapshot(s, object_id, material_id, dt):
    import warp as wp
    st,mo=s.mpm_state,s.mpm_model
    get=lambda name: wp.to_torch(getattr(st,'particle_'+name))
    x=get('x').clone()
    data=support(x,mo.n_grid,mo.grid_lim)
    data.update(x=x,v=get('v').clone(),F=get('F_trial').clone().reshape(-1,3,3),
                elastic_F=get('F').clone().reshape(-1,3,3),C=get('C').clone().reshape(-1,3,3),
                mass=get('mass').clone(),E=wp.to_torch(mo.E).clone(),
                nu=wp.to_torch(mo.nu).clone(),rho=get('density').clone(),
                object_id=object_id,material_id=material_id,dt=dt,
                gv=wp.to_torch(st.grid_v_out).reshape(-1,3)[data['ids']].clone(),
                gm=wp.to_torch(st.grid_m).flatten()[data['ids']].clone())
    return data

def corrected_state(data,dv,base):
    # Exact local derivative for a fixed elastic, interior native G2P. This is
    # deliberately not a differentiable full-MPM rollout.
    local=dv[data['inv']]
    pv=(local*data['weight'][...,None]).sum(1)
    grad=torch.einsum('nki,nkj->nij',local,data['grad'])
    return (base[0]+data['dt']*pv,base[1]+pv,
            base[2]+data['dt']*(grad@data['elastic_F']),
            base[3]+torch.einsum('nki,nkj,nk->nij',local,data['dpos'],data['weight'])*4/data['dx'])

class GridCorrector(nn.Module):
    schema='worldparticle-spatial-grid-residual-v1'
    def __init__(self,width=32,global_attention=True,velocity_scale=.3):
        super().__init__()
        self.width,self.global_attention=width,global_attention
        self.velocity_scale=float(velocity_scale)
        self.node=nn.Sequential(nn.Linear(27,width),nn.SiLU(),nn.Linear(width,width))
        self.local=nn.Linear(3,width,bias=False)
        self.qkv=nn.Linear(width,3*width)
        self.mix=nn.Linear(width,width)
        self.query=nn.Sequential(nn.Linear(8,width),nn.SiLU(),nn.Linear(width,width))
        self.key=nn.Linear(width,width)
        self.value=nn.Linear(width,width)
        self.output=nn.Linear(width,3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,d):
        x,m=d['x'],d['mass']
        cell=d['dx']*.65
        coord=torch.floor(x/cell).long()
        keys=torch.cat((coord,d['object_id'][:,None],d['material_id'][:,None]),1)
        _,group=torch.unique(keys,dim=0,sorted=True,return_inverse=True)
        k=int(group.max())+1
        if k>256:
            raise ValueError("Prototype token ceiling is 256; increase spatial cell size for larger scenes")
        tm=m.new_zeros(k).index_add(0,group,m)
        pool=lambda a: a.new_zeros((k,a.shape[1])).index_add(0,group,a*m[:,None])/tm[:,None]
        center=pool(x)
        # ponytail: cell-local moments replace Open3D CConv for this small probe;
        # resolve subcell contacts with radius-based continuous kernels if needed.
        feature=torch.cat(((x-center[group])/cell,d['v']/self.velocity_scale,
            (d['F']-torch.eye(3,device=x.device)).flatten(1),d['C'].flatten(1)*d['dt'],
            torch.stack((torch.log(d['E']/2000.),d['nu'],torch.log(d['rho']/1000.)),1)),1)
        token=pool(self.node(feature))
        rel=(center[None]-center[:,None])/cell
        neighbor=(rel.square().sum(-1)<9)&(d['object_id'].new_zeros(k).scatter(0,group,d['object_id'])[:,None]
            ==d['object_id'].new_zeros(k).scatter(0,group,d['object_id'])[None])
        token=token+torch.tanh((self.local(rel)*neighbor[...,None]).sum(1)/neighbor.sum(1).clamp_min(1)[:,None])
        if self.global_attention:
            q,key,value=self.qkv(token).chunk(3,-1)
            logits=q@key.T/math.sqrt(self.width)-.1*rel.square().sum(-1)
            token=token+self.mix(logits.softmax(-1)@value)
        centroid=(x*m[:,None]).sum(0)/m.sum()
        qfeat=torch.cat(((d['pos']-centroid)/.1,d['gv']/self.velocity_scale,
            x.new_full((len(d['pos']),1),d['dx']/.02),
            x.new_full((len(d['pos']),1),d['dt']/.001)),1)
        q=self.query(qfeat)
        bias=-((d['pos'][:,None]-center[None])/cell).square().sum(-1)
        attn=(q@self.key(token).T/math.sqrt(self.width)+bias).softmax(-1)
        raw=self.output(q+attn@self.value(token))
        dv=.15*torch.tanh(raw)
        gm=d['gm']
        dv=dv-(dv*gm[:,None]).sum(0)/gm.sum().clamp_min(1e-15)
        return dv*(gm>1e-15)[:,None]

def attach_corrector(solver,model,object_id,material_id):
    import warp as wp
    if int(solver.mpm_model.material)!=0:
        raise ValueError("This validation prototype supports elastic jelly only")
    if len(solver.particle_velocity_modifiers) or bool(wp.to_torch(solver.mpm_state.particle_selection).any()):
        raise ValueError("Selected/kinematic particles require a separately verified training bridge")
    model.eval()
    @torch.no_grad()
    def apply(s,dt):
        data=snapshot(s,object_id,material_id,dt)
        dv=model(data)
        wp.to_torch(s.mpm_state.grid_v_out).reshape(-1,3).index_add_(0,data['ids'],dv)
    solver.grid_correction_step=apply


class WorldParticlePhysCoreCorrector(nn.Module):
    """Original WorldParticle core with particle-state inputs and a physical P2G output adapter."""
    worldparticle_core = True

    def __init__(self, source_root, n_materials, max_delta_v=.002,
                 width=32, merge_iterations=3, decoder_layers=2):
        super().__init__()
        root = Path(source_root).resolve(strict=True)
        if (root.name != 'WorldParticle' or not (root/'models/particle_network_cross_attn_feat.py').is_file()
                or width < 8 or width % 8 or min(n_materials, merge_iterations, decoder_layers) < 1
                or not math.isfinite(max_delta_v) or max_delta_v <= 0):
            raise ValueError('Require original WorldParticle source, width divisible by 8, positive dimensions and velocity bound')
        for path in (root.parent, root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        os.environ.setdefault('ATTN_IMPL', 'sdpa')
        from WorldParticle.models.particle_network_cross_attn_feat import ParticleNetworkCrossAttnLocalFeature
        if Path(inspect.getfile(ParticleNetworkCrossAttnLocalFeature)).resolve() != root/'models/particle_network_cross_attn_feat.py':
            raise RuntimeError('WorldParticle resolved to a different checkout')
        self.source_root, self.merge_iterations = root, int(merge_iterations)
        self.max_delta_v = float(max_delta_v)
        self.material_embedding = nn.Embedding(n_materials, 4)
        self.backbone = ParticleNetworkCrossAttnLocalFeature(
            cconv_embedding_dim=width, particle_radius=.01, radius_scale=1.5,
            data='real_world', use_topology_conv=False, other_feats_channels=32,
            obstacle_feats_channels=3, slot_attn_iters=merge_iterations, slot_attn_heads=4,
            particle_position_rope_dim=width*3//4, num_decoder_layers=decoder_layers,
            decoder_attn_heads=4, decoder_attn_ffn_hidden_dim=width*6,
            decoder_attn_dropout=0., output_hidden_dim=width*2, output_layers=2,
            gravity=(0., 0., 0.), velhead=False)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self._boundary_cache = {}

    @property
    def head(self):
        return self.backbone.output_network[-1]

    def provenance(self):
        names = ('models/particle_network_cross_attn_feat.py', 'models/continuous_conv.py',
                 'models/super_particle_layers.py', 'models/cross_attn_feat.py',
                 'models/layers/attention.py', 'models/layers/merge.py', 'models/encodings/rope.py', 'TrajDataset.py')
        return dict(core_class=f'{type(self.backbone).__module__}.{type(self.backbone).__name__}',
            source_files={str(self.source_root/name): hashlib.sha256((self.source_root/name).read_bytes()).hexdigest() for name in names},
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            attention_backend=os.environ['ATTN_IMPL'],
            pipeline='original CConv -> k-partite merging -> RoPE -> SuperParticleDecoder -> original output head',
            physical_features='F-I, dt*C, log(E/2000), nu, log(rho/1000), log(dt/.001), log(volume/1e-6), Neo-Hookean flag, gravity/9.8, sticky flag, material embedding(4)',
            correction='original position correction / dt, bounded particle velocity residual, mass-weighted P2G, no momentum projection',
            boundary='actual axis-aligned source planes sampled at simulation dx',
            predictor='MPM supplies the physics; compute_correction bypasses WorldParticle gravity/integration',
            topology='disabled: numerical particle tracks do not supply mesh topology')

    def _boundaries(self, physics, x):
        key = json.dumps((physics['n_grid'], physics['boundary_conditions']), sort_keys=True)
        if key not in self._boundary_cache:
            grid = int(physics['n_grid'])
            axis = torch.arange(grid + 1, device=x.device, dtype=x.dtype) / grid
            uv = torch.cartesian_prod(axis, axis)
            points, normals = [], []
            for bc in physics['boundary_conditions']:
                # ponytail: current GT uses axis-aligned planes; general planes need a sampled tangent basis.
                normal = x.new_tensor(bc['normal'])
                if torch.count_nonzero(normal) != 1 or normal.abs().max() != 1:
                    raise ValueError('WorldParticle boundary adapter requires axis-aligned unit normals')
                k = int(normal.abs().argmax())
                surface = x.new_empty((len(uv), 3))
                surface[:, k] = float(bc['point'][k])
                surface[:, [i for i in range(3) if i != k]] = uv
                points.append(surface); normals.append(normal.expand(len(uv), 3))
            self._boundary_cache[key] = (torch.cat(points), torch.cat(normals))
        return self._boundary_cache[key]

    def forward(self, x, v, F, C, log_E, nu, grid_m, physics, material_id):
        # ponytail: one scene per call, batches use gradient accumulation; original batch_info supports future batched calls.
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 2**self.merge_iterations:
            raise ValueError('This adapter requires one complete scene per call and enough particles for merging')
        dt, volume = float(physics['dt']), float(physics['volume'])
        if dt <= 0 or volume <= 0 or material_id.shape != (x.shape[1],):
            raise ValueError('Invalid physical step/volume or material labels')
        positions, velocities = x[0], v[0]
        n = len(positions)
        # Source radii follow the supplied rest particle volume; units are never silently rescaled.
        self.backbone.apply_radius_override(radius_scale=1.5, particle_radius=volume**(1/3)/2)
        params = positions.new_tensor([math.log(dt/.001), math.log(volume/1e-6),
            float(physics.get('elasticity_model') == 'NeoHookeanElasticity'),
            *(g/9.8 for g in physics['gravity']), float(any(bc['surface'] == 'sticky' for bc in physics['boundary_conditions']))])
        features = torch.cat(((F[0]-torch.eye(3, device=x.device)).flatten(1), (dt*C[0]).flatten(1),
            (log_E[0]-math.log(2000.))[:, None], nu[0, :, None],
            positions.new_full((n, 1), math.log(physics['density']/1000.)),
            params.expand(n, -1), self.material_embedding(material_id)), dim=1)
        boundary, normals = self._boundaries(physics, positions)
        delta_x, _ = self.backbone.compute_correction(positions, velocities, features, boundary, normals, k_partite=2)
        particle_dv = self.max_delta_v * torch.tanh(delta_x / (dt*self.max_delta_v))
        projection = support(positions, int(physics['n_grid']), 1.)
        weighted = projection['weight'] * float(physics['mass'])
        contribution = (weighted[..., None] * particle_dv[:, None, :]).reshape(-1, 3)
        index = projection['ids'][projection['inv']].reshape(-1)
        grid_dv = torch.zeros_like(grid_m).unsqueeze(-1).expand(-1, 3).clone().index_add(0, index, contribution)
        grid_dv = grid_dv / grid_m.clamp_min(1e-15)[:, None]
        return grid_dv * (grid_m > 1e-15)[:, None]
