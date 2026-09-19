import argparse, json, numpy as np, trimesh
from pathlib import Path
from plyfile import PlyData, PlyElement

def main():
 p=argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('out'); a=p.parse_args()
 rows=json.load(open(a.manifest)); xyz=[]; rgb=[]; ids=[]
 for row in rows:
  m=trimesh.load(row['path'], force='mesh'); v=np.asarray(m.vertices,np.float32); c=np.full((len(v),3),128,np.uint8)
  if hasattr(m.visual,'vertex_colors') and len(m.visual.vertex_colors)==len(v): c=np.asarray(m.visual.vertex_colors[:,:3],np.uint8)
  xyz.append(v); rgb.append(c); ids.extend([row['object_id']]*len(v))
 xyz=np.concatenate(xyz); rgb=np.concatenate(rgb); n=len(xyz); scale=np.full(n,-5.,np.float32)
 arr=np.empty(n,dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4')]+[(f'f_rest_{i}','f4') for i in range(45)]+[('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')])
 for name in arr.dtype.names: arr[name]=0
 arr['x'],arr['y'],arr['z']=xyz.T; arr['f_dc_0']=rgb[:,0]/255.; arr['f_dc_1']=rgb[:,1]/255.; arr['f_dc_2']=rgb[:,2]/255.; arr['opacity']=2.; arr['scale_0']=arr['scale_1']=arr['scale_2']=scale; arr['rot_0']=1.
 PlyData([PlyElement.describe(arr,'vertex')],text=True).write(a.out); np.save(str(Path(a.out).with_suffix('.object_id.npy')),np.asarray(ids,np.int64)); print(n,a.out)
if __name__=='__main__': main()
