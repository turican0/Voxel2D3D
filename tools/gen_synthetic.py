#!/usr/bin/env python3
"""Render synthetic orthographic views of a known 3D scene (ground truth for testing).

usage: gen_synthetic.py PITCH_DEG YAW1,YAW2,...   (writes v0.png, v1.png, ... to the cwd)
Camera convention is the same as in src/model.rs. Views are cropped to the silhouette
bounding box, exactly like sprite sheets, so the image anchors are unknown to the tool.
"""
import numpy as np, sys, json
from PIL import Image
rng=np.random.default_rng(1)
# ground-truth voxel scene in world units (px), grid 1 unit
N=120
ax=np.arange(N)-N/2+0.5
X,Y,Z=np.meshgrid(ax,ax,ax,indexing='ij')   # x,y,z
occ=np.zeros((N,N,N),bool)
# carpet: flat plate  (x: +-45, z: +-30, y: -22..-18) slightly rotated
th=np.radians(20); xr=X*np.cos(th)+Z*np.sin(th); zr=-X*np.sin(th)+Z*np.cos(th)
occ|=(abs(xr)<42)&(abs(zr)<28)&(Y>-22)&(Y<-18)
# legs / body
occ|=((X-0)**2+(Z+5)**2<6**2)&(Y>-18)&(Y<5)
occ|=((X-0)**2+(Z+5)**2<11**2)&(Y>=5)&(Y<25)
# head sphere
occ|=(X**2+(Y-33)**2+(Z+5)**2)<8**2
# one arm sticking out in +x+z direction (asymmetry)
occ|=(abs(Y-15)<3)&(X>8)&(X<30)&(abs(Z-8-0.3*(X-8))<3)
# nose marker -z side
occ|=((X-0)**2+(Y-33)**2+(Z+16)**2)<4**2
col=np.zeros((N,N,N,3),np.uint8)
col[...]=(200,60,60)
col[(Y<-17)]=(120,80,50)
col[(Y>26)]=(210,180,150)
# 3D texture (stripes/patches) so that colour is a property of the surface point
tex=(np.sin(X*0.35)+np.sin(Z*0.31+1)+np.sin(Y*0.4+2))
col=np.clip(col.astype(int)+ (tex[...,None]*[28,-20,24]).astype(int),0,255).astype(np.uint8)
pitch=float(sys.argv[1]); yaws=[float(a) for a in sys.argv[2].split(',')]
def render(yaw,pitch):
    ty,tp=np.radians(yaw),np.radians(pitch)
    r=np.array([np.cos(ty),0,-np.sin(ty)])
    c=np.array([np.sin(ty)*np.cos(tp),np.sin(tp),np.cos(ty)*np.cos(tp)])
    u=np.cross(c,r)
    pts=np.stack([X[occ],Y[occ],Z[occ]],1)
    a=pts@r; b=pts@u; t=pts@c
    S=260
    img=np.zeros((S,S,4),np.uint8); zb=np.full((S,S),-1e9)
    ii=np.round(a+S/2).astype(int); jj=np.round(S/2-b).astype(int)
    cols=col[occ]
    order=np.argsort(t)
    for k in order:
        for di in (0,1):
            for dj in (0,1):
                i=ii[k]+di; j=jj[k]+dj
                img[j,i,:3]=cols[k]; img[j,i,3]=255
    return img
for n,yw in enumerate(yaws):
    im=render(yw,pitch)
    a=im[...,3]>0
    ys,xs=np.where(a); 
    # tight crop, plus random hand-drawn wobble: shift edge pixels a bit
    crop=im[ys.min():ys.max()+1, xs.min():xs.max()+1]
    Image.fromarray(crop,'RGBA').save(f'v{n}.png')
    print(n,yw,crop.shape)
