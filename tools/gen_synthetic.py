#!/usr/bin/env python3
"""Render synthetic orthographic sprite views of a known 3D scene (ground truth for tests).

usage: gen_synthetic.py PITCH_DEG YAW1,YAW2,... [WOBBLE_PX]   (writes v0.png, v1.png, ... to the cwd)

The camera convention is the one of src/model.rs. The scene is a little "character on a
carpet" with two separate legs, flat-coloured parts (markers) and an asymmetric arm. Views
are cropped to the silhouette bounding box like a sprite sheet (anchors unknown to the tool).
WOBBLE_PX adds a smooth random warp of up to that many pixels to every view (hand-drawn
inaccuracy).
"""
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, zoom

rng = np.random.default_rng(7)
N = 130
ax = np.arange(N) - N / 2 + 0.5
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
occ = np.zeros((N, N, N), bool)
col = np.zeros((N, N, N, 3), np.uint8)


def put(mask, color):
    occ[mask] = True
    col[mask] = color


th = np.radians(20)
xr = X * np.cos(th) + Z * np.sin(th)
zr = -X * np.sin(th) + Z * np.cos(th)
carpet = (abs(xr) < 42) & (abs(zr) < 28) & (Y > -22) & (Y < -18)
put(carpet, (120, 80, 50))
put(carpet & ((abs(xr) > 37) | (abs(zr) > 23)), (110, 110, 170))          # border line
for sx in (-7, 7):                                                             # two legs
    put(((X - sx) ** 2 + (Z + 3) ** 2 < 5.5 ** 2) & (Y > -18) & (Y < 7), (140, 150, 205))
put(((X / 10.5) ** 2 + ((Z + 3) / 7.5) ** 2 < 1) & (Y >= 7) & (Y < 27), (210, 30, 30))  # vest
put(((X / 10.5) ** 2 + ((Z + 3) / 7.5) ** 2 < 1) & (Y >= 7) & (Y < 10), (240, 240, 240))  # belt
put((X ** 2 + (Y - 34) ** 2 + (Z + 3) ** 2) < 7 ** 2, (210, 170, 140))         # head
put((X ** 2 + (Y - 38) ** 2 + (Z + 3) ** 2) < 8 ** 2, (200, 210, 240))         # hat
# arms: the left one stretches far forward, the right one is short (asymmetry)
for k in range(0, 26):
    t = k / 25.0
    cx, cy, cz = 11 + 14 * t, 22 - 4 * t, -3 + 22 * t
    put((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 < 3.2 ** 2, (40, 150, 60) if t > 0.8 else (210, 170, 140))
for k in range(0, 12):
    t = k / 11.0
    cx, cy, cz = -11 - 6 * t, 22 - 8 * t, -3 + 6 * t
    put((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 < 3.2 ** 2, (40, 150, 60) if t > 0.7 else (210, 170, 140))
# sash flying behind the character
put((abs(Y - 12 - 0.15 * (Z + 3)) < 1.2) & (abs(X + 14 + 0.3 * (Z + 3)) < 8) & (Z < -3) & (Z > -26), (245, 245, 245))

pitch = float(sys.argv[1])
yaws = [float(a) for a in sys.argv[2].split(",")]
wobble = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
pts = np.stack([X[occ], Y[occ], Z[occ]], 1)
cols = col[occ]


def render(yaw, pitch):
    ty, tp = np.radians(yaw), np.radians(pitch)
    r = np.array([np.cos(ty), 0, -np.sin(ty)])
    c = np.array([np.sin(ty) * np.cos(tp), np.sin(tp), np.cos(ty) * np.cos(tp)])
    u = np.cross(c, r)
    a, b, t = pts @ r, pts @ u, pts @ c
    S = 300
    img = np.zeros((S, S, 4), np.uint8)
    ii = np.round(a + S / 2).astype(int)
    jj = np.round(S / 2 - b).astype(int)
    for k in np.argsort(t):
        for di in (0, 1):
            for dj in (0, 1):
                img[jj[k] + dj, ii[k] + di, :3] = cols[k]
                img[jj[k] + dj, ii[k] + di, 3] = 255
    return img


def wobble_img(img, amp):
    if amp <= 0:
        return img
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    dx = zoom(rng.uniform(-amp, amp, (4, 4)), (h / 4, w / 4), order=1)[:h, :w]
    dy = zoom(rng.uniform(-amp, amp, (4, 4)), (h / 4, w / 4), order=1)[:h, :w]
    yy, xx = np.mgrid[0:h, 0:w]
    for ch in range(4):
        out[..., ch] = map_coordinates(img[..., ch], [yy + dy, xx + dx], order=0, mode="constant")
    return out


for n, yw in enumerate(yaws):
    im = wobble_img(render(yw, pitch), wobble)
    ys, xs = np.where(im[..., 3] > 0)
    crop = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    Image.fromarray(crop, "RGBA").save(f"v{n}.png")
    print(n, yw, crop.shape)
