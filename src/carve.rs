//! Final reconstruction at the requested resolution: most probable occupancy from the
//! summed evidence, removal of floating fragments, colour transfer and debug overlays.

use crate::estimate::Solution;
use crate::exchange::Prior;
use crate::model::*;
use crate::silhouette::{View, LEVELS};
use rayon::prelude::*;

pub struct VoxelModel {
    pub grid: Grid,
    pub occ: Vec<bool>,
    pub color: Vec<[u8; 3]>,
}

impl VoxelModel {
    pub fn count(&self) -> usize {
        self.occ.iter().filter(|&&o| o).count()
    }
}

/// `voxel`: voxel edge in world units (1 world unit = 1 pixel of the reference scale).
/// `fine_y`: optional range of the model height (fractions 0..1 from the bottom) that is
/// evaluated with half the voxel size (2x resolution), e.g. the legs.
/// `prior`: optional occupancy prior (e.g. from a 3D model) and its weight, added to the
/// silhouette evidence before the decision (see `exchange::Prior`).
pub fn reconstruct(views: &[View], sol: &Solution, voxel: Option<f32>, resolution: usize, colors: bool, fine_y: Option<(f32, f32)>, repair: bool, prior: Option<(&Prior, f32)>) -> VoxelModel {
    let (lo, hi) = sol.bounds;
    let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let longest = ext.iter().cloned().fold(0.0, f32::max);
    // never go finer than one source pixel: there is no more information
    let v = voxel.unwrap_or_else(|| (longest / resolution as f32).max(1.0));
    let n = [
        ((ext[0] / v).ceil() as usize).max(1),
        ((ext[1] / v).ceil() as usize).max(1),
        ((ext[2] / v).ceil() as usize).max(1),
    ];
    let grid = Grid { n, v, o: lo };

    // summed evidence, parallel over z-slabs
    let slab = n[0] * n[1];
    let mut total = vec![0f32; grid.len()];
    let cams: Vec<Cam> = (0..views.len()).map(|k| sol.cam(k)).collect();
    total.par_chunks_mut(slab).enumerate().for_each(|(iz, chunk)| {
        for (k, view) in views.iter().enumerate() {
            llr_slab(view, LEVELS - 1, &cams[k], &grid, iz, &sol.noise, chunk, true);
        }
    });
    if let Some((p, w)) = prior {
        p.add_to(&grid, &mut total, w, 0.5);
    }
    let mut occ: Vec<bool> = total.iter().map(|&t| t > sol.noise.thr).collect();
    let mut grid = grid;
    if let Some((a, b)) = fine_y {
        let (g2, o2, t2) = refine_region(views, &cams, sol, &grid, &occ, a, b, prior);
        grid = g2;
        occ = o2;
        total = t2;
    }
    // the strict intersection first loses floating fragments, then the repair brings back
    // what the silhouettes demand (the repaired voxels are kept even when thin / detached)
    keep_largest_component(&mut occ, &grid);
    close_holes(&mut occ, &grid, 1);
    if repair {
        let margin: f32 = std::env::var("VOX_REPAIR_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(6.0);
        let added = repair_coverage(views, &cams, &grid, &total, &mut occ, sol.noise.thr, margin, 2);
        if added > 0 {
            eprintln!("coverage repair: {added} voxels added so that every silhouette pixel is explained");
        }
    }

    let color = if colors { transfer_colors(views, &cams, &grid, &occ) } else { vec![[200, 200, 200]; grid.len()] };
    VoxelModel { grid, occ, color }
}

/// Re-evaluates the evidence with half the voxel size inside a height range; elsewhere the
/// coarse decision is simply copied to the finer lattice.
fn refine_region(views: &[View], cams: &[Cam], sol: &Solution, g: &Grid, occ: &[bool], a: f32, b: f32, prior: Option<(&Prior, f32)>) -> (Grid, Vec<bool>, Vec<f32>) {
    let [nx, ny, nz] = g.n;
    let (mut ylo, mut yhi) = (usize::MAX, 0usize);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if occ[g.idx(x, y, z)] {
                    ylo = ylo.min(y);
                    yhi = yhi.max(y);
                }
            }
        }
    }
    let g2 = Grid { n: [nx * 2, ny * 2, nz * 2], v: g.v * 0.5, o: g.o };
    if ylo == usize::MAX {
        return (g2, vec![false; g2.len()], vec![0.0; g2.len()]);
    }
    let slab = g2.n[0] * g2.n[1];
    let mut total = vec![0f32; g2.len()];
    total.par_chunks_mut(slab).enumerate().for_each(|(iz, chunk)| {
        for (k, view) in views.iter().enumerate() {
            llr_slab(view, LEVELS - 1, &cams[k], &g2, iz, &sol.noise, chunk, true);
        }
    });
    if let Some((p, w)) = prior {
        p.add_to(&g2, &mut total, w, 0.5);
    }
    let span = (yhi - ylo + 1) as f32;
    let mut out = vec![false; g2.len()];
    for z in 0..g2.n[2] {
        for y in 0..g2.n[1] {
            let frac = ((y / 2) as f32 + 0.5 - ylo as f32) / span;
            let fine = frac >= a && frac <= b;
            for x in 0..g2.n[0] {
                let i = g2.idx(x, y, z);
                out[i] = if fine { total[i] > sol.noise.thr } else { occ[g.idx(x / 2, y / 2, z / 2)] };
            }
        }
    }
    (g2, out, total)
}

/// Silhouette of an occupancy grid in one view (image sized).
fn occ_mask(grid: &Grid, occ: &[bool], cam: &Cam, w: usize, h: usize) -> Vec<bool> {
    let b = basis(cam.yaw, cam.pitch, cam.vs);
    let mut out = vec![false; w * h];
    let r = (grid.v * cam.scale * 0.55).max(0.6);
    for z in 0..grid.n[2] {
        for y in 0..grid.n[1] {
            for x in 0..grid.n[0] {
                if !occ[grid.idx(x, y, z)] {
                    continue;
                }
                let [px, py] = cam.project(&b, grid.center(x, y, z));
                for yy in (py - r).floor() as i32..(py + r).ceil() as i32 {
                    for xx in (px - r).floor() as i32..(px + r).ceil() as i32 {
                        if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                            out[yy as usize * w + xx as usize] = true;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Every pixel of an input silhouette that the model does not explain gets the most
/// probable voxel along its viewing ray (highest evidence, but not worse than `margin`
/// below the occupancy threshold). This brings back thin parts (arms, cloth, tips) that
/// the strict intersection of all silhouettes loses because of small drawing errors.
fn repair_coverage(views: &[View], cams: &[Cam], g: &Grid, total: &[f32], occ: &mut [bool], thr: f32, margin: f32, rounds: usize) -> usize {
    let mut added = 0usize;
    // only voxels close to the existing model qualify: no ghosts floating at a wrong depth
    let near = dilate(occ, g, 3);
    let diag = ((g.n[0] * g.n[0] + g.n[1] * g.n[1] + g.n[2] * g.n[2]) as f32).sqrt() * g.v;
    let centre = [g.o[0] + g.n[0] as f32 * g.v * 0.5, g.o[1] + g.n[1] as f32 * g.v * 0.5, g.o[2] + g.n[2] as f32 * g.v * 0.5];
    for _ in 0..rounds {
        for (k, view) in views.iter().enumerate() {
            let cam = &cams[k];
            let b = basis(cam.yaw, cam.pitch, cam.vs);
            let model = occ_mask(g, occ, cam, view.w, view.h);
            let mut todo: Vec<(f32, f32)> = Vec::new();
            for y in 0..view.h {
                for x in 0..view.w {
                    let i = y * view.w + x;
                    if view.sd[i] > 0.5 && !model[i] {
                        todo.push((x as f32 + 0.5, y as f32 + 0.5));
                    }
                }
            }
            // ray: p = a*r + (b/vs)*u0 + t*c  with u0 the unstretched up vector
            let found: Vec<Option<usize>> = todo
                .par_iter()
                .map(|&(px, py)| {
                    let a = (px - cam.off[0]) / cam.scale;
                    let bb = (cam.off[1] - py) / cam.scale / cam.vs;
                    let u0 = [b.u[0] / cam.vs, b.u[1] / cam.vs, b.u[2] / cam.vs];
                    let p0 = [a * b.r[0] + bb * u0[0], a * b.r[1] + bb * u0[1], a * b.r[2] + bb * u0[2]];
                    // start at the point of the ray closest to the grid centre
                    let t0 = (centre[0] - p0[0]) * b.c[0] + (centre[1] - p0[1]) * b.c[1] + (centre[2] - p0[2]) * b.c[2];
                    let step = g.v * 0.5;
                    let n = (diag / step) as i32;
                    let (mut best, mut best_i) = (f32::NEG_INFINITY, None);
                    for s in -n..=n {
                        let t = t0 + s as f32 * step;
                        let p = [p0[0] + t * b.c[0], p0[1] + t * b.c[1], p0[2] + t * b.c[2]];
                        let (fx, fy, fz) = ((p[0] - g.o[0]) / g.v, (p[1] - g.o[1]) / g.v, (p[2] - g.o[2]) / g.v);
                        if fx < 0.0 || fy < 0.0 || fz < 0.0 {
                            continue;
                        }
                        let (ix, iy, iz) = (fx as usize, fy as usize, fz as usize);
                        if ix >= g.n[0] || iy >= g.n[1] || iz >= g.n[2] {
                            continue;
                        }
                        let idx = g.idx(ix, iy, iz);
                        if near[idx] && total[idx] > best {
                            best = total[idx];
                            best_i = Some(idx);
                        }
                    }
                    if best > thr - margin {
                        best_i
                    } else {
                        None
                    }
                })
                .collect();
            for f in found.into_iter().flatten() {
                if !occ[f] {
                    occ[f] = true;
                    added += 1;
                }
            }
        }
    }
    added
}

/// Cube dilation (max filter) of radius `r` voxels, separable.
fn dilate(occ: &[bool], g: &Grid, r: usize) -> Vec<bool> {
    let [nx, ny, nz] = g.n;
    let mut cur = occ.to_vec();
    for axis in 0..3 {
        let mut next = vec![false; cur.len()];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let c = [x, y, z];
                    let n = g.n[axis];
                    let lo = c[axis].saturating_sub(r);
                    let hi = (c[axis] + r).min(n - 1);
                    let mut v = false;
                    for t in lo..=hi {
                        let mut q = c;
                        q[axis] = t;
                        if cur[g.idx(q[0], q[1], q[2])] {
                            v = true;
                            break;
                        }
                    }
                    next[g.idx(x, y, z)] = v;
                }
            }
        }
        cur = next;
    }
    cur
}

fn erode(occ: &[bool], g: &Grid, r: usize) -> Vec<bool> {
    let inv: Vec<bool> = occ.iter().map(|&o| !o).collect();
    dilate(&inv, g, r).into_iter().map(|d| !d).collect()
}

/// Morphological closing: small holes (pixel-level misregistration) are filled, gaps wider
/// than 2r voxels (between the legs) stay open.
fn close_holes(occ: &mut [bool], g: &Grid, r: usize) {
    let d = dilate(occ, g, r);
    let c = erode(&d, g, r);
    for i in 0..occ.len() {
        occ[i] = occ[i] || c[i];
    }
}

fn keep_largest_component(occ: &mut [bool], g: &Grid) {
    let [nx, ny, nz] = g.n;
    let mut label = vec![0u32; occ.len()];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    for s in 0..occ.len() {
        if !occ[s] || label[s] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        sizes.push(0);
        label[s] = id;
        stack.push(s);
        while let Some(i) = stack.pop() {
            sizes[id as usize] += 1;
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            let mut visit = |j: usize| {
                if occ[j] && label[j] == 0 {
                    label[j] = id;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < nx {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - nx);
            }
            if y + 1 < ny {
                visit(i + nx);
            }
            if z > 0 {
                visit(i - nx * ny);
            }
            if z + 1 < nz {
                visit(i + nx * ny);
            }
        }
    }
    if sizes.len() <= 2 {
        return;
    }
    let best = (1..sizes.len()).max_by_key(|&i| sizes[i]).unwrap() as u32;
    for i in 0..occ.len() {
        if occ[i] && label[i] != best {
            occ[i] = false;
        }
    }
}

/// Estimated outward normal of a surface voxel from its empty neighbours.
fn surface_normal(occ: &[bool], g: &Grid, x: usize, y: usize, z: usize) -> Option<[f32; 3]> {
    let [nx, ny, nz] = g.n;
    let mut n = [0f32; 3];
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let (X, Y, Z) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                let empty = X < 0 || Y < 0 || Z < 0 || X >= nx as i32 || Y >= ny as i32 || Z >= nz as i32 || !occ[g.idx(X as usize, Y as usize, Z as usize)];
                if empty {
                    n[0] += dx as f32;
                    n[1] += dy as f32;
                    n[2] += dz as f32;
                }
            }
        }
    }
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l < 1e-3 {
        None
    } else {
        Some([n[0] / l, n[1] / l, n[2] / l])
    }
}

fn transfer_colors(views: &[View], cams: &[Cam], g: &Grid, occ: &[bool]) -> Vec<[u8; 3]> {
    let [nx, ny, nz] = g.n;
    let v = g.v;
    let bases: Vec<Basis> = cams.iter().map(|c| basis(c.yaw, c.pitch, c.vs)).collect();

    // depth buffers (nearest surface towards each camera) in the view plane
    let rn = {
        let e = [nx as f32 * v, ny as f32 * v, nz as f32 * v];
        let c = [g.o[0] + e[0] * 0.5, g.o[1] + e[1] * 0.5, g.o[2] + e[2] * 0.5];
        let r = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt() * 0.5;
        let cn = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        (r + cn) * 1.6
    };
    let s = (2.0 * rn / v).ceil() as usize + 3;
    let half = s as f32 * 0.5;
    let mut depth = vec![f32::NEG_INFINITY; views.len() * s * s];
    let cell = |b: &Basis, p: [f32; 3]| -> Option<(usize, usize, f32)> {
        let a = p[0] * b.r[0] + p[1] * b.r[1] + p[2] * b.r[2];
        let u = p[0] * b.u[0] + p[1] * b.u[1] + p[2] * b.u[2];
        let t = p[0] * b.c[0] + p[1] * b.c[1] + p[2] * b.c[2];
        let i = (a / v + half).floor();
        let j = (half - u / v).floor();
        if i < 0.0 || j < 0.0 || i >= s as f32 || j >= s as f32 {
            None
        } else {
            Some((i as usize, j as usize, t))
        }
    };
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if !occ[g.idx(x, y, z)] {
                    continue;
                }
                let p = g.center(x, y, z);
                for (k, b) in bases.iter().enumerate() {
                    if let Some((i, j, t)) = cell(b, p) {
                        let d = &mut depth[k * s * s + j * s + i];
                        if t > *d {
                            *d = t;
                        }
                    }
                }
            }
        }
    }

    let mut color = vec![[0u8; 3]; g.len()];
    let mut have = vec![false; g.len()];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = g.idx(x, y, z);
                if !occ[idx] {
                    continue;
                }
                let normal = surface_normal(occ, g, x, y, z);
                let p = g.center(x, y, z);
                let (mut wsum, mut acc) = (0.0f32, [0.0f32; 3]);
                for (k, b) in bases.iter().enumerate() {
                    let Some((i, j, t)) = cell(b, p) else { continue };
                    if t + 1.5 * v < depth[k * s * s + j * s + i] {
                        continue; // hidden behind something nearer
                    }
                    let [px, py] = cams[k].project(b, p);
                    let sd = views[k].sample_sd(px, py);
                    if sd < 0.5 {
                        continue;
                    }
                    let edge = ((sd - 0.5) / 2.0).clamp(0.0, 1.0);
                    let facing = match normal {
                        Some(n) => (n[0] * b.c[0] + n[1] * b.c[1] + n[2] * b.c[2]).max(0.0),
                        None => 1.0,
                    };
                    let w = edge * (facing.powi(12) + 1e-4);
                    let c = views[k].color_at(px, py);
                    for d in 0..3 {
                        acc[d] += w * c[d] as f32;
                    }
                    wsum += w;
                }
                if wsum > 1e-6 {
                    color[idx] = [(acc[0] / wsum) as u8, (acc[1] / wsum) as u8, (acc[2] / wsum) as u8];
                    have[idx] = true;
                }
            }
        }
    }

    // voxels seen by no view: spread colours from the nearest coloured voxels
    let mut queue: std::collections::VecDeque<usize> = (0..g.len()).filter(|&i| occ[i] && have[i]).collect();
    if queue.is_empty() {
        return vec![[200, 200, 200]; g.len()];
    }
    while let Some(i) = queue.pop_front() {
        let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
        let mut nb = [usize::MAX; 6];
        if x > 0 {
            nb[0] = i - 1;
        }
        if x + 1 < nx {
            nb[1] = i + 1;
        }
        if y > 0 {
            nb[2] = i - nx;
        }
        if y + 1 < ny {
            nb[3] = i + nx;
        }
        if z > 0 {
            nb[4] = i - nx * ny;
        }
        if z + 1 < nz {
            nb[5] = i + nx * ny;
        }
        for &j in &nb {
            if j != usize::MAX && occ[j] && !have[j] {
                color[j] = color[i];
                have[j] = true;
                queue.push_back(j);
            }
        }
    }
    color
}

/// Silhouette of the reconstructed model as seen by view `k` (image-sized mask).
pub fn project_mask(model: &VoxelModel, view: &View, cam: &Cam) -> Vec<bool> {
    let b = basis(cam.yaw, cam.pitch, cam.vs);
    let g = &model.grid;
    let mut out = vec![false; view.w * view.h];
    let r = (g.v * cam.scale * 0.5).max(0.5);
    for z in 0..g.n[2] {
        for y in 0..g.n[1] {
            for x in 0..g.n[0] {
                if !model.occ[g.idx(x, y, z)] {
                    continue;
                }
                let [px, py] = cam.project(&b, g.center(x, y, z));
                let (x0, x1) = ((px - r).floor() as i32, (px + r).ceil() as i32);
                let (y0, y1) = ((py - r).floor() as i32, (py + r).ceil() as i32);
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        if xx >= 0 && yy >= 0 && (xx as usize) < view.w && (yy as usize) < view.h {
                            out[yy as usize * view.w + xx as usize] = true;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Fraction of (input silhouette ∪ model silhouette) that both share.
pub fn iou(a: &[bool], b: &[bool]) -> f32 {
    let (mut i, mut u) = (0usize, 0usize);
    for k in 0..a.len() {
        if a[k] && b[k] {
            i += 1;
        }
        if a[k] || b[k] {
            u += 1;
        }
    }
    if u == 0 {
        0.0
    } else {
        i as f32 / u as f32
    }
}
