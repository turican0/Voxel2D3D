//! Exchange with external tools: saved camera solutions, the export frame of the GLB and
//! an occupancy prior (e.g. voxelised from a 3D model of the same object).
//!
//! The loop this enables: reconstruct -> fit / correct a detailed 3D model against the
//! voxels -> voxelise that model into a prior -> reconstruct again *with the same cameras*
//! (`--solution`) and the prior added to the silhouette evidence. Where the views clearly
//! say "empty" the views win; where they cannot decide (concave regions that a visual hull
//! fills, thin parts lost by drawing errors) the model decides.

use crate::carve::VoxelModel;
use crate::estimate::{Pose, Solution};
use crate::model::{Grid, Noise};
use serde_json::{json, Value};
use std::path::Path;

pub fn save_solution(path: &Path, sol: &Solution, names: &[String]) -> Result<(), String> {
    let v = json!({
        "pitch": sol.pitch,
        "vscale": sol.vscale,
        "views": (0..sol.poses.len()).map(|k| json!({
            "name": names.get(k).cloned().unwrap_or_default(),
            "yaw": sol.poses[k].yaw,
            "shift": sol.poses[k].shift,
            "scale": sol.scales[k],
            "anchor": sol.anchors[k],
            "max_shift": sol.max_shift[k],
            "iou": sol.ious.get(k).copied().unwrap_or(0.0),
        })).collect::<Vec<_>>(),
        "score": sol.score,
        "bounds": [sol.bounds.0, sol.bounds.1],
        "noise": [sol.noise.a, sol.noise.b, sol.noise.thr],
    });
    std::fs::write(path, serde_json::to_string_pretty(&v).unwrap()).map_err(|e| format!("{}: {e}", path.display()))
}

fn f(v: &Value) -> Result<f32, String> {
    v.as_f64().map(|x| x as f32).ok_or_else(|| format!("expected a number, got {v}"))
}

fn f2(v: &Value) -> Result<[f32; 2], String> {
    Ok([f(&v[0])?, f(&v[1])?])
}

fn f3(v: &Value) -> Result<[f32; 3], String> {
    Ok([f(&v[0])?, f(&v[1])?, f(&v[2])?])
}

pub fn load_solution(path: &Path, n_views: usize) -> Result<Solution, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
    let views = v["views"].as_array().ok_or("solution: missing 'views'")?;
    if views.len() != n_views {
        return Err(format!("solution has {} views, {} images given", views.len(), n_views));
    }
    let mut sol = Solution {
        pitch: f(&v["pitch"])?,
        vscale: f(&v["vscale"])?,
        poses: vec![],
        scales: vec![],
        anchors: vec![],
        max_shift: vec![],
        score: f(&v["score"]).unwrap_or(0.0),
        ious: vec![],
        bounds: (f3(&v["bounds"][0])?, f3(&v["bounds"][1])?),
        noise: Noise { a: f(&v["noise"][0])?, b: f(&v["noise"][1])?, thr: f(&v["noise"][2])? },
    };
    for w in views {
        sol.poses.push(Pose { yaw: f(&w["yaw"])?, shift: f2(&w["shift"])? });
        sol.scales.push(f(&w["scale"])?);
        sol.anchors.push(f2(&w["anchor"])?);
        sol.max_shift.push(f(&w["max_shift"]).unwrap_or(0.0));
        sol.ious.push(f(&w["iou"]).unwrap_or(0.0));
    }
    Ok(sol)
}

/// Transform of the exported GLB: `out = rotY(rotate) * ((world - origin) * k)`.
/// `world` is the reconstruction space (1 unit = 1 source pixel, cameras of the solution).
pub struct Frame {
    pub origin: [f32; 3],
    pub k: f32,
    pub rotate_deg: f32,
}

pub fn export_frame(model: &VoxelModel, height: f32, rotate_deg: f32) -> Option<Frame> {
    let g = &model.grid;
    let [nx, ny, nz] = g.n;
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if model.occ[g.idx(x, y, z)] {
                    for (d, c) in [x, y, z].into_iter().enumerate() {
                        lo[d] = lo[d].min(c);
                        hi[d] = hi[d].max(c);
                    }
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        return None;
    }
    // must match mesh::build_mesh
    let k = height / ((hi[1] - lo[1] + 1) as f32 * g.v);
    let cx = (lo[0] + hi[0] + 1) as f32 * 0.5;
    let cz = (lo[2] + hi[2] + 1) as f32 * 0.5;
    let origin = [g.o[0] + cx * g.v, g.o[1] + lo[1] as f32 * g.v, g.o[2] + cz * g.v];
    Some(Frame { origin, k, rotate_deg })
}

pub fn save_frame(path: &Path, fr: &Frame, sol: &Solution, grid: &Grid) -> Result<(), String> {
    let v = json!({
        "about": "out = rotY(rotate_deg) * ((world - origin) * k); world = reconstruction space, 1 unit = 1 source pixel",
        "origin": fr.origin,
        "k": fr.k,
        "rotate_deg": fr.rotate_deg,
        "grid": { "n": grid.n, "voxel": grid.v, "min_corner": grid.o },
        "pitch": sol.pitch,
        "vscale": sol.vscale,
        "cameras": (0..sol.poses.len()).map(|k| {
            let c = sol.cam(k);
            json!({ "yaw": c.yaw, "scale": c.scale, "off": c.off })
        }).collect::<Vec<_>>(),
    });
    std::fs::write(path, serde_json::to_string_pretty(&v).unwrap()).map_err(|e| format!("{}: {e}", path.display()))
}

/// Occupancy probability on a regular lattice in world space.
/// File format (little endian): b"VXPR", u32 version (1), f32 origin[3] (centre of the
/// first cell), f32 cell size, u32 n[3], then n0*n1*n2 bytes, p = byte / 255,
/// index = x + n0 * (y + n1 * z).
pub struct Prior {
    pub origin: [f32; 3],
    pub cell: f32,
    pub n: [usize; 3],
    pub p: Vec<u8>,
}

impl Prior {
    pub fn load(path: &Path) -> Result<Prior, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if b.len() < 36 || &b[0..4] != b"VXPR" {
            return Err(format!("{}: not a VXPR prior file", path.display()));
        }
        let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let fl = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let origin = [fl(8), fl(12), fl(16)];
        let cell = fl(20);
        let n = [u(24) as usize, u(28) as usize, u(32) as usize];
        let len = n[0] * n[1] * n[2];
        if b.len() < 36 + len {
            return Err(format!("{}: truncated prior ({} of {} cells)", path.display(), b.len() - 36, len));
        }
        Ok(Prior { origin, cell, n, p: b[36..36 + len].to_vec() })
    }

    /// Probability at a world point (nearest cell, trilinear would blur thin parts);
    /// `None` outside the lattice.
    pub fn sample(&self, w: [f32; 3]) -> Option<f32> {
        let mut i = [0usize; 3];
        for d in 0..3 {
            let t = ((w[d] - self.origin[d]) / self.cell).round();
            if t < 0.0 || t >= self.n[d] as f32 {
                return None;
            }
            i[d] = t as usize;
        }
        Some(self.p[i[0] + self.n[0] * (i[1] + self.n[1] * i[2])] as f32 / 255.0)
    }

    /// Adds `weight * logit(p)` (clamped) to the evidence of every grid cell.
    /// Outside the lattice the prior is `outside` (0.5 = neutral: the model may cover only a
    /// part of the scene, e.g. a figure without the carpet it stands on).
    pub fn add_to(&self, grid: &Grid, total: &mut [f32], weight: f32, outside: f32) {
        let logit = |p: f32| {
            let p = p.clamp(0.02, 0.98);
            (p / (1.0 - p)).ln()
        };
        for z in 0..grid.n[2] {
            for y in 0..grid.n[1] {
                for x in 0..grid.n[0] {
                    let p = self.sample(grid.center(x, y, z)).unwrap_or(outside);
                    total[grid.idx(x, y, z)] += weight * logit(p);
                }
            }
        }
    }
}
