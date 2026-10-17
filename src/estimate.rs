//! Estimation of the unknown camera parameters.
//!
//! Unknowns: one yaw per image, one shared pitch, and one 2D shift per image (the
//! images are usually cropped, so the pixel position of the rotation axis is not
//! known; the nominal position is the silhouette bounding-box centre or a value from
//! `--anchors`, the shift is bounded by `max_shift` and penalised quadratically).
//!
//! Objective (maximised):
//!
//! ```text
//! score = mean_k IoU( project(hull, view k), silhouette k )
//!         - color_weight * (colour spread of the hull surface seen from different views)
//!         - shift_penalty * mean_k (|shift_k| / max_shift_k)^2
//!         - small prior pulling yaws towards the even-spacing guess
//! ```
//!
//! where `hull` is the most probable occupancy given all views (see `model::Noise`).
//! The optimiser is a coordinate descent (yaw scan -> shift pattern search -> pitch
//! scan) run coarse-to-fine in grid resolution and in silhouette softness, started
//! from several yaw layouts (full circle / half circle, both rotation directions).

use crate::model::*;
use crate::silhouette::View;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Config {
    pub e: f32,
    pub q: f32,
    pub prior: f32,
    pub pitch: Option<f32>,
    pub pitch_min: f32,
    pub pitch_max: f32,
    /// vertical stretch of the image plane (None = estimated)
    pub vscale: Option<f32>,
    pub yaw_hints: Vec<Option<f32>>,
    pub anchors: Vec<Option<[f32; 2]>>,
    pub max_shift: Option<f32>,
    pub shift_penalty: f32,
    pub color_weight: f32,
    pub normalize_height: bool,
    /// Assume the images are ordered and evenly spaced in angle (yaw_k = k * step),
    /// each image may deviate by at most `yaw_dev` degrees (penalised).
    pub turntable: bool,
    pub yaw_dev: f32,
    /// Known angular step between consecutive images (degrees); both rotation directions are tried.
    pub yaw_step: Option<f32>,
    pub verbose: bool,
}

impl Config {
    pub fn new(n_views: usize) -> Self {
        Self {
            e: 0.04,
            q: 0.35,
            prior: 0.5,
            pitch: None,
            pitch_min: 0.0,
            pitch_max: 80.0,
            vscale: None,
            yaw_hints: vec![None; n_views],
            anchors: vec![None; n_views],
            max_shift: None,
            shift_penalty: 0.02,
            color_weight: 1.0,
            normalize_height: false,
            turntable: true,
            yaw_dev: 8.0,
            yaw_step: None,
            verbose: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub yaw: f32,
    pub shift: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub pitch: f32,
    pub vscale: f32,
    pub poses: Vec<Pose>,
    pub scales: Vec<f32>,
    pub anchors: Vec<[f32; 2]>,
    pub max_shift: Vec<f32>,
    pub score: f32,
    pub ious: Vec<f32>,
    /// world-space box around the hull found at the end of the estimation
    pub bounds: ([f32; 3], [f32; 3]),
    pub noise: Noise,
}

impl Solution {
    pub fn cam(&self, k: usize) -> Cam {
        let p = &self.poses[k];
        Cam {
            yaw: p.yaw,
            pitch: self.pitch,
            vs: self.vscale,
            scale: self.scales[k],
            off: [self.anchors[k][0] + p.shift[0], self.anchors[k][1] + p.shift[1]],
        }
    }
}

struct Cache {
    poses: Vec<Pose>,
    pitch: f32,
    llr: Vec<Vec<f32>>,
    total: Vec<f32>,
    pw: Vec<Vec<f32>>,
    score: f32,
    ious: Vec<f32>,
}

#[derive(Clone)]
struct Problem<'a> {
    views: &'a [View],
    scales: Vec<f32>,
    anchors: Vec<[f32; 2]>,
    max_shift: Vec<f32>,
    noise: Noise,
    lam_s: f32,
    lam_a: f32,
    lam_c: f32,
    yaw_window: Option<f32>,
    vs: f32,
    pmin: f32,
    pmax: f32,
    init_yaw: Vec<f32>,
    hinted: Vec<bool>,
    grid: Grid,
    s_rast: usize,
    level: usize,
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

impl<'a> Problem<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        views: &'a [View],
        scales: &[f32],
        anchors: &[[f32; 2]],
        max_shift: &[f32],
        noise: Noise,
        lam_s: f32,
        lam_c: f32,
        yaw_window: Option<f32>,
        vs: f32,
        (pmin, pmax): (f32, f32),
        init_yaw: Vec<f32>,
        hinted: Vec<bool>,
        voxel: f32,
        level: usize,
    ) -> Self {
        let max_w = views.iter().zip(scales).map(|(v, s)| v.bbox_w() / s).fold(0.0, f32::max);
        let max_h = views.iter().zip(scales).map(|(v, s)| v.bbox_h() / s).fold(0.0, f32::max);
        // generous box: the origin is only known up to the (bounded) anchor error
        let rx = 0.75 * max_w;
        let ry = 1.0 * max_h;
        let nx = ((2.0 * rx / voxel).ceil() as usize).max(4);
        let ny = ((2.0 * ry / voxel).ceil() as usize).max(4);
        let grid = Grid {
            n: [nx, ny, nx],
            v: voxel,
            o: [-(nx as f32) * voxel * 0.5, -(ny as f32) * voxel * 0.5, -(nx as f32) * voxel * 0.5],
        };
        let rn = (rx * rx * 2.0 + ry * ry).sqrt() * 1.6;
        let s_rast = (2.0 * rn / voxel).ceil() as usize + 3;
        Self {
            views,
            scales: scales.to_vec(),
            anchors: anchors.to_vec(),
            max_shift: max_shift.to_vec(),
            noise,
            lam_s,
            lam_a: if yaw_window.is_some() { 0.03 } else { 0.01 },
            lam_c,
            yaw_window,
            vs,
            pmin,
            pmax,
            init_yaw,
            hinted,
            grid,
            s_rast,
            level,
        }
    }

    fn with_vs(&self, vs: f32) -> Problem<'a> {
        let mut p = self.clone();
        p.vs = vs;
        p
    }

    fn cam(&self, k: usize, pose: &Pose, pitch: f32) -> Cam {
        Cam {
            yaw: pose.yaw,
            pitch,
            vs: self.vs,
            scale: self.scales[k],
            off: [self.anchors[k][0] + pose.shift[0], self.anchors[k][1] + pose.shift[1]],
        }
    }

    fn llr_view(&self, k: usize, pose: &Pose, pitch: f32, out: &mut [f32]) {
        let cam = self.cam(k, pose, pitch);
        let slab = self.grid.n[0] * self.grid.n[1];
        for iz in 0..self.grid.n[2] {
            llr_slab(&self.views[k], self.level, &cam, &self.grid, iz, &self.noise, &mut out[iz * slab..(iz + 1) * slab], false);
        }
    }

    /// Soft silhouette of view k resampled into the common plane raster.
    fn pin_raster(&self, k: usize, shift: [f32; 2]) -> Vec<f32> {
        let s = self.s_rast;
        let v = self.grid.v;
        let half = s as f32 * 0.5;
        let sc = self.scales[k];
        let ox = self.anchors[k][0] + shift[0];
        let oy = self.anchors[k][1] + shift[1];
        let mut out = vec![0f32; s * s];
        for j in 0..s {
            let b = (half - j as f32 - 0.5) * v;
            for i in 0..s {
                let a = (i as f32 + 0.5 - half) * v;
                out[j * s + i] = self.views[k].sample(self.level, ox + sc * a, oy - sc * b);
            }
        }
        out
    }

    fn penalty(&self, poses: &[Pose]) -> f32 {
        let k_n = poses.len() as f32;
        let mut ps = 0.0;
        for (k, p) in poses.iter().enumerate() {
            let r = self.max_shift[k].max(1e-3);
            ps += (p.shift[0] * p.shift[0] + p.shift[1] * p.shift[1]) / (r * r);
        }
        let mut pa = 0.0;
        let mut na: f32 = 0.0;
        for (k, p) in poses.iter().enumerate() {
            if !self.hinted[k] {
                pa += ang_diff(p.yaw, self.init_yaw[k]) / 180.0;
                na += 1.0;
            }
        }
        self.lam_s * ps / k_n + self.lam_a * pa / na.max(1.0)
    }

    fn score(&self, total: &[f32], poses: &[Pose], pitch: f32, pw: &[&[f32]]) -> (f32, Vec<f32>) {
        let k_n = self.views.len();
        let s = self.s_rast;
        let ss = s * s;
        let mut cover = vec![0f32; k_n * ss];
        let mut dep = vec![f32::NEG_INFINITY; k_n * ss];
        let bases: Vec<Basis> = poses.iter().map(|p| basis(p.yaw, pitch, self.vs)).collect();
        let cams: Vec<Cam> = (0..k_n).map(|k| self.cam(k, &poses[k], pitch)).collect();
        let [nx, ny, nz] = self.grid.n;
        let v = self.grid.v;
        let o = self.grid.o;
        let half = s as f32 * 0.5;
        let inv_v = 1.0 / v;
        let thr = self.noise.thr;
        for iz in 0..nz {
            let z = o[2] + (iz as f32 + 0.5) * v;
            for iy in 0..ny {
                let y = o[1] + (iy as f32 + 0.5) * v;
                let row = (iz * ny + iy) * nx;
                for ix in 0..nx {
                    let e = total[row + ix] - thr;
                    if e <= -2.0 {
                        continue;
                    }
                    // soft occupancy: smooth objective, much easier for the pattern searches
                    let pocc = (0.5 + e * 0.25).clamp(0.0, 1.0);
                    let x = o[0] + (ix as f32 + 0.5) * v;
                    for (k, b) in bases.iter().enumerate() {
                        let a = x * b.r[0] + y * b.r[1] + z * b.r[2];
                        let u = x * b.u[0] + y * b.u[1] + z * b.u[2];
                        let t = x * b.c[0] + y * b.c[1] + z * b.c[2];
                        let i0 = (a * inv_v + half - 0.5).floor() as i32;
                        let j0 = (half - u * inv_v - 0.5).floor() as i32;
                        if i0 >= 0 && j0 >= 0 && (i0 as usize) + 1 < s && (j0 as usize) + 1 < s {
                            let base = k * ss + j0 as usize * s + i0 as usize;
                            for off in [0, 1, s, s + 1] {
                                if pocc > cover[base + off] {
                                    cover[base + off] = pocc;
                                }
                                if e > 0.0 && t > dep[base + off] {
                                    dep[base + off] = t;
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut ious = Vec::with_capacity(k_n);
        for k in 0..k_n {
            let c = &cover[k * ss..(k + 1) * ss];
            let p = pw[k];
            let (mut inter, mut sum_p, mut sum_c) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..ss {
                let cc = c[i];
                inter += cc.min(p[i]);
                sum_p += p[i];
                sum_c += cc;
            }
            let union = sum_p + sum_c - inter;
            ious.push(if union > 0.0 { inter / union } else { 0.0 });
        }
        let mean = ious.iter().sum::<f32>() / k_n as f32;

        // photo-consistency: colours seen by different cameras at the same surface voxel
        let mut color_cost = 0.25f32;
        if self.lam_c > 0.0 {
            let (mut sum_std, mut n_valid, mut n_surf) = (0.0f32, 0usize, 0usize);
            let dirs: [(i32, i32, i32); 6] = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        if total[(iz * ny + iy) * nx + ix] <= thr {
                            continue;
                        }
                        let mut nrm = [0f32; 3];
                        let mut surf = false;
                        for &(dx, dy, dz) in &dirs {
                            let (xx, yy, zz) = (ix as i32 + dx, iy as i32 + dy, iz as i32 + dz);
                            let empty = xx < 0
                                || yy < 0
                                || zz < 0
                                || xx >= nx as i32
                                || yy >= ny as i32
                                || zz >= nz as i32
                                || total[(zz as usize * ny + yy as usize) * nx + xx as usize] <= thr;
                            if empty {
                                nrm[0] += dx as f32;
                                nrm[1] += dy as f32;
                                nrm[2] += dz as f32;
                                surf = true;
                            }
                        }
                        if !surf {
                            continue;
                        }
                        n_surf += 1;
                        let nl = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt().max(1e-6);
                        let nrm = [nrm[0] / nl, nrm[1] / nl, nrm[2] / nl];
                        let p = [o[0] + (ix as f32 + 0.5) * v, o[1] + (iy as f32 + 0.5) * v, o[2] + (iz as f32 + 0.5) * v];
                        let (mut ws, mut sc, mut sc2, mut nv) = (0.0f32, [0.0f32; 3], [0.0f32; 3], 0);
                        for (k, b) in bases.iter().enumerate() {
                            let facing = nrm[0] * b.c[0] + nrm[1] * b.c[1] + nrm[2] * b.c[2];
                            if facing < 0.25 {
                                continue;
                            }
                            let a = p[0] * b.r[0] + p[1] * b.r[1] + p[2] * b.r[2];
                            let u = p[0] * b.u[0] + p[1] * b.u[1] + p[2] * b.u[2];
                            let t = p[0] * b.c[0] + p[1] * b.c[1] + p[2] * b.c[2];
                            let ci = (a * inv_v + half).floor();
                            let cj = (half - u * inv_v).floor();
                            if ci < 0.0 || cj < 0.0 || ci >= s as f32 || cj >= s as f32 {
                                continue;
                            }
                            if t + 1.5 * v < dep[k * ss + cj as usize * s + ci as usize] {
                                continue;
                            }
                            let [px, py] = cams[k].project(b, p);
                            if self.views[k].sample_sd(px, py) < 1.5 {
                                continue;
                            }
                            let c = self.views[k].sample_color(px, py);
                            ws += facing;
                            for d in 0..3 {
                                sc[d] += facing * c[d];
                                sc2[d] += facing * c[d] * c[d];
                            }
                            nv += 1;
                        }
                        if nv >= 2 {
                            let mut var = 0.0;
                            for d in 0..3 {
                                let m = sc[d] / ws;
                                var += (sc2[d] / ws - m * m).max(0.0);
                            }
                            sum_std += (var / 3.0).sqrt();
                            n_valid += 1;
                        }
                    }
                }
            }
            if n_surf > 0 {
                let f = n_valid as f32 / n_surf as f32;
                let mean_std = if n_valid > 0 { sum_std / n_valid as f32 } else { 0.0 };
                color_cost = f * mean_std + (1.0 - f) * 0.25;
            }
        }
        (mean - self.penalty(poses) - self.lam_c * color_cost, ious)
    }

    fn build_cache(&self, poses: &[Pose], pitch: f32) -> Cache {
        let n = self.grid.len();
        let k_n = self.views.len();
        let llr: Vec<Vec<f32>> = (0..k_n)
            .into_par_iter()
            .map(|k| {
                let mut o = vec![0f32; n];
                self.llr_view(k, &poses[k], pitch, &mut o);
                o
            })
            .collect();
        let pw: Vec<Vec<f32>> = (0..k_n).into_par_iter().map(|k| self.pin_raster(k, poses[k].shift)).collect();
        let mut total = vec![0f32; n];
        for l in &llr {
            for i in 0..n {
                total[i] += l[i];
            }
        }
        let refs: Vec<&[f32]> = pw.iter().map(|v| v.as_slice()).collect();
        let (score, ious) = self.score(&total, poses, pitch, &refs);
        Cache { poses: poses.to_vec(), pitch, llr, total, pw, score, ious }
    }

    fn eval_view(&self, cache: &Cache, k: usize, cand: Pose) -> f32 {
        let n = self.grid.len();
        let mut new = vec![0f32; n];
        self.llr_view(k, &cand, cache.pitch, &mut new);
        let old = &cache.llr[k];
        for i in 0..n {
            new[i] = cache.total[i] - old[i] + new[i];
        }
        let mut poses = cache.poses.clone();
        let fresh;
        let pwk: &[f32] = if cand.shift == cache.poses[k].shift {
            &cache.pw[k]
        } else {
            fresh = self.pin_raster(k, cand.shift);
            &fresh
        };
        poses[k] = cand;
        let refs: Vec<&[f32]> = (0..self.views.len()).map(|i| if i == k { pwk } else { cache.pw[i].as_slice() }).collect();
        self.score(&new, &poses, cache.pitch, &refs).0
    }

    fn eval_pitch(&self, cache: &Cache, pitch: f32) -> f32 {
        let n = self.grid.len();
        let mut total = vec![0f32; n];
        let mut tmp = vec![0f32; n];
        for k in 0..self.views.len() {
            self.llr_view(k, &cache.poses[k], pitch, &mut tmp);
            for i in 0..n {
                total[i] += tmp[i];
            }
        }
        let refs: Vec<&[f32]> = cache.pw.iter().map(|v| v.as_slice()).collect();
        self.score(&total, &cache.poses, pitch, &refs).0
    }

    fn apply_view(&self, cache: &mut Cache, k: usize, pose: Pose) {
        let n = self.grid.len();
        let mut new = vec![0f32; n];
        self.llr_view(k, &pose, cache.pitch, &mut new);
        for i in 0..n {
            cache.total[i] += new[i] - cache.llr[k][i];
        }
        cache.llr[k] = new;
        if pose.shift != cache.poses[k].shift {
            cache.pw[k] = self.pin_raster(k, pose.shift);
        }
        cache.poses[k] = pose;
        let refs: Vec<&[f32]> = cache.pw.iter().map(|v| v.as_slice()).collect();
        let (s, i) = self.score(&cache.total, &cache.poses, cache.pitch, &refs);
        cache.score = s;
        cache.ious = i;
    }

    fn search_yaw(&self, cache: &Cache, k: usize) -> (f32, f32) {
        let cur = cache.poses[k];
        let mut cands: Vec<f32> = match self.yaw_window {
            Some(w) => {
                let n = (w / 2.0).ceil() as i32;
                (-n..=n).map(|i| self.init_yaw[k] + i as f32 * (w / n.max(1) as f32)).collect()
            }
            None => (0..48).map(|i| i as f32 * 7.5).collect(),
        };
        cands.push(cur.yaw);
        let scores: Vec<f32> = cands
            .par_iter()
            .map(|&y| self.eval_view(cache, k, Pose { yaw: y, shift: cur.shift }))
            .collect();
        let (bi, mut best_s) = argmax(&scores);
        let mut best_y = cands[bi];
        for &step in &[3.0f32, 1.5, 0.75, 0.4] {
            for _ in 0..8 {
                let c: Vec<f32> = [best_y - step, best_y + step]
                    .into_iter()
                    .filter(|y| self.yaw_window.map_or(true, |w| ang_diff(*y, self.init_yaw[k]) <= w + 1e-3))
                    .collect();
                if c.is_empty() {
                    break;
                }
                let sc: Vec<f32> = c
                    .par_iter()
                    .map(|&y| self.eval_view(cache, k, Pose { yaw: y, shift: cur.shift }))
                    .collect();
                let (i, s) = argmax(&sc);
                if s > best_s + 1e-6 {
                    best_s = s;
                    best_y = c[i];
                } else {
                    break;
                }
            }
        }
        (best_y.rem_euclid(360.0), best_s)
    }

    fn search_shift(&self, cache: &Cache, k: usize) -> ([f32; 2], f32) {
        let ms = self.max_shift[k];
        let cur = cache.poses[k];
        let mut best = cur.shift;
        let mut best_s = cache.score;
        let mut step = (ms * 0.25).max(0.5);
        const DIRS: [(f32, f32); 8] = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
        while step >= 0.12 {
            for _ in 0..10 {
                let cands: Vec<[f32; 2]> = DIRS
                    .iter()
                    .map(|&(dx, dy)| [best[0] + dx * step, best[1] + dy * step])
                    .filter(|c| c[0].abs() <= ms + 1e-4 && c[1].abs() <= ms + 1e-4)
                    .collect();
                if cands.is_empty() {
                    break;
                }
                let sc: Vec<f32> = cands
                    .par_iter()
                    .map(|&c| self.eval_view(cache, k, Pose { yaw: cur.yaw, shift: c }))
                    .collect();
                let (i, s) = argmax(&sc);
                if s > best_s + 1e-6 {
                    best_s = s;
                    best = cands[i];
                } else {
                    break;
                }
            }
            step *= 0.5;
        }
        (best, best_s)
    }

    fn search_pitch(&self, cache: &Cache) -> (f32, f32) {
        let mut cands: Vec<f32> = Vec::new();
        let mut pv = self.pmin;
        while pv <= self.pmax + 1e-3 {
            cands.push(pv);
            pv += 5.0;
        }
        let scores: Vec<f32> = cands.par_iter().map(|&p| self.eval_pitch(cache, p)).collect();
        let (bi, mut best_s) = argmax(&scores);
        let mut best_p = cands[bi];
        for &step in &[2.5f32, 1.25, 0.6] {
            for _ in 0..6 {
                let c: Vec<f32> = [best_p - step, best_p + step].into_iter().filter(|p| *p >= self.pmin && *p <= self.pmax).collect();
                if c.is_empty() {
                    break;
                }
                let sc: Vec<f32> = c.par_iter().map(|&p| self.eval_pitch(cache, p)).collect();
                let (i, s) = argmax(&sc);
                if s > best_s + 1e-6 {
                    best_s = s;
                    best_p = c[i];
                } else {
                    break;
                }
            }
        }
        (best_p, best_s)
    }

    /// Cheap evaluation of a starting layout: pitch scan only.
    fn quick_score(&self, poses: &[Pose], pitch: f32, pitch_fixed: bool) -> (f32, f32) {
        let mut cache = self.build_cache(poses, pitch);
        // images are cropped, so align them first or every layout looks equally bad
        for k in 0..self.views.len() {
            let (sh, sc) = self.search_shift(&cache, k);
            if sc > cache.score {
                let p = Pose { yaw: cache.poses[k].yaw, shift: sh };
                self.apply_view(&mut cache, k, p);
            }
        }
        if pitch_fixed {
            return (cache.score, pitch);
        }
        let (p, s) = self.search_pitch(&cache);
        (s.max(cache.score), if s > cache.score { p } else { pitch })
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize(
        &self,
        poses: &mut Vec<Pose>,
        pitch: &mut f32,
        vs: &mut f32,
        fixed: &[bool],
        pitch_fixed: bool,
        vs_fixed: bool,
        rounds: usize,
        tag: &str,
        verbose: bool,
    ) -> f32 {
        let k_n = self.views.len();
        let mut last = f32::NEG_INFINITY;
        let mut me = self.with_vs(*vs);
        for r in 0..rounds {
            let mut cache = me.build_cache(poses, *pitch);
            for k in 0..k_n {
                if fixed[k] {
                    continue;
                }
                let (y, s) = me.search_yaw(&cache, k);
                if s > cache.score {
                    let p = Pose { yaw: y, shift: cache.poses[k].shift };
                    me.apply_view(&mut cache, k, p);
                }
            }
            for k in 0..k_n {
                let (sh, s) = me.search_shift(&cache, k);
                if s > cache.score {
                    let p = Pose { yaw: cache.poses[k].yaw, shift: sh };
                    me.apply_view(&mut cache, k, p);
                }
            }
            if !pitch_fixed || !vs_fixed {
                // pitch and vertical stretch are strongly coupled: search them together
                let factors: &[f32] = if vs_fixed {
                    &[1.0]
                } else if r == 0 {
                    &[0.85, 1.0, 1.15]
                } else {
                    &[0.93, 1.0, 1.07]
                };
                let mut best = (cache.score, me.vs, cache.pitch);
                for &f in factors {
                    let v = (me.vs * f).clamp(0.4, 1.8);
                    let p2 = me.with_vs(v);
                    let (p, s) = if pitch_fixed { (cache.pitch, p2.eval_pitch(&cache, cache.pitch)) } else { p2.search_pitch(&cache) };
                    if s > best.0 + 1e-6 {
                        best = (s, v, p);
                    }
                }
                if best.0 > cache.score + 1e-6 {
                    me = self.with_vs(best.1);
                    cache = me.build_cache(&cache.poses.clone(), best.2);
                }
            }
            *poses = cache.poses.clone();
            *pitch = cache.pitch;
            *vs = me.vs;
            if verbose {
                eprintln!("  [{tag}] round {}: score {:.4}  pitch {:.1}  vscale {:.2}", r + 1, cache.score, cache.pitch, me.vs);
            }
            if (cache.score - last).abs() < 1e-4 {
                last = cache.score;
                break;
            }
            last = cache.score;
        }
        last
    }
}

pub fn estimate(views: &[View], cfg: &Config) -> Solution {
    let k_n = views.len();
    assert!(k_n >= 2, "at least two images are required");

    let scales: Vec<f32> = if cfg.normalize_height {
        let mut hs: Vec<f32> = views.iter().map(|v| v.bbox_h()).collect();
        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = hs[hs.len() / 2];
        views.iter().map(|v| v.bbox_h() / med).collect()
    } else {
        vec![1.0; k_n]
    };
    let anchors: Vec<[f32; 2]> = (0..k_n).map(|k| cfg.anchors[k].unwrap_or_else(|| views[k].bbox_center())).collect();
    let max_shift: Vec<f32> = (0..k_n)
        .map(|k| cfg.max_shift.unwrap_or_else(|| 0.25 * views[k].bbox_w().max(views[k].bbox_h())))
        .collect();
    let noise = Noise::new(cfg.e, cfg.q, cfg.prior);

    let any_hint = cfg.yaw_hints.iter().any(|h| h.is_some());
    let hinted: Vec<bool> = (0..k_n).map(|k| cfg.yaw_hints[k].is_some() || (!any_hint && k == 0)).collect();
    let hint_or0 = |k: usize| cfg.yaw_hints[k].unwrap_or(0.0);

    let max_dim = views.iter().zip(&scales).map(|(v, s)| v.bbox_w().max(v.bbox_h()) / s).fold(0.0, f32::max);
    let pitch_fixed = cfg.pitch.is_some();
    let pitch0 = cfg.pitch.unwrap_or(30.0f32.clamp(cfg.pitch_min, cfg.pitch_max));
    let vs_fixed = cfg.vscale.is_some();
    let vs0 = cfg.vscale.unwrap_or(1.0);
    let prange = (cfg.pitch_min, cfg.pitch_max);

    // ---- candidate yaw layouts
    let zero_poses = |yaws: &[f32]| -> Vec<Pose> { yaws.iter().map(|&y| Pose { yaw: y, shift: [0.0, 0.0] }).collect() };
    let mut layouts: Vec<Vec<f32>> = Vec::new();
    let all_hinted = any_hint && cfg.yaw_hints.iter().all(|h| h.is_some());
    let add_layout = |y: Vec<f32>, layouts: &mut Vec<Vec<f32>>| {
        if !layouts.iter().any(|s| s.iter().zip(&y).all(|(a, b)| ang_diff(*a, *b) < 1.0)) {
            layouts.push(y);
        }
    };
    if all_hinted {
        layouts.push((0..k_n).map(hint_or0).collect());
    } else if cfg.turntable {
        let mut steps: Vec<f32> = vec![10.0, 12.0, 15.0, 18.0, 20.0, 22.5, 25.0, 30.0, 36.0, 45.0, 60.0, 90.0];
        steps.push(360.0 / k_n as f32);
        steps.push(180.0 / (k_n as f32 - 1.0));
        steps.push(180.0 / k_n as f32);
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        steps.dedup_by(|a, b| (*a - *b).abs() < 0.3);
        if let Some(st) = cfg.yaw_step {
            steps = vec![st];
        }
        for &st in &steps {
            for dir in [1.0f32, -1.0] {
                let y: Vec<f32> = (0..k_n).map(|k| cfg.yaw_hints[k].unwrap_or((dir * st * k as f32).rem_euclid(360.0))).collect();
                add_layout(y, &mut layouts);
            }
        }
    } else {
        let mut spans = vec![360.0 / k_n as f32, 180.0 / (k_n as f32 - 1.0), 180.0 / k_n as f32];
        spans.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        for &sp in &spans {
            for dir in [1.0f32, -1.0] {
                let y: Vec<f32> = (0..k_n).map(|k| cfg.yaw_hints[k].unwrap_or((dir * sp * k as f32).rem_euclid(360.0))).collect();
                add_layout(y, &mut layouts);
            }
        }
    }
    let window = if cfg.turntable && !all_hinted { Some(cfg.yaw_dev) } else { None };
    let mk = |yaws: &[f32], vs: f32, voxel: f32, level: usize| {
        Problem::new(views, &scales, &anchors, &max_shift, noise, cfg.shift_penalty, cfg.color_weight, window, vs, prange, yaws.to_vec(), hinted.clone(), voxel, level)
    };
    let signed_step = |y: &[f32]| -> f32 {
        let d = (y[1.min(k_n - 1)] - y[0]).rem_euclid(360.0);
        if d > 180.0 {
            d - 360.0
        } else {
            d
        }
    };

    // ---- phase A1: rank layouts with a cheap test (pitch scan only)
    let mut ranked: Vec<(f32, f32, usize)> = layouts
        .iter()
        .enumerate()
        .map(|(i, y)| {
            let (sc, p) = mk(y, vs0, max_dim / 16.0, 0).quick_score(&zero_poses(y), pitch0, pitch_fixed);
            if cfg.verbose {
                eprintln!("  layout {:>2}/{}: yaw step {:6.1}  score {:.4}  pitch {:.1}", i + 1, layouts.len(), signed_step(y), sc, p);
            }
            (sc, p, i)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let top = layouts.len().min(3);

    // ---- every promising layout goes through the whole pipeline; the best final score wins
    //      (the cheap ranking above is too coarse to trust on its own)
    let vs_cands: Vec<f32> = if vs_fixed { vec![vs0] } else { vec![0.6, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5] };
    let mut best: Option<(f32, Vec<Pose>, f32, f32, Vec<f32>)> = None;
    for (n, &(_, p0, i)) in ranked.iter().take(top).enumerate() {
        let yaws = &layouts[i];
        // vertical stretch for this layout
        let (mut bvs, mut bsc, mut bp) = (vs0, f32::NEG_INFINITY, p0);
        for &v in &vs_cands {
            let (sc, p) = mk(yaws, v, max_dim / 16.0, 0).quick_score(&zero_poses(yaws), p0, pitch_fixed);
            if sc > bsc {
                bsc = sc;
                bvs = v;
                bp = p;
            }
        }
        // coarse grid, blurry silhouettes
        let mut poses = zero_poses(yaws);
        let mut pitch = bp;
        let mut vs = bvs;
        let mut init_yaw = yaws.clone();
        let tag = format!("candidate {}/{} (step {:.1})", n + 1, top, signed_step(yaws));
        mk(&init_yaw, vs, max_dim / 24.0, 0).optimize(&mut poses, &mut pitch, &mut vs, &hinted, pitch_fixed, vs_fixed, 2, &tag, cfg.verbose);

        // continuous refinement of the even angular step (not when the user gave the step)
        if cfg.turntable && !all_hinted && cfg.yaw_step.is_none() && k_n > 2 {
            let step0 = signed_step(&init_yaw);
            let dir = if step0 < 0.0 { -1.0f32 } else { 1.0 };
            let layout_of = |st: f32| -> Vec<f32> { (0..k_n).map(|k| cfg.yaw_hints[k].unwrap_or((dir * st * k as f32).rem_euclid(360.0))).collect() };
            let base = mk(&init_yaw, vs, max_dim / 24.0, 0).quick_score(&zero_poses(&init_yaw), pitch, pitch_fixed).0;
            let (mut best_step, mut best_sc) = (step0.abs(), base + 0.005); // must win clearly
            for j in -4..=4 {
                if j == 0 {
                    continue;
                }
                let st = step0.abs() + j as f32 * 0.4;
                let y = layout_of(st);
                let sc = mk(&y, vs, max_dim / 24.0, 0).quick_score(&zero_poses(&y), pitch, pitch_fixed).0;
                if sc > best_sc {
                    best_sc = sc;
                    best_step = st;
                }
            }
            if (best_step - step0.abs()).abs() > 1e-3 {
                init_yaw = layout_of(best_step);
                for k in 0..k_n {
                    poses[k].yaw = init_yaw[k];
                }
            }
        }

        // finer grids, sharper silhouettes
        let mut score = 0.0;
        for (level, div, rounds) in [(1usize, 40.0f32, 2usize), (2, 56.0, 2)] {
            let prob = mk(&init_yaw, vs, max_dim / div, level);
            score = prob.optimize(&mut poses, &mut pitch, &mut vs, &hinted, pitch_fixed, vs_fixed, rounds, &format!("{tag} refine L{level}"), cfg.verbose);
        }
        if best.as_ref().map_or(true, |b| score > b.0) {
            best = Some((score, poses, pitch, vs, init_yaw));
        }
    }
    let (score, poses, pitch, vs, init_yaw) = best.unwrap();

    // ---- final bookkeeping on the finest estimation grid
    let prob = mk(&init_yaw, vs, max_dim / 56.0, 2);
    let cache = prob.build_cache(&poses, pitch);
    let [nx, ny, nz] = prob.grid.n;
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if cache.total[prob.grid.idx(ix, iy, iz)] > noise.thr {
                    let c = [ix, iy, iz];
                    for d in 0..3 {
                        lo[d] = lo[d].min(c[d]);
                        hi[d] = hi[d].max(c[d]);
                    }
                }
            }
        }
    }
    let g = prob.grid;
    let bounds = if lo[0] == usize::MAX {
        let a = g.o;
        (a, [a[0] + g.n[0] as f32 * g.v, a[1] + g.n[1] as f32 * g.v, a[2] + g.n[2] as f32 * g.v])
    } else {
        let m = 1.5 * g.v;
        (
            [g.o[0] + lo[0] as f32 * g.v - m, g.o[1] + lo[1] as f32 * g.v - m, g.o[2] + lo[2] as f32 * g.v - m],
            [g.o[0] + (hi[0] + 1) as f32 * g.v + m, g.o[1] + (hi[1] + 1) as f32 * g.v + m, g.o[2] + (hi[2] + 1) as f32 * g.v + m],
        )
    };

    Solution { pitch, vscale: vs, poses, scales, anchors, max_shift, score: score.max(cache.score), ious: cache.ious.clone(), bounds, noise }
}
