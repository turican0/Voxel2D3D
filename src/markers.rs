//! Marker based calibration.
//!
//! Distinct colour regions of the pictures (red vest, white belt, green cuffs, a line on
//! the carpet, ...) are extracted per view and followed from view to view. Every such
//! track is the image of one 3D point seen from different yaw angles, i.e. it moves on
//! an ellipse whose axis ratio is sin(pitch). Fitting all tracks gives
//!
//!   * the yaw of every view, the common pitch and the vertical stretch,
//!   * the pixel position of the rotation axis / world origin in every (cropped) view,
//!   * the residual of every observation, which becomes a smooth, bounded warp of the
//!     image ("nudge parts of the picture so that everything fits").
//!
//! For fixed angles the offsets and the 3D points are a *linear* least squares problem
//! (observation = offset_k + scale * M_k * P_m), so only a handful of angle parameters
//! have to be searched.

use crate::model::basis;
use crate::silhouette::View;

#[derive(Clone, Debug)]
pub struct Track {
    /// (view index, centroid in image pixels)
    pub obs: Vec<(usize, [f32; 2])>,
    pub color: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Calib {
    pub yaw: Vec<f32>,
    pub pitch: f32,
    pub vs: f32,
    /// pixel position of the world origin in every view
    pub off: Vec<[f32; 2]>,
    pub rms: f32,
    pub tracks: Vec<Track>,
    pub points: Vec<[f32; 3]>,
    /// per track, per observation: predicted - observed (pixels), robust weight
    pub resid: Vec<Vec<([f32; 2], f32)>>,
}

// ---------------------------------------------------------------- extraction

pub fn kmeans(samples: &[[f32; 3]], k: usize, iters: usize) -> Vec<[f32; 3]> {
    let d2 = |a: &[f32; 3], b: &[f32; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let mut centers = vec![samples[0]];
    let mut mind: Vec<f32> = samples.iter().map(|s| d2(s, &centers[0])).collect();
    while centers.len() < k {
        let (bi, _) = mind.iter().enumerate().fold((0, -1.0f32), |a, (i, &d)| if d > a.1 { (i, d) } else { a });
        centers.push(samples[bi]);
        for (i, s) in samples.iter().enumerate() {
            mind[i] = mind[i].min(d2(s, centers.last().unwrap()));
        }
    }
    for _ in 0..iters {
        let mut acc = vec![[0f32; 4]; k];
        for s in samples {
            let (mut bi, mut bd) = (0, f32::MAX);
            for (i, c) in centers.iter().enumerate() {
                let d = d2(s, c);
                if d < bd {
                    bd = d;
                    bi = i;
                }
            }
            for c in 0..3 {
                acc[bi][c] += s[c];
            }
            acc[bi][3] += 1.0;
        }
        for i in 0..k {
            if acc[i][3] > 0.0 {
                centers[i] = [acc[i][0] / acc[i][3], acc[i][1] / acc[i][3], acc[i][2] / acc[i][3]];
            }
        }
    }
    centers
}

struct Marker {
    cluster: usize,
    rel: [f32; 2],
    c: [f32; 2],
    area: f32,
}

pub fn extract_tracks(views: &[View], n_colors: usize) -> Vec<Track> {
    // colour palette from all silhouettes
    let mut samples: Vec<[f32; 3]> = Vec::new();
    for v in views {
        let step = ((v.w * v.h) / 1500).max(1);
        for i in (0..v.w * v.h).step_by(step) {
            if v.sd[i] > 1.5 {
                samples.push(v.col[i]);
            }
        }
    }
    if samples.len() < n_colors * 4 {
        return vec![];
    }
    let centers = kmeans(&samples, n_colors, 12);

    // components per view and colour
    let mut all: Vec<Vec<Marker>> = Vec::new();
    for v in views {
        let n = v.w * v.h;
        let mut lab = vec![usize::MAX; n];
        for i in 0..n {
            if v.sd[i] > 1.0 {
                let c = v.col[i];
                let (mut bi, mut bd) = (0, f32::MAX);
                for (j, ce) in centers.iter().enumerate() {
                    let d = (c[0] - ce[0]).powi(2) + (c[1] - ce[1]).powi(2) + (c[2] - ce[2]).powi(2);
                    if d < bd {
                        bd = d;
                        bi = j;
                    }
                }
                lab[i] = bi;
            }
        }
        let area_sil = v.mask.iter().filter(|&&m| m).count() as f32;
        let min_area = (0.004 * area_sil).max(8.0);
        let bc = v.bbox_center();
        let mut seen = vec![false; n];
        let mut ms = Vec::new();
        let mut stack = Vec::new();
        for s in 0..n {
            if lab[s] == usize::MAX || seen[s] {
                continue;
            }
            let cl = lab[s];
            seen[s] = true;
            stack.push(s);
            let (mut sx, mut sy, mut cnt) = (0f64, 0f64, 0f64);
            while let Some(i) = stack.pop() {
                let (x, y) = (i % v.w, i / v.w);
                sx += x as f64 + 0.5;
                sy += y as f64 + 0.5;
                cnt += 1.0;
                let mut nb = [usize::MAX; 4];
                if x > 0 {
                    nb[0] = i - 1;
                }
                if x + 1 < v.w {
                    nb[1] = i + 1;
                }
                if y > 0 {
                    nb[2] = i - v.w;
                }
                if y + 1 < v.h {
                    nb[3] = i + v.w;
                }
                for &j in &nb {
                    if j != usize::MAX && !seen[j] && lab[j] == cl {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            if cnt as f32 >= min_area {
                let c = [(sx / cnt) as f32, (sy / cnt) as f32];
                ms.push(Marker { cluster: cl, rel: [c[0] - bc[0], c[1] - bc[1]], c, area: cnt as f32 });
            }
        }
        all.push(ms);
    }

    // match neighbouring views (k -> k+1, and k -> k+2 for what stayed unmatched)
    let n_views = views.len();
    let mut parent: Vec<Vec<usize>> = Vec::new(); // union-find over (view, marker)
    let mut base = vec![0usize; n_views + 1];
    for k in 0..n_views {
        base[k + 1] = base[k] + all[k].len();
    }
    let total = base[n_views];
    parent.push((0..total).collect());
    fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut matched_fwd = vec![false; total];
    for gap in [1usize, 2] {
        for k in 0..n_views.saturating_sub(gap) {
            let k2 = k + gap;
            let gate = 0.38 * views[k].bbox_w().max(views[k2].bbox_w()) * gap as f32;
            let cost = |a: &Marker, b: &Marker| -> Option<f32> {
                if a.cluster != b.cluster {
                    return None;
                }
                let ratio = (a.area / b.area).max(b.area / a.area);
                if ratio > 4.0 {
                    return None;
                }
                let d = ((a.rel[0] - b.rel[0]).powi(2) + (a.rel[1] - b.rel[1]).powi(2)).sqrt();
                if d > gate {
                    return None;
                }
                Some(d + 6.0 * ratio.ln())
            };
            // mutual best matches
            let best_fwd: Vec<Option<usize>> = all[k]
                .iter()
                .map(|a| all[k2].iter().enumerate().filter_map(|(j, b)| cost(a, b).map(|c| (j, c))).min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).map(|x| x.0))
                .collect();
            let best_bwd: Vec<Option<usize>> = all[k2]
                .iter()
                .map(|b| all[k].iter().enumerate().filter_map(|(i, a)| cost(a, b).map(|c| (i, c))).min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).map(|x| x.0))
                .collect();
            for (i, bf) in best_fwd.iter().enumerate() {
                if let Some(j) = *bf {
                    if best_bwd[j] == Some(i) {
                        let ia = base[k] + i;
                        let ib = base[k2] + j;
                        if gap == 2 && (matched_fwd[ia] || matched_fwd[ib]) {
                            continue;
                        }
                        matched_fwd[ia] = true;
                        let (ra, rb) = (find(&mut parent[0], ia), find(&mut parent[0], ib));
                        parent[0][ra] = rb;
                    }
                }
            }
        }
    }
    // collect tracks
    let mut groups: std::collections::HashMap<usize, Vec<(usize, usize)>> = std::collections::HashMap::new();
    for k in 0..n_views {
        for i in 0..all[k].len() {
            let r = find(&mut parent[0], base[k] + i);
            groups.entry(r).or_default().push((k, i));
        }
    }
    let mut tracks = Vec::new();
    for (_, mut g) in groups {
        g.sort();
        // one observation per view: keep the larger blob
        let mut obs: Vec<(usize, usize)> = Vec::new();
        for (k, i) in g {
            if let Some(last) = obs.last_mut() {
                if last.0 == k {
                    if all[k][i].area > all[k][last.1].area {
                        last.1 = i;
                    }
                    continue;
                }
            }
            obs.push((k, i));
        }
        if obs.len() >= 3 {
            let cl = all[obs[0].0][obs[0].1].cluster;
            tracks.push(Track { obs: obs.iter().map(|&(k, i)| (k, all[k][i].c)).collect(), color: centers[cl] });
        }
    }
    tracks.sort_by(|a, b| b.obs.len().cmp(&a.obs.len()));
    tracks
}

// ---------------------------------------------------------------- calibration

/// Gaussian elimination with partial pivoting on a dense symmetric system.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for c in 0..n {
        let mut p = c;
        for r in c + 1..n {
            if a[r * n + c].abs() > a[p * n + c].abs() {
                p = r;
            }
        }
        if a[p * n + c].abs() < 1e-12 {
            return None;
        }
        if p != c {
            for k in 0..n {
                a.swap(c * n + k, p * n + k);
            }
            b.swap(c, p);
        }
        let d = a[c * n + c];
        for r in c + 1..n {
            let f = a[r * n + c] / d;
            if f != 0.0 {
                for k in c..n {
                    a[r * n + k] -= f * a[c * n + k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for k in r + 1..n {
            s -= a[r * n + k] * x[k];
        }
        x[r] = s / a[r * n + r];
    }
    Some(x)
}

struct Fit {
    off: Vec<[f32; 2]>,
    points: Vec<[f32; 3]>,
    loss: f32,
    resid: Vec<Vec<([f32; 2], f32)>>,
}

/// Linear solve (robust, IRLS) for offsets and 3D points at fixed angles.
fn fit_fixed(tracks: &[Track], n_views: usize, scales: &[f32], yaw: &[f32], pitch: f32, vs: f32, irls: usize) -> Option<Fit> {
    let m = tracks.len();
    let nu = 2 * n_views + 3 * m;
    let bases: Vec<_> = yaw.iter().map(|&y| basis(y, pitch, vs)).collect();
    let mut weights: Vec<Vec<f32>> = tracks.iter().map(|t| vec![1.0; t.obs.len()]).collect();
    let mut result: Option<Fit> = None;
    for it in 0..irls.max(1) {
        let mut ata = vec![0f64; nu * nu];
        let mut atb = vec![0f64; nu];
        let mut add = |idx: &[(usize, f64)], rhs: f64, w: f64| {
            for &(i, vi) in idx {
                atb[i] += w * vi * rhs;
                for &(j, vj) in idx {
                    ata[i * nu + j] += w * vi * vj;
                }
            }
        };
        for (mi, t) in tracks.iter().enumerate() {
            for (oi, &(k, o)) in t.obs.iter().enumerate() {
                let b = &bases[k];
                let s = scales[k] as f64;
                let w = weights[mi][oi] as f64;
                // px = off_x + s * (P . r)
                let ix = [
                    (2 * k, 1.0),
                    (2 * n_views + 3 * mi, s * b.r[0] as f64),
                    (2 * n_views + 3 * mi + 1, s * b.r[1] as f64),
                    (2 * n_views + 3 * mi + 2, s * b.r[2] as f64),
                ];
                add(&ix, o[0] as f64, w);
                // py = off_y - s * (P . u)
                let iy = [
                    (2 * k + 1, 1.0),
                    (2 * n_views + 3 * mi, -s * b.u[0] as f64),
                    (2 * n_views + 3 * mi + 1, -s * b.u[1] as f64),
                    (2 * n_views + 3 * mi + 2, -s * b.u[2] as f64),
                ];
                add(&iy, o[1] as f64, w);
            }
        }
        // gauge: the 3D points are centred on the origin
        for d in 0..3 {
            let idx: Vec<(usize, f64)> = (0..m).map(|mi| (2 * n_views + 3 * mi + d, 1.0)).collect();
            add(&idx, 0.0, 50.0);
        }
        // tiny ridge for numerical safety
        for i in 0..nu {
            ata[i * nu + i] += 1e-6;
        }
        let x = solve(ata, atb, nu)?;
        let off: Vec<[f32; 2]> = (0..n_views).map(|k| [x[2 * k] as f32, x[2 * k + 1] as f32]).collect();
        let points: Vec<[f32; 3]> = (0..m).map(|mi| [x[2 * n_views + 3 * mi] as f32, x[2 * n_views + 3 * mi + 1] as f32, x[2 * n_views + 3 * mi + 2] as f32]).collect();
        // residuals + robust weights (Huber, delta = 2 px)
        let mut loss = 0.0f32;
        let mut cnt = 0.0f32;
        let mut resid = Vec::with_capacity(m);
        for (mi, t) in tracks.iter().enumerate() {
            let p = points[mi];
            let mut rr = Vec::with_capacity(t.obs.len());
            for (oi, &(k, o)) in t.obs.iter().enumerate() {
                let b = &bases[k];
                let s = scales[k];
                let px = off[k][0] + s * (p[0] * b.r[0] + p[1] * b.r[1] + p[2] * b.r[2]);
                let py = off[k][1] - s * (p[0] * b.u[0] + p[1] * b.u[1] + p[2] * b.u[2]);
                let (dx, dy) = (px - o[0], py - o[1]);
                let e = (dx * dx + dy * dy).sqrt();
                let delta = 2.0f32;
                let w = if e <= delta { 1.0 } else { delta / e };
                weights[mi][oi] = w;
                loss += if e <= delta { 0.5 * e * e } else { delta * (e - 0.5 * delta) };
                cnt += 1.0;
                rr.push(([dx, dy], w));
            }
            resid.push(rr);
        }
        result = Some(Fit { off, points, loss: loss / cnt.max(1.0), resid });
        if it + 1 >= irls {
            break;
        }
    }
    result
}

/// Robust mean reprojection loss of the tracks for given angles (offsets and 3D points are
/// solved internally). Used as one term of the angle objective.
pub fn marker_loss(tracks: &[Track], n: usize, scales: &[f32], yaw: &[f32], pitch: f32, vs: f32) -> f32 {
    fit_fixed(tracks, n, scales, yaw, pitch, vs, 2).map_or(10.0, |f| f.loss.min(10.0))
}

pub struct CalibOptions {
    pub pitch: Option<f32>,
    pub vs: Option<f32>,
    pub yaw_hints: Vec<Option<f32>>,
    /// known step between consecutive images
    pub step: Option<f32>,
    pub turntable: bool,
    pub yaw_dev: f32,
    pub max_candidates: usize,
    pub verbose: bool,
}

/// Several candidate calibrations (best marker residual first).
pub fn calibrate(views: &[View], tracks: &[Track], scales: &[f32], opt: &CalibOptions) -> Vec<Calib> {
    let n = views.len();
    if tracks.len() < 4 {
        return vec![];
    }
    let pitch_range = (0.0f32, 80.0f32);
    let eval = |yaw: &[f32], pitch: f32, vs: f32, tr: &[Track]| -> f32 { fit_fixed(tr, n, scales, yaw, pitch, vs, 2).map_or(f32::MAX, |f| f.loss) };
    let layout = |step: f32, dir: f32| -> Vec<f32> { (0..n).map(|k| opt.yaw_hints[k].unwrap_or((dir * step * k as f32).rem_euclid(360.0))).collect() };

    // ---- coarse search over step, direction, pitch, vertical stretch (run twice:
    //      the second time without the tracks that fit the first result worst)
    let steps: Vec<f32> = match opt.step {
        Some(s) => vec![s],
        None => (8..=120).map(|i| i as f32 * 0.75 + 2.0).chain([360.0 / n as f32, 180.0 / (n as f32 - 1.0), 180.0 / n as f32]).collect(),
    };
    let pitches: Vec<f32> = match opt.pitch {
        Some(p) => vec![p],
        None => (0..=8).map(|i| i as f32 * 8.0 + 4.0).collect(),
    };
    let vss: Vec<f32> = match opt.vs {
        Some(v) => vec![v],
        None => vec![0.7, 0.85, 1.0, 1.2, 1.45],
    };
    let all_hinted = opt.yaw_hints.iter().all(|h| h.is_some());
    let search = |tr: &[Track]| -> Vec<(f32, f32, f32, f32, f32)> {
        let mut cands: Vec<(f32, f32, f32, f32, f32)> = Vec::new(); // loss, step, dir, pitch, vs
        if all_hinted {
            let y: Vec<f32> = opt.yaw_hints.iter().map(|h| h.unwrap()).collect();
            for &p in &pitches {
                for &v in &vss {
                    cands.push((eval(&y, p, v, tr), 0.0, 1.0, p, v));
                }
            }
        } else {
            for &st in &steps {
                for dir in [1.0f32, -1.0] {
                    let y = layout(st, dir);
                    for &p in &pitches {
                        for &v in &vss {
                            cands.push((eval(&y, p, v, tr), st, dir, p, v));
                        }
                    }
                }
            }
        }
        cands.retain(|c| c.0 < f32::MAX);
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cands
    };
    let mut used: Vec<Track> = tracks.to_vec();
    let mut cands = search(&used);
    for _pass in 0..2 {
        if cands.is_empty() || used.len() < 9 {
            break;
        }
        let b = cands[0];
        let y = if all_hinted { opt.yaw_hints.iter().map(|h| h.unwrap()).collect() } else { layout(b.1, b.2) };
        let Some(f) = fit_fixed(&used, n, scales, &y, b.3, b.4, 3) else { break };
        let mut per: Vec<(f32, usize)> = f
            .resid
            .iter()
            .enumerate()
            .map(|(i, r)| (r.iter().map(|(d, _)| (d[0] * d[0] + d[1] * d[1]).sqrt()).sum::<f32>() / r.len() as f32, i))
            .collect();
        per.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let keep = ((used.len() as f32 * 0.7).ceil() as usize).max(8);
        if keep >= used.len() {
            break;
        }
        let kept: Vec<Track> = per.iter().take(keep).map(|&(_, i)| used[i].clone()).collect();
        used = kept;
        cands = search(&used);
    }
    let tracks: &[Track] = &used;
    cands.retain(|c| c.0 < f32::MAX);
    if cands.is_empty() {
        return vec![];
    }
    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // best candidate of every (direction, step bucket)
    let mut picked: Vec<(f32, f32, f32, f32, f32)> = Vec::new();
    for c in &cands {
        let dup = picked.iter().any(|p| p.2 == c.2 && (p.1 - c.1).abs() < 2.5);
        if !dup {
            picked.push(*c);
        }
        if picked.len() >= opt.max_candidates.max(1) {
            break;
        }
    }

    let refine = |start: (f32, f32, f32, f32, f32)| -> Option<Calib> {
        let mut tr: Vec<Track> = tracks.to_vec();
        let (_l, mut step, dir, mut pitch, mut vs) = start;
        let mut yaw: Vec<f32> = if all_hinted { opt.yaw_hints.iter().map(|h| h.unwrap()).collect() } else { layout(step, dir) };

        // drop tracks that do not fit, refit
        if let Some(f) = fit_fixed(&tr, n, scales, &yaw, pitch, vs, 3) {
            let per: Vec<f32> = f.resid.iter().map(|r| r.iter().map(|(d, _)| (d[0] * d[0] + d[1] * d[1]).sqrt()).sum::<f32>() / r.len() as f32).collect();
            let mut sorted = per.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = sorted[sorted.len() / 2];
            let lim = (3.0 * med).max(3.0);
            let kept: Vec<Track> = tr.iter().zip(&per).filter(|(_, &e)| e <= lim).map(|(t, _)| t.clone()).collect();
            if kept.len() >= 4 {
                tr = kept;
            }
        }

        // continuous refinement: step, pitch, vs, then every yaw separately (within the window)
        let nominal = yaw.clone();
        let mut cur = eval(&yaw, pitch, vs, &tr);
        for round in 0..4 {
            let sc = 1.0 / (1 << round) as f32;
            if opt.step.is_none() && !all_hinted && opt.turntable {
                for d in [-1.0f32, 1.0] {
                    let st = step + d * 0.5 * sc;
                    let y: Vec<f32> = (0..n).map(|k| opt.yaw_hints[k].unwrap_or((dir * st * k as f32).rem_euclid(360.0))).collect();
                    let l = eval(&y, pitch, vs, &tr);
                    if l < cur {
                        cur = l;
                        step = st;
                        yaw = y;
                    }
                }
            }
            if opt.pitch.is_none() {
                for d in [-1.0f32, 1.0] {
                    let p = (pitch + d * 2.0 * sc).clamp(pitch_range.0, pitch_range.1);
                    let l = eval(&yaw, p, vs, &tr);
                    if l < cur {
                        cur = l;
                        pitch = p;
                    }
                }
            }
            if opt.vs.is_none() {
                for d in [-1.0f32, 1.0] {
                    let v = (vs * (1.0 + d * 0.06 * sc)).clamp(0.4, 1.8);
                    let l = eval(&yaw, pitch, v, &tr);
                    if l < cur {
                        cur = l;
                        vs = v;
                    }
                }
            }
            for k in 0..n {
                if opt.yaw_hints[k].is_some() || (k == 0 && opt.yaw_hints.iter().all(|h| h.is_none())) {
                    continue;
                }
                for d in [-1.0f32, 1.0] {
                    let mut y = yaw.clone();
                    y[k] = (y[k] + d * 6.0 * sc).rem_euclid(360.0);
                    if opt.turntable && crate::model::ang_diff(y[k], nominal[k]) > opt.yaw_dev + 0.01 {
                        continue;
                    }
                    let l = eval(&y, pitch, vs, &tr);
                    if l < cur {
                        cur = l;
                        yaw = y;
                    }
                }
            }
        }

        let fit = fit_fixed(&tr, n, scales, &yaw, pitch, vs, 4)?;
        let inl: Vec<f32> = fit.resid.iter().flat_map(|r| r.iter().map(|(d, _)| (d[0] * d[0] + d[1] * d[1]).sqrt())).collect();
        let rms = (inl.iter().map(|e| e * e).sum::<f32>() / inl.len().max(1) as f32).sqrt();
        if opt.verbose {
            eprintln!(
                "  markers: {} tracks ({} obs), step {:.2}, pitch {:.1}, vscale {:.2}, residual rms {:.2} px",
                tr.len(),
                inl.len(),
                if dir < 0.0 { -step } else { step },
                pitch,
                vs,
                rms
            );
        }
        Some(Calib { yaw, pitch, vs, off: fit.off, rms, tracks: tr, points: fit.points, resid: fit.resid })
    };

    let mut out: Vec<Calib> = picked.into_iter().filter_map(refine).collect();
    out.sort_by(|a, b| a.rms.partial_cmp(&b.rms).unwrap());
    out
}

/// Copies of the views whose colours are reduced to `k` palette colours. The first
/// (coarse) stage works with this posterised version: shading and pixel noise do not
/// disturb the consistency measure, the angles are found from the big colour regions.
pub fn quantize_views(views: &[View], k: usize) -> Vec<View> {
    let mut samples: Vec<[f32; 3]> = Vec::new();
    for v in views {
        let step = ((v.w * v.h) / 1500).max(1);
        for i in (0..v.w * v.h).step_by(step) {
            if v.sd[i] > 1.5 {
                samples.push(v.col[i]);
            }
        }
    }
    if samples.len() < k * 4 {
        return views.to_vec();
    }
    let centers = kmeans(&samples, k, 12);
    views
        .iter()
        .map(|v| {
            let mut q = v.clone();
            for i in 0..q.col.len() {
                let c = v.col[i];
                let mut best = (f32::MAX, centers[0]);
                for ce in &centers {
                    let d = (c[0] - ce[0]).powi(2) + (c[1] - ce[1]).powi(2) + (c[2] - ce[2]).powi(2);
                    if d < best.0 {
                        best = (d, *ce);
                    }
                }
                q.col[i] = best.1;
            }
            q
        })
        .collect()
}
