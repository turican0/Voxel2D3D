//! Search for deviations between the reconstructed model and the input images, and
//! correction of the images by the smallest possible, smooth displacement.
//!
//! The coarse model is rendered into every view (with colours). The image is divided into
//! cells; for each cell the displacement (bounded by `max_shift`, sub-pixel) that makes
//! the picture agree best with the rendering is searched. A displacement is only accepted
//! where it clearly improves the fit, it is penalised by its size, and the accepted
//! displacements are blended into a smooth field (inverse-distance weights, shrinking
//! towards zero away from evidence). The image is then resampled with that field, so that
//! parts of a drawing are nudged by a fraction of a pixel up to a few pixels while the
//! rest stays untouched - an outlier drawing is pulled towards the others, not the
//! others towards the outlier.

use crate::carve::VoxelModel;
use crate::model::{basis, Cam};
use crate::silhouette::{build_view, LoadOptions, View};

pub struct Rendered {
    pub w: usize,
    pub h: usize,
    pub col: Vec<[f32; 3]>,
    pub cov: Vec<bool>,
}

/// Colour rendering of the voxel model with the camera of one view (z-buffered splats).
pub fn render_model(model: &VoxelModel, view: &View, cam: &Cam) -> Rendered {
    let b = basis(cam.yaw, cam.pitch, cam.vs);
    let g = &model.grid;
    let [nx, ny, nz] = g.n;
    let (w, h) = (view.w, view.h);
    let r = (g.v * cam.scale * 0.55).max(0.6);
    let mut depth = vec![f32::NEG_INFINITY; w * h];
    let mut col = vec![[0f32; 3]; w * h];
    let mut cov = vec![false; w * h];
    let occ = |x: i32, y: i32, z: i32| -> bool {
        x >= 0 && y >= 0 && z >= 0 && x < nx as i32 && y < ny as i32 && z < nz as i32 && model.occ[g.idx(x as usize, y as usize, z as usize)]
    };
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = g.idx(x, y, z);
                if !model.occ[idx] {
                    continue;
                }
                let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                let surface = !(occ(xi + 1, yi, zi) && occ(xi - 1, yi, zi) && occ(xi, yi + 1, zi) && occ(xi, yi - 1, zi) && occ(xi, yi, zi + 1) && occ(xi, yi, zi - 1));
                if !surface {
                    continue;
                }
                let p = g.center(x, y, z);
                let [px, py] = cam.project(&b, p);
                let t = p[0] * b.c[0] + p[1] * b.c[1] + p[2] * b.c[2];
                let c = model.color[idx];
                let (x0, x1) = ((px - r).floor() as i32, (px + r).ceil() as i32);
                let (y0, y1) = ((py - r).floor() as i32, (py + r).ceil() as i32);
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                            let i = yy as usize * w + xx as usize;
                            if t > depth[i] {
                                depth[i] = t;
                                col[i] = [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0];
                                cov[i] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    Rendered { w, h, col, cov }
}

pub struct Field {
    cell: usize,
    gw: usize,
    gh: usize,
    d: Vec<[f32; 2]>,
    conf: Vec<f32>,
    max_shift: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DevStats {
    pub mean_shift: f32,
    pub max_shift: f32,
    pub moved_cells: usize,
    pub cells: usize,
    pub err_before: f32,
    pub err_after: f32,
}

impl Field {
    pub fn eval(&self, x: f32, y: f32) -> [f32; 2] {
        let sigma = 1.1 * self.cell as f32;
        let (mut sx, mut sy, mut sw) = (0.0f32, 0.0f32, 0.0f32);
        let (cx, cy) = ((x / self.cell as f32) as i32, (y / self.cell as f32) as i32);
        for j in (cy - 3).max(0)..=(cy + 3).min(self.gh as i32 - 1) {
            for i in (cx - 3).max(0)..=(cx + 3).min(self.gw as i32 - 1) {
                let k = j as usize * self.gw + i as usize;
                if self.conf[k] <= 0.0 {
                    continue;
                }
                let (mx, my) = ((i as f32 + 0.5) * self.cell as f32, (j as f32 + 0.5) * self.cell as f32);
                let w = self.conf[k] * (-((x - mx).powi(2) + (y - my).powi(2)) / (2.0 * sigma * sigma)).exp();
                sx += w * self.d[k][0];
                sy += w * self.d[k][1];
                sw += w;
            }
        }
        // shrink towards zero where there is little evidence
        let (dx, dy) = (sx / (sw + 0.15), sy / (sw + 0.15));
        let l = (dx * dx + dy * dy).sqrt();
        if l > self.max_shift {
            [dx * self.max_shift / l, dy * self.max_shift / l]
        } else {
            [dx, dy]
        }
    }
}

/// Per cell displacement search. `lambda` penalises large shifts (relative cost).
pub fn deviation_field(view: &View, rend: &Rendered, max_shift: f32, lambda: f32) -> (Field, DevStats) {
    let (w, h) = (view.w, view.h);
    let cell = (w.max(h) / 9).max(6);
    let (gw, gh) = (w.div_ceil(cell), h.div_ceil(cell));
    let rad = cell as i32; // patch radius around the cell centre
    let mut d = vec![[0f32; 2]; gw * gh];
    let mut conf = vec![0f32; gw * gh];
    let step = 0.5f32;
    let n = (max_shift / step).round() as i32;

    let (mut e0_sum, mut e1_sum, mut n_sum) = (0.0f64, 0.0f64, 0.0f64);
    let (mut moved, mut msum, mut mmax) = (0usize, 0.0f32, 0.0f32);
    for gy in 0..gh {
        for gx in 0..gw {
            let (cxp, cyp) = (gx * cell + cell / 2, gy * cell + cell / 2);
            // pixels of the patch that matter: covered by the rendering or inside the picture
            let mut pix: Vec<usize> = Vec::new();
            for yy in (cyp as i32 - rad).max(0)..(cyp as i32 + rad).min(h as i32) {
                for xx in (cxp as i32 - rad).max(0)..(cxp as i32 + rad).min(w as i32) {
                    let i = yy as usize * w + xx as usize;
                    if rend.cov[i] || view.pins[2][i] > 0.05 {
                        pix.push(i);
                    }
                }
            }
            if pix.len() < 24 {
                continue;
            }
            let cost = |dx: f32, dy: f32| -> f32 {
                let mut s = 0.0f32;
                for &i in &pix {
                    let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                    let p = view.sample(2, x + dx, y + dy);
                    if rend.cov[i] {
                        let c = view.sample_color(x + dx, y + dy);
                        let r = rend.col[i];
                        s += (c[0] - r[0]).powi(2) + (c[1] - r[1]).powi(2) + (c[2] - r[2]).powi(2) + (0.7 * (1.0 - p)).powi(2);
                    } else {
                        s += (0.7 * p).powi(2);
                    }
                }
                s / pix.len() as f32
            };
            let c0 = cost(0.0, 0.0);
            e0_sum += c0 as f64 * pix.len() as f64;
            n_sum += pix.len() as f64;
            let (mut best, mut bd) = (c0, [0.0f32, 0.0f32]);
            for j in -n..=n {
                for i in -n..=n {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let (dx, dy) = (i as f32 * step, j as f32 * step);
                    let c = cost(dx, dy) + lambda * (dx * dx + dy * dy) / (max_shift * max_shift);
                    if c < best {
                        best = c;
                        bd = [dx, dy];
                    }
                }
            }
            let gain = (c0 - best) / (c0 + 1e-4);
            if c0 > 0.02 && gain > 0.15 {
                let k = gy * gw + gx;
                d[k] = bd;
                conf[k] = gain.min(1.0);
                moved += 1;
                let l = (bd[0] * bd[0] + bd[1] * bd[1]).sqrt();
                msum += l;
                mmax = mmax.max(l);
                e1_sum += best as f64 * pix.len() as f64;
            } else {
                e1_sum += c0 as f64 * pix.len() as f64;
            }
        }
    }
    let stats = DevStats {
        mean_shift: if moved > 0 { msum / moved as f32 } else { 0.0 },
        max_shift: mmax,
        moved_cells: moved,
        cells: gw * gh,
        err_before: if n_sum > 0.0 { (e0_sum / n_sum) as f32 } else { 0.0 },
        err_after: if n_sum > 0.0 { (e1_sum / n_sum) as f32 } else { 0.0 },
    };
    (Field { cell, gw, gh, d, conf, max_shift }, stats)
}

/// Resamples the picture with the displacement field and rebuilds the view.
pub fn warp_view(view: &View, field: &Field, opt: &LoadOptions) -> Result<View, String> {
    let (w, h) = (view.w, view.h);
    let src = &view.rgba;
    let mut out = image::RgbaImage::new(w as u32, h as u32);
    let sample = |x: f32, y: f32| -> [f32; 4] {
        // premultiplied bilinear
        let (xf, yf) = ((x - 0.5).floor(), (y - 0.5).floor());
        let (fx, fy) = (x - 0.5 - xf, y - 0.5 - yf);
        let (x0, y0) = (xf as i32, yf as i32);
        let mut acc = [0f32; 4];
        for (ox, oy, wt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            let (xx, yy) = (x0 + ox, y0 + oy);
            if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                continue;
            }
            let p = src.get_pixel(xx as u32, yy as u32).0;
            let a = p[3] as f32 / 255.0;
            acc[0] += wt * a * p[0] as f32;
            acc[1] += wt * a * p[1] as f32;
            acc[2] += wt * a * p[2] as f32;
            acc[3] += wt * a;
        }
        acc
    };
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let d = field.eval(fx, fy);
            let mut a = sample(fx + d[0], fy + d[1]);
            if d[0].abs() + d[1].abs() < 1e-3 {
                // untouched pixel: keep it exactly (pixel art stays crisp)
                out.put_pixel(x as u32, y as u32, *src.get_pixel(x as u32, y as u32));
                continue;
            }
            let alpha = a[3];
            if alpha > 1e-4 {
                for c in 0..3 {
                    a[c] /= alpha;
                }
            }
            out.put_pixel(x as u32, y as u32, image::Rgba([a[0].round() as u8, a[1].round() as u8, a[2].round() as u8, (alpha * 255.0).round() as u8]));
        }
    }
    build_view(view.name.clone(), out, opt)
}
