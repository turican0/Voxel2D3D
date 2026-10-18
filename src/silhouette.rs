//! Loading of input images: silhouette extraction (alpha channel or background
//! colour), hole filling, signed distance field and "soft" inside-probability maps.
//!
//! Hand-drawn pictures never match perfectly, so the rest of the pipeline never works
//! with a hard binary mask. Instead every view carries several probability maps
//! `p_in(x, y) = sigmoid(sd / sigma)` with decreasing `sigma` (blurry -> sharp). The
//! estimator starts with the blurry maps (wide basin of attraction) and finishes
//! with the sharp one.

use std::path::Path;

pub const LEVELS: usize = 3;

#[derive(Clone)]
pub struct View {
    pub name: String,
    pub w: usize,
    pub h: usize,
    /// the source image (needed to rebuild the view after small corrections)
    pub rgba: image::RgbaImage,
    pub rgb: Vec<[u8; 3]>,
    /// Colours in 0..1, lightly blurred inside the silhouette (used for photo-consistency).
    pub col: Vec<[f32; 3]>,
    pub mask: Vec<bool>,
    /// Signed distance to the silhouette border in pixels (positive inside).
    pub sd: Vec<f32>,
    /// Soft inside probability, level 0 = blurriest ... LEVELS-1 = sharpest.
    pub pins: Vec<Vec<f32>>,
    /// Silhouette bounding box in pixels: x0, y0, x1, y1 (exclusive).
    pub bbox: [usize; 4],
}

#[derive(Clone, Debug)]
pub struct LoadOptions {
    pub bg_threshold: Option<f32>,
    /// Fill enclosed background regions (only for images WITHOUT an alpha channel: there a
    /// light interior can be mistaken for background; with alpha the gaps are real).
    pub fill_holes: bool,
    /// Also fill enclosed transparent regions of images with an alpha channel.
    pub fill_alpha_holes: bool,
    /// Sharpest softness in pixels (None = automatic, ~0.8 % of the silhouette height).
    pub softness: Option<f32>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { bg_threshold: None, fill_holes: true, fill_alpha_holes: false, softness: None }
    }
}

impl View {
    pub fn bbox_w(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]) as f32
    }
    pub fn bbox_h(&self) -> f32 {
        (self.bbox[3] - self.bbox[1]) as f32
    }
    pub fn bbox_center(&self) -> [f32; 2] {
        [(self.bbox[0] + self.bbox[2]) as f32 * 0.5, (self.bbox[1] + self.bbox[3]) as f32 * 0.5]
    }

    /// Bilinear sample of a probability level. `u`, `v` are continuous pixel
    /// coordinates (pixel centres at +0.5). Outside the image the probability is 0.
    #[inline]
    pub fn sample(&self, level: usize, u: f32, v: f32) -> f32 {
        let x = u - 0.5;
        let y = v - 0.5;
        let xf = x.floor();
        let yf = y.floor();
        let fx = x - xf;
        let fy = y - yf;
        let x0 = xf as i32;
        let y0 = yf as i32;
        let w = self.w as i32;
        let h = self.h as i32;
        if x0 < -1 || y0 < -1 || x0 >= w || y0 >= h {
            return 0.0;
        }
        let m = &self.pins[level];
        let g = |xx: i32, yy: i32| -> f32 {
            if xx < 0 || yy < 0 || xx >= w || yy >= h {
                0.0
            } else {
                m[yy as usize * self.w + xx as usize]
            }
        };
        let a = g(x0, y0) * (1.0 - fx) + g(x0 + 1, y0) * fx;
        let b = g(x0, y0 + 1) * (1.0 - fx) + g(x0 + 1, y0 + 1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Signed distance (bilinear), outside the image: -distance to the border.
    pub fn sample_sd(&self, u: f32, v: f32) -> f32 {
        let x = u - 0.5;
        let y = v - 0.5;
        let xf = x.floor();
        let yf = y.floor();
        let fx = x - xf;
        let fy = y - yf;
        let (x0, y0) = (xf as i32, yf as i32);
        let g = |xx: i32, yy: i32| -> f32 {
            let cx = xx.clamp(0, self.w as i32 - 1) as usize;
            let cy = yy.clamp(0, self.h as i32 - 1) as usize;
            let mut s = self.sd[cy * self.w + cx];
            // outside of the image there is nothing
            if xx < 0 || yy < 0 || xx >= self.w as i32 || yy >= self.h as i32 {
                s = -0.5;
            }
            s
        };
        let a = g(x0, y0) * (1.0 - fx) + g(x0 + 1, y0) * fx;
        let b = g(x0, y0 + 1) * (1.0 - fx) + g(x0 + 1, y0 + 1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Bilinear sample of the blurred colour image.
    #[inline]
    pub fn sample_color(&self, u: f32, v: f32) -> [f32; 3] {
        let x = u - 0.5;
        let y = v - 0.5;
        let xf = x.floor();
        let yf = y.floor();
        let fx = x - xf;
        let fy = y - yf;
        let cx = |i: i32| i.clamp(0, self.w as i32 - 1) as usize;
        let cy = |i: i32| i.clamp(0, self.h as i32 - 1) as usize;
        let (x0, y0) = (xf as i32, yf as i32);
        let g = |xx: i32, yy: i32| self.col[cy(yy) * self.w + cx(xx)];
        let (a, b, c, d) = (g(x0, y0), g(x0 + 1, y0), g(x0, y0 + 1), g(x0 + 1, y0 + 1));
        let mut o = [0f32; 3];
        for i in 0..3 {
            o[i] = (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy;
        }
        o
    }

    #[inline]
    pub fn color_at(&self, u: f32, v: f32) -> [u8; 3] {
        let x = (u.floor() as i32).clamp(0, self.w as i32 - 1) as usize;
        let y = (v.floor() as i32).clamp(0, self.h as i32 - 1) as usize;
        self.rgb[y * self.w + x]
    }
}

pub fn load_view(path: &Path, opt: &LoadOptions) -> Result<View, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgba8();
    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    build_view(name, img, opt).map_err(|e| format!("{}: {e}", path.display()))
}

/// Builds all derived maps (silhouette, distance field, probabilities, colours) from an image.
pub fn build_view(name: String, img: image::RgbaImage, opt: &LoadOptions) -> Result<View, String> {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let px = img.as_raw().clone();
    let n = w * h;

    let rgb: Vec<[u8; 3]> = (0..n).map(|i| [px[i * 4], px[i * 4 + 1], px[i * 4 + 2]]).collect();

    let transparent = (0..n).filter(|&i| px[i * 4 + 3] < 128).count();
    let use_alpha = transparent as f32 > 0.01 * n as f32;

    let mut mask = vec![false; n];
    if use_alpha {
        for i in 0..n {
            mask[i] = px[i * 4 + 3] >= 128;
        }
    } else {
        // Background colour = per-channel median of the border pixels.
        let mut border: Vec<[u8; 3]> = Vec::new();
        for x in 0..w {
            border.push(rgb[x]);
            border.push(rgb[(h - 1) * w + x]);
        }
        for y in 0..h {
            border.push(rgb[y * w]);
            border.push(rgb[y * w + w - 1]);
        }
        let med = |c: usize| {
            let mut v: Vec<u8> = border.iter().map(|p| p[c]).collect();
            v.sort_unstable();
            v[v.len() / 2] as f32
        };
        let bg = [med(0), med(1), med(2)];
        let dist: Vec<f32> = rgb
            .iter()
            .map(|p| {
                let d = [p[0] as f32 - bg[0], p[1] as f32 - bg[1], p[2] as f32 - bg[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .collect();
        let thr = opt.bg_threshold.unwrap_or_else(|| otsu(&dist).max(12.0));
        for i in 0..n {
            mask[i] = dist[i] > thr;
        }
    }

    if (use_alpha && opt.fill_alpha_holes) || (!use_alpha && opt.fill_holes) {
        fill_holes(&mut mask, w, h);
    }
    drop_small_components(&mut mask, w, h, 0.03);

    // bounding box
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 == 0 {
        return Err("no silhouette found (empty mask)".to_string());
    }

    // signed distance
    let bg_src: Vec<bool> = mask.iter().map(|m| !m).collect();
    let d_in = edt(&bg_src, w, h); // distance to nearest background pixel
    let d_out = edt(&mask, w, h); // distance to nearest foreground pixel
    let sd: Vec<f32> = (0..n)
        .map(|i| if mask[i] { (d_in[i] - 0.5).min(1.0e4) } else { -(d_out[i] - 0.5).min(1.0e4) })
        .collect();

    let bh = (y1 - y0) as f32;
    let sharp = opt.softness.unwrap_or_else(|| (0.008 * bh).max(0.8));
    let sigmas = [(0.07 * bh).max(sharp * 4.0), (0.025 * bh).max(sharp * 2.0), sharp];
    let pins: Vec<Vec<f32>> = sigmas
        .iter()
        .map(|&s| sd.iter().map(|&d| 1.0 / (1.0 + (-d / s).exp())).collect())
        .collect();

    // blurred colours: 3x3 mean over silhouette pixels only (no black bleeding from the background)
    let mut col = vec![[0f32; 3]; n];
    for y in 0..h {
        for x in 0..w {
            let (mut acc, mut ws) = ([0f32; 3], 0f32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if mask[j] {
                        for c in 0..3 {
                            acc[c] += rgb[j][c] as f32 / 255.0;
                        }
                        ws += 1.0;
                    }
                }
            }
            let i = y * w + x;
            col[i] = if ws > 0.0 { [acc[0] / ws, acc[1] / ws, acc[2] / ws] } else { [rgb[i][0] as f32 / 255.0, rgb[i][1] as f32 / 255.0, rgb[i][2] as f32 / 255.0] };
        }
    }

    Ok(View { name, w, h, rgba: img.clone(), rgb, col, mask, sd, pins, bbox: [x0, y0, x1, y1] })
}

/// Otsu threshold on a 0..255 histogram of the (clamped) values.
fn otsu(vals: &[f32]) -> f32 {
    let mut hist = [0f64; 256];
    for &v in vals {
        hist[(v.clamp(0.0, 255.0)) as usize] += 1.0;
    }
    let total: f64 = hist.iter().sum();
    let sum_all: f64 = hist.iter().enumerate().map(|(i, c)| i as f64 * c).sum();
    let (mut wb, mut sb, mut best, mut thr) = (0.0, 0.0, -1.0, 0usize);
    for t in 0..256 {
        wb += hist[t];
        if wb == 0.0 {
            continue;
        }
        let wf = total - wb;
        if wf == 0.0 {
            break;
        }
        sb += t as f64 * hist[t];
        let mb = sb / wb;
        let mf = (sum_all - sb) / wf;
        let var = wb * wf * (mb - mf) * (mb - mf);
        if var > best {
            best = var;
            thr = t;
        }
    }
    thr as f32
}

fn fill_holes(mask: &mut [bool], w: usize, h: usize) {
    let mut outside = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let push = |i: usize, outside: &mut Vec<bool>, stack: &mut Vec<usize>| {
        if !mask[i] && !outside[i] {
            outside[i] = true;
            stack.push(i);
        }
    };
    for x in 0..w {
        push(x, &mut outside, &mut stack);
        push((h - 1) * w + x, &mut outside, &mut stack);
    }
    for y in 0..h {
        push(y * w, &mut outside, &mut stack);
        push(y * w + w - 1, &mut outside, &mut stack);
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        if x > 0 {
            push(i - 1, &mut outside, &mut stack);
        }
        if x + 1 < w {
            push(i + 1, &mut outside, &mut stack);
        }
        if y > 0 {
            push(i - w, &mut outside, &mut stack);
        }
        if y + 1 < h {
            push(i + w, &mut outside, &mut stack);
        }
    }
    for i in 0..w * h {
        mask[i] = !outside[i];
    }
}

fn drop_small_components(mask: &mut [bool], w: usize, h: usize, rel: f32) {
    let mut label = vec![0u32; w * h];
    let mut areas: Vec<usize> = vec![0];
    let mut stack = Vec::new();
    for s in 0..w * h {
        if !mask[s] || label[s] != 0 {
            continue;
        }
        let id = areas.len() as u32;
        areas.push(0);
        label[s] = id;
        stack.push(s);
        while let Some(i) = stack.pop() {
            areas[id as usize] += 1;
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && label[j] == 0 {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
    }
    let max = areas.iter().copied().max().unwrap_or(0) as f32;
    for i in 0..w * h {
        if mask[i] && (areas[label[i] as usize] as f32) < rel * max {
            mask[i] = false;
        }
    }
}

/// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher):
/// distance of every pixel to the nearest `true` pixel of `src`.
fn edt(src: &[bool], w: usize, h: usize) -> Vec<f32> {
    const INF: f64 = 1.0e10;
    let n = w.max(h);
    let mut g = vec![0f64; w * h];
    let mut v = vec![0usize; n];
    let mut z = vec![0f64; n + 1];
    let mut f = vec![0f64; n];
    let mut d = vec![0f64; n];
    for x in 0..w {
        for y in 0..h {
            f[y] = if src[y * w + x] { 0.0 } else { INF };
        }
        edt_1d(&f[..h], &mut d[..h], &mut v, &mut z);
        for y in 0..h {
            g[y * w + x] = d[y];
        }
    }
    for y in 0..h {
        for x in 0..w {
            f[x] = g[y * w + x];
        }
        edt_1d(&f[..w], &mut d[..w], &mut v, &mut z);
        for x in 0..w {
            g[y * w + x] = d[x];
        }
    }
    g.iter().map(|&q| q.sqrt() as f32).collect()
}

fn edt_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s;
        loop {
            let p = v[k];
            s = ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * q as f64 - 2.0 * p as f64);
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        let dq = q as f64 - p as f64;
        d[q] = dq * dq + f[p];
    }
}
