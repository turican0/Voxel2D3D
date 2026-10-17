//! Camera model, voxel grid and the per-view evidence term.
//!
//! Camera: orthographic, looking at the origin. `yaw` turns the camera around the
//! vertical Y axis (yaw 0 = camera on the +Z side, yaw 90 = camera on the +X side),
//! `pitch` is the elevation above the horizon (positive = looking down on the object).
//! World axes follow glTF: +Y up, +Z "front".
//!
//! A world point `p` lands in the image at
//! `px = off.x + scale * dot(p, right)`, `py = off.y - scale * dot(p, up)`.
//! The image offset `off` is the pixel position of the world origin; because the
//! input images are usually cropped to the silhouette it is not known and is
//! estimated per view (nominal anchor + a bounded, penalised shift).

use crate::silhouette::View;

#[derive(Clone, Copy, Debug)]
pub struct Basis {
    pub r: [f32; 3],
    pub u: [f32; 3],
    pub c: [f32; 3],
}

/// `vs` is a vertical stretch of the image plane (1 = true orthographic rotation).
/// Many 2.5D game sprites are rendered with an oblique projection in which the height
/// is not foreshortened by cos(pitch); `vs` > 1 models that.
pub fn basis(yaw_deg: f32, pitch_deg: f32, vs: f32) -> Basis {
    let (sy, cy) = yaw_deg.to_radians().sin_cos();
    let (sp, cp) = pitch_deg.to_radians().sin_cos();
    Basis {
        r: [cy, 0.0, -sy],
        u: [-sp * sy * vs, cp * vs, -sp * cy * vs],
        c: [sy * cp, sp, cy * cp],
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cam {
    pub yaw: f32,
    pub pitch: f32,
    pub vs: f32,
    /// pixels per world unit
    pub scale: f32,
    /// pixel position of the world origin
    pub off: [f32; 2],
}

impl Cam {
    #[inline]
    pub fn project(&self, b: &Basis, p: [f32; 3]) -> [f32; 2] {
        let a = p[0] * b.r[0] + p[1] * b.r[1] + p[2] * b.r[2];
        let u = p[0] * b.u[0] + p[1] * b.u[1] + p[2] * b.u[2];
        [self.off[0] + self.scale * a, self.off[1] - self.scale * u]
    }
}

/// Observation model. A voxel is "occupied" or "empty"; a view reports the voxel as
/// inside / outside its silhouette.
/// P(outside | occupied) = e   (drawing error)
/// P(outside | empty)    = q   (empty voxels may still project into the silhouette)
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    /// log-likelihood ratio contributed by "inside"
    pub a: f32,
    /// log-likelihood ratio contributed by "outside" (negative)
    pub b: f32,
    /// occupied iff the summed evidence exceeds this
    pub thr: f32,
}

impl Noise {
    pub fn new(e: f32, q: f32, prior_occupied: f32) -> Self {
        let prior = (prior_occupied / (1.0 - prior_occupied)).ln();
        Self { a: ((1.0 - e) / (1.0 - q)).ln(), b: (e / q).ln(), thr: -prior }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub n: [usize; 3],
    pub v: f32,
    /// world position of the minimum corner
    pub o: [f32; 3],
}

impl Grid {
    pub fn len(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }
    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.n[0] * (y + self.n[1] * z)
    }
    #[inline]
    pub fn center(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        [
            self.o[0] + (x as f32 + 0.5) * self.v,
            self.o[1] + (y as f32 + 0.5) * self.v,
            self.o[2] + (z as f32 + 0.5) * self.v,
        ]
    }
}

/// Evidence of one view for one z-slab of the grid (`out` has nx*ny entries).
pub fn llr_slab(view: &View, level: usize, cam: &Cam, grid: &Grid, iz: usize, noise: &Noise, out: &mut [f32], add: bool) {
    let b = basis(cam.yaw, cam.pitch, cam.vs);
    let [nx, ny, _] = grid.n;
    let v = grid.v;
    let z = grid.o[2] + (iz as f32 + 0.5) * v;
    let x0 = grid.o[0] + 0.5 * v;
    let da = v * b.r[0];
    let db = v * b.u[0];
    for iy in 0..ny {
        let y = grid.o[1] + (iy as f32 + 0.5) * v;
        let mut a = x0 * b.r[0] + y * b.r[1] + z * b.r[2];
        let mut u = x0 * b.u[0] + y * b.u[1] + z * b.u[2];
        let row = &mut out[iy * nx..(iy + 1) * nx];
        for o in row.iter_mut() {
            let px = cam.off[0] + cam.scale * a;
            let py = cam.off[1] - cam.scale * u;
            let p = view.sample(level, px, py);
            let l = p * noise.a + (1.0 - p) * noise.b;
            if add {
                *o += l;
            } else {
                *o = l;
            }
            a += da;
            u += db;
        }
    }
}

/// Smallest absolute difference of two angles in degrees (0..180).
pub fn ang_diff(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}
