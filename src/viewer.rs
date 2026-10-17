//! Interactive SDL3 preview of the reconstructed voxel model.
//!
//! Software "point splat" renderer: surface voxels are sorted back-to-front every
//! frame and drawn as filled squares (orthographic projection, headlight shading).
//!
//! Controls: left mouse drag = rotate, wheel = zoom, Space = auto-rotate on/off,
//! R = reset view, Esc = quit.

use crate::carve::VoxelModel;
use crate::model::basis;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::pixels::Color;
use sdl3::render::FRect;
use std::time::Duration;

struct Vox {
    p: [f32; 3],
    n: [f32; 3],
    c: [u8; 3],
}

fn surface_voxels(model: &VoxelModel) -> (Vec<Vox>, f32) {
    let g = &model.grid;
    let [nx, ny, nz] = g.n;
    let occ = |x: i32, y: i32, z: i32| -> bool {
        x >= 0 && y >= 0 && z >= 0 && x < nx as i32 && y < ny as i32 && z < nz as i32 && model.occ[g.idx(x as usize, y as usize, z as usize)]
    };
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut v = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if !model.occ[g.idx(x, y, z)] {
                    continue;
                }
                let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                let mut n = [0f32; 3];
                for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
                    if !occ(xi + dx, yi + dy, zi + dz) {
                        n[0] += dx as f32;
                        n[1] += dy as f32;
                        n[2] += dz as f32;
                    }
                }
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if l < 1e-3 {
                    continue; // interior or fully ambiguous
                }
                let p = g.center(x, y, z);
                for d in 0..3 {
                    lo[d] = lo[d].min(p[d]);
                    hi[d] = hi[d].max(p[d]);
                }
                v.push(Vox { p, n: [n[0] / l, n[1] / l, n[2] / l], c: model.color[g.idx(x, y, z)] });
            }
        }
    }
    let ctr = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let mut extent = 1.0f32;
    for d in 0..3 {
        extent = extent.max(hi[d] - lo[d]);
    }
    for s in v.iter_mut() {
        for d in 0..3 {
            s.p[d] -= ctr[d];
        }
    }
    (v, extent)
}

pub fn run(model: &VoxelModel, title: &str) -> Result<(), String> {
    let (vox, extent) = surface_voxels(model);
    if vox.is_empty() {
        return Err("nothing to show".into());
    }
    let vsize = model.grid.v;

    let sdl = sdl3::init().map_err(|e| e.to_string())?;
    let video = sdl.video().map_err(|e| e.to_string())?;
    let window = video
        .window(&format!("Voxel2D3D - {title}  (drag: rotate, wheel: zoom, space: spin, R: reset, Esc: quit)"), 960, 720)
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;
    let mut canvas = window.into_canvas();
    let mut pump = sdl.event_pump().map_err(|e| e.to_string())?;

    let (yaw0, pitch0) = (25.0f32, 25.0f32);
    let (mut yaw, mut pitch, mut zoom, mut spin) = (yaw0, pitch0, 1.0f32, true);
    let mut order: Vec<(f32, usize)> = Vec::with_capacity(vox.len());

    'main: loop {
        for ev in pump.poll_iter() {
            match ev {
                Event::Quit { .. } => break 'main,
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'main,
                Event::KeyDown { keycode: Some(Keycode::Space), .. } => spin = !spin,
                Event::KeyDown { keycode: Some(Keycode::R), .. } => {
                    yaw = yaw0;
                    pitch = pitch0;
                    zoom = 1.0;
                }
                Event::MouseMotion { mousestate, xrel, yrel, .. } => {
                    if mousestate.left() {
                        yaw -= xrel * 0.4;
                        pitch = (pitch + yrel * 0.4).clamp(-89.0, 89.0);
                        spin = false;
                    }
                }
                Event::MouseWheel { y, .. } => zoom = (zoom * 1.1f32.powf(y)).clamp(0.1, 20.0),
                _ => {}
            }
        }
        if spin {
            yaw += 0.4;
        }

        let (w, h) = canvas.output_size().map_err(|e| e.to_string())?;
        let scale = zoom * 0.85 * w.min(h) as f32 / extent;
        let b = basis(yaw, pitch, 1.0);
        let size = (scale * vsize * 1.05).max(1.0);

        order.clear();
        for (i, s) in vox.iter().enumerate() {
            let facing = s.n[0] * b.c[0] + s.n[1] * b.c[1] + s.n[2] * b.c[2];
            if facing < -0.05 {
                continue; // back face
            }
            let t = s.p[0] * b.c[0] + s.p[1] * b.c[1] + s.p[2] * b.c[2];
            order.push((t, i));
        }
        order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        canvas.set_draw_color(Color::RGB(28, 30, 36));
        canvas.clear();
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        for &(_, i) in &order {
            let s = &vox[i];
            let sx = cx + scale * (s.p[0] * b.r[0] + s.p[1] * b.r[1] + s.p[2] * b.r[2]);
            let sy = cy - scale * (s.p[0] * b.u[0] + s.p[1] * b.u[1] + s.p[2] * b.u[2]);
            let facing = (s.n[0] * b.c[0] + s.n[1] * b.c[1] + s.n[2] * b.c[2]).max(0.0);
            let shade = 0.55 + 0.45 * facing;
            canvas.set_draw_color(Color::RGB(
                (s.c[0] as f32 * shade) as u8,
                (s.c[1] as f32 * shade) as u8,
                (s.c[2] as f32 * shade) as u8,
            ));
            let _ = canvas.fill_rect(FRect::new(sx - size * 0.5, sy - size * 0.5, size, size));
        }
        canvas.present();
        std::thread::sleep(Duration::from_millis(8));
    }
    Ok(())
}
