//! Voxel surface -> triangle mesh (per-voxel-face quads, vertex colours) -> GLB.

use crate::carve::VoxelModel;
use std::io::Write;
use std::path::Path;

pub struct Mesh {
    pub pos: Vec<[f32; 3]>,
    pub nrm: Vec<[f32; 3]>,
    /// linear RGBA
    pub col: Vec<[f32; 4]>,
    pub idx: Vec<u32>,
}

// (normal, 4 corners counter-clockwise seen from outside)
const FACES: [([i32; 3], [[f32; 3]; 4]); 6] = [
    ([1, 0, 0], [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]]),
    ([-1, 0, 0], [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.], [0., 1., 0.]]),
    ([0, 1, 0], [[0., 1., 0.], [0., 1., 1.], [1., 1., 1.], [1., 1., 0.]]),
    ([0, -1, 0], [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]]),
    ([0, 0, 1], [[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]]),
    ([0, 0, -1], [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]]),
];

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// `height`: height of the exported model in glTF units (metres); the model is
/// centred on X/Z, stands on y = 0 and is rotated by `rotate_deg` around Y.
pub fn build_mesh(model: &VoxelModel, height: f32, rotate_deg: f32) -> Mesh {
    let g = &model.grid;
    let [nx, ny, nz] = g.n;
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if model.occ[g.idx(x, y, z)] {
                    let c = [x, y, z];
                    for d in 0..3 {
                        lo[d] = lo[d].min(c[d]);
                        hi[d] = hi[d].max(c[d]);
                    }
                }
            }
        }
    }
    let mut mesh = Mesh { pos: vec![], nrm: vec![], col: vec![], idx: vec![] };
    if lo[0] == usize::MAX {
        return mesh;
    }
    let k = height / ((hi[1] - lo[1] + 1) as f32 * g.v);
    let cx = (lo[0] + hi[0] + 1) as f32 * 0.5;
    let cz = (lo[2] + hi[2] + 1) as f32 * 0.5;
    let (sr, cr) = rotate_deg.to_radians().sin_cos();
    // rotation around Y (counter-clockwise seen from above)
    let rot = |p: [f32; 3]| [p[0] * cr + p[2] * sr, p[1], -p[0] * sr + p[2] * cr];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if !model.occ[g.idx(x, y, z)] {
                    continue;
                }
                let c = model.color[g.idx(x, y, z)];
                let col = [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), 1.0];
                for (n, corners) in FACES.iter() {
                    let (X, Y, Z) = (x as i32 + n[0], y as i32 + n[1], z as i32 + n[2]);
                    let neighbour = X >= 0
                        && Y >= 0
                        && Z >= 0
                        && X < nx as i32
                        && Y < ny as i32
                        && Z < nz as i32
                        && model.occ[g.idx(X as usize, Y as usize, Z as usize)];
                    if neighbour {
                        continue;
                    }
                    let base = mesh.pos.len() as u32;
                    let nn = rot([n[0] as f32, n[1] as f32, n[2] as f32]);
                    for q in corners {
                        let p = [
                            ((x as f32 + q[0]) - cx) * g.v * k,
                            ((y - lo[1]) as f32 + q[1]) * g.v * k,
                            ((z as f32 + q[2]) - cz) * g.v * k,
                        ];
                        mesh.pos.push(rot(p));
                        mesh.nrm.push(nn);
                        mesh.col.push(col);
                    }
                    mesh.idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }
    }
    mesh
}

pub fn write_glb(path: &Path, mesh: &Mesh, name: &str) -> std::io::Result<()> {
    if mesh.pos.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "the reconstructed model is empty"));
    }
    let mut bin: Vec<u8> = Vec::new();
    let push_f32 = |bin: &mut Vec<u8>, v: &[f32]| {
        for x in v {
            bin.extend_from_slice(&x.to_le_bytes());
        }
    };
    let off_pos = bin.len();
    for p in &mesh.pos {
        push_f32(&mut bin, p);
    }
    let off_nrm = bin.len();
    for n in &mesh.nrm {
        push_f32(&mut bin, n);
    }
    let off_col = bin.len();
    for c in &mesh.col {
        // COLOR_0 as normalized unsigned bytes (linear): the most widely supported form
        for ch in c {
            bin.push((ch.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    let off_idx = bin.len();
    for i in &mesh.idx {
        bin.extend_from_slice(&i.to_le_bytes());
    }

    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &mesh.pos {
        for d in 0..3 {
            mn[d] = mn[d].min(p[d]);
            mx[d] = mx[d].max(p[d]);
        }
    }
    let nv = mesh.pos.len();
    let json = serde_json::json!({
        "asset": { "version": "2.0", "generator": "Voxel2D3D" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": name }],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1, "COLOR_0": 2 },
                "indices": 3,
                "material": 0,
                "mode": 4
            }]
        }],
        "materials": [{
            "name": "voxel",
            "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0 }
        }],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": mn, "max": mx },
            { "bufferView": 1, "componentType": 5126, "count": nv, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5121, "normalized": true, "count": nv, "type": "VEC4" },
            { "bufferView": 3, "componentType": 5125, "count": mesh.idx.len(), "type": "SCALAR" }
        ],
        "bufferViews": [
            { "buffer": 0, "byteOffset": off_pos, "byteLength": nv * 12, "target": 34962 },
            { "buffer": 0, "byteOffset": off_nrm, "byteLength": nv * 12, "target": 34962 },
            { "buffer": 0, "byteOffset": off_col, "byteLength": nv * 4, "target": 34962 },
            { "buffer": 0, "byteOffset": off_idx, "byteLength": mesh.idx.len() * 4, "target": 34963 }
        ],
        "buffers": [{ "byteLength": bin.len() }]
    });
    let mut jbytes = serde_json::to_vec(&json).unwrap();
    while jbytes.len() % 4 != 0 {
        jbytes.push(b' ');
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let total = 12 + 8 + jbytes.len() + 8 + bin.len();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&0x46546C67u32.to_le_bytes())?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(total as u32).to_le_bytes())?;
    f.write_all(&(jbytes.len() as u32).to_le_bytes())?;
    f.write_all(&0x4E4F534Au32.to_le_bytes())?;
    f.write_all(&jbytes)?;
    f.write_all(&(bin.len() as u32).to_le_bytes())?;
    f.write_all(&0x004E4942u32.to_le_bytes())?;
    f.write_all(&bin)?;
    f.flush()
}
