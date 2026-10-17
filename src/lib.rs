#![allow(non_snake_case)]
//! Voxel2D3D: reconstruct the most probable voxel model from a handful of
//! hand-drawn 2D views (viewing angles estimated automatically) and export it to GLB.

pub mod carve;
pub mod estimate;
pub mod mesh;
pub mod model;
pub mod silhouette;

#[cfg(feature = "viewer")]
pub mod viewer;

#[cfg(test)]
mod tests {
    use super::model::*;

    #[test]
    fn camera_basis_is_orthonormal() {
        for yaw in [0.0f32, 33.0, 90.0, 215.0] {
            for pitch in [-20.0f32, 0.0, 30.0, 80.0] {
                let b = basis(yaw, pitch, 1.0);
                let dot = |a: [f32; 3], c: [f32; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
                assert!((dot(b.r, b.r) - 1.0).abs() < 1e-5);
                assert!((dot(b.u, b.u) - 1.0).abs() < 1e-5);
                assert!((dot(b.c, b.c) - 1.0).abs() < 1e-5);
                assert!(dot(b.r, b.u).abs() < 1e-5 && dot(b.r, b.c).abs() < 1e-5 && dot(b.u, b.c).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn angle_difference_wraps() {
        assert!((ang_diff(350.0, 10.0) - 20.0).abs() < 1e-4);
        assert!((ang_diff(0.0, 180.0) - 180.0).abs() < 1e-4);
    }

    #[test]
    fn glb_has_valid_header() {
        use crate::{carve::VoxelModel, mesh};
        let grid = Grid { n: [2, 2, 2], v: 1.0, o: [0.0; 3] };
        let mut occ = vec![false; 8];
        occ[0] = true;
        occ[1] = true;
        let model = VoxelModel { grid, occ, color: vec![[200, 30, 30]; 8] };
        let m = mesh::build_mesh(&model, 1.0, 0.0);
        assert_eq!(m.idx.len(), 10 * 6); // two adjacent cubes share a hidden face: 10 quads
        let path = std::env::temp_dir().join("voxel2d3d_test.glb");
        mesh::write_glb(&path, &m, "t").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, bytes.len());
    }
}
