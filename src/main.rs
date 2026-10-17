use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use voxel2d3d::carve::{self, VoxelModel};
use voxel2d3d::estimate::{self, Config};
use voxel2d3d::mesh;
use voxel2d3d::silhouette::{self, LoadOptions, View};

const HELP: &str = "\
Voxel2D3D - hand-drawn 2D views -> most probable voxel model -> GLB

USAGE
    voxel2d3d [OPTIONS] <IMAGE|DIR>...

    Images are used in the given order (a directory contributes its images sorted by name).
    The order is only the starting guess for the angles; they are estimated automatically.

OUTPUT
    -o, --output FILE        output GLB (default: model.glb)
        --height H           height of the exported model in metres (default: 1.0)
        --rotate DEG         rotate the exported model around Y (to turn its front to +Z)
        --report DIR         write report.json and one overlay PNG per input
                             (red = only input silhouette, blue = only model, grey = both)
        --view               open the interactive SDL3 preview afterwards
        --no-color           skip colour transfer

RESOLUTION
        --resolution N       longest side of the model in voxels (default: 128, never finer
                             than one source pixel)
        --voxel PX           voxel edge in source pixels (overrides --resolution)

CAMERA / ALIGNMENT
        --angles FILE        known yaws, lines `image.png 45` (those views are not estimated)
        --anchors FILE       known pixel position of the rotation axis / origin per image,
                             lines `image.png x y` (default: silhouette bbox centre, because
                             cropped sprites carry no position information)
        --pitch DEG          fix the camera elevation (default: estimated)
        --pitch-range MIN MAX  allowed elevation range in degrees (default: 0 80, i.e. the
                             camera looks down on the object; silhouettes alone cannot tell
                             'from above' from 'from below')
        --vscale X           vertical stretch of the image plane, 1 = true orthographic
                             (default: estimated; >1 for oblique game-style sprites)
        --max-shift PX       largest allowed nudge of an image in pixels
                             (default: 3 with --anchors, otherwise 25% of the image size)
        --shift-penalty X    score penalty for shifted images (default: 0.02)
        --color-weight X     weight of the colour-consistency term in the angle estimation
                             (default 1.0; 0 = silhouettes only)
        --free-angles        do not assume evenly spaced, ordered images; search every angle
                             independently (needs many images / distinctive shapes)
        --yaw-step DEG       known angle between consecutive images (e.g. 25.714 for 8 images over
                             180 degrees); both rotation directions are still tried
        --yaw-dev DEG        largest deviation of an image from the even spacing (default 8)
        --normalize-height   scale every image so the silhouette heights match
                             (for drawings made at different zoom levels)

PROBABILITY MODEL
        --noise E            P(voxel outside a silhouette | occupied), default 0.04
        --occlusion Q        P(voxel outside a silhouette | empty), default 0.35
        --prior P            prior probability of a voxel being occupied, default 0.5

SILHOUETTES
        --bg-threshold T     colour distance from the background colour (default: Otsu)
        --keep-holes         do not fill enclosed background regions
        --softness PX        edge softness of the sharpest probability map

    -q, --quiet              less output
    -h, --help               this help
";

struct Opts {
    inputs: Vec<PathBuf>,
    output: PathBuf,
    height: f32,
    rotate: f32,
    report: Option<PathBuf>,
    view: bool,
    colors: bool,
    resolution: usize,
    voxel: Option<f32>,
    angles: Option<PathBuf>,
    anchors: Option<PathBuf>,
    pitch: Option<f32>,
    pitch_min: f32,
    pitch_max: f32,
    vscale: Option<f32>,
    max_shift: Option<f32>,
    shift_penalty: f32,
    color_weight: f32,
    normalize: bool,
    free_angles: bool,
    yaw_dev: f32,
    yaw_step: Option<f32>,
    e: f32,
    q: f32,
    prior: f32,
    load: LoadOptions,
    quiet: bool,
}

fn parse_args() -> Result<Opts, String> {
    let mut o = Opts {
        inputs: vec![],
        output: PathBuf::from("model.glb"),
        height: 1.0,
        rotate: 0.0,
        report: None,
        view: false,
        colors: true,
        resolution: 128,
        voxel: None,
        angles: None,
        anchors: None,
        pitch: None,
        pitch_min: 0.0,
        pitch_max: 80.0,
        vscale: None,
        max_shift: None,
        shift_penalty: 0.02,
        color_weight: 1.0,
        normalize: false,
        free_angles: false,
        yaw_dev: 8.0,
        yaw_step: None,
        e: 0.04,
        q: 0.35,
        prior: 0.5,
        load: LoadOptions::default(),
        quiet: false,
    };
    let mut it = std::env::args().skip(1);
    fn num<T: std::str::FromStr>(it: &mut impl Iterator<Item = String>, name: &str) -> Result<T, String> {
        it.next().ok_or(format!("{name} needs a value"))?.parse::<T>().map_err(|_| format!("{name}: invalid value"))
    }
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "-o" | "--output" => o.output = PathBuf::from(it.next().ok_or("-o needs a value")?),
            "--height" => o.height = num(&mut it, "--height")?,
            "--rotate" => o.rotate = num(&mut it, "--rotate")?,
            "--report" => o.report = Some(PathBuf::from(it.next().ok_or("--report needs a value")?)),
            "--view" => o.view = true,
            "--no-color" => o.colors = false,
            "--resolution" => o.resolution = num(&mut it, "--resolution")?,
            "--voxel" => o.voxel = Some(num(&mut it, "--voxel")?),
            "--angles" => o.angles = Some(PathBuf::from(it.next().ok_or("--angles needs a value")?)),
            "--anchors" => o.anchors = Some(PathBuf::from(it.next().ok_or("--anchors needs a value")?)),
            "--pitch" => o.pitch = Some(num(&mut it, "--pitch")?),
            "--pitch-range" => {
                o.pitch_min = num(&mut it, "--pitch-range")?;
                o.pitch_max = num(&mut it, "--pitch-range")?;
            }
            "--vscale" => o.vscale = Some(num(&mut it, "--vscale")?),
            "--max-shift" => o.max_shift = Some(num(&mut it, "--max-shift")?),
            "--shift-penalty" => o.shift_penalty = num(&mut it, "--shift-penalty")?,
            "--color-weight" => o.color_weight = num(&mut it, "--color-weight")?,
            "--free-angles" => o.free_angles = true,
            "--yaw-step" => o.yaw_step = Some(num(&mut it, "--yaw-step")?),
            "--yaw-dev" => o.yaw_dev = num(&mut it, "--yaw-dev")?,
            "--normalize-height" => o.normalize = true,
            "--noise" => o.e = num(&mut it, "--noise")?,
            "--occlusion" => o.q = num(&mut it, "--occlusion")?,
            "--prior" => o.prior = num(&mut it, "--prior")?,
            "--bg-threshold" => o.load.bg_threshold = Some(num(&mut it, "--bg-threshold")?),
            "--keep-holes" => o.load.fill_holes = false,
            "--softness" => o.load.softness = Some(num(&mut it, "--softness")?),
            "-q" | "--quiet" => o.quiet = true,
            s if s.starts_with('-') => return Err(format!("unknown option {s} (see --help)")),
            s => o.inputs.push(PathBuf::from(s)),
        }
    }
    if o.inputs.is_empty() {
        return Err("no input images (see --help)".into());
    }
    Ok(o)
}

fn expand_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                .map_err(|e| format!("{}: {e}", p.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif"))
                        .unwrap_or(false)
                })
                .collect();
            v.sort();
            out.extend(v);
        } else {
            out.push(p.clone());
        }
    }
    Ok(out)
}

/// `name value [value]` lines, matched by file name.
fn read_table(path: &Path, n_values: usize) -> Result<HashMap<String, Vec<f32>>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut m = HashMap::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 1 + n_values {
            return Err(format!("{}:{}: expected `name` followed by {n_values} number(s)", path.display(), ln + 1));
        }
        let vals: Result<Vec<f32>, _> = parts[1..].iter().map(|s| s.parse::<f32>()).collect();
        m.insert(parts[0].to_string(), vals.map_err(|_| format!("{}:{}: invalid number", path.display(), ln + 1))?);
    }
    Ok(m)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let o = parse_args()?;
    let t0 = Instant::now();
    let files = expand_inputs(&o.inputs)?;
    if files.len() < 2 {
        return Err("at least two images are required".into());
    }

    let mut views: Vec<View> = Vec::new();
    for f in &files {
        let v = silhouette::load_view(f, &o.load)?;
        if !o.quiet {
            eprintln!("loaded {:<28} {}x{}  silhouette bbox {}x{}", v.name, v.w, v.h, v.bbox[2] - v.bbox[0], v.bbox[3] - v.bbox[1]);
        }
        views.push(v);
    }

    let mut cfg = Config::new(views.len());
    cfg.e = o.e;
    cfg.q = o.q;
    cfg.prior = o.prior;
    cfg.pitch = o.pitch;
    cfg.pitch_min = o.pitch_min;
    cfg.pitch_max = o.pitch_max;
    cfg.vscale = o.vscale;
    cfg.max_shift = o.max_shift;
    cfg.shift_penalty = o.shift_penalty;
    cfg.color_weight = o.color_weight;
    cfg.turntable = !o.free_angles;
    cfg.yaw_dev = o.yaw_dev;
    cfg.yaw_step = o.yaw_step;
    cfg.normalize_height = o.normalize;
    cfg.verbose = !o.quiet;

    let key = |k: usize| -> Vec<String> { vec![files[k].to_string_lossy().into_owned(), views[k].name.clone()] };
    if let Some(p) = &o.angles {
        let t = read_table(p, 1)?;
        for k in 0..views.len() {
            cfg.yaw_hints[k] = key(k).iter().find_map(|n| t.get(n)).map(|v| v[0]);
        }
    }
    if let Some(p) = &o.anchors {
        let t = read_table(p, 2)?;
        for k in 0..views.len() {
            cfg.anchors[k] = key(k).iter().find_map(|n| t.get(n)).map(|v| [v[0], v[1]]);
        }
        if cfg.anchors.iter().all(|a| a.is_some()) && cfg.max_shift.is_none() {
            cfg.max_shift = Some(3.0);
        }
    }

    if !o.quiet {
        eprintln!("estimating angles, pitch and image shifts ({} images)...", views.len());
    }
    let sol = estimate::estimate(&views, &cfg);

    if !o.quiet {
        eprintln!("\nresult (yaw 0 = camera on +Z; positive pitch = looking down):");
        eprintln!("  pitch {:.1} deg, vertical scale {:.2}, score {:.4}", sol.pitch, sol.vscale, sol.score);
        for k in 0..views.len() {
            eprintln!(
                "  {:<28} yaw {:6.1}  shift ({:+5.1}, {:+5.1}) px  IoU {:.3}",
                views[k].name, sol.poses[k].yaw, sol.poses[k].shift[0], sol.poses[k].shift[1], sol.ious[k]
            );
        }
    }

    let model = carve::reconstruct(&views, &sol, o.voxel, o.resolution, o.colors);
    let g = model.grid;
    if !o.quiet {
        eprintln!("voxel grid {}x{}x{} (voxel {:.2} px), {} occupied voxels", g.n[0], g.n[1], g.n[2], g.v, model.count());
    }
    if model.count() == 0 {
        return Err("the reconstruction is empty - the images probably do not show the same object, try --pitch or --angles".into());
    }

    let m = mesh::build_mesh(&model, o.height, o.rotate);
    let name = o.output.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    mesh::write_glb(&o.output, &m, &name).map_err(|e| format!("{}: {e}", o.output.display()))?;
    if !o.quiet {
        eprintln!("wrote {} ({} triangles) in {:.1}s", o.output.display(), m.idx.len() / 3, t0.elapsed().as_secs_f32());
    }

    if let Some(dir) = &o.report {
        write_report(dir, &views, &sol, &model)?;
        if !o.quiet {
            eprintln!("report written to {}", dir.display());
        }
    }

    if o.view {
        #[cfg(feature = "viewer")]
        voxel2d3d::viewer::run(&model, &name)?;
        #[cfg(not(feature = "viewer"))]
        eprintln!("this build has no viewer (built with --no-default-features)");
    }
    Ok(())
}

fn write_report(dir: &Path, views: &[View], sol: &estimate::Solution, model: &VoxelModel) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut rows = Vec::new();
    for (k, v) in views.iter().enumerate() {
        let cam = sol.cam(k);
        let mm = carve::project_mask(model, v, &cam);
        let iou = carve::iou(&v.mask, &mm);
        let mut img = image::RgbImage::new(v.w as u32, v.h as u32);
        for y in 0..v.h {
            for x in 0..v.w {
                let i = y * v.w + x;
                let px = match (v.mask[i], mm[i]) {
                    (true, true) => {
                        let c = v.rgb[i];
                        [c[0] / 2 + 64, c[1] / 2 + 64, c[2] / 2 + 64]
                    }
                    (true, false) => [230, 60, 60],
                    (false, true) => [60, 110, 240],
                    (false, false) => [24, 24, 24],
                };
                img.put_pixel(x as u32, y as u32, image::Rgb(px));
            }
        }
        let stem = Path::new(&v.name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        img.save(dir.join(format!("{stem}_overlay.png"))).map_err(|e| e.to_string())?;
        rows.push(serde_json::json!({
            "image": v.name,
            "yaw": sol.poses[k].yaw,
            "shift_px": sol.poses[k].shift,
            "anchor_px": [sol.anchors[k][0] + sol.poses[k].shift[0], sol.anchors[k][1] + sol.poses[k].shift[1]],
            "iou_estimate": sol.ious[k],
            "iou_final": iou,
        }));
    }
    let rep = serde_json::json!({ "pitch": sol.pitch, "vscale": sol.vscale, "score": sol.score, "views": rows });
    std::fs::write(dir.join("report.json"), serde_json::to_string_pretty(&rep).unwrap()).map_err(|e| e.to_string())
}
