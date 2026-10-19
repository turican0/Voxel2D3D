use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use voxel2d3d::carve::{self, VoxelModel};
use voxel2d3d::estimate::{self, Config};
use voxel2d3d::exchange;
use voxel2d3d::markers;
use voxel2d3d::mesh;
use voxel2d3d::refine;
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
        --export-frame FILE  write the transform world -> GLB and the cameras (JSON), so that an
                             external 3D model can be placed into / projected like the views

EXCHANGE WITH A 3D MODEL
        --save-solution FILE save the estimated cameras (JSON)
        --save-views DIR     save the (nudged) input images the solution belongs to
        --solution FILE      use these cameras instead of estimating them (pass the images
                             written by --save-views); the reconstruction space is then the same
                             as in the run that saved them
        --prior-model FILE   occupancy prior on a world-space lattice (VXPR file, see
                             src/exchange.rs), e.g. voxelised from a 3D model of the object;
                             added to the silhouette evidence
        --prior-model-weight W  weight of the model prior in log-odds (default 1.0; one view that clearly
                             sees 'outside' counts about 2.2)

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
        --span DEG           the images evenly cover this angle, first to last (180 = half circle,
                             typical for mirrored sprite sheets; 360 = full circle)
        --yaw-dev DEG        each image's angle is searched separately within this window around
                             the starting estimate from the markers (default 10)
        --no-repair          do not add voxels for silhouette pixels the strict intersection loses
                             (thin parts: arms, cloth, tips)
        --fine-y A,B         evaluate this part of the model height (fractions 0..1 from the bottom,
                             e.g. 0.1,0.55 for the legs) with 2x the resolution
        --nudge PX           largest shift when parts of an image are nudged to fit (default 3)
        --no-stages          single pass (no coarse model / nudging / full-colour repeat)
        --no-markers         do not use colour markers for the starting angles
        --normalize-height   scale every image so the silhouette heights match
                             (for drawings made at different zoom levels)

PROBABILITY MODEL
        --noise E            P(voxel outside a silhouette | occupied), default 0.04
        --occlusion Q        P(voxel outside a silhouette | empty), default 0.35
        --prior P            prior probability of a voxel being occupied, default 0.5

SILHOUETTES
        --bg-threshold T     colour distance from the background colour (default: Otsu)
        --keep-holes         do not fill enclosed background regions (images without alpha)
        --fill-holes         also fill enclosed transparent regions of images with alpha
                             (by default transparent gaps, e.g. between the legs, stay open)
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
    no_markers: bool,
    stages: bool,
    repair: bool,
    fine_y: Option<(f32, f32)>,
    nudge: f32,
    yaw_dev: f32,
    yaw_step: Option<f32>,
    span: Option<f32>,
    e: f32,
    q: f32,
    prior: f32,
    load: LoadOptions,
    quiet: bool,
    export_frame: Option<PathBuf>,
    save_solution: Option<PathBuf>,
    save_views: Option<PathBuf>,
    solution: Option<PathBuf>,
    prior_file: Option<PathBuf>,
    prior_weight: f32,
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
        no_markers: false,
        stages: true,
        repair: true,
        fine_y: None,
        nudge: 3.0,
        yaw_dev: 10.0,
        yaw_step: None,
        span: None,
        e: 0.04,
        q: 0.35,
        prior: 0.5,
        load: LoadOptions::default(),
        quiet: false,
        export_frame: None,
        save_solution: None,
        save_views: None,
        solution: None,
        prior_file: None,
        prior_weight: 1.0,
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
            "--export-frame" => o.export_frame = Some(PathBuf::from(it.next().ok_or("--export-frame needs a value")?)),
            "--save-solution" => o.save_solution = Some(PathBuf::from(it.next().ok_or("--save-solution needs a value")?)),
            "--save-views" => o.save_views = Some(PathBuf::from(it.next().ok_or("--save-views needs a value")?)),
            "--solution" => o.solution = Some(PathBuf::from(it.next().ok_or("--solution needs a value")?)),
            "--prior-model" => o.prior_file = Some(PathBuf::from(it.next().ok_or("--prior-model needs a value")?)),
            "--prior-model-weight" => o.prior_weight = num(&mut it, "--prior-model-weight")?,
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
            "--no-markers" => o.no_markers = true,
            "--no-stages" => o.stages = false,
            "--no-repair" => o.repair = false,
            "--fine-y" => {
                let v = it.next().ok_or("--fine-y needs a value like 0.1,0.55")?;
                let p: Vec<&str> = v.split(',').collect();
                if p.len() != 2 {
                    return Err("--fine-y: expected A,B".into());
                }
                let a: f32 = p[0].parse().map_err(|_| "--fine-y: invalid number")?;
                let b: f32 = p[1].parse().map_err(|_| "--fine-y: invalid number")?;
                o.fine_y = Some((a, b));
            }
            "--nudge" => o.nudge = num(&mut it, "--nudge")?,
            "--free-angles" => o.free_angles = true,
            "--span" => {
                let span: f32 = num(&mut it, "--span")?;
                o.span = Some(span);
            }
            "--yaw-step" => o.yaw_step = Some(num(&mut it, "--yaw-step")?),
            "--yaw-dev" => o.yaw_dev = num(&mut it, "--yaw-dev")?,
            "--normalize-height" => o.normalize = true,
            "--noise" => o.e = num(&mut it, "--noise")?,
            "--occlusion" => o.q = num(&mut it, "--occlusion")?,
            "--prior" => o.prior = num(&mut it, "--prior")?,
            "--bg-threshold" => o.load.bg_threshold = Some(num(&mut it, "--bg-threshold")?),
            "--keep-holes" => o.load.fill_holes = false,
            "--fill-holes" => {
                o.load.fill_holes = true;
                o.load.fill_alpha_holes = true;
            }
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
    cfg.yaw_step = o.yaw_step.or_else(|| {
        o.span.map(|sp| if sp >= 360.0 { sp / views.len() as f32 } else { sp / (views.len() as f32 - 1.0) })
    });
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

    let loaded = match &o.solution {
        Some(p) => Some(exchange::load_solution(p, views.len())?),
        None => None,
    };

    // ---- marker based calibration (angles, pitch, image offsets)
    let mut calib: Option<markers::Calib> = None;
    if !o.no_markers && loaded.is_none() {
        let tracks = markers::extract_tracks(&views, 12);
        if !o.quiet {
            eprintln!("markers: {} tracks found", tracks.len());
        }
        let step = cfg.yaw_step.or_else(|| o.span.map(|sp| if sp >= 360.0 { sp / views.len() as f32 } else { sp / (views.len() as f32 - 1.0) }));
        let copt = markers::CalibOptions { pitch: o.pitch, vs: o.vscale, yaw_hints: cfg.yaw_hints.clone(), step, turntable: !o.free_angles, yaw_dev: o.yaw_dev, verbose: !o.quiet, max_candidates: 4 };
        let scales = vec![1.0f32; views.len()];
        let cands = markers::calibrate(&views, &tracks, &scales, &copt);
        // candidates whose markers fit much worse than the best one are not worth a full run
        let best_rms = cands.first().map_or(0.0, |c| c.rms);
        let mut cands: Vec<markers::Calib> = cands.into_iter().filter(|c| c.rms <= 1.5 * best_rms + 0.3).collect();
        if best_rms > 3.0 {
            if !o.quiet {
                eprintln!("  markers do not agree well enough ({:.1} px) - not used, searching without them", best_rms);
            }
            cands.clear();
        }
        for c in &cands {
            if !o.quiet {
                eprintln!("  marker yaws: {:?}", c.yaw.iter().map(|y| (y * 10.0).round() / 10.0).collect::<Vec<_>>());
            }
            cfg.inits.push(estimate::Init { yaw: c.yaw.clone(), pitch: c.pitch, vs: c.vs, off: c.off.clone(), tracks: c.tracks.clone() });
        }
        if !cands.is_empty() {
            calib = cands.into_iter().next();
            if cfg.max_shift.is_none() {
                cfg.max_shift = Some(4.0);
            }
        }
    }
    let _ = &calib;

    if !o.quiet {
        eprintln!("estimating angles, pitch and image shifts ({} images)...", views.len());
    }
    // ---- stage 1: angles of every image, found separately (posterised colours: robust against shading)
    let stages = o.stages && loaded.is_none();
    let mut sol = match loaded {
        Some(s) => {
            if !o.quiet {
                eprintln!("cameras loaded from the solution file - no estimation");
            }
            s
        }
        None => {
            let views_q = if stages { markers::quantize_views(&views, 8) } else { views.clone() };
            estimate::estimate(&views_q, &cfg)
        }
    };
    print_result(&views, &sol, o.quiet, "stage 1");

    let mut views = views;
    if stages {
        // ---- stage 2: coarse model -> where does it not fit? -> nudge the images by the minimum
        let coarse = carve::reconstruct(&views, &sol, None, 64, true, None, false, None);
        let mut nudged = Vec::with_capacity(views.len());
        for k in 0..views.len() {
            let rend = refine::render_model(&coarse, &views[k], &sol.cam(k));
            let (field, st) = refine::deviation_field(&views[k], &rend, o.nudge, 0.05);
            if !o.quiet {
                eprintln!(
                    "  deviations {:<26} {:>2}/{} cells moved, mean {:.2} px, max {:.2} px, error {:.4} -> {:.4}",
                    views[k].name, st.moved_cells, st.cells, st.mean_shift, st.max_shift, st.err_before, st.err_after
                );
            }
            nudged.push(refine::warp_view(&views[k], &field, &o.load)?);
        }
        views = nudged;

        // ---- stage 3: the same again with full colours, starting from the stage 1 result
        let init = estimate::Init {
            yaw: sol.poses.iter().map(|p| p.yaw).collect(),
            pitch: sol.pitch,
            vs: sol.vscale,
            off: (0..views.len()).map(|k| [sol.anchors[k][0] + sol.poses[k].shift[0], sol.anchors[k][1] + sol.poses[k].shift[1]]).collect(),
            tracks: vec![],
        };
        let mut cfg2 = cfg.clone();
        cfg2.inits = vec![init];
        cfg2.yaw_dev = cfg.yaw_dev.min(6.0);
        cfg2.max_shift = Some(o.nudge.max(1.0));
        if !o.quiet {
            eprintln!("stage 3: repeating with full colours...");
        }
        sol = estimate::estimate(&views, &cfg2);
        print_result(&views, &sol, o.quiet, "stage 3");
    }
    let views = views;

    if let Some(p) = &o.save_solution {
        let names: Vec<String> = views.iter().map(|v| v.name.clone()).collect();
        exchange::save_solution(p, &sol, &names)?;
    }
    if let Some(dir) = &o.save_views {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for v in &views {
            let p = dir.join(&v.name);
            v.rgba.save(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        }
    }
    let prior = match &o.prior_file {
        Some(p) => Some(exchange::Prior::load(p)?),
        None => None,
    };
    let model = carve::reconstruct(&views, &sol, o.voxel, o.resolution, o.colors, o.fine_y, o.repair, prior.as_ref().map(|p| (p, o.prior_weight)));
    let g = model.grid;
    if !o.quiet {
        eprintln!("voxel grid {}x{}x{} (voxel {:.2} px), {} occupied voxels", g.n[0], g.n[1], g.n[2], g.v, model.count());
    }
    if model.count() == 0 {
        return Err("the reconstruction is empty - the images probably do not show the same object, try --pitch or --angles".into());
    }

    let m = mesh::build_mesh(&model, o.height, o.rotate);
    if let Some(p) = &o.export_frame {
        if let Some(fr) = exchange::export_frame(&model, o.height, o.rotate) {
            exchange::save_frame(p, &fr, &sol, &model.grid)?;
        }
    }
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

fn print_result(views: &[View], sol: &estimate::Solution, quiet: bool, tag: &str) {
    if quiet {
        return;
    }
    eprintln!("\nresult {tag} (yaw 0 = camera on +Z; positive pitch = looking down):");
    eprintln!("  pitch {:.1} deg, vertical scale {:.2}, score {:.4}", sol.pitch, sol.vscale, sol.score);
    for k in 0..views.len() {
        eprintln!(
            "  {:<28} yaw {:6.1}  shift ({:+5.1}, {:+5.1}) px  IoU {:.3}",
            views[k].name, sol.poses[k].yaw, sol.poses[k].shift[0], sol.poses[k].shift[1], sol.ious[k]
        );
    }
}

fn write_report(dir: &Path, views: &[View], sol: &estimate::Solution, model: &VoxelModel) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut rows = Vec::new();
    for (k, v) in views.iter().enumerate() {
        let cam = sol.cam(k);
        let mm = carve::project_mask(model, v, &cam);
        let iou = carve::iou(&v.mask, &mm);
        let (mut inter, mut a_cnt, mut b_cnt) = (0usize, 0usize, 0usize);
        for i in 0..v.mask.len() {
            if v.mask[i] && mm[i] {
                inter += 1;
            }
            a_cnt += v.mask[i] as usize;
            b_cnt += mm[i] as usize;
        }
        let (recall, precision) = (inter as f32 / a_cnt.max(1) as f32, inter as f32 / b_cnt.max(1) as f32);
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
        // side by side: input | model rendered with the estimated camera (4x, nearest neighbour)
        let rend = refine::render_model(model, v, &cam);
        const S: usize = 4;
        let mut cmp = image::RgbImage::from_pixel((v.w * S * 2 + 8) as u32, (v.h * S) as u32, image::Rgb([90, 120, 90]));
        for y in 0..v.h * S {
            for x in 0..v.w * S {
                let i = (y / S) * v.w + x / S;
                if v.mask[i] {
                    let c = v.rgb[i];
                    cmp.put_pixel(x as u32, y as u32, image::Rgb(c));
                }
                if rend.cov[i] {
                    let c = rend.col[i];
                    cmp.put_pixel((x + v.w * S + 8) as u32, y as u32, image::Rgb([(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8]));
                }
            }
        }
        cmp.save(dir.join(format!("{stem}_compare.png"))).map_err(|e| e.to_string())?;
        rows.push(serde_json::json!({
            "image": v.name,
            "yaw": sol.poses[k].yaw,
            "shift_px": sol.poses[k].shift,
            "anchor_px": [sol.anchors[k][0] + sol.poses[k].shift[0], sol.anchors[k][1] + sol.poses[k].shift[1]],
            "iou_estimate": sol.ious[k],
            "iou_final": iou,
            "recall": recall,
            "precision": precision,
        }));
    }
    let rep = serde_json::json!({ "pitch": sol.pitch, "vscale": sol.vscale, "score": sol.score, "views": rows });
    std::fs::write(dir.join("report.json"), serde_json::to_string_pretty(&rep).unwrap()).map_err(|e| e.to_string())
}
