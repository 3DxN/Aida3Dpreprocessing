configfile: "config.yaml"

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


ROOT = Path(os.getcwd()).resolve()


def _section(name):
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f'Config section "{name}" must be a mapping.')
    return section


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _list_default(value, default):
    values = _as_list(value)
    return values if values else _as_list(default)


def _abs_path(path):
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    return str(candidate)


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _touch(path):
    _ensure_dir(Path(path).parent)
    Path(path).touch()


def _add_arg(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _add_flag(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)


def _latest_run_id(analysis_dir):
    prefixes = ['pct_cd8', 'pct_gh2ax', 'mean_intensity_gh2ax']
    run_sets = []
    for prefix in prefixes:
        matches = list(Path(analysis_dir).glob(f'{prefix}_*.json'))
        run_ids = set()
        for match in matches:
            stem = match.stem
            if stem.startswith(prefix + '_'):
                run_ids.add(stem[len(prefix) + 1 :])
        run_sets.append(run_ids)
    if not run_sets:
        return None
    common = set.intersection(*run_sets) if all(run_sets) else set()
    if not common:
        return None
    return sorted(common)[-1]


def _latest_by_mtime(paths):
    return max(paths, key=lambda path: path.stat().st_mtime)


def _resolve_latest_jsons(analysis_dir):
    prefixes = ['pct_cd8', 'pct_gh2ax', 'mean_intensity_gh2ax']
    run_id = _latest_run_id(analysis_dir)
    resolved = {}
    for prefix in prefixes:
        if run_id:
            candidate = Path(analysis_dir) / f'{prefix}_{run_id}.json'
            if candidate.exists():
                resolved[prefix] = candidate
                continue
        matches = list(Path(analysis_dir).glob(f'{prefix}_*.json'))
        if not matches:
            raise FileNotFoundError(f'No {prefix}_*.json files found in {analysis_dir}')
        resolved[prefix] = _latest_by_mtime(matches)
    return resolved, run_id


def _ensure_csv_alias(analysis_dir, class_name, run_id):
    analysis_dir = Path(analysis_dir)
    dest = analysis_dir / f'{class_name}_texture_features_w_neighbors.csv'
    candidates = []
    if run_id:
        src = analysis_dir / f'{class_name}_texture_features_w_neighbors_{run_id}.csv'
        if src.exists():
            candidates = [src]
    if not candidates:
        candidates = list(analysis_dir.glob(f'{class_name}_texture_features_w_neighbors_*.csv'))
    if candidates:
        src = _latest_by_mtime(candidates)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.symlink(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        return dest
    if dest.exists():
        return dest
    return None


hne_cfg = _section("hne")
seg_cfg = _section("segmentation")
analysis_cfg = _section("analysis")
plot_cfg = _section("plot")
geometry_cfg = _section("geometry")

_output_dir_raw = Path(config.get("output_dir", "data"))
output_dir = _output_dir_raw if _output_dir_raw.is_absolute() else (ROOT / _output_dir_raw).resolve()

_input_dir_raw = Path(config.get("input_dir", output_dir))
input_dir = _input_dir_raw if _input_dir_raw.is_absolute() else (ROOT / _input_dir_raw).resolve()


def _abs_output_path(path):
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    if not _output_dir_raw.is_absolute():
        out_parts = _output_dir_raw.parts
        if out_parts and candidate.parts[:len(out_parts)] == out_parts:
            return str((ROOT / candidate).resolve())
    return str((output_dir / candidate).resolve())


def _abs_input_path(path):
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)

    input_candidate = (input_dir / candidate).resolve()
    if input_candidate.exists():
        return str(input_candidate)

    if input_dir.is_absolute() and candidate.parts:
        if input_dir.name == candidate.parts[0]:
            parent_candidate = (input_dir.parent / candidate).resolve()
            if parent_candidate.exists():
                return str(parent_candidate)

    root_candidate = (ROOT / candidate).resolve()
    if root_candidate.exists():
        return str(root_candidate)

    return str(input_candidate)


state_dir = Path(_abs_output_path(config.get("workflow_state_dir", output_dir / "workflow_state")))
analysis_dir = Path(_abs_output_path(analysis_cfg.get("output_dir", output_dir / "analysis")))

imaris_xml = _abs_input_path(config.get("imaris_xml") or hne_cfg.get("imaris_xml"))
tiff_dir = Path(_abs_output_path(hne_cfg.get("tiff_dir", output_dir / "TIFFtiles")))
tile_json = Path(_abs_output_path(hne_cfg.get("tile_arrangement_file", output_dir / "tileArrangement.json")))

seg_in_dir = Path(_abs_output_path(seg_cfg.get("in_dir", tiff_dir)))
seg_out_dir = Path(_abs_output_path(seg_cfg.get("out_dir", output_dir / "CellposeSegmentations")))
seg_tile_json = Path(_abs_output_path(seg_cfg.get("tile_json", tile_json)))

geometry_enabled = bool(geometry_cfg.get("enabled", False))
geometry_run_starconvex = bool(geometry_cfg.get("run_labels2starconvex", True))
geometry_depends_on_segmentation = bool(geometry_cfg.get("depends_on_segmentation", True))
geometry_input_dir = Path(_abs_output_path(geometry_cfg.get("input_dir", seg_out_dir)))
geometry_output_dir = Path(
    _abs_output_path(geometry_cfg.get("output_dir", output_dir / "Cellpose3DGeometryAndFeatures"))
)
geometry_tile_json = Path(_abs_output_path(geometry_cfg.get("tile_json", tile_json)))
geometry_anisotropy = geometry_cfg.get("anisotropy", 0.0)

geometry_prereq = (
    [str(state_dir / "segmentation.done")] if geometry_depends_on_segmentation else []
)

final_targets = [str(state_dir / "plot.done")]
if geometry_enabled:
    final_targets.append(str(state_dir / "geometry.done"))


rule all:
    input:
        final_targets


rule analysis_config:
    output:
        str(state_dir / "analysis_config.json")
    run:
        classes = _list_default(analysis_cfg.get("classes"), [""])
        num_classes = max(len(classes), 1)

        def expand_list(value, fallback):
            values = _list_default(value, [fallback] * num_classes)
            if len(values) == 1 and num_classes > 1:
                values = values * num_classes
            return values

        if not imaris_xml and not analysis_cfg.get("xml_files_path"):
            raise ValueError('Set "analysis.xml_files_path" or top-level "imaris_xml" in config.yaml.')

        segmentation_path = expand_list(
            analysis_cfg.get("segmentation_path"),
            seg_out_dir,
        )
        raw_image_path = expand_list(
            analysis_cfg.get("raw_image_path"),
            input_dir ,#/ "raw",
        )
        xml_files_path = expand_list(
            analysis_cfg.get("xml_files_path"),
            imaris_xml,
        )

        payload = {
            "analysis": {
                "classes": classes,
                "segmentation_path": [_abs_output_path(p) for p in segmentation_path],
                "raw_image_path": [_abs_input_path(p) for p in raw_image_path],
                "xml_files_path": [_abs_input_path(p) for p in xml_files_path],
            }
        }

        optional_keys = {
            "gh2ax_threshold": "gh2ax_threshold",
            "cd8_threshold": "cd8_threshold",
            "gh2ax_pixel_count_threshold": "gh2ax_pixel_count_threshold",
            "gH2AX__pixel_count_threshold": "gH2AX__pixel_count_threshold",
            "cell_size_threshold": "cell_size_threshold",
            "radii": "radii",
            "cd8_dapi_box_overlap_threshold": "cd8_dapi_box_overlap_threshold",
            "n_pseudo_classes": "n_pseudo_classes",
        }

        for cfg_key, out_key in optional_keys.items():
            if cfg_key in analysis_cfg:
                payload["analysis"][out_key] = analysis_cfg[cfg_key]

        _ensure_dir(Path(output[0]).parent)
        with open(output[0], "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


rule hne:
    input:
        str(imaris_xml)
    output:
        str(state_dir / "hne.done")
    run:
        if not imaris_xml:
            raise ValueError('Missing top-level "imaris_xml" in config.yaml.')

        _ensure_dir(tiff_dir)

        hne_res_level = hne_cfg.get("hne_res_level")
        z_anisotropy = hne_cfg.get("z_anisotropy", 1.0)
        try:
            if hne_res_level is not None and float(hne_res_level) > 0:
                print(
                    f"Warning: hne.hne_res_level={hne_res_level} will downsample XY "
                    "relative to the native Imaris resolution."
                )
        except (TypeError, ValueError):
            print(f"Warning: hne.hne_res_level={hne_res_level} is not a number.")

        try:
            if z_anisotropy is not None and float(z_anisotropy) != 1.0:
                print(
                    f"Warning: hne.z_anisotropy={z_anisotropy} will downsample Z in TIFF output."
                )
        except (TypeError, ValueError):
            print(f"Warning: hne.z_anisotropy={z_anisotropy} is not a number.")

        mask_output = Path(hne_cfg.get("mask_output", "clusteringMask.tif")).name
        nucl_output = Path(hne_cfg.get("nucl_output", "NUCLmaxIntensity.tif")).name
        cyto_output = Path(hne_cfg.get("cyto_output", "CYTOmaxIntensity.tif")).name
        hne_output = Path(hne_cfg.get("hne_output", "HNEmaxIntensity.tif")).name

        cmd = [
            sys.executable,
            str(ROOT / "src/HnE/genHnE.py"),
            str(imaris_xml),
            "--maskFile",
            str(mask_output),
            "--outfileNUCLsubstackMaxIntensity",
            str(nucl_output),
            "--outfileCYTOsubstackMaxIntensity",
            str(cyto_output),
            "--outfileHNEmaxIntensity",
            str(hne_output),
            "--tileArrangementFile",
            str(tile_json),
            "--TIFFwriteout",
            str(tiff_dir),
            str(z_anisotropy),
        ]

        _add_arg(cmd, "--HnEresLevel", hne_cfg.get("hne_res_level"))
        _add_arg(cmd, "--nuclearChannel", hne_cfg.get("nuclear_channel"))
        _add_arg(cmd, "--cytoChannel", hne_cfg.get("cyto_channel"))
        _add_arg(cmd, "--maskThresh", hne_cfg.get("mask_thresh"))
        _add_arg(cmd, "--colorWeightNUCL", hne_cfg.get("color_weight_nucl"))
        _add_arg(cmd, "--colorWeightCYTO", hne_cfg.get("color_weight_cyto"))
        _add_arg(cmd, "--substackThickness", hne_cfg.get("substack_thickness"))
        _add_arg(cmd, "--useZStackFraction", hne_cfg.get("use_z_stack_fraction"))
        _add_arg(cmd, "--fixedZposition", hne_cfg.get("fixed_z_position"))
        _add_arg(cmd, "--imsFilenameFilter", hne_cfg.get("ims_filename_filter"))
        _add_flag(cmd, "--padImgDimsToPowerOf2", hne_cfg.get("pad_img_dims_to_power_of_2", False))
        _add_flag(cmd, "--computeMaskOnly", hne_cfg.get("compute_mask_only", False))

        subprocess.run(cmd, check=True)
        _touch(output[0])


rule segmentation:
    input:
        str(state_dir / "hne.done")
    output:
        str(state_dir / "segmentation.done")
    run:
        _ensure_dir(seg_out_dir)
        cmd = [
            sys.executable,
            str(ROOT / "src/segmentation/predCellpose3D.py"),
            "--inDir",
            str(seg_in_dir),
            "--outDir",
            str(seg_out_dir),
            "--tileJsonFilename",
            str(seg_tile_json),
        ]

        _add_arg(cmd, "--cellposeModel", seg_cfg.get("cellpose_model"))
        _add_arg(cmd, "--cellposeDiameter", seg_cfg.get("cellpose_diameter"))
        _add_arg(cmd, "--batchSize", seg_cfg.get("batch_size"))
        _add_arg(cmd, "--anisotropy", seg_cfg.get("anisotropy"))
        _add_flag(cmd, "--useGPU", seg_cfg.get("use_gpu", False))
        _add_arg(cmd, "--gpu_device", seg_cfg.get("gpu_device"))

        subprocess.run(cmd, check=True)
        _touch(output[0])


rule analysis:
    input:
        str(state_dir / "segmentation.done"),
        str(state_dir / "analysis_config.json"),
    output:
        str(state_dir / "analysis.done")
    run:
        analysis_dir_abs = Path(analysis_dir).resolve()
        _ensure_dir(analysis_dir_abs)
        env = os.environ.copy()
        env["AIDA_CONFIG_PATH"] = str(Path(input[1]).resolve())

        subprocess.run(
            [sys.executable, str(ROOT / "src/proximityAnalysis/analyze_multi.py")],
            check=True,
            cwd=str(analysis_dir_abs),
            env=env,
        )
        _touch(output[0])


rule plot:
    input:
        str(state_dir / "analysis.done"),
    output:
        str(state_dir / "plot.done")
    run:
        plot_classes = plot_cfg.get("classes") or analysis_cfg.get("classes")
        if not plot_classes or len(plot_classes) != 2:
            raise ValueError('Plotting requires exactly two classes in plot.classes or analysis.classes.')

        analysis_dir_abs = Path(analysis_dir).resolve()
        json_inputs, run_id = _resolve_latest_jsons(analysis_dir_abs)
        json_inputs = {key: Path(path).resolve() for key, path in json_inputs.items()}  # Avoid 'path-doubling' issues

        for class_name in plot_classes:
            csv_alias = _ensure_csv_alias(analysis_dir_abs, class_name, run_id)
            if csv_alias is None:
                raise FileNotFoundError(
                    f'No texture feature CSV found for class "{class_name}" in {analysis_dir_abs}'
                )

        num_pseudo = plot_cfg.get("num_pseudo_classes", analysis_cfg.get("n_pseudo_classes", 5))
        plot_format = plot_cfg.get("plot_output_format", "pdf")

        cmd = [
            sys.executable,
            str(ROOT / "src/proximityAnalysis/plotRatios.py"),
            str(json_inputs["pct_cd8"]),
            str(json_inputs["pct_gh2ax"]),
            str(json_inputs["mean_intensity_gh2ax"]),
            "-p",
            str(plot_format),
            "-c",
            str(plot_classes[0]),
            str(plot_classes[1]),
            "--num_pseudo_classes",
            str(num_pseudo),
        ]

        subprocess.run(cmd, check=True, cwd=str(analysis_dir_abs))
        _touch(output[0])


rule labels2starconvex3d:
    input:
        geometry_prereq
    output:
        str(state_dir / "starconvex.done")
    run:
        if not geometry_run_starconvex:
            _touch(output[0])
            return

        if geometry_anisotropy <= 0 and not geometry_tile_json:
            raise ValueError('Set geometry.tile_json when geometry.anisotropy <= 0.')

        cmd = [
            sys.executable,
            str(ROOT / "src/geometry/labels2starconvex3D.py"),
            str(geometry_input_dir),
            str(geometry_anisotropy),
        ]
        if geometry_anisotropy <= 0:
            cmd.append(str(geometry_tile_json))

        subprocess.run(cmd, check=True)
        _touch(output[0])


rule extract_vis_data:
    input:
        str(state_dir / "starconvex.done")
    output:
        str(state_dir / "geometry.done")
    run:
        _ensure_dir(geometry_output_dir)
        cmd = [
            sys.executable,
            str(ROOT / "src/geometry/extractVisData.py"),
            "--inDir",
            str(geometry_input_dir),
            "--outDir",
            str(geometry_output_dir),
            "--tileFile",
            str(geometry_tile_json),
        ]

        _add_flag(cmd, "--computeHaralick", geometry_cfg.get("compute_haralick", False))
        _add_flag(cmd, "--separateFeatureFiles", geometry_cfg.get("separate_feature_files", False))

        subprocess.run(cmd, check=True)
        _touch(output[0])
