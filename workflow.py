#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_config(path):
    with open(path, 'r', encoding='utf-8') as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config file must be a JSON object: {path}')
    return cfg


def resolve_path(path, base_dir):
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def get_section(cfg, name):
    section = cfg.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f'Config section "{name}" must be an object.')
    return section


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def cmd_to_str(cmd):
    return ' '.join(shlex.quote(str(part)) for part in cmd)


def run_cmd(cmd, cwd=None, env=None, dry_run=False):
    print(f'  cmd: {cmd_to_str(cmd)}')
    if cwd:
        print(f'  cwd: {cwd}')
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def has_any(directory, pattern):
    return any(Path(directory).glob(pattern))


def should_skip_hne(tile_json, tiff_dir):
    return tile_json.exists() and has_any(tiff_dir, '*.tif')


def should_skip_segmentation(seg_out_dir):
    return seg_out_dir.exists() and has_any(seg_out_dir, '*_CELLPOSE-LABELS*.tif')


def should_skip_analysis(analysis_dir):
    return analysis_dir.exists() and has_any(analysis_dir, 'pct_cd8_*.json')


def should_skip_plot(analysis_dir, output_format):
    if not analysis_dir.exists():
        return False
    return (
        has_any(analysis_dir, f'gh2ax_ratio_*.{output_format}')
        and has_any(analysis_dir, f'cd8_ratio_*.{output_format}')
        and has_any(analysis_dir, f'mean_intensity_gh2ax_*.{output_format}')
    )


def add_arg(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


def add_flag(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)


def select_latest_run_id(analysis_dir):
    prefixes = ['pct_cd8', 'pct_gh2ax', 'mean_intensity_gh2ax']
    run_sets = []
    for prefix in prefixes:
        matches = list(analysis_dir.glob(f'{prefix}_*.json'))
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


def latest_by_mtime(paths):
    return max(paths, key=lambda path: path.stat().st_mtime)


def ensure_csv_alias(analysis_dir, class_name, run_id, dry_run=False):
    dest = analysis_dir / f'{class_name}_texture_features_w_neighbors.csv'
    if run_id:
        src = analysis_dir / f'{class_name}_texture_features_w_neighbors_{run_id}.csv'
        if src.exists():
            if not dry_run:
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


def resolve_analysis_inputs(analysis_dir):
    prefixes = ['pct_cd8', 'pct_gh2ax', 'mean_intensity_gh2ax']
    run_id = select_latest_run_id(analysis_dir)
    resolved = {}
    for prefix in prefixes:
        if run_id:
            candidate = analysis_dir / f'{prefix}_{run_id}.json'
            if candidate.exists():
                resolved[prefix] = candidate
                continue
        matches = list(analysis_dir.glob(f'{prefix}_*.json'))
        if not matches:
            raise FileNotFoundError(f'No {prefix}_*.json files found in {analysis_dir}')
        resolved[prefix] = latest_by_mtime(matches)
    return resolved, run_id


def build_context(cfg, config_path):
    base_dir = config_path.parent
    output_dir = resolve_path(cfg.get('output_dir', 'data'), base_dir)
    hne_cfg = get_section(cfg, 'hne')
    seg_cfg = get_section(cfg, 'segmentation')
    analysis_cfg = get_section(cfg, 'analysis')
    plot_cfg = get_section(cfg, 'plot')

    imaris_xml = resolve_path(cfg.get('imaris_xml') or hne_cfg.get('imaris_xml'), base_dir)
    tiff_dir = resolve_path(hne_cfg.get('tiff_dir', output_dir / 'TIFFtiles'), base_dir)
    tile_json = resolve_path(hne_cfg.get('tile_arrangement_file', output_dir / 'tileArrangement.json'), base_dir)

    seg_in_dir = resolve_path(seg_cfg.get('in_dir', tiff_dir), base_dir)
    seg_out_dir = resolve_path(seg_cfg.get('out_dir', output_dir / 'CellposeSegmentations'), base_dir)
    seg_tile_json = resolve_path(seg_cfg.get('tile_json', tile_json), base_dir)

    analysis_dir = resolve_path(analysis_cfg.get('output_dir', output_dir / 'analysis'), base_dir)

    return {
        'base_dir': base_dir,
        'output_dir': output_dir,
        'hne_cfg': hne_cfg,
        'seg_cfg': seg_cfg,
        'analysis_cfg': analysis_cfg,
        'plot_cfg': plot_cfg,
        'imaris_xml': imaris_xml,
        'tiff_dir': tiff_dir,
        'tile_json': tile_json,
        'seg_in_dir': seg_in_dir,
        'seg_out_dir': seg_out_dir,
        'seg_tile_json': seg_tile_json,
        'analysis_dir': analysis_dir,
    }


def run_hne(ctx, force=False, dry_run=False):
    if ctx['imaris_xml'] is None:
        raise ValueError('Missing "imaris_xml" in config (top-level or hne section).')
    output_dir = ctx['output_dir']
    tiff_dir = ctx['tiff_dir']
    tile_json = ctx['tile_json']
    hne_cfg = ctx['hne_cfg']

    if not dry_run:
        ensure_dir(output_dir)

    if not force and should_skip_hne(tile_json, tiff_dir):
        print('HnE step: outputs detected, skipping. Use --force to re-run.')
        return

    cmd = [
        sys.executable,
        str(ROOT / 'src/HnE/genHnE.py'),
        str(ctx['imaris_xml']),
        '--maskFile',
        str(resolve_path(hne_cfg.get('mask_output', output_dir / 'clusteringMask.tif'), ctx['base_dir'])),
        '--outfileNUCLsubstackMaxIntensity',
        str(resolve_path(hne_cfg.get('nucl_output', output_dir / 'NUCLmaxIntensity.tif'), ctx['base_dir'])),
        '--outfileCYTOsubstackMaxIntensity',
        str(resolve_path(hne_cfg.get('cyto_output', output_dir / 'CYTOmaxIntensity.tif'), ctx['base_dir'])),
        '--outfileHNEmaxIntensity',
        str(resolve_path(hne_cfg.get('hne_output', output_dir / 'HNEmaxIntensity.tif'), ctx['base_dir'])),
        '--tileArrangementFile',
        str(tile_json),
        '--TIFFwriteout',
        str(tiff_dir),
        str(hne_cfg.get('z_anisotropy', 2.0)),
    ]

    add_arg(cmd, '--HnEresLevel', hne_cfg.get('hne_res_level'))
    add_arg(cmd, '--nuclearChannel', hne_cfg.get('nuclear_channel'))
    add_arg(cmd, '--cytoChannel', hne_cfg.get('cyto_channel'))
    add_arg(cmd, '--maskThresh', hne_cfg.get('mask_thresh'))
    add_arg(cmd, '--colorWeightNUCL', hne_cfg.get('color_weight_nucl'))
    add_arg(cmd, '--colorWeightCYTO', hne_cfg.get('color_weight_cyto'))
    add_arg(cmd, '--substackThickness', hne_cfg.get('substack_thickness'))
    add_arg(cmd, '--useZStackFraction', hne_cfg.get('use_z_stack_fraction'))
    add_arg(cmd, '--fixedZposition', hne_cfg.get('fixed_z_position'))
    add_arg(cmd, '--imsFilenameFilter', hne_cfg.get('ims_filename_filter'))
    add_flag(cmd, '--padImgDimsToPowerOf2', hne_cfg.get('pad_img_dims_to_power_of_2', False))
    add_flag(cmd, '--computeMaskOnly', hne_cfg.get('compute_mask_only', False))

    print('HnE step:')
    run_cmd(cmd, dry_run=dry_run)


def run_segmentation(ctx, force=False, dry_run=False):
    seg_out_dir = ctx['seg_out_dir']
    seg_cfg = ctx['seg_cfg']
    if not dry_run:
        ensure_dir(seg_out_dir)

    if not force and should_skip_segmentation(seg_out_dir):
        print('Segmentation step: outputs detected, skipping. Use --force to re-run.')
        return

    cmd = [
        sys.executable,
        str(ROOT / 'src/segmentation/predCellpose3D.py'),
        '--inDir',
        str(ctx['seg_in_dir']),
        '--outDir',
        str(seg_out_dir),
        '--tileJsonFilename',
        str(ctx['seg_tile_json']),
    ]

    add_arg(cmd, '--cellposeModel', seg_cfg.get('cellpose_model'))
    add_arg(cmd, '--cellposeDiameter', seg_cfg.get('cellpose_diameter'))
    add_arg(cmd, '--batchSize', seg_cfg.get('batch_size'))
    add_arg(cmd, '--anisotropy', seg_cfg.get('anisotropy'))
    add_flag(cmd, '--useGPU', seg_cfg.get('use_gpu', False))
    add_arg(cmd, '--gpu_device', seg_cfg.get('gpu_device'))

    print('Segmentation step:')
    run_cmd(cmd, dry_run=dry_run)


def run_analysis(ctx, config_path, force=False, dry_run=False):
    if not ctx['analysis_cfg']:
        raise ValueError('Missing "analysis" section in config.')

    analysis_dir = ctx['analysis_dir']
    if not dry_run:
        ensure_dir(analysis_dir)

    if not force and should_skip_analysis(analysis_dir):
        print('Analysis step: outputs detected, skipping. Use --force to re-run.')
        return

    cmd = [sys.executable, str(ROOT / 'src/proximityAnalysis/analyze_multi.py')]
    env = os.environ.copy()
    env['AIDA_CONFIG_PATH'] = str(config_path)

    print('Analysis step:')
    run_cmd(cmd, cwd=analysis_dir, env=env, dry_run=dry_run)


def run_plot(ctx, force=False, dry_run=False):
    plot_cfg = ctx['plot_cfg']
    analysis_cfg = ctx['analysis_cfg']
    analysis_dir = ctx['analysis_dir']

    if not analysis_dir.exists():
        if dry_run:
            print(f'Plot step: analysis directory not found: {analysis_dir}. Skipping in dry-run.')
            return
        raise FileNotFoundError(f'Analysis directory not found: {analysis_dir}')

    plot_format = plot_cfg.get('plot_output_format', 'pdf')
    if not force and should_skip_plot(analysis_dir, plot_format):
        print('Plot step: outputs detected, skipping. Use --force to re-run.')
        return

    classes = plot_cfg.get('classes') or analysis_cfg.get('classes')
    if not classes or len(classes) != 2:
        raise ValueError('Plot step requires exactly two classes (set plot.classes or analysis.classes).')

    try:
        inputs, run_id = resolve_analysis_inputs(analysis_dir)
    except FileNotFoundError as exc:
        if dry_run:
            print(f'Plot step: {exc}. Skipping in dry-run.')
            return
        raise

    for class_name in classes:
        csv_alias = ensure_csv_alias(analysis_dir, class_name, run_id, dry_run=dry_run)
        if csv_alias is None:
            raise FileNotFoundError(
                f'No texture feature CSV found for class "{class_name}" in {analysis_dir}'
            )

    num_pseudo = plot_cfg.get('num_pseudo_classes', analysis_cfg.get('n_pseudo_classes', 5))

    cmd = [
        sys.executable,
        str(ROOT / 'src/proximityAnalysis/plotRatios.py'),
        str(inputs['pct_cd8']),
        str(inputs['pct_gh2ax']),
        str(inputs['mean_intensity_gh2ax']),
        '-p',
        str(plot_format),
        '-c',
        str(classes[0]),
        str(classes[1]),
        '--num_pseudo_classes',
        str(num_pseudo),
    ]

    print('Plot step:')
    run_cmd(cmd, cwd=analysis_dir, dry_run=dry_run)


def default_config_path():
    candidate = ROOT / 'workflow.json'
    return candidate if candidate.exists() else None


def main():
    parser = argparse.ArgumentParser(description='Workflow manager for AIDA-3D preprocessing.')
    parser.add_argument('--config', type=str, default=None, help='Path to workflow JSON config.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing.')
    parser.add_argument('--force', action='store_true', help='Re-run steps even if outputs exist.')

    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ['hne', 'segmentation', 'analysis', 'plot', 'all']:
        subparsers.add_parser(name)

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path()
    if config_path is None:
        raise FileNotFoundError('No config provided. Use --config to specify a workflow JSON file.')
    config_path = config_path.expanduser().resolve()

    cfg = load_config(config_path)
    ctx = build_context(cfg, config_path)

    if args.command == 'hne':
        run_hne(ctx, force=args.force, dry_run=args.dry_run)
        return
    if args.command == 'segmentation':
        run_segmentation(ctx, force=args.force, dry_run=args.dry_run)
        return
    if args.command == 'analysis':
        run_analysis(ctx, config_path, force=args.force, dry_run=args.dry_run)
        return
    if args.command == 'plot':
        run_plot(ctx, force=args.force, dry_run=args.dry_run)
        return

    run_hne(ctx, force=args.force, dry_run=args.dry_run)
    run_segmentation(ctx, force=args.force, dry_run=args.dry_run)
    if ctx['analysis_cfg']:
        run_analysis(ctx, config_path, force=args.force, dry_run=args.dry_run)
        run_plot(ctx, force=args.force, dry_run=args.dry_run)
    else:
        print('Analysis section not found in config. Skipping analysis and plot steps.')


if __name__ == '__main__':
    main()
