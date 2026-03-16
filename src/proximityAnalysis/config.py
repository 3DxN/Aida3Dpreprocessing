import json
import os
from pathlib import Path


_DEFAULT_CLASSES = ['']
_DEFAULT_SEGMENTATION_PATH = ['./data/CellposeSegmentations', './data/CellposeSegmentations']
_DEFAULT_RAW_IMAGE_PATH = ['./data/raw']
_DEFAULT_XML_FILES_PATH = [
    './data/raw/012523_303_1_63X_stitchable.xml',
    './data/raw/012523_303_1_63X_stitchable.xml',
]

_DEFAULT_GH2AX_THRESHOLD = [1300, 1300]
_DEFAULT_CD8_THRESHOLD = [750, 1000]
_DEFAULT_GH2AX_PIXEL_COUNT_THRESHOLD = 100
_DEFAULT_CELL_SIZE_THRESHOLD = 200
_DEFAULT_RADII = [32, 48, 64, 128]
_DEFAULT_CD8_DAPI_BOX_OVERLAP_THRESHOLD = 0.2
_DEFAULT_N_PSEUDO_CLASSES = 5


def _load_config_file():
    config_path = os.getenv('AIDA_CONFIG_PATH')
    if not config_path:
        return None, None
    cfg_path = Path(config_path).expanduser()
    try:
        cfg_path = cfg_path.resolve()
        with cfg_path.open('r', encoding='utf-8') as handle:
            cfg = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f'Failed to load config file: {cfg_path}') from exc

    if isinstance(cfg, dict) and isinstance(cfg.get('analysis'), dict):
        cfg = cfg['analysis']

    if not isinstance(cfg, dict):
        raise RuntimeError(f'Config file must be a JSON object: {cfg_path}')

    return cfg, cfg_path.parent


def _csv_env(name):
    value = os.getenv(name)
    if value is None:
        return None
    parts = [part.strip() for part in value.split(',') if part.strip()]
    return parts or None


def _csv_env_typed(name, cast):
    parts = _csv_env(name)
    if parts is None:
        return None
    try:
        return [cast(part) for part in parts]
    except ValueError as exc:
        raise RuntimeError(f'Invalid value for {name}: {parts}') from exc


def _env_number(name, cast):
    value = os.getenv(name)
    if value is None or value == '':
        return None
    try:
        return cast(value)
    except ValueError as exc:
        raise RuntimeError(f'Invalid value for {name}: {value}') from exc


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_paths(paths, base_dir):
    if base_dir is None or paths is None:
        return paths
    resolved = []
    for path in paths:
        if path is None:
            resolved.append(path)
            continue
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        resolved.append(str(candidate))
    return resolved


_cfg, _cfg_base = _load_config_file()

_classes = _csv_env('AIDA_CLASSES')
if _classes is None:
    _classes = _as_list((_cfg or {}).get('classes', _DEFAULT_CLASSES))

_segmentation_path = _csv_env('AIDA_SEGMENTATION_PATH')
if _segmentation_path is None:
    _segmentation_path = _as_list((_cfg or {}).get('segmentation_path', _DEFAULT_SEGMENTATION_PATH))
_segmentation_path = _resolve_paths(_segmentation_path, _cfg_base)

_raw_image_path = _csv_env('AIDA_RAW_IMAGE_PATH')
if _raw_image_path is None:
    _raw_image_path = _as_list((_cfg or {}).get('raw_image_path', _DEFAULT_RAW_IMAGE_PATH))
_raw_image_path = _resolve_paths(_raw_image_path, _cfg_base)

_xml_files_path = _csv_env('AIDA_XML_FILES_PATH')
if _xml_files_path is None:
    _xml_files_path = _as_list((_cfg or {}).get('xml_files_path', _DEFAULT_XML_FILES_PATH))
_xml_files_path = _resolve_paths(_xml_files_path, _cfg_base)

_gh2ax_threshold = _csv_env_typed('AIDA_GH2AX_THRESHOLD', int)
if _gh2ax_threshold is None:
    _gh2ax_threshold = _as_list((_cfg or {}).get('gh2ax_threshold', _DEFAULT_GH2AX_THRESHOLD))

_cd8_threshold = _csv_env_typed('AIDA_CD8_THRESHOLD', int)
if _cd8_threshold is None:
    _cd8_threshold = _as_list((_cfg or {}).get('cd8_threshold', _DEFAULT_CD8_THRESHOLD))

_gh2ax_pixel_count_threshold = _env_number('AIDA_GH2AX_PIXEL_COUNT_THRESHOLD', int)
if _gh2ax_pixel_count_threshold is None:
    _gh2ax_pixel_count_threshold = (
        (_cfg or {}).get('gh2ax_pixel_count_threshold')
        if (_cfg or {}).get('gh2ax_pixel_count_threshold') is not None
        else (_cfg or {}).get('gH2AX__pixel_count_threshold', _DEFAULT_GH2AX_PIXEL_COUNT_THRESHOLD)
    )

_cell_size_threshold = _env_number('AIDA_CELL_SIZE_THRESHOLD', int)
if _cell_size_threshold is None:
    _cell_size_threshold = (_cfg or {}).get('cell_size_threshold', _DEFAULT_CELL_SIZE_THRESHOLD)

_radii = _csv_env_typed('AIDA_RADII', int)
if _radii is None:
    _radii = _as_list((_cfg or {}).get('radii', _DEFAULT_RADII))

_cd8_overlap = _env_number('AIDA_CD8_DAPI_BOX_OVERLAP_THRESHOLD', float)
if _cd8_overlap is None:
    _cd8_overlap = (
        (_cfg or {}).get('cd8_dapi_box_overlap_threshold', _DEFAULT_CD8_DAPI_BOX_OVERLAP_THRESHOLD)
    )

_n_pseudo_classes = _env_number('AIDA_N_PSEUDO_CLASSES', int)
if _n_pseudo_classes is None:
    _n_pseudo_classes = (_cfg or {}).get('n_pseudo_classes', _DEFAULT_N_PSEUDO_CLASSES)


CLASSES = _classes
SEGMENTATION_PATH = _segmentation_path
RAW_IMAGE_PATH = _raw_image_path
XML_FILES_PATH = _xml_files_path

gH2AX_threshold = _gh2ax_threshold
cd8_threshold = _cd8_threshold
gH2AX__pixel_count_threshold = _gh2ax_pixel_count_threshold
cell_size_threshold = _cell_size_threshold

radii = _radii
cd8_dapi_box_overlap_threshold = _cd8_overlap

N_PSEUDO_CLASSES = _n_pseudo_classes
