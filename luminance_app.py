import sys
import os
import tempfile
import json
import math
import re
import numpy as np
import rawpy
import cv2
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit,
                             QProgressBar, QMessageBox, QListWidget, QPlainTextEdit,
                             QGroupBox, QSlider, QAbstractSpinBox, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QTextCursor, QImage, QPixmap


STRETCH_PERCENT_SCALE = 1000

APP_NAME = "Luminance Extractor for Photometric Stereo (Beta)"
APP_VERSION = "1.0.0"
APP_WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION}"

def get_exiftool_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'exiftool.exe')

def get_subprocess_flags():
    if sys.platform == 'win32':
        return subprocess.CREATE_NO_WINDOW
    return 0

def is_tiff_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return ext in ['.tif', '.tiff']

def validate_tiff_16bit(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read TIFF file: {filepath}")
    if img.dtype != np.uint16:
        raise ValueError(f"TIFF must be 16-bit linear: {os.path.basename(filepath)}")
    return img


def parse_lp_light_positions(lp_path, lights_distance_mm):
    with open(lp_path, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(".lp file is empty")

    if len(lines) > 0 and re.fullmatch(r"[-+]?\d+", lines[0].strip()):
        lines = lines[1:]

    raw_positions = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 4:
            continue
        try:
            x = float(tokens[-3])
            y = float(tokens[-2])
            z = float(tokens[-1])
        except ValueError:
            continue
        raw_positions.append((x, y, z))

    if not raw_positions:
        raise ValueError("No valid light directions found in .lp file")
    if lights_distance_mm is None:
        raise ValueError("lights distance (mm) is required for .lp files")

    positions = []
    radius = float(lights_distance_mm)
    for x, y, z in raw_positions:
        n = math.sqrt(x * x + y * y + z * z)
        if n <= 1e-9:
            continue
        scale = radius / n
        positions.append((x * scale, y * scale, z * scale))

    if not positions:
        raise ValueError("No valid light directions found in .lp file")
    return positions


def get_embedded_dome_radius_mm(dome_path):
    with open(dome_path, 'r', encoding='utf-8-sig') as f:
        dome_data = json.load(f)

    diameter = dome_data.get('domeDiameter')
    if diameter is None:
        return None

    try:
        diameter_value = float(diameter)
    except (TypeError, ValueError):
        return None

    if diameter_value <= 0:
        return None
    return diameter_value / 2.0


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text):
        self.textWritten.emit(str(text))
    def flush(self):
        pass

class DarkLevelAnalysisThread(QThread):
    progress = pyqtSignal(int)
    analysis_done = pyqtSignal(dict)
    analysis_failed = pyqtSignal(str)

    def __init__(self, files, metadata_enabled, current_coeff, sample_step=4,
                 manual_dark_active=False, manual_dark_cap_on=0.0, manual_dark_offset=0.0):
        super().__init__()
        self.files = list(files)
        self.metadata_enabled = bool(metadata_enabled)
        self.current_coeff = float(np.clip(current_coeff, -0.99, 0.99))
        self.sample_step = max(1, int(sample_step))
        self.manual_dark_active = bool(manual_dark_active)
        self.manual_dark_cap_on = max(0.0, float(manual_dark_cap_on))
        self.manual_dark_offset = max(0.0, float(manual_dark_offset))

    def _effective_dark_level(self, dark_level):
        return max(0.0, float(dark_level) * (1.0 - self.current_coeff))

    def _manual_dark_total_65535(self):
        scale = max(0.0, 1.0 - self.current_coeff)
        return (self.manual_dark_cap_on + self.manual_dark_offset) * 65535.0 * scale

    def run(self):
        try:
            raw_exts = {'cr2', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2', 'raw'}
            raw_files = [f for f in self.files if os.path.splitext(f)[1].lower().lstrip('.') in raw_exts]
            if not raw_files:
                self.analysis_done.emit({'status': 'no_raw'})
                return

            neg_ratios = []
            coeff_needs = []
            clipped_files = []
            failed_files = []
            total_pixels = 0
            total_negative = 0

            for idx_path, path in enumerate(raw_files):
                try:
                    with rawpy.imread(path) as raw:
                        raw_vis = raw.raw_image_visible.astype(np.float32)
                        colors = raw.raw_colors_visible
                        black_levels = [float(v) for v in list(raw.black_level_per_channel)]

                        if not any(v > 0 for v in black_levels):
                            continue

                        residual = raw_vis.copy()
                        per_file_coeff_need = 0.0

                        if self.metadata_enabled:
                            for chan_idx, level in enumerate(black_levels):
                                level_eff = self._effective_dark_level(level)
                                mask = (colors == chan_idx)
                                if np.any(mask):
                                    residual[mask] -= level_eff

                                    if level > 0:
                                        chan = raw_vis[mask][::self.sample_step]
                                        if chan.size > 0:
                                            p01 = float(np.percentile(chan, 1.0))
                                            needed = max(0.0, min(0.99, (level - p01) / level))
                                            per_file_coeff_need = max(per_file_coeff_need, needed)
                        elif self.manual_dark_active:
                            dark_total = self._manual_dark_total_65535()
                            residual -= dark_total
                            if dark_total > 0:
                                sample_raw = raw_vis[::self.sample_step, ::self.sample_step]
                                if sample_raw.size > 0:
                                    p01 = float(np.percentile(sample_raw, 1.0))
                                    needed = max(-0.99, min(0.99, 1.0 - (p01 / dark_total)))
                                    per_file_coeff_need = max(per_file_coeff_need, needed)

                        sample = residual[::self.sample_step, ::self.sample_step]
                        neg_count = int(np.count_nonzero(sample < 0.0))
                        px_count = int(sample.size)
                        neg_ratio = (float(neg_count) / float(px_count) * 100.0) if px_count > 0 else 0.0

                        total_negative += neg_count
                        total_pixels += px_count
                        neg_ratios.append(neg_ratio)
                        coeff_needs.append(per_file_coeff_need)

                        if neg_ratio > 0.05:
                            clipped_files.append((os.path.basename(path), neg_ratio, per_file_coeff_need))
                except Exception as e:
                    failed_files.append((os.path.basename(path), str(e)))

                self.progress.emit(int((idx_path + 1) / len(raw_files) * 100))

            if not neg_ratios and not failed_files:
                self.analysis_done.emit({'status': 'no_valid'})
                return

            clipped_files.sort(key=lambda x: x[1], reverse=True)
            clipped_count = len(clipped_files)
            analyzed_count = len(neg_ratios)
            global_neg_ratio = (float(total_negative) / float(total_pixels) * 100.0) if total_pixels > 0 else 0.0
            max_neg_ratio = max(neg_ratios) if neg_ratios else 0.0
            mean_neg_ratio = float(np.mean(neg_ratios)) if neg_ratios else 0.0
            negligible_clipping = (global_neg_ratio > 0.0) and (global_neg_ratio < 1e-6)

            suggested_coeff = self.current_coeff
            if self.metadata_enabled and coeff_needs:
                robust_need = float(np.percentile(np.array(coeff_needs, dtype=np.float32), 90.0))
                suggested_coeff = max(self.current_coeff, min(0.99, robust_need))
            if clipped_count == 0:
                suggested_coeff = self.current_coeff

            self.analysis_done.emit({
                'status': 'ok',
                'metadata_enabled': self.metadata_enabled,
                'manual_dark_active': self.manual_dark_active,
                'current_coeff': self.current_coeff,
                'suggested_coeff': suggested_coeff,
                'raw_files_considered': len(raw_files),
                'raw_files_analyzed': analyzed_count,
                'clipped_count': clipped_count,
                'mean_neg_ratio': mean_neg_ratio,
                'max_neg_ratio': max_neg_ratio,
                'global_neg_ratio': global_neg_ratio,
                'negligible_clipping': negligible_clipping,
                'clipped_files': clipped_files,
                'failed_files': failed_files
            })
        except Exception as e:
            self.analysis_failed.emit(str(e))

class ProcessThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)
    analysis_threshold = pyqtSignal(float)

    def __init__(self, files, options):
        super().__init__()
        self.files = files
        self.options = options
        self._is_running = True
        self.excluded_burned_files = []
        self._excluded_burned_file_set = set()
        self._linearity_lut_x = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
        self._linearity_lut_values = self._build_linearity_lut_values()

    def stop(self):
        self._is_running = False

    def _build_linearity_lut_values(self):
        if not bool(self.options.get('linearity_calibration_enabled', False)):
            return None

        points = self.options.get('linearity_lut_control_points', None)
        if not isinstance(points, list) or len(points) < 2:
            return None

        clean = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            try:
                xin = float(p[0])
                yout = float(p[1])
            except Exception:
                continue
            clean.append((float(np.clip(xin, 0.0, 1.0)), float(np.clip(yout, 0.0, 1.0))))

        if len(clean) < 2:
            return None

        clean.sort(key=lambda t: (t[0], t[1]))
        collapsed = []
        for xin, yout in clean:
            if collapsed and abs(xin - collapsed[-1][0]) < 1e-8:
                collapsed[-1] = (collapsed[-1][0], max(collapsed[-1][1], yout))
            else:
                collapsed.append((xin, yout))

        if collapsed[0][0] > 0.0:
            collapsed.insert(0, (0.0, 0.0))
        if collapsed[-1][0] < 1.0:
            collapsed.append((1.0, 1.0))

        x = np.array([p[0] for p in collapsed], dtype=np.float32)
        y = np.array([p[1] for p in collapsed], dtype=np.float32)
        y = np.maximum.accumulate(y)
        y = np.clip(y, 0.0, 1.0)
        return np.interp(self._linearity_lut_x, x, y).astype(np.float32)

    def _apply_linearity_lut(self, y_norm):
        if self._linearity_lut_values is None:
            return y_norm
        src = np.clip(y_norm, 0.0, 1.0).astype(np.float32)
        dst = np.interp(src.reshape(-1), self._linearity_lut_x, self._linearity_lut_values)
        return dst.reshape(src.shape).astype(np.float32)

    def _is_raw_file(self, filepath):
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        return ext in {'cr2', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2', 'raw'}

    def _read_linear_rgb_65535(self, filepath, demosaic):
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        if ext in {'tif', 'tiff'}:
            img = validate_tiff_16bit(filepath)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.float32)

        with rawpy.imread(filepath) as raw:
            rgb = raw.postprocess(
                half_size=False,
                demosaic_algorithm=demosaic,
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=True,
                no_auto_scale=True,
                use_camera_wb=False,
                user_black=0,
                user_wb=[1.0, 1.0, 1.0, 1.0]
            ).astype(np.float32)
        return rgb

    def _match_dark_map_shape(self, dark_map, target_h, target_w):
        if dark_map is None:
            return None
        if dark_map.shape[0] == target_h and dark_map.shape[1] == target_w:
            return dark_map
        return cv2.resize(dark_map, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _bayer2x2_mean_map(self, mono_2d):
        h, w = mono_2d.shape[:2]
        h2 = (h // 2) * 2
        w2 = (w // 2) * 2
        if h2 == 0 or w2 == 0:
            return np.zeros((1, 1), dtype=np.float32)
        src = mono_2d[:h2, :w2]
        return (src[0::2, 0::2] + src[0::2, 1::2] + src[1::2, 0::2] + src[1::2, 1::2]) / 4.0

    def _read_bayer2x2_luma_65535(self, filepath):
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        if ext in {'tif', 'tiff'}:
            img = validate_tiff_16bit(filepath)
            if len(img.shape) == 2:
                mono = img.astype(np.float32)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                mono = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) / 3.0
            return self._bayer2x2_mean_map(mono)

        with rawpy.imread(filepath) as raw:
            raw_vis = raw.raw_image_visible.astype(np.float32)
        return self._bayer2x2_mean_map(raw_vis)

    def _compute_luminance_from_rgb(self, rgb, use_weighted=False, weight_r=1.0, weight_g=1.0, weight_b=1.0):
        if use_weighted:
            denom = max(1e-8, float(weight_r + weight_g + weight_b))
            lum = (weight_r * rgb[:, :, 0] + weight_g * rgb[:, :, 1] + weight_b * rgb[:, :, 2]) / denom
        else:
            lum = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) / 3.0
        return np.clip(lum / 65535.0, 0.0, None)

    def _get_raw_dark_level(self, raw):
        levels = [float(v) for v in list(raw.black_level_per_channel) if float(v) > 0]
        if not levels:
            return None
        return float(sum(levels) / len(levels))

    def _build_light_gain_lookup(self):
        raw_map = self.options.get('light_compensation_map', {})
        if not isinstance(raw_map, dict):
            return {}

        gain_map = {}
        for name, gain in raw_map.items():
            try:
                gain_val = float(gain)
            except Exception:
                continue
            gain_val = float(np.clip(gain_val, 0.05, 20.0))
            gain_map[str(name).lower()] = gain_val
        return gain_map

    def _gain_for_file(self, filepath, gain_map):
        if not gain_map:
            return 1.0
        base = os.path.basename(filepath).lower()
        return float(gain_map.get(base, 1.0))

    def _build_flatfield_lookup(self):
        raw_map = self.options.get('flatfield_map', {})
        if not isinstance(raw_map, dict):
            return {}
        lookup = {}
        for input_name, grey_path in raw_map.items():
            if not grey_path:
                continue
            key = str(input_name).lower()
            lookup[key] = str(grey_path)
        return lookup

    def _apply_grey_rotation_to_map(self, y_map):
        if y_map is None or not bool(self.options.get('grey_rotation_enabled', False)):
            return y_map
        angle_idx = int(self.options.get('grey_rotation_angle', 0))
        if angle_idx == 0:
            return np.rot90(y_map, k=3)
        if angle_idx == 1:
            return np.rot90(y_map, k=2)
        return np.rot90(y_map, k=1)

    def _center_patch_trimmed_mean(self, map_2d, patch_size=20):
        h, w = map_2d.shape[:2]
        p = max(8, min(int(patch_size), h, w))
        y0 = (h - p) // 2
        x0 = (w - p) // 2
        patch = map_2d[y0:y0 + p, x0:x0 + p]
        if patch.size == 0:
            return float(np.mean(map_2d))
        low = float(np.percentile(patch, 5.0))
        high = float(np.percentile(patch, 95.0))
        trimmed = patch[(patch >= low) & (patch <= high)]
        if trimmed.size == 0:
            return float(np.mean(patch))
        return float(np.mean(trimmed))

    def _load_flatfield_luma_norm(self, grey_path, demosaic, apply_dark_level, dark_frame_offset,
                                  dark_map, dark_bayer_map, dark_lift_coeff):
        luminance_mode = self.options.get('luminance_mode', 'demosaic_mean')

        if luminance_mode == 'raw_bayer_2x2':
            flat_y, _ = self.extract_sensor_bayer2x2_map(
                filepath=grey_path,
                sharp_radius=1.0,
                sharp_amount=0.0,
                downsample_factor=1,
                apply_dark_level=apply_dark_level,
                dark_frame_offset=dark_frame_offset,
                dark_bayer_map=dark_bayer_map,
                dark_lift_coeff=dark_lift_coeff,
            )
        else:
            rgb_linear = self.load_and_linearize(
                grey_path,
                demosaic=demosaic,
                apply_dark_level=apply_dark_level,
                dark_frame_offset=dark_frame_offset,
                dark_map=dark_map,
                dark_lift_coeff=dark_lift_coeff,
            )
            use_weighted = luminance_mode == 'demosaic_weighted'
            weight_r = float(self.options.get('weight_r', 1.0))
            weight_g = float(self.options.get('weight_g', 2.0))
            weight_b = float(self.options.get('weight_b', 1.0))
            flat_y = self._compute_luminance_from_rgb(rgb_linear, use_weighted, weight_r, weight_g, weight_b)

        # Keep flat-map photometry in the same domain as input maps before division.
        return self._apply_grey_rotation_to_map(self._apply_linearity_lut(flat_y))

    def _build_flatfield_gain_map(self, flat_y):
        if flat_y is None or flat_y.size == 0:
            return None

        src = np.maximum(flat_y.astype(np.float32), 1e-8)
        min_side = float(max(1, min(src.shape[0], src.shape[1])))
        sigma_rel = float(np.clip(self.options.get('flatfield_smooth_sigma_rel', 0.02), 0.0, 0.25))
        sigma = sigma_rel * min_side

        if sigma >= 0.5:
            log_src = np.log(src)
            log_lp = cv2.GaussianBlur(
                log_src,
                (0, 0),
                sigmaX=float(sigma),
                sigmaY=float(sigma),
                borderType=cv2.BORDER_REPLICATE
            )
            src_lp = np.exp(log_lp).astype(np.float32)
        else:
            src_lp = src

        center_ref = self._center_patch_trimmed_mean(src_lp, patch_size=20)
        if center_ref <= 1e-8:
            return None

        gain = center_ref / np.maximum(src_lp, 1e-8)
        return gain.astype(np.float32)

    def _apply_flatfield_to_luma_map(self, input_path, y_norm, flat_lookup, demosaic,
                                     apply_dark_level, dark_frame_offset, dark_map,
                                     dark_bayer_map, dark_lift_coeff):
        if not flat_lookup:
            return y_norm
        key = os.path.basename(input_path).lower()
        grey_path = flat_lookup.get(key)
        if not grey_path or (not os.path.isfile(grey_path)):
            return y_norm

        flat_y = self._load_flatfield_luma_norm(
            grey_path,
            demosaic=demosaic,
            apply_dark_level=apply_dark_level,
            dark_frame_offset=dark_frame_offset,
            dark_map=dark_map,
            dark_bayer_map=dark_bayer_map,
            dark_lift_coeff=dark_lift_coeff,
        )
        if flat_y.shape[:2] != y_norm.shape[:2]:
            flat_y = cv2.resize(flat_y, (y_norm.shape[1], y_norm.shape[0]), interpolation=cv2.INTER_LINEAR)

        gain_map = self._build_flatfield_gain_map(flat_y)
        if gain_map is None:
            return y_norm
        corrected = y_norm * gain_map
        return np.clip(corrected, 0.0, None)

    def _effective_dark_level(self, dark_level, dark_lift_coeff):
        coeff = float(np.clip(dark_lift_coeff, -0.99, 0.99))
        return max(0.0, float(dark_level) * (1.0 - coeff))

    def _apply_raw_black_subtraction_per_channel(self, raw, dark_lift_coeff):
        dark_levels = [float(v) for v in list(raw.black_level_per_channel)]
        if not any(v > 0 for v in dark_levels):
            return None

        effective_levels = [self._effective_dark_level(v, dark_lift_coeff) for v in dark_levels]
        raw_vis_ref = raw.raw_image_visible
        raw_vis = raw_vis_ref.astype(np.float32)
        colors = raw.raw_colors_visible

        for chan_idx, level_eff in enumerate(effective_levels):
            if level_eff <= 0:
                continue
            mask = (colors == chan_idx)
            if np.any(mask):
                raw_vis[mask] -= level_eff

        raw_vis = np.clip(raw_vis, 0.0, 65535.0)
        raw_vis_ref[:] = raw_vis.astype(raw_vis_ref.dtype)

        positive = [v for v in effective_levels if v > 0]
        if not positive:
            return None
        return float(sum(positive) / len(positive))

    def _manual_dark_scale(self, dark_lift_coeff):
        return max(0.0, 1.0 - float(np.clip(dark_lift_coeff, -0.99, 0.99)))

    def _postprocess_raw_linear_rgb(self, raw, demosaic, half_size, apply_dark_level,
                                    dark_frame_offset=0.0, dark_map=None, dark_lift_coeff=0.0):
        kwargs = dict(
            half_size=half_size,
            demosaic_algorithm=demosaic,
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            no_auto_scale=True,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0]
        )

        dark_used = None
        if apply_dark_level:
            dark_used = self._apply_raw_black_subtraction_per_channel(raw, dark_lift_coeff)
            kwargs['user_black'] = 0
        else:
            kwargs['user_black'] = 0

        rgb = raw.postprocess(**kwargs).astype(np.float32)
        manual_dark_scale = self._manual_dark_scale(dark_lift_coeff)
        if dark_map is not None:
            dark_map_m = self._match_dark_map_shape(dark_map, rgb.shape[0], rgb.shape[1])
            rgb = np.clip(rgb - (dark_map_m * manual_dark_scale), 0.0, None)
        if dark_frame_offset > 0:
            rgb = np.clip(rgb - (dark_frame_offset * 65535.0 * manual_dark_scale), 0.0, None)
        return rgb, dark_used

    def _compute_master_dark_map(self, cap_on_paths, demosaic):
        if not cap_on_paths:
            return None

        acc = None
        count = 0
        for path in cap_on_paths:
            try:
                rgb = self._read_linear_rgb_65535(path, demosaic)
                if acc is None:
                    acc = rgb
                    count = 1
                elif acc.shape == rgb.shape:
                    acc += rgb
                    count += 1
                else:
                    print(f"Warning: skipping cap-on frame with mismatched shape: {os.path.basename(path)}")
            except Exception as e:
                print(f"Warning: unable to decode cap-on dark frame {path}: {e}")

        if acc is None or count == 0:
            return None
        return acc / float(count)

    def _compute_master_dark_bayer_map(self, cap_on_paths):
        if not cap_on_paths:
            return None

        acc = None
        count = 0
        for path in cap_on_paths:
            try:
                y = self._read_bayer2x2_luma_65535(path)
                if acc is None:
                    acc = y
                    count = 1
                elif acc.shape == y.shape:
                    acc += y
                    count += 1
                else:
                    print(f"Warning: skipping cap-on frame with mismatched Bayer 2x2 shape: {os.path.basename(path)}")
            except Exception as e:
                print(f"Warning: unable to decode cap-on dark frame {path}: {e}")

        if acc is None or count == 0:
            return None
        return acc / float(count)

    def _compute_ambient_dark_offset(self, cap_off_paths, demosaic, master_dark_map):
        if not cap_off_paths:
            return 0.0

        per_frame_means = []
        for path in cap_off_paths:
            try:
                rgb = self._read_linear_rgb_65535(path, demosaic)
                if master_dark_map is not None:
                    dark_map_m = self._match_dark_map_shape(master_dark_map, rgb.shape[0], rgb.shape[1])
                    rgb = np.clip(rgb - dark_map_m, 0.0, None)
                per_frame_means.append(float(np.mean(rgb) / 65535.0))
            except Exception as e:
                print(f"Warning: unable to decode cap-off dark frame {path}: {e}")

        if not per_frame_means:
            return 0.0
        return float(np.mean(per_frame_means))

    def _compute_ambient_dark_offset_bayer(self, cap_off_paths, master_dark_bayer_map):
        if not cap_off_paths:
            return 0.0

        per_frame_means = []
        for path in cap_off_paths:
            try:
                y = self._read_bayer2x2_luma_65535(path)
                if master_dark_bayer_map is not None:
                    dark_m = self._match_dark_map_shape(master_dark_bayer_map, y.shape[0], y.shape[1])
                    y = np.clip(y - dark_m, 0.0, None)
                per_frame_means.append(float(np.mean(y) / 65535.0))
            except Exception as e:
                print(f"Warning: unable to decode cap-off dark frame {path}: {e}")

        if not per_frame_means:
            return 0.0
        return float(np.mean(per_frame_means))

    def extract_sensor_bayer2x2_map(self, filepath, sharp_radius, sharp_amount, downsample_factor=1,
                                    apply_dark_level=True, dark_frame_offset=0.0, dark_bayer_map=None,
                                    dark_lift_coeff=0.0):
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        ds = max(1, int(downsample_factor))
        dark_used = None

        if ext in {'tif', 'tiff'}:
            img_linear = self.load_and_linearize(filepath, apply_dark_level=False)
            y = self._compute_luminance_from_rgb(img_linear, use_weighted=False)
            y = self.apply_unsharp_mask(y, sharp_radius, sharp_amount)
            if ds > 1:
                y = y[::ds, ::ds]
            return y, dark_used

        with rawpy.imread(filepath) as raw:
            raw_vis = raw.raw_image_visible.astype(np.float32)

            if apply_dark_level:
                dark_levels = [float(v) for v in list(raw.black_level_per_channel)]
                colors = raw.raw_colors_visible
                for idx, level in enumerate(dark_levels):
                    raw_vis[colors == idx] -= self._effective_dark_level(level, dark_lift_coeff)
                positive_levels = [v for v in dark_levels if v > 0]
                if positive_levels:
                    dark_used = self._effective_dark_level(float(sum(positive_levels) / len(positive_levels)), dark_lift_coeff)

            if dark_bayer_map is not None:
                dark_bayer_m = self._match_dark_map_shape(dark_bayer_map, raw_vis.shape[0] // 2, raw_vis.shape[1] // 2)
            else:
                dark_bayer_m = None

            y_65535 = self._bayer2x2_mean_map(np.clip(raw_vis, 0.0, None))

            if dark_bayer_m is not None:
                y_65535 = np.clip(y_65535 - (dark_bayer_m * self._manual_dark_scale(dark_lift_coeff)), 0.0, None)

            if dark_frame_offset > 0:
                y_65535 = np.clip(y_65535 - (dark_frame_offset * 65535.0 * self._manual_dark_scale(dark_lift_coeff)), 0.0, None)

        y = np.clip(y_65535 / 65535.0, 0.0, None)
        y = self.apply_unsharp_mask(y, sharp_radius, sharp_amount)
        if ds > 1:
            y = y[::ds, ::ds]
        return y, dark_used

    def extract_sensor_weighted_rgb_map_sharp(self, filepath, demosaic, apply_dark_level=True,
                                              use_weighted=False, weight_r=1.0, weight_g=1.0, weight_b=1.0,
                                              dark_frame_offset=0.0, dark_map=None, dark_lift_coeff=0.0):
        with rawpy.imread(filepath) as raw:
            rgb, dark_used = self._postprocess_raw_linear_rgb(
                raw=raw,
                demosaic=demosaic,
                half_size=False,
                apply_dark_level=apply_dark_level,
                dark_frame_offset=dark_frame_offset,
                dark_map=dark_map,
                dark_lift_coeff=dark_lift_coeff
            )
        y = self._compute_luminance_from_rgb(rgb, use_weighted, weight_r, weight_g, weight_b)
        return y, dark_used

    def apply_unsharp_mask(self, image, radius, amount):
        if amount <= 0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        sharpened = image + amount * (image - blurred)
        return np.clip(sharpened, 0.0, 1.0)

    def extract_sensor_luminance_map(self, filepath, demosaic, sharp_radius, sharp_amount, downsample_factor=1,
                                     apply_dark_level=True, use_weighted=False,
                                     weight_r=1.0, weight_g=1.0, weight_b=1.0, dark_frame_offset=0.0,
                                     dark_map=None, dark_lift_coeff=0.0):
        ext = filepath.lower().split('.')[-1]
        ds = max(1, int(downsample_factor))
        dark_used = None

        if ext in ['tif', 'tiff']:
            img_linear = self.load_and_linearize(
                filepath,
                demosaic=demosaic,
                apply_dark_level=apply_dark_level,
                dark_frame_offset=dark_frame_offset,
                dark_map=dark_map,
                dark_lift_coeff=dark_lift_coeff
            )
            y = self._compute_luminance_from_rgb(img_linear, use_weighted, weight_r, weight_g, weight_b)
            y = self.apply_unsharp_mask(y, sharp_radius, sharp_amount)
            if ds > 1:
                y = y[::ds, ::ds]
            return y, dark_used

        half_size = ds > 1 and ds % 2 == 0
        post_ds = max(1, ds // 2) if half_size else ds
        with rawpy.imread(filepath) as raw:
            rgb, dark_used = self._postprocess_raw_linear_rgb(
                raw=raw,
                demosaic=demosaic,
                half_size=half_size,
                apply_dark_level=apply_dark_level,
                dark_frame_offset=dark_frame_offset,
                dark_map=dark_map,
                dark_lift_coeff=dark_lift_coeff
            )

        y = self._compute_luminance_from_rgb(rgb, use_weighted, weight_r, weight_g, weight_b)

        y = self.apply_unsharp_mask(y, sharp_radius, sharp_amount)
        if post_ds > 1:
            y = y[::post_ds, ::post_ds]
        return y, dark_used

    def compute_peak_from_map(self, image_map, undersample_n, threshold_pct, peak_percentile=99.8, return_burned_info=False):
        n = max(1, undersample_n)
        t = float(threshold_pct) / 100.0
        p = float(np.clip(peak_percentile, 50.0, 100.0))

        sub = image_map[::n, ::n]
        burned = sub > t
        black = sub <= 1e-8
        valid = sub[~burned]
        if valid.size > 0:
            file_max = float(np.percentile(valid, p))
        else:
            file_max = float(sub.max())
        if return_burned_info:
            return file_max, int(np.count_nonzero(burned)), int(sub.size), int(np.count_nonzero(black))
        return file_max

    def load_and_linearize(self, filepath, demosaic=rawpy.DemosaicAlgorithm.AAHD,
                           apply_dark_level=True, dark_frame_offset=0.0, dark_map=None,
                           dark_lift_coeff=0.0):
        ext = filepath.lower().split('.')[-1]
        if ext in ['tif', 'tiff']:
            img = validate_tiff_16bit(filepath)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.float32)
        else:
            with rawpy.imread(filepath) as raw:
                rgb, _ = self._postprocess_raw_linear_rgb(
                    raw=raw,
                    demosaic=demosaic,
                    half_size=False,
                    apply_dark_level=apply_dark_level,
                    dark_frame_offset=dark_frame_offset,
                    dark_map=dark_map,
                    dark_lift_coeff=dark_lift_coeff
                )
            return rgb.astype(np.float32)

    def _rotate_img(self, img, angle_idx):
        if angle_idx == 0:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle_idx == 1:
            return cv2.rotate(img, cv2.ROTATE_180)
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def get_rotation_label(self, angle_idx):
        return {
            0: '90° CW',
            1: '180°',
            2: '270° CW'
        }.get(angle_idx, '90° CW')

    def save_output_image(self, out_path, image, out_format):
        ext = os.path.splitext(out_path)[1].lower()
        encode_params = []

        if out_format == 'JPG':
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

        success, encoded = cv2.imencode(ext, image, encode_params)
        if not success:
            raise ValueError(f"Unable to encode output image for {out_path}")

        with open(out_path, 'wb') as output_file:
            output_file.write(encoded.tobytes())

    def _selected_colorspace_metadata_args(self, color_space):
        if color_space == 'sRGB':
            return [
                "-EXIF:ColorSpace=1",
                "-XMP-exif:ColorSpace=sRGB",
                "-XMP-photoshop:ICCProfile=sRGB IEC61966-2.1",
                "-PNG:sRGBRendering=Perceptual"
            ]
        if color_space == 'Display P3':
            return [
                "-EXIF:ColorSpace=65535",
                "-XMP-exif:ColorSpace=Uncalibrated",
                "-XMP-photoshop:ICCProfile=Display P3"
            ]
        if color_space == 'ProPhoto RGB':
            return [
                "-EXIF:ColorSpace=65535",
                "-XMP-exif:ColorSpace=Uncalibrated",
                "-XMP-photoshop:ICCProfile=ProPhoto RGB"
            ]
        return [
            "-EXIF:ColorSpace=65535",
            "-XMP-exif:ColorSpace=Uncalibrated",
            "-XMP-photoshop:ICCProfile=Linear"
        ]

    def copy_metadata_preserving_output_colorspace(self, exiftool_exe, src_path, out_path, color_space, normalize_orientation=False):
        tmp_args = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8',
                    suffix='.args', delete=False) as tmp:
                tmp.write("-m\n-TagsFromFile\n")
                tmp.write(src_path + "\n")
                # Copy metadata from source but keep output color-space tags under our control.
                tmp.write("-all:all\n")
                tmp.write("--ICC_Profile\n")
                tmp.write("--EXIF:ColorSpace\n")
                tmp.write("--XMP-exif:ColorSpace\n")
                tmp.write("--XMP-photoshop:ICCProfile\n")
                tmp.write("--PNG:sRGBRendering\n")
                tmp.write("--PNG:Gamma\n")
                tmp.write("--PNG:Chromaticities\n")
                # Output pixels already contain the final orientation.
                # Never carry over source orientation tags, otherwise Windows Photos may rotate again.
                tmp.write("--Orientation\n")
                tmp.write("--IFD0:Orientation\n")
                tmp.write("--EXIF:Orientation\n")
                tmp.write("--XMP-tiff:Orientation\n")

                # Clear common orientation tags on the exported file.
                tmp.write("-Orientation=\n")
                tmp.write("-IFD0:Orientation=\n")
                tmp.write("-EXIF:Orientation=\n")
                tmp.write("-XMP-tiff:Orientation=\n")

                # Write color-space metadata matching the app setting.
                for arg in self._selected_colorspace_metadata_args(color_space):
                    tmp.write(arg + "\n")

                tmp.write("-unsafe\n-overwrite_original\n")
                tmp.write(out_path + "\n")
                tmp_args = tmp.name

            cmd = [exiftool_exe, "-charset", "filename=UTF8", "-@", tmp_args]
            creationflags = get_subprocess_flags()
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=creationflags
            )
            if res.returncode != 0:
                print(f"  [ExifTool Error]: {res.stderr.strip()}")
        finally:
            if tmp_args and os.path.exists(tmp_args):
                os.unlink(tmp_args)

    def apply_color_space(self, Y_norm, color_space):
        if color_space == 'Linear':
            return Y_norm
        elif color_space in ['sRGB', 'Display P3']:
            return np.where(Y_norm <= 0.0031308, 12.92 * Y_norm, 1.055 * np.power(Y_norm, 1/2.4) - 0.055)
        elif color_space == 'ProPhoto RGB':
            return np.power(Y_norm, 1/1.8)
        return Y_norm

    def run(self):
        out_folder = ""
        self.status.emit("Initializing processing...")
        self.progress.emit(0)
        if not self.files:
            self.finished.emit(False, "No files selected.", out_folder)
            return

        phase1_only = self.options.get('phase1_only', False)

        out_format = self.options['format']
        overwrite = self.options['overwrite']
        out_paths = []
        skipped_paths = []
        self.excluded_burned_files = []
        self._excluded_burned_file_set = set()
        has_exiftool = False
        exiftool_exe = ""

        if not phase1_only:
            self.status.emit("Preparing output folder and file list...")
            self.progress.emit(2)
            out_folder = self.options.get('output_folder', '').strip()
            if not out_folder:
                out_folder = os.path.join(os.path.dirname(self.files[0]), f"Luminance_Export_{out_format}")
            os.makedirs(out_folder, exist_ok=True)

            for f in self.files:
                base = os.path.splitext(os.path.basename(f))[0]
                ext = out_format.lower()
                out_path = os.path.join(out_folder, f"{base}_lum.{ext}")

                if not overwrite and os.path.exists(out_path):
                    skipped_paths.append(out_path)
                    print(f"Skipping existing output: {os.path.basename(out_path)}")
                    continue

                out_paths.append((f, out_path))

            if skipped_paths:
                print(f"Skipped {len(skipped_paths)} existing file(s) because overwrite is disabled.")

            if not out_paths:
                self.finished.emit(True, "No new files exported: all output files already exist.", out_folder)
                return

            exiftool_exe = get_exiftool_path()
            has_exiftool = os.path.exists(exiftool_exe)
            if not has_exiftool:
                print(f"WARNING: exiftool.exe not found in {os.path.dirname(exiftool_exe)}")
                print("Ensure exiftool.exe and the exiftool_files folder are in the directory.")
            else:
                print(f"ExifTool found in: {exiftool_exe}")

        try:
            undersample_n = self.options['undersample_n']
            percentile_threshold = self.options['burnt_percentile']
            peak_percentile = float(np.clip(self.options.get('peak_percentile', 99.8), 50.0, 100.0))
            demosaic = self.options['demosaic']
            sharp_amount = float(self.options.get('sharp_amount', 0.0))
            sharp_radius = float(self.options.get('sharp_radius', 1.0))
            apply_dark_level = bool(self.options.get('apply_dark_level', True))
            dark_frame_paths_cap_on = list(self.options.get('dark_frame_paths_cap_on', []))
            dark_frame_paths_cap_off = list(self.options.get('dark_frame_paths_cap_off', []))
            luminance_mode = self.options.get('luminance_mode', 'raw_bayer_2x2')
            use_weighted_luminance = bool(self.options.get('use_weighted_luminance', False))
            weight_r = float(self.options.get('weight_r', 1.0))
            weight_g = float(self.options.get('weight_g', 2.0))
            weight_b = float(self.options.get('weight_b', 1.0))
            dark_lift_coeff = float(np.clip(self.options.get('dark_lift_coeff', 0.0), -0.99, 0.99))
            output_downsample_factor = max(1, int(self.options.get('output_downsample_factor', 1)))
            process_at_output_scale = bool(self.options.get('process_at_output_scale', False))
            light_comp_enabled = bool(self.options.get('light_compensation_enabled', False))
            light_gain_map = self._build_light_gain_lookup() if light_comp_enabled else {}
            flatfield_enabled = bool(self.options.get('flatfield_enabled', False))
            flatfield_lookup = self._build_flatfield_lookup() if flatfield_enabled else {}
            dark_frame_offset = 0.0
            dark_map = None
            dark_bayer_map = None

            def _prepare_phase1_sensor_norm(filepath):
                if luminance_mode == 'raw_bayer_2x2':
                    sensor_norm, dark_used = self.extract_sensor_bayer2x2_map(
                        filepath=filepath,
                        sharp_radius=sharp_radius,
                        sharp_amount=sharp_amount,
                        downsample_factor=undersample_n,
                        apply_dark_level=apply_dark_level,
                        dark_frame_offset=dark_frame_offset,
                        dark_bayer_map=dark_bayer_map,
                        dark_lift_coeff=dark_lift_coeff
                    )
                else:
                    sensor_norm, dark_used = self.extract_sensor_luminance_map(
                        filepath=filepath,
                        demosaic=demosaic,
                        sharp_radius=sharp_radius,
                        sharp_amount=sharp_amount,
                        downsample_factor=undersample_n,
                        apply_dark_level=apply_dark_level,
                        use_weighted=use_weighted_luminance,
                        weight_r=weight_r,
                        weight_g=weight_g,
                        weight_b=weight_b,
                        dark_frame_offset=dark_frame_offset,
                        dark_map=dark_map,
                        dark_lift_coeff=dark_lift_coeff
                    )
                if dark_used is not None:
                    print(f"  RAW dark level used: {dark_used:.2f}")
                sensor_norm = self._apply_linearity_lut(sensor_norm)
                gain = self._gain_for_file(filepath, light_gain_map)
                if gain != 1.0:
                    sensor_norm = np.clip(sensor_norm * gain, 0.0, None)
                if flatfield_enabled:
                    sensor_norm = self._apply_flatfield_to_luma_map(
                        input_path=filepath,
                        y_norm=sensor_norm,
                        flat_lookup=flatfield_lookup,
                        demosaic=demosaic,
                        apply_dark_level=apply_dark_level,
                        dark_frame_offset=dark_frame_offset,
                        dark_map=dark_map,
                        dark_bayer_map=dark_bayer_map,
                        dark_lift_coeff=dark_lift_coeff,
                    )
                return sensor_norm

            self.status.emit("Preparing calibration settings...")
            self.progress.emit(5)

            if light_comp_enabled:
                if light_gain_map:
                    print(f"Light-intensity variance calibration: enabled ({len(light_gain_map)} gain entries loaded)")
                else:
                    print("Light-intensity variance calibration: enabled (no gain entries loaded, fallback gain=1)")
            else:
                print("Light-intensity variance calibration: disabled")

            if self._linearity_lut_values is not None:
                print("Sensor linearity calibration (monotonic LUT): enabled")
            elif bool(self.options.get('linearity_calibration_enabled', False)):
                print("Sensor linearity calibration (monotonic LUT): enabled but invalid/empty LUT, fallback disabled")
            else:
                print("Sensor linearity calibration (monotonic LUT): disabled")

            if flatfield_enabled:
                if flatfield_lookup:
                    print(f"Flatfielding: enabled ({len(flatfield_lookup)} input-grey pairs)")
                else:
                    print("Flatfielding: enabled (no valid pairs found, fallback disabled per file)")
            else:
                print("Flatfielding: disabled")

            if luminance_mode == 'raw_bayer_2x2':
                use_weighted_luminance = False
                print("Luminance mode: RAW Bayer 2x2 mean (recommended)")
            elif luminance_mode == 'demosaic_weighted':
                use_weighted_luminance = True
                print(f"Luminance mode: demosaic weighted RGB (R={weight_r:.3f}, G={weight_g:.3f}, B={weight_b:.3f})")
            else:
                use_weighted_luminance = False
                print("Luminance mode: demosaic arithmetic RGB mean")

            if dark_frame_paths_cap_on or dark_frame_paths_cap_off:
                apply_dark_level = False
                print("Dark calibration mode: manual dark-frame workflow")
                if luminance_mode == 'raw_bayer_2x2':
                    if dark_frame_paths_cap_on:
                        print(f"Cap-on frames loaded: {len(dark_frame_paths_cap_on)}")
                        dark_bayer_map = self._compute_master_dark_bayer_map(dark_frame_paths_cap_on)
                        if dark_bayer_map is not None:
                            print("Cap-on per-pixel dark map (Bayer 2x2): enabled")
                        else:
                            print("Cap-on per-pixel dark map (Bayer 2x2): unavailable")
                    if dark_frame_paths_cap_off:
                        print(f"Cap-off frames loaded: {len(dark_frame_paths_cap_off)}")
                        dark_frame_offset = self._compute_ambient_dark_offset_bayer(
                            dark_frame_paths_cap_off,
                            dark_bayer_map
                        )
                        print(f"Ambient dark offset (after cap-on removal): {dark_frame_offset:.8f} (normalized linear)")
                else:
                    if dark_frame_paths_cap_on:
                        print(f"Cap-on frames loaded: {len(dark_frame_paths_cap_on)}")
                        dark_map = self._compute_master_dark_map(dark_frame_paths_cap_on, demosaic)
                        if dark_map is not None:
                            print("Cap-on per-pixel dark map: enabled")
                        else:
                            print("Cap-on per-pixel dark map: unavailable")
                    if dark_frame_paths_cap_off:
                        print(f"Cap-off frames loaded: {len(dark_frame_paths_cap_off)}")
                        dark_frame_offset = self._compute_ambient_dark_offset(
                            dark_frame_paths_cap_off,
                            demosaic,
                            dark_map
                        )
                        print(f"Ambient dark offset (after cap-on removal): {dark_frame_offset:.8f} (normalized linear)")

            if apply_dark_level:
                print("Dark-level calibration: enabled (use RAW black level when available)")
                print(f"Dark lift coefficient: {dark_lift_coeff:.4f}")
            else:
                print("Dark-level calibration: disabled")

            if output_downsample_factor > 1:
                print(f"Output downsample: enabled (factor {output_downsample_factor}x)")
            else:
                print("Output downsample: disabled (full resolution)")

            process_downsample_factor = output_downsample_factor if (process_at_output_scale and output_downsample_factor > 1) else 1
            if process_downsample_factor > 1:
                print(f"Fast export mode: processing at output scale (factor {process_downsample_factor}x)")
            else:
                print("Fast export mode: disabled (full-resolution processing)")

            if phase1_only:
                self.status.emit("Phase 1/2: Global luminance peak analysis...")
                self.progress.emit(10)
                print("Starting Phase 1: Global luminance peak calculation (Raw Array Method)...")
                global_max = 0.0
                phase1_peaks = []
                phase1_burned_ratios = []
                phase1_black_ratios = []

                files_to_process = list(self.files)

                for i, f in enumerate(files_to_process):
                    if not self._is_running:
                        self.finished.emit(False, "Process interrupted by user (Phase 1).", out_folder)
                        return

                    print(f"Fast peak analysis for: {os.path.basename(f)}")

                    sensor_norm = _prepare_phase1_sensor_norm(f)
                    img_max, burned_count, burned_total, black_count = self.compute_peak_from_map(
                        sensor_norm,
                        1,
                        percentile_threshold,
                        peak_percentile,
                        return_burned_info=True
                    )

                    burned_ratio_pct = (float(burned_count) / float(max(1, burned_total)) * 100.0)
                    black_ratio_pct = (float(black_count) / float(max(1, burned_total)) * 100.0)
                    phase1_burned_ratios.append(burned_ratio_pct)
                    phase1_black_ratios.append(black_ratio_pct)

                    global_max = max(global_max, img_max)
                    phase1_peaks.append(float(img_max))
                    phase1_pct = 10 + int((i + 1) / max(1, len(files_to_process)) * 45)
                    self.progress.emit(phase1_pct)

                if global_max <= 0:
                    global_max = 1.0

                print(f"Calculated global peak (sensor normalized): {global_max:.6f}")
                print(f"Stretch factor applied to full images: 1 / {global_max:.6f}")
                self.analysis_threshold.emit(float(np.clip(global_max * 100.0, 0.0, 100.0)))

                median_peak = float(np.median(np.array(phase1_peaks, dtype=np.float64))) if phase1_peaks else 0.0
                p99_burned_high = float(np.percentile(np.array(phase1_burned_ratios, dtype=np.float64), 99.0)) if phase1_burned_ratios else 0.0
                p99_burned_black = float(np.percentile(np.array(phase1_black_ratios, dtype=np.float64), 99.0)) if phase1_black_ratios else 0.0
                frames_excluded_ratio = (
                    float(len(self.excluded_burned_files)) / float(max(1, len(files_to_process))) * 100.0
                )
                print(
                    "Phase-1 QA summary: "
                    f"median_peak={median_peak:.6f}, "
                    f"p99_burned_high={p99_burned_high:.6f}%, "
                    f"p99_burned_black={p99_burned_black:.6f}%, "
                    f"excluded_ratio={frames_excluded_ratio:.3f}%"
                )

                self.status.emit("Phase 1 completed")
                self.progress.emit(100)
                self.finished.emit(True, f"Stretch calculation completed. Global peak: {global_max:.6f}", out_folder)
                return

            self.status.emit("Phase 2/2: Development and export...")
            self.progress.emit(55)
            print(f"Starting Develop Luminance using Burned threshold slider: {percentile_threshold:.2f}%")
            global_max = float(np.clip(percentile_threshold, 0.1, 100.0)) / 100.0
            stretch_denominator = max(global_max, 1e-8)
            print(f"Develop stretch factor applied to full images: 1 / {global_max:.6f}")

            for i, (f, out_path) in enumerate(out_paths):
                if not self._is_running:
                    self.finished.emit(False, "Process interrupted by user (Phase 2).", out_folder)
                    return

                self.status.emit(f"Phase 2/2: Developing {i + 1}/{len(out_paths)} - {os.path.basename(f)}")
                print(f"Developing image: {os.path.basename(f)}")
                ext = f.lower().split('.')[-1]

                if luminance_mode == 'raw_bayer_2x2':
                    Y_norm, dark_used = self.extract_sensor_bayer2x2_map(
                        filepath=f,
                        sharp_radius=sharp_radius,
                        sharp_amount=0.0,
                        downsample_factor=process_downsample_factor,
                        apply_dark_level=apply_dark_level,
                        dark_frame_offset=dark_frame_offset,
                        dark_bayer_map=dark_bayer_map,
                        dark_lift_coeff=dark_lift_coeff
                    )
                    if dark_used is not None:
                        print(f"  RAW dark level used: {dark_used:.2f}")
                else:
                    if ext in ['tif', 'tiff']:
                        img_linear = self.load_and_linearize(
                            f,
                            demosaic=demosaic,
                            apply_dark_level=apply_dark_level,
                            dark_frame_offset=dark_frame_offset,
                            dark_map=dark_map,
                            dark_lift_coeff=dark_lift_coeff,
                        )
                        Y_norm = self._compute_luminance_from_rgb(img_linear, use_weighted_luminance, weight_r, weight_g, weight_b)
                        if process_downsample_factor > 1:
                            Y_norm = Y_norm[::process_downsample_factor, ::process_downsample_factor]
                    else:
                        Y_norm, dark_used = self.extract_sensor_luminance_map(
                            filepath=f,
                            demosaic=demosaic,
                            sharp_radius=sharp_radius,
                            sharp_amount=0.0,
                            downsample_factor=process_downsample_factor,
                            apply_dark_level=apply_dark_level,
                            use_weighted=use_weighted_luminance,
                            weight_r=weight_r,
                            weight_g=weight_g,
                            weight_b=weight_b,
                            dark_frame_offset=dark_frame_offset,
                            dark_map=dark_map,
                            dark_lift_coeff=dark_lift_coeff
                        )
                        if dark_used is not None:
                            print(f"  RAW dark level used: {dark_used:.2f}")

                Y_norm = self._apply_linearity_lut(Y_norm)

                gain = self._gain_for_file(f, light_gain_map)
                if gain != 1.0:
                    Y_norm = np.clip(Y_norm * gain, 0.0, None)
                if flatfield_enabled:
                    Y_norm = self._apply_flatfield_to_luma_map(
                        input_path=f,
                        y_norm=Y_norm,
                        flat_lookup=flatfield_lookup,
                        demosaic=demosaic,
                        apply_dark_level=apply_dark_level,
                        dark_frame_offset=dark_frame_offset,
                        dark_map=dark_map,
                        dark_bayer_map=dark_bayer_map,
                        dark_lift_coeff=dark_lift_coeff,
                    )

                Y_norm = self.apply_unsharp_mask(Y_norm, sharp_radius, sharp_amount)

                Y_stretched = np.clip(Y_norm / stretch_denominator, 0, 1)

                Y_out = self.apply_color_space(Y_stretched, self.options['color_space'])
                Y_out_3c = np.stack((Y_out, Y_out, Y_out), axis=-1)

                rot_enabled = self.options.get('rotation_enabled', False)
                rot_angle = self.options.get('rotation_angle', 0)

                if rot_enabled:
                    print(f"  Applying output rotation: {self.get_rotation_label(rot_angle)}")

                out_bit_depth = self.options.get('bit_depth', 8)
                if out_format in ['TIFF', 'PNG'] and out_bit_depth == 16:
                    out_img = (Y_out_3c * 65535).astype(np.uint16)
                    if rot_enabled:
                        out_img = self._rotate_img(out_img, rot_angle)
                    if output_downsample_factor > 1 and process_downsample_factor == 1:
                        h, w = out_img.shape[:2]
                        out_w = max(1, w // output_downsample_factor)
                        out_h = max(1, h // output_downsample_factor)
                        out_img = cv2.resize(out_img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    self.save_output_image(out_path, out_img, out_format)
                else:
                    out_img = (Y_out_3c * 255).astype(np.uint8)
                    if rot_enabled:
                        out_img = self._rotate_img(out_img, rot_angle)
                    if output_downsample_factor > 1 and process_downsample_factor == 1:
                        h, w = out_img.shape[:2]
                        out_w = max(1, w // output_downsample_factor)
                        out_h = max(1, h // output_downsample_factor)
                        out_img = cv2.resize(out_img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    self.save_output_image(out_path, out_img, out_format)

                if has_exiftool:
                    self.status.emit(f"Phase 2/2: Writing metadata {i + 1}/{len(out_paths)}")
                    print(f"  Copying metadata to {os.path.basename(out_path)}...")
                    self.copy_metadata_preserving_output_colorspace(
                        exiftool_exe,
                        f,
                        out_path,
                        self.options['color_space'],
                        normalize_orientation=rot_enabled
                    )

                phase2_pct = 55 + int((i + 1) / max(1, len(out_paths)) * 45)
                self.progress.emit(phase2_pct)

            message = "Development completed successfully!"
            if skipped_paths:
                message += f" Skipped {len(skipped_paths)} existing file(s)."
            self.status.emit("Finalizing...")
            self.finished.emit(True, message, out_folder)

        except Exception as e:
            print(f"CRITICAL ERROR: {str(e)}")
            self.finished.emit(False, f"Processing error: {str(e)}", out_folder)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_WINDOW_TITLE)
        self.resize(1360, 900)
        self.setMinimumSize(1280, 520)
        
        self.files = []
        self.dark_frame_files_cap_off = []
        self.dark_frame_files_cap_on = []
        self.grey_card_files = []
        self.grey_to_input_pairs = []
        self.dome_file_path = ""
        self.light_comp_file_path = ""
        self.light_comp_gain_map = {}
        self.light_comp_gain_values = []
        self.light_comp_metadata = {}
        # Dark frame caching: stores frame means (key = tuple(paths))
        self.dark_frame_mean_cache_cap_on = {}
        self.dark_frame_mean_cache_cap_off = {}
        # Dark map caching: stores full dark maps (key = tuple(paths), luminance_mode, demosaic)
        self.dark_map_cache_cap_on = {}
        self.dark_bayer_map_cache_cap_on = {}
        self.dark_map_cache_cap_off = {}
        # Legacy single-value cache (kept for backward compatibility with grey calibration)
        self.dark_signal_cache_key = None
        self.dark_signal_cache_value = 0.0
        self.linearity_calibration_files = []
        self._linearity_lut_cache_text = ""
        self._linearity_lut_cache_points = []
        self._linearity_cached_rows = []
        self._linearity_cached_frame_report = []
        self._linearity_cached_consistency_warnings = []
        self.preview_index = 0
        self._stack_cache = None       # cached raw luminance stack array
        self._stack_cache_key_val = None  # key tuple that matches cached stack
        self.last_output_folder = ""
        self.last_auto_output_path = ""
        self.edit_output_folder = None
        self.thread = None
        self.dark_analysis_thread = None
        self.current_phase1_only = False
        self.initUI()
        
        sys.stdout = EmittingStream()
        sys.stdout.textWritten.connect(self.append_to_console)
        sys.stderr = EmittingStream()
        sys.stderr.textWritten.connect(self.append_to_console)
        print("Application started successfully.")

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and isinstance(obj, (QAbstractSpinBox, QComboBox, QSlider)):
            self._scroll_main_area_by_wheel(event)
            return True
        return super().eventFilter(obj, event)

    def _scroll_main_area_by_wheel(self, event):
        if not hasattr(self, 'main_scroll_area') or self.main_scroll_area is None:
            return
        scrollbar = self.main_scroll_area.verticalScrollBar()
        if scrollbar is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = scrollbar.singleStep() if scrollbar.singleStep() > 0 else 20
        # Positive delta means wheel up; vertical scrollbar decreases when moving up.
        scrollbar.setValue(scrollbar.value() - int(np.sign(delta) * step * 3))

    def disable_wheel_on_inputs(self):
        for widget in self.findChildren((QAbstractSpinBox, QComboBox, QSlider)):
            widget.installEventFilter(self)

    def _configure_section_toggle_button(self, button):
        button.setCheckable(True)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet("""
            QPushButton {
                text-align: left;
                border: none;
                color: #7f8c8d;
                text-decoration: underline;
                padding: 5px 0px;
            }
            QPushButton:hover { color: #2c3e50; }
        """)

    def _update_section_visibility(self, button, section_widget, title):
        is_open = button.isChecked()
        section_widget.setVisible(is_open)
        button.setText(("▼ " if is_open else "▶ ") + title)

    def _create_section_toggle(self, title, section_widget, default_open=True):
        button = QPushButton()
        self._configure_section_toggle_button(button)
        button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        button.setChecked(default_open)
        button.clicked.connect(lambda _=False, b=button, w=section_widget, t=title: self._update_section_visibility(b, w, t))
        self._update_section_visibility(button, section_widget, title)
        return button

    def _pin_section_to_top(self, widget):
        if widget is not None:
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def _effective_dark_level(self, dark_level):
        coeff = self.get_dark_lift_coeff_value()
        return max(0.0, float(dark_level) * (1.0 - coeff))

    def _manual_dark_scale(self):
        return max(0.0, 1.0 - self.get_dark_lift_coeff_value())

    def _format_dark_lift_coeff(self, coeff):
        return f"{float(np.clip(coeff, -0.99, 0.99)):.6f}".replace('.', ',')

    def get_dark_lift_coeff_value(self):
        text = self.edit_dark_lift_coeff.text().strip().replace(',', '.')
        try:
            value = float(text)
        except ValueError:
            value = 0.0
        return float(np.clip(value, -0.99, 0.99))

    def set_dark_lift_coeff_value(self, coeff):
        self.edit_dark_lift_coeff.setText(self._format_dark_lift_coeff(coeff))

    def on_dark_lift_coeff_edited(self):
        self.set_dark_lift_coeff_value(self.get_dark_lift_coeff_value())
        self.update_clip_preview()

    def initUI(self):
        root_widget = QWidget()
        root_layout = QVBoxLayout(root_widget)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)
        self.setCentralWidget(root_widget)

        self.main_scroll_area = QScrollArea()
        self.main_scroll_area.setWidgetResizable(True)
        self.main_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.main_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root_layout.addWidget(self.main_scroll_area, 1)

        central_widget = QWidget()
        self.main_scroll_area.setWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignTop)

        files_box = QGroupBox("")
        self._pin_section_to_top(files_box)
        files_layout = QVBoxLayout(files_box)
        files_layout.setAlignment(Qt.AlignTop)
        self.list_widget = QListWidget()
        files_layout.addWidget(QLabel("RAW/TIFF Files (TIFF must be 16-bit linear):"))
        files_layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add Files")
        btn_add.clicked.connect(self.add_files)
        btn_clear = QPushButton("Clear List")
        btn_clear.clicked.connect(self.clear_files)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_clear)
        files_layout.addLayout(btn_layout)

        files_toggle = self._create_section_toggle("1) Input Files", files_box, default_open=True)
        layout.addWidget(files_toggle)
        layout.addWidget(files_box)

        output_box = QGroupBox("")
        self._pin_section_to_top(output_box)
        output_layout = QVBoxLayout(output_box)
        output_layout.setAlignment(Qt.AlignTop)

        output_profile_box = QGroupBox("Output Profile")
        output_profile_layout = QVBoxLayout(output_profile_box)
        output_profile_layout.setAlignment(Qt.AlignTop)

        profile_row = QHBoxLayout()
        self.combo_format = QComboBox()
        self.combo_format.addItems(["TIFF", "PNG", "JPG"])
        self.combo_format.setCurrentText("TIFF")
        profile_row.addWidget(QLabel("Output Format:"))
        profile_row.addWidget(self.combo_format)
        profile_row.addSpacing(18)

        self.combo_bit_depth = QComboBox()
        self.combo_bit_depth.addItems(["8-bit", "16-bit"])
        self.combo_bit_depth.setCurrentText("16-bit")
        profile_row.addWidget(QLabel("Bit Depth:"))
        profile_row.addWidget(self.combo_bit_depth)
        profile_row.addSpacing(18)

        self.combo_color = QComboBox()
        self.combo_color.addItems(["Linear", "sRGB", "Display P3", "ProPhoto RGB"])
        self.combo_color.setCurrentText("Linear")
        profile_row.addWidget(QLabel("Colour Space:"))
        profile_row.addWidget(self.combo_color)
        profile_row.addStretch()
        output_profile_layout.addLayout(profile_row)
        self.combo_color.currentTextChanged.connect(self.update_clip_preview)

        overwrite_row = QHBoxLayout()
        self.check_overwrite = QCheckBox("Overwrite existing files")
        overwrite_row.addWidget(self.check_overwrite)
        overwrite_row.addStretch()
        output_profile_layout.addLayout(overwrite_row)

        self.combo_format.currentTextChanged.connect(self.update_bit_depth_controls)
        self.update_bit_depth_controls(self.combo_format.currentText())
        output_layout.addWidget(output_profile_box)

        self.luminance_box = QGroupBox("Luminance Source")
        luminance_layout = QVBoxLayout(self.luminance_box)
        luminance_layout.setAlignment(Qt.AlignTop)

        lum_top_row = QHBoxLayout()
        self.combo_luminance_source = QComboBox()
        self.combo_luminance_source.addItems([
            "RAW Bayer 2x2 (recommended)",
            "Demosaic RGB mean",
            "Demosaic RGB weighted"
        ])
        self.combo_luminance_source.setCurrentText("Demosaic RGB mean")
        lum_top_row.addWidget(QLabel("Luminance Source:"))
        lum_top_row.addWidget(self.combo_luminance_source)
        lum_top_row.addSpacing(18)
        self.lbl_demosaic_label = QLabel("Demosaic Algorithm:")
        self.combo_demosaic = QComboBox()
        self.demosaic_map = {
            "AAHD": rawpy.DemosaicAlgorithm.AAHD,
            "AHD": rawpy.DemosaicAlgorithm.AHD,
            "VNG": rawpy.DemosaicAlgorithm.VNG,
            "LINEAR": rawpy.DemosaicAlgorithm.LINEAR
        }
        self.combo_demosaic.addItems(self.demosaic_map.keys())
        self.combo_demosaic.setCurrentText("AAHD")
        lum_top_row.addWidget(self.lbl_demosaic_label)
        lum_top_row.addWidget(self.combo_demosaic)
        lum_top_row.addStretch()
        luminance_layout.addLayout(lum_top_row)

        self.weighted_controls_widget = QWidget()
        weighted_controls_layout = QHBoxLayout(self.weighted_controls_widget)
        weighted_controls_layout.setContentsMargins(0, 0, 0, 0)
        weighted_controls_layout.addWidget(QLabel("Weights -> R:"))
        self.spin_weight_r = QDoubleSpinBox()
        self.spin_weight_r.setRange(0.0, 10.0)
        self.spin_weight_r.setSingleStep(0.1)
        self.spin_weight_r.setValue(1.0)
        weighted_controls_layout.addWidget(self.spin_weight_r)
        weighted_controls_layout.addWidget(QLabel("G:"))
        self.spin_weight_g = QDoubleSpinBox()
        self.spin_weight_g.setRange(0.0, 10.0)
        self.spin_weight_g.setSingleStep(0.1)
        self.spin_weight_g.setValue(2.0)
        weighted_controls_layout.addWidget(self.spin_weight_g)
        weighted_controls_layout.addWidget(QLabel("B:"))
        self.spin_weight_b = QDoubleSpinBox()
        self.spin_weight_b.setRange(0.0, 10.0)
        self.spin_weight_b.setSingleStep(0.1)
        self.spin_weight_b.setValue(1.0)
        weighted_controls_layout.addWidget(self.spin_weight_b)
        weighted_controls_layout.addStretch()
        luminance_layout.addWidget(self.weighted_controls_widget)

        output_layout.addWidget(self.luminance_box)

        self.sharp_box = QGroupBox("Sharpness")
        sharp_layout = QHBoxLayout(self.sharp_box)
        sharp_layout.setContentsMargins(8, 4, 8, 8)

        self.check_use_sharpness = QCheckBox("Use sharpness")
        self.check_use_sharpness.setChecked(False)
        sharp_layout.addWidget(self.check_use_sharpness)

        self.lbl_sharp_amount = QLabel("Sharpness Amount:")
        sharp_layout.addWidget(self.lbl_sharp_amount)
        self.spin_sharp_amount = QDoubleSpinBox()
        self.spin_sharp_amount.setRange(0.0, 5.0)
        self.spin_sharp_amount.setSingleStep(0.1)
        self.spin_sharp_amount.setValue(0.6)
        sharp_layout.addWidget(self.spin_sharp_amount)

        self.lbl_sharp_radius = QLabel("Radius:")
        sharp_layout.addWidget(self.lbl_sharp_radius)
        self.spin_sharp_radius = QDoubleSpinBox()
        self.spin_sharp_radius.setRange(0.1, 10.0)
        self.spin_sharp_radius.setSingleStep(0.5)
        self.spin_sharp_radius.setValue(1.0)
        sharp_layout.addWidget(self.spin_sharp_radius)
        sharp_layout.addStretch()
        output_layout.addWidget(self.sharp_box)

        downsample_row = QHBoxLayout()
        downsample_row.addWidget(QLabel("Export downsample factor:"))
        self.spin_output_downsample = QSpinBox()
        self.spin_output_downsample.setRange(1, 32)
        self.spin_output_downsample.setValue(1)
        self.spin_output_downsample.setToolTip("Save output images downsampled by this factor. 1 = full resolution.")
        downsample_row.addWidget(self.spin_output_downsample)
        self.check_process_at_output_scale = QCheckBox("Process at output scale (faster)")
        self.check_process_at_output_scale.setChecked(False)
        self.check_process_at_output_scale.setToolTip(
            "When enabled and downsample factor > 1, processing runs on reduced-size maps to speed up export."
        )
        downsample_row.addSpacing(10)
        downsample_row.addWidget(self.check_process_at_output_scale)
        downsample_row.addStretch()
        output_layout.addLayout(downsample_row)

        output_toggle = self._create_section_toggle("2) Output And Development Settings", output_box, default_open=True)
        layout.addWidget(output_toggle)
        layout.addWidget(output_box)

        dark_box = QGroupBox("")
        self._pin_section_to_top(dark_box)
        dark_layout = QVBoxLayout(dark_box)
        dark_layout.setAlignment(Qt.AlignTop)

        dark_meta_row = QHBoxLayout()
        self.check_apply_dark_level = QCheckBox("Dark level calibration from RAW metadata")
        self.check_apply_dark_level.setChecked(True)
        dark_meta_row.addWidget(self.check_apply_dark_level)
        dark_meta_row.addSpacing(10)
        dark_meta_row.addWidget(QLabel("Lift coefficient:"))
        self.edit_dark_lift_coeff = QLineEdit(self._format_dark_lift_coeff(0.0))
        self.edit_dark_lift_coeff.setFixedWidth(96)
        self.edit_dark_lift_coeff.setToolTip("Adjusts RAW metadata black subtraction coefficient (range: -0.99 to 0.99).")
        dark_meta_row.addWidget(self.edit_dark_lift_coeff)
        dark_meta_row.addStretch()
        dark_layout.addLayout(dark_meta_row)

        self.dark_frames_controls = QWidget()
        dark_controls_layout = QVBoxLayout(self.dark_frames_controls)
        dark_controls_layout.setContentsMargins(0, 0, 0, 0)

        frames_row = QHBoxLayout()

        cap_off_col = QVBoxLayout()
        cap_off_col.addWidget(QLabel("Cap off + dome off (ambient bias, TIFF must be 16-bit linear):"))
        cap_off_buttons = QHBoxLayout()
        self.btn_add_dark_frames_cap_off = QPushButton("Add")
        self.btn_add_dark_frames_cap_off.clicked.connect(self.add_dark_frames_cap_off)
        cap_off_buttons.addWidget(self.btn_add_dark_frames_cap_off)
        self.btn_clear_dark_frames_cap_off = QPushButton("Clear")
        self.btn_clear_dark_frames_cap_off.clicked.connect(self.clear_dark_frames_cap_off)
        cap_off_buttons.addWidget(self.btn_clear_dark_frames_cap_off)
        cap_off_col.addLayout(cap_off_buttons)
        self.dark_frames_list_cap_off = QListWidget()
        self.dark_frames_list_cap_off.setMaximumHeight(80)
        cap_off_col.addWidget(self.dark_frames_list_cap_off)
        frames_row.addLayout(cap_off_col)

        cap_on_col = QVBoxLayout()
        cap_on_col.addWidget(QLabel("Cap on (sensor dark per-pixel, TIFF must be 16-bit linear):"))
        cap_on_buttons = QHBoxLayout()
        self.btn_add_dark_frames_cap_on = QPushButton("Add")
        self.btn_add_dark_frames_cap_on.clicked.connect(self.add_dark_frames_cap_on)
        cap_on_buttons.addWidget(self.btn_add_dark_frames_cap_on)
        self.btn_clear_dark_frames_cap_on = QPushButton("Clear")
        self.btn_clear_dark_frames_cap_on.clicked.connect(self.clear_dark_frames_cap_on)
        cap_on_buttons.addWidget(self.btn_clear_dark_frames_cap_on)
        cap_on_col.addLayout(cap_on_buttons)
        self.dark_frames_list_cap_on = QListWidget()
        self.dark_frames_list_cap_on.setMaximumHeight(80)
        cap_on_col.addWidget(self.dark_frames_list_cap_on)
        frames_row.addLayout(cap_on_col)

        dark_controls_layout.addLayout(frames_row)

        self.lbl_dark_frames_info = QLabel("No dark frames loaded.")
        dark_controls_layout.addWidget(self.lbl_dark_frames_info)

        dark_layout.addWidget(self.dark_frames_controls)

        self.btn_analyze_dark_clipping = QPushButton("Dark level overestimation check")
        self.btn_analyze_dark_clipping.clicked.connect(self.analyze_dark_clipping)
        dark_layout.addWidget(self.btn_analyze_dark_clipping)

        dark_toggle = self._create_section_toggle("3) Dark Calibration", dark_box, default_open=False)
        layout.addWidget(dark_toggle)
        layout.addWidget(dark_box)

        linearity_box = QGroupBox("")
        self._pin_section_to_top(linearity_box)
        linearity_layout = QVBoxLayout(linearity_box)
        linearity_layout.setAlignment(Qt.AlignTop)

        linearity_top = QHBoxLayout()
        self.check_linearity_enable = QCheckBox("Use sensor linearity calibration (monotonic LUT)")
        self.check_linearity_enable.setChecked(False)
        linearity_top.addWidget(self.check_linearity_enable)
        linearity_top.addStretch()
        linearity_layout.addLayout(linearity_top)

        linearity_files_row = QHBoxLayout()
        linearity_left = QVBoxLayout()
        linearity_left.addWidget(QLabel("Calibration frames (RAW/TIFF16, varying exposure):"))
        linearity_btns = QHBoxLayout()
        self.btn_add_linearity_frames = QPushButton("Add")
        self.btn_add_linearity_frames.clicked.connect(self.add_linearity_calibration_frames)
        linearity_btns.addWidget(self.btn_add_linearity_frames)
        self.btn_clear_linearity_frames = QPushButton("Clear")
        self.btn_clear_linearity_frames.clicked.connect(self.clear_linearity_calibration_frames)
        linearity_btns.addWidget(self.btn_clear_linearity_frames)
        linearity_left.addLayout(linearity_btns)
        self.linearity_list = QListWidget()
        self.linearity_list.setMaximumHeight(84)
        linearity_left.addWidget(self.linearity_list)
        linearity_files_row.addLayout(linearity_left)
        linearity_layout.addLayout(linearity_files_row)

        self.btn_analyze_linearity = QPushButton("Analyze linearity and build inverse LUT")
        self.btn_analyze_linearity.clicked.connect(self.analyze_sensor_linearity)
        linearity_layout.addWidget(self.btn_analyze_linearity)

        linearity_threshold_row = QHBoxLayout()
        linearity_threshold_row.addWidget(QLabel("Exclude last N high-exposure frames:"))
        self.spin_linearity_exclude_last_n = QSpinBox()
        self.spin_linearity_exclude_last_n.setRange(0, 999)
        self.spin_linearity_exclude_last_n.setValue(0)
        self.spin_linearity_exclude_last_n.setToolTip("Manually excludes the highest-exposure calibration frames from fit/plot/LUT.")
        linearity_threshold_row.addWidget(self.spin_linearity_exclude_last_n)
        self.btn_apply_linearity_exclusion = QPushButton("Apply N exclusion")
        self.btn_apply_linearity_exclusion.clicked.connect(self.apply_linearity_tail_exclusion)
        linearity_threshold_row.addWidget(self.btn_apply_linearity_exclusion)
        linearity_threshold_row.addStretch()
        linearity_layout.addLayout(linearity_threshold_row)

        self.lbl_linearity_info = QLabel("No linearity calibration frames loaded. If empty, current input files will be used.")
        linearity_layout.addWidget(self.lbl_linearity_info)

        self.linearity_plot_label = QLabel("No linearity plot")
        self.linearity_plot_label.setFixedSize(620, 280)
        self.linearity_plot_label.setAlignment(Qt.AlignCenter)
        self.linearity_plot_label.setStyleSheet("border: 1px solid #bbb; background: #f5f5f5;")
        linearity_layout.addWidget(self.linearity_plot_label)

        linearity_layout.addWidget(QLabel("Editable LUT control points (input,output in [0..1]):"))
        self.edit_linearity_lut = QPlainTextEdit("")
        self.edit_linearity_lut.setMaximumHeight(130)
        linearity_layout.addWidget(self.edit_linearity_lut)

        linearity_lut_btn_row = QHBoxLayout()
        self.btn_load_linearity_lut = QPushButton("Load LUT preset")
        self.btn_load_linearity_lut.clicked.connect(self.load_linearity_lut_preset)
        linearity_lut_btn_row.addWidget(self.btn_load_linearity_lut)
        self.btn_save_linearity_lut = QPushButton("Save LUT preset")
        self.btn_save_linearity_lut.clicked.connect(self.save_linearity_lut_preset)
        linearity_lut_btn_row.addWidget(self.btn_save_linearity_lut)
        self.btn_apply_linearity_lut = QPushButton("Validate and apply LUT text")
        self.btn_apply_linearity_lut.clicked.connect(self.apply_linearity_lut_text)
        linearity_lut_btn_row.addWidget(self.btn_apply_linearity_lut)
        linearity_lut_btn_row.addStretch()
        linearity_layout.addLayout(linearity_lut_btn_row)

        linearity_toggle = self._create_section_toggle("4) Sensor Linearity Calibration", linearity_box, default_open=False)
        layout.addWidget(linearity_toggle)
        layout.addWidget(linearity_box)

        phase1_box = QGroupBox("")
        self._pin_section_to_top(phase1_box)
        phase1_layout = QVBoxLayout(phase1_box)
        phase1_layout.setAlignment(Qt.AlignTop)
        clip_widget = QWidget()
        clip_layout = QHBoxLayout(clip_widget)
        clip_layout.setContentsMargins(0, 0, 0, 0)

        clip_controls = QVBoxLayout()

        ds_row = QHBoxLayout()
        self.spin_undersample = QSpinBox()
        self.spin_undersample.setRange(1, 64)
        self.spin_undersample.setValue(32)
        ds_row.addWidget(QLabel("Downscale preview and stretch"))
        ds_row.addWidget(self.spin_undersample)
        clip_controls.addLayout(ds_row)

        burnt_row = QHBoxLayout()
        self.slider_percentile = QSlider(Qt.Horizontal)
        self.slider_percentile.setRange(1, 100000)
        self.slider_percentile.setValue(99000)
        self.slider_percentile.setTickInterval(1000)
        self.slider_percentile.setSingleStep(1)
        self.lbl_percentile_value = QLabel("99.000%")
        self.lbl_percentile_value.setMinimumWidth(70)
        self.edit_percentile = QDoubleSpinBox()
        self.edit_percentile.setRange(0.001, 100.0)
        self.edit_percentile.setDecimals(3)
        self.edit_percentile.setSingleStep(0.001)
        self.edit_percentile.setValue(99.0)
        self.edit_percentile.setSuffix(" %")
        self.edit_percentile.setFixedWidth(98)
        self.edit_percentile.setAlignment(Qt.AlignCenter)
        burnt_row.addWidget(QLabel("Burned threshold (%):"))
        burnt_row.addWidget(self.slider_percentile, 1)
        burnt_row.addWidget(self.edit_percentile)
        burnt_row.addWidget(self.lbl_percentile_value)
        clip_controls.addLayout(burnt_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Preview mode:"))
        self.combo_preview_mode = QComboBox()
        self.combo_preview_mode.addItems(["Current image", "Stack MAX", "Stack MIN"])
        self.combo_preview_mode.setToolTip(
            "Current image: stretched preview of the selected image\n"
            "Stack MAX: pixel-wise maximum across all images (shows worst high-clipping)\n"
            "Stack MIN: pixel-wise minimum across all images (shows worst black-clipping)"
        )
        mode_row.addWidget(self.combo_preview_mode, 1)
        clip_controls.addLayout(mode_row)

        nav_row = QHBoxLayout()
        self.btn_prev_preview = QPushButton("<")
        self.btn_prev_preview.setFixedWidth(34)
        self.btn_prev_preview.clicked.connect(self.prev_preview_image)
        nav_row.addWidget(self.btn_prev_preview)

        self.edit_preview_index = QLineEdit("1")
        self.edit_preview_index.setFixedWidth(64)
        self.edit_preview_index.setAlignment(Qt.AlignCenter)
        self.edit_preview_index.editingFinished.connect(self.on_preview_index_edited)
        nav_row.addWidget(self.edit_preview_index)

        self.btn_next_preview = QPushButton(">")
        self.btn_next_preview.setFixedWidth(34)
        self.btn_next_preview.clicked.connect(self.next_preview_image)
        nav_row.addWidget(self.btn_next_preview)
        self.nav_widget = QWidget()
        self.nav_widget.setLayout(nav_row)
        clip_controls.addWidget(self.nav_widget)

        self.lbl_preview_file = QLabel("Image: -/-")
        self.lbl_preview_file.setAlignment(Qt.AlignCenter)
        clip_controls.addWidget(self.lbl_preview_file)

        self.lbl_preview_stats = QLabel("Clipped pixels: -")
        clip_controls.addWidget(self.lbl_preview_stats)
        clip_controls.addStretch()
        clip_layout.addLayout(clip_controls, 1)

        self.clip_preview_label = QLabel("No preview")
        self.clip_preview_label.setFixedSize(280, 190)
        self.clip_preview_label.setAlignment(Qt.AlignCenter)
        self.clip_preview_label.setStyleSheet("border: 1px solid #bbb; background: #f5f5f5;")
        clip_layout.addWidget(self.clip_preview_label)

        phase1_layout.addWidget(clip_widget)

        self.btn_calc_stretch = QPushButton("Calculate Global Burn Threshold")
        self.btn_calc_stretch.setMinimumHeight(36)
        self.btn_calc_stretch.setStyleSheet("background-color: #4f6f4f; color: white; font-weight: bold;")
        self.btn_calc_stretch.clicked.connect(self.start_stretch_calculation)
        phase1_layout.addWidget(self.btn_calc_stretch)
        phase1_toggle = self._create_section_toggle("6) Stretch Analysis", phase1_box, default_open=True)
        layout.addWidget(phase1_toggle)
        layout.addWidget(phase1_box)

        self.spin_undersample.valueChanged.connect(self.update_clip_preview)
        self.combo_preview_mode.currentIndexChanged.connect(self.update_clip_preview)
        self.slider_percentile.valueChanged.connect(self.on_percentile_changed)
        self.edit_percentile.valueChanged.connect(self.on_percentile_spin_changed)
        self.combo_luminance_source.currentTextChanged.connect(self.on_luminance_mode_changed)
        self.combo_demosaic.currentTextChanged.connect(self.update_clip_preview)
        self.spin_sharp_amount.valueChanged.connect(self.update_clip_preview)
        self.spin_sharp_radius.valueChanged.connect(self.update_clip_preview)
        self.check_apply_dark_level.stateChanged.connect(self.on_dark_mode_changed)
        self.check_linearity_enable.stateChanged.connect(self.on_linearity_toggle_changed)
        self.list_widget.currentRowChanged.connect(self.on_list_selection_changed)

        self.rot_widget = QGroupBox("")
        self._pin_section_to_top(self.rot_widget)
        rot_main_layout = QHBoxLayout(self.rot_widget)
        rot_main_layout.setContentsMargins(15, 0, 0, 5)

        rot_controls_layout = QVBoxLayout()

        self.check_rotation_enable = QCheckBox("Enable rotation for export")
        self.check_rotation_enable.stateChanged.connect(self.update_rotation_preview)
        rot_controls_layout.addWidget(self.check_rotation_enable)

        rot_angle_row = QHBoxLayout()
        rot_angle_row.addWidget(QLabel("Angle:"))
        self.combo_rotation = QComboBox()
        self.combo_rotation.addItems(["90° CW", "180°", "270° CW"])
        self.combo_rotation.currentIndexChanged.connect(self.update_rotation_preview)
        rot_angle_row.addWidget(self.combo_rotation)
        rot_angle_row.addStretch()
        rot_controls_layout.addLayout(rot_angle_row)
        rot_controls_layout.addStretch()
        rot_main_layout.addLayout(rot_controls_layout)

        self.rot_preview_label = QLabel("No preview")
        self.rot_preview_label.setFixedSize(180, 130)
        self.rot_preview_label.setAlignment(Qt.AlignCenter)
        self.rot_preview_label.setStyleSheet("border: 1px solid #bbb; background: #f5f5f5;")
        rot_main_layout.addWidget(self.rot_preview_label)

        rotation_toggle = self._create_section_toggle("7) Image rotation", self.rot_widget, default_open=False)

        self.grey_box = QGroupBox("")
        self._pin_section_to_top(self.grey_box)
        grey_layout = QVBoxLayout(self.grey_box)
        grey_layout.setAlignment(Qt.AlignTop)

        grey_top_row = QHBoxLayout()
        self.check_calibrate_light_variance = QCheckBox("calibrate light intensity variance")
        self.check_calibrate_light_variance.setChecked(False)
        grey_top_row.addWidget(self.check_calibrate_light_variance)
        grey_top_row.addSpacing(10)
        grey_top_row.addWidget(QLabel("lights distance (mm):"))
        self.edit_lights_distance_mm = QLineEdit("300")
        self.edit_lights_distance_mm.setFixedWidth(90)
        self.edit_lights_distance_mm.setToolTip("Radius of the hemisphere in mm used to project directions from .dome/.lp files onto the light sphere.")
        grey_top_row.addWidget(self.edit_lights_distance_mm)
        grey_top_row.addSpacing(10)
        grey_top_row.addWidget(QLabel("ROI side (% short side):"))
        self.spin_grey_roi_short_side_pct = QDoubleSpinBox()
        self.spin_grey_roi_short_side_pct.setRange(1.0, 10.0)
        self.spin_grey_roi_short_side_pct.setSingleStep(0.5)
        self.spin_grey_roi_short_side_pct.setDecimals(1)
        self.spin_grey_roi_short_side_pct.setValue(3.0)
        self.spin_grey_roi_short_side_pct.setSuffix(" %")
        self.spin_grey_roi_short_side_pct.setToolTip("ROI side for grey-card calibration as percentage of the image short side (1% to 10%).")
        grey_top_row.addWidget(self.spin_grey_roi_short_side_pct)
        grey_top_row.addStretch()
        grey_layout.addLayout(grey_top_row)

        grey_files_row = QHBoxLayout()
        grey_left_col = QVBoxLayout()
        grey_left_col.addWidget(QLabel("Grey frames (RAW/TIFF16, one per input image):"))
        grey_files_buttons = QHBoxLayout()
        self.btn_add_grey_frames = QPushButton("Add")
        self.btn_add_grey_frames.clicked.connect(self.add_grey_frames)
        grey_files_buttons.addWidget(self.btn_add_grey_frames)
        self.btn_clear_grey_frames = QPushButton("Clear")
        self.btn_clear_grey_frames.clicked.connect(self.clear_grey_frames)
        grey_files_buttons.addWidget(self.btn_clear_grey_frames)
        grey_left_col.addLayout(grey_files_buttons)
        self.grey_frames_list = QListWidget()
        self.grey_frames_list.setMaximumHeight(84)
        grey_left_col.addWidget(self.grey_frames_list)
        grey_files_row.addLayout(grey_left_col)
        grey_layout.addLayout(grey_files_row)

        grey_rotation_row = QHBoxLayout()
        self.check_grey_rotation_enable = QCheckBox("Rotate grey-card frames before processing")
        self.check_grey_rotation_enable.stateChanged.connect(self.on_grey_rotation_setting_changed)
        grey_rotation_row.addWidget(self.check_grey_rotation_enable)
        grey_rotation_row.addWidget(QLabel("Angle:"))
        self.combo_grey_rotation = QComboBox()
        self.combo_grey_rotation.addItems(["90° CW", "180°", "270° CW"])
        self.combo_grey_rotation.setEnabled(False)
        self.combo_grey_rotation.currentIndexChanged.connect(self.on_grey_rotation_setting_changed)
        grey_rotation_row.addWidget(self.combo_grey_rotation)
        grey_rotation_row.addStretch()
        grey_layout.addLayout(grey_rotation_row)

        dome_row = QHBoxLayout()
        dome_row.addWidget(QLabel("Dome/LP file (.dome/.lp):"))
        self.edit_dome_file = QLineEdit("")
        dome_row.addWidget(self.edit_dome_file, 1)
        self.btn_browse_dome = QPushButton("Browse...")
        self.btn_browse_dome.clicked.connect(self.browse_dome_file)
        dome_row.addWidget(self.btn_browse_dome)
        grey_layout.addLayout(dome_row)

        comp_row = QHBoxLayout()
        comp_row.addWidget(QLabel("led intensity compensation file:"))
        self.edit_light_comp_file = QLineEdit("")
        comp_row.addWidget(self.edit_light_comp_file, 1)
        self.btn_browse_light_comp = QPushButton("Browse...")
        self.btn_browse_light_comp.clicked.connect(self.browse_light_comp_file)
        comp_row.addWidget(self.btn_browse_light_comp)
        grey_layout.addLayout(comp_row)

        self.btn_save_light_comp = QPushButton("Save led intensity compensation")
        self.btn_save_light_comp.clicked.connect(self.save_led_intensity_compensation)
        grey_layout.addWidget(self.btn_save_light_comp)

        self.lbl_grey_info = QLabel("Grey-card calibration disabled.")
        grey_layout.addWidget(self.lbl_grey_info)

        flatfield_row = QHBoxLayout()
        self.check_flatfield_enable = QCheckBox("enable flatfielding")
        self.check_flatfield_enable.setChecked(False)
        flatfield_row.addWidget(self.check_flatfield_enable)
        flatfield_row.addStretch()
        grey_layout.addLayout(flatfield_row)

        flatfield_params_row = QHBoxLayout()
        flatfield_params_row.addWidget(QLabel("Flatfield smooth sigma (% short side):"))
        self.spin_flatfield_sigma_pct = QDoubleSpinBox()
        self.spin_flatfield_sigma_pct.setRange(0.0, 25.0)
        self.spin_flatfield_sigma_pct.setSingleStep(0.1)
        self.spin_flatfield_sigma_pct.setDecimals(1)
        self.spin_flatfield_sigma_pct.setValue(2.0)
        self.spin_flatfield_sigma_pct.setSuffix(" %")
        self.spin_flatfield_sigma_pct.setToolTip("Low-frequency smoothing for flat map (higher values = smoother correction, less ring risk).")
        flatfield_params_row.addWidget(self.spin_flatfield_sigma_pct)
        flatfield_params_row.addStretch()
        grey_layout.addLayout(flatfield_params_row)

        self.grey_flat_preview_title = QLabel("Flatfield preview (current input pair):")
        grey_layout.addWidget(self.grey_flat_preview_title)

        grey_preview_row = QHBoxLayout()

        before_col = QVBoxLayout()
        before_col.addWidget(QLabel("Before"))
        self.grey_flat_before_label = QLabel("No preview")
        self.grey_flat_before_label.setFixedSize(180, 130)
        self.grey_flat_before_label.setAlignment(Qt.AlignCenter)
        self.grey_flat_before_label.setStyleSheet("border: 1px solid #bbb; background: #f5f5f5;")
        before_col.addWidget(self.grey_flat_before_label)
        grey_preview_row.addLayout(before_col)

        after_col = QVBoxLayout()
        after_col.addWidget(QLabel("After"))
        self.grey_flat_after_label = QLabel("No preview")
        self.grey_flat_after_label.setFixedSize(180, 130)
        self.grey_flat_after_label.setAlignment(Qt.AlignCenter)
        self.grey_flat_after_label.setStyleSheet("border: 1px solid #bbb; background: #f5f5f5;")
        after_col.addWidget(self.grey_flat_after_label)
        grey_preview_row.addLayout(after_col)

        flat_col = QVBoxLayout()
        flat_col.addWidget(QLabel("Flat frame"))
        self.grey_flat_frame_label = QLabel("No preview")
        self.grey_flat_frame_label.setFixedSize(180, 130)
        self.grey_flat_frame_label.setAlignment(Qt.AlignCenter)
        self.grey_flat_frame_label.setStyleSheet("border: 1px solid #bbb; background: #f5f5f5;")
        flat_col.addWidget(self.grey_flat_frame_label)
        grey_preview_row.addLayout(flat_col)

        grey_layout.addLayout(grey_preview_row)

        self.lbl_flatfield_preview_info = QLabel("")
        grey_layout.addWidget(self.lbl_flatfield_preview_info)

        grey_toggle = self._create_section_toggle("5) Grey Card Calibration", self.grey_box, default_open=False)
        grey_insert_index = layout.indexOf(phase1_toggle)
        layout.insertWidget(grey_insert_index, grey_toggle)
        layout.insertWidget(grey_insert_index + 1, self.grey_box)
        layout.addWidget(rotation_toggle)
        layout.addWidget(self.rot_widget)

        self.check_use_sharpness.stateChanged.connect(self.on_use_sharpness_changed)
        self.check_apply_dark_level.stateChanged.connect(self.update_clip_preview)
        self.edit_dark_lift_coeff.editingFinished.connect(self.on_dark_lift_coeff_edited)
        self.spin_weight_r.valueChanged.connect(self.update_clip_preview)
        self.spin_weight_g.valueChanged.connect(self.update_clip_preview)
        self.spin_weight_b.valueChanged.connect(self.update_clip_preview)
        self.check_calibrate_light_variance.stateChanged.connect(self.update_grey_calibration_ui_state)
        self.spin_grey_roi_short_side_pct.valueChanged.connect(self.update_grey_calibration_ui_state)
        self.check_flatfield_enable.stateChanged.connect(self.update_clip_preview)
        self.edit_dome_file.textChanged.connect(self._on_light_file_path_changed)
        self.check_flatfield_enable.stateChanged.connect(self.update_grey_calibration_ui_state)
        self.check_flatfield_enable.stateChanged.connect(self.update_flatfield_preview_panel)
        self.spin_flatfield_sigma_pct.valueChanged.connect(self.update_clip_preview)
        self.spin_flatfield_sigma_pct.valueChanged.connect(self.update_flatfield_preview_panel)
        self.combo_rotation.currentIndexChanged.connect(self.on_rotation_setting_changed)
        self.check_rotation_enable.stateChanged.connect(self.on_rotation_setting_changed)
        self.edit_light_comp_file.editingFinished.connect(self.on_light_comp_file_edited)

        execution_box = QGroupBox("")
        self._pin_section_to_top(execution_box)
        execution_box.setStyleSheet("QGroupBox { border: 1px solid #9d9d9d; background: #e7e7e7; border-radius: 4px; }")
        execution_layout = QVBoxLayout(execution_box)
        execution_layout.setAlignment(Qt.AlignTop)
        self.lbl_status = QLabel("Ready")
        execution_layout.addWidget(self.lbl_status)

        execution_output_row = QHBoxLayout()
        execution_output_row.addWidget(QLabel("Output Folder:"))
        self.edit_output_folder = QLineEdit("")
        execution_output_row.addWidget(self.edit_output_folder, 1)
        self.btn_browse_output = QPushButton("Browse...")
        self.btn_browse_output.clicked.connect(self.browse_output_folder)
        execution_output_row.addWidget(self.btn_browse_output)
        self.btn_open_set_output = QPushButton("Open Output Folder")
        self.btn_open_set_output.clicked.connect(self.open_selected_output_folder)
        execution_output_row.addWidget(self.btn_open_set_output)
        execution_layout.addLayout(execution_output_row)
        
        self.progress = QProgressBar()
        self.progress.setValue(0)
        execution_layout.addWidget(self.progress)

        action_row = QHBoxLayout()

        self.btn_run = QPushButton()
        self.btn_run.setMinimumHeight(50)
        self.set_button_develop()
        self.btn_run.clicked.connect(self.toggle_processing)
        action_row.addWidget(self.btn_run, 1)

        self.btn_export_current = QPushButton("Export Current Frame")
        self.btn_export_current.setMinimumHeight(34)
        self.btn_export_current.setEnabled(False)
        self.btn_export_current.clicked.connect(self.export_current_preview)
        action_row.addWidget(self.btn_export_current, 0)

        execution_layout.addLayout(action_row)
        console_box = QGroupBox("")
        self._pin_section_to_top(console_box)
        console_layout = QVBoxLayout(console_box)
        console_layout.setAlignment(Qt.AlignTop)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        font = self.console.font()
        font.setFamily("Courier")
        self.console.setFont(font)
        console_layout.addWidget(self.console)
        console_toggle = self._create_section_toggle("8) Python Console Log", console_box, default_open=True)
        layout.addWidget(console_toggle)
        layout.addWidget(console_box)

        root_layout.addWidget(execution_box, 0)

        self.sync_default_output_folder(force=True)
        self.on_luminance_mode_changed()
        self.on_use_sharpness_changed()
        self.on_linearity_toggle_changed()
        self.update_dark_calibration_ui_state()
        self.update_grey_calibration_ui_state()
        self.on_percentile_changed(self.slider_percentile.value())
        self.disable_wheel_on_inputs()

    def _dark_files_enabled_extensions_filter(self):
        return "Images (*.cr2 *.nef *.arw *.dng *.raf *.orf *.rw2 *.raw *.tif *.tiff)"

    def add_dark_frames_cap_off(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Dark Frames", "", self._dark_files_enabled_extensions_filter())
        added = False
        for f in files:
            if is_tiff_file(f):
                try:
                    validate_tiff_16bit(f)
                except Exception as e:
                    print(f"Skipped cap-off TIFF: {e}")
                    continue
            if f not in self.dark_frame_files_cap_off:
                self.dark_frame_files_cap_off.append(f)
                self.dark_frames_list_cap_off.addItem(os.path.basename(f))
                added = True

        if added:
            self.dark_frame_mean_cache_cap_on.clear()
            self.dark_frame_mean_cache_cap_off.clear()
            self.dark_map_cache_cap_on.clear()
            self.dark_bayer_map_cache_cap_on.clear()
            self.dark_map_cache_cap_off.clear()
            self.dark_signal_cache_key = None
            self.dark_signal_cache_value = 0.0
            self.check_apply_dark_level.setChecked(False)
            self.update_dark_calibration_ui_state()
            self.update_clip_preview()

    def clear_dark_frames_cap_off(self):
        self.dark_frame_files_cap_off = []
        self.dark_frames_list_cap_off.clear()
        self.dark_frame_mean_cache_cap_off.clear()
        self.dark_map_cache_cap_off.clear()
        self.dark_signal_cache_key = None
        self.dark_signal_cache_value = 0.0
        self.update_dark_calibration_ui_state()
        self.update_clip_preview()

    def add_dark_frames_cap_on(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Cap-On Dark Frames", "", self._dark_files_enabled_extensions_filter())
        added = False
        for f in files:
            if is_tiff_file(f):
                try:
                    validate_tiff_16bit(f)
                except Exception as e:
                    print(f"Skipped cap-on TIFF: {e}")
                    continue
            if f not in self.dark_frame_files_cap_on:
                self.dark_frame_files_cap_on.append(f)
                self.dark_frames_list_cap_on.addItem(os.path.basename(f))
                added = True

        if added:
            self.dark_frame_mean_cache_cap_on.clear()
            self.dark_map_cache_cap_on.clear()
            self.dark_bayer_map_cache_cap_on.clear()
            self.dark_map_cache_cap_off.clear()
            self.dark_signal_cache_key = None
            self.dark_signal_cache_value = 0.0
            self.check_apply_dark_level.setChecked(False)
            self.update_dark_calibration_ui_state()
            self.update_clip_preview()

    def clear_dark_frames_cap_on(self):
        self.dark_frame_files_cap_on = []
        self.dark_frames_list_cap_on.clear()
        self.dark_frame_mean_cache_cap_on.clear()
        self.dark_map_cache_cap_on.clear()
        self.dark_bayer_map_cache_cap_on.clear()
        self.dark_signal_cache_key = None
        self.dark_signal_cache_value = 0.0
        self.update_dark_calibration_ui_state()
        self.update_clip_preview()

    def _sorted_paths_by_basename(self, paths):
        return sorted(list(paths), key=lambda p: (os.path.basename(p).lower(), p.lower()))

    def _paired_input_and_grey_paths(self):
        input_sorted = self._sorted_paths_by_basename(self.files)
        grey_sorted = self._sorted_paths_by_basename(self.grey_card_files)
        pairs = list(zip(input_sorted, grey_sorted))
        return input_sorted, grey_sorted, pairs

    def _build_flatfield_map_for_inputs(self):
        if not self.files:
            return {}, "No input files loaded."
        if not self.grey_card_files:
            return {}, "Load grey-card frames to use flatfielding."
        if len(self.grey_card_files) != len(self.files):
            return {}, "Grey-card frame count must match input frame count for flatfielding."

        input_sorted, _grey_sorted, pairs = self._paired_input_and_grey_paths()
        if len(pairs) != len(input_sorted):
            return {}, "Unable to pair input and grey-card frames for flatfielding."

        mapping = {}
        for input_path, grey_path in pairs:
            mapping[os.path.basename(input_path).lower()] = grey_path
        return mapping, None

    def _percentile_slider_to_float(self, slider_value):
        return float(slider_value) / float(STRETCH_PERCENT_SCALE)

    def _percentile_float_to_slider(self, percentile_value):
        scaled = int(round(float(percentile_value) * float(STRETCH_PERCENT_SCALE)))
        return max(self.slider_percentile.minimum(), min(self.slider_percentile.maximum(), scaled))

    def _current_percentile_value(self):
        return self._percentile_slider_to_float(self.slider_percentile.value())

    def _format_percentile_text(self, percentile_value):
        return f"{float(percentile_value):.3f}%"

    def _grey_rotation_enabled(self):
        return bool(self.check_grey_rotation_enable.isChecked())

    def _apply_grey_rotation_to_map(self, y_map):
        if y_map is None or not self._grey_rotation_enabled():
            return y_map
        angle_idx = self.combo_grey_rotation.currentIndex()
        if angle_idx == 0:
            return np.rot90(y_map, k=3)
        if angle_idx == 1:
            return np.rot90(y_map, k=2)
        return np.rot90(y_map, k=1)

    def _parse_positive_float(self, text):
        raw_text = str(text).strip().replace(',', '.')
        if not raw_text:
            return None
        try:
            value = float(raw_text)
        except ValueError:
            return None
        if value <= 0:
            return None
        return value

    def _parse_exposure_time_value(self, raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float)):
            val = float(raw_value)
            return val if val > 0 else None

        text = str(raw_value).strip().replace(',', '.')
        if not text:
            return None

        frac_match = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*$', text)
        if frac_match:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den > 0:
                val = num / den
                return val if val > 0 else None

        try:
            val = float(text)
            return val if val > 0 else None
        except ValueError:
            return None

    def _norm_path_key(self, path):
        if not path:
            return ""
        return os.path.normcase(os.path.normpath(os.path.abspath(str(path)))).replace('/', '\\')

    def _read_exposure_times_batch(self, filepaths):
        paths = [p for p in filepaths if os.path.isfile(p)]
        if not paths:
            return {}, "No valid files for exposure-time reading."

        exiftool_exe = get_exiftool_path()
        if not os.path.isfile(exiftool_exe):
            return {}, "exiftool.exe not found. Exposure time is required for linearity calibration."

        cmd = [
            exiftool_exe,
            "-j",
            "-n",
            "-ExposureTime",
            "-EXIF:ExposureTime",
            "-ShutterSpeedValue",
            "-Composite:ShutterSpeed",
            "-charset",
            "filename=UTF8"
        ] + paths

        creationflags = get_subprocess_flags()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=creationflags
            )
        except Exception as e:
            return {}, f"Unable to run ExifTool: {e}"

        if res.returncode != 0:
            return {}, f"ExifTool failed: {res.stderr.strip()}"

        try:
            payload = json.loads(res.stdout)
        except Exception as e:
            return {}, f"Unable to parse ExifTool JSON: {e}"

        out = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            src = row.get('SourceFile')
            if not src:
                continue

            exp = self._parse_exposure_time_value(row.get('ExposureTime'))
            if exp is None:
                exp = self._parse_exposure_time_value(row.get('EXIF:ExposureTime'))
            if exp is None:
                sv = self._parse_exposure_time_value(row.get('ShutterSpeedValue'))
                if sv is not None:
                    exp = float(2.0 ** (-sv))
            if exp is None:
                exp = self._parse_exposure_time_value(row.get('Composite:ShutterSpeed'))

            if exp is not None and exp > 0:
                exp_val = float(exp)
                src_key = self._norm_path_key(src)
                if src_key:
                    out[src_key] = exp_val
                # Fallback key by basename to tolerate minor path-format differences from ExifTool.
                out[os.path.basename(str(src)).lower()] = exp_val

        return out, None

    def _read_capture_consistency_batch(self, filepaths):
        paths = [p for p in filepaths if os.path.isfile(p)]
        if not paths:
            return None, None, "No valid files for capture-consistency check."

        exiftool_exe = get_exiftool_path()
        if not os.path.isfile(exiftool_exe):
            return None, None, "exiftool.exe not found."

        cmd = [
            exiftool_exe,
            "-j",
            "-n",
            "-ISO",
            "-EXIF:ISO",
            "-FNumber",
            "-EXIF:FNumber",
            "-charset",
            "filename=UTF8"
        ] + paths

        creationflags = get_subprocess_flags()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=creationflags
            )
        except Exception as e:
            return None, None, f"Unable to run ExifTool: {e}"

        if res.returncode != 0:
            return None, None, f"ExifTool failed: {res.stderr.strip()}"

        try:
            payload = json.loads(res.stdout)
        except Exception as e:
            return None, None, f"Unable to parse ExifTool JSON: {e}"

        iso_values = []
        fnum_values = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            iso_raw = row.get('ISO', row.get('EXIF:ISO'))
            fnum_raw = row.get('FNumber', row.get('EXIF:FNumber'))

            try:
                iso_val = float(iso_raw)
                if iso_val > 0:
                    iso_values.append(iso_val)
            except Exception:
                pass

            try:
                fnum_val = float(fnum_raw)
                if fnum_val > 0:
                    fnum_values.append(fnum_val)
            except Exception:
                pass

        iso_summary = None
        fnum_summary = None
        if iso_values:
            iso_unique = sorted({round(v, 6) for v in iso_values})
            iso_summary = {'count': len(iso_values), 'unique': iso_unique}
        if fnum_values:
            fnum_unique = sorted({round(v, 6) for v in fnum_values})
            fnum_summary = {'count': len(fnum_values), 'unique': fnum_unique}

        return iso_summary, fnum_summary, None

    def _isotonic_regression_non_decreasing(self, y):
        arr = np.array(y, dtype=np.float64)
        n = int(arr.size)
        if n == 0:
            return arr

        blocks = []
        for i in range(n):
            blocks.append([i, i, arr[i], 1.0])
            while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
                b2 = blocks.pop()
                b1 = blocks.pop()
                w = b1[3] + b2[3]
                avg = (b1[2] * b1[3] + b2[2] * b2[3]) / w
                blocks.append([b1[0], b2[1], avg, w])

        out = np.zeros(n, dtype=np.float64)
        for s, e, avg, _w in blocks:
            out[s:e + 1] = avg
        return out

    def _build_monotonic_inverse_lut_points(self, x_norm, y_norm):
        x = np.array(x_norm, dtype=np.float64)
        y = np.array(y_norm, dtype=np.float64)
        y_iso = self._isotonic_regression_non_decreasing(y)
        y_iso = np.clip(y_iso, 0.0, 1.0)

        cp = [(0.0, 0.0)]
        for xin, yin in zip(x, y_iso):
            cp.append((float(yin), float(xin)))
        cp.append((1.0, 1.0))

        cp.sort(key=lambda t: (t[0], t[1]))
        collapsed = []
        for a, b in cp:
            if collapsed and abs(a - collapsed[-1][0]) < 1e-8:
                collapsed[-1] = (collapsed[-1][0], max(collapsed[-1][1], b))
            else:
                collapsed.append((a, b))

        out = []
        max_b = 0.0
        for a, b in collapsed:
            max_b = max(max_b, float(np.clip(b, 0.0, 1.0)))
            out.append((float(np.clip(a, 0.0, 1.0)), max_b))

        if out[0][0] > 0.0:
            out.insert(0, (0.0, 0.0))
        if out[-1][0] < 1.0:
            out.append((1.0, 1.0))
        return out, y_iso

    def _linearity_points_to_text(self, points):
        lines = ["# input_luma_norm,output_luma_norm"]
        for xin, yout in points:
            lines.append(f"{float(xin):.8f},{float(yout):.8f}")
        return "\n".join(lines)

    def _parse_linearity_lut_text(self, text):
        rows = str(text).splitlines()
        pts = []
        for row in rows:
            line = row.strip()
            if not line or line.startswith('#'):
                continue
            line = line.replace(';', ',').replace('\t', ',')
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0].replace(',', '.'))
                b = float(parts[1].replace(',', '.'))
            except ValueError:
                continue
            pts.append((float(np.clip(a, 0.0, 1.0)), float(np.clip(b, 0.0, 1.0))))

        if len(pts) < 2:
            raise ValueError("At least two LUT points are required.")

        pts.sort(key=lambda t: (t[0], t[1]))
        collapsed = []
        for a, b in pts:
            if collapsed and abs(a - collapsed[-1][0]) < 1e-8:
                collapsed[-1] = (collapsed[-1][0], max(collapsed[-1][1], b))
            else:
                collapsed.append((a, b))

        monotonic = []
        max_b = 0.0
        for a, b in collapsed:
            max_b = max(max_b, b)
            monotonic.append((a, float(np.clip(max_b, 0.0, 1.0))))

        if monotonic[0][0] > 0.0:
            monotonic.insert(0, (0.0, 0.0))
        if monotonic[-1][0] < 1.0:
            monotonic.append((1.0, 1.0))
        return monotonic

    def _linearity_lut_values_from_points(self, points):
        x = np.array([p[0] for p in points], dtype=np.float32)
        y = np.array([p[1] for p in points], dtype=np.float32)
        grid = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
        return np.interp(grid, x, y).astype(np.float32)

    def _current_linearity_lut_points(self):
        text = self.edit_linearity_lut.toPlainText().strip()
        if not text:
            return []
        if text == self._linearity_lut_cache_text and self._linearity_lut_cache_points:
            return list(self._linearity_lut_cache_points)
        pts = self._parse_linearity_lut_text(text)
        self._linearity_lut_cache_text = text
        self._linearity_lut_cache_points = list(pts)
        return pts

    def _apply_linearity_lut_if_enabled(self, y_norm):
        if not self.check_linearity_enable.isChecked():
            return y_norm
        pts = self._current_linearity_lut_points()
        if not pts:
            return y_norm
        lut = self._linearity_lut_values_from_points(pts)
        grid = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
        src = np.clip(y_norm, 0.0, 1.0).astype(np.float32)
        dst = np.interp(src.reshape(-1), grid, lut)
        return dst.reshape(src.shape).astype(np.float32)

    def _render_linearity_plot(self, x_s, y_obs_norm, y_fit_norm):
        w, h = 620, 280
        canvas = np.full((h, w, 3), 245, dtype=np.uint8)

        m_left, m_right, m_top, m_bottom = 56, 18, 20, 38
        x0, y0 = m_left, h - m_bottom
        x1, y1 = w - m_right, m_top

        cv2.rectangle(canvas, (x0, y1), (x1, y0), (255, 255, 255), thickness=-1)
        cv2.rectangle(canvas, (x0, y1), (x1, y0), (170, 170, 170), thickness=1)

        min_x = float(np.min(x_s))
        max_x = float(np.max(x_s))
        span_x = max(1e-12, max_x - min_x)

        def _px(tx, ty):
            xx = int(x0 + (float(tx) - min_x) / span_x * (x1 - x0))
            yy = int(y0 - float(np.clip(ty, 0.0, 1.0)) * (y0 - y1))
            return xx, yy

        for t in np.linspace(0.0, 1.0, 6):
            gx = int(x0 + t * (x1 - x0))
            gy = int(y0 - t * (y0 - y1))
            cv2.line(canvas, (gx, y1), (gx, y0), (235, 235, 235), 1)
            cv2.line(canvas, (x0, gy), (x1, gy), (235, 235, 235), 1)

        p_a = _px(min_x, 0.0)
        p_b = _px(max_x, 1.0)
        cv2.line(canvas, p_a, p_b, (175, 175, 175), 1, cv2.LINE_AA)

        fit_pts = [_px(xv, yv) for xv, yv in zip(x_s, y_fit_norm)]
        for i in range(1, len(fit_pts)):
            cv2.line(canvas, fit_pts[i - 1], fit_pts[i], (40, 140, 40), 2, cv2.LINE_AA)

        for xv, yv in zip(x_s, y_obs_norm):
            px, py = _px(xv, yv)
            cv2.circle(canvas, (px, py), 3, (30, 70, 190), thickness=-1, lineType=cv2.LINE_AA)

        cv2.putText(canvas, "Y: Grey level (normalized)", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(canvas, "X: Exposure time (s)", (x0, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{min_x:.6g}", (x0 - 10, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{max_x:.6g}", (x1 - 40, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(self.linearity_plot_label.width() - 2, self.linearity_plot_label.height() - 2, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def add_linearity_calibration_frames(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Linearity Calibration Frames",
            "",
            self._dark_files_enabled_extensions_filter()
        )
        added = False
        for f in files:
            if is_tiff_file(f):
                try:
                    validate_tiff_16bit(f)
                except Exception as e:
                    print(f"Skipped linearity TIFF: {e}")
                    continue
            if f not in self.linearity_calibration_files:
                self.linearity_calibration_files.append(f)
                self.linearity_list.addItem(os.path.basename(f))
                added = True
        if added:
            self.lbl_linearity_info.setText(f"Linearity calibration frames: {len(self.linearity_calibration_files)}")

    def clear_linearity_calibration_frames(self):
        self.linearity_calibration_files = []
        self._linearity_cached_rows = []
        self._linearity_cached_frame_report = []
        self._linearity_cached_consistency_warnings = []
        self.spin_linearity_exclude_last_n.setValue(0)
        self.spin_linearity_exclude_last_n.setMaximum(0)
        self.linearity_list.clear()
        self.lbl_linearity_info.setText("No linearity calibration frames loaded. If empty, current input files will be used.")

    def on_linearity_toggle_changed(self):
        enabled = self.check_linearity_enable.isChecked()
        self.btn_load_linearity_lut.setEnabled(enabled)
        self.btn_save_linearity_lut.setEnabled(enabled)
        self.edit_linearity_lut.setEnabled(enabled)
        self.btn_apply_linearity_lut.setEnabled(enabled)
        self.spin_linearity_exclude_last_n.setEnabled(enabled)
        self.btn_apply_linearity_exclusion.setEnabled(enabled)
        self.update_clip_preview()

    def _default_linearity_lut_preset_path(self):
        if self.grey_card_files:
            base_dir = os.path.dirname(self.grey_card_files[0])
        elif self.files:
            base_dir = os.path.dirname(self.files[0])
        elif self.linearity_calibration_files:
            base_dir = os.path.dirname(self.linearity_calibration_files[0])
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "sensor_linearity_lut.txt")

    def _autosave_linearity_lut_preset_silent(self, points):
        if not points:
            return None
        path = self._default_linearity_lut_preset_path()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._linearity_points_to_text(points))
                f.write('\n')
            return path
        except Exception as e:
            print(f"Warning: unable to autosave linearity LUT preset: {e}")
            return None

    def _default_linearity_report_preset_path(self):
        if self.files:
            base_dir = os.path.dirname(self.files[0])
        elif self.linearity_calibration_files:
            base_dir = os.path.dirname(self.linearity_calibration_files[0])
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "sensor_linearity_report.txt")

    def _autosave_linearity_report_silent(self, report_text):
        if not report_text:
            return None
        path = self._default_linearity_report_preset_path()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(str(report_text).rstrip())
                f.write('\n')
            return path
        except Exception as e:
            print(f"Warning: unable to autosave linearity report: {e}")
            return None

    def save_linearity_lut_preset(self):
        try:
            pts = self._current_linearity_lut_points()
            if not pts:
                raise ValueError("LUT text is empty.")
        except Exception as e:
            QMessageBox.warning(self, "Linearity Calibration", f"Invalid LUT format: {e}")
            return

        default_path = self._default_linearity_lut_preset_path()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save LUT Preset",
            default_path,
            "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith('.txt'):
            path += '.txt'

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._linearity_points_to_text(pts))
                f.write('\n')
        except Exception as e:
            QMessageBox.warning(self, "Linearity Calibration", f"Unable to save LUT preset: {e}")
            return

        self.lbl_linearity_info.setText(f"LUT preset saved: {os.path.basename(path)} ({len(pts)} points)")
        QMessageBox.information(self, "Linearity Calibration", f"LUT preset saved:\n{path}")

    def load_linearity_lut_preset(self):
        default_path = self._default_linearity_lut_preset_path()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load LUT Preset",
            default_path,
            "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                txt = f.read()
            pts = self._parse_linearity_lut_text(txt)
        except Exception as e:
            QMessageBox.warning(self, "Linearity Calibration", f"Unable to load LUT preset: {e}")
            return

        normalized_text = self._linearity_points_to_text(pts)
        self.edit_linearity_lut.setPlainText(normalized_text)
        self._linearity_lut_cache_text = normalized_text.strip()
        self._linearity_lut_cache_points = list(pts)
        self.update_clip_preview()
        self.lbl_linearity_info.setText(f"LUT preset loaded: {os.path.basename(path)} ({len(pts)} points)")
        QMessageBox.information(self, "Linearity Calibration", f"LUT preset loaded:\n{path}")

    def apply_linearity_lut_text(self):
        try:
            pts = self._current_linearity_lut_points()
            if not pts:
                raise ValueError("LUT text is empty.")
        except Exception as e:
            QMessageBox.warning(self, "Linearity Calibration", f"Invalid LUT format: {e}")
            return

        self.edit_linearity_lut.setPlainText(self._linearity_points_to_text(pts))
        self._linearity_lut_cache_text = self.edit_linearity_lut.toPlainText().strip()
        self._linearity_lut_cache_points = list(pts)
        autosave_path = self._autosave_linearity_lut_preset_silent(pts)
        self.update_clip_preview()
        if autosave_path:
            self.lbl_linearity_info.setText(f"LUT validated and autosaved: {os.path.basename(autosave_path)} ({len(pts)} points)")
        QMessageBox.information(self, "Linearity Calibration", f"LUT loaded successfully ({len(pts)} control points).")

    def _format_linearity_frame_report(self, frame_rows, analysis_info=None):
        if not frame_rows:
            return "No frame diagnostics available."

        lines = []
        if isinstance(analysis_info, dict) and analysis_info:
            lines.append(
                f"Manual exclusion: drop last N high-exposure frames = {int(analysis_info.get('excluded_tail_n', 0))}"
            )
            lines.append(
                f"Burned indicator threshold (for diagnostics only): {float(analysis_info.get('burnt_threshold_pct', 0.0)):.2f}%"
            )
            lines.append("")

        header = f"{'Status':<10} {'File':<44} {'Exposure(s)':>12} {'CenterMean':>12} {'Burned%':>9}  Reason"
        sep = "-" * len(header)
        lines.extend([header, sep])

        for row in frame_rows:
            status = str(row.get('status', '-'))[:10]
            name = os.path.basename(str(row.get('file', '-')))
            if len(name) > 44:
                name = name[:41] + "..."

            exp_s = row.get('exp_s', None)
            mean_y = row.get('mean_y', None)
            burned_pct = row.get('burned_pct', None)
            reason = str(row.get('reason', ''))

            exp_txt = f"{float(exp_s):.6g}" if exp_s is not None else "-"
            mean_txt = f"{float(mean_y):.6f}" if mean_y is not None else "-"
            burn_txt = f"{float(burned_pct):.2f}" if burned_pct is not None else "-"

            lines.append(f"{status:<10} {name:<44} {exp_txt:>12} {mean_txt:>12} {burn_txt:>9}  {reason}")

        return "\n".join(lines)

    def analyze_sensor_linearity(self):
        calib_files = list(self.linearity_calibration_files) if self.linearity_calibration_files else list(self.files)
        if len(calib_files) < 3:
            QMessageBox.warning(self, "Linearity Calibration", "Load at least 3 calibration frames (or 3 input files).")
            return

        self.lbl_status.setText("Analyzing sensor linearity...")
        self.progress.setValue(0)
        QApplication.processEvents()

        exp_map, err = self._read_exposure_times_batch(calib_files)
        if err:
            self.lbl_status.setText("Ready")
            QMessageBox.warning(self, "Linearity Calibration", err)
            return

        iso_summary, fnum_summary, consistency_err = self._read_capture_consistency_batch(calib_files)
        consistency_warnings = []
        if consistency_err:
            consistency_warnings.append(f"Capture-consistency check unavailable: {consistency_err}")
        else:
            if iso_summary and len(iso_summary.get('unique', [])) > 1:
                consistency_warnings.append(
                    "ISO varies across calibration frames: "
                    + ", ".join(str(int(v)) if abs(v - round(v)) < 1e-6 else f"{v:g}" for v in iso_summary['unique'])
                )
            if fnum_summary and len(fnum_summary.get('unique', [])) > 1:
                consistency_warnings.append(
                    "Aperture (FNumber) varies across calibration frames: "
                    + ", ".join(f"f/{v:g}" for v in fnum_summary['unique'])
                )
        for warn in consistency_warnings:
            print(f"Linearity warning: {warn}")

        rows = []
        missing_exposure = []
        failed_roi = []
        frame_report = []
        dark_ctx = self._build_grey_dark_context()
        burnt_threshold_pct = self._current_percentile_value()
        analysis_info = {'excluded_tail_n': 0, 'burnt_threshold_pct': burnt_threshold_pct}
        for i, f in enumerate(calib_files):
            exp_s = exp_map.get(self._norm_path_key(f), None)
            if exp_s is None:
                exp_s = exp_map.get(os.path.basename(f).lower(), None)
            if exp_s is None or exp_s <= 0:
                missing_exposure.append(f)
                frame_report.append({
                    'status': 'excluded',
                    'file': f,
                    'exp_s': None,
                    'mean_y': None,
                    'burned_pct': None,
                    'reason': 'Missing exposure metadata'
                })
                self.progress.setValue(int((i + 1) / max(1, len(calib_files)) * 100))
                QApplication.processEvents()
                continue

            try:
                mean_y, burned_ratio, burned_count, burned_total, roi_p99 = self._compute_center_patch_stats(
                    f,
                    patch_size=20,
                    dark_ctx=dark_ctx,
                    burnt_threshold_pct=burnt_threshold_pct
                )
            except Exception:
                failed_roi.append(f)
                frame_report.append({
                    'status': 'excluded',
                    'file': f,
                    'exp_s': float(exp_s),
                    'mean_y': None,
                    'burned_pct': None,
                    'reason': 'ROI sampling failed'
                })
                mean_y = None

            if mean_y is not None and exp_s is not None and exp_s > 0:
                rows.append((f, float(exp_s), float(mean_y)))
                burned_pct = float(burned_ratio * 100.0)
                reason = 'Included'
                frame_report.append({
                    'status': 'included',
                    'file': f,
                    'exp_s': float(exp_s),
                    'mean_y': float(mean_y),
                    'burned_pct': burned_pct,
                    'reason': reason
                })

            self.progress.setValue(int((i + 1) / max(1, len(calib_files)) * 100))
            QApplication.processEvents()

        if len(rows) < 3:
            self.lbl_status.setText("Ready")
            diag = [
                f"Total frames: {len(calib_files)}",
                f"Valid frames: {len(rows)}",
                f"Missing exposure time: {len(missing_exposure)}",
                f"ROI sampling failures: {len(failed_roi)}"
            ]
            if consistency_warnings:
                diag.append("")
                diag.append("Capture-consistency warnings:")
                for w in consistency_warnings:
                    diag.append(f"- {w}")
            if missing_exposure:
                diag.append("")
                diag.append("Missing exposure examples:")
                for fp in missing_exposure[:8]:
                    diag.append(f"- {os.path.basename(fp)}")
            if failed_roi:
                diag.append("")
                diag.append("ROI failure examples:")
                for fp in failed_roi[:8]:
                    diag.append(f"- {os.path.basename(fp)}")
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Linearity Calibration")
            msg.setText("Not enough valid samples with exposure time and center patch value.")
            msg.setInformativeText("\n".join(diag))
            detailed = self._format_linearity_frame_report(frame_report, analysis_info)
            msg.setDetailedText(detailed)
            self._autosave_linearity_report_silent("\n".join([
                "Linearity Calibration Report",
                "",
                "Result: insufficient valid samples",
                "",
                *diag,
                "",
                detailed
            ]))
            msg.exec()
            return

        self._linearity_cached_rows = list(rows)
        self._linearity_cached_frame_report = list(frame_report)
        self._linearity_cached_consistency_warnings = list(consistency_warnings)
        self.spin_linearity_exclude_last_n.setMaximum(max(0, len(rows) - 3))
        self._rebuild_linearity_from_cached_rows(show_message=True)

    def apply_linearity_tail_exclusion(self):
        if not self._linearity_cached_rows:
            QMessageBox.information(self, "Linearity Calibration", "Run linearity analysis first.")
            return
        self._rebuild_linearity_from_cached_rows(show_message=False)

    def _rebuild_linearity_from_cached_rows(self, show_message=False):
        rows = list(self._linearity_cached_rows)
        if len(rows) < 3:
            QMessageBox.warning(self, "Linearity Calibration", "Not enough cached valid frames. Re-run analysis.")
            return

        rows.sort(key=lambda t: t[1])
        max_drop = max(0, len(rows) - 3)
        drop_n = int(np.clip(self.spin_linearity_exclude_last_n.value(), 0, max_drop))
        if self.spin_linearity_exclude_last_n.value() != drop_n:
            self.spin_linearity_exclude_last_n.setValue(drop_n)

        kept_rows = rows[:-drop_n] if drop_n > 0 else list(rows)
        grouped = {}
        for _f, ex, yy in kept_rows:
            grouped.setdefault(ex, []).append(yy)

        exps = sorted(grouped.keys())
        if len(exps) < 3:
            QMessageBox.warning(self, "Linearity Calibration", "Not enough exposure points after manual exclusion.")
            return

        yvals = [float(np.mean(grouped[e])) for e in exps]
        x = np.array(exps, dtype=np.float64)
        y = np.array(yvals, dtype=np.float64)

        if np.max(x) <= 0 or np.max(y) <= 0:
            self.lbl_status.setText("Ready")
            QMessageBox.warning(self, "Linearity Calibration", "Invalid calibration data range.")
            return

        x_norm = x / float(np.max(x))
        y_norm = y / float(np.max(y))

        mask = (y_norm > 0.01) & (y_norm < 0.99)
        if np.count_nonzero(mask) >= 3:
            x_fit = x_norm[mask]
            y_fit = y_norm[mask]
        else:
            x_fit = x_norm
            y_fit = y_norm

        cp, _y_iso = self._build_monotonic_inverse_lut_points(x_fit, y_fit)

        denom = float(np.sum(x_fit * x_fit))
        a = float(np.sum(x_fit * y_fit) / denom) if denom > 1e-12 else 1.0
        pred = a * x_fit
        ss_res = float(np.sum((y_fit - pred) ** 2))
        ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        max_abs_err = float(np.max(np.abs(y_fit - pred))) if y_fit.size > 0 else 0.0

        y_iso_plot = np.clip(self._isotonic_regression_non_decreasing(y_norm), 0.0, 1.0)
        pix = self._render_linearity_plot(x, y_norm, y_iso_plot)
        self.linearity_plot_label.setPixmap(pix)
        self.linearity_plot_label.setText("")

        lut_text = self._linearity_points_to_text(cp)
        self.edit_linearity_lut.setPlainText(lut_text)
        self._linearity_lut_cache_text = lut_text.strip()
        self._linearity_lut_cache_points = list(cp)
        autosave_path = self._autosave_linearity_lut_preset_silent(cp)

        state_txt = "close-to-linear" if (r2 >= 0.995 and max_abs_err <= 0.02) else "non-linear behavior detected"
        self.lbl_linearity_info.setText(
            f"Samples: {len(x)} | Excluded tail: {drop_n} | R^2: {r2:.6f} | max abs err: {max_abs_err:.4f} | {state_txt}."
        )
        if autosave_path:
            print(f"Linearity LUT autosaved: {autosave_path}")
        self.lbl_status.setText("Ready")

        self.progress.setValue(100)
        self.update_clip_preview()

        warnings = list(self._linearity_cached_consistency_warnings)
        analysis_info = {
            'excluded_tail_n': drop_n,
            'burnt_threshold_pct': self._current_percentile_value()
        }
        frame_report = list(self._linearity_cached_frame_report)
        if drop_n > 0:
            for fpath, exp_s, mean_y in rows[-drop_n:]:
                frame_report.append({
                    'status': 'excluded',
                    'file': fpath,
                    'exp_s': float(exp_s),
                    'mean_y': float(mean_y),
                    'burned_pct': None,
                    'reason': 'Manual tail exclusion'
                })

        detailed = self._format_linearity_frame_report(frame_report, analysis_info)
        report_lines = [
            "Linearity Calibration Report",
            "",
            f"Valid samples after exclusion: {len(x)}",
            f"Excluded tail frames: {drop_n}",
            f"R^2 (linear model): {r2:.6f}",
            f"Max abs error: {max_abs_err:.4f}",
        ]
        if warnings:
            report_lines.extend(["", "Capture-consistency warnings:"])
            report_lines.extend([f"- {w}" for w in warnings])
        report_lines.extend(["", detailed])
        report_path = self._autosave_linearity_report_silent("\n".join(report_lines))
        if report_path:
            print(f"Linearity report autosaved: {report_path}")

        if show_message:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Linearity Calibration")
            msg.setText("Analysis completed.")
            warning_block = "\n".join([f"Warning: {w}" for w in warnings])
            if warning_block:
                warning_block += "\n\n"
            else:
                warning_block = "\n"
            msg.setInformativeText(
                f"Valid samples after exclusion: {len(x)}\n"
                f"Excluded tail frames: {drop_n}\n"
                f"R^2 (linear model): {r2:.6f}\n"
                f"Max abs error: {max_abs_err:.4f}\n"
                + warning_block
                + "Monotonic inverse LUT generated and loaded into editable text area."
            )
            msg.setDetailedText(detailed)
            msg.exec()

    def _compute_center_patch_stats(self, filepath, patch_size=20, dark_ctx=None, burnt_threshold_pct=None):
        ctx = dark_ctx if isinstance(dark_ctx, dict) else self._build_grey_dark_context()

        def _roi_stats(values):
            values = np.asarray(values, dtype=np.float32)
            if burnt_threshold_pct is None:
                burned_mask = np.zeros(values.shape, dtype=bool)
            else:
                burned_mask = values > (float(burnt_threshold_pct) / 100.0)

            burned_count = int(np.count_nonzero(burned_mask))
            burned_total = int(values.size)
            valid = values[~burned_mask]
            if valid.size == 0:
                valid = values
            mean_val = float(np.mean(valid)) if valid.size > 0 else 0.0
            burned_ratio = float(burned_count) / float(max(1, burned_total))
            p99 = float(np.percentile(values, 99.0)) if values.size > 0 else 0.0
            return mean_val, burned_ratio, burned_count, burned_total, p99

        y = self._read_linear_luma_norm_for_grey(filepath, dark_ctx=ctx)
        h, w = y.shape[:2]
        p = max(4, min(int(patch_size), h, w))
        if p <= 0:
            raise ValueError(f"Invalid image size for center patch: {os.path.basename(filepath)}")

        x0 = (w - p) // 2
        y0 = (h - p) // 2
        patch = y[y0:y0 + p, x0:x0 + p].astype(np.float32)
        if patch.size == 0:
            raise ValueError(f"Unable to sample center patch: {os.path.basename(filepath)}")
        return _roi_stats(patch)

    def _build_grey_dark_context(self):
        has_manual_dark = (len(self.dark_frame_files_cap_on) > 0) or (len(self.dark_frame_files_cap_off) > 0)
        apply_dark_level = self.check_apply_dark_level.isChecked() and (not has_manual_dark)
        cap_on_offset = self._preview_compute_frame_mean(self.dark_frame_files_cap_on) if self.dark_frame_files_cap_on else 0.0
        cap_off_raw = self._preview_compute_frame_mean(self.dark_frame_files_cap_off) if self.dark_frame_files_cap_off else 0.0
        dark_frame_offset = max(0.0, cap_off_raw - cap_on_offset)
        return {
            'apply_dark_level': apply_dark_level,
            'cap_on_offset': cap_on_offset,
            'dark_frame_offset': dark_frame_offset,
            'scale': self._manual_dark_scale()
        }

    def _read_linear_luma_norm_for_grey(self, filepath, dark_ctx=None):
        demosaic = self.demosaic_map[self.combo_demosaic.currentText()]
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        ctx = dark_ctx if isinstance(dark_ctx, dict) else self._build_grey_dark_context()
        apply_dark_level = bool(ctx.get('apply_dark_level', False))
        cap_on_offset = float(ctx.get('cap_on_offset', 0.0))
        dark_frame_offset = float(ctx.get('dark_frame_offset', 0.0))
        scale = float(ctx.get('scale', 1.0))

        if ext in {'tif', 'tiff'}:
            rgb = validate_tiff_16bit(filepath)
            if len(rgb.shape) == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32)
            if cap_on_offset > 0:
                rgb = np.clip(rgb - (cap_on_offset * 65535.0 * scale), 0.0, None)
            if dark_frame_offset > 0:
                rgb = np.clip(rgb - (dark_frame_offset * 65535.0 * scale), 0.0, None)
            y = self._compute_grey_luminance_from_rgb(rgb)
            return self._apply_grey_rotation_to_map(y)

        with rawpy.imread(filepath) as raw:
            kwargs = dict(
                half_size=False,
                demosaic_algorithm=demosaic,
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=True,
                no_auto_scale=True,
                use_camera_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0]
            )
            if apply_dark_level:
                dark_level = self._get_raw_dark_level(raw)
                kwargs['user_black'] = self._effective_dark_level(dark_level) if dark_level is not None else 0
            else:
                kwargs['user_black'] = 0

            rgb = raw.postprocess(**kwargs).astype(np.float32)
            if cap_on_offset > 0:
                rgb = np.clip(rgb - (cap_on_offset * 65535.0 * scale), 0.0, None)
            if dark_frame_offset > 0:
                rgb = np.clip(rgb - (dark_frame_offset * 65535.0 * scale), 0.0, None)
            y = self._compute_grey_luminance_from_rgb(rgb)
            return self._apply_grey_rotation_to_map(y)

    def _preview_apply_flatfield(self, input_path, y_norm, dark_ctx=None):
        if not self.check_flatfield_enable.isChecked():
            return y_norm

        flat_map, err = self._build_flatfield_map_for_inputs()
        if err:
            return y_norm

        grey_path = flat_map.get(os.path.basename(input_path).lower())
        if not grey_path or (not os.path.isfile(grey_path)):
            return y_norm

        flat_y = self._preview_grey_luma_norm_map(grey_path, dark_ctx=dark_ctx, target_shape=y_norm.shape[:2])
        if flat_y is None:
            return y_norm

        gain_map = self._preview_build_flatfield_gain_map(flat_y)
        if gain_map is None:
            return y_norm

        corrected = y_norm * gain_map
        return np.clip(corrected, 0.0, None)

    def _preview_build_flatfield_gain_map(self, flat_y):
        if flat_y is None or flat_y.size == 0:
            return None

        src = np.maximum(flat_y.astype(np.float32), 1e-8)
        min_side = float(max(1, min(src.shape[0], src.shape[1])))
        sigma_rel = float(np.clip(self.spin_flatfield_sigma_pct.value() / 100.0, 0.0, 0.25))
        sigma = sigma_rel * min_side

        if sigma >= 0.5:
            log_src = np.log(src)
            log_lp = cv2.GaussianBlur(
                log_src,
                (0, 0),
                sigmaX=float(sigma),
                sigmaY=float(sigma),
                borderType=cv2.BORDER_REPLICATE
            )
            src_lp = np.exp(log_lp).astype(np.float32)
        else:
            src_lp = src

        h, w = src_lp.shape[:2]
        p = max(8, min(20, h, w))
        y0 = (h - p) // 2
        x0 = (w - p) // 2
        patch = src_lp[y0:y0 + p, x0:x0 + p]
        if patch.size == 0:
            return None
        low = float(np.percentile(patch, 5.0))
        high = float(np.percentile(patch, 95.0))
        trimmed = patch[(patch >= low) & (patch <= high)]
        center_ref = float(np.mean(trimmed)) if trimmed.size > 0 else float(np.mean(patch))
        if center_ref <= 1e-8:
            return None

        gain = center_ref / np.maximum(src_lp, 1e-8)
        return gain.astype(np.float32)

    def _apply_rotation_to_map(self, y_map):
        if y_map is None or not self.check_rotation_enable.isChecked():
            return y_map
        angle_idx = self.combo_rotation.currentIndex()
        if angle_idx == 0:
            return np.rot90(y_map, k=3)
        if angle_idx == 1:
            return np.rot90(y_map, k=2)
        return np.rot90(y_map, k=1)

    def _preview_grey_luma_norm_map(self, filepath, dark_ctx=None, target_shape=None):
        ext = filepath.lower().split('.')[-1]
        raw_exts = {'cr2', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2', 'raw'}
        demosaic = self.demosaic_map[self.combo_demosaic.currentText()]
        downsample = max(1, self.spin_undersample.value())
        ctx = dark_ctx if isinstance(dark_ctx, dict) else self._build_grey_dark_context()
        apply_dark_level = bool(ctx.get('apply_dark_level', False))
        cap_on_offset = float(ctx.get('cap_on_offset', 0.0))
        dark_frame_offset = float(ctx.get('dark_frame_offset', 0.0))
        scale = float(ctx.get('scale', 1.0))

        if ext in raw_exts:
            half_size = downsample > 1 and downsample % 2 == 0
            post_ds = max(1, downsample // 2) if half_size else downsample
            with rawpy.imread(filepath) as raw:
                kwargs = dict(
                    half_size=half_size,
                    demosaic_algorithm=demosaic,
                    output_bps=16,
                    gamma=(1, 1),
                    no_auto_bright=True,
                    no_auto_scale=True,
                    use_camera_wb=False,
                    user_wb=[1.0, 1.0, 1.0, 1.0]
                )
                if apply_dark_level:
                    dark_level = self._get_raw_dark_level(raw)
                    kwargs['user_black'] = self._effective_dark_level(dark_level) if dark_level is not None else 0
                else:
                    kwargs['user_black'] = 0

                rgb = raw.postprocess(**kwargs).astype(np.float32)
            if cap_on_offset > 0:
                rgb = np.clip(rgb - (cap_on_offset * 65535.0 * scale), 0.0, None)
            if dark_frame_offset > 0:
                rgb = np.clip(rgb - (dark_frame_offset * 65535.0 * scale), 0.0, None)
            y = self._compute_grey_luminance_from_rgb(rgb)
            y = self._apply_grey_rotation_to_map(y)
            if post_ds > 1:
                y = y[::post_ds, ::post_ds]
        else:
            y = self._read_linear_luma_norm_for_grey(filepath, dark_ctx=ctx)
            if y is None:
                return None
            if downsample > 1:
                y = y[::downsample, ::downsample]

        if target_shape is not None and y.shape[:2] != tuple(target_shape):
            y = cv2.resize(y, (int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_LINEAR)
        return y

    def _compute_center_patch_mean(self, filepath, patch_size=20, dark_ctx=None):
        ctx = dark_ctx if isinstance(dark_ctx, dict) else self._build_grey_dark_context()

        def _trimmed_mean(arr):
            low = float(np.percentile(arr, 5.0))
            high = float(np.percentile(arr, 95.0))
            trimmed = arr[(arr >= low) & (arr <= high)]
            if trimmed.size == 0:
                return float(np.mean(arr))
            return float(np.mean(trimmed))

        y = self._read_linear_luma_norm_for_grey(filepath, dark_ctx=ctx)
        h, w = y.shape[:2]
        p = max(4, min(int(patch_size), h, w))
        if p <= 0:
            raise ValueError(f"Invalid image size for center patch: {os.path.basename(filepath)}")

        x0 = (w - p) // 2
        y0 = (h - p) // 2
        patch = y[y0:y0 + p, x0:x0 + p].astype(np.float32)
        if patch.size == 0:
            raise ValueError(f"Unable to sample center patch: {os.path.basename(filepath)}")
        return _trimmed_mean(patch)

    def _load_dome_light_positions_mm(self, dome_path, lights_distance_mm):
        with open(dome_path, 'r', encoding='utf-8-sig') as f:
            dome_data = json.load(f)

        positions3d = dome_data.get('positions3d', [])
        directions = dome_data.get('directions', [])

        positions = []
        if isinstance(positions3d, list) and positions3d:
            for p in positions3d:
                x = float(p.get('x', 0.0))
                y = float(p.get('y', 0.0))
                z = float(p.get('z', 0.0))
                if lights_distance_mm is not None:
                    r = math.sqrt(x * x + y * y + z * z)
                    if r > 1e-9:
                        s = float(lights_distance_mm) / r
                        x, y, z = x * s, y * s, z * s
                positions.append((x, y, z))
            return positions

        if not isinstance(directions, list) or not directions:
            raise ValueError(".dome file does not contain positions3d or directions")
        if lights_distance_mm is None:
            raise ValueError("lights distance (mm) is required when .dome has only directions")

        for d in directions:
            x = float(d.get('x', 0.0))
            y = float(d.get('y', 0.0))
            z = float(d.get('z', 0.0))
            n = math.sqrt(x * x + y * y + z * z)
            if n <= 1e-9:
                continue
            s = float(lights_distance_mm) / n
            positions.append((x * s, y * s, z * s))
        if not positions:
            raise ValueError("No valid light directions found in .dome file")
        return positions

    def _load_light_positions_mm(self, light_path, lights_distance_mm):
        ext = os.path.splitext(light_path)[1].lower()
        if ext == '.lp':
            return parse_lp_light_positions(light_path, lights_distance_mm)
        return self._load_dome_light_positions_mm(light_path, lights_distance_mm)

    def _compute_light_compensation_payload(self, patch_size=20, progress_callback=None):
        if not self.files:
            raise ValueError("No input files loaded")
        if not self.grey_card_files:
            raise ValueError("No grey-card files loaded")
        if len(self.grey_card_files) != len(self.files):
            raise ValueError("Grey-card file count must match input file count")

        dome_path = self.edit_dome_file.text().strip()
        if not dome_path:
            raise ValueError("Please select a .dome file")
        if not os.path.isfile(dome_path):
            raise ValueError("Selected .dome file does not exist")

        lights_distance_mm = self._parse_positive_float(self.edit_lights_distance_mm.text())
        embedded_radius = None
        if os.path.splitext(dome_path)[1].lower() == '.dome':
            embedded_radius = get_embedded_dome_radius_mm(dome_path)
            if embedded_radius is not None:
                lights_distance_mm = embedded_radius
                self.edit_lights_distance_mm.setText(str(embedded_radius))
        positions = self._load_light_positions_mm(dome_path, lights_distance_mm)

        input_sorted, grey_sorted, pairs = self._paired_input_and_grey_paths()
        if len(positions) < len(pairs):
            raise ValueError(".dome light count is lower than input image count")

        dark_ctx = self._build_grey_dark_context()
        total = len(pairs)
        if callable(progress_callback):
            progress_callback(0, total)

        rows = []
        corrected_levels = []
        roi_short_side_percent = float(np.clip(self.spin_grey_roi_short_side_pct.value(), 1.0, 10.0))
        for i, (input_path, grey_path) in enumerate(pairs):
            y_map = self._read_linear_luma_norm_for_grey(grey_path, dark_ctx=dark_ctx)
            if y_map is None or y_map.size == 0:
                raise ValueError(f"Unable to read grey-card luminance: {os.path.basename(grey_path)}")

            h, w = y_map.shape[:2]
            short_side = max(1, min(h, w))
            roi_side = int(round(short_side * (roi_short_side_percent / 100.0)))
            roi_side = max(1, min(roi_side, h, w))
            x0 = (w - roi_side) // 2
            y0 = (h - roi_side) // 2
            patch = y_map[y0:y0 + roi_side, x0:x0 + roi_side].astype(np.float32)
            if patch.size == 0:
                raise ValueError(f"Unable to sample ROI from grey-card: {os.path.basename(grey_path)}")

            grey_level = float(np.mean(patch))
            x, y, z = positions[i]
            r = math.sqrt(x * x + y * y + z * z)
            z_eff = max(1e-9, float(z))
            if r <= 1e-9:
                raise ValueError(f"Invalid light distance for LED index {i + 1}")
            geometry_factor = z_eff / (r ** 3)
            if geometry_factor <= 1e-12:
                raise ValueError(f"Invalid geometry factor for LED index {i + 1}")

            corrected_level = grey_level / geometry_factor
            corrected_levels.append(corrected_level)
            rows.append({
                'index': i + 1,
                'grey_file': os.path.basename(grey_path),
                'roi_short_side_percent': float(roi_short_side_percent),
                'roi_side_px': int(roi_side),
                'grey_level': grey_level,
                'geometry_factor': geometry_factor,
                'led_intensity_rel': corrected_level,
                'gain': 1.0
            })
            if callable(progress_callback):
                progress_callback(i + 1, total)

        target = float(np.median(np.array(corrected_levels, dtype=np.float64)))
        gain_values = []
        for row in rows:
            raw_gain = target / max(1e-12, float(row['led_intensity_rel']))
            gain = float(np.clip(raw_gain, 0.05, 20.0))
            row['gain'] = gain
            gain_values.append(gain)

        gain_map = {}
        for input_path, gain in zip(input_sorted, gain_values):
            gain_map[os.path.basename(input_path).lower()] = float(gain)

        payload = {
            'version': 3,
            'label': 'led intensity compensation',
            'roi_short_side_percent': float(roi_short_side_percent),
            'grey_rotation_enabled': self._grey_rotation_enabled(),
            'grey_rotation_angle_index': int(self.combo_grey_rotation.currentIndex()),
            'grey_rotation_angle_label': self.combo_grey_rotation.currentText(),
            'dome_file': dome_path,
            'lights_distance_mm': lights_distance_mm,
            'luminance_mode': self.current_luminance_mode(),
            'weights': {
                'r': float(self.spin_weight_r.value()),
                'g': float(self.spin_weight_g.value()),
                'b': float(self.spin_weight_b.value())
            },
            'grey_files_sorted': [os.path.basename(p) for p in grey_sorted],
            'gain_values_sorted': gain_values,
            'entries': rows
        }
        return payload, gain_map

    def _parse_light_compensation_file(self, path):
        with open(path, 'r', encoding='utf-8-sig') as f:
            payload = json.load(f)

        entries = payload.get('entries', [])
        if not isinstance(entries, list) or not entries:
            raise ValueError("Compensation file has no entries")

        gain_values = []
        explicit_values = payload.get('gain_values_sorted', [])
        if isinstance(explicit_values, list) and explicit_values:
            for g in explicit_values:
                gain_values.append(float(np.clip(float(g), 0.05, 20.0)))
        else:
            # Backward compatibility: extract gains from entries order/index.
            sortable = []
            for idx, row in enumerate(entries):
                row_idx = int(row.get('index', idx + 1))
                row_gain = float(np.clip(float(row.get('gain', 1.0)), 0.05, 20.0))
                sortable.append((row_idx, idx, row_gain))
            sortable.sort(key=lambda t: (t[0], t[1]))
            gain_values = [t[2] for t in sortable]

        if not gain_values:
            raise ValueError("Compensation file has no valid gain values")
        return payload, gain_values

    def _validate_compensation_against_current_inputs(self, payload, gain_values):
        input_paths = self._sorted_paths_by_basename(self.files)
        input_names = [os.path.basename(p).lower() for p in input_paths]
        if not input_names:
            raise ValueError("No input files loaded")

        if len(gain_values) != len(input_names):
            raise ValueError(
                f"Compensation count mismatch. Input images: {len(input_names)}, compensation values: {len(gain_values)}"
            )

        gain_map = {}
        for input_path, gain in zip(input_paths, gain_values):
            gain_map[os.path.basename(input_path).lower()] = float(np.clip(float(gain), 0.05, 20.0))

        file_rotation_enabled = payload.get('grey_rotation_enabled', None)
        file_rotation_idx = payload.get('grey_rotation_angle_index', None)
        current_enabled = self._grey_rotation_enabled()
        current_idx = int(self.combo_grey_rotation.currentIndex())
        rotation_mismatch = (
            file_rotation_enabled is not None and file_rotation_idx is not None and
            (bool(file_rotation_enabled) != current_enabled or int(file_rotation_idx) != current_idx)
        )
        return gain_map, rotation_mismatch

    def add_grey_frames(self):
        if not self.files:
            QMessageBox.warning(self, "Grey Card Calibration", "Load input files before adding grey-card frames.")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Grey-Card Frames",
            "",
            self._dark_files_enabled_extensions_filter()
        )
        if not files:
            return

        valid_files = []
        for f in files:
            if is_tiff_file(f):
                try:
                    validate_tiff_16bit(f)
                except Exception as e:
                    print(f"Skipped grey TIFF: {e}")
                    continue
            valid_files.append(f)

        unique_files = []
        seen = set()
        for f in valid_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        if len(unique_files) != len(self.files):
            QMessageBox.warning(
                self,
                "Grey Card Calibration",
                f"Grey-card frame count must match input frame count.\n"
                f"Input: {len(self.files)}\nGrey: {len(unique_files)}\n\n"
                "Loading rejected."
            )
            return

        self.grey_card_files = self._sorted_paths_by_basename(unique_files)
        _, _, self.grey_to_input_pairs = self._paired_input_and_grey_paths()

        self.grey_frames_list.clear()
        for f in self.grey_card_files:
            self.grey_frames_list.addItem(os.path.basename(f))

        if self.grey_card_files:
            default_comp_path = os.path.join(os.path.dirname(self.grey_card_files[0]), "led intensity compensation.txt")
            self.edit_light_comp_file.setText(default_comp_path)

        self.update_grey_calibration_ui_state()

    def clear_grey_frames(self):
        self.grey_card_files = []
        self.grey_to_input_pairs = []
        self.grey_frames_list.clear()
        self.update_grey_calibration_ui_state()

    def browse_dome_file(self):
        selected, _ = QFileDialog.getOpenFileName(self, "Select Dome/LP File", "", "Light Files (*.dome *.lp);;JSON Files (*.json);;All Files (*)")
        if not selected:
            return
        self.dome_file_path = selected
        self.edit_dome_file.setText(selected)
        self._refresh_radius_field_state(selected)

    def _refresh_radius_field_state(self, light_path):
        if not light_path:
            self.edit_lights_distance_mm.setEnabled(True)
            self.edit_lights_distance_mm.setToolTip("Radius of the hemisphere in mm used to project directions from .dome/.lp files onto the light sphere.")
            return

        ext = os.path.splitext(light_path)[1].lower()
        if ext == '.dome':
            embedded_radius = get_embedded_dome_radius_mm(light_path)
            if embedded_radius is not None:
                self.edit_lights_distance_mm.setText(str(embedded_radius))
                self.edit_lights_distance_mm.setEnabled(False)
                self.edit_lights_distance_mm.setToolTip("Radius is taken from the .dome file metadata.")
                return

        self.edit_lights_distance_mm.setEnabled(True)
        self.edit_lights_distance_mm.setToolTip("Radius of the hemisphere in mm used to project directions from .dome/.lp files onto the light sphere.")

    def _on_light_file_path_changed(self):
        self._refresh_radius_field_state(self.edit_dome_file.text().strip())

    def on_light_comp_file_edited(self):
        path = self.edit_light_comp_file.text().strip()
        if not path:
            self.light_comp_file_path = ""
            self.light_comp_gain_map = {}
            self.light_comp_gain_values = []
            self.light_comp_metadata = {}
            self.update_grey_calibration_ui_state()
            return

        self.light_comp_file_path = path
        if not os.path.isfile(path):
            self.light_comp_gain_map = {}
            self.light_comp_gain_values = []
            self.light_comp_metadata = {}
            self.update_grey_calibration_ui_state()
            return

        try:
            payload, gain_values = self._parse_light_compensation_file(path)
            gain_map, _ = self._validate_compensation_against_current_inputs(payload, gain_values)
            self.light_comp_gain_map = gain_map
            self.light_comp_gain_values = gain_values
            self.light_comp_metadata = payload
            print(f"Loaded compensation file: {os.path.basename(path)} ({len(gain_values)} gain values)")
        except Exception as e:
            self.light_comp_gain_map = {}
            self.light_comp_gain_values = []
            self.light_comp_metadata = {}
            QMessageBox.warning(self, "Grey Card Calibration", f"Invalid compensation file: {e}")
        self.update_grey_calibration_ui_state()

    def browse_light_comp_file(self):
        selected, _ = QFileDialog.getOpenFileName(self, "Select led intensity compensation", "", "Text Files (*.txt);;All Files (*)")
        if not selected:
            return
        self.edit_light_comp_file.setText(selected)
        self.on_light_comp_file_edited()

    def on_rotation_setting_changed(self):
        self.update_grey_calibration_ui_state()

    def update_grey_calibration_ui_state(self):
        enabled = self.check_calibrate_light_variance.isChecked()

        self.edit_lights_distance_mm.setEnabled(enabled)
        self.spin_grey_roi_short_side_pct.setEnabled(enabled)
        self.btn_add_grey_frames.setEnabled(enabled)
        self.btn_clear_grey_frames.setEnabled(enabled)
        self.grey_frames_list.setEnabled(enabled)
        self.edit_dome_file.setEnabled(enabled)
        self.btn_browse_dome.setEnabled(enabled)
        self.edit_light_comp_file.setEnabled(enabled)
        self.btn_browse_light_comp.setEnabled(enabled)
        self.btn_save_light_comp.setEnabled(enabled)
        self.check_flatfield_enable.setEnabled(True)
        self.check_grey_rotation_enable.setEnabled(True)
        self.combo_grey_rotation.setEnabled(self._grey_rotation_enabled())

        flat_on = self.check_flatfield_enable.isChecked()
        self.spin_flatfield_sigma_pct.setEnabled(flat_on)
        if not enabled and not flat_on:
            self.lbl_grey_info.setText("Grey-card calibration disabled.")
            return

        flat_map, flat_err = self._build_flatfield_map_for_inputs()
        flat_status = "Flatfielding: OFF"
        if flat_on:
            if flat_err:
                flat_status = f"Flatfielding: ON (invalid setup: {flat_err})"
            else:
                flat_status = f"Flatfielding: ON ({len(flat_map)} paired grey frames)"

        comp_path = self.edit_light_comp_file.text().strip()
        has_comp = bool(comp_path) and os.path.isfile(comp_path)
        if has_comp and self.light_comp_gain_map:
            self.lbl_grey_info.setText(
                f"Compensation file ready ({len(self.light_comp_gain_map)} entries). ROI side: {self.spin_grey_roi_short_side_pct.value():.1f}% short side. "
                f"Flat sigma: {self.spin_flatfield_sigma_pct.value():.1f}%. "
                f"Grey rotation: {'ON' if self._grey_rotation_enabled() else 'OFF'}. {flat_status}"
            )
        else:
            self.lbl_grey_info.setText(
                f"Grey frames loaded: {len(self.grey_card_files)} / {len(self.files)}. "
                f"ROI side: {self.spin_grey_roi_short_side_pct.value():.1f}% short side. "
                f"Flat sigma: {self.spin_flatfield_sigma_pct.value():.1f}%. "
                f"Grey rotation: {'ON' if self._grey_rotation_enabled() else 'OFF'}. "
                f"Load .dome and save compensation file or select an existing one. {flat_status}"
            )

    def save_led_intensity_compensation(self):
        if not self.check_calibrate_light_variance.isChecked():
            QMessageBox.warning(self, "Grey Card Calibration", "Enable calibrate light intensity variance first.")
            return

        def _on_progress(done, total):
            denom = max(1, int(total))
            pct = int((int(done) / denom) * 100)
            self.progress.setValue(max(0, min(100, pct)))
            QApplication.processEvents()

        self._set_light_compensation_ui_running(True)

        try:
            payload, gain_map = self._compute_light_compensation_payload(
                progress_callback=_on_progress
            )
        except Exception as e:
            self._set_light_compensation_ui_running(False)
            QMessageBox.warning(self, "Grey Card Calibration", f"Unable to compute compensation: {e}")
            return

        default_folder = os.path.dirname(self.grey_card_files[0]) if self.grey_card_files else os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(default_folder, "led intensity compensation.txt")

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
                f.write('\n')
        except Exception as e:
            self._set_light_compensation_ui_running(False)
            QMessageBox.warning(self, "Grey Card Calibration", f"Unable to save file: {e}")
            return

        self.light_comp_file_path = path
        self.light_comp_gain_map = gain_map
        self.light_comp_gain_values = [float(row.get('gain', 1.0)) for row in payload.get('entries', [])]
        self.light_comp_metadata = payload
        self.edit_light_comp_file.setText(path)
        self.update_grey_calibration_ui_state()
        self.progress.setValue(100)
        self._set_light_compensation_ui_running(False)
        QMessageBox.information(self, "Grey Card Calibration", f"Compensation file saved:\n{path}")

    def on_dark_mode_changed(self):
        self.update_dark_calibration_ui_state()
        self.update_clip_preview()

    def update_dark_calibration_ui_state(self):
        has_dark_frames = (len(self.dark_frame_files_cap_on) > 0) or (len(self.dark_frame_files_cap_off) > 0)
        metadata_checked = self.check_apply_dark_level.isChecked()

        if has_dark_frames:
            self.check_apply_dark_level.setEnabled(False)
            self.edit_dark_lift_coeff.setEnabled(True)
            self.dark_frames_controls.setEnabled(True)
            self.lbl_dark_frames_info.setText(
                f"Cap-on: {len(self.dark_frame_files_cap_on)} | Cap-off: {len(self.dark_frame_files_cap_off)}. "
                "RAW metadata dark calibration disabled."
            )
        else:
            self.check_apply_dark_level.setEnabled(True)
            self.edit_dark_lift_coeff.setEnabled(True)
            self.dark_frames_controls.setEnabled(not metadata_checked)
            if metadata_checked:
                self.lbl_dark_frames_info.setText("Manual dark frames disabled while RAW metadata calibration is active.")
            else:
                self.lbl_dark_frames_info.setText("No dark frames loaded.")

    def _preview_compute_frame_mean(self, paths, downsample=4):
        """Compute average value across multiple dark frames.
        Args:
            paths: list of dark frame file paths
            downsample: factor to downsample RAW for speed (default 4, recommended for preview)
        Returns:
            Average normalized value (0.0-1.0) or 0.0 if no frames
        """
        if not paths:
            return 0.0

        demosaic = self.demosaic_map[self.combo_demosaic.currentText()]
        cache_key = (tuple(paths), str(demosaic), int(downsample))
        
        # Check dictionary cache based on file paths and demosaic
        cache_dict = self.dark_frame_mean_cache_cap_on if paths == self.dark_frame_files_cap_on else self.dark_frame_mean_cache_cap_off
        if cache_key in cache_dict:
            return cache_dict[cache_key]

        per_frame_means = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            if ext in {'tif', 'tiff'}:
                try:
                    img = validate_tiff_16bit(path)
                    if downsample > 1:
                        img = img[::downsample, ::downsample]
                except Exception:
                    continue
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                per_frame_means.append(float(np.mean(img.astype(np.float32) / 65535.0)))
            else:
                try:
                    with rawpy.imread(path) as raw:
                        # Use half_size if downsample >= 2 for efficiency
                        half_size = downsample >= 2
                        rgb = raw.postprocess(
                            half_size=half_size,
                            demosaic_algorithm=demosaic,
                            output_bps=16,
                            gamma=(1, 1),
                            no_auto_bright=True,
                            no_auto_scale=True,
                            use_camera_wb=False,
                            user_black=0,
                            user_wb=[1.0, 1.0, 1.0, 1.0]
                        ).astype(np.float32)
                        if downsample > 2:
                            rgb = rgb[::max(1, downsample // 2), ::max(1, downsample // 2)]
                    per_frame_means.append(float(np.mean(rgb) / 65535.0))
                except Exception:
                    continue

        if per_frame_means:
            offset = float(np.mean(per_frame_means))
        else:
            offset = 0.0

        # Store in persistent cache
        cache_dict[cache_key] = offset
        # Keep legacy cache for backward compatibility
        self.dark_signal_cache_key = cache_key
        self.dark_signal_cache_value = offset
        return offset

    def on_use_sharpness_changed(self):
        enabled = self.check_use_sharpness.isChecked()
        self.lbl_sharp_amount.setEnabled(enabled)
        self.lbl_sharp_radius.setEnabled(enabled)
        self.spin_sharp_amount.setEnabled(enabled)
        self.spin_sharp_radius.setEnabled(enabled)
        self.update_clip_preview()

    def current_luminance_mode(self):
        label = self.combo_luminance_source.currentText()
        if label.startswith("RAW Bayer 2x2"):
            return 'raw_bayer_2x2'
        if label.startswith("Demosaic RGB weighted"):
            return 'demosaic_weighted'
        return 'demosaic_mean'

    def _grey_luminance_params(self):
        mode = self.current_luminance_mode()
        return (
            mode == 'demosaic_weighted',
            float(self.spin_weight_r.value()),
            float(self.spin_weight_g.value()),
            float(self.spin_weight_b.value())
        )

    def _compute_grey_luminance_from_rgb(self, rgb):
        rgb = rgb.astype(np.float32)
        use_weighted, weight_r, weight_g, weight_b = self._grey_luminance_params()
        if use_weighted:
            denom = max(1e-8, float(weight_r + weight_g + weight_b))
            y = (weight_r * rgb[:, :, 0] + weight_g * rgb[:, :, 1] + weight_b * rgb[:, :, 2]) / denom
        else:
            y = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) / 3.0
        return np.clip(y / 65535.0, 0.0, None)

    def on_luminance_mode_changed(self):
        mode = self.current_luminance_mode()
        uses_demosaic = mode != 'raw_bayer_2x2'
        weighted = mode == 'demosaic_weighted'

        self.lbl_demosaic_label.setEnabled(uses_demosaic)
        self.combo_demosaic.setEnabled(uses_demosaic)

        self.weighted_controls_widget.setVisible(weighted)
        self.update_clip_preview()

    def update_bit_depth_controls(self, out_format):
        supports_16 = out_format in ['TIFF', 'PNG']
        self.combo_bit_depth.setEnabled(supports_16)
        if not supports_16:
            self.combo_bit_depth.setCurrentText("8-bit")
        self.sync_default_output_folder()

    def default_output_folder_path(self):
        if self.files:
            base_dir = os.path.dirname(self.files[0])
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, f"Luminance_Export_{self.combo_format.currentText()}")

    def sync_default_output_folder(self, force=False):
        if self.edit_output_folder is None:
            return
        new_default = self.default_output_folder_path()
        current = self.edit_output_folder.text().strip()
        if force or (not current) or (current == self.last_auto_output_path):
            self.edit_output_folder.setText(new_default)
            self.last_auto_output_path = new_default

    def browse_output_folder(self):
        current = self.edit_output_folder.text().strip() or self.default_output_folder_path()
        selected = QFileDialog.getExistingDirectory(self, "Select Output Folder", current)
        if selected:
            self.edit_output_folder.setText(selected)

    def open_selected_output_folder(self):
        folder = self.edit_output_folder.text().strip()
        if not folder:
            folder = self.default_output_folder_path()
            self.edit_output_folder.setText(folder)
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        try:
            if sys.platform == 'win32':
                os.startfile(folder)
            else:
                QMessageBox.information(self, "Output Folder", folder)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Unable to open output folder: {e}")

    def on_percentile_changed(self, value):
        percentile_value = self._percentile_slider_to_float(value)
        self.lbl_percentile_value.setText(self._format_percentile_text(percentile_value))
        self.edit_percentile.blockSignals(True)
        self.edit_percentile.setValue(percentile_value)
        self.edit_percentile.blockSignals(False)
        self.update_clip_preview()

    def on_stretch_threshold_ready(self, threshold_pct):
        self.slider_percentile.setValue(self._percentile_float_to_slider(threshold_pct))

    def on_percentile_spin_changed(self, value):
        slider_value = self._percentile_float_to_slider(value)
        if self.slider_percentile.value() != slider_value:
            self.slider_percentile.setValue(slider_value)
            return
        self.lbl_percentile_value.setText(self._format_percentile_text(value))
        self.update_clip_preview()

    def on_grey_rotation_setting_changed(self):
        self.combo_grey_rotation.setEnabled(self._grey_rotation_enabled())
        self.update_grey_calibration_ui_state()
        self.update_clip_preview()

    def on_preview_index_edited(self):
        if not self.files:
            self.edit_preview_index.setText("1")
            return
        text = self.edit_preview_index.text().strip()
        try:
            idx = int(text)
        except ValueError:
            idx = self.preview_index + 1
        idx = max(1, min(len(self.files), idx))
        self.preview_index = idx - 1
        self.list_widget.setCurrentRow(self.preview_index)
        self.update_clip_preview()

    def _apply_preview_color_space(self, y_norm):
        color_space = self.combo_color.currentText()
        if color_space == 'Linear':
            return y_norm
        if color_space in ['sRGB', 'Display P3']:
            return np.where(y_norm <= 0.0031308, 12.92 * y_norm, 1.055 * np.power(y_norm, 1 / 2.4) - 0.055)
        if color_space == 'ProPhoto RGB':
            return np.power(y_norm, 1 / 1.8)
        return y_norm

    def _preview_luminance_from_rgb(self, rgb_norm):
        use_weighted = self.current_luminance_mode() == 'demosaic_weighted'
        if use_weighted:
            wr = float(self.spin_weight_r.value())
            wg = float(self.spin_weight_g.value())
            wb = float(self.spin_weight_b.value())
            denom = max(1e-8, wr + wg + wb)
            return (wr * rgb_norm[:, :, 0] + wg * rgb_norm[:, :, 1] + wb * rgb_norm[:, :, 2]) / denom
        return (rgb_norm[:, :, 0] + rgb_norm[:, :, 1] + rgb_norm[:, :, 2]) / 3.0

    def _get_raw_dark_level(self, raw):
        levels = [float(v) for v in list(raw.black_level_per_channel) if float(v) > 0]
        if not levels:
            return None
        return float(sum(levels) / len(levels))

    def apply_unsharp_mask(self, image, radius, amount):
        if amount <= 0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        sharpened = image + amount * (image - blurred)
        return np.clip(sharpened, 0.0, 1.0)

    def _gray_map_to_pixmap(self, y_map, width=178, height=128):
        if y_map is None or y_map.size == 0:
            return QPixmap()
        y = np.clip(self._apply_rotation_to_map(y_map).astype(np.float32), 0.0, None)
        peak = float(np.percentile(y, 99.0)) if y.size > 0 else 1.0
        if peak <= 0:
            peak = 1.0
        show = np.clip(y / peak, 0.0, 1.0)
        img8 = (show * 255.0).astype(np.uint8)
        rgb = np.ascontiguousarray(np.stack((img8, img8, img8), axis=-1))
        h, w, c = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, w * c, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        return pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def update_flatfield_preview_panel(self):
        if not self.files:
            self.grey_flat_before_label.setPixmap(QPixmap())
            self.grey_flat_before_label.setText("No preview")
            self.grey_flat_after_label.setPixmap(QPixmap())
            self.grey_flat_after_label.setText("No preview")
            self.grey_flat_frame_label.setPixmap(QPixmap())
            self.grey_flat_frame_label.setText("No preview")
            self.lbl_flatfield_preview_info.setText("")
            return

        filepath = self.files[self.preview_index]
        dark_ctx = self._build_grey_dark_context()
        try:
            before = self._preview_norm_map(filepath, apply_flatfield=False)
            if self.check_flatfield_enable.isChecked():
                after = self._preview_apply_flatfield(filepath, before.copy(), dark_ctx=dark_ctx)
            else:
                after = before

            flat_map, flat_err = self._build_flatfield_map_for_inputs()
            flat_y = None
            if flat_err is None:
                grey_path = flat_map.get(os.path.basename(filepath).lower())
                if grey_path and os.path.isfile(grey_path):
                    flat_y = self._preview_grey_luma_norm_map(grey_path, dark_ctx=dark_ctx, target_shape=before.shape[:2])

            self.grey_flat_before_label.setPixmap(self._gray_map_to_pixmap(before))
            self.grey_flat_before_label.setText("")
            self.grey_flat_after_label.setPixmap(self._gray_map_to_pixmap(after))
            self.grey_flat_after_label.setText("")

            if flat_y is not None:
                self.grey_flat_frame_label.setPixmap(self._gray_map_to_pixmap(flat_y))
                self.grey_flat_frame_label.setText("")
            else:
                self.grey_flat_frame_label.setPixmap(QPixmap())
                self.grey_flat_frame_label.setText("No flat")

            pair_txt = os.path.basename(filepath)
            if flat_err:
                self.lbl_flatfield_preview_info.setText(f"{pair_txt} | {flat_err}")
            else:
                self.lbl_flatfield_preview_info.setText(pair_txt)
        except Exception as e:
            self.grey_flat_before_label.setPixmap(QPixmap())
            self.grey_flat_before_label.setText("Preview\nerror")
            self.grey_flat_after_label.setPixmap(QPixmap())
            self.grey_flat_after_label.setText("Preview\nerror")
            self.grey_flat_frame_label.setPixmap(QPixmap())
            self.grey_flat_frame_label.setText("Preview\nerror")
            self.lbl_flatfield_preview_info.setText(f"Flatfield preview error: {e}")

    def _get_cached_master_dark_map(self, cap_on_paths, demosaic, luminance_mode):
        """Get cached dark map or compute and cache it. Avoids repeated computation."""
        if not cap_on_paths:
            return None
        
        cache_key = (tuple(cap_on_paths), str(demosaic), luminance_mode)
        if luminance_mode == 'raw_bayer_2x2':
            if cache_key in self.dark_bayer_map_cache_cap_on:
                return self.dark_bayer_map_cache_cap_on[cache_key]
            # Compute and cache
            worker = ProcessThread([], {})
            dark_map = worker._compute_master_dark_bayer_map(cap_on_paths)
            self.dark_bayer_map_cache_cap_on[cache_key] = dark_map
            return dark_map
        else:
            if cache_key in self.dark_map_cache_cap_on:
                return self.dark_map_cache_cap_on[cache_key]
            # Compute and cache
            worker = ProcessThread([], {})
            dark_map = worker._compute_master_dark_map(cap_on_paths, demosaic)
            self.dark_map_cache_cap_on[cache_key] = dark_map
            return dark_map

    def _get_cached_ambient_dark_offset(self, cap_off_paths, demosaic, dark_map, luminance_mode):
        """Get cached ambient dark offset or compute and cache it."""
        if not cap_off_paths:
            return 0.0
        
        cache_key = (tuple(cap_off_paths), str(demosaic), luminance_mode)
        if luminance_mode == 'raw_bayer_2x2':
            if cache_key in self.dark_map_cache_cap_off:
                return self.dark_map_cache_cap_off[cache_key]
            # Compute and cache
            worker = ProcessThread([], {})
            offset = worker._compute_ambient_dark_offset_bayer(cap_off_paths, dark_map)
            self.dark_map_cache_cap_off[cache_key] = offset
            return offset
        else:
            if cache_key in self.dark_map_cache_cap_off:
                return self.dark_map_cache_cap_off[cache_key]
            # Compute and cache
            worker = ProcessThread([], {})
            offset = worker._compute_ambient_dark_offset(cap_off_paths, demosaic, dark_map)
            self.dark_map_cache_cap_off[cache_key] = offset
            return offset

    def _preview_norm_map(self, filepath, apply_flatfield=True):
        ext = filepath.lower().split('.')[-1]
        raw_exts = {'cr2', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2', 'raw'}
        mode = self.current_luminance_mode()
        demosaic = self.demosaic_map[self.combo_demosaic.currentText()]
        sharp_amount = self.spin_sharp_amount.value() if self.check_use_sharpness.isChecked() else 0.0
        sharp_radius = self.spin_sharp_radius.value()
        downsample = max(1, self.spin_undersample.value())
        has_manual_dark = (len(self.dark_frame_files_cap_on) > 0) or (len(self.dark_frame_files_cap_off) > 0)
        apply_dark_level = self.check_apply_dark_level.isChecked() and (not has_manual_dark)

        cap_on_offset = self._preview_compute_frame_mean(self.dark_frame_files_cap_on) if self.dark_frame_files_cap_on else 0.0
        cap_off_raw = self._preview_compute_frame_mean(self.dark_frame_files_cap_off) if self.dark_frame_files_cap_off else 0.0
        dark_frame_offset = max(0.0, cap_off_raw - cap_on_offset)
        dark_ctx = {
            'apply_dark_level': apply_dark_level,
            'cap_on_offset': cap_on_offset,
            'dark_frame_offset': dark_frame_offset,
            'scale': self._manual_dark_scale()
        }

        if ext in raw_exts:
            if mode == 'raw_bayer_2x2':
                with rawpy.imread(filepath) as raw:
                    raw_vis = raw.raw_image_visible.astype(np.float32)
                    if apply_dark_level:
                        dark_levels = [float(v) for v in list(raw.black_level_per_channel)]
                        colors = raw.raw_colors_visible
                        for idx, level in enumerate(dark_levels):
                            raw_vis[colors == idx] -= self._effective_dark_level(level)

                    if cap_on_offset > 0:
                        raw_vis = np.clip(raw_vis - (cap_on_offset * 65535.0 * self._manual_dark_scale()), 0.0, None)
                    if dark_frame_offset > 0:
                        raw_vis = np.clip(raw_vis - (dark_frame_offset * 65535.0 * self._manual_dark_scale()), 0.0, None)

                    h, w = raw_vis.shape[:2]
                    h2 = (h // 2) * 2
                    w2 = (w // 2) * 2
                    if h2 == 0 or w2 == 0:
                        raise ValueError("RAW frame too small for Bayer 2x2 preview")

                    src = raw_vis[:h2, :w2]
                    y_65535 = (src[0::2, 0::2] + src[0::2, 1::2] + src[1::2, 0::2] + src[1::2, 1::2]) / 4.0
                    y = np.clip(y_65535 / 65535.0, 0.0, None)
                    y = self.apply_unsharp_mask(y, sharp_radius, sharp_amount)
                    if downsample > 1:
                        y = y[::downsample, ::downsample]
                    y = self._apply_linearity_lut_if_enabled(y)
                    if apply_flatfield:
                        return self._preview_apply_flatfield(filepath, y, dark_ctx=dark_ctx)
                    return y

            half_size = downsample > 1 and downsample % 2 == 0
            post_ds = max(1, downsample // 2) if half_size else downsample
            with rawpy.imread(filepath) as raw:
                kwargs = dict(
                    half_size=half_size,
                    demosaic_algorithm=demosaic,
                    output_bps=16,
                    gamma=(1, 1),
                    no_auto_bright=True,
                    no_auto_scale=True,
                    use_camera_wb=False,
                    user_wb=[1.0, 1.0, 1.0, 1.0]
                )
                if apply_dark_level:
                    dark_level = self._get_raw_dark_level(raw)
                    if dark_level is not None:
                        kwargs['user_black'] = self._effective_dark_level(dark_level)
                else:
                    kwargs['user_black'] = 0

                rgb = raw.postprocess(**kwargs).astype(np.float32)
                if cap_on_offset > 0:
                    rgb = np.clip(rgb - (cap_on_offset * 65535.0 * self._manual_dark_scale()), 0.0, None)
                if dark_frame_offset > 0:
                    rgb = np.clip(rgb - (dark_frame_offset * 65535.0 * self._manual_dark_scale()), 0.0, None)
                rgb_norm = rgb / 65535.0
                y = np.clip(self._preview_luminance_from_rgb(rgb_norm), 0.0, None)
                y = self.apply_unsharp_mask(y, sharp_radius, sharp_amount)
                if post_ds > 1:
                    y = y[::post_ds, ::post_ds]
                y = self._apply_linearity_lut_if_enabled(y)
                if apply_flatfield:
                    return self._preview_apply_flatfield(filepath, y, dark_ctx=dark_ctx)
                return y

        if is_tiff_file(filepath):
            img = validate_tiff_16bit(filepath)
        else:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Unable to read {filepath}")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb = img.astype(np.float32) / 65535.0

        y = self._preview_luminance_from_rgb(rgb)
        y = np.clip(y, 0.0, None)
        y = self.apply_unsharp_mask(y, sharp_radius, sharp_amount)
        if downsample > 1:
            y = y[::downsample, ::downsample]
        y = self._apply_linearity_lut_if_enabled(y)
        if apply_flatfield:
            return self._preview_apply_flatfield(filepath, y, dark_ctx=dark_ctx)
        return y

    def on_list_selection_changed(self, row):
        if row >= 0 and row < len(self.files):
            self.preview_index = row
        self.update_clip_preview()

    def prev_preview_image(self):
        if not self.files:
            return
        self.preview_index = (self.preview_index - 1) % len(self.files)
        self.list_widget.setCurrentRow(self.preview_index)
        self.update_clip_preview()

    def next_preview_image(self):
        if not self.files:
            return
        self.preview_index = (self.preview_index + 1) % len(self.files)
        self.list_widget.setCurrentRow(self.preview_index)
        self.update_clip_preview()

    def update_clip_preview(self, *_):
        if not self.files:
            self.clip_preview_label.setPixmap(QPixmap())
            self.clip_preview_label.setText("No preview")
            self.lbl_preview_file.setText("Image: -/-")
            self.edit_preview_index.setText("1")
            self.lbl_preview_stats.setText("Clipped pixels: -")
            self.btn_export_current.setEnabled(False)
            self.update_flatfield_preview_panel()
            return

        preview_mode = self.combo_preview_mode.currentText()
        is_stack = preview_mode in ("Stack MAX", "Stack MIN")
        self.nav_widget.setVisible(not is_stack)
        self.btn_export_current.setEnabled(not is_stack)

        if is_stack:
            self._update_clip_preview_stack(preview_mode)
            return

        self.preview_index = max(0, min(self.preview_index, len(self.files) - 1))
        filepath = self.files[self.preview_index]
        self.btn_export_current.setEnabled(True)
        self.lbl_preview_file.setText(f"Image: {self.preview_index + 1}/{len(self.files)}")
        self.edit_preview_index.setText(str(self.preview_index + 1))

        try:
            y = self._preview_norm_map(filepath)
            t = self._current_percentile_value() / 100.0

            sub = y
            clipped_high = sub > t
            clipped_black = sub <= 1e-8

            # Stretch preview with the same logic used for phase-1 peak exclusion.
            valid = sub[~clipped_high]
            preview_peak = float(valid.max()) if valid.size > 0 else float(sub.max())
            if preview_peak <= 0:
                preview_peak = 1.0
            stretched = np.clip(sub / preview_peak, 0.0, 1.0)

            # Show preview in selected output color space (gamma/transfer included).
            shown = np.clip(self._apply_preview_color_space(stretched), 0.0, 1.0)

            gray = shown
            gray8 = (gray * 255).astype(np.uint8)
            rgb = np.ascontiguousarray(np.stack((gray8, gray8, gray8), axis=-1))
            rgb[clipped_black] = [0, 0, 255]
            rgb[clipped_high] = [255, 0, 0]

            name = os.path.basename(filepath)
            y_text = max(14, rgb.shape[0] - 8)
            cv2.putText(rgb, name, (6, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(rgb, name, (6, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)

            high_clip_ratio = (float(np.count_nonzero(clipped_high)) / float(clipped_high.size) * 100.0) if clipped_high.size > 0 else 0.0
            black_clip_ratio = (float(np.count_nonzero(clipped_black)) / float(clipped_black.size) * 100.0) if clipped_black.size > 0 else 0.0
            flat_state = "ON" if self.check_flatfield_enable.isChecked() else "OFF"
            self.lbl_preview_stats.setText(
                f"Flatfielding: {flat_state} | Clipped pixels - High (red): {high_clip_ratio:.2f}% | Black (blue): {black_clip_ratio:.2f}%"
            )

            h, w, c = rgb.shape
            qimg = QImage(rgb.tobytes(), w, h, w * c, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(
                self.clip_preview_label.width() - 2,
                self.clip_preview_label.height() - 2,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.clip_preview_label.setPixmap(pixmap)
            self.clip_preview_label.setText("")
            self.update_flatfield_preview_panel()
        except Exception as e:
            self.clip_preview_label.setPixmap(QPixmap())
            self.clip_preview_label.setText("Preview\nerror")
            self.lbl_preview_stats.setText("Clipped pixels: -")
            self.update_flatfield_preview_panel()
            print(f"Clip preview error: {e}")

    def _build_stack_cache_key(self, mode):
        """Return a tuple that identifies all parameters affecting the raw stack pixels."""
        return (
            mode,
            tuple(self.files),
            self.spin_undersample.value(),
            self.current_luminance_mode(),
            self.combo_demosaic.currentText(),
            self.spin_sharp_amount.value() if self.check_use_sharpness.isChecked() else 0.0,
            self.spin_sharp_radius.value(),
            self.check_apply_dark_level.isChecked(),
            tuple(sorted(self.dark_frame_files_cap_on)),
            tuple(sorted(self.dark_frame_files_cap_off)),
            self.check_flatfield_enable.isChecked(),
            tuple(sorted(self.grey_card_files)),
            self.check_linearity_enable.isChecked(),
        )

    def _update_clip_preview_stack(self, mode):
        """Display pixel-wise MAX or MIN stack with clipping overlay; raw stack is cached."""
        use_max = mode == "Stack MAX"
        self.lbl_preview_file.setText(f"Stack {mode.split()[1]}: {len(self.files)} images")
        self.edit_preview_index.setText("-")

        try:
            key = self._build_stack_cache_key(mode)
            if self._stack_cache is None or self._stack_cache_key_val != key:
                self.clip_preview_label.setPixmap(QPixmap())
                self.clip_preview_label.setText("Computing stack…")
                QApplication.processEvents()

                stack = None
                for filepath in self.files:
                    try:
                        y = self._preview_norm_map(filepath)
                    except Exception as e:
                        print(f"Stack preview: skipping {filepath}: {e}")
                        continue
                    if stack is None:
                        stack = y.copy()
                    else:
                        if y.shape != stack.shape:
                            y = cv2.resize(y, (stack.shape[1], stack.shape[0]), interpolation=cv2.INTER_AREA)
                        if use_max:
                            np.maximum(stack, y, out=stack)
                        else:
                            np.minimum(stack, y, out=stack)

                if stack is None:
                    self.clip_preview_label.setText("Stack error:\nno images loaded")
                    self.lbl_preview_stats.setText("Clipped pixels: -")
                    return

                self._stack_cache = stack
                self._stack_cache_key_val = key
            else:
                stack = self._stack_cache

            # Apply stretch and clipping overlay (fast — no file I/O)
            t = self._current_percentile_value() / 100.0
            clipped_high = stack > t
            clipped_black = stack <= 1e-8

            valid = stack[~clipped_high]
            preview_peak = float(valid.max()) if valid.size > 0 else float(stack.max())
            if preview_peak <= 0:
                preview_peak = 1.0
            stretched = np.clip(stack / preview_peak, 0.0, 1.0)
            shown = np.clip(self._apply_preview_color_space(stretched), 0.0, 1.0)

            gray8 = (shown * 255).astype(np.uint8)
            rgb = np.ascontiguousarray(np.stack((gray8, gray8, gray8), axis=-1))
            rgb[clipped_black] = [0, 0, 255]
            rgb[clipped_high] = [255, 0, 0]

            label = f"{mode} ({len(self.files)} imgs)"
            y_text = max(14, rgb.shape[0] - 8)
            cv2.putText(rgb, label, (6, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(rgb, label, (6, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)

            high_clip_ratio = (float(np.count_nonzero(clipped_high)) / float(clipped_high.size) * 100.0) if clipped_high.size > 0 else 0.0
            black_clip_ratio = (float(np.count_nonzero(clipped_black)) / float(clipped_black.size) * 100.0) if clipped_black.size > 0 else 0.0
            flat_state = "ON" if self.check_flatfield_enable.isChecked() else "OFF"
            self.lbl_preview_stats.setText(
                f"Flatfielding: {flat_state} | Clipped pixels - High (red): {high_clip_ratio:.2f}% | Black (blue): {black_clip_ratio:.2f}%"
            )

            h, w, c = rgb.shape
            qimg = QImage(rgb.tobytes(), w, h, w * c, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(
                self.clip_preview_label.width() - 2,
                self.clip_preview_label.height() - 2,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.clip_preview_label.setPixmap(pixmap)
            self.clip_preview_label.setText("")
            self.update_flatfield_preview_panel()
        except Exception as e:
            self.clip_preview_label.setPixmap(QPixmap())
            self.clip_preview_label.setText("Stack\nerror")
            self.lbl_preview_stats.setText("Clipped pixels: -")
            self.update_flatfield_preview_panel()
            print(f"Stack preview error: {e}")

    def update_rotation_preview(self):
        if not self.files:
            self.rot_preview_label.setPixmap(QPixmap())
            self.rot_preview_label.setText("No preview")
            return
        filepath = self.files[0]
        ext = filepath.lower().split('.')[-1]
        raw_exts = {'cr2', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2', 'raw'}
        try:
            if ext in raw_exts:
                try:
                    with rawpy.imread(filepath) as raw:
                        thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        nparr = np.frombuffer(thumb.data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = np.array(thumb.data)
                except Exception:
                    with rawpy.imread(filepath) as raw:
                        img = raw.postprocess(half_size=True, output_bps=8)
            else:
                img = cv2.imread(filepath)
                if img is None:
                    self.rot_preview_label.setText("Preview error")
                    return
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.check_rotation_enable.isChecked():
                angle_idx = self.combo_rotation.currentIndex()
                if angle_idx == 0:
                    img = np.rot90(img, k=3)   # 90° CW
                elif angle_idx == 1:
                    img = np.rot90(img, k=2)   # 180°
                else:
                    img = np.rot90(img, k=1)   # 270° CW

            img = np.ascontiguousarray(img)
            h, w, c = img.shape
            qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(178, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.rot_preview_label.setPixmap(pixmap)
            self.rot_preview_label.setText("")
        except Exception as e:
            self.rot_preview_label.setText("Preview\nerror")
            print(f"Preview error: {e}")

    def set_button_develop(self):
        self.btn_run.setText("Develop Luminance (Step 2)")
        self.btn_run.setStyleSheet("background-color: #5b7da6; color: white; font-weight: bold;")

    def set_button_stop(self):
        self.btn_run.setText("Stop")
        self.btn_run.setStyleSheet("background-color: #a65b5b; color: white; font-weight: bold;")

    def append_to_console(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select RAW/TIFF", "", "Images (*.cr2 *.nef *.arw *.dng *.tif *.tiff)")
        added = False
        for f in files:
            if is_tiff_file(f):
                try:
                    validate_tiff_16bit(f)
                except Exception as e:
                    print(f"Skipped TIFF: {e}")
                    continue
            if f not in self.files:
                self.files.append(f)
                self.list_widget.addItem(os.path.basename(f))
                print(f"Added: {os.path.basename(f)}")
                added = True
        if self.files and self.list_widget.currentRow() < 0:
            self.preview_index = 0
            self.list_widget.setCurrentRow(0)

        if self.grey_card_files and len(self.grey_card_files) != len(self.files):
            self.clear_grey_frames()
            QMessageBox.information(
                self,
                "Grey Card Calibration",
                "Input list changed: grey-card list was cleared because frame counts no longer match."
            )

        if added:
            self.sync_default_output_folder(force=False)
            self.btn_export_current.setEnabled(True)
        self.update_clip_preview()
        self.update_rotation_preview()
        self.update_grey_calibration_ui_state()

    def clear_files(self):
        self.files.clear()
        self.preview_index = 0
        self.list_widget.clear()
        self.clear_grey_frames()
        self.clip_preview_label.setPixmap(QPixmap())
        self.clip_preview_label.setText("No preview")
        self.lbl_preview_file.setText("Image: -/-")
        self.edit_preview_index.setText("1")
        self.lbl_preview_stats.setText("Clipped pixels: -")
        self.rot_preview_label.setPixmap(QPixmap())
        self.rot_preview_label.setText("No preview")
        self.btn_export_current.setEnabled(False)
        self.sync_default_output_folder(force=True)
        self.update_grey_calibration_ui_state()
        print("File list cleared.")

    def _rotate_output(self, img, angle_idx):
        if angle_idx == 0:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle_idx == 1:
            return cv2.rotate(img, cv2.ROTATE_180)
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _fullres_norm_map(self, filepath, apply_flatfield=True, processing_downsample_factor=1):
        options = self._build_base_options()
        worker = ProcessThread([], options)

        demosaic = options['demosaic']
        sharp_amount = float(options.get('sharp_amount', 0.0))
        sharp_radius = float(options.get('sharp_radius', 1.0))
        apply_dark_level = bool(options.get('apply_dark_level', True))
        dark_frame_paths_cap_on = list(options.get('dark_frame_paths_cap_on', []))
        dark_frame_paths_cap_off = list(options.get('dark_frame_paths_cap_off', []))
        luminance_mode = options.get('luminance_mode', 'raw_bayer_2x2')
        use_weighted_luminance = bool(options.get('use_weighted_luminance', False))
        weight_r = float(options.get('weight_r', 1.0))
        weight_g = float(options.get('weight_g', 2.0))
        weight_b = float(options.get('weight_b', 1.0))
        dark_lift_coeff = float(np.clip(options.get('dark_lift_coeff', 0.0), -0.99, 0.99))

        dark_frame_offset = 0.0
        dark_map = None
        dark_bayer_map = None
        if dark_frame_paths_cap_on or dark_frame_paths_cap_off:
            apply_dark_level = False
            if luminance_mode == 'raw_bayer_2x2':
                if dark_frame_paths_cap_on:
                    dark_bayer_map = self._get_cached_master_dark_map(dark_frame_paths_cap_on, demosaic, 'raw_bayer_2x2')
                if dark_frame_paths_cap_off:
                    dark_frame_offset = self._get_cached_ambient_dark_offset(dark_frame_paths_cap_off, demosaic, dark_bayer_map, 'raw_bayer_2x2')
            else:
                if dark_frame_paths_cap_on:
                    dark_map = self._get_cached_master_dark_map(dark_frame_paths_cap_on, demosaic, 'demosaic_rgb')
                if dark_frame_paths_cap_off:
                    dark_frame_offset = self._get_cached_ambient_dark_offset(dark_frame_paths_cap_off, demosaic, dark_map, 'demosaic_rgb')

        ext = filepath.lower().split('.')[-1]
        if luminance_mode == 'raw_bayer_2x2':
            y_norm, _dark_used = worker.extract_sensor_bayer2x2_map(
                filepath=filepath,
                sharp_radius=sharp_radius,
                sharp_amount=0.0,
                downsample_factor=max(1, int(processing_downsample_factor)),
                apply_dark_level=apply_dark_level,
                dark_frame_offset=dark_frame_offset,
                dark_bayer_map=dark_bayer_map,
                dark_lift_coeff=dark_lift_coeff,
            )
        else:
            if ext in ['tif', 'tiff']:
                img_linear = worker.load_and_linearize(
                    filepath,
                    demosaic=demosaic,
                    apply_dark_level=apply_dark_level,
                    dark_frame_offset=dark_frame_offset,
                    dark_map=dark_map,
                    dark_lift_coeff=dark_lift_coeff,
                )
                y_norm = worker._compute_luminance_from_rgb(img_linear, use_weighted_luminance, weight_r, weight_g, weight_b)
                if processing_downsample_factor > 1:
                    y_norm = y_norm[::processing_downsample_factor, ::processing_downsample_factor]
            else:
                y_norm, _dark_used = worker.extract_sensor_luminance_map(
                    filepath=filepath,
                    demosaic=demosaic,
                    sharp_radius=sharp_radius,
                    sharp_amount=0.0,
                    downsample_factor=max(1, int(processing_downsample_factor)),
                    apply_dark_level=apply_dark_level,
                    use_weighted=use_weighted_luminance,
                    weight_r=weight_r,
                    weight_g=weight_g,
                    weight_b=weight_b,
                    dark_frame_offset=dark_frame_offset,
                    dark_map=dark_map,
                    dark_lift_coeff=dark_lift_coeff,
                )

        y_norm = worker.apply_unsharp_mask(y_norm, sharp_radius, sharp_amount)
        y_norm = self._apply_linearity_lut_if_enabled(y_norm)
        if apply_flatfield and self.check_flatfield_enable.isChecked():
            flat_map, flat_err = self._build_flatfield_map_for_inputs()
            if flat_err is None:
                y_norm = worker._apply_flatfield_to_luma_map(
                    input_path=filepath,
                    y_norm=y_norm,
                    flat_lookup={k: str(v) for k, v in flat_map.items()},
                    demosaic=demosaic,
                    apply_dark_level=apply_dark_level,
                    dark_frame_offset=dark_frame_offset,
                    dark_map=dark_map,
                    dark_bayer_map=dark_bayer_map,
                    dark_lift_coeff=dark_lift_coeff,
                )
        return y_norm

    def export_current_preview(self):
        if not self.files:
            QMessageBox.warning(self, "Export Current Frame", "Load at least one input file first.")
            return

        out_folder = self.edit_output_folder.text().strip() or self.default_output_folder_path()
        os.makedirs(out_folder, exist_ok=True)

        filepath = self.files[self.preview_index]
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_format = self.combo_format.currentText().upper()
        if out_format == 'JPG':
            ext = 'jpg'
        elif out_format == 'PNG':
            ext = 'png'
        elif out_format == 'TIFF':
            ext = 'tiff'
        else:
            raise ValueError(f"Unsupported output format: {out_format}")
        out_path = os.path.join(out_folder, f"{base}_lum_preview.{ext}")

        try:
            output_downsample_factor = max(1, int(self.spin_output_downsample.value()))
            process_at_output_scale = self.check_process_at_output_scale.isChecked()
            processing_downsample_factor = output_downsample_factor if (process_at_output_scale and output_downsample_factor > 1) else 1

            y = self._fullres_norm_map(
                filepath,
                apply_flatfield=True,
                processing_downsample_factor=processing_downsample_factor
            )
            t = self._current_percentile_value() / 100.0
            valid = y[y <= t]
            preview_peak = float(valid.max()) if valid.size > 0 else float(y.max())
            if preview_peak <= 0:
                preview_peak = 1.0

            stretched = np.clip(y / preview_peak, 0.0, 1.0)
            shown = np.clip(self._apply_preview_color_space(stretched), 0.0, 1.0)
            rgb = np.stack((shown, shown, shown), axis=-1)

            bit_depth = 16 if self.combo_bit_depth.currentText() == '16-bit' else 8
            if out_format in ['TIFF', 'PNG'] and bit_depth == 16:
                out_img = (rgb * 65535.0).astype(np.uint16)
            else:
                out_img = (rgb * 255.0).astype(np.uint8)

            if self.check_rotation_enable.isChecked():
                out_img = self._rotate_output(out_img, self.combo_rotation.currentIndex())

            if output_downsample_factor > 1 and processing_downsample_factor == 1:
                h, w = out_img.shape[:2]
                out_w = max(1, w // output_downsample_factor)
                out_h = max(1, h // output_downsample_factor)
                out_img = cv2.resize(out_img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            saver = ProcessThread([], {})
            saver.save_output_image(out_path, out_img, out_format)

            self.last_output_folder = out_folder
            print(f"Exported current preview: {out_path}")
            QMessageBox.information(self, "Export Current Frame", f"Saved:\n{out_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Current Frame", f"Export failed: {e}")

    def _build_base_options(self):
        use_dark_frames = (len(self.dark_frame_files_cap_on) > 0) or (len(self.dark_frame_files_cap_off) > 0)
        luminance_mode = self.current_luminance_mode()
        return {
            'format': self.combo_format.currentText(),
            'bit_depth': 16 if self.combo_bit_depth.currentText() == '16-bit' else 8,
            'color_space': self.combo_color.currentText(),
            'luminance_mode': luminance_mode,
            'demosaic': self.demosaic_map[self.combo_demosaic.currentText()],
            'sharp_amount': self.spin_sharp_amount.value() if self.check_use_sharpness.isChecked() else 0.0,
            'sharp_radius': self.spin_sharp_radius.value(),
            'overwrite': self.check_overwrite.isChecked(),
            'undersample_n': self.spin_undersample.value(),
            'output_downsample_factor': self.spin_output_downsample.value(),
            'process_at_output_scale': self.check_process_at_output_scale.isChecked(),
            'burnt_percentile': self._current_percentile_value(),
            'output_folder': self.edit_output_folder.text().strip(),
            'rotation_enabled': self.check_rotation_enable.isChecked(),
            'rotation_angle': self.combo_rotation.currentIndex(),
            'grey_rotation_enabled': self._grey_rotation_enabled(),
            'grey_rotation_angle': self.combo_grey_rotation.currentIndex(),
            'apply_dark_level': self.check_apply_dark_level.isChecked() and (not use_dark_frames),
            'dark_lift_coeff': self.get_dark_lift_coeff_value(),
            'dark_frame_paths_cap_on': list(self.dark_frame_files_cap_on),
            'dark_frame_paths_cap_off': list(self.dark_frame_files_cap_off),
            'use_weighted_luminance': luminance_mode == 'demosaic_weighted',
            'weight_r': self.spin_weight_r.value(),
            'weight_g': self.spin_weight_g.value(),
            'weight_b': self.spin_weight_b.value(),
            'light_compensation_enabled': self.check_calibrate_light_variance.isChecked(),
            'light_compensation_map': {},
            'flatfield_enabled': self.check_flatfield_enable.isChecked(),
            'flatfield_map': {},
            'flatfield_smooth_sigma_rel': float(np.clip(self.spin_flatfield_sigma_pct.value() / 100.0, 0.0, 0.25)),
            'linearity_calibration_enabled': self.check_linearity_enable.isChecked(),
            'linearity_lut_control_points': [],
            'peak_percentile': 99.8
        }

    def _resolve_light_compensation_for_processing(self):
        if not self.check_calibrate_light_variance.isChecked():
            return True, {}, None

        if not self.files:
            return False, {}, "No input files loaded for light compensation."

        path = self.edit_light_comp_file.text().strip()
        if path and os.path.isfile(path):
            try:
                payload, gain_values = self._parse_light_compensation_file(path)
                gain_map, rotation_mismatch = self._validate_compensation_against_current_inputs(payload, gain_values)
                if rotation_mismatch:
                    reply = QMessageBox.question(
                        self,
                        "Grey Card Calibration",
                        "Grey-card rotation settings differ from those saved in the compensation file.\n"
                        "Do you want to continue anyway?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return False, {}, "Processing cancelled due to rotation mismatch."
                self.light_comp_gain_map = gain_map
                self.light_comp_gain_values = gain_values
                self.light_comp_metadata = payload
                return True, gain_map, payload
            except Exception as e:
                return False, {}, f"Invalid compensation file: {e}"

        try:
            payload, gain_map = self._compute_light_compensation_payload()
            return True, gain_map, payload
        except Exception as e:
            return False, {}, (
                "No valid compensation file found and unable to compute from grey cards. "
                f"Reason: {e}"
            )

    def _set_dark_analysis_ui_running(self, running):
        self.btn_analyze_dark_clipping.setEnabled(not running)
        self.btn_run.setEnabled(not running)
        self.btn_calc_stretch.setEnabled(not running)
        if running:
            self.lbl_status.setText("Dark analysis running...")
            self.progress.setValue(0)
        else:
            self.lbl_status.setText("Ready")

    def _set_light_compensation_ui_running(self, running):
        self.btn_save_light_comp.setEnabled(not running)
        self.btn_analyze_dark_clipping.setEnabled(not running)
        self.btn_run.setEnabled(not running)
        self.btn_calc_stretch.setEnabled(not running)
        if running:
            self.lbl_status.setText("Computing light compensation...")
            self.progress.setValue(0)
        else:
            self.lbl_status.setText("Ready")

    def _build_dark_analysis_report_text(self, result):
        metadata_enabled = bool(result.get('metadata_enabled', False))
        manual_dark_active = bool(result.get('manual_dark_active', False))
        current_coeff = float(result.get('current_coeff', 0.0))
        suggested_coeff = float(result.get('suggested_coeff', current_coeff))
        clipped_count = int(result.get('clipped_count', 0))
        clipped_files = result.get('clipped_files', [])
        failed_files = result.get('failed_files', [])

        report_lines = [
            "Dark level overestimation check report",
            "",
            f"Metadata dark calibration active: {'Yes' if metadata_enabled else 'No'}",
            f"Manual dark frames active: {'Yes' if manual_dark_active else 'No'}",
            f"RAW files considered: {int(result.get('raw_files_considered', 0))}",
            f"RAW files analyzed: {int(result.get('raw_files_analyzed', 0))}",
            f"Files with possible dark clipping: {clipped_count}",
            f"Current lift coefficient: {current_coeff:.6f}",
            f"Mean negative ratio: {float(result.get('mean_neg_ratio', 0.0)):.6f}%",
            f"Max negative ratio: {float(result.get('max_neg_ratio', 0.0)):.6f}%",
            f"Global negative ratio: {float(result.get('global_neg_ratio', 0.0)):.6f}%"
        ]

        if bool(result.get('negligible_clipping', False)):
            report_lines.extend([
                "",
                "Clipping is negligible because it is below the tolerance threshold (0.000001%)."
            ])

        if (metadata_enabled or manual_dark_active) and clipped_count > 0:
            report_lines.extend([
                "",
                f"Suggested lift coefficient: {suggested_coeff:.6f}",
                "",
                "Top files with clipping:"
            ])
            for name, ratio, need in clipped_files[:12]:
                report_lines.append(f"- {name}: {ratio:.6f}% negative, estimated need {need:.6f}")
        elif not metadata_enabled and not manual_dark_active:
            report_lines.extend([
                "",
                "Metadata dark calibration is currently disabled.",
                "No RAW black-level subtraction was applied in this analysis.",
                "Enable metadata dark calibration to use Set coefficient on an estimated value."
            ])
        else:
            report_lines.extend(["", "No relevant dark clipping detected with current settings."])

        if failed_files:
            report_lines.append("")
            report_lines.append("Files skipped due to read errors:")
            for name, err in failed_files[:10]:
                report_lines.append(f"- {name}: {err}")

        return "\n".join(report_lines)

    def _show_dark_analysis_report(self, result):
        metadata_enabled = bool(result.get('metadata_enabled', False))
        manual_dark_active = bool(result.get('manual_dark_active', False))
        current_coeff = float(result.get('current_coeff', 0.0))
        suggested_coeff = float(result.get('suggested_coeff', current_coeff))
        clipped_count = int(result.get('clipped_count', 0))
        report_text = self._build_dark_analysis_report_text(result)

        msg = QMessageBox(self)
        msg.setWindowTitle("Dark Level Overestimation Check")
        msg.setIcon(QMessageBox.Information if clipped_count == 0 else QMessageBox.Warning)
        msg.setText("Dark level overestimation check completed.")
        if (metadata_enabled or manual_dark_active) and clipped_count > 0:
            msg.setInformativeText(
                f"Detected possible dark clipping in {clipped_count} file(s). Suggested lift coefficient: {suggested_coeff:.6f}"
            )
        elif clipped_count > 0:
            msg.setInformativeText(f"Detected possible dark clipping in {clipped_count} file(s).")
        elif bool(result.get('negligible_clipping', False)):
            msg.setInformativeText("Clipping is negligible because it is below the tolerance threshold (0.000001%).")
        else:
            msg.setInformativeText("No significant dark clipping detected.")
        msg.setDetailedText(report_text)

        set_btn = msg.addButton("Set coefficient", QMessageBox.AcceptRole)
        close_btn = msg.addButton("Close", QMessageBox.RejectRole)

        if (not metadata_enabled and not manual_dark_active) or (clipped_count == 0):
            set_btn.setEnabled(False)

        msg.exec_()

        if msg.clickedButton() == set_btn:
            self.set_dark_lift_coeff_value(suggested_coeff)
            self.update_clip_preview()
            print(f"Applied suggested dark lift coefficient: {suggested_coeff:.6f}")

        if msg.clickedButton() == close_btn:
            return

    def on_dark_analysis_progress(self, value):
        self.progress.setValue(int(value))

    def on_dark_analysis_failed(self, message):
        self._set_dark_analysis_ui_running(False)
        if self.dark_analysis_thread is not None:
            self.dark_analysis_thread.deleteLater()
        self.dark_analysis_thread = None
        self.progress.setValue(0)
        QMessageBox.warning(self, "Dark Analysis", f"Dark analysis failed: {message}")

    def on_dark_analysis_finished(self, result):
        self._set_dark_analysis_ui_running(False)
        if self.dark_analysis_thread is not None:
            self.dark_analysis_thread.deleteLater()
        self.dark_analysis_thread = None

        status = result.get('status')
        if status == 'no_raw':
            self.progress.setValue(0)
            QMessageBox.information(self, "Dark Analysis", "No RAW files found in selection.")
            return
        if status == 'no_valid':
            self.progress.setValue(0)
            QMessageBox.information(self, "Dark Analysis", "No valid RAW metadata black level found to analyze.")
            return

        self.progress.setValue(100)
        self._show_dark_analysis_report(result)

    def analyze_dark_clipping(self):
        if not self.files:
            QMessageBox.warning(self, "Dark Analysis", "Please add files before running analysis.")
            return

        if self.dark_analysis_thread is not None and self.dark_analysis_thread.isRunning():
            return

        metadata_enabled = self.check_apply_dark_level.isChecked()
        current_coeff = self.get_dark_lift_coeff_value()
        manual_dark_active = (len(self.dark_frame_files_cap_on) > 0) or (len(self.dark_frame_files_cap_off) > 0)
        cap_on_offset = self._preview_compute_frame_mean(self.dark_frame_files_cap_on) if self.dark_frame_files_cap_on else 0.0
        cap_off_raw = self._preview_compute_frame_mean(self.dark_frame_files_cap_off) if self.dark_frame_files_cap_off else 0.0
        dark_frame_offset = max(0.0, cap_off_raw - cap_on_offset)

        self.dark_analysis_thread = DarkLevelAnalysisThread(
            files=self.files,
            metadata_enabled=metadata_enabled,
            current_coeff=current_coeff,
            sample_step=4,
            manual_dark_active=manual_dark_active,
            manual_dark_cap_on=cap_on_offset,
            manual_dark_offset=dark_frame_offset
        )
        self.dark_analysis_thread.progress.connect(self.on_dark_analysis_progress)
        self.dark_analysis_thread.analysis_done.connect(self.on_dark_analysis_finished)
        self.dark_analysis_thread.analysis_failed.connect(self.on_dark_analysis_failed)

        self._set_dark_analysis_ui_running(True)
        self.dark_analysis_thread.start()

    def start_stretch_calculation(self):
        self.start_processing(phase1_only=True)

    def start_processing(self, phase1_only=False):
        if self.thread is not None and self.thread.isRunning():
            print("Interrupt request in progress...")
            self.thread.stop()
            self.btn_run.setEnabled(False)
            self.btn_calc_stretch.setEnabled(False)
            return

        if not self.files:
            QMessageBox.warning(self, "Warning", "Please add at least one file!")
            return

        options = self._build_base_options()

        if options.get('linearity_calibration_enabled', False):
            try:
                points = self._current_linearity_lut_points()
                if not points:
                    raise ValueError("LUT text area is empty.")
                options['linearity_lut_control_points'] = points
            except Exception as e:
                QMessageBox.warning(self, "Linearity Calibration", f"Invalid LUT format: {e}")
                return

        if options.get('flatfield_enabled', False):
            flat_map, flat_err = self._build_flatfield_map_for_inputs()
            if flat_err:
                QMessageBox.warning(self, "Flatfielding", flat_err)
                return
            options['flatfield_map'] = flat_map

        ok_light, gain_map, light_payload = self._resolve_light_compensation_for_processing()
        if not ok_light:
            QMessageBox.warning(self, "Grey Card Calibration", light_payload)
            return
        options['light_compensation_enabled'] = self.check_calibrate_light_variance.isChecked()
        options['light_compensation_map'] = gain_map
        options['phase1_only'] = phase1_only

        self.current_phase1_only = phase1_only
        if phase1_only:
            self.lbl_status.setText("Starting Stretch Analysis...")
        else:
            self.lbl_status.setText("Starting Develop Luminance...")
        self.progress.setValue(0)
        self.set_button_stop()
        self.btn_calc_stretch.setEnabled(False)
        self.thread = ProcessThread(self.files, options)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.status.connect(self.lbl_status.setText)
        self.thread.analysis_threshold.connect(self.on_stretch_threshold_ready)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def toggle_processing(self):
        self.start_processing(phase1_only=False)

    def on_finished(self, success, message, out_folder):
        phase1_only = self.current_phase1_only
        self.set_button_develop()
        self.btn_run.setEnabled(True)
        self.btn_calc_stretch.setEnabled(True)
        self.current_phase1_only = False

        if (not phase1_only) and out_folder and os.path.isdir(out_folder):
            self.last_output_folder = out_folder
        
        if success:
            print(f"SUCCESS: {message}")
            self.progress.setValue(100)
            if phase1_only:
                self.lbl_status.setText("Stretch calculated.")
            else:
                self.lbl_status.setText("Completed.")
        else:
            print(f"FAILED/INTERRUPTED: {message}")
            self.progress.setValue(0)
            self.lbl_status.setText("Interrupted.")

        QApplication.processEvents()

        if out_folder and os.path.exists(out_folder):
            try:
                log_path = os.path.join(out_folder, "export_log.txt")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(self.console.toPlainText())
            except Exception as e:
                print(f"Error saving log: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())