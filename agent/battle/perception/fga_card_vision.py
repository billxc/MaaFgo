"""Fast OpenCV 720p template matching for FGA command card color recognition."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..core.enums import CardColor
from ..core.models import Confidence
from .card_owner import BASE_HEIGHT, BASE_WIDTH, _normalize_image

# FGA typeRegion(card): 1440p x=[-1280, -768, -256, 256, 768] from center, y=1060, w=512, h=200
#   -> 720p x=[0, 256, 512, 768, 1024], y=500..645.
COLOR_ROIS = {
    1: (0, 500, 256, 145),
    2: (256, 500, 256, 145),
    3: (512, 500, 256, 145),
    4: (768, 500, 256, 145),
    5: (1024, 500, 256, 145),
}


@dataclass(frozen=True)
class _PreparedTemplate:
    name: str
    bgr: np.ndarray
    gray: np.ndarray
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class FastCardColorDetection:
    ui_slot: int
    color: CardColor
    confidence: Confidence
    score: float = 0.0
    margin: float = 0.0
    template: str = ""


def default_fga_card_roots() -> tuple[Path, ...]:
    """Return source-tree and packaged fga_cards resource directories."""
    project = Path(__file__).resolve().parents[3]
    roots: list[Path] = []
    for root in (project, project.parent):
        for candidate in (
            root / "assets" / "resource" / "base" / "image" / "fga_cards",
            root / "resource" / "base" / "image" / "fga_cards",
        ):
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    return tuple(roots)


def _default_image_roots() -> tuple[Path, ...]:
    """Return base image roots (assets/resource/base/image or resource/base/image)."""
    project = Path(__file__).resolve().parents[3]
    roots: list[Path] = []
    for root in (project, project.parent):
        for candidate in (
            root / "assets" / "resource" / "base" / "image",
            root / "resource" / "base" / "image",
        ):
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    return tuple(roots)


def _load_template_file(path: Path, name: str | None = None) -> _PreparedTemplate | None:
    if not path.is_file():
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim not in (2, 3):
        return None
    mask: np.ndarray | None = None
    if raw.ndim == 2:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        gray = raw
    elif raw.shape[2] == 4:
        bgr = np.ascontiguousarray(raw[:, :, :3], dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        alpha = raw[:, :, 3]
        if int(np.min(alpha)) < 250:
            m = np.where(alpha >= 128, 255, 0).astype(np.uint8)
            if cv2.countNonZero(m) >= 16:
                mask = m
    elif raw.shape[2] == 3:
        bgr = np.ascontiguousarray(raw, dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        return None
    return _PreparedTemplate(name=name or path.name, bgr=bgr, gray=gray, mask=mask)


def _normalize_720p(image) -> np.ndarray | None:
    frame = _normalize_image(image)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if h < 90 or w < 160:
        return None
    if (w, h) != (BASE_WIDTH, BASE_HEIGHT):
        interp = cv2.INTER_AREA if w > BASE_WIDTH else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (BASE_WIDTH, BASE_HEIGHT), interpolation=interp)
    return frame


def _match_best(
    crop_bgr: np.ndarray,
    crop_gray: np.ndarray,
    templates: Sequence[_PreparedTemplate],
    *,
    use_bgr: bool = True,
) -> tuple[float, str]:
    best_score = 0.0
    best_name = ""
    ch, cw = crop_bgr.shape[:2]
    for tpl in templates:
        th, tw = tpl.bgr.shape[:2]
        if th > ch or tw > cw:
            continue
        target = crop_bgr if use_bgr else crop_gray
        needle = tpl.bgr if use_bgr else tpl.gray
        if tpl.mask is not None:
            res = cv2.matchTemplate(target, needle, cv2.TM_CCOEFF_NORMED, mask=tpl.mask)
        else:
            res = cv2.matchTemplate(target, needle, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = float(score)
            best_name = tpl.name
    return best_score, best_name


def _load_composed_text_template(path: Path, size: tuple[int, int], name: str) -> _PreparedTemplate | None:
    if not path.is_file():
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
        return None
    floating = raw.astype(np.float32) / 255.0
    floating[:, :, :3] *= floating[:, :, 3:4]
    resized = cv2.resize(floating, size, interpolation=cv2.INTER_LINEAR)
    alpha = resized[:, :, 3:4]
    colors = np.divide(resized[:, :, :3], alpha, out=np.zeros_like(resized[:, :, :3]), where=alpha > 0)
    bgra = np.rint(np.clip(np.concatenate((colors, alpha), axis=2) * 255.0, 0, 255)).astype(np.uint8)
    mask = np.where(bgra[:, :, 3] >= 128, 255, 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)
    if w < 8 or h < 8:
        return None
    bgr = np.ascontiguousarray(bgra[y : y + h, x : x + w, :3])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cropped_mask = np.ascontiguousarray(mask[y : y + h, x : x + w])
    return _PreparedTemplate(name=name, bgr=bgr, gray=gray, mask=cropped_mask)


@lru_cache(maxsize=8)
def _load_color_bank(
    fga_roots: tuple[str, ...],
    image_roots: tuple[str, ...],
) -> dict[CardColor, tuple[_PreparedTemplate, ...]]:
    mapping = {
        CardColor.BUSTER: ("buster", "buster.png"),
        CardColor.ARTS: ("arts", "art.png"),
        CardColor.QUICK: ("quick", "quick.png"),
    }
    bank: dict[CardColor, list[_PreparedTemplate]] = {
        CardColor.BUSTER: [],
        CardColor.ARTS: [],
        CardColor.QUICK: [],
    }
    for root_str in fga_roots:
        root = Path(root_str)
        for color, (_, filename) in mapping.items():
            for region in ("en", "cn"):
                tpl = _load_template_file(root / region / filename, f"fga/{region}/{filename}")
                if tpl is not None:
                    bank[color].append(tpl)

    for img_root_str in image_roots:
        img_root = Path(img_root_str)
        res_root = img_root.parent.parent
        for reg_folder, size in (("base", (round(150 * 1.25), round(73 * 1.25))),
                                 ("cn", (round(126 * 1.25), round(66 * 1.25)))):
            shared_dir = res_root / reg_folder / "image" / "CommandCards" / "_composition" / "shared"
            if not shared_dir.is_dir():
                continue
            for color, (color_key, _) in mapping.items():
                tpl = _load_composed_text_template(
                    shared_dir / f"card_txt_{color_key}.png",
                    size,
                    f"shared/{reg_folder}/card_txt_{color_key}.png",
                )
                if tpl is not None:
                    bank[color].append(tpl)

    return {color: tuple(items) for color, items in bank.items()}


class FgaCardVision:
    """Fast 720p OpenCV recognizer for command card color (Buster / Arts / Quick)."""

    def __init__(
        self,
        fga_roots: Iterable[str | Path] | None = None,
        *,
        image_roots: Iterable[str | Path] | None = None,
        min_color_score: float = 0.74,
        min_color_margin: float = 0.08,
    ) -> None:
        selected_fga = tuple(Path(p) for p in (fga_roots or default_fga_card_roots()))
        self.fga_roots = tuple(str(p.resolve()) for p in selected_fga if p.is_dir())
        selected_img = tuple(Path(p) for p in (image_roots or _default_image_roots()))
        self.image_roots = tuple(str(p.resolve()) for p in selected_img if p.is_dir())
        self.min_color_score = float(min_color_score)
        self.min_color_margin = float(min_color_margin)

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "FgaCardVision":
        return cls()

    def detect_card_color(self, image, ui_slot: int) -> FastCardColorDetection:
        """Fast OpenCV template matching for face card color (BUSTER / ARTS / QUICK)."""
        frame = _normalize_720p(image)
        if frame is None or ui_slot not in COLOR_ROIS or not (self.fga_roots or self.image_roots):
            return FastCardColorDetection(
                ui_slot, CardColor.UNKNOWN, Confidence(0.0, "fga_template"),
            )
        x, y, w, h = COLOR_ROIS[ui_slot]
        crop_bgr = frame[y : y + h, x : x + w]
        if crop_bgr.size == 0 or not np.any(crop_bgr):
            return FastCardColorDetection(
                ui_slot, CardColor.UNKNOWN, Confidence(0.0, "fga_template"),
            )
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        bank = _load_color_bank(self.fga_roots, self.image_roots)
        ranked: list[tuple[float, CardColor, str]] = []
        for color in (CardColor.BUSTER, CardColor.ARTS, CardColor.QUICK):
            score, tpl_name = _match_best(crop_bgr, crop_gray, bank.get(color, ()), use_bgr=True)
            ranked.append((score, color, tpl_name))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_color, best_tpl = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        if best_score >= self.min_color_score and margin >= self.min_color_margin:
            conf_val = max(best_score, 0.92) if best_score >= self.min_color_score else best_score
            return FastCardColorDetection(
                ui_slot=ui_slot,
                color=best_color,
                confidence=Confidence(conf_val, "fga_template"),
                score=best_score,
                margin=margin,
                template=best_tpl,
            )
        return FastCardColorDetection(
            ui_slot=ui_slot,
            color=CardColor.UNKNOWN,
            confidence=Confidence(0.0, "fga_template"),
            score=best_score,
            margin=margin,
            template=best_tpl,
        )


def detect_card_color_fast(image, ui_slot: int) -> FastCardColorDetection:
    """Convenience function to detect card color using FGA + shared composition templates."""
    return FgaCardVision.default().detect_card_color(image, ui_slot)
