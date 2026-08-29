"""Render SF Symbols to monochrome PNGs for use as menu bar template icons."""
from __future__ import annotations

from pathlib import Path

from AppKit import (NSBitmapImageFileTypePNG, NSBitmapImageRep, NSCompositingOperationSourceOver,
                    NSFontWeightMedium, NSGraphicsContext, NSImage, NSImageSymbolConfiguration, NSMakeRect)

from .config import DATA_DIR

SYMBOLS = {"watching": "eye", "paused": "eye.slash", "trouble": "eye.trianglebadge.exclamationmark"}


def render_symbol(name: str, out: Path, px: int = 44) -> Path:
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
    if img is None:
        raise RuntimeError(f"SF Symbol not found: {name}")
    img = img.imageWithSymbolConfiguration_(
        NSImageSymbolConfiguration.configurationWithPointSize_weight_(px * 0.75, NSFontWeightMedium))
    w, h = img.size().width, img.size().height
    scale = px / max(w, h)
    W, H = int(round(w * scale)), int(round(h * scale))
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, W, H, 8, 4, True, False, "NSCalibratedRGBColorSpace", 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    img.drawInRect_fromRect_operation_fraction_(NSMakeRect(0, 0, W, H), NSMakeRect(0, 0, w, h),
                                                NSCompositingOperationSourceOver, 1.0)
    NSGraphicsContext.restoreGraphicsState()
    rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, None).writeToFile_atomically_(str(out), True)
    return out


def menu_icons() -> dict[str, str]:
    """Return {state: png_path}, rendering into DATA_DIR/icons if needed."""
    d = DATA_DIR / "icons"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for state, sym in SYMBOLS.items():
        p = d / f"{state}.png"
        if not p.exists():
            render_symbol(sym, p)
        out[state] = str(p)
    return out
