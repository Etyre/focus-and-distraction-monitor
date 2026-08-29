"""Build dist/Focus Monitor.app - a standalone menu bar app wrapping this project's venv.

    .venv/bin/python build_app.py
    open "dist/Focus Monitor.app"

The bundle's main executable is a tiny native launcher (so macOS attributes Screen Recording /
Accessibility / Automation permissions to "Focus Monitor" rather than to Python). It execs the
venv's Python with `-m focus_monitor app`. The bundle references this directory - move the
project and rebuild.
"""
from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"
DIST = ROOT / "dist"
APP = DIST / "Focus Monitor.app"
BUNDLE_ID = "com.elityre.focusmonitor"

LAUNCHER_C = r"""
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
/* Spawn Python as a child and wait, so the .app (not Python) stays the responsible process
   that macOS attributes Screen Recording / Accessibility / Automation permissions to. */
static pid_t child = 0;
static void forward(int sig) { if (child > 0) kill(child, sig); }
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    if (chdir("%(root)s") != 0) { perror("chdir"); return 1; }
    setenv("FOCUS_MONITOR_CONFIG", "%(root)s/config.toml", 0);
    setenv("PYTHONUNBUFFERED", "1", 1);
    signal(SIGTERM, forward); signal(SIGINT, forward); signal(SIGHUP, forward);
    child = fork();
    if (child == 0) {
        char *args[] = {"%(python)s", "-m", "focus_monitor", "app", NULL};
        execv("%(python)s", args);
        perror("execv");
        _exit(1);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {}
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
"""


def draw_icon(path: Path) -> None:
    """A simple eye: dark iris on a white almond, on a rounded blue tile."""
    size = 1024
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    m = 90
    d.rounded_rectangle((m, m, size - m, size - m), radius=210, fill=(42, 120, 214, 255))
    # almond shape = intersection of two circles, drawn as a polygon
    import math
    pts = []
    cx, cy, w, h = size / 2, size / 2, 340, 200
    for i in range(0, 181):
        a = math.radians(i)
        pts.append((cx - w * math.cos(a), cy - h * math.sin(a)))
    for i in range(0, 181):
        a = math.radians(i)
        pts.append((cx + w * math.cos(a), cy + h * math.sin(a)))
    d.polygon(pts, fill=(252, 252, 251, 255))
    d.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), fill=(11, 11, 11, 255))
    d.ellipse((cx - 70, cy - 70, cx + 70, cy + 70), fill=(42, 120, 214, 255))
    d.ellipse((cx - 30, cy - 95, cx + 30, cy - 35), fill=(252, 252, 251, 255))
    iconset = path.with_suffix(".iconset")
    iconset.mkdir(exist_ok=True)
    for s in (16, 32, 64, 128, 256, 512, 1024):
        im.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
        if s <= 512:
            im.resize((s * 2, s * 2), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(path)], check=True)
    shutil.rmtree(iconset)


def main() -> None:
    if not PYTHON.exists():
        sys.exit(f"venv python not found at {PYTHON}")
    if APP.exists():
        shutil.rmtree(APP)
    (APP / "Contents/MacOS").mkdir(parents=True)
    (APP / "Contents/Resources").mkdir(parents=True)

    src = DIST / "launcher.c"
    src.write_text(LAUNCHER_C % {"root": ROOT, "python": PYTHON})
    subprocess.run(["clang", "-O2", "-o", str(APP / "Contents/MacOS/FocusMonitor"), str(src)], check=True)
    src.unlink()

    draw_icon(APP / "Contents/Resources/AppIcon.icns")

    plist = {
        "CFBundleName": "Focus Monitor",
        "CFBundleDisplayName": "Focus Monitor",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": "FocusMonitor",
        "CFBundleIconFile": "AppIcon",
        "LSUIElement": True,  # menu bar only, no Dock icon
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",
        "NSAppleEventsUsageDescription": "Focus Monitor reads the active window title and browser tab URL to track focus.",
        "NSScreenCaptureUsageDescription": "Focus Monitor takes screenshots when you switch windows so the model can judge attention events.",
    }
    with open(APP / "Contents/Info.plist", "wb") as f:
        plistlib.dump(plist, f)
    (APP / "Contents/PkgInfo").write_text("APPL????")

    subprocess.run(["xattr", "-cr", str(APP)], check=True)  # strip Finder detritus that blocks signing
    subprocess.run(["codesign", "--force", "--sign", "-", "--identifier", BUNDLE_ID, str(APP)], check=True)
    print(f"built {APP}\n  open \"{APP}\"   or drag it to /Applications and add to Login Items")


if __name__ == "__main__":
    main()
