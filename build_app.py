"""Build dist/Focus Monitor.app - a standalone menu bar app wrapping this project's venv.

    .venv/bin/python build_app.py
    open "dist/Focus Monitor.app"

The bundle's main executable is a tiny native launcher (so macOS attributes Screen Recording /
Accessibility / Automation permissions to "Focus Monitor" rather than to Python). It runs the
venv's Python with `-m focus_monitor app` and supervises it: a crash, or a heartbeat that stops
(the collector touches a file every poll), gets the Python process killed and restarted, logged
to `~/Library/Application Support/FocusMonitor/launcher.log`. Exit status 75 means "relaunch me
now" (POST /api/restart, or `python -m focus_monitor restart`) and is not counted as a crash. The bundle references this
directory - move the project and rebuild.
"""
from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"
DIST = Path(os.environ.get("FOCUS_MONITOR_DIST", ROOT / "dist"))  # override for test builds
APP = DIST / "Focus Monitor.app"
BUNDLE_ID = "com.elityre.focusmonitor"

LAUNCHER_C = r"""
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
/* Spawn Python as a child and supervise it, so the .app (not Python) stays the responsible
   process that macOS attributes Screen Recording / Accessibility / Automation permissions to.

   The child is a copy of the interpreter stub that lives INSIDE this bundle, not Homebrew's
   Python.app. 2026-11-04: on macOS 26.6.2 Control Center hosts menu bar items only for
   processes whose bundle carries a sealed signature. Homebrew's Python.app is ad-hoc signed
   with its Info.plist unbound, so its status item was created, reported visible, and never
   got a slot - the eye vanished. The same stub run from a signed bundle is placed at once.
   The venv is reached through PYTHONPATH because the stub no longer sits in .venv/bin.

   The supervisor restarts the child when it crashes, and when it stops ticking: the collector
   touches HEARTBEAT every poll. 2026-09-04: macOS froze the Python process for three days
   (every thread stalled at zero CPU, menu unresponsive, no log lines) while the machine was
   awake. A frozen process cannot recover itself, so the check has to live outside it. */

#define CHECK_SECONDS   30
#define STALE_SECONDS   180   /* heartbeat age that counts as frozen */
#define STALE_CHECKS    2     /* consecutive awake checks at that age before restarting */
#define GRACE_SECONDS   120   /* after a (re)start, before the heartbeat is judged */
#define STOP_SECONDS    60    /* a frozen child gets this long to honour SIGTERM before SIGKILL */
#define MAX_RESTARTS    5     /* this many restarts within RESTART_WINDOW -> give up */
#define RESTART_WINDOW  600
#define RESTART_CODE    75    /* python asks to be relaunched (dev restart) - see focus_monitor/config.py */

static pid_t child = 0;
static volatile sig_atomic_t stopping = 0;
static char heartbeat[1024], logpath[1024];

static void forward(int sig) { stopping = 1; if (child > 0) kill(child, sig); }

static void logline(const char *fmt, ...) {
    FILE *f = fopen(logpath, "a");
    if (!f) return;
    time_t t = time(NULL);
    char ts[32];
    strftime(ts, sizeof ts, "%%Y-%%m-%%d %%H:%%M:%%S", localtime(&t));
    fprintf(f, "%%s launcher ", ts);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
}

static pid_t spawn(void) {
    pid_t pid = fork();
    if (pid == 0) {
        char *args[] = {"%(python)s", "-m", "focus_monitor", "app", NULL};
        setenv("PYTHONPATH", "%(pythonpath)s", 1);
        setenv("PYTHONNOUSERSITE", "1", 1);
        execv("%(python)s", args);
        perror("execv");
        _exit(1);
    }
    return pid;
}

/* Seconds the machine has been awake - does not advance during system sleep. */
static double awake_now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e9; }

static void reap(int *status) { while (waitpid(child, status, 0) < 0 && errno == EINTR) {} }

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    if (chdir("%(root)s") != 0) { perror("chdir"); return 1; }
    setenv("FOCUS_MONITOR_CONFIG", "%(root)s/config.toml", 0);
    setenv("PYTHONUNBUFFERED", "1", 1);
    const char *home = getenv("HOME");
    if (!home) home = "";
    snprintf(heartbeat, sizeof heartbeat, "%%s/Library/Application Support/FocusMonitor/heartbeat", home);
    snprintf(logpath, sizeof logpath, "%%s/Library/Application Support/FocusMonitor/launcher.log", home);
    signal(SIGTERM, forward); signal(SIGINT, forward); signal(SIGHUP, forward);

    time_t restarts[MAX_RESTARTS] = {0};
    int nrestarts = 0;
    for (;;) {
        child = spawn();
        if (child < 0) { logline("fork failed: %%s", strerror(errno)); return 1; }
        time_t started = time(NULL);
        logline("started python pid %%d", (int)child);
        int status = 0, stale = 0, stop_wait = 0, frozen = 0;
        double last_awake = awake_now();
        time_t last_wall = started;
        for (;;) {
            for (int i = 0; i < CHECK_SECONDS && !stopping; i++) sleep(1);  /* 1 s slices: a signal is noticed promptly */
            if (stopping) sleep(1);
            pid_t r = waitpid(child, &status, WNOHANG);
            if (r == child) break;                          /* exited on its own */
            if (r < 0 && errno != EINTR) { status = 0; break; }
            if (stopping) {                                 /* we forwarded a signal */
                if (++stop_wait >= STOP_SECONDS) { kill(child, SIGKILL); reap(&status); break; }
                continue;
            }
            double awake = awake_now();
            time_t wall = time(NULL);
            int slept = (double)(wall - last_wall) - (awake - last_awake) > 60;
            last_awake = awake; last_wall = wall;
            if (slept) { stale = 0; continue; }            /* a stale heartbeat after sleep is normal */
            if (wall - started < GRACE_SECONDS) continue;
            struct stat st;
            double age = stat(heartbeat, &st) == 0 ? (double)(wall - st.st_mtime) : 1e9;
            if (age <= STALE_SECONDS) { stale = 0; continue; }
            if (++stale < STALE_CHECKS) continue;
            logline("heartbeat %%.0fs old on %%d consecutive checks - killing frozen python pid %%d", age, stale, (int)child);
            kill(child, SIGKILL);
            reap(&status);
            frozen = 1;
            break;
        }
        if (stopping) { logline("stopped by signal"); return 0; }
        if (!frozen && WIFEXITED(status) && WEXITSTATUS(status) == RESTART_CODE) {
            logline("python asked to be relaunched");
            continue;                                   /* at once, and outside the crash budget */
        }
        if (!frozen) {
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) { logline("python quit cleanly"); return 0; }
            if (WIFSIGNALED(status)) logline("python died: signal %%d", WTERMSIG(status));
            else logline("python died: exit status %%d", WEXITSTATUS(status));
        }
        time_t now = time(NULL);
        restarts[nrestarts %% MAX_RESTARTS] = now;
        nrestarts++;
        if (nrestarts >= MAX_RESTARTS && now - restarts[nrestarts %% MAX_RESTARTS] < RESTART_WINDOW) {
            logline("%%d restarts within %%d s - giving up", MAX_RESTARTS, RESTART_WINDOW);
            return 1;
        }
        logline("restarting in 5 s");
        sleep(5);
    }
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


def signing_identity() -> str:
    """A certificate from the keychain, or "-" (ad hoc).

    TCC (the privacy permissions database) records the app's designated requirement when a
    permission is granted. With a real certificate that is `identifier + certificate`, which a
    rebuild leaves unchanged. Ad hoc, it is the binary's hash, so every rebuild invalidates the
    Screen Recording grant and the "would like to record this computer's screen" prompt returns
    on every screenshot (2026-09-07).
    """
    out = subprocess.run(["security", "find-identity", "-v", "-p", "codesigning"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].endswith(")") and len(parts[1]) == 40:
            return parts[1]  # the SHA-1 of the first valid identity
    return "-"


def interpreter_facts() -> tuple[Path, str]:
    """The framework interpreter stub behind the venv, and the PYTHONPATH that reproduces the venv."""
    out = subprocess.run([str(PYTHON), "-c", "import sys, sysconfig; print(sys.base_prefix); "
                          "print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True,
                         check=True).stdout.split()
    stub = Path(out[0]) / "Resources/Python.app/Contents/MacOS/Python"
    if not stub.exists():
        sys.exit(f"framework interpreter not found at {stub}")
    return stub, f"{ROOT}:{out[1]}"


def main() -> None:
    if not PYTHON.exists():
        sys.exit(f"venv python not found at {PYTHON}")
    stub, pythonpath = interpreter_facts()
    if APP.exists():
        shutil.rmtree(APP)
    (APP / "Contents/MacOS").mkdir(parents=True)
    (APP / "Contents/Resources").mkdir(parents=True)

    src = DIST / "launcher.c"
    embedded = APP / "Contents/MacOS/Python"
    shutil.copy2(stub, embedded)
    src.write_text(LAUNCHER_C % {"root": ROOT, "python": embedded, "pythonpath": pythonpath})
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
        "NSAppSleepDisabled": True,  # never App Nap: the collector must poll while invisible
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",
        "NSAppleEventsUsageDescription": "Focus Monitor reads the active window title and browser tab URL to track focus.",
        "NSScreenCaptureUsageDescription": "Focus Monitor takes screenshots when you switch windows so the model can judge attention events.",
    }
    with open(APP / "Contents/Info.plist", "wb") as f:
        plistlib.dump(plist, f)
    (APP / "Contents/PkgInfo").write_text("APPL????")

    identity = signing_identity()
    # The embedded interpreter is a second executable in Contents/MacOS: the bundle signature
    # below does not cover it, and Control Center refuses status items from an unsealed binary.
    subprocess.run(["codesign", "--force", "--sign", identity, "--identifier", BUNDLE_ID, str(embedded)], check=True)
    for attempt in range(5):
        # Strip Finder detritus that blocks signing. Finder (this is an iCloud-synced folder) can
        # re-tag the bundle with FinderInfo between the strip and the sign, so retry.
        subprocess.run(["xattr", "-cr", str(APP)], check=True)
        r = subprocess.run(["codesign", "--force", "--sign", identity, "--identifier", BUNDLE_ID, str(APP)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            break
        if attempt == 4 or "detritus" not in r.stderr:
            sys.exit(f"codesign failed: {r.stderr.strip()}")
        time.sleep(0.5)
    if identity == "-":
        print("WARNING: no code-signing identity in the keychain - signed ad hoc. macOS ties Screen Recording /"
              " Accessibility grants to the signature, and an ad-hoc signature changes on every build, so the"
              " permissions will have to be granted again after each rebuild.")
    print(f"built {APP}\n  open \"{APP}\"   or drag it to /Applications and add to Login Items")


if __name__ == "__main__":
    main()
