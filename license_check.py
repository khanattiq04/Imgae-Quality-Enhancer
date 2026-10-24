"""Remote on/off switch for the desktop app.

Every time the app starts (and every 30 minutes while it is open) it reads a
small file on the owner's website, public/license.json:

    {"enabled": true, "blocked": ["AB12-CD34"], "message": "..."}

* "enabled": false turns every copy off.
* "blocked" turns off only the listed computers (the Device ID is shown in
  the app's title bar).

The owner changes that file with app-switch.bat (license_admin.py), which
pushes it to GitHub; Vercel redeploys it within a minute or two.

Without internet the app keeps working for OFFLINE_GRACE_DAYS after the last
successful check; a copy that was turned off stays off while offline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import time
import urllib.request
import uuid

# The address of public/license.json on your Vercel site.
LICENSE_URL = "https://YOUR-SITE.vercel.app/license.json"
OFFLINE_GRACE_DAYS = 3
CHECK_EVERY_SECONDS = 30 * 60

DEFAULT_MESSAGE = "This copy of Screenshot HD Print Enhancer has been turned off by its owner."
_STATE_KEY = b"shd-print-enhancer-license-state-v1"


def configured() -> bool:
    return "YOUR-SITE" not in LICENSE_URL


def _machine_guid() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography",
                            0, winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0)) as key:
            return str(winreg.QueryValueEx(key, "MachineGuid")[0])
    except Exception:
        return f"{uuid.getnode():x}-{platform.node()}"


def device_id() -> str:
    """Short, stable ID for this computer, e.g. "AB12-CD34"."""
    digest = hashlib.sha256(("shd:" + _machine_guid()).encode()).hexdigest().upper()
    return f"{digest[:4]}-{digest[4:8]}"


def _state_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "ScreenshotHDPrintEnhancer", "license_state.json")


def _sign(payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(_STATE_KEY + device_id().encode(), body, hashlib.sha256).hexdigest()


def _save_state(allowed: bool, message: str) -> None:
    state = {"allowed": allowed, "message": message, "checked": time.time()}
    state["sig"] = _sign(state)
    try:
        os.makedirs(os.path.dirname(_state_path()), exist_ok=True)
        with open(_state_path(), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError:
        pass


def _load_state():
    try:
        with open(_state_path(), encoding="utf-8") as fh:
            state = json.load(fh)
        sig = state.pop("sig", "")
        if hmac.compare_digest(sig, _sign(state)):
            return state
    except (OSError, ValueError, AttributeError):
        pass
    return None


def decide(data: dict, device: str) -> tuple[bool, str]:
    """Whether this computer may run, from the contents of license.json."""
    message = str(data.get("message") or DEFAULT_MESSAGE)
    blocked = {str(d).strip().upper() for d in data.get("blocked", []) if str(d).strip()}
    if not data.get("enabled", True):
        return False, message
    if device.upper() in blocked:
        return False, message
    return True, ""


def check(timeout: float = 6.0) -> tuple[bool, str]:
    """(allowed, message). Reads the switch online, falls back to the last result offline."""
    if not configured():
        return True, ""
    try:
        url = f"{LICENSE_URL}?t={int(time.time())}"
        request = urllib.request.Request(url, headers={"Cache-Control": "no-cache",
                                                       "User-Agent": "ScreenshotHDPrintEnhancer"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("license file is not an object")
        allowed, message = decide(data, device_id())
        _save_state(allowed, message)
        return allowed, message
    except Exception:
        state = _load_state()
        if state is None:
            return False, ("Connect this computer to the internet once to start "
                           "Screenshot HD Print Enhancer.")
        if not state.get("allowed"):
            return False, state.get("message") or DEFAULT_MESSAGE
        if time.time() - float(state.get("checked", 0)) <= OFFLINE_GRACE_DAYS * 86400:
            return True, ""
        return False, ("Screenshot HD Print Enhancer has not been able to check its licence for "
                       f"{OFFLINE_GRACE_DAYS} days. Connect to the internet and open it again.")
