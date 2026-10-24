"""Turn the desktop app on or off on other computers (run app-switch.bat).

Edits public/license.json and pushes it to GitHub; Vercel then redeploys it,
and every copy of the app reads it on start and every 30 minutes.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LICENSE_FILE = ROOT / "public" / "license.json"
DEFAULT = {"enabled": True, "blocked": [],
           "message": "This copy of Screenshot HD Print Enhancer has been turned off by its owner."}


def load():
    try:
        data = json.loads(LICENSE_FILE.read_text(encoding="utf-8"))
        return {**DEFAULT, **data}
    except (OSError, ValueError):
        return dict(DEFAULT)


def show(data):
    print()
    print("  App status :", "ON for everyone" if data["enabled"] else "OFF for everyone")
    print("  Blocked    :", ", ".join(data["blocked"]) or "none")
    print("  Message    :", data["message"])
    print()


def publish(data, what):
    LICENSE_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {what}. Sending to GitHub...")
    steps = [["git", "add", "public/license.json"],
             ["git", "commit", "-m", f"App switch: {what}"],
             ["git", "push"]]
    for step in steps:
        result = subprocess.run(step, cwd=ROOT)
        if result.returncode != 0:
            print("\nCould not send it to GitHub. Check your internet / git login and run this again.")
            return
    print("\nDone. Vercel updates the site in about 1-2 minutes.")
    print("Open copies of the app switch off within 30 minutes, or straight away when restarted.")


def device(prompt):
    value = input(prompt).strip().upper()
    return value if value else None


def main():
    data = load()
    show(data)
    print("  1. Turn OFF for everyone")
    print("  2. Turn ON for everyone")
    print("  3. Turn off one computer (Device ID)")
    print("  4. Turn on one computer again (Device ID)")
    print("  5. Change the message people see")
    print("  0. Exit")
    choice = input("\nChoose: ").strip()
    if choice == "1":
        data["enabled"] = False
        publish(data, "turned OFF for everyone")
    elif choice == "2":
        data["enabled"] = True
        publish(data, "turned ON for everyone")
    elif choice == "3":
        dev = device("Device ID shown in the app's title bar (e.g. AB12-CD34): ")
        if dev:
            if dev not in data["blocked"]:
                data["blocked"].append(dev)
            publish(data, f"turned off {dev}")
    elif choice == "4":
        dev = device("Device ID to turn on again: ")
        if dev:
            data["blocked"] = [d for d in data["blocked"] if d.upper() != dev]
            publish(data, f"turned on {dev}")
    elif choice == "5":
        text = input("New message: ").strip()
        if text:
            data["message"] = text
            publish(data, "changed the message")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
