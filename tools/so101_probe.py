#!/usr/bin/env python3
"""Read-only probe of a connected SO-ARM101 follower. Writes NOTHING to the arm.

Opens LeRobot's serial bus, checks that the six STS3215 servos answer, and prints what they
report: positions (raw and, when the servos carry a LeRobot calibration, degrees and radians
in the convention the dexforge SO-101 model uses), torque state, voltage, temperature, and the
calibration stored in their memory. It never writes a register and never changes torque: it
closes with ``disconnect(disable_torque=False)`` (LeRobot's default would switch torque off,
which makes a holding arm go limp), and it does not call ``SO101Follower.connect()``, which
would write the motor configuration and re-enable torque.

Run it with LeRobot's interpreter, not dexforge's (it needs ``lerobot``)::

    /path/to/lerobot/.venv/bin/python tools/so101_probe.py --port /dev/tty.usbmodemXXXX
    ... --watch 30        # keep printing positions at 5 Hz for 30 s (move the arm by hand)

The port is the one ``lerobot-find-port`` reports.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

# The SO-101 follower's motors, as LeRobot defines them (ids are the bus ids).
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = MOTORS[:5]  # the five joints dexforge's `so101` model covers; the gripper is separate
# the dexforge model's joint limits (rad), for the sanity check below
LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}


def make_bus(port: str) -> SerialMotorsBus:
    motors = {
        name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES)
        for i, name in enumerate(MOTORS)
    }
    return SerialMotorsBus(port=port, motors=motors)


def try_read(bus: SerialMotorsBus, register: str, motor: str):
    try:
        return bus.read(register, motor, normalize=False)
    except Exception as exc:  # a register some firmware does not expose; the probe goes on
        return f"n/a ({type(exc).__name__})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--watch", type=float, default=0.0, help="then print positions at 5 Hz for this many seconds")
    ap.add_argument("--scan", action="store_true", help="only list which servo ids answer at each baud rate")
    ap.add_argument("--status", action="store_true", help="read the fault flags, voltage and temperature of whichever servos answer")
    args = ap.parse_args()

    bus = make_bus(args.port)
    if args.scan:
        return scan(bus)
    if args.status:
        return status(bus)
    try:
        try:
            bus.connect()  # opens the port and pings every servo; reads only
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            print("\nSome servos did not answer. `--scan` lists every id that does, at every baud rate.", file=sys.stderr)
            return 2
        print(f"connected on {args.port}: all {len(MOTORS)} servos answered\n")

        raw = bus.sync_read("Present_Position", normalize=False)
        print(f"{'motor':14s} {'raw':>6s}  torque  voltage  temp")
        for m in MOTORS:
            print(
                f"{m:14s} {raw[m]:6d}  {str(try_read(bus, 'Torque_Enable', m)):6s}  "
                f"{str(try_read(bus, 'Present_Voltage', m)):7s}  {try_read(bus, 'Present_Temperature', m)}"
            )

        cal = bus.read_calibration()
        print("\ncalibration stored in the servos (LeRobot's `lerobot-calibrate` writes it):")
        print(f"{'motor':14s} {'homing':>7s} {'min':>6s} {'max':>6s}")
        for m in MOTORS:
            c = cal[m]
            print(f"{m:14s} {c.homing_offset:7d} {c.range_min:6d} {c.range_max:6d}")
        # wrist_roll is calibrated as a full-turn joint (0..4095) on purpose; the others should not be
        body = [m for m in MOTORS if m not in ("wrist_roll",)]
        factory = all(cal[m].range_min == 0 and cal[m].range_max == 4095 for m in body)
        if factory:
            print(
                "\n!! every servo reports the factory range 0..4095: this arm does not carry a LeRobot calibration.\n"
                "   Degrees below would be meaningless. Run `lerobot-calibrate` first (it moves nothing by itself;\n"
                "   you move the arm by hand)."
            )
        else:
            show_degrees(bus, cal)
            if args.watch > 0:
                end = time.time() + args.watch
                print(f"\nwatching for {args.watch:g} s (read-only); move the arm by hand:")
                while time.time() < end:
                    show_degrees(bus, cal, header=False)
                    time.sleep(0.2)
    finally:
        # NOT bus.disconnect(): its default switches torque off, which drops a holding arm.
        bus.disconnect(disable_torque=False)
    return 0


def scan(bus: SerialMotorsBus) -> int:
    """List the ids that answer a ping at each baud rate the servo model supports.

    A ping is a read-type instruction: no servo setting changes. Only the host port's baud rate
    does, and it is put back. (This uses LeRobot's own `_connect` / `_scan`, as its motor setup does.)
    """
    bus._connect(handshake=False)
    try:
        definition = bus._definition("shoulder_pan")
        rates = sorted(definition.baudrates, key=lambda r: r != bus.default_baudrate)
        print(f"servos expected: ids 1..{len(MOTORS)} at {bus.default_baudrate} baud\n")
        for rate in rates:
            bus.set_baudrate(rate)
            found = bus._scan(definition)
            note = "  <- the rate LeRobot uses" if rate == bus.default_baudrate else ""
            print(f"{rate:>9d} baud: ids {sorted(found)}{note}")
        bus.set_baudrate(bus.default_baudrate)
    finally:
        bus.disconnect(disable_torque=False)
    return 0


# Feetech's hardware-error bits (the `Status` register), as the SCS/STS documentation lists them.
FAULT_BITS = {0: "input voltage", 1: "angle sensor", 2: "overheat", 3: "over-current", 5: "overload"}


def status(bus: SerialMotorsBus) -> int:
    """For every servo that answers: fault flags, the voltage it measures, temperature, load.

    A servo whose LED is red or blinking has tripped a protection; this says which. Reads only.
    The voltage is the supply the servo actually sees, in 0.1 V units (74 = 7.4 V).
    """
    bus._connect(handshake=False)
    try:
        definition = bus._definition("shoulder_pan")
        found = bus._scan(definition)
        names = {m.id: n for n, m in bus.motors.items()}
        print(f"answering at {bus.default_baudrate} baud: ids {sorted(found)} (expected 1..{len(MOTORS)})\n")
        regs = ["Status", "Present_Voltage", "Present_Temperature", "Present_Current", "Present_Load", "Torque_Enable"]
        print(f"{'id':>2s} {'motor':14s} " + " ".join(f"{r.replace('Present_', ''):>11s}" for r in regs) + "   faults")
        for i in sorted(found):
            name = names.get(i)
            if name is None:
                print(f"{i:>2d} (not an SO-101 id)")
                continue
            vals = [try_read(bus, r, name) for r in regs]
            st = vals[0] if isinstance(vals[0], int) else 0
            faults = [label for bit, label in FAULT_BITS.items() if st & (1 << bit)]
            print(f"{i:>2d} {name:14s} " + " ".join(f"{str(v):>11s}" for v in vals) + "   " + (", ".join(faults) or "none flagged"))
        missing = [n for n, m in bus.motors.items() if m.id not in found]
        if missing:
            print(f"\nnot answering: {', '.join(missing)}")
    finally:
        bus.disconnect(disable_torque=False)
    return 0


def show_degrees(bus: SerialMotorsBus, cal, header: bool = True) -> None:
    bus.calibration = cal  # in memory only: lets the bus normalise; nothing is written to the servos
    deg = bus.sync_read("Present_Position")  # degrees for the joints, 0..100 for the gripper
    rad = {m: math.radians(deg[m]) for m in ARM}
    flag = {m: "  <-- outside the dexforge model's joint limit" for m in ARM if not (LIMITS[m][0] <= rad[m] <= LIMITS[m][1])}
    if header:
        print("\nposition as LeRobot normalises it, and in the radians the dexforge model uses:")
        print(f"{'motor':14s} {'deg':>9s} {'rad':>9s}")
        for m in ARM:
            print(f"{m:14s} {deg[m]:9.2f} {rad[m]:9.4f}{flag.get(m, '')}")
        print(f"{'gripper':14s} {deg['gripper']:9.2f}   (0..100, not modelled)")
    else:
        print("  " + "  ".join(f"{m[:8]}={deg[m]:7.2f}" for m in ARM))


if __name__ == "__main__":
    sys.exit(main())
