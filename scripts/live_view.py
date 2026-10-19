"""Live view for two HiSeq 2500 cameras using pyseq2500.

Controls (click the image window first):
    space   pause / resume acquisition
    s       save a snapshot (PNG, raw 16-bit)
    q / Esc quit
Ctrl+C in the terminal also shuts down cleanly.

Design:
    * All DCAM calls are blocking, so they run on ONE dedicated worker thread
      (DCAM handles should not hop between threads) via ``run_cam``.
    * Frame waiting is polled with short ``await asyncio.sleep`` calls instead
      of a blocking busy loop, so pause/stop take effect within ~ms.
    * cv2 windows are only touched from the main thread (display task).
    * Cameras are always returned to TDI mode in ``finally``.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyseq

WINDOW = "DCAM Live (space=pause, s=snapshot, q=quit)"
LIVE_BUNDLE_HEIGHT = 64
TDI_BUNDLE_HEIGHT = 128
FRAME_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.005
ARM_RETRIES = 10

# Single worker thread: serialises DCAM access and keeps thread affinity.
_cam_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dcam")


async def run_cam(fn, *args):
    """Run a blocking camera call on the DCAM thread without blocking asyncio."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_cam_executor, fn, *args)


# ---------------------------------------------------------------- camera ops
def configure_live(cam):
    if cam.sensor_mode != "AREA":
        cam.setAREA()
    cam.setPropertyValue("sensor_mode_line_bundle_height", LIVE_BUNDLE_HEIGHT)
    cam.setTriggerMode("internal")


def restore_tdi(cam):
    cam.setTriggerMode("TDI")
    if cam.sensor_mode != "TDI":
        cam.setTDI()
    cam.setPropertyValue("sensor_mode_line_bundle_height", TDI_BUNDLE_HEIGHT)


def arm(cam):
    """Stop, reset and re-allocate so the camera is ready to acquire one frame."""
    cam.stopAcquisition()
    cam.freeFrames()
    for _ in range(ARM_RETRIES):
        cam.captureSetup()
        if cam.get_status() == 3:
            break
        cam.stopAcquisition()
        cam.freeFrames()
    else:
        raise RuntimeError("Camera failed to reach capture-ready status (3)")
    cam.allocFrame(1)
    cam.startAcquisition()


def halt(cam):
    """Best-effort stop; never raises."""
    try:
        cam.stopAcquisition()
        cam.freeFrames()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: halt failed: {exc}")


def frame_counts(cams):
    return [c.getFrameCount() for c in cams]


def fetch_frame(cam):
    frame, (fx, fy) = cam.getFrames()
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim == 1:  # flat buffer -> image (assumes fx wide, fy tall)
        frame = frame.reshape(fy, fx)
    return frame


# ------------------------------------------------------------- async tasks
async def acquire(cams, frames_q, run_evt, stop_evt):
    """Acquire frames from both cameras; honours pause (run_evt) and stop."""
    armed = False
    while not stop_evt.is_set():
        if not run_evt.is_set():
            if armed:  # paused: stop the hardware once
                await asyncio.gather(*(run_cam(halt, c) for c in cams))
                armed = False
            await asyncio.sleep(0.02)
            continue

        await asyncio.gather(*(run_cam(arm, c) for c in cams))
        armed = True

        t0 = time.monotonic()
        while not stop_evt.is_set() and run_evt.is_set():
            counts = await asyncio.gather(*(run_cam(c.getFrameCount) for c in cams))
            if all(n >= 1 for n in counts):
                break
            if time.monotonic() - t0 > FRAME_TIMEOUT_S:
                print("Imaging took too long; re-arming cameras.")
                break
            await asyncio.sleep(POLL_INTERVAL_S)
        else:
            continue  # stopped or paused mid-wait; top of loop handles it

        if all(n >= 1 for n in counts):
            frames = await asyncio.gather(*(run_cam(fetch_frame, c) for c in cams))
            if all(f is not None for f in frames):
                frame = np.hstack(frames)
                # Keep only the newest frame so display never lags behind.
                if frames_q.full():
                    frames_q.get_nowait()
                frames_q.put_nowait(frame)

    await asyncio.gather(*(run_cam(halt, c) for c in cams))


def to_8bit(frame):
    lo, hi = np.percentile(frame, (1, 99.8))
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


async def display(frames_q, run_evt, stop_evt):
    """Show frames and handle keys. Runs on the main thread."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    last_raw = None
    fps, t_prev = 0.0, time.monotonic()

    while not stop_evt.is_set():
        try:
            last_raw = frames_q.get_nowait()
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
        except asyncio.QueueEmpty:
            pass

        if last_raw is not None:
            img = cv2.cvtColor(to_8bit(last_raw), cv2.COLOR_GRAY2BGR)
            status = "PAUSED" if not run_evt.is_set() else f"{fps:4.1f} fps"
            cv2.putText(
                img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            cv2.imshow(WINDOW, img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            stop_evt.set()
        elif key == ord(" "):
            run_evt.clear() if run_evt.is_set() else run_evt.set()
        elif key == ord("s") and last_raw is not None:
            name = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, last_raw.astype(np.uint16))
            print(f"Saved {name}")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            stop_evt.set()  # window closed with the X button

        await asyncio.sleep(0.01)


async def main(cams):
    run_evt = asyncio.Event()
    run_evt.set()
    stop_evt = asyncio.Event()
    frames_q: asyncio.Queue = asyncio.Queue(maxsize=1)

    for cam in cams:
        await run_cam(configure_live, cam)

    acq = asyncio.create_task(acquire(cams, frames_q, run_evt, stop_evt))
    try:
        await display(frames_q, run_evt, stop_evt)
    finally:
        stop_evt.set()
        await acq  # waits for cameras to be halted
        cv2.destroyAllWindows()
        for cam in cams:
            try:
                await run_cam(restore_tdi, cam)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: could not restore TDI mode: {exc}")


if __name__ == "__main__":
    hs = pyseq.HiSeq()
    hs.initializeCams()
    try:
        asyncio.run(main([hs.cams[0], hs.cams[1]]))
    except KeyboardInterrupt:
        pass  # asyncio.run cancels main(); its finally block restores cameras