"""
Reading frames of a video file with the package `av` (PyAV).

Used for the scene video of head-mounted eye trackers (see
:class:`~pypillometry.HeadMountedEyeData`). The functions that call these
helpers are decorated with ``requires_package("av")``.
"""
import numpy as np


def _video_info(fname: str) -> dict:
    """Size (width, height) in pixels, frames per second and duration (s) of a video."""
    import av
    with av.open(str(fname)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = np.nan
        return dict(size=(stream.width, stream.height),
                    fps=float(stream.average_rate), duration=duration)


def _read_frame(fname: str, t: float) -> np.ndarray:
    """
    The frame shown at time `t` (seconds from the start of the video) as an
    RGB array (height x width x 3, uint8).
    """
    import av
    with av.open(str(fname)) as container:
        stream = container.streams.video[0]
        dt = 1. / float(stream.average_rate)
        # go to the keyframe before t and decode up to the frame shown at t
        container.seek(int(max(t, 0) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time + dt > t:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"no frame at {t:.3f} s in '{fname}'")


def _iter_gray_frames(fname: str, size: tuple):
    """
    Iterate over all frames of a video: tuples of the time of the frame
    (seconds from the start of the video) and the frame as a gray-scale
    array (height x width, uint8), scaled to `size` (width, height).
    """
    import av
    w, h = size
    with av.open(str(fname)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.time, frame.reformat(width=w, height=h, format="gray").to_ndarray()
