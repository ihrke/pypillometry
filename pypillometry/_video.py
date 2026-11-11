"""
Reading and writing frames of a video file with the package `av` (PyAV).

Used for the scene video of head-mounted eye trackers (see
:class:`~pypillometry.HeadMountedEyeData`). The functions that call these
helpers are decorated with ``requires_package("av")``.
"""
import numpy as np


def _threads(threads) -> int:
    """Check the argument `threads` of the calling methods (0: all cores)."""
    if isinstance(threads, bool) or int(threads) != threads or threads < 0:
        raise ValueError(f"threads must be 0 or a positive integer, got {threads!r}")
    return int(threads)


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


def _render(fname: str, out: str, t0: float, t1: float, draw, audio: bool = True,
            every: int = 1, threads: int = 0) -> int:
    """
    Write the frames of the video `fname` between `t0` and `t1` (seconds
    from the start of the video) to the video file `out` (H.264) after
    drawing on them: `draw(t, image)` is called with the time of the frame
    and the frame as a PIL image, which it changes. Only one of `every`
    frames is written (with a frame rate that is lower by this factor).
    The audio track is copied if `audio`. `threads` is the number of
    threads for decoding and for encoding (0: all cores). Returns the
    number of frames written.
    """
    import av
    from fractions import Fraction
    with av.open(str(fname)) as src, av.open(str(out), "w") as dst:
        vin = src.streams.video[0]
        vin.thread_type = "AUTO"
        vin.codec_context.thread_count = threads
        dt = 1. / float(vin.average_rate)
        rate = vin.average_rate / every
        vout = dst.add_stream("h264", rate=rate, options={"crf": "20", "preset": "veryfast"})
        vout.width, vout.height, vout.pix_fmt = vin.width, vin.height, "yuv420p"
        vout.thread_type = "AUTO"
        vout.codec_context.thread_count = threads
        streams = [vin]
        if audio and len(src.streams.audio) > 0:
            ain = src.streams.audio[0]
            aout = dst.add_stream_from_template(ain)
            streams.append(ain)
        src.seek(int(t0 * av.time_base), backward=True)
        # time of the first frame written (the frame shown at t0)
        first = np.floor(t0 / dt + 1e-6) * dt
        n, nread, done, t = 0, 0, False, -np.inf
        for packet in src.demux(streams):
            if packet.stream is vin:
                for frame in packet.decode():
                    if frame.time + dt <= t0 or done:
                        continue
                    if frame.time >= t1:
                        done = True
                        continue
                    nread += 1
                    if (nread - 1) % every != 0:
                        continue
                    img = frame.to_image()
                    draw(frame.time, img)
                    new = av.VideoFrame.from_image(img)
                    new.pts, new.time_base = n, Fraction(1, 1) / rate
                    dst.mux(vout.encode(new))
                    n += 1
            elif packet.dts is not None:
                t = float(packet.pts * packet.time_base)
                if first <= t < t1:
                    shift = int(round(first / packet.time_base))
                    packet.pts, packet.dts = packet.pts - shift, packet.dts - shift
                    packet.stream = aout
                    dst.mux(packet)
            if done and (len(streams) == 1 or t >= t1):
                break
        dst.mux(vout.encode())
    return n


def _iter_gray_frames(fname: str, size: tuple, threads: int = 0):
    """
    Iterate over all frames of a video: tuples of the time of the frame
    (seconds from the start of the video) and the frame as a gray-scale
    array (height x width, uint8), scaled to `size` (width, height).
    `threads` is the number of threads for decoding (0: all cores).
    """
    import av
    w, h = size
    with av.open(str(fname)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = threads
        for frame in container.decode(stream):
            yield frame.time, frame.reformat(width=w, height=h, format="gray").to_ndarray()
