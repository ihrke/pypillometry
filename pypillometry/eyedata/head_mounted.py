"""
head_mounted.py
===============

Data from head-mounted eye trackers (eye-tracking glasses, e.g. Tobii Pro
Glasses 3): :class:`HeadMountedEyeData` and its namespace ``d.scene``
(:class:`SceneNamespace`) for the scene video and the motion sensors.
"""
import inspect
import os
from typing import Optional

import numpy as np
from loguru import logger

from .. import io, _video
from ..convenience import requires_package
from ..events import Events
from ..logging import _show_progress
from ..plot import HeadMountedPlotNamespace
from ..units import parse_time
from .eyedata import EyeData
from .eyedatadict import EyeDataDict
from .history import keephistory
from .namespace import Namespace, accessor
from .pipes import step

#: width (pixels) to which the scene video is scaled for the luminance
_LUMINANCE_WIDTH = 320
#: duration (ms) of the pupil trace shown in a rendered video
_PUPIL_WINDOW = 10000.
#: colors (RGB) of the pupil traces in a rendered video
_PUPIL_COLORS = ((107, 174, 214), (253, 141, 60), (161, 217, 155))


class SceneNamespace(Namespace):
    """
    Methods for the scene video and the motion sensors of a head-mounted
    eye tracker, accessed as ``d.scene`` (e.g., ``d.scene.luminance()``) on
    a :class:`~pypillometry.HeadMountedEyeData` object.

    The transforming methods store their result as channels in ``d.aux``
    and return the data object (see
    :class:`~pypillometry.eyedata.namespace.Namespace`).
    """

    _name = "scene"

    @requires_package("av")
    def frame(self, time, units: str = "ms") -> np.ndarray:
        """
        The frame of the scene video that was shown at a time of the recording.

        Requires the optional package `av`.

        Parameters
        ----------
        time : float, str or Quantity
            time on the clock of the data (as in ``d.tx``), e.g. ``12000``,
            ``"12 sec"`` or ``12 * u.sec``
        units : str, optional
            units of `time` if it is a plain number; default "ms"

        Returns
        -------
        np.ndarray
            the frame as an RGB image (height x width x 3, uint8)

        Raises
        ------
        ValueError
            if the object has no scene video or the video has no frame at
            `time`

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> img = d.scene.frame("12 sec")  # doctest: +SKIP
        """
        obj = self._d
        t = float(parse_time(time, to="ms", default=units))
        return _video._read_frame(obj._scene_video_file(),
                                  (t - obj.scene_video_offset) / 1000.)

    @requires_package("av")
    def render_video(self, fname: str, plot_range: tuple = (-np.inf, +np.inf),
                     trail=500, radius: float = 30, color="red",
                     pupil: bool = True, pupil_alpha: float = 0.3,
                     audio: bool = True, fps: Optional[float] = None,
                     threads: Optional[int] = None, units: str = "sec") -> str:
        """
        Write the scene video with the gaze position drawn on it.

        Each frame gets a ring at the gaze position (``both_x``,
        ``both_y``; the mean of the usable samples during the frame) and a
        fading line through the positions before it. Frames without a
        usable gaze position get no ring. A strip at the bottom of the
        frame shows the pupil size of each eye (usable samples) from 5
        seconds before to 5 seconds after the frame; the white line in
        its middle is the time of the frame. The scale of the strip (the
        numbers at its right) is the same in all frames and leaves out
        the most extreme 1% of the values at each end. The video is useful for
        checking the data and the timing of the video relative to it
        (`scene_video_offset`).

        Requires the optional package `av`. All frames are decoded and
        encoded again (H.264); a progress bar counts the frames. With
        the default settings, this takes about as long as the part of
        the video that is written. For a fast preview, write fewer frames
        (`fps`). An existing file `fname` is overwritten.

        Parameters
        ----------
        fname : str
            name of the video file to write (e.g., ``"gaze.mp4"``)
        plot_range : tuple
            The time range to write, on the clock of the data (plain
            numbers are in `units`, or with units, e.g. ``(3, 5) * u.min``).
            Default is (-np.inf, +np.inf), i.e. all data. Only the part of
            the video for which the object has samples is written.
        trail : float, str or Quantity, optional
            duration of the line behind the gaze position, e.g. ``500``
            (ms) or ``"1 sec"``; default 500 ms. Use 0 for no line.
        radius : float, optional
            radius of the ring in pixels of the scene video; default 30
        color : color, optional
            color of the ring and the line (any matplotlib color);
            default "red"
        pupil : bool, optional
            if True (default), draw the strip with the pupil size (if the
            object has pupil data); it covers the lower fifth of the frame
        pupil_alpha : float, optional
            opacity of the dark background of the strip, from 0 (no
            background, the traces are drawn on the scene) to 1 (black);
            default 0.3
        audio : bool, optional
            if True (default), copy the sound of the scene video (if any)
        fps : float, optional
            frames per second of the written video. Default: as the scene
            video. With a lower value, only every n-th frame is written
            (e.g., every fifth frame for ``fps=5`` and a scene video with
            25 frames per second), which is faster by about this factor.
        threads : int, optional
            number of processor cores used for reading and for writing
            the video. Default (None): half of the cores, so that the
            computer stays usable; 0: all cores (fastest).
        units : str
            The units of plain numbers in `plot_range` ("ms", "sec",
            "min", "h"). Default "sec".

        Returns
        -------
        str
            `fname`

        Raises
        ------
        ValueError
            if the object has no scene video or no gaze position,
            `plot_range` does not overlap with the data and the video, or
            `fps` is not positive

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> d.scene.render_video("gaze.mp4", (10, 30))  # doctest: +SKIP
        >>> d.scene.render_video("preview.mp4", fps=5, threads=2)  # doctest: +SKIP
        """
        from matplotlib.colors import to_rgb
        from PIL import ImageDraw, ImageFont
        from tqdm import tqdm
        obj = self._d
        if "both_x" not in obj.data or "both_y" not in obj.data:
            raise ValueError("the object has no gaze position (both_x, both_y)")
        video = obj._scene_video_file()
        info = _video._video_info(video)
        threads = _video._threads(threads)
        if fps is not None and not fps > 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        every = 1 if fps is None else max(1, int(round(info["fps"] / fps)))
        start, end = (float(v) for v in parse_time(plot_range, to="ms", default=units))
        # only the part with samples (e.g., after cropping)
        start, end = max(start, obj.tx[0]), min(end, obj.tx[-1] + 1000. / obj.fs)
        t0 = max((start - obj.scene_video_offset) / 1000., 0.)
        t1 = (end - obj.scene_video_offset) / 1000.
        if not np.isnan(info["duration"]):
            t1 = min(t1, info["duration"])
        if not t0 < t1:
            raise ValueError(f"plot_range {tuple(plot_range)} ({units}) does not "
                             "overlap with the data and the scene video")

        # gaze position in pixels of the video, NaN where not usable
        sx, sy = (v / s for v, s in zip(info["size"], obj.scene_size or info["size"]))
        usable = ~obj.mask("both", "x") & ~obj.mask("both", "y")
        gx = np.where(usable, obj.data["both_x"], np.nan) * sx
        gy = np.where(usable, obj.data["both_y"], np.nan) * sy
        rgb = tuple(int(round(255 * v)) for v in to_rgb(color))
        ntrail = float(parse_time(trail, to="ms", default="ms"))
        dt = 1000. / info["fps"]
        # a frame shows the samples during it (at least the latest sample)
        current = max(dt, 1000. / obj.fs)
        width = max(2, int(round(radius / 6.)))

        # strip with the pupil size: the lower fifth of the frame
        W, H = info["size"]
        top, pad, lw = H - H // 5, max(1, H // 100), max(1, H // 400)
        traces = {}  # eye -> y-position (pixels) of the pupil size per sample
        background = int(round(255 * min(max(pupil_alpha, 0.), 1.)))
        if pupil:
            sizes = {eye: np.where(obj.mask(eye, "pupil"), np.nan, obj.data[f"{eye}_pupil"])
                     for eye in obj.eyes if f"{eye}_pupil" in obj.data}
            shown = (obj.tx >= start) & (obj.tx <= end)
            vals = np.concatenate([v[shown] for v in sizes.values()] or [[np.nan]])
            if not np.isnan(vals).all():
                # range without the extreme values (artifacts), which are cut off
                lo, hi = np.nanpercentile(vals, [1, 99])
                lo, hi = lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)
                hi = hi if hi > lo else lo + 1.
                traces = {eye: np.clip((H - pad) - (v - lo) / (hi - lo) * (H - top - 2 * pad),
                                       top, H) for eye, v in sizes.items()}
                try:
                    font = ImageFont.load_default(size=max(10, H // 45))
                except TypeError:  # Pillow < 10.1: one size only
                    font = ImageFont.load_default()

        def draw_pupil(canvas, now):
            canvas.rectangle([0, top, W, H], fill=(0, 0, 0, background))
            a, b = np.searchsorted(obj.tx, [now - _PUPIL_WINDOW / 2., now + _PUPIL_WINDOW / 2.])
            x = W / 2. + (obj.tx[a:b] - now) / _PUPIL_WINDOW * W
            for i, (eye, ypos) in enumerate(traces.items()):
                col = _PUPIL_COLORS[i % len(_PUPIL_COLORS)]
                y = ypos[a:b]
                ok = ~np.isnan(y)
                # one line per run of usable samples
                edges = np.flatnonzero(np.diff(np.concatenate([[False], ok, [False]]).astype(int)))
                for r0, r1 in zip(edges[::2], edges[1::2]):
                    if r1 - r0 > 1:
                        canvas.line(list(zip(x[r0:r1], y[r0:r1])), fill=col + (255,), width=lw)
                if H >= 200:
                    canvas.text((pad * 2, top + pad + i * font.size * 1.2), f"{eye} pupil",
                                fill=col + (255,), font=font)
            if H >= 200:
                for val, y in ((hi, top + pad), (lo, H - pad - font.size * 1.2)):
                    canvas.text((W - pad * 2, y), f"{val:.2f}", fill=(255, 255, 255, 255),
                                font=font, anchor="ra")
            canvas.line([(W / 2., top), (W / 2., H)], fill=(255, 255, 255, 255), width=lw)

        def draw(t, img):
            now = t * 1000. + obj.scene_video_offset + dt
            j0, i0, i1 = np.searchsorted(obj.tx, [now - dt - ntrail, now - current, now])
            canvas = ImageDraw.Draw(img, "RGBA")
            if traces:
                draw_pupil(canvas, now - dt / 2.)
            for k in range(j0, i1 - 1):
                if usable[k] and usable[k + 1]:
                    alpha = int(200 * (k + 1 - j0) / max(i1 - 1 - j0, 1))
                    canvas.line([(gx[k], gy[k]), (gx[k + 1], gy[k + 1])],
                                fill=rgb + (alpha,), width=width)
            if usable[i0:i1].any():
                x, y = np.nanmean(gx[i0:i1]), np.nanmean(gy[i0:i1])
                canvas.ellipse([x - radius, y - radius, x + radius, y + radius],
                               outline=rgb + (255,), width=width)
            bar.update(1)

        logger.info(f"Writing '{fname}' ({t1 - t0:.1f} s of '{video}')")
        nframes = int(np.ceil(t1 * info["fps"] - 1e-6) - np.floor(t0 * info["fps"] + 1e-6))
        with tqdm(desc=os.path.basename(str(fname)), total=-(-nframes // every),
                  unit="frames", disable=not _show_progress()) as bar:
            _video._render(video, fname, t0, t1, draw, audio=audio, every=every,
                           threads=threads)
        return str(fname)

    @step
    @keephistory
    @requires_package("av")
    def luminance(self, radius: float = 100, gamma: float = 2.2,
                  threads: Optional[int] = None, inplace=None):
        """
        Calculate the luminance of the scene video over time.

        Light is the strongest driver of pupil size, and outside the lab it
        changes with every movement of the head and the eyes. This method
        reads all frames of the scene video and stores two channels in
        ``d.aux`` that can be used as covariates of the pupil size:

        - ``"scene_luminance"``: mean luminance of the whole frame
        - ``"gaze_luminance"``: mean luminance in a circle around the gaze
          position (``both_x``, ``both_y``); NaN where the gaze position
          is not usable or outside the frame

        Each sample gets the value of the frame that was shown at its time
        (NaN outside the video). The values are between 0 (black) and 1
        (white). They are relative: the scene camera adapts its exposure
        to the light, so that the same value can stand for different
        amounts of light at different times.

        Requires the optional package `av`. All frames are decoded, which
        takes about a tenth of the duration of the video.

        Parameters
        ----------
        radius : float, optional
            radius of the circle around the gaze position in pixels of the
            scene video; default 100 (about 5 degrees for Tobii Pro Glasses 3)
        gamma : float, optional
            the pixel values (0 to 1) are raised to this power before
            averaging, which undoes the gamma encoding of the video
            (approximately, values proportional to the light); default 2.2.
            Use 1 for the pixel values as they are.
        threads : int, optional
            number of processor cores used for reading the video. Default
            (None): half of the cores, so that the computer stays usable;
            0: all cores (fastest).
        inplace : bool or None
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object-level setting.

        Returns
        -------
        :class:`~pypillometry.HeadMountedEyeData`
            the object with the new channels in `aux` (`self` or a copy,
            depending on `inplace`)

        Raises
        ------
        ValueError
            if the object has no scene video

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> d = d.scene.luminance()  # doctest: +SKIP
        >>> d.aux["gaze_luminance"]  # doctest: +SKIP
        """
        obj = self._d._get_inplace(inplace)
        fname = obj._scene_video_file()
        info = _video._video_info(fname)
        width, height = info["size"]
        scale = _LUMINANCE_WIDTH / width
        size = (_LUMINANCE_WIDTH, max(2, int(round(height * scale / 2)) * 2))
        lut = (np.arange(256) / 255.) ** gamma

        has_gaze = "both_x" in obj.data and "both_y" in obj.data
        if has_gaze:
            usable = ~obj.mask("both", "x") & ~obj.mask("both", "y")
            gx, gy = obj.data["both_x"] * scale, obj.data["both_y"] * scale
        r = radius * scale
        yy, xx = np.mgrid[:size[1], :size[0]]
        dt = 1000. / info["fps"]

        scene = np.full(len(obj), np.nan)
        local = np.full(len(obj), np.nan)
        logger.info(f"Reading the frames of '{fname}'")
        for t, img in _video._iter_gray_frames(fname, size, _video._threads(threads)):
            # the samples during which the frame was shown
            start = t * 1000. + obj.scene_video_offset
            i0, i1 = np.searchsorted(obj.tx, [start, start + dt])
            if i0 == i1:
                continue
            lum = lut[img]
            scene[i0:i1] = lum.mean()
            ok = usable[i0:i1] if has_gaze else np.zeros(i1 - i0, dtype=bool)
            if ok.any():
                x, y = gx[i0:i1][ok].mean(), gy[i0:i1][ok].mean()
                circle = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
                if circle.any():
                    local[i0:i1] = lum[circle].mean()
        obj._set_aux("scene_luminance", scene)
        obj._set_aux("gaze_luminance", local)
        return obj

    @step
    @keephistory
    def distance(self, window=500, inplace=None):
        """
        Calculate the viewing distance from the 3D gaze point.

        The eye tracker calculates the point in space at which the lines of
        sight of the two eyes meet (``gaze3d_x``, ``gaze3d_y``,
        ``gaze3d_z`` in ``d.aux``, in mm from the scene camera). Its
        distance tells how far away the person looks, which matters for
        the pupil size because the pupil constricts when looking at near
        objects (near response). The distance of single samples is very
        noisy for far objects (the lines of sight are almost parallel), so
        it is smoothed with a running median of the vergence (1/distance).

        The result is stored as ``"gaze_distance"`` (in mm) in ``d.aux``;
        NaN where there is no 3D gaze point.

        Parameters
        ----------
        window : float, str or Quantity, optional
            width of the running median, e.g. ``500`` (ms), ``"1 sec"``;
            default 500 ms. Use 0 for the distance of the single samples.
        inplace : bool or None
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object-level setting.

        Returns
        -------
        :class:`~pypillometry.HeadMountedEyeData`
            the object with the new channel in `aux` (`self` or a copy,
            depending on `inplace`)

        Raises
        ------
        ValueError
            if the object has no 3D gaze point

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> d = d.scene.distance(window="1 sec")  # doctest: +SKIP
        """
        import pandas as pd
        obj = self._d._get_inplace(inplace)
        obj._require_aux("gaze3d_x", "gaze3d_y", "gaze3d_z")
        dist = np.sqrt(sum(obj.aux[f"gaze3d_{ax}"] ** 2 for ax in "xyz"))
        dist[dist == 0] = np.nan
        nwin = obj._samples(window, rounding="nearest")
        if nwin > 1:
            vergence = pd.Series(1. / dist).rolling(nwin, center=True, min_periods=1).median()
            dist = np.where(np.isnan(dist), np.nan, 1. / vergence.to_numpy())
        obj._set_aux("gaze_distance", dist)
        return obj

    @step
    @keephistory
    def head_movement(self, inplace=None):
        """
        Calculate the speed of the head rotation from the gyroscope.

        The result (the length of the vector ``gyro_x``, ``gyro_y``,
        ``gyro_z``, in degrees per second) is stored as ``"head_speed"``
        in ``d.aux``. It can be used as a covariate or to find periods in
        which the head moved.

        Parameters
        ----------
        inplace : bool or None
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object-level setting.

        Returns
        -------
        :class:`~pypillometry.HeadMountedEyeData`
            the object with the new channel in `aux` (`self` or a copy,
            depending on `inplace`)

        Raises
        ------
        ValueError
            if the object has no gyroscope data

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> d = d.scene.head_movement()  # doctest: +SKIP
        """
        obj = self._d._get_inplace(inplace)
        obj._require_aux("gyro_x", "gyro_y", "gyro_z")
        obj._set_aux("head_speed", np.sqrt(sum(obj.aux[f"gyro_{ax}"] ** 2 for ax in "xyz")))
        return obj


class HeadMountedEyeData(EyeData):
    """
    Data from a head-mounted eye tracker (eye-tracking glasses).

    An :class:`~pypillometry.EyeData` object (all its methods work) with
    the additional data of such trackers:

    - the gaze position is the point at which the person looks in the
      video of the scene camera: ``both_x`` and ``both_y`` in pixels of the
      video (one position for both eyes, y pointing down)
    - `aux`: channels that do not belong to an eye, with one value per
      sample (e.g., the motion sensors ``gyro_x``, ``acc_x``, ``mag_x``).
      They are cut, downsampled and saved together with the data.
    - the scene video: the object stores the name of the video file
      (`scene_video`), not the video itself
    - the namespace ``d.scene`` (:class:`SceneNamespace`) with methods for
      the scene video and the motion sensors, and the plot
      ``d.plot.plot_scene_scanpath()``

    The camera moves with the head, so there is no screen: an experimental
    setup and the methods that need one (e.g., the correction of the
    foreshortening) do not apply.

    Parameters
    ----------
    time : np.ndarray or None
        timing array (in ms), see :class:`~pypillometry.EyeData`
    left_x, left_y, left_pupil, right_x, right_y, right_pupil : np.ndarray or None
        data of the left and right eye (see :class:`~pypillometry.EyeData`);
        at least one of these channels must be provided
    both_x, both_y : np.ndarray or None
        gaze position in the scene video (pixels); both or neither
    aux : dict or None
        additional channels with one value per sample: name -> array.
        The names have two parts joined by ``_`` (e.g., ``"gyro_x"``).
    scene_video : str or None
        filename of the scene video
    scene_video_offset : float
        time of the first frame of the scene video on the clock of the
        data (ms); default 0
    scene_size : tuple or None
        width and height of the scene video in pixels
    **kwargs
        other parameters of :class:`~pypillometry.EyeData` (e.g.,
        `sampling_rate`, `events`, `name`, `info`)

    Examples
    --------
    >>> import numpy as np
    >>> import pypillometry as pp
    >>> rng = np.random.default_rng(1)
    >>> d = pp.HeadMountedEyeData(time=np.arange(0, 1000, 10),
    ...                           left_pupil=rng.normal(5, 0.1, size=100),
    ...                           both_x=rng.uniform(0, 1920, size=100),
    ...                           both_y=rng.uniform(0, 1080, size=100),
    ...                           aux={"gyro_x": rng.normal(size=100)},
    ...                           scene_size=(1920, 1080))
    >>> d.eyes, list(d.aux)
    (['left', 'both'], ['gyro_x'])
    """
    scene = accessor(SceneNamespace)
    plot = accessor(HeadMountedPlotNamespace)

    aux: EyeDataDict  ## channels that do not belong to an eye
    scene_video: Optional[str]  ## filename of the scene video
    scene_video_offset: float  ## time of the first video frame (ms)
    scene_size: Optional[tuple]  ## (width, height) of the scene video (pixels)

    # state (see pypillometry.state): adds to that of EyeData
    _STATE_FIELDS = (("aux", "EyeDataDict"), "scene_video", "scene_video_offset",
                     "scene_size")
    # the additional channels follow crop, downsample etc. with the data
    _SAMPLE_ALIGNED = ("data", "aux")

    def __init__(self, time=None,
                 left_x=None, left_y=None, left_pupil=None,
                 right_x=None, right_y=None, right_pupil=None,
                 both_x=None, both_y=None, aux: Optional[dict] = None,
                 scene_video: Optional[str] = None, scene_video_offset: float = 0.,
                 scene_size: Optional[tuple] = None,
                 fill_time_discontinuities: bool = True, keep_orig: bool = False,
                 **kwargs):
        """Constructor for the HeadMountedEyeData class."""
        if (both_x is None) != (both_y is None):
            raise ValueError("both_x and both_y must be provided together")
        self.aux = EyeDataDict()
        self.scene_video = None if scene_video is None else str(scene_video)
        self.scene_video_offset = float(scene_video_offset)
        self.scene_size = None if scene_size is None else tuple(int(v) for v in scene_size)
        # gaps are filled in after all channels are there
        super().__init__(time=time, left_x=left_x, left_y=left_y, left_pupil=left_pupil,
                         right_x=right_x, right_y=right_y, right_pupil=right_pupil,
                         fill_time_discontinuities=False, keep_orig=False, **kwargs)
        if both_x is not None:
            self.data["both_x"] = both_x
            self.data["both_y"] = both_y
        for key, values in (aux or {}).items():
            self._set_aux(key, values)
        if fill_time_discontinuities:
            inspect.unwrap(type(self).fill_time_discontinuities)(self, inplace=True)
        if keep_orig:
            self.original = self.copy()

    def _set_aux(self, key: str, values):
        """Store `values` (one per sample) as channel `key` in `aux`."""
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self),):
            raise ValueError(f"aux channel '{key}' must have one value per sample "
                             f"({len(self)}), got shape {values.shape}")
        self.aux[key] = values

    def _require_aux(self, *keys):
        """Raise a ValueError if one of the channels `keys` is not in `aux`."""
        missing = [k for k in keys if k not in self.aux]
        if missing:
            raise ValueError(f"no channel(s) {', '.join(missing)} in aux "
                             f"(available: {', '.join(self.aux.keys()) or 'none'})")

    def _scene_video_file(self) -> str:
        """Filename of the scene video (ValueError if there is none)."""
        if self.scene_video is None:
            raise ValueError("the object has no scene video: set the attribute "
                             "`scene_video` to the name of the video file")
        if not os.path.exists(self.scene_video):
            raise ValueError(f"scene video '{self.scene_video}' not found: set the "
                             "attribute `scene_video` to the name of the video file")
        return self.scene_video

    def __repr__(self):
        """String-representation with the additional channels and the scene video."""
        scene = (f" Scene: video={self.scene_video}, size={self.scene_size}, "
                 f"aux={list(self.aux.keys())}\n")
        head, sep, history = super().__repr__().partition(" History:")
        return head + scene + sep + history

    @classmethod
    def from_tobii_prolab(cls, fname: str, video: Optional[str] = None,
                          video_offset: Optional[float] = None,
                          remove_sync_events: bool = True, **kwargs):
        """
        Reads a recording of Tobii Pro Glasses from a data export of Tobii Pro Lab.

        The export is read with :func:`pypillometry.io.read_tobii_prolab`
        (Excel files require the optional package `openpyxl`). The object
        holds

        - ``left_pupil``, ``right_pupil``: pupil diameter in mm
        - ``both_x``, ``both_y``: gaze position in the scene video (pixels)
        - in `aux`: the 3D gaze point in mm from the scene camera
          (``gaze3d_x``, ``gaze3d_y``, ``gaze3d_z``) and, interpolated to
          the times of the eye-tracker samples, the gyroscope
          (``gyro_x``, ..., deg/s), the accelerometer (``acc_x``, ...,
          m/s²) and the magnetometer (``mag_x``, ..., µT)
        - the events of the export
        - the description of the recording (participant, date, ...) in `info`

        Samples without valid data are empty in the export; they are NaN
        (and therefore masked). Gaps in the samples are filled in.

        Parameters
        ----------
        fname : str
            filename of the export (``.xlsx`` or tab-separated text)
        video : str, optional
            filename of the scene video; default: the file with the name
            of the export and the extension ``.mp4``, if it exists
        video_offset : float, optional
            time of the first frame of the video on the clock of the data
            (ms); default: the first timestamp of the export (the video
            that Tobii Pro Lab exports covers the same period as the data)
        remove_sync_events : bool, optional
            if True (default), remove the events of the sync port
            (``SyncPortOutHigh``, ``SyncPortInLow``, ...) from the events
        **kwargs
            additional arguments passed to the constructor (e.g., `name`,
            `keep_orig`)

        Returns
        -------
        :class:`~pypillometry.HeadMountedEyeData`
            object of the class this method is called on

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> d.plot.plot_scene_scanpath((10, 12))  # doctest: +SKIP
        """
        tabs = io.read_tobii_prolab(fname)
        eye = tabs["eye"].drop_duplicates("time").sort_values("time")
        if len(eye) < 2:
            raise ValueError(f"'{fname}' has no samples of the eye tracker")
        tx = eye["time"].to_numpy(dtype=float)

        def col(name):
            return eye[name].to_numpy(dtype=float) if name in eye.columns else None

        info = dict(tabs["info"], source=str(fname), format="Tobii Pro Lab data export")
        first = min([tx[0]] + [float(tabs[k]["time"].min()) for k in ("gyro", "acc", "mag", "events")
                               if len(tabs[k]) > 0])
        if video is None:
            guess = os.path.splitext(str(fname))[0] + ".mp4"
            video = guess if os.path.exists(guess) else None
        size = None
        if "media_width" in info and "media_height" in info:
            size = (info["media_width"], info["media_height"])

        events = tabs["events"]
        if remove_sync_events:
            events = events[~events["label"].astype(str).str.startswith("SyncPort")]
        # events before the first or after the last sample (e.g., the start
        # of the recording) are moved to that sample
        onsets = np.clip(events["time"].to_numpy(dtype=float), tx[0], tx[-1])

        aux = {f"gaze3d_{ax}": col(f"Gaze point 3D {ax.upper()}") for ax in "xyz"}
        kwargs.setdefault("name", info.get("recording"))
        obj = cls(time=tx,
                  left_pupil=col("Pupil diameter left"),
                  right_pupil=col("Pupil diameter right"),
                  both_x=col("Gaze point X"), both_y=col("Gaze point Y"),
                  aux={k: v for k, v in aux.items() if v is not None},
                  events=Events(onsets, events["label"].to_numpy(dtype=str), units="ms"),
                  scene_video=video,
                  scene_video_offset=first if video_offset is None else video_offset,
                  scene_size=size, info=info, **kwargs)
        # motion sensors: own sampling rates, interpolated to the samples
        for key in ("gyro", "acc", "mag"):
            imu = tabs[key].dropna()
            if len(imu) < 2:
                continue
            t = imu["time"].to_numpy(dtype=float)
            for ax in "xyz":
                obj._set_aux(f"{key}_{ax}", np.interp(obj.tx, t, imu[ax].to_numpy(dtype=float),
                                                      left=np.nan, right=np.nan))
        return obj
