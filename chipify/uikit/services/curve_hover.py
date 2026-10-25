# Copyright (c) 2026 Santiago Hofwimmer
"""
curve_hover.py – Hover tooltip for the per-run waveform overlays.

The Plots tab and the dashboard's ``Plots`` cell draw one curve per run; this
puts the scatter plot's tooltip on those curves, so pointing at an outlier
answers "which run is that?" without cross-referencing the legend. The bubble
itself is the shared :class:`~chipify.uikit.services.hover_bubble.HoverBubble`,
so the two surfaces look identical.

The input is the ``line_map`` (``Line2D -> (run_id, signal)``) that
``PlotManager.draw_transient_plot`` / ``draw_dc_sweep`` / ``draw_bode_plot``
already return, plus the results frame the overlay was drawn from.

Two things make this different from the scatter hover, and both come from the
same place — an overlay may hold ``transient_loader.RUN_CAP`` (500) runs of
several thousand samples each:

- **Hit testing.** Looping ``Line2D.contains(event)`` re-transforms every
  vertex of every curve on every mouse move (~150 ms at that size) and answers
  "some curve is near" rather than "this one is nearest", which is the useful
  answer when curves overlap. Instead each axes gets a lazily built index of
  its curves in *scale space* (``log10`` applied on a log axis, identity
  otherwise). Only the remaining ``transLimits + transAxes`` affine is re-read
  per event, so pan, zoom and resize need no rebuild, and the nearest curve is
  found with one vectorised point-to-segment pass in **pixels** — the metric
  the user is actually pointing with.
- **Repainting.** ``canvas.draw_idle()`` per motion event, which is what the
  scatter hover can afford, re-renders the whole overlay (~1 s at 500 curves).
  The bubble is blitted over a saved background instead, with a one-way
  fallback to ``draw_idle()`` if a canvas cannot blit.

No GUI-toolkit imports — see CONTRIBUTING, "Architecture conventions".
"""
from __future__ import annotations

import logging

import numpy as np

from chipify.uikit.services.hover_bubble import SEPARATOR, HoverBubble, fmt_value

log = logging.getLogger("chipify.uikit.curve_hover")

#: How close (in pixels) the cursor must come to a curve to select it.
HIT_RADIUS_PX = 8.0

#: How much thicker the hovered curve is drawn, and the floor that keeps a
#: hairline curve (an overlay draws at 0.8) visibly picked out.
HIGHLIGHT_WIDTH_FACTOR = 2.5
HIGHLIGHT_MIN_WIDTH = 2.0

#: Chunks the segments of one axes are grouped into for the coarse pass of a
#: hit test. A 500-run transient overlay is millions of segments; each motion
#: event tests at most this many chunk bounding boxes, then measures only the
#: segments of the chunks near the cursor. The index used to *stride* through
#: the samples instead, and the decimated polylines cut corners: at 500 runs
#: the hover named the wrong run for 476 of 600 cursors placed on a curve.
MAX_INDEX_CHUNKS = 120_000

#: Fallback when the hovered axes carries no x label of its own.
_X_FALLBACK = "x"


def x_axis_label(ax) -> str:
    """The x-axis name to print in a tooltip on *ax*.

    The Bode figure labels only its lower (phase) axes — the magnitude axes is
    its ``sharex`` partner and returns ``""`` — so fall back to whichever
    sibling does carry a label rather than printing a bare value.
    """
    label = ax.get_xlabel()
    if label:
        return label
    try:
        for sib in ax.get_shared_x_axes().get_siblings(ax):
            if sib is not ax and sib.get_xlabel():
                return sib.get_xlabel()
    except Exception:  # noqa: BLE001 — older/newer mpl grouper APIs
        log.debug("Could not read a shared x label.", exc_info=True)
    return _X_FALLBACK


def run_detail_map(df, stim) -> dict[str, list[str]]:
    """``padded run_id -> the run's tooltip detail lines``.

    The PASS/FAIL verdict and the genuinely swept input parameters, exactly
    what the scatter tooltip appends below its separator. Built once per
    redraw: doing it per motion event would scan the frame hundreds of times a
    second.

    Keys are zero-padded to six digits because a ``line_map``'s run ids come
    from the ``run_<id>__<tb>.csv`` filenames while a results frame read back
    from CSV holds ``run_id`` as an int — the trap
    ``test_padded_run_ids_survive_a_csv_round_trip`` documents.
    """
    from chipify.uikit.services.transient_loader import pad_run_id

    if df is None or "run_id" not in getattr(df, "columns", []):
        return {}

    varying: list[str] = []
    for name in (getattr(stim, "params", None) or {}):
        if name not in df.columns:
            continue
        try:
            if df[name].nunique() > 1:
                varying.append(name)
        except Exception:  # noqa: BLE001 — unhashable/object column
            continue

    # Absent rather than assumed: a history run whose frame predates the
    # column would otherwise have every curve labelled FAIL.
    has_verdict = "global_pass" in df.columns
    keep = ["run_id"] + (["global_pass"] if has_verdict else []) + varying

    details: dict[str, list[str]] = {}
    for rec in df[keep].to_dict("records"):
        lines = []
        if has_verdict:
            lines.append("PASS" if bool(rec["global_pass"]) else "FAIL")
        lines.extend(f"{name}: {rec[name]}" for name in varying)
        details[pad_run_id(rec["run_id"])] = lines
    return details


def curve_tooltip_lines(run_id, signal: str, x_label: str, x_value, y_value,
                        details=None) -> list[str]:
    """The tooltip text for one hovered sample, in the scatter tooltip's shape."""
    lines = [
        f"Run #{str(run_id).zfill(6)}",
        SEPARATOR,
        f"{x_label}: {fmt_value(x_value)}",
        f"{signal}: {fmt_value(y_value)}",
    ]
    if details:
        lines.append(SEPARATOR)
        lines.extend(details)
    return lines


def _scale_space(axis, raw):
    """Map raw data onto *axis*' scale, with a validity mask.

    A log axis maps values ``<= 0`` to the sentinel ``-1000`` rather than to
    ``-inf``, so an ``isfinite`` filter alone would plant a phantom point a
    thousand decades left of the plot that wins every hit test near the left
    edge. Mask on the raw values first.
    """
    raw = np.asarray(raw, dtype=float)
    ok = np.isfinite(raw)
    if axis.get_scale() == "linear":
        return raw, ok      # the identity transform: skip it at millions of samples
    if axis.get_scale() == "log":
        ok &= raw > 0.0
    out = np.full(raw.shape, np.nan)
    if ok.any():
        out[ok] = axis.get_transform().transform(raw[ok])
    ok &= np.isfinite(out)
    return out, ok


class _CurveIndex:
    """Nearest-curve lookup over every hoverable curve of one axes.

    Holds each curve's samples in scale space (view-independent) plus their raw
    values for display. ``hit()`` applies only the remaining affine, so pan and
    zoom cost nothing; a change of axis *scale* does invalidate it.

    Every sample is kept. Consecutive segments are grouped into chunks with a
    scale-space bounding box, so a hit test first rejects the chunks whose box
    is out of reach and measures the exact distance only to the segments of
    the rest: the same answer as measuring every segment, at a fraction of the
    cost.
    """

    def __init__(self, ax, entries):
        self.ax = ax
        self.entries = list(entries)
        self.scales = (ax.get_xscale(), ax.get_yscale())
        self.keys: list = []
        # Parallel to keys, and NOT to entries: a curve with nothing finite in
        # it is drawn but indexed away, so the two lists drift apart.
        self.lines: list = []

        sx, sy, raw_x, raw_y, owner, pos = [], [], [], [], [], []
        for line, key in self.entries:
            sel = np.asarray(line.get_xydata(), dtype=float)
            if sel.size == 0:
                continue
            xs, xok = _scale_space(ax.xaxis, sel[:, 0])
            ys, yok = _scale_space(ax.yaxis, sel[:, 1])
            ok = xok & yok
            n = int(ok.sum())
            if n == 0:
                continue          # an all-NaN curve is drawn but not hoverable
            if n == 1:
                # Duplicated into a zero-length segment: the segment pass is
                # the only hit test, and a lone sample would otherwise be
                # unreachable. The positions must stay adjacent for it to
                # survive the break filter below.
                take = np.flatnonzero(ok)[[0, 0]]
                at = np.array([0, 1], dtype=np.int64)
            elif n == len(ok):
                take = slice(None)      # every sample usable: views, no copies
                at = np.arange(n, dtype=np.int64)
            else:
                take = np.flatnonzero(ok)
                at = take.astype(np.int64)
            sx.append(xs[take])
            sy.append(ys[take])
            raw_x.append(sel[take, 0])
            raw_y.append(sel[take, 1])
            owner.append(np.full(len(at), len(self.keys), dtype=np.int32))
            pos.append(at)
            self.keys.append(key)
            self.lines.append(line)

        if not self.keys:
            self.xs = self.ys = np.empty(0)
            self.raw_x = self.raw_y = np.empty(0)
            self.owner = np.empty(0, dtype=np.int32)
            self.pos = np.empty(0, dtype=np.int64)
            self.seg = np.empty(0, dtype=np.int64)
            self._box = np.empty((4, 0))
            self._seg_at = np.zeros(1, dtype=np.int64)
            return

        self.xs = np.concatenate(sx)
        self.ys = np.concatenate(sy)
        self.raw_x = np.concatenate(raw_x)
        self.raw_y = np.concatenate(raw_y)
        self.owner = np.concatenate(owner)
        self.pos = np.concatenate(pos)
        # Segment starts: samples whose successor is the next sample of the
        # same curve. Distance to the *segment*, not to the vertex, is what
        # makes a sparse DC sweep (a handful of samples, tens of pixels apart)
        # hoverable between them — while requiring adjacency keeps a NaN
        # dropout, which matplotlib draws as a gap, from being bridged into a
        # hoverable line that is not on screen.
        self.seg = np.flatnonzero((self.owner[:-1] == self.owner[1:])
                                  & (self.pos[1:] == self.pos[:-1] + 1))

        # Chunks of `size` consecutive samples, each box also taking in the next
        # chunk's first sample: a segment starting in a chunk then lies inside
        # its box, including the one that crosses into the next chunk.
        n = self.xs.size
        size = max(1, -(-n // MAX_INDEX_CHUNKS))
        n_chunks = -(-n // size)
        pad = n_chunks * size - n

        def _per_chunk(arr, pick, fill):
            arr = np.concatenate([arr, np.full(pad, fill)]) if pad else arr
            arr = arr.reshape(n_chunks, size)
            # Column by column: min(axis=1) over this short inner axis took
            # about six times as long, most of the index build at 500 runs.
            out = arr[:, 0].copy()
            for j in range(1, size):
                pick(out, arr[:, j], out=out)
            return out

        box = [_per_chunk(self.xs, np.minimum, np.inf),
               _per_chunk(self.xs, np.maximum, -np.inf),
               _per_chunk(self.ys, np.minimum, np.inf),
               _per_chunk(self.ys, np.maximum, -np.inf)]
        first_of_next = np.arange(1, n_chunks) * size
        for k, (vals, pick) in enumerate(((self.xs, np.minimum), (self.xs, np.maximum),
                                          (self.ys, np.minimum), (self.ys, np.maximum))):
            box[k][:-1] = pick(box[k][:-1], vals[first_of_next])
        self._box = np.vstack(box)
        # Positions in `seg` where each chunk's segment starts begin and end.
        self._seg_at = np.searchsorted(self.seg, np.arange(n_chunks + 1) * size)

    def __len__(self) -> int:
        return len(self.keys)

    def stale(self) -> bool:
        """True once an axis scale changed — the cached coordinates baked it in."""
        return self.scales != (self.ax.get_xscale(), self.ax.get_yscale())

    def _near_segments(self, m, x_px, y_px, radius_px):
        """Segment starts in the chunks whose pixel box is within reach."""
        x0, x1, y0, y1 = self._box

        def _span(cx, cy, c):
            # Pixel extent of the boxes under row (cx, cy, c) of the affine.
            p, q, r, s = cx * x0, cx * x1, cy * y0, cy * y1
            return (np.minimum(p, q) + np.minimum(r, s) + c,
                    np.maximum(p, q) + np.maximum(r, s) + c)

        px_lo, px_hi = _span(m[0, 0], m[0, 1], m[0, 2])
        py_lo, py_hi = _span(m[1, 0], m[1, 1], m[1, 2])
        near = np.flatnonzero((px_lo - radius_px <= x_px) & (x_px <= px_hi + radius_px)
                              & (py_lo - radius_px <= y_px) & (y_px <= py_hi + radius_px))
        starts, ends = self._seg_at[near], self._seg_at[near + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        # Concatenate the ranges [start, end) without a Python loop.
        shift = np.repeat(starts - np.cumsum(lengths) + lengths, lengths)
        return self.seg[shift + np.arange(total)]

    def hit(self, x_px, y_px, radius_px=HIT_RADIUS_PX):
        """``(line, key, x_value, y_value, sample)`` for the nearest curve, else None."""
        if self.seg.size == 0:
            return None
        m = (self.ax.transLimits + self.ax.transAxes).get_matrix()
        i = self._near_segments(m, x_px, y_px, radius_px)
        if i.size == 0:
            return None
        ax_ = m[0, 0] * self.xs[i] + m[0, 1] * self.ys[i] + m[0, 2]
        ay = m[1, 0] * self.xs[i] + m[1, 1] * self.ys[i] + m[1, 2]
        bx = m[0, 0] * self.xs[i + 1] + m[0, 1] * self.ys[i + 1] + m[0, 2]
        by = m[1, 0] * self.xs[i + 1] + m[1, 1] * self.ys[i + 1] + m[1, 2]
        dx, dy = bx - ax_, by - ay
        length2 = dx * dx + dy * dy
        t = np.where(
            length2 > 0.0,
            ((x_px - ax_) * dx + (y_px - ay) * dy) / np.where(length2 > 0.0, length2, 1.0),
            0.0,
        )
        np.clip(t, 0.0, 1.0, out=t)
        cx = ax_ + t * dx
        cy = ay + t * dy
        d2 = (cx - x_px) ** 2 + (cy - y_px) ** 2

        best = int(np.argmin(d2))
        if not d2[best] <= radius_px * radius_px:
            return None
        # Report an actual sample from the CSV, not a point interpolated along
        # the segment: whichever end of the winning segment is nearer.
        lo = int(i[best])
        d_lo = (ax_[best] - x_px) ** 2 + (ay[best] - y_px) ** 2
        d_hi = (bx[best] - x_px) ** 2 + (by[best] - y_px) ** 2
        j = lo if d_lo <= d_hi else lo + 1
        owner = int(self.owner[j])
        return self.lines[owner], self.keys[owner], self.raw_x[j], self.raw_y[j], j


class CurveHoverState:
    """Snapshot of everything the curve hover needs for one overlay.

    Built once per redraw and returned unchanged from the owner's ``get_state``
    (which runs on every motion event); the manager caches its index against
    this object's identity.
    """

    def __init__(self, line_map, df=None, stim=None):
        self.line_map = line_map or {}
        self.df = df           # the results frame the overlay was drawn from
        self.stim = stim       # util.Stimuli or None


class CurveHoverManager:
    """Hover tooltip over the per-run curves of one mpl canvas.

    Parameters
    ----------
    canvas, fig:
        The matplotlib canvas and its Figure.
    get_state:
        Callable returning a :class:`CurveHoverState` when curve hover is
        currently applicable, else ``None`` (e.g. another plot mode is active).
    radius_px:
        Cursor-to-curve distance, in pixels, that counts as pointing at it.
    """

    def __init__(self, canvas, fig, get_state, *, radius_px: float = HIT_RADIUS_PX):
        self._canvas = canvas
        self._fig = fig
        self._get_state = get_state
        self._radius = float(radius_px)
        self._bubble = HoverBubble(canvas, fig, animated=True)
        self._state = None
        self._details: dict[str, list[str]] = {}
        self._index: dict = {}
        self._last = None
        self._bg = None
        self._highlight = None    # (Line2D, the properties it had before)
        self._can_blit = all(hasattr(canvas, name) for name in
                             ("copy_from_bbox", "restore_region", "blit"))
        self._bubble.animated = self._can_blit

    def connect(self) -> None:
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        # Without these the bubble stays stranded when the pointer leaves the
        # canvas quickly — and, because an animated artist *is* rendered by
        # savefig, it would be baked into an Export… or toolbar Save taken
        # right afterwards.
        self._canvas.mpl_connect("axes_leave_event", self._on_leave)
        self._canvas.mpl_connect("figure_leave_event", self._on_leave)
        self._canvas.mpl_connect("draw_event", self._on_draw)

    def invalidate(self) -> None:
        """Drop every cached artist — call around each ``fig.clf()`` redraw."""
        self._clear_highlight()
        self._bubble.invalidate()
        self._state = None
        self._details = {}
        self._index = {}
        self._last = None
        self._bg = None

    def hide(self) -> None:
        """Drop the tooltip and the highlight, e.g. before exporting the figure."""
        self._last = None
        restored = self._clear_highlight()
        if self._bubble.hide() or restored:
            self._repaint()

    # ── Highlighting the hovered curve ────────────────────────────────────────

    def _apply_highlight(self, line) -> None:
        """Draw *line* thicker and fully opaque, remembering how it was."""
        if self._highlight is not None and self._highlight[0] is line:
            return
        self._clear_highlight()
        if line is None:
            return
        saved = {
            "linewidth": line.get_linewidth(),
            "alpha": line.get_alpha(),
            "zorder": line.get_zorder(),
        }
        line.set_linewidth(max(saved["linewidth"] * HIGHLIGHT_WIDTH_FACTOR,
                               HIGHLIGHT_MIN_WIDTH))
        # An overlay auto-fades to alpha 0.05 at 500 curves; the one being
        # pointed at has to come back out of that crowd.
        line.set_alpha(1.0)
        line.set_zorder(saved["zorder"] + 3)
        self._highlight = (line, saved)

    def _clear_highlight(self) -> bool:
        """Put the highlighted curve back as it was. True if one was changed."""
        if self._highlight is None:
            return False
        line, saved = self._highlight
        self._highlight = None
        try:
            line.set_linewidth(saved["linewidth"])
            line.set_alpha(saved["alpha"])
            line.set_zorder(saved["zorder"])
        except Exception:  # noqa: BLE001 — artist already discarded by fig.clf()
            return False
        return True

    # ── Internals ─────────────────────────────────────────────────────────────

    def _on_leave(self, _event) -> None:
        self.hide()

    def _on_draw(self, event) -> None:
        """Save the freshly drawn figure as the blit background.

        The bubble is animated, so a normal draw skips it and the background is
        clean. That same draw wiped whatever was blitted, so the tooltip's own
        state is reset here too. While blitting is unavailable the artists are
        the only state there is, so this leaves them alone.

        Exports draw too: a PNG export renders on this canvas at export
        resolution, an SVG one on a temporary canvas, and both emit this event.
        Saving either as the background left a frame of the wrong size to be
        restored by the next hover, so they are ignored.
        """
        source = getattr(event, "canvas", self._canvas)
        if source is not self._canvas or getattr(source, "is_saving", bool)():
            return
        if not self._can_blit:
            self._bg = None
            return
        # Unlike the bubble, a highlighted curve is an ordinary artist: this
        # draw rendered it thickened.
        restored = self._clear_highlight()
        self._bubble.hide()
        self._last = None
        try:
            self._bg = self._canvas.copy_from_bbox(self._fig.bbox)
        except Exception:  # noqa: BLE001 — a canvas that cannot blit
            log.debug("Could not save a blit background.", exc_info=True)
            self._fall_back()
            return
        if restored:
            # …so the frame just saved still shows the thick curve. Drop it and
            # schedule the repaint that puts the curve back to normal width.
            self._bg = None
            self._canvas.draw_idle()

    def _fall_back(self) -> None:
        """Give up on blitting for good and repaint whole figures instead.

        One-way on purpose: while ``_can_blit`` is true the bubble is animated
        and therefore absent from the saved background, and while it is false
        the bubble is an ordinary artist and no background is ever saved.
        Flip-flopping between the two would bake a bubble into a background
        and smear it across the plot.
        """
        self._can_blit = False
        self._bubble.animated = False
        self._bg = None

    def _ensure_background(self) -> None:
        """Save a clean background before the first blit of a redraw cycle."""
        if not self._can_blit or self._bg is not None:
            return
        try:
            self._canvas.draw()      # _on_draw saves it; the bubble is animated
        except Exception:  # noqa: BLE001
            log.debug("Could not draw a blit background.", exc_info=True)
            self._fall_back()

    def _repaint(self) -> None:
        annot = self._bubble.annot
        if self._can_blit and self._bg is not None:
            try:
                self._canvas.restore_region(self._bg)
                # The thickened curve covers the thin one already in the
                # background; dropping it is what un-highlights.
                if self._highlight is not None and self._highlight[0].axes is not None:
                    line = self._highlight[0]
                    line.axes.draw_artist(line)
                if annot is not None and annot.get_visible() and annot.axes is not None:
                    annot.axes.draw_artist(annot)
                self._canvas.blit(self._fig.bbox)
                return
            except Exception:  # noqa: BLE001
                log.debug("Blitting the hover tooltip failed.", exc_info=True)
        self._fall_back()
        if annot is not None:
            annot.set_animated(False)
        self._canvas.draw_idle()

    def _interaction_locked(self, event) -> bool:
        """True while a drag or a toolbar tool (pan / zoom rectangle) owns the mouse."""
        if getattr(event, "button", None) is not None:
            return True
        lock = getattr(self._canvas, "widgetlock", None)
        try:
            return bool(lock.locked())
        except Exception:  # noqa: BLE001 — no widgetlock on this canvas
            return False

    def _index_for(self, state, ax):
        """The hit index for *ax*, built on the first hover over it."""
        if state is not self._state:
            self._state = state
            self._index = {}
            self._details = run_detail_map(state.df, state.stim)
        index = self._index.get(ax, False)
        if index is False:
            entries = [(line, key) for line, key in state.line_map.items()
                       if line.axes is ax]
            index = _CurveIndex(ax, entries) if entries else None
            if index is not None and not len(index):
                index = None
            self._index[ax] = index
        elif index is not None and index.stale():
            index = _CurveIndex(ax, index.entries)
            self._index[ax] = index
        return index

    def _hide(self) -> None:
        self._last = None
        restored = self._clear_highlight()
        if self._bubble.hide() or restored:
            self._repaint()

    def _on_motion(self, event) -> None:
        if self._interaction_locked(event):
            self._hide()
            return
        state = self._get_state()
        ax = getattr(event, "inaxes", None)
        if state is None or ax is None or event.x is None or event.y is None:
            self._hide()
            return
        index = self._index_for(state, ax)
        if index is None:
            self._hide()
            return
        hit = index.hit(event.x, event.y, self._radius)
        if hit is None:
            self._hide()
            return

        line, (run_id, signal), x_value, y_value, sample = hit
        token = (id(ax), run_id, signal, sample)
        if token == self._last and self._bubble.visible():
            return          # same sample: nothing to repaint

        from chipify.uikit.services.transient_loader import pad_run_id
        text = "\n".join(curve_tooltip_lines(
            run_id, signal, x_axis_label(ax), x_value, y_value,
            self._details.get(pad_run_id(run_id)),
        ))
        # Saving a background runs a draw, so do it before the highlight and
        # the bubble exist — neither belongs in the saved frame.
        if self._can_blit and self._bg is None:
            self._clear_highlight()
            self._ensure_background()
        self._apply_highlight(line)
        self._bubble.show(ax, (x_value, y_value), text, event)
        self._last = token
        self._repaint()
