"""Bounded, local webcam motion context; never acknowledgement or hand recognition.

Only two small grayscale samples and a face exclusion mask are kept in memory.
No frame, face identity, key input or image-derived baseline is persisted.
"""
import math
import threading
import time

import numpy as np


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class DeskActivity:
    ROI = (.05, .50, .90, .48)

    def __init__(self, *, hold_s=2., freshness_s=3., frozen_s=3.):
        for value, minimum in ((hold_s, 2.), (freshness_s, .1), (frozen_s, 2.)):
            if not _finite(value) or value < minimum:
                raise ValueError('Desk activity durations must be finite positive seconds')
        self.hold_s, self.freshness_s, self.frozen_s = map(float, (hold_s, freshness_s, frozen_s))
        self._lock = threading.RLock()
        self._previous = self._face_mask = None
        self._observed_at = self._movement_at = self._identical_since = None
        self._report = self._unknown('not_started')

    def _unknown(self, reason):
        return {'state': 'unknown', 'reason': reason, 'observed_at': self._observed_at,
                'coverage': 0., 'global_change_fraction': 0., 'exposure_shift': 0.,
                'roi': list(self.ROI), 'face_excluded': False,
                'acknowledges': False, 'method': 'lower-view motion; not hand or button recognition'}

    def invalidate(self, reason='camera_unavailable'):
        with self._lock:
            self._previous = self._face_mask = None
            self._movement_at = self._identical_since = None
            self._report = self._unknown(reason)

    def _fresh(self, stamp, now):
        return _finite(stamp) and 0 <= now-stamp < self.freshness_s

    def update(self, frame, observed_at, *, now=None, face_box=None, camera_ready=True):
        """Measure a fresh BGR webcam frame; face_box is original-pixel xywh."""
        import cv2
        now = time.monotonic() if now is None else now
        if not _finite(now):
            raise ValueError('now must be a finite monotonic timestamp')
        with self._lock:
            if camera_ready is not True or not self._fresh(observed_at, now):
                self.invalidate('camera_unavailable' if camera_ready is not True else 'stale_frame')
                return self.status(now)
            if self._observed_at is not None and observed_at <= self._observed_at:
                return self.status(now)
            pixels = np.asarray(frame)
            if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
                    or min(pixels.shape[:2]) < 16 or max(pixels.shape[:2]) > 4096):
                self.invalidate('invalid_frame')
                return self.status(now)
            old_stamp = self._observed_at
            self._observed_at = float(observed_at)
            small = cv2.resize(pixels, (160, 120), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # A covered, featureless or saturated view is not a desk gesture.
            p10, p90 = np.percentile(gray, [10, 90])
            if p90 <= 12 or p10 >= 243 or float(np.std(gray)) < 2.:
                self.invalidate('view_obscured_or_featureless')
                return self.status(now)
            mask = np.zeros(gray.shape, dtype=bool)
            mask[60:118, 8:152] = True
            face_mask = np.zeros(gray.shape, dtype=bool)
            if isinstance(face_box, (tuple, list)) and len(face_box) == 4 and all(_finite(v) for v in face_box):
                x, y, width, height = face_box
                if width > 0 and height > 0:
                    sx, sy = 160. / pixels.shape[1], 120. / pixels.shape[0]
                    x1, y1 = max(0, int((x-.15*width)*sx)), max(0, int((y-.15*height)*sy))
                    x2, y2 = min(160, int((x+1.15*width)*sx)), min(120, int((y+1.15*height)*sy))
                    if x2 > x1 and y2 > y1:
                        face_mask[y1:y2, x1:x2] = True
            mask &= ~face_mask
            if self._face_mask is not None:
                mask &= ~self._face_mask
            previous, old_face_mask = self._previous, self._face_mask
            self._previous, self._face_mask = gray.copy(), face_mask
            if old_stamp is None or observed_at-old_stamp >= self.freshness_s or previous is None:
                self._movement_at = None
                self._identical_since = observed_at
                self._report = self._unknown('establishing_fresh_view')
                return self.status(now)
            if int(mask.sum()) < 800:
                self.invalidate('desk_region_obscured_by_face')
                return self.status(now)
            if np.array_equal(gray, previous):
                if self._identical_since is None:
                    self._identical_since = old_stamp
            else:
                self._identical_since = observed_at
            if observed_at-self._identical_since >= self.frozen_s:
                self._movement_at = None
                self._report = self._unknown('static_or_frozen_view')
                return self.status(now)
            signed = gray.astype(np.float32)-previous.astype(np.float32)
            exposure = float(np.median(signed))
            residual = np.abs(signed-exposure)
            changed = residual >= 14.
            global_fraction = float(np.mean(changed))
            # Broad changes could be camera movement or occlusion. Do not label
            # a jolt, screen flash or covered lens as an operator action.
            broad = global_fraction > .35
            report = {'state': 'quiet', 'reason': 'no_recent_desk_movement',
                      'observed_at': self._observed_at, 'coverage': 0.,
                      'global_change_fraction': round(global_fraction, 4),
                      'exposure_shift': round(exposure, 2), 'roi': list(self.ROI),
                      'face_excluded': bool(face_mask.any() or (old_face_mask is not None and old_face_mask.any())),
                      'acknowledges': False,
                      'method': 'lower-view motion; not hand or button recognition'}
            if broad:
                self._movement_at = None
                report.update(state='unknown', reason='broad_view_change')
            else:
                binary = (changed & mask).astype(np.uint8)
                # Require local spatial support; scattered sensor noise is not
                # foreground movement. This is deliberately not a skin detector.
                neighbors = cv2.filter2D(binary, cv2.CV_16U, np.ones((3, 3), dtype=np.uint8))
                binary &= (neighbors >= 5).astype(np.uint8)
                count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
                coverage = float(np.count_nonzero(binary)/np.count_nonzero(mask))
                report['coverage'] = round(coverage, 4)
                if largest >= 20 and .004 <= coverage <= .35:
                    self._movement_at = observed_at
                elif coverage > .35:
                    self._movement_at = None
                    report.update(state='unknown', reason='desk_view_obscured_or_shifted')
                if report['state'] != 'unknown' and self._movement_at is not None and observed_at-self._movement_at <= self.hold_s:
                    report.update(state='movement', reason='recent_localized_desk_movement')
            self._report = report
            return self.status(now)

    def status(self, now=None):
        now = time.monotonic() if now is None else now
        if not _finite(now):
            raise ValueError('now must be a finite monotonic timestamp')
        with self._lock:
            report = {**self._report, 'roi': list(self.ROI)}
            fresh = self._fresh(self._observed_at, now)
            report['age_s'] = round(max(0., now-self._observed_at), 3) if _finite(self._observed_at) else None
            age = now-self._movement_at if self._movement_at is not None else None
            report['movement_age_s'] = round(max(0., age), 3) if age is not None else None
            report['hold_remaining_s'] = round(max(0., self.hold_s-age), 3) if age is not None and fresh else 0.
            report['ready'] = fresh and report['state'] != 'unknown'
            if not fresh:
                report.update(state='unknown', reason='stale_frame' if self._observed_at is not None else report['reason'],
                              ready=False, hold_remaining_s=0.)
            elif report['state'] == 'movement' and age is not None and age > self.hold_s:
                report.update(state='quiet', reason='no_recent_desk_movement')
            return report
