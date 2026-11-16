"""Bounded route-focused views for the existing detector, in aligned pixels.

These views change the model's image scale, not its learned classes or weights.
Results from every view of a frame are merged before temporal confirmation.
"""
from __future__ import annotations

import math
from itertools import islice
from numbers import Real


def _bounds(zones, width, height, margins):
    side, head, foot = margins
    return (max(0, int((min(z.rect[0] for z in zones) - side) * width)),
            max(0, int((min(z.rect[1] for z in zones) - head) * height)),
            min(width, int((max(z.rect[0] + z.rect[2] for z in zones) + side) * width)),
            min(height, int((max(z.rect[1] + z.rect[3] for z in zones) + foot) * height)))


def inference_rects(zones, width, height):
    """Return one legacy union or at most two contextual route crops (xyxy).

Role metadata opts a complete site configuration into focused views. Unknown or
missing roles retain the original union so no configured region is omitted.
Exit vehicles and pedestrians share a view with extra headroom; the entrance
gets its own view. Padding allows boxes to extend beyond ground-point polygons.
"""
    if width < 1 or height < 1:
        raise ValueError('Inference frame must have positive dimensions')
    eligible = [z for z in zones
                if getattr(z, 'ai_enabled', getattr(z, 'motion', True))
                and getattr(z, 'classes', None) != []]
    if not eligible:
        return []
    union = _bounds(eligible, width, height, (.06, .10, .04))
    roles = {getattr(z, 'route_role', '') for z in eligible}
    if (not roles <= {'exit', 'entrance', 'pedestrian'}
            or 'entrance' not in roles or not roles & {'exit', 'pedestrian'}):
        return [union]
    groups = ([z for z in eligible if z.route_role != 'entrance'],
              [z for z in eligible if z.route_role == 'entrance'])
    rects = [_bounds(group, width, height, (.08, .12, .06)) for group in groups]
    # Avoid spending two passes on essentially identical wide configurations.
    if _iou(rects[0], rects[1]) >= .85:
        return [union]
    return rects


def _family(label):
    return 'road_vehicle' if label in {'car', 'motorcycle', 'bus', 'truck'} else label


def _iou(a, b):
    overlap = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - overlap)
    return overlap / max(union, 1e-9)


def _crop_edge_fragment(part, part_rect, whole, whole_rect, margin=2.):
    """A small crop-cut slice contained in a complete box from another view.

    IoU alone cannot merge a narrow slice with its full vehicle. Require crop
    boundary evidence, a complete alternative and strong containment; a merely
    smaller overlapping detection must not erase another car.
    """
    if part_rect == whole_rect:
        return False
    if not (whole[0] > whole_rect[0]+margin and whole[1] > whole_rect[1]+margin
            and whole[2] < whole_rect[2]-margin and whole[3] < whole_rect[3]-margin):
        return False
    part_area = (part[2]-part[0])*(part[3]-part[1])
    whole_area = (whole[2]-whole[0])*(whole[3]-whole[1])
    if part_area > .5*whole_area:
        return False
    overlap = max(0.,min(part[2],whole[2])-max(part[0],whole[0])) * max(
        0.,min(part[3],whole[3])-max(part[1],whole[1]))
    if overlap < .75*part_area:
        return False
    return any((abs(part[edge]-part_rect[edge]) <= margin
                and whole[low] < part_rect[edge]-margin
                and whole[high] > part_rect[edge]+margin)
               for edge,low,high in ((0,0,2),(1,1,3),(2,0,2),(3,1,3)))


def merge_view_detections(view_results, iou_threshold=.45):
    """Map crop detections into full-frame pixels and suppress family duplicates.

``view_results`` contains (detections, xyxy crop rectangle) for one camera frame.
Person, bicycle and motor-vehicle detections remain separate even when their
boxes overlap. Work is bounded to two detector passes / 100 input detections.
"""
    if not 0 < iou_threshold <= 1:
        raise ValueError('Inference merge IoU must be in (0, 1]')
    candidates = []
    for objects, rect in list(view_results)[:2]:
        left, top, right, bottom = rect
        for obj in objects[:50]:
            try:
                label = obj['label']
                if label not in {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'}:
                    continue
                raw = tuple(float(v) for v in obj['bbox'])
                confidence = float(obj.get('confidence', 0.))
                if (len(raw) != 4 or not all(math.isfinite(v) for v in raw)
                        or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                    continue
                box = [max(left, min(right, raw[0] + left)),
                       max(top, min(bottom, raw[1] + top)),
                       max(left, min(right, raw[2] + left)),
                       max(top, min(bottom, raw[3] + top))]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                candidates.append(({**obj, 'bbox': box, 'confidence': confidence}, rect))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
    # Compare both directions before confidence sorting: a clipped fragment
    # can score above the complete vehicle. Input remains bounded to 100 boxes.
    complete = [obj for obj,rect in candidates
                if not any(_family(obj['label']) == _family(other['label'])
                           and _crop_edge_fragment(obj['bbox'],rect,other['bbox'],other_rect)
                           for other,other_rect in candidates)]
    kept = []
    for obj in sorted(complete, key=lambda item: item['confidence'], reverse=True):
        if not any(_family(obj['label']) == _family(prior['label'])
                   and _iou(obj['bbox'], prior['bbox']) > iou_threshold for prior in kept):
            kept.append(obj)
            if len(kept) >= 50:
                break
    return kept


def add_context_detections(merged, context_objects, focused_rects, overlap=.2):
    """Objects from the whole-picture view, kept only where the focused crops do not look.

    ``merged`` are full-frame detections from the focused crops; ``context_objects`` come from the whole picture
    (already in full-frame pixels). An object whose centre lies inside any focused crop, or which overlaps a
    focused detection of the same kind, is left to the focused view (sharper), so nothing is counted twice.
    """
    kept = list(merged)
    extra = []
    for obj in list(context_objects)[:50]:
        try:
            label = obj['label']
            if label not in {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'}:
                continue
            box = [float(v) for v in obj['bbox']]
            confidence = float(obj.get('confidence', 0.))
            if (len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0]
                    or box[3] <= box[1] or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in focused_rects):
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        clash = False
        for prior in kept + extra:
            if _family(prior['label']) != _family(label):
                continue
            p = prior['bbox']
            inter = max(0., min(box[2], p[2]) - max(box[0], p[0])) * max(0., min(box[3], p[3]) - max(box[1], p[1]))
            smaller = max(1e-9, min(area, (p[2] - p[0]) * (p[3] - p[1])))
            if _iou(box, p) > overlap or inter / smaller > .5:
                clash = True
                break
        if not clash:
            extra.append({**obj, 'bbox': box, 'confidence': confidence, 'view': 'whole picture'})
    return kept + extra[:max(0, 50 - len(kept))]


def merge_corrob_view(primary, supplemental, *, primary_rects=(),
                      supplemental_rect=None, iou_threshold=.45):
    """Merge one same-frame supplemental view without adding observations.

    Both inputs are already in full-frame pixels and filtered by the caller's
    class, region and confidence policy. At most 50 objects from each input are
    inspected and 50 returned. A unique same-family match uses the higher exact
    score, except that a proven crop-cut fragment cannot replace a complete box.
    Ambiguous matches keep the primary objects rather than join distinct cars.
    The caller must submit the result to temporal tracking once for this frame.
    """
    if (isinstance(iou_threshold, bool) or not isinstance(iou_threshold, Real)
            or not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1):
        raise ValueError('Corroborating merge IoU must be in (0, 1]')

    def number(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    def rectangle(value):
        if (not isinstance(value, (list, tuple)) or len(value) != 4
                or not all(number(v) for v in value)
                or value[2] <= value[0] or value[3] <= value[1]):
            return None
        return tuple(float(v) for v in value)

    def candidates(values):
        result = []
        for obj in islice(values, 50):
            if not isinstance(obj, dict):
                continue
            label, confidence = obj.get('label'), obj.get('confidence')
            if not isinstance(label, str) or label not in {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'}:
                continue
            box = rectangle(obj.get('bbox'))
            score = obj.get('raw_confidence', confidence)
            if (box is None or not number(confidence) or not 0 <= confidence <= 1
                    or not number(score) or not 0 <= score <= 1):
                continue
            result.append(({**obj, 'bbox': list(box)}, float(score)))
        return result

    primaries, supplements = candidates(primary), candidates(supplemental)
    # One detector pass should already perform NMS. Defensively prevent its own
    # duplicate boxes from creating extra observations or ambiguous cross-matches.
    unique = []
    for obj, score in sorted(supplements, key=lambda item: item[1], reverse=True):
        if not any(_family(obj['label']) == _family(prior['label'])
                   and _iou(obj['bbox'], prior['bbox']) > iou_threshold for prior, _ in unique):
            unique.append((obj, score))
    supplements = unique
    rects = [r for value in islice(primary_rects, 2) if (r := rectangle(value)) is not None]
    extra_rect = rectangle(supplemental_rect)

    def containing(box, rect, margin=0.):
        return (box[0] >= rect[0]+margin and box[1] >= rect[1]+margin
                and box[2] <= rect[2]-margin and box[3] <= rect[3]-margin)

    def primary_rect(box):
        # Merged primary boxes no longer carry exact crop provenance. If any
        # supplied crop contains this box completely, clipping is not proven.
        if any(containing(box, rect, 2.) for rect in rects):
            return None
        possible = [rect for rect in rects if containing(box, rect)]
        return possible[0] if len(possible) == 1 else None

    links = {}
    primary_links = [[] for _ in primaries]
    supplement_links = [[] for _ in supplements]
    for p_index, (p, _) in enumerate(primaries):
        for s_index, (s, _) in enumerate(supplements):
            if _family(p['label']) != _family(s['label']):
                continue
            p_rect = primary_rect(p['bbox'])
            preference = None
            if p_rect and extra_rect and _crop_edge_fragment(p['bbox'], p_rect, s['bbox'], extra_rect):
                preference = 'supplement'
            elif extra_rect and any(_crop_edge_fragment(s['bbox'], extra_rect, p['bbox'], rect)
                                    for rect in rects if containing(p['bbox'], rect, 2.)):
                preference = 'primary'
            elif _iou(p['bbox'], s['bbox']) > iou_threshold:
                preference = 'score'
            if preference:
                links[p_index, s_index] = preference
                primary_links[p_index].append(s_index)
                supplement_links[s_index].append(p_index)

    result = [obj for obj, _ in primaries]
    for s_index, (s, score) in enumerate(supplements):
        matches = supplement_links[s_index]
        if not matches:
            if len(result) < 50:
                result.append(s)
        elif len(matches) == 1 and len(primary_links[matches[0]]) == 1:
            p_index = matches[0]
            preference = links[p_index, s_index]
            if preference == 'supplement' or (preference == 'score' and score > primaries[p_index][1]):
                result[p_index] = s
        # A supplemental box spanning two cars, or two supplemental boxes
        # splitting one primary car, cannot remove the existing identities.
    return result

