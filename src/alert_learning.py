"""Learn which kinds of alert turn out to be real, from the system's own evidence
and from the operator's "Wrong alarm" / "Real" reports.

Every image or headlight alert is followed for FOLLOW_S seconds. If the AI then
recognises a vehicle or person on the same route, that alert counts as
"followed". When the operator marks an alert Wrong or Real on the dashboard,
that verdict replaces the automatic guess for that alert (and can be undone).
Per alert kind (zone + channel) the record keeps totals and the outcome of the
most recent alerts. The results are used three ways, all safe:

* shown in Settings, so the operator can see what is real;
* suggestions to tighten a threshold, applied only when the operator clicks;
* reliable kinds (almost always real) climb on the faster tier.

Nothing here ever makes an alarm quieter or suppresses one by itself.
Cheap: a few dictionary updates per alert, one small JSON file.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

FOLLOW_S = 20.0          # how long after an alert a recognised object still counts
RECENT = 60              # outcomes kept per alert kind
MIN_EVIDENCE = 15        # alerts needed before a kind is judged
RELIABLE = 0.85          # real this often: promote to the faster tier
UNRELIABLE = 0.30        # real less often than this: suggest tightening
OPERATOR_WRONG_MIN = 3   # operator "wrong" reports before a kind gets a suggestion
VERDICTS = ('wrong', 'real', 'clear')


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class AlertLearning:
    def __init__(self, path: Path | str | None, clock=time.monotonic):
        self.path = Path(path) if path else None
        self.clock = clock
        self._lock = threading.Lock()
        self._kinds: dict[str, dict] = {}
        self._pending: list[dict] = []
        self._dirty = False
        self._saved_at = 0.0
        self._load()

    # -- storage ------------------------------------------------------------
    def _load(self):
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            kinds = data.get('kinds', {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return
        if not isinstance(kinds, dict):
            return
        for key, row in kinds.items():
            if not isinstance(key, str) or not isinstance(row, dict):
                continue
            raw_recent = row.get('recent', [])
            recent = [1 if value else 0 for value in raw_recent if value in (0, 1, True, False)] \
                if isinstance(raw_recent, list) else []
            alerts, followed = row.get('alerts', 0), row.get('followed', 0)
            if not (isinstance(alerts, int) and isinstance(followed, int) and 0 <= followed <= alerts):
                continue
            recent = recent[-RECENT:]
            ids = row.get('ids') if isinstance(row.get('ids'), list) else []
            ids = [value if isinstance(value, str) and 0 < len(value) <= 128 else None for value in ids]
            ids = ([None] * max(0, len(recent) - len(ids)) + ids)[-len(recent):] if recent else []
            labels = {}
            for alarm_id, label in (row.get('labels') or {}).items() if isinstance(row.get('labels'), dict) else ():
                if (alarm_id in ids and isinstance(label, dict) and label.get('verdict') in ('wrong', 'real')
                        and label.get('auto') in (0, 1, None)):
                    labels[alarm_id] = {'verdict': label['verdict'], 'auto': label['auto'],
                                        'cross': label.get('cross') is True}
            self._kinds[key] = {'name': str(row.get('name', key))[:120], 'route': str(row.get('route', ''))[:120],
                                'source': str(row.get('source', ''))[:20], 'zone': str(row.get('zone', ''))[:80],
                                'alerts': alerts, 'followed': followed, 'recent': recent, 'ids': ids,
                                'labels': labels}

    def save(self, force=False):
        if self.path is None:
            return
        with self._lock:
            if not self._dirty or (not force and self.clock() - self._saved_at < 30):
                return
            payload = json.dumps({'version': 2, 'kinds': self._kinds}, indent=1)
            self._dirty = False
            self._saved_at = self.clock()
        temporary = self.path.with_name(self.path.name + '.tmp')
        try:
            temporary.write_text(payload, encoding='utf-8')
            temporary.replace(self.path)
        except OSError:
            with self._lock:
                self._dirty = True

    # -- evidence -----------------------------------------------------------
    @staticmethod
    def kind_of(details) -> tuple[str, dict] | None:
        details = details if isinstance(details, dict) else {}
        source = details.get('source')
        zone = details.get('zone_name')
        if source not in ('light', 'motion', 'ai') or not isinstance(zone, str):
            return None
        label = {'light': 'headlights', 'motion': 'image change', 'ai': 'AI'}[source]
        if details.get('strict_change'):
            label = 'car-sized change'
        return f'{zone}|{source}', {'name': f'{zone}: {label}', 'zone': zone, 'source': source,
                                    'route': str(details.get('route_group') or zone)}

    def _row(self, key, meta):
        return self._kinds.setdefault(key, {**meta, 'alerts': 0, 'followed': 0, 'recent': [], 'ids': [],
                                            'labels': {}})

    def note_alarm(self, alarm_id, details, now=None):
        """An alert sounded. AI alerts count as real unless the operator says otherwise."""
        found = self.kind_of(details)
        if found is None:
            return
        key, meta = found
        now = self.clock() if now is None else now
        with self._lock:
            row = self._row(key, meta)
            if meta['source'] == 'ai':
                self._record(row, True, alarm_id)
                return
            self._pending.append({'id': alarm_id, 'key': key, 'route': meta['route'],
                                  'at': now, 'followed': False, 'operator': None,
                                  'cross': bool((details or {}).get('cross_lane'))})

    def note_recognised(self, route, now=None):
        """The AI recognised a vehicle or person on this route."""
        if not isinstance(route, str) or not route:
            return
        now = self.clock() if now is None else now
        with self._lock:
            for item in self._pending:
                if item['route'] == route and 0 <= now - item['at'] <= FOLLOW_S:
                    item['followed'] = True

    def tick(self, now=None):
        now = self.clock() if now is None else now
        with self._lock:
            done = [item for item in self._pending if now - item['at'] > FOLLOW_S]
            self._pending = [item for item in self._pending if now - item['at'] <= FOLLOW_S]
            for item in done:
                row = self._kinds.get(item['key'])
                if row is None:
                    continue
                verdict = item.get('operator')
                if verdict in ('wrong', 'real'):
                    self._record(row, verdict == 'real', item['id'])
                    if isinstance(item['id'], str):
                        row['labels'][item['id']] = {'verdict': verdict, 'auto': 1 if item['followed'] else 0,
                                                     'cross': bool(item.get('cross'))}
                else:
                    self._record(row, item['followed'], item['id'])
        if done:
            self.save()

    def _record(self, row, followed, alarm_id=None):
        row['alerts'] += 1
        row['followed'] += 1 if followed else 0
        row['recent'] = (row['recent'] + [1 if followed else 0])[-RECENT:]
        ids = row.setdefault('ids', [])
        ids.append(alarm_id if isinstance(alarm_id, str) else None)
        row['ids'] = ids[-RECENT:]
        labels = row.setdefault('labels', {})
        for old in [key for key in labels if key not in row['ids']]:
            del labels[old]
        self._dirty = True

    # -- operator reports ---------------------------------------------------
    def note_feedback(self, alarm_id, verdict, details=None) -> dict:
        """The operator marked one alert Wrong or Real (or took that back with clear).

        The verdict replaces the automatic follow-up guess for that alert. An
        alert the learner never saw (e.g. from before learning started) is
        added as a new outcome when its kind is known.
        """
        if verdict not in VERDICTS:
            raise ValueError('Verdict must be wrong, real or clear.')
        if not isinstance(alarm_id, str) or not 0 < len(alarm_id) <= 128:
            raise ValueError('An alarm identifier is required.')
        result = {'applied': None}
        with self._lock:
            for item in self._pending:
                if item['id'] == alarm_id:
                    item['operator'] = None if verdict == 'clear' else verdict
                    result = {'applied': 'pending', 'kind': item['key']}
                    break
            else:
                result = self._apply_feedback(alarm_id, verdict, details)
            if result.get('applied'):
                self._dirty = True
        if result.get('applied'):
            self.save(force=True)
        return result

    def _apply_feedback(self, alarm_id, verdict, details):
        for key, row in self._kinds.items():
            ids = row.get('ids', [])
            if alarm_id not in ids:
                continue
            index = ids.index(alarm_id)
            old = row['recent'][index]
            label = row['labels'].get(alarm_id)
            auto = label['auto'] if label else old
            if verdict == 'clear':
                if label is None:
                    return {'applied': 'unchanged', 'kind': key}
                del row['labels'][alarm_id]
                if auto is None:          # added by the operator: take it out again
                    del row['recent'][index]
                    del row['ids'][index]
                    row['alerts'] = max(0, row['alerts'] - 1)
                    row['followed'] = max(0, min(row['alerts'], row['followed'] - old))
                    return {'applied': 'removed', 'kind': key}
                new = auto
            else:
                row['labels'][alarm_id] = {'verdict': verdict, 'auto': auto,
                                           'cross': bool((details or {}).get('cross_lane'))}
                new = 1 if verdict == 'real' else 0
            row['recent'][index] = new
            row['followed'] = max(0, min(row['alerts'], row['followed'] + new - old))
            return {'applied': 'recorded', 'kind': key}
        if verdict == 'clear':
            return {'applied': 'unchanged'}
        found = self.kind_of(details)
        if found is None:
            return {'applied': None, 'reason': 'This kind of alert is not tracked by learning.'}
        key, meta = found
        row = self._row(key, meta)
        self._record(row, verdict == 'real', alarm_id)
        row['labels'][alarm_id] = {'verdict': verdict, 'auto': None,
                                   'cross': bool((details or {}).get('cross_lane'))}
        return {'applied': 'added', 'kind': key}

    # -- use ----------------------------------------------------------------
    @staticmethod
    def _rate(row):
        recent = row['recent']
        return (sum(recent) / len(recent)) if len(recent) >= MIN_EVIDENCE else None

    @staticmethod
    def _marks(row):
        labels = row.get('labels', {}).values()
        return (sum(1 for label in labels if label['verdict'] == 'wrong'),
                sum(1 for label in labels if label['verdict'] == 'real'))

    def reliable(self, details) -> bool:
        """Almost always real: may climb faster."""
        found = self.kind_of(details)
        if found is None or found[1]['source'] == 'ai':
            return False
        with self._lock:
            row = self._kinds.get(found[0])
            rate = self._rate(row) if row else None
        return rate is not None and rate >= RELIABLE

    def status(self, zones_config=None) -> dict:
        with self._lock:
            rows = sorted(self._kinds.items(), key=lambda item: -item[1]['alerts'])
            routes = []
            for key, row in rows:
                wrong, real = self._marks(row)
                routes.append({'key': key, 'name': row['name'], 'alerts': len(row['recent']),
                               'followed': sum(row['recent']), 'all_time': row['alerts'],
                               'rate': self._rate(row), 'reliable': (self._rate(row) or 0) >= RELIABLE,
                               'marked_wrong': wrong, 'marked_real': real})
            pending = len(self._pending)
        suggestions = self.suggestions(zones_config) if zones_config else []
        total = sum(r['alerts'] for r in routes)
        marked = sum(r['marked_wrong'] + r['marked_real'] for r in routes)
        summary = (f'{total} recent alerts checked; an alert counts as real when the AI recognises a vehicle '
                   f'or person on the same route within {FOLLOW_S:.0f} seconds, or when you mark it Real. '
                   f'Your Wrong/Real marks always win ({marked} so far).' if total else '')
        return {'routes': routes, 'pending': pending, 'suggestions': suggestions, 'summary': summary,
                'follow_s': FOLLOW_S, 'min_evidence': MIN_EVIDENCE, 'operator_marks': marked}

    def suggestions(self, zones_config) -> list[dict]:
        zones = {z.get('name'): z for z in zones_config.get('zones', []) if isinstance(z, dict)}
        result = []
        with self._lock:
            items = [(key, {**row, 'marks': self._marks(row),
                            'cross_wrong': sum(1 for label in row.get('labels', {}).values()
                                               if label['verdict'] == 'wrong' and label.get('cross'))})
                     for key, row in self._kinds.items()]
        for key, row in items:
            rate = self._rate(row)
            wrong, real = row['marks']
            by_operator = wrong >= OPERATOR_WRONG_MIN and wrong > real
            if not by_operator and (rate is None or rate >= UNRELIABLE):
                continue
            zone = zones.get(row['zone'])
            if zone is None:
                continue
            why = (f"you marked {wrong} recent alert{'s' if wrong != 1 else ''} wrong" if by_operator else
                   f"only {sum(row['recent'])} of {len(row['recent'])} recent alerts had a vehicle")
            name = row['name']
            if by_operator and row['cross_wrong'] * 2 >= wrong:
                result.append({'key': None, 'value': None, 'by_operator': True, 'advice': True,
                               'text': f"{name}: {why}, and {row['cross_wrong']} of them came just after a vehicle "
                                       f"on the other lane. That is headlight spill from the other lane: a higher "
                                       f"threshold would not stop it and could hide real cars. Keep marking; the "
                                       f"other-lane label is being checked against your marks."})
                continue
            if row['source'] == 'light' and _finite(zone.get('light_delta')):
                value = round(min(80., zone['light_delta'] * 1.25 + 1), 1)
                if value <= zone['light_delta']:
                    continue
                result.append({'key': f"zone:{row['zone']}:light_delta", 'value': value, 'by_operator': by_operator,
                               'text': f"{name}: {why}. Raise its headlight threshold "
                                       f"from {zone['light_delta']:g} to {value:g}?"})
            elif row['source'] == 'motion' and zone.get('strict_change') and _finite(zone.get('connected_pct')):
                value = round(min(60., zone['connected_pct'] * 1.25), 1)
                if value <= zone['connected_pct']:
                    continue
                result.append({'key': f"zone:{row['zone']}:connected_pct", 'value': value, 'by_operator': by_operator,
                               'text': f"{name}: {why}. Raise its car-sized change "
                                       f"from {zone['connected_pct']:g}% to {value:g}%?"})
            elif (row['source'] == 'ai' and by_operator and zone.get('trigger') in ('ai', 'ai_light')
                    and _finite(zone.get('ai_confidence'))):
                value = round(min(.95, zone['ai_confidence'] + .05), 2)
                if value <= zone['ai_confidence']:
                    continue
                result.append({'key': f"zone:{row['zone']}:ai_confidence", 'value': value, 'by_operator': True,
                               'text': f"{name}: {why}. Make the AI need {value * 100:.0f}% certainty "
                                       f"instead of {zone['ai_confidence'] * 100:.0f}%?"})
        return result
