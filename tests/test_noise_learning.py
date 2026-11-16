"""Restless pixels (wind in foliage, rain) must learn to stop raising alarms."""
import unittest

from test_watch import watch, watcher


class NoiseLearningTests(unittest.TestCase):
    def steady_and_restless(self, frames=300, amplitude=30.):
        obj = watcher()
        obj.semantic = None
        obj.args.warmup = 0
        rng = watch.np.random.default_rng(5)
        base = watch.np.full((48, 64, 3), 60., dtype=watch.np.float32)
        for _ in range(frames):
            frame = base.copy()
            frame[8:20, 8:20] += rng.uniform(-amplitude, amplitude)   # leaves in the wind
            obj.process(frame)
        return obj, float(obj.bg_noise[30:45, 40:60].mean()), float(obj.bg_noise[8:20, 8:20].mean())

    def test_flicker_is_learned_only_where_it_happens(self):
        obj, steady, restless = self.steady_and_restless()
        self.assertGreater(restless, 5 * steady)
        self.assertLessEqual(steady, watch.INITIAL_NOISE)
        self.assertGreaterEqual(steady, watch.MIN_NOISE)
        self.assertLessEqual(restless, watch.MAX_NOISE)

    def test_relearning_starts_cautious_after_a_rearm(self):
        obj, _, restless = self.steady_and_restless(frames=60)
        obj._pending_rearm = True
        obj.process(watch.np.full((48, 64, 3), 60., dtype=watch.np.float32))
        self.assertIsNotNone(obj.bg_noise)
        self.assertEqual(float(obj.bg_noise.max()), watch.INITIAL_NOISE)

    def test_noise_map_reaches_the_light_measurement(self):
        obj = watcher()
        obj.semantic = None
        obj.args.warmup = 0
        base = watch.np.full((48, 64, 3), 60., dtype=watch.np.float32)
        obj.process(base)
        seen = {}
        original = watch.measure_local_brightness

        def record(gray, background, region, **kwargs):
            seen.update(kwargs)
            return original(gray, background, region, **kwargs)

        with unittest.mock.patch.object(watch, "measure_local_brightness", record):
            obj.process(base)
        if seen:   # only when an ai_light zone is configured
            self.assertIsNotNone(seen["noise"])
            self.assertEqual(seen["noise"].shape, obj.bg_noise.shape)
            self.assertEqual(seen["noise_k"], watch.NOISE_K)


if __name__ == "__main__":
    unittest.main()
