import math
import unittest
from region_geometry import (MAX_POLYGON_CORNERS, validate_polygon, point_in_polygon,
                             polygon_mask, region_contains)


class RegionGeometryTests(unittest.TestCase):
    def test_detailed_polygons_preserve_every_corner_with_bounded_limit(self):
        self.assertEqual(MAX_POLYGON_CORNERS, 32)
        for count in (18, 21, MAX_POLYGON_CORNERS):
            points = [[.5 + .4 * math.cos(2 * math.pi * i / count),
                       .5 + .4 * math.sin(2 * math.pi * i / count)]
                      for i in range(count)]
            with self.subTest(count=count):
                self.assertEqual(validate_polygon(points), points)
        points = [[.5 + .4 * math.cos(2 * math.pi * i / 33),
                   .5 + .4 * math.sin(2 * math.pi * i / 33)] for i in range(33)]
        with self.assertRaisesRegex(ValueError, 'needs 3–32 corners'):
            validate_polygon(points)

    def test_concave_region_and_boundary(self):
        polygon = validate_polygon([[0,0],[1,0],[1,.3],[.3,.3],[.3,1],[0,1]])
        self.assertTrue(point_in_polygon(.2,.8,polygon))
        self.assertTrue(point_in_polygon(.3,.8,polygon))
        self.assertFalse(point_in_polygon(.8,.8,polygon))

    def test_excluded_area_and_boundary_are_ignored(self):
        polygon = [[0,0],[1,0],[1,1],[0,1]]
        hole = [[.3,.3],[.7,.3],[.7,.7],[.3,.7]]
        self.assertFalse(region_contains(.3,.5,polygon,[hole]))
        self.assertTrue(region_contains(.2,.5,polygon,[hole]))
        self.assertFalse(region_contains(math.nan,.5,polygon))

    def test_raster_matches_anchor_for_each_pixel(self):
        polygon = [[0,0],[1,0],[.65,.7],[.2,1]]
        holes = [[[.2,.2],[.4,.2],[.4,.4],[.2,.4]]]
        mask = polygon_mask(40,30,polygon,holes)
        for y in range(30):
            for x in range(40):
                self.assertEqual(mask[y,x],region_contains((x+.5)/40,(y+.5)/30,polygon,holes))

    def test_bad_geometry_rejected(self):
        cases = [[], [[0,0],[1,1]], [[0,0],[1,1],[0,1],[1,0]],
                 [[0,0],[.5,.5],[1,1]], [[0,0],[1,0],[1,1],[0,0]],
                 [[0,0],[1,0],[math.inf,1]], [[0,0],[1,0],[True,1]],
                 [[0,0],[1,0],[1.1,1]]]
        for points in cases:
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_polygon(points)
