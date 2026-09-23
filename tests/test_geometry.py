import os
os.environ['MPLBACKEND'] = 'Agg'
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import simpson
import v23flask as app


class GeometryTests(unittest.TestCase):
    def test_ellipse_area(self):
        self.assertAlmostEqual(app.calculate_asymmetric_area(15, 6, m=2), np.pi * 90, places=8)

    def test_requested_volumes_independently_integrated(self):
        total = app.calculate_asymmetric_area(15, 6) * 3 * 0.9478
        for px, py in [(3, 4), (0, 0), (0, 12), (15, 7.8), (10, 7.8)]:
            for target in [0.1, 10, 200, 500, total]:
                with self.subTest(px=px, py=py, target=target):
                    result = app.keep_only_requested_volume(30, 12, 3, total, px, py, target)
                    radius = result['radius']
                    x = np.linspace(max(-15, px-radius), min(15, px+radius), 100001)
                    edge = 6 * np.maximum(0, 1-(x/15)**2)**(1/1.2)
                    disk = np.sqrt(np.maximum(0, radius**2-(x-px)**2))
                    cy = 7.8-py
                    height = np.maximum(0, np.minimum(1.3*edge, cy+disk)-np.maximum(-0.7*edge, cy-disk))
                    cut = app.zone_boundaries(30)[2]
                    # Independent dense Simpson integration, split at thickness steps.
                    edges = sorted({x[0], x[-1]} | {c for c in (-cut, 0, cut) if x[0] < c < x[-1]})
                    independent = 0
                    for left, right in zip(edges[:-1], edges[1:]):
                        sx = np.linspace(left, right, 20001)
                        edge = 6 * np.maximum(0, 1-(sx/15)**2)**(1/1.2)
                        disk = np.sqrt(np.maximum(0, radius**2-(sx-px)**2))
                        height = np.maximum(0, np.minimum(1.3*edge, cy+disk)-np.maximum(-0.7*edge, cy-disk))
                        factor = 1 if abs((left+right)/2) <= cut else 0.82
                        independent += 3*factor*simpson(height, x=sx)
                    self.assertAlmostEqual(independent, target, delta=max(1e-5, target*1e-5))
                    self.assertAlmostEqual(result['volume'], target, delta=1e-5)

    def test_deterministic(self):
        args = (30, 12, 3, 756, 3, 4, 500)
        one = app.keep_only_requested_volume(*args)
        two = app.keep_only_requested_volume(*args)
        self.assertEqual(one['radius'], two['radius'])
        np.testing.assert_array_equal(one['high'], two['high'])

    def test_zone_area_shares_and_total_volume(self):
        for width, length in [(30, 12), (40, 8), (12, 20)]:
            a, b = width/2, length/2
            cuts = (-a, *app.zone_boundaries(width), a)
            area = app.calculate_asymmetric_area(a, b)
            for left, right, expected in zip(cuts[:-1], cuts[1:], (.145, .355, .355, .145)):
                x = np.linspace(left, right, 100001)
                height = 2*b*np.maximum(0, 1-(x/a)**2)**(1/1.2)
                self.assertAlmostEqual(simpson(height, x=x)/area, expected, places=8)
            volumes = app.calculate_zone_volumes(area, 3)
            self.assertAlmostEqual(sum(volumes), area*3*.9478, places=8)
            self.assertEqual(volumes[0], volumes[3])
            self.assertEqual(volumes[1], volumes[2])
            self.assertAlmostEqual(volumes[0]/volumes[1], .145*.82/.355)

    def test_small_regions_use_local_thickness(self):
        total = app.calculate_asymmetric_area(15, 6)*3*.9478
        for px, factor in [(0, 1), (10, .82), (-10, .82)]:
            result = app.keep_only_requested_volume(30, 12, 3, total, px, 7.8, 1)
            self.assertAlmostEqual(np.pi*result['radius']**2*3*factor, 1, places=6)

    def test_cut_crossings_and_symmetry(self):
        total = app.calculate_asymmetric_area(15, 6)*3*.9478
        cut = app.zone_boundaries(30)[2]
        central = app.keep_only_requested_volume(30, 12, 3, total, 0, 7.8, 500)
        self.assertGreater(central['radius'], cut)  # Crosses both boundaries.
        right = app.keep_only_requested_volume(30, 12, 3, total, cut, 7.8, 10)
        left = app.keep_only_requested_volume(30, 12, 3, total, -cut, 7.8, 10)
        self.assertLess(right['radius'], cut)  # Crosses just one boundary.
        self.assertAlmostEqual(right['radius'], left['radius'], places=8)

    def test_http_validation_and_results(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        data = dict(width='30', length='12', thickness='3', Px='3', Py='4', requested_volume='500')
        with patch.object(app, 'visualize_extraction') as plot:
            for change in [dict(requested_volume='1000'), dict(requested_volume='730'), dict(requested_volume='0'), dict(width='nan'), dict(thickness='-1'), dict(Px='100')]:
                response = client.post('/index', data=data | change)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn(b'extraction.png', response.data)
            plot.assert_not_called()
            response = client.post('/index', data=data)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'500.0 cc', response.data)
            self.assertIn(b'Left lateral', response.data)
            self.assertIn(b'14.5%', response.data)
            self.assertIn(b'35.5%', response.data)
            self.assertIn(b'2.46', response.data)
            self.assertIn(b'717 cc', response.data)
            plot.assert_called_once()

if __name__ == '__main__':
    unittest.main()
