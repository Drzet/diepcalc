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
        total = app.calculate_asymmetric_area(15, 6) * 3
        for px, py in [(3, 4), (0, 0), (0, 12), (15, 7.8)]:
            for target in [0.1, 10, 200, 500, total]:
                with self.subTest(px=px, py=py, target=target):
                    result = app.keep_only_requested_volume(30, 12, 3, total, px, py, target)
                    radius = result['radius']
                    x = np.linspace(max(-15, px-radius), min(15, px+radius), 100001)
                    edge = 6 * np.maximum(0, 1-(x/15)**2)**(1/1.2)
                    disk = np.sqrt(np.maximum(0, radius**2-(x-px)**2))
                    cy = 7.8-py
                    height = np.maximum(0, np.minimum(1.3*edge, cy+disk)-np.maximum(-0.7*edge, cy-disk))
                    independent = 3*simpson(height, x=x)
                    self.assertAlmostEqual(independent, target, delta=max(1e-5, target*1e-5))
                    self.assertAlmostEqual(result['volume'], target, delta=1e-5)

    def test_deterministic(self):
        args = (30, 12, 3, 756, 3, 4, 500)
        one = app.keep_only_requested_volume(*args)
        two = app.keep_only_requested_volume(*args)
        self.assertEqual(one['radius'], two['radius'])
        np.testing.assert_array_equal(one['high'], two['high'])

    def test_zones_match_uniform_grid_area(self):
        x = np.linspace(-1, 1, 200001)
        profile = (1-x*x)**(1/1.2)
        mask = abs(x) <= .85
        fraction = simpson(profile[mask], x=x[mask])/simpson(profile, x=x)
        central = app.calculate_zone_volume(30, 0, 1000)
        outer = app.calculate_zone_volume(30, 14, 1000)
        self.assertAlmostEqual(central/1000, fraction, delta=1e-5)
        self.assertAlmostEqual(central + outer, 1000)
        self.assertEqual(outer, app.calculate_zone_volume(30, -14, 1000))
        self.assertGreater(central, 850)

    def test_http_validation_and_results(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        data = dict(width='30', length='12', thickness='3', Px='3', Py='4', requested_volume='500')
        with patch.object(app, 'visualize_extraction') as plot:
            for change in [dict(requested_volume='1000'), dict(requested_volume='0'), dict(width='nan'), dict(thickness='-1'), dict(Px='100')]:
                response = client.post('/index', data=data | change)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn(b'extraction.png', response.data)
            plot.assert_not_called()
            response = client.post('/index', data=data)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'500.0 cc', response.data)
            self.assertIn(b'Central geometric band', response.data)
            plot.assert_called_once()

if __name__ == '__main__':
    unittest.main()
