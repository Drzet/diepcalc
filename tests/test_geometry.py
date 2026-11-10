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
        for px, pyc in [(3, 3.8), (0, 7.8), (0, -4.2), (15, 0), (10, 0)]:
            for target in [0.1, 10, 200, 500, total]:
                with self.subTest(px=px, pyc=pyc, target=target):
                    result = app.keep_only_requested_volume(30, 12, 3, total, px, pyc, target)
                    radius = result['radius']
                    x = np.linspace(max(-15, px-radius), min(15, px+radius), 100001)
                    edge = 6 * np.maximum(0, 1-(x/15)**2)**(1/1.2)
                    disk = np.sqrt(np.maximum(0, radius**2-(x-px)**2))
                    cy = pyc
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
        args = (30, 12, 3, 756, 3, 3.8, 500)
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
            result = app.keep_only_requested_volume(30, 12, 3, total, px, 0, 1)
            self.assertAlmostEqual(np.pi*result['radius']**2*3*factor, 1, places=6)

    def test_cut_crossings_and_symmetry(self):
        total = app.calculate_asymmetric_area(15, 6)*3*.9478
        cut = app.zone_boundaries(30)[2]
        central = app.keep_only_requested_volume(30, 12, 3, total, 0, 0, 500)
        self.assertGreater(central['radius'], cut)  # Crosses both boundaries.
        right = app.keep_only_requested_volume(30, 12, 3, total, cut, 0, 10)
        left = app.keep_only_requested_volume(30, 12, 3, total, -cut, 0, 10)
        self.assertLess(right['radius'], cut)  # Crosses just one boundary.
        self.assertAlmostEqual(right['radius'], left['radius'], places=8)

    def test_http_validation_and_results(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        data = dict(width='30', length='12', thickness='3', Px='3', Py='4', requested_weight='500')
        with patch.object(app, 'visualize_extraction') as plot:
            for change in [dict(requested_weight='1000'), dict(requested_weight='730'), dict(requested_weight='0'), dict(width='nan'), dict(thickness='-1'), dict(Px='100')]:
                response = client.post('/index', data=data | change)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn(b'extraction.png', response.data)
            plot.assert_not_called()
            response = client.post('/index', data=data)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'500.0 g', response.data)
            self.assertIn(b'717 cc', response.data)
            self.assertNotIn(b'Geometric zone</th>', response.data)
            self.assertNotIn(b'Trial geometry', response.data)
            self.assertLess(response.data.index(b'id="flap-summary"'), response.data.index(b'<img'))
            self.assertLess(response.data.index(b'<img'), response.data.index(b'Total area:'))
            dims = app.extraction_dimensions(plot.call_args.args[4])
            self.assertIn(f'{dims[0]:.1f} × {dims[1]:.1f} cm'.encode(), response.data)
            plot.assert_called_once()
            self.assertAlmostEqual(plot.call_args.args[4]['volume'], 500 / .9, places=5)
            self.assertIn(b'555.56 cc', response.data)
            # The extraction and plot receive the same local-edge Cartesian Y.
            expected_pyc = app.flap_bounds(3, 15, 6)[1] - 4
            self.assertAlmostEqual(plot.call_args.args[3], expected_pyc)

    def test_perforator_y_is_measured_from_local_upper_edge(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        base = dict(width='30', length='12', thickness='3',
                    Px='10', requested_weight='1')
        low, high = app.flap_bounds(10, 15, 6)
        local_height = high - low

        with patch.object(app, 'visualize_extraction') as plot:
            for measured_py, expected_y in [(0, high), (local_height, low),
                                            (2, high - 2)]:
                with self.subTest(py=measured_py):
                    response = client.post('/index',
                        data=base | dict(Py=str(measured_py)))
                    self.assertEqual(response.status_code, 200)
                    self.assertAlmostEqual(plot.call_args.args[3],
                                           expected_y, places=10)

            # Beyond the local lower incision must be rejected even if
            # it would fall inside the flap using the former midline origin.
            response = client.post('/index',
                data=base | dict(Py=str(local_height + 0.1)))
            self.assertEqual(response.status_code, 400)

    def test_internal_geometry_uses_cartesian_y_without_reconversion(self):
        total = app.calculate_asymmetric_area(15, 6) * 3 * .9478
        pyc = 0.75
        extraction = app.keep_only_requested_volume(
            30, 12, 3, total, 10, pyc, 1)
        self.assertGreater(extraction['radius'], 0)
        found = []
        def capture(fig, *args, **kwargs):
            found.extend(line.get_ydata()[0] for line in fig.axes[0].lines
                         if line.get_label() == 'Perforator')
        with patch('matplotlib.figure.Figure.savefig', capture):
            app.visualize_extraction(30, 12, 10, pyc, extraction, 1, total)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0], pyc)

    def test_plot_y_axis_relabel_does_not_change_geometry(self):
        # The superior midline is 0 cm; the lower midline is the full
        # flap length. The underlying Cartesian coordinates remain intact.
        width, length, thickness, px, pyc = 30, 12, 3, 10, 0.75
        a, b = width / 2, length / 2
        total = app.calculate_asymmetric_area(a, b) * thickness * .9478
        extraction = app.keep_only_requested_volume(
            width, length, thickness, total, px, pyc, 1)

        def capture(fig, *args, **kwargs):
            ax = fig.axes[0]
            ticks = ax.get_yticks()
            labels = [tick.get_text() for tick in ax.get_yticklabels()]
            self.assertEqual(len(ticks), 7)
            self.assertEqual(labels[-1], '0.0')
            self.assertEqual(labels[0], '12.0')
            for tick, label in zip(ticks, labels):
                self.assertAlmostEqual(float(label), b * 1.3 - tick,
                                       places=1)
            self.assertEqual(ax.get_ylabel(),
                             "Distance below highest superior incision (cm)")
            self.assertEqual(ax.get_xlim(), (-a - 5, a + 5))
            self.assertEqual(ax.get_ylim(), (-b * .7 - 5, b * 1.3 + 5))
            edge = next(line for line in ax.lines
                        if line.get_label() == 'Abdominoplasty Boundary')
            np.testing.assert_allclose(
                edge.get_ydata(),
                app.flap_bounds(edge.get_xdata(), a, b)[1])
            perforator = next(line for line in ax.lines
                              if line.get_label() == 'Perforator')
            self.assertAlmostEqual(perforator.get_xdata()[0], px)
            self.assertAlmostEqual(perforator.get_ydata()[0], pyc)

        with patch('matplotlib.figure.Figure.savefig', capture):
            app.visualize_extraction(
                width, length, px, pyc, extraction, 1, total)

    def test_legacy_volume_input_not_reinterpreted_as_grams(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        response = client.post('/index', data=dict(width=35, length=12,
            thickness=2.6, Px=3.5, Py=1, requested_volume=500))
        self.assertEqual(response.status_code, 400)

    def test_plot_title_uses_weight_and_larger_region(self):
        total = app.calculate_asymmetric_area(17.5, 6)*2.6*.9478
        pyc = app.flap_bounds(3.5, 17.5, 6)[1] - 1
        old = app.keep_only_requested_volume(35, 12, 2.6, total, 3.5, pyc, 500)
        new = app.keep_only_requested_volume(35, 12, 2.6, total, 3.5, pyc, 500/.9)
        self.assertGreater(new['radius'], old['radius'])
        titles = []
        def capture(fig, *args, **kwargs):
            titles.append(fig.axes[0].get_title())
        with patch('matplotlib.figure.Figure.savefig', capture):
            app.visualize_extraction(35, 12, 3.5, pyc, new, 500/.9, total)
        self.assertIn('500.0 g', titles[0])
        self.assertNotIn('cc', titles[0])

if __name__ == '__main__':
    unittest.main()
