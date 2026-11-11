import os
import base64
import re
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
        for px, pyc in [(3, 3.8), (0.01, 7.7), (0.01, -4.1), (15, 0), (10, 0)]:
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
        for px, factor in [(0.01, 1), (10, .82), (-10, .82)]:
            result = app.keep_only_requested_volume(30, 12, 3, total, px, 0, 1)
            self.assertAlmostEqual(np.pi*result['radius']**2*3*factor, 1, places=6)

    def test_cut_crossings_and_symmetry(self):
        total = app.calculate_asymmetric_area(15, 6)*3*.9478
        cut = app.zone_boundaries(30)[2]
        central = app.keep_only_requested_volume(30, 12, 3, total, 0.01, 0, 500)
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
            self.assertIn(b'Estimated weight:', response.data)
            self.assertIn(b'Estimated volume:', response.data)
            self.assertIn(b'Zone 4 contribution:', response.data)
            self.assertNotIn(b'Estimated retained weight:', response.data)
            self.assertNotIn(b'Estimated retained volume:', response.data)
            # The extraction and plot receive the CT midline-referenced Y.
            expected_pyc = app.flap_bounds(0, 15, 6)[1] - 4
            self.assertAlmostEqual(plot.call_args.args[3], expected_pyc)

    def test_plot_is_embedded_in_memory_per_request(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        base = dict(width='30', length='12', thickness='3',
                    Px='3', Py='4', requested_weight='500')
        encoded_images = []
        for data in (base, base | dict(Px='4', requested_weight='450')):
            response = client.post('/index', data=data)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(b'/static/extraction.png', response.data)
            match = re.search(rb'<img src="data:image/png;base64,([^"]+)"',
                              response.data)
            self.assertIsNotNone(match)
            png = base64.b64decode(match.group(1), validate=True)
            self.assertTrue(png.startswith(bytes.fromhex('89504e470d0a1a0a')))
            self.assertGreater(len(png), 1000)
            encoded_images.append(png)

        # A later request must not replace or reuse an earlier response image.
        self.assertNotEqual(encoded_images[0], encoded_images[1])

    def test_perforator_y_is_measured_from_ct_superior_midline(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        base = dict(width='30', length='12', thickness='3',
                    requested_weight='1')
        ct_top = app.flap_bounds(0, 15, 6)[1]

        with patch.object(app, 'visualize_extraction') as plot:
            for px in (-10, 10):
                local_low, local_high = app.flap_bounds(px, 15, 6)
                # The CT origin stays at the superior MIDLINE, so the
                # local upper incision is below 0 cm in CT measurements.
                for py in (ct_top - local_high, ct_top - local_low, 4):
                    with self.subTest(px=px, py=py):
                        response = client.post('/index',
                            data=base | dict(Px=str(px), Py=str(py)))
                        self.assertEqual(response.status_code, 200)
                        self.assertAlmostEqual(plot.call_args.args[3],
                                               ct_top - py, places=10)

                # A perforator above its local incision is invalid, even
                # if the supplied CT slice remains below the midline apex.
                response = client.post('/index', data=base | dict(
                    Px=str(px), Py=str(ct_top - local_high - 0.1)))
                self.assertEqual(response.status_code, 400)

                # Likewise a CT slice below the local inferior incision.
                response = client.post('/index', data=base | dict(
                    Px=str(px), Py=str(ct_top - local_low + 0.1)))
                self.assertEqual(response.status_code, 400)

            # The same CT Y produces the same Cartesian Y for different X.
            self.assertEqual(ct_top, app.flap_bounds(0, 15, 6)[1])

    def test_ct_visualization_draws_midline_origin_and_xy_guides(self):
        width, length, thickness, px, measured_y = 30, 12, 3, -10, 4
        a, b = width / 2, length / 2
        ct_top = app.flap_bounds(0, a, b)[1]
        pyc = ct_top - measured_y
        total = app.calculate_asymmetric_area(a, b) * thickness * .9478
        extraction = app.keep_only_requested_volume(
            width, length, thickness, total, px, pyc, 1)

        def capture(fig, *args, **kwargs):
            ax = fig.axes[0]
            origin = next(line for line in ax.lines
                          if line.get_label() == 'CT origin (0, 0)')
            x_guide = next(line for line in ax.lines
                           if line.get_label() == 'CT X guide')
            y_guide = next(line for line in ax.lines
                           if line.get_label() == 'CT Y guide')
            perforator = next(line for line in ax.lines
                              if line.get_label() == 'Perforator')
            np.testing.assert_allclose(
                [origin.get_xdata()[0], origin.get_ydata()[0]],
                [0, ct_top])
            np.testing.assert_allclose(
                x_guide.get_xdata(), [0, px])
            np.testing.assert_allclose(
                x_guide.get_ydata(), [ct_top, ct_top])
            np.testing.assert_allclose(
                y_guide.get_xdata(), [px, px])
            np.testing.assert_allclose(
                y_guide.get_ydata(), [ct_top, pyc])
            np.testing.assert_allclose(
                [perforator.get_xdata()[0], perforator.get_ydata()[0]],
                [px, pyc])
            labels = [text.get_text() for text in ax.texts]
            self.assertIn('X = -10.0 cm', labels)
            self.assertIn('Y = 4.0 cm', labels)

        with patch('matplotlib.figure.Figure.savefig', capture):
            app.visualize_extraction(
                width, length, px, pyc, extraction, 1, total)

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

    def test_zone4_identification_relative_to_perforator(self):
        for px, expected in [(-12, (2, 3)), (-3, (3,)),
                             (3, (0,)), (12, (0, 1))]:
            with self.subTest(px=px):
                self.assertEqual(app.perfusion_zone4_indices(px, 30), expected)

    def test_midline_perforator_rejected(self):
        total = app.calculate_asymmetric_area(15, 6) * 3 * .9478
        for mode in (False, True):
            with self.subTest(exclude_zone4=mode):
                with self.assertRaisesRegex(ValueError, "Perforator X cannot equal 0"):
                    app.keep_only_requested_volume(
                        30, 12, 3, total, 0, 3.8, 100, exclude_zone4=mode)

        with self.assertRaisesRegex(ValueError, "Perforator X cannot equal 0"):
            app.perfusion_zone4_indices(0, 30)

        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        data = dict(width='30', length='12', thickness='3',
                    Px='0', Py='4', requested_weight='500')
        with patch.object(app, 'visualize_extraction') as plot:
            for selection in ({}, {'exclude_zone4': 'on'}):
                response = client.post('/index', data=data | selection)
                self.assertEqual(response.status_code, 400)
                self.assertIn(b'Perforator X cannot equal 0', response.data)
            plot.assert_not_called()

    def test_zone4_medial_lateral_boundary_is_mirror_symmetric(self):
        width, length, thickness = 30, 12, 3
        cut = app.zone_boundaries(width)[2]
        self.assertEqual(app.perfusion_zone4_indices(-cut, width), (3,))
        self.assertEqual(app.perfusion_zone4_indices(cut, width), (0,))

        # Immediately lateral to each boundary, both cases classify
        # two contralateral zones as IV; immediately medial, only one.
        eps = 1e-6
        for offset, expected_left, expected_right in [
            (-eps, (2, 3), (0, 1)),
            (0, (3,), (0,)),
            (eps, (3,), (0,)),
        ]:
            with self.subTest(offset=offset):
                self.assertEqual(
                    app.perfusion_zone4_indices(-cut + offset, width), expected_left)
                self.assertEqual(
                    app.perfusion_zone4_indices(cut - offset, width), expected_right)

        total = app.calculate_asymmetric_area(width / 2, length / 2) * thickness * .9478
        results = []
        for px in (-cut, cut):
            # Same 4 cm distance from the local upper incision.
            pyc = app.flap_bounds(px, width / 2, length / 2)[1] - 4
            extracted = app.keep_only_requested_volume(
                width, length, thickness, total, px, pyc,
                600 / .9, exclude_zone4=True)
            results.append(extracted)
            self.assertTrue(extracted['zone4_limited'])
            self.assertAlmostEqual(
                extracted['volume'] * app.TISSUE_DENSITY_G_PER_CC, 564.11,
                delta=.02)
            self.assertEqual(extracted['zone4_volume'], 0)

        self.assertAlmostEqual(results[0]['volume'], results[1]['volume'], places=7)
        self.assertAlmostEqual(results[0]['radius'], results[1]['radius'], places=7)
        np.testing.assert_allclose(
            results[0]['zone_volumes'], results[1]['zone_volumes'][::-1],
            atol=1e-5, rtol=1e-6)

    def test_zone4_unticked_preserves_requested_volume_and_tracks_zone4(self):
        total = app.calculate_asymmetric_area(15, 6) * 3 * .9478
        pyc = app.flap_bounds(-3, 15, 6)[1] - 4
        target = 600 / .9
        original = app.keep_only_requested_volume(30, 12, 3, total, -3, pyc, target)
        self.assertAlmostEqual(original['volume'], target, places=5)
        self.assertAlmostEqual(sum(original['zone_volumes']), target, places=5)
        self.assertGreater(original['zone4_volume'], 0)
        self.assertAlmostEqual(original['zone4_volume'],
                               original['zone_volumes'][3], places=5)
        self.assertFalse(original['zone4_limited'])

        mirrored = app.keep_only_requested_volume(30, 12, 3, total, 3, pyc, target)
        self.assertAlmostEqual(mirrored['zone4_volume'],
                               original['zone4_volume'], places=5)

    def test_zone4_exclusion_solves_target_or_caps_at_three_zone_capacity(self):
        total = app.calculate_asymmetric_area(15, 6) * 3 * .9478
        pyc = app.flap_bounds(-3, 15, 6)[1] - 4
        smaller = app.keep_only_requested_volume(
            30, 12, 3, total, -3, pyc, 300 / .9, exclude_zone4=True)
        self.assertAlmostEqual(smaller['volume'], 300 / .9, places=5)
        self.assertEqual(smaller['zone4_volume'], 0)
        self.assertFalse(smaller['zone4_limited'])

        limited = app.keep_only_requested_volume(
            30, 12, 3, total, -3, pyc, 600 / .9, exclude_zone4=True)
        max_volume = total - app.calculate_zone_volumes(
            app.calculate_asymmetric_area(15, 6), 3)[3]
        self.assertTrue(limited['zone4_limited'])
        self.assertEqual(limited['zone4_volume'], 0)
        self.assertEqual(limited['zone_volumes'][3], 0)
        self.assertAlmostEqual(limited['volume'], max_volume, places=5)
        self.assertAlmostEqual(limited['max_valid_volume'], max_volume, places=5)
        self.assertLess(app.extraction_dimensions(limited)[0], 30)

        # Even a request above the whole-flap capacity is capped when
        # exclusion is enabled, rather than rejected.
        bigger = app.keep_only_requested_volume(
            30, 12, 3, total, -3, pyc, 800 / .9, exclude_zone4=True)
        self.assertAlmostEqual(bigger['volume'], max_volume, places=5)

    def test_zone4_plot_colours_and_exclusion(self):
        total = app.calculate_asymmetric_area(15, 6) * 3 * .9478
        pyc = app.flap_bounds(-3, 15, 6)[1] - 4
        for exclude, target, expected_label in [
            (False, 600 / .9, "Zone 4 Portion"),
            (True, 600 / .9, "Excluded Zone 4"),
        ]:
            with self.subTest(exclude=exclude):
                extraction = app.keep_only_requested_volume(
                    30, 12, 3, total, -3, pyc, target,
                    exclude_zone4=exclude)
                def capture(fig, *args, **kwargs):
                    labels = [c.get_label() for c in fig.axes[0].collections]
                    self.assertIn("Extracted Region", labels)
                    self.assertIn(expected_label, labels)
                    self.assertEqual(
                        "Excluded Zone 4" in labels, exclude)
                with patch('matplotlib.figure.Figure.savefig', capture):
                    app.visualize_extraction(
                        30, 12, -3, pyc, extraction, target, total)

    def test_zone4_checkbox_http_and_capped_weight(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        data = dict(width='30', length='12', thickness='3',
                    Px='-3', Py='4', requested_weight='600')
        with patch.object(app, 'visualize_extraction'):
            included = client.post('/index', data=data)
            self.assertEqual(included.status_code, 200)
            self.assertNotIn(b'zone 4 needed to satisfy request', included.data)
            self.assertIn(b'Exclude zone 4', included.data)
            excluded = client.post('/index',
                                   data=data | dict(exclude_zone4='on'))
            self.assertEqual(excluded.status_code, 200)
            self.assertIn(b'zone 4 needed to satisfy request', excluded.data)
            self.assertIn(b'Estimated weight:', excluded.data)
            self.assertIn(b'Zone 4 contribution: 0.0 g', excluded.data)
            self.assertIn(b'checked', excluded.data)
            # Request above full capacity remains an error unless Zone IV
            # is excluded, in which case the attainable maximum is shown.
            self.assertEqual(client.post('/index',
                data=data | dict(requested_weight='800')).status_code, 400)
            self.assertEqual(client.post('/index',
                data=data | dict(requested_weight='800', exclude_zone4='on')).status_code, 200)

    def test_legacy_volume_input_not_reinterpreted_as_grams(self):
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'test'
        response = client.post('/index', data=dict(width=35, length=12,
            thickness=2.6, Px=3.5, Py=1, requested_volume=500))
        self.assertEqual(response.status_code, 400)

    def test_weight_to_volume_conversion_affects_extraction_radius(self):
        total = app.calculate_asymmetric_area(17.5, 6) * 2.6 * .9478
        pyc = app.flap_bounds(3.5, 17.5, 6)[1] - 1
        smaller = app.keep_only_requested_volume(
            35, 12, 2.6, total, 3.5, pyc, 500)
        larger = app.keep_only_requested_volume(
            35, 12, 2.6, total, 3.5, pyc, 500 / .9)
        self.assertGreater(larger['radius'], smaller['radius'])

    def test_plot_has_no_title_and_legend_is_upper_right(self):
        total = app.calculate_asymmetric_area(15, 6) * 3 * .9478
        px = -3
        pyc = app.flap_bounds(0, 15, 6)[1] - 4

        for exclude, target, has_zone4 in (
            (False, 600/.9, True),
            (True, 600/.9, True),
            (False, 1, False),
        ):
            with self.subTest(exclude=exclude, target=target):
                result = app.keep_only_requested_volume(
                    30, 12, 3, total, px, pyc, target,
                    exclude_zone4=exclude)

                def capture(fig, *args, **kwargs):
                    ax = fig.axes[0]
                    self.assertEqual(ax.get_title(), '')
                    self.assertEqual(tuple(fig.get_size_inches()), (10, 6.5))
                    legend = ax.get_legend()
                    self.assertEqual(legend._loc, 1)  # upper right
                    expected = ['Geometric boundaries', 'Perforator',
                                'Estimated flap']
                    if has_zone4:
                        expected.append('Zone 4')
                    self.assertEqual(
                        [text.get_text() for text in legend.get_texts()],
                        expected)
                    # The outline and CT guides remain plotted.
                    labels = {line.get_label() for line in ax.lines}
                    self.assertIn('Abdominoplasty Boundary', labels)
                    self.assertIn('CT X guide', labels)
                    self.assertIn('CT Y guide', labels)

                with patch('matplotlib.figure.Figure.savefig', capture):
                    app.visualize_extraction(
                        30, 12, px, pyc, result, target, total)


if __name__ == '__main__':
    unittest.main()
