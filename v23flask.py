from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import matplotlib.pyplot as plt
#import math
import os
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.integrate import quad
from scipy.optimize import brentq
from functools import lru_cache


app = Flask(__name__)
def auth_bypass_enabled() -> bool:
    return os.getenv("DIEP_AUTH_BYPASS", "").lower() in ("1", "true", "yes", "on")
app.secret_key = 'your_secret_key'  # Change this for security

# User authentication with password hashing
users = {
"admin": generate_password_hash("password123"),
"Adam" : generate_password_hash("qvh_1_test")
}  # Change this for real use

# Function to log visits
def log_visit(user):
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    endpoint = request.path
    log_message = f"{datetime.datetime.now()} - User: {user}, IP: {ip}, User Agent: {user_agent}, Endpoint: {endpoint}\n"
    with open("logs/visits.log", "a") as log_file:
        log_file.write(log_message)

@app.route('/', methods=['GET', 'POST'])
def login():
    if auth_bypass_enabled():
        session['user'] = 'bypass'
        log_visit(session['user'])
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['user'] = username
            log_visit(username)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Trial geometry/thickness model; independent of perfusion.
LATERAL_AREA_SHARE = 0.145  # Each lateral zone, as a share of the full flap.
MEDIAL_AREA_SHARE = 0.5 - LATERAL_AREA_SHARE
LATERAL_THICKNESS_FACTOR = 0.82
TISSUE_DENSITY_G_PER_CC = 0.9


@lru_cache(maxsize=32)
def normalized_zone_boundary(n=2, m=1.2):
    """Positive cut / half-width enclosing the specified medial AREA share."""
    profile = lambda u: (1 - u ** n) ** (1 / m)
    half_area = quad(profile, 0, 1, epsabs=1e-11, epsrel=1e-11)[0]
    return brentq(lambda cut: quad(profile, 0, cut)[0] / half_area
                  - 2 * MEDIAL_AREA_SHARE, 0, 1, xtol=1e-12)


def zone_boundaries(width, n=2, m=1.2):
    cut = width / 2 * normalized_zone_boundary(n, m)
    return (-cut, 0.0, cut)


def total_volume_factor():
    return 2 * (MEDIAL_AREA_SHARE + LATERAL_AREA_SHARE * LATERAL_THICKNESS_FACTOR)


def calculate_zone_volumes(total_area, thickness):
    """Four full-zone volumes, ordered from left to right."""
    medial = total_area * thickness * MEDIAL_AREA_SHARE
    lateral = total_area * thickness * LATERAL_AREA_SHARE * LATERAL_THICKNESS_FACTOR
    return (lateral, medial, medial, lateral)


def flap_bounds(x, a, b, n=2, m=1.2, upper_scale=1.3, lower_scale=0.7):
    """Vertical bounds of the existing asymmetric superellipse."""
    height = b * np.maximum(0, 1 - (np.abs(x) / a) ** n) ** (1 / m)
    return -lower_scale * height, upper_scale * height


def calculate_asymmetric_area(a, b, n=2, m=1.2, upper_scale=1.3, lower_scale=0.7, num_points=300):
    # Integrate one quadrant; num_points remains for call compatibility.
    return 2 * a * b * (upper_scale + lower_scale) * quad(
        lambda u: (1 - u ** n) ** (1 / m), 0, 1,
        epsabs=1e-10, epsrel=1e-10)[0]


def perfusion_zone4_indices(Px, width, n=2, m=1.2):
    """Zones separated from the perforator by at least one whole geometric zone.

    Zone definitions run from left lateral (0) to right lateral (3).
    A perforator at X=0 is invalid because its side is undefined.
    At either medial/lateral boundary, the perforator belongs to the medial zone.
    """
    cuts = zone_boundaries(width, n, m)
    if Px == 0:
        raise ValueError('Perforator X cannot equal 0 (midline).')
    # At an exact medial/lateral boundary, assign the perforator to the
    # medial zone on either side of the midline (mirror-symmetric tie rule).
    perforator_zone = int(np.searchsorted(cuts, Px,
                            side='right' if Px < 0 else 'left'))
    return tuple(i for i in range(4) if abs(i - perforator_zone) >= 2)


def zone4_mask(x, zone_indices, cuts):
    return np.isin(np.searchsorted(cuts, x, side='right'), zone_indices)


@app.route('/index', methods=['GET', 'POST'])
def index():
    if auth_bypass_enabled():
        session.setdefault('user', 'bypass')
    elif 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        fields = ('width', 'length', 'thickness', 'Px', 'Py', 'requested_weight')
        try:
            values = {name: float(request.form[name]) for name in fields}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError("All inputs must be finite numbers.")
            width, length, thickness, Px, Py, requested_weight = (values[name] for name in fields)
            exclude_zone4 = request.form.get('exclude_zone4') == 'on'
            if min(width, length, thickness, requested_weight) <= 0:
                raise ValueError("Dimensions and desired weight must be greater than zero.")
            requested_volume = requested_weight / TISSUE_DENSITY_G_PER_CC
            a, b = width / 2, length / 2
            low, high = flap_bounds(Px, a, b)
            # Py is measured from the upper flap edge at the perforator X.
            Pyc = high - Py
            if abs(Px) > a or not low - 1e-10 <= Pyc <= high + 1e-10:
                raise ValueError("The perforator must lie within the flap boundary.")
            total_area = calculate_asymmetric_area(a, b)
            total_volume = total_area * thickness * total_volume_factor()
            if not np.isfinite(total_volume):
                raise ValueError("The dimensions are too large to calculate.")
            if not exclude_zone4 and requested_volume > total_volume:
                raise ValueError(f"Desired weight exceeds the available {total_volume * TISSUE_DENSITY_G_PER_CC:.2f} g.")
            extraction = keep_only_requested_volume(
                width, length, thickness, total_volume, Px, Pyc, requested_volume,
                exclude_zone4=exclude_zone4)
        except (ValueError, KeyError) as exc:
            return render_template('index.html', error=str(exc), user=session['user'],
                                   exclude_zone4=request.form.get('exclude_zone4') == 'on'), 400

        hemi_volume = total_volume / 2
        total_weight = total_volume * TISSUE_DENSITY_G_PER_CC
        hemi_weight = hemi_volume * TISSUE_DENSITY_G_PER_CC
        zone_rows = list(zip(
            ('Left lateral', 'Left medial', 'Right medial', 'Right lateral'),
            (LATERAL_AREA_SHARE, MEDIAL_AREA_SHARE, MEDIAL_AREA_SHARE, LATERAL_AREA_SHARE),
            (LATERAL_THICKNESS_FACTOR * thickness, thickness, thickness, LATERAL_THICKNESS_FACTOR * thickness),
            calculate_zone_volumes(total_area, thickness)))
        extracted_width, extracted_length = extraction_dimensions(extraction)
        visualize_extraction(width, length, Px, Pyc, extraction, requested_volume, total_volume)

        total_volume = int(round(total_volume))
        total_weight = int(round(total_weight))
        hemi_volume = int(round(hemi_volume))
        hemi_weight = int(round(hemi_weight))
        total_area = int(round(total_area))



        return render_template(
            'index.html', width=width, total_area=total_area, length=length,
            thickness=thickness, Px=Px, Py=Py, requested_weight=requested_weight,
            density=TISSUE_DENSITY_G_PER_CC,
            retained_weight=round(extraction['volume'] * TISSUE_DENSITY_G_PER_CC, 2),
            total_volume=total_volume, hemi_volume=hemi_volume,
            hemi_weight=hemi_weight, total_weight=total_weight,
            zone_rows=zone_rows, extracted_width=extracted_width,
            extracted_length=extracted_length, retained_volume=round(extraction['volume'], 2),
            user=session['user'], exclude_zone4=exclude_zone4,
            zone4_volume=round(extraction['zone4_volume'], 2),
            zone4_weight=round(extraction['zone4_volume'] * TISSUE_DENSITY_G_PER_CC, 2),
            zone4_limited=extraction['zone4_limited'],
            max_valid_weight=round(extraction['max_valid_volume'] * TISSUE_DENSITY_G_PER_CC, 2))

    return render_template('index.html', user=session['user'], exclude_zone4=False)

def extraction_dimensions(extraction):
    """Bounding dimensions shared by the summary and plot."""
    valid = extraction['high'] >= extraction['low']
    if extraction.get('exclude_zone4', False):
        cuts = zone_boundaries(extraction['width'], extraction['n'], extraction['m'])
        valid &= ~zone4_mask(extraction['x'], extraction['zone4_indices'], cuts)
    return (np.ptp(extraction['x'][valid]),
            max(extraction['high'][valid]) - min(extraction['low'][valid]))


def retained_bounds(x, a, b, Px, Pyc, radius, n=2, m=1.2, upper_scale=1.3, lower_scale=0.7):
    low, high = flap_bounds(x, a, b, n, m, upper_scale, lower_scale)
    circle_height = np.sqrt(np.maximum(0, radius ** 2 - (x - Px) ** 2))
    return np.maximum(low, Pyc - circle_height), np.minimum(high, Pyc + circle_height)


def keep_only_requested_volume(width, length, thickness, total_volume, Px, Pyc, required_volume,
                               n=2, m=1.2, upper_scale=1.3, lower_scale=0.7,
                               exclude_zone4=False):
    """Extract a perforator-centred disk, optionally excluding perfusion Zone IV.

    All radius integrations are split at the existing four geometric zones.
    Pyc is the Cartesian perforator Y calculated by index(); no conversion here.
    """
    if not np.isfinite([width, length, thickness, Px, Pyc, required_volume]).all():
        raise ValueError("Inputs must be finite.")
    if min(width, length, thickness, required_volume) <= 0:
        raise ValueError("Dimensions and required volume must be positive.")
    a, b = width / 2, length / 2
    low, high = flap_bounds(Px, a, b, n, m, upper_scale, lower_scale)
    if abs(Px) > a or not low - 1e-10 <= Pyc <= high + 1e-10:
        raise ValueError("The perforator must lie within the flap boundary.")

    total_area = calculate_asymmetric_area(a, b, n, m, upper_scale, lower_scale)
    available = total_area * thickness * total_volume_factor()
    cuts = zone_boundaries(width, n, m)
    zone4_indices = perfusion_zone4_indices(Px, width, n, m)
    zone_maxima = calculate_zone_volumes(total_area, thickness)
    max_valid_volume = available - sum(zone_maxima[i] for i in zone4_indices)

    if not exclude_zone4 and required_volume > available:
        raise ValueError(f"Required volume exceeds the available {available:.2f} cc.")

    def per_zone_volumes(radius):
        left, right = max(-a, Px - radius), min(a, Px + radius)
        volumes = np.zeros(4)
        if left >= right:
            return volumes

        def height(x):
            bottom, top = retained_bounds(x, a, b, Px, Pyc, radius,
                                          n, m, upper_scale, lower_scale)
            return max(0.0, top - bottom)

        edges = [left] + [cut for cut in cuts if left < cut < right] + [right]
        for left_edge, right_edge in zip(edges[:-1], edges[1:]):
            idx = int(np.searchsorted(cuts, (left_edge + right_edge) / 2, side='right'))
            factor = 1.0 if idx in (1, 2) else LATERAL_THICKNESS_FACTOR
            integral = quad(height, left_edge, right_edge,
                            points=np.linspace(left_edge, right_edge, 17)[1:-1],
                            epsabs=max(1e-12, required_volume / thickness * 1e-7 / len(edges)),
                            epsrel=1e-7, limit=250)[0]
            volumes[idx] += thickness * factor * integral
        return volumes

    def included_volumes(radius):
        volumes = per_zone_volumes(radius)
        if exclude_zone4:
            volumes[list(zone4_indices)] = 0
        return volumes

    def volume(radius):
        return float(included_volumes(radius).sum())

    max_radius = max(np.hypot(x - Px, y - Pyc)
                     for x in (-a, a) for y in (-b * lower_scale, b * upper_scale))
    capacity = max_valid_volume if exclude_zone4 else available
    target_volume = min(required_volume, capacity) if exclude_zone4 else required_volume
    zone4_limited = bool(exclude_zone4 and required_volume > capacity + 1e-8)

    if np.isclose(target_volume, capacity, rtol=1e-12, atol=1e-9):
        radius = max_radius
    else:
        radius = brentq(lambda r: volume(r) - target_volume, 0, max_radius,
                        xtol=1e-10, rtol=1e-12)

    zone_volumes = included_volumes(radius)
    actual_volume = float(zone_volumes.sum())
    x = np.linspace(max(-a, Px - radius), min(a, Px + radius), 2001)
    low, high = retained_bounds(x, a, b, Px, Pyc, radius,
                                n, m, upper_scale, lower_scale)
    return dict(x=x, low=low, high=high, radius=radius, volume=actual_volume,
                zone_volumes=zone_volumes, zone4_indices=zone4_indices,
                zone4_volume=float(sum(zone_volumes[i] for i in zone4_indices)),
                max_valid_volume=max_valid_volume,
                exclude_zone4=exclude_zone4, zone4_limited=zone4_limited,
                width=width, n=n, m=m)

def visualize_extraction(width, length, Px, Pyc, extraction, requested_volume, total_volume,
                         n=2, m=1.2, upper_scale=1.3, lower_scale=0.7):
    a, b = width / 2, length / 2
    x = np.linspace(-a, a, 2001)
    low, high = flap_bounds(x, a, b, n, m, upper_scale, lower_scale)
    fig, ax = plt.subplots(figsize=(10, 10))
    try:
        ax.set_xlim(-a - 5, a + 5)
        ax.set_ylim(-b * lower_scale - 5, b * upper_scale + 5)
        # Display-only Y coordinates: 0 at the highest superior incision
        # (x=0), increasing inferiorly. All geometry stays Cartesian.
        y_ticks = np.linspace(-b * lower_scale, b * upper_scale, 7)
        ax.set_yticks(y_ticks, [f'{b * upper_scale - y:.1f}' for y in y_ticks])
        ax.set_ylabel("Distance below highest superior incision (cm)")
        ax.set_aspect('equal')
        ax.plot(x, high, 'b-', label="Abdominoplasty Boundary")
        ax.plot(x, low, 'b-')
        for i, cut in enumerate(zone_boundaries(width, n, m)):
            zone_low, zone_high = flap_bounds(cut, a, b, n, m, upper_scale, lower_scale)
            ax.plot([cut, cut], [zone_low, zone_high], color='gray', linestyle='--',
                    label="Geometric zone boundaries" if i == 0 else None)
        ax.plot(Px, Pyc, 'ko', markersize=8, label="Perforator")
        valid = extraction['high'] >= extraction['low']
        zone4 = zone4_mask(extraction['x'], extraction['zone4_indices'],
                           zone_boundaries(width, n, m))
        ax.fill_between(extraction['x'], extraction['low'], extraction['high'],
                        where=valid & ~zone4, color='red', alpha=0.5,
                        label="Extracted Region")
        if extraction['exclude_zone4']:
            excluded_on_flap = zone4_mask(x, extraction['zone4_indices'],
                                          zone_boundaries(width, n, m))
            ax.fill_between(x, low, high, where=excluded_on_flap,
                            color='lightgray', alpha=0.8, label="Excluded Zone 4")
        elif np.any(valid & zone4):
            ax.fill_between(extraction['x'], extraction['low'], extraction['high'],
                            where=valid & zone4, color='purple', alpha=0.6,
                            label="Zone 4 Portion")
        extracted_width, extracted_length = extraction_dimensions(extraction)
        ax.set_title(f"DIEP Flap {extraction['volume'] * TISSUE_DENSITY_G_PER_CC:.1f} g | Approx. dimensions: {extracted_width:.1f}cm x {extracted_length:.1f}cm")
        ax.legend()
        ax.grid()
        fig.savefig('static/extraction.png', bbox_inches='tight', pad_inches=0.5)
    finally:
        plt.close(fig)


if __name__ == '__main__':
    app.run(debug=True)
