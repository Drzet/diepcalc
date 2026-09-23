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


@app.route('/index', methods=['GET', 'POST'])
def index():
    if auth_bypass_enabled():
        session.setdefault('user', 'bypass')
    elif 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        fields = ('width', 'length', 'thickness', 'Px', 'Py', 'requested_volume')
        try:
            values = {name: float(request.form[name]) for name in fields}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError("All inputs must be finite numbers.")
            width, length, thickness, Px, Py, requested_volume = (values[name] for name in fields)
            if min(width, length, thickness, requested_volume) <= 0:
                raise ValueError("Dimensions and required volume must be greater than zero.")
            a, b = width / 2, length / 2
            low, high = flap_bounds(Px, a, b)
            Pyc = b * 1.3 - Py
            if abs(Px) > a or not low - 1e-10 <= Pyc <= high + 1e-10:
                raise ValueError("The perforator must lie within the flap boundary.")
            total_area = calculate_asymmetric_area(a, b)
            total_volume = total_area * thickness * total_volume_factor()
            if not np.isfinite(total_volume):
                raise ValueError("The dimensions are too large to calculate.")
            if requested_volume > total_volume:
                raise ValueError(f"Required volume exceeds the available {total_volume:.2f} cc.")
            extraction = keep_only_requested_volume(
                width, length, thickness, total_volume, Px, Py, requested_volume)
        except (ValueError, KeyError) as exc:
            return render_template('index.html', error=str(exc), user=session['user']), 400

        hemi_volume = total_volume / 2
        total_weight = total_volume * 0.9
        hemi_weight = hemi_volume * 0.9
        zone_rows = list(zip(
            ('Left lateral', 'Left medial', 'Right medial', 'Right lateral'),
            (LATERAL_AREA_SHARE, MEDIAL_AREA_SHARE, MEDIAL_AREA_SHARE, LATERAL_AREA_SHARE),
            (LATERAL_THICKNESS_FACTOR * thickness, thickness, thickness, LATERAL_THICKNESS_FACTOR * thickness),
            calculate_zone_volumes(total_area, thickness)))
        visualize_extraction(width, length, Px, Py, extraction, requested_volume, total_volume)

        total_volume = int(round(total_volume))
        total_weight = int(round(total_weight))
        hemi_volume = int(round(hemi_volume))
        hemi_weight = int(round(hemi_weight))
        total_area = int(round(total_area))



        return render_template('index.html', width=width, total_area=total_area, length=length, thickness=thickness, Px=Px, Py=Py, requested_volume=requested_volume, total_volume=total_volume, hemi_volume=hemi_volume, hemi_weight=hemi_weight, total_weight=total_weight, zone_rows=zone_rows, retained_volume=round(extraction['volume'], 2), user=session['user'])

    return render_template('index.html', user=session['user'])

def retained_bounds(x, a, b, Px, Pyc, radius, n=2, m=1.2, upper_scale=1.3, lower_scale=0.7):
    low, high = flap_bounds(x, a, b, n, m, upper_scale, lower_scale)
    circle_height = np.sqrt(np.maximum(0, radius ** 2 - (x - Px) ** 2))
    return np.maximum(low, Pyc - circle_height), np.minimum(high, Pyc + circle_height)


def keep_only_requested_volume(width, length, thickness, total_volume, Px, Py, required_volume,
                               n=2, m=1.2, upper_scale=1.3, lower_scale=0.7):
    """Intersect the flap with a perforator-centred disk of solved volume.

    The disk preserves nearest-to-perforator selection, weighted by local
    thickness: full measured thickness medially, 82% laterally.
    Integration and radius solving do not depend on plotting resolution.
    """
    if not np.isfinite([width, length, thickness, Px, Py, required_volume]).all():
        raise ValueError("Inputs must be finite.")
    if min(width, length, thickness, required_volume) <= 0:
        raise ValueError("Dimensions and required volume must be positive.")
    a, b = width / 2, length / 2
    Pyc = b * upper_scale - Py
    low, high = flap_bounds(Px, a, b, n, m, upper_scale, lower_scale)
    if abs(Px) > a or not low - 1e-10 <= Pyc <= high + 1e-10:
        raise ValueError("The perforator must lie within the flap boundary.")
    available = calculate_asymmetric_area(a, b, n, m, upper_scale, lower_scale) * thickness * total_volume_factor()
    cuts = zone_boundaries(width, n, m)
    if required_volume > available:
        raise ValueError(f"Required volume exceeds the available {available:.2f} cc.")

    def volume(radius):
        left, right = max(-a, Px - radius), min(a, Px + radius)
        if left >= right:
            return 0.0
        def height(x):
            low, high = retained_bounds(x, a, b, Px, Pyc, radius, n, m, upper_scale, lower_scale)
            local_factor = 1.0 if abs(x) <= cuts[2] else LATERAL_THICKNESS_FACTOR
            return max(0.0, high - low) * local_factor
        # Integrate each constant-thickness band separately, including the
        # midline. Interior subintervals keep narrow intersections visible.
        edges = [left] + [cut for cut in cuts if left < cut < right] + [right]
        integral = 0.0
        for start, end in zip(edges[:-1], edges[1:]):
            integral += quad(height, start, end,
                points=np.linspace(start, end, 17)[1:-1],
                epsabs=max(1e-12, required_volume / thickness * 1e-7 / len(edges)),
                epsrel=1e-7, limit=250)[0]
        return thickness * integral

    max_radius = max(np.hypot(x - Px, y - Pyc)
                     for x in (-a, a) for y in (-b * lower_scale, b * upper_scale))
    if required_volume == available:
        radius, actual_volume = max_radius, available
    else:
        radius = brentq(lambda r: volume(r) - required_volume, 0, max_radius,
                        xtol=1e-10, rtol=1e-12)
        actual_volume = volume(radius)
    x = np.linspace(max(-a, Px - radius), min(a, Px + radius), 2001)
    low, high = retained_bounds(x, a, b, Px, Pyc, radius, n, m, upper_scale, lower_scale)
    return dict(x=x, low=low, high=high, radius=radius, volume=actual_volume)


def visualize_extraction(width, length, Px, Py, extraction, requested_volume, total_volume,
                         n=2, m=1.2, upper_scale=1.3, lower_scale=0.7):
    a, b = width / 2, length / 2
    Pyc = b * upper_scale - Py
    x = np.linspace(-a, a, 2001)
    low, high = flap_bounds(x, a, b, n, m, upper_scale, lower_scale)
    fig, ax = plt.subplots(figsize=(10, 10))
    try:
        ax.set_xlim(-a - 5, a + 5)
        ax.set_ylim(-b * lower_scale - 5, b * upper_scale + 5)
        ax.set_aspect('equal')
        ax.plot(x, high, 'b-', label="Abdominoplasty Boundary")
        ax.plot(x, low, 'b-')
        for i, cut in enumerate(zone_boundaries(width, n, m)):
            zone_low, zone_high = flap_bounds(cut, a, b, n, m, upper_scale, lower_scale)
            ax.plot([cut, cut], [zone_low, zone_high], color='gray', linestyle='--',
                    label="Geometric zone boundaries" if i == 0 else None)
        ax.plot(Px, Pyc, 'ko', markersize=8, label="Perforator")
        valid = extraction['high'] >= extraction['low']
        ax.fill_between(extraction['x'], extraction['low'], extraction['high'],
                        where=valid, color='red', alpha=0.5, label="Extracted Region")
        extracted_width = np.ptp(extraction['x'][valid])
        extracted_length = max(extraction['high'][valid]) - min(extraction['low'][valid])
        ax.set_title(f"DIEP Flap {extraction['volume']:.2f}cc | Approx. dimensions: {extracted_width:.1f}cm x {extracted_length:.1f}cm")
        ax.legend()
        ax.grid()
        fig.savefig('static/extraction.png', bbox_inches='tight', pad_inches=0.5)
    finally:
        plt.close(fig)


if __name__ == '__main__':
    app.run(debug=True)
