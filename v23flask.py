from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import matplotlib.pyplot as plt
#import math
#import os
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.integrate import trapezoid

app = Flask(__name__)
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

@app.route('/extract', methods=['GET', 'POST'])

def calculate_asymmetric_area(a, b, n=2, m=1.2, upper_scale=1.3, lower_scale=0.7, num_points=300):
    # Generate asymmetric superellipse
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = a * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2 / n))
    y = b * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2 / m))

    # Apply asymmetric scaling while keeping total height constant
    y_adjusted = np.where(y > 0, y * upper_scale, y * lower_scale)

    # Sort x values to ensure proper integration order
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y_adjusted[sorted_indices]

    # Integrate the positive values only
    total_area = trapezoid(np.abs(y_sorted), x_sorted) * 2  # Multiply by 2 to account for both halves

    return total_area

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        width = float(request.form['width'])
        length = float(request.form['length'])
        thickness = float(request.form['thickness'])
        Px = float(request.form['Px'])
        Py = float(request.form['Py'])
        requested_volume = float(request.form['requested_volume'])
        
       
        
        # Calculate total volume using the new asymmetric superellipse
        a = width / 2  # Semi-major axis
        b = length / 2  # Semi-minor axis
        total_area = calculate_asymmetric_area(a, b)
        total_volume = total_area * thickness  # Volume of the flap
        hemi_volume = total_volume / 2
        total_weight = total_volume * 0.9
        hemi_weight = hemi_volume * 0.9


        extraction_area_points = keep_only_requested_volume(width, length, thickness, total_volume, Px, Py, requested_volume)
        
        if extraction_area_points:
            visualize_extraction(width, length, Px, Py, extraction_area_points, requested_volume)

        total_volume = int(round(total_volume))
        total_weight = int(round(total_weight))
        hemi_volume = int(round(hemi_volume))
        hemi_weight = int(round(hemi_weight))
        total_area = int(round(total_area))

        
        
        return render_template('index.html', width=width, total_area=total_area, length=length, thickness=thickness, Px=Px, Py=Py, requested_volume=requested_volume, total_volume=total_volume, hemi_volume=hemi_volume, hemi_weight=hemi_weight, total_weight=total_weight, user=session['user'])
    
    return render_template('index.html', user=session['user'])

# Functions for DIEP flap extraction logic
def calculate_zone_volume(width, Px, total_volume):
    print(f"total_volume={total_volume}")
    medial_width = width * 0.85
    lateral_width = width * 0.15
    print(f"medial_width={medial_width}")
    medial_zone_x = (-medial_width / 2, medial_width / 2)
    print(f"Px={Px}, Medial Zone Bounds={medial_zone_x}")
    
    if medial_zone_x[0] <= Px <= medial_zone_x[1]:
        zone_area_fraction = 0.85
    else:
        zone_area_fraction = 0.15
    zone1_volume = total_volume * zone_area_fraction
    print(f"zone1_volume={zone1_volume}")
    return total_volume * zone_area_fraction


def keep_only_requested_volume(width, length, thickness, total_volume, Px, Py, requested_volume, n=1.2, m=2, upper_scale=1.3, lower_scale=0.7):
    print(f"Received total_volume in keep_only_requested_volume: {total_volume}")
    a = width / 2  # Semi-major axis
    b = length / 2  # Semi-minor axis
    global Pyc
    Pyc = b - Py  # Convert to Cartesian coordinate
    total_zone1_volume = calculate_zone_volume(width, Px, total_volume)
    excess_volume = total_zone1_volume - requested_volume
    print(f"EXtracted_volume={excess_volume}")
    
    if excess_volume <= 0:
        return None
    # Generate candidate points inside the asymmetric Lame superellipse
    all_points = [(x, y) for x in np.linspace(-a, a, 100) for y in np.linspace(-b * lower_scale, b * upper_scale, 100)
                  if (abs(x) / a) ** (2 / n) + (abs(y) / (b * (upper_scale if y > 0 else lower_scale))) ** (2 / m) <= 1]

    # Sort points based on proximity to perforator
    all_points.sort(key=lambda p: np.sqrt((p[0] - Px) ** 2 + (p[1] - Pyc) ** 2))

    # Extract points until required volume is reached
    kept_volume = 0
    kept_points = []
    
    for point in all_points:
        if kept_volume >= requested_volume:
            break
        kept_points.append(point)
        kept_volume += thickness * (width / 100) * (length / 100)  # Same method as the original script

    return kept_points


def visualize_extraction(width, length, Px, Py, extraction_area_points, requested_volume, n=2, m=1.2, upper_scale=1.3, lower_scale=0.7, num_points=300):
    a = width / 2
    b = length / 2
    Pyc = b - Py  # Convert Py to Cartesian coordinate

    # Generate the asymmetric Lame superellipse
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_vals = a * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2 / n))
    y_vals = b * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2 / m))
    
    # Apply asymmetric scaling
    y_vals_adjusted = np.where(y_vals > 0, y_vals * upper_scale, y_vals * lower_scale)

    # Plot updated asymmetric shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-a - 5, a + 5)
    ax.set_ylim(-b - 5, b + 5)
    ax.set_aspect('equal')

    extracted_width = max(x for x, y in extraction_area_points) - min(x for x, y in extraction_area_points)
    extracted_length = max(y for x, y in extraction_area_points) - min(y for x, y in extraction_area_points)
    print(f"extracted_width={extracted_width}")
    # Plot new asymmetric boundary
    ax.plot(x_vals, y_vals_adjusted, 'b-', linewidth=2, label="Asymmetric Superellipse Boundary")

    # Mark perforator point
    ax.plot(Px, Pyc, 'ko', markersize=8, label="Perforator")

    # Plot extracted region if available
    if extraction_area_points:
        extracted_x, extracted_y = zip(*extraction_area_points)
        ax.scatter(extracted_x, extracted_y, color='red', s=2, label="Extracted Region")

    ax.set_title(f"DIEP Flap {requested_volume:.1f}cc | Dimensions: {extracted_width:.1f}cm x {extracted_length:.1f}cm")
    ax.legend()
    plt.grid()
    plt.savefig('static/extraction.png')

    
# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
