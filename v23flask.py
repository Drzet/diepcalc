from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

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
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        width = float(request.form['width'])
        length = float(request.form['length'])
        thickness = float(request.form['thickness'])
        Px = float(request.form['Px'])
        Py = float(request.form['Py'])
        required_volume = float(request.form['required_volume'])
        
        extraction_area_points = keep_only_required_volume(width, length, thickness, Px, Py,  required_volume)
        
        if extraction_area_points:
            visualize_extraction(width, length, Px, Py, extraction_area_points, required_volume)
        # Calculate total volume
        n = 1.3
        m = 2

        x_vals = np.linspace(-a, a, 500)
        y_vals = np.linspace(-b, b, 500)
        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]

        total_area = sum(
        1 for x in x_vals for y in y_vals if (np.abs(x/a)**m + np.abs(y/b)**n) <= 1
        ) * dx * dy
        total_volume = total_area * thickness  # Total volume of the flap
        hemi_volume = total_volume / 2
        total_weight = total_volume * 0.9
        hemi_weight = hemi_volume * 0.9

        total_volume = int(round(total_volume))
        total_weight = int(round(total_weight))
        hemi_volume = int(round(hemi_volume))
        hemi_weight = int(round(hemi_weight))
        
        return render_template('index.html', width=width, length=length, thickness=thickness, Px=Px, Py=Py, required_volume=required_volume, total_volume=total_volume, hemi_volume=hemi_volume, hemi_weight=hemi_weight, total_weight=total_weight, user=session['user'])
    
    return render_template('index.html', user=session['user'])

# Functions for DIEP flap extraction logic
def calculate_zone_volume(width, length, thickness, Px):
    a = width / 2  # Semi-major axis
    b = length / 2  # Semi-minor axis
    total_area = math.pi * a * b  # Total ellipse area
    global total_volume
    total_volume = total_area * thickness  # Total volume of the flap
    
    medial_width = width * 0.75
    lateral_width = width * 0.25
    
    medial_zone_x = (-medial_width / 2, medial_width / 2)
    
    if medial_zone_x[0] <= Px <= medial_zone_x[1]:
        zone_area_fraction = 0.75
    else:
        zone_area_fraction = 0.25
    
    return total_volume * zone_area_fraction

def keep_only_required_volume(width, length, thickness, Px, Py, required_volume):
    a = width / 2 
    b = length / 2
    global Pyc
    Pyc = b - Py  # Convert to Cartesian coordinate
    
    total_zone1_volume = calculate_zone_volume(width, length, thickness, Px)
    excess_volume = total_zone1_volume - required_volume
    
    if excess_volume <= 0:
        return None
    
    n = 1.3
    m = 2
    all_points = [
    (x, y) for x in np.linspace(-a, a, 100) for y in np.linspace(-b, b, 100)
    if (np.abs(x/a)**m + np.abs(y/b)**n) <= 1]
    all_points.sort(key=lambda p: np.sqrt((p[0] - Px) ** 2 + (p[1] - Pyc) ** 2))
    
    kept_volume = 0
    kept_points = []
    for point in all_points:
        if kept_volume >= required_volume:
            break
        kept_points.append(point)
        kept_volume += thickness * (width / 100) * (length / 100)
    
    return kept_points

def visualize_extraction(width, length, Px, Py, extraction_area_points, required_volume):
    a = width / 2
    b = length / 2
    Pyc = b - Py
    
    extracted_width = max(x for x, y in extraction_area_points) - min(x for x, y in extraction_area_points)
    extracted_length = max(y for x, y in extraction_area_points) - min(y for x, y in extraction_area_points)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-a - 5, a + 5)
    ax.set_ylim(-b - 5, b + 5)
    ax.set_aspect('equal')
    
    n = 1.3  # vertical curvature (pointier ends)
    m = 2    # horizontal curvature (standard ellipse)

    theta = np.linspace(0, 2 * np.pi, 300)
    ellipse_x = a * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** (2 / m)
    ellipse_y = b * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** (2 / n)

    ax.plot(ellipse_x, ellipse_y, 'b-', linewidth=2, label="Lamé Superellipse Boundary")
    
    ax.plot(Px, Pyc, 'ko', markersize=8, label="Perforator")
    
    if extraction_area_points:
        extracted_x, extracted_y = zip(*extraction_area_points)
        ax.scatter(extracted_x, extracted_y, color='red', s=2)
    
    ax.set_title(f"DIEP Flap {required_volume:.1f}cc | Dimensions: {extracted_width:.1f}cm x {extracted_length:.1f}cm")
    ax.legend()
    plt.grid()
    plt.savefig('static/extraction.png')
    
# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
