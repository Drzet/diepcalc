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
users = {"admin": generate_password_hash("password123")}  # Change this for real use

# Function to log visits
def log_visit(user):
    with open("visits.log", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - User: {user}\n")

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
        
        extraction_area_points = keep_only_required_volume(width, length, thickness, Px, Py, required_volume)
        
        if extraction_area_points:
            visualize_extraction(width, length, Px, Py, extraction_area_points, required_volume)
        # Calculate total volume
        a = width / 2  # Semi-major axis
        b = length / 2  # Semi-minor axis
        total_area = math.pi * a * b  # Total ellipse area
        total_volume = total_area * thickness  # Total volume of the flap
        hemi_volume = total_volume / 2
        
        return render_template('index.html', width=width, length=length, thickness=thickness, Px=Px, Py=Py, required_volume=required_volume, user=session['user'])
    
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
    
    all_points = [(x, y) for x in np.linspace(-a, a, 100) for y in np.linspace(-b, b, 100) if (x**2 / a**2) + (y**2 / b**2) <= 1]
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
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-a - 5, a + 5)
    ax.set_ylim(-b - 5, b + 5)
    ax.set_aspect('equal')
    
    theta = np.linspace(0, 2 * np.pi, 300)
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)
    ax.plot(ellipse_x, ellipse_y, 'b-', linewidth=2, label="Ellipse Boundary")
    
    ax.plot(Px, Pyc, 'ro', markersize=8, label="Perforator")
    
    if extraction_area_points:
        extracted_x, extracted_y = zip(*extraction_area_points)
        ax.scatter(extracted_x, extracted_y, color='red', s=2, label="Remaining Volume")
    
    ax.set_title(f"DIEP Flap {required_volume:.1f}cc | Extracted: {extracted_width:.1f}cm x {extracted_length:.1f}cm")
    ax.legend()
    plt.grid()
    plt.savefig('static/extraction.png')
    
# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
