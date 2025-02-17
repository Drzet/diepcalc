import streamlit as st
import math

def calculate_flap_volume(flap_length, flap_width, Max_first, Max_middle, Max_last, Min_first, Min_middle, Min_last):
    major_axis = flap_width
    minor_axis = flap_length
    avg_Max = (Max_first + Max_middle + Max_last) / 3
    avg_Min = (Min_first + Min_middle + Min_last) / 3
    avg_thickness = (avg_Max + avg_Min) / 2
    area = (math.pi / 4) * major_axis * minor_axis
    volume = area * avg_thickness
    return volume

def update_middle_slice():
    try:
        first_slice = st.session_state["first_slice"]
        slice_width = st.session_state["slice_width"]
        
        if not st.session_state["use_flap_length"]:
            flap_length = st.session_state["flap_length"]
            last_slice = first_slice + int(flap_length / slice_width)
        else:
            last_slice = st.session_state["last_slice"]
            flap_length = (last_slice - first_slice) * slice_width

        middle_slice = (first_slice + last_slice) // 2

        st.session_state["last_slice"] = last_slice
        st.session_state["flap_length"] = flap_length
        st.session_state["middle_slice"] = middle_slice

    except ValueError:
        pass

st.title("DIEP Flap Volume Calculator")

if "use_flap_length" not in st.session_state:
    st.session_state["use_flap_length"] = False

slice_width = st.number_input("Slice Width (cm)", min_value=0.1, step=0.1, format="%.2f", key="slice_width")
first_slice = st.number_input("First Slice Number", min_value=1, step=1, format="%d", key="first_slice", on_change=update_middle_slice)

use_flap_length = st.checkbox("Input Last Slice Number Instead", value=st.session_state["use_flap_length"])
st.session_state["use_flap_length"] = use_flap_length

if not use_flap_length:
    flap_length = st.number_input("Flap Length (cm, vertical)", min_value=0.1, step=0.1, format="%.2f", key="flap_length", on_change=update_middle_slice)
    last_slice = first_slice + int(flap_length / slice_width)
else:
    last_slice = st.number_input("Last Slice Number", min_value=first_slice+1, step=1, format="%d", key="last_slice", on_change=update_middle_slice)
    flap_length = (last_slice - first_slice) * slice_width

middle_slice = (first_slice + last_slice) // 2

flap_width = st.number_input("Flap Width (cm, frontal)", min_value=0.1, step=0.1, format="%.2f", key="flap_width")

st.subheader("Max Thickness at Slices (cm)")
Max_first = st.number_input(f"Max Thickness at Slice {first_slice}:", min_value=0.1, step=0.1, format="%.2f", key="Max_first")
Max_middle = st.number_input(f"Max Thickness at Slice {middle_slice}:", min_value=0.1, step=0.1, format="%.2f", key="Max_middle")
Max_last = st.number_input(f"Max Thickness at Slice {last_slice}:", min_value=0.1, step=0.1, format="%.2f", key="Max_last")

st.subheader("Min Thickness at Slices (cm)")
Min_first = st.number_input(f"Min Thickness at Slice {first_slice}:", min_value=0.1, step=0.1, format="%.2f", key="Min_first")
Min_middle = st.number_input(f"Min Thickness at Slice {middle_slice}:", min_value=0.1, step=0.1, format="%.2f", key="Min_middle")
Min_last = st.number_input(f"Min Thickness at Slice {last_slice}:", min_value=0.1, step=0.1, format="%.2f", key="Min_last")

if st.button("Calculate Volume"):
    try:
        volume = calculate_flap_volume(flap_length, flap_width, Max_first, Max_middle, Max_last, Min_first, Min_middle, Min_last)
        diep_volume = volume * 0.75
        hemi_diep_volume = volume * 0.50
        total_weight = volume * 0.9
        diep_weight = diep_volume * 0.9
        hemi_diep_weight = hemi_diep_volume * 0.9

        result_text = (
            f"Total Abdominoplasty Volume: {volume:.2f} cm³\n  Weight: {total_weight:.2f} g\n"
            f"DIEP Flap Volume (3 zones): {diep_volume:.2f} cm³\n  Weight: {diep_weight:.2f} g\n"
            f"HemiDIEP Flap Volume (2 zones): {hemi_diep_volume:.2f} cm³\n  Weight: {hemi_diep_weight:.2f} g\n"
        )

        st.text_area("Results", value=result_text, height=150)

    except ValueError as e:
        st.error(f"Error: {e}")

if st.button("Clear All Inputs"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

