# app.py
# EcoNav — Smart Streamlit EV Dashboard with OSRM Routing & OpenChargeMap Integration

import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests

# -------------------- API Keys --------------------
OPENCHARGEMAP_API_KEY = "89ebf209-8d75-43a9-acfa-4f730f54cfc2"

# -------------------- Helper Functions --------------------
def osrm_route(start_lon, start_lat, end_lon, end_lat):
    """Get driving route distance & coordinates from OSRM public API."""
    base = "http://router.project-osrm.org/route/v1/driving/"
    coords_str = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {"overview": "full", "geometries": "geojson"}
    resp = requests.get(base + coords_str, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if 'routes' not in data or len(data['routes']) == 0:
        raise ValueError("No route returned by OSRM")
    route = data['routes'][0]
    distance_m = route.get('distance', 0)
    geometry = route.get('geometry', {})
    coords = [(lat, lon) for lon, lat in geometry.get('coordinates', [])]
    return {"distance_m": distance_m, "coords": coords}


def get_charging_stations(lat, lon, distance_km=50):
    """Fetch nearby charging stations using OpenChargeMap API."""
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "key": OPENCHARGEMAP_API_KEY,
        "latitude": lat,
        "longitude": lon,
        "distance": distance_km,
        "distanceunit": "KM",
        "maxresults": 20
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    stations = []
    for item in data:
        addr = item.get("AddressInfo", {})
        stations.append({
            "name": addr.get("Title", "Unknown"),
            "lat": addr.get("Latitude"),
            "lon": addr.get("Longitude"),
        })
    return stations


def cumulative_route_distances(route_coords):
    cum = [0.0]
    for i in range(1, len(route_coords)):
        d = geodesic(route_coords[i-1], route_coords[i]).km
        cum.append(cum[-1] + d)
    return cum


def nearest_station_along_route(route_coords, stations):
    if not route_coords:
        return None
    cum = cumulative_route_distances(route_coords)
    best = None
    for stn in stations:
        st_point = (stn['lat'], stn['lon'])
        min_d = float("inf")
        min_idx = 0
        for idx, pt in enumerate(route_coords):
            d = geodesic(st_point, pt).km
            if d < min_d:
                min_d = d
                min_idx = idx
        route_dist = cum[min_idx]
        candidate = {
            **stn,
            "min_dist_km": round(min_d, 3),
            "route_dist_from_start_km": round(route_dist, 3)
        }
        if best is None or candidate["route_dist_from_start_km"] < best["route_dist_from_start_km"]:
            best = candidate
    return best


# -------------------- Train ML Model --------------------
@st.cache_data
def load_and_train(csv_path="ADAS_EV_Dataset.csv"):
    df = pd.read_csv(csv_path)
    categorical_cols = ['weather_condition', 'road_type', 'brake_intensity', 'ADAS_output']
    le_dict = {}
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        le.fit(df[col])
        le_dict[col] = le
        df[col] = le.transform(df[col])
    le_target = le_dict['ADAS_output']
    features = [
        'speed_kmh', 'acceleration_mps2', 'brake_intensity', 'battery_level',
        'regen_braking_usage', 'lane_deviation', 'obstacle_distance',
        'traffic_density', 'weather_condition', 'road_type',
        'steering_angle', 'reaction_time'
    ]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X, y = df[features], df['ADAS_output']
    model.fit(X, y)
    return model, le_dict, le_target, features


try:
    model, le_dict, le_target, features = load_and_train()
except FileNotFoundError:
    st.error("⚠️ Please place ADAS_EV_Dataset.csv in the same folder as this app.")
    st.stop()


# -------------------- Streamlit Layout --------------------
st.set_page_config(page_title="EcoNav EV Dashboard", layout="wide")
st.title("⚡ EcoNav — Smart EV Dashboard with Live Charging Stations")
st.markdown("Enter **start** and **destination** to calculate the **real route**, fetch **nearest live charging stations**, and view **ADAS predictions.**")

st.subheader("🧭 Trip Input")
start = st.text_input("Start location", value="Madurai")
end = st.text_input("Destination location", value="Thirumangalam")

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Analyze Trip"):
    geolocator = Nominatim(user_agent="econav_streamlit_app")
    start_loc = geolocator.geocode(start)
    end_loc = geolocator.geocode(end)

    if not start_loc or not end_loc:
        st.error("❌ Could not find one or both locations. Please try again.")
    else:
        start_coords = (start_loc.latitude, start_loc.longitude)
        end_coords = (end_loc.latitude, end_loc.longitude)

        try:
            osrm = osrm_route(start_coords[1], start_coords[0], end_coords[1], end_coords[0])
            route_distance_km = round(osrm["distance_m"] / 1000.0, 2)
            route_coords = osrm["coords"]
        except Exception as e:
            st.warning(f"⚠️ OSRM routing failed ({e}). Using straight-line distance.")
            route_coords = [start_coords, end_coords]
            route_distance_km = round(geodesic(start_coords, end_coords).km, 2)

        # Fetch live charging stations from OpenChargeMap near midpoint
        mid_lat = (start_coords[0] + end_coords[0]) / 2
        mid_lon = (start_coords[1] + end_coords[1]) / 2
        charging_stations = get_charging_stations(mid_lat, mid_lon)

        nearest = nearest_station_along_route(route_coords, charging_stations)
        ev = {
            'battery_level': random.randint(20, 100),
            'speed_kmh': random.randint(30, 90),
            'acceleration_mps2': round(random.uniform(0.2, 2.0), 2),
            'regen_braking_usage': random.randint(20, 80),
            'lane_deviation': round(random.uniform(0, 0.5), 2),
            'obstacle_distance': random.randint(5, 120),
            'traffic_density': random.randint(1, 5),
            'weather_condition': random.choice(['Sunny', 'Rainy', 'Foggy']),
            'road_type': random.choice(['City', 'Highway']),
            'steering_angle': random.randint(-20, 20),
            'reaction_time': round(random.uniform(0.4, 1.8), 2),
            'brake_intensity': random.choice(['Low', 'Medium', 'High', 'Maintain'])
        }

        encoded = ev.copy()
        for col in ['weather_condition', 'road_type', 'brake_intensity']:
            encoded[col] = -1 if encoded[col] not in le_dict[col].classes_ else le_dict[col].transform([encoded[col]])[0]

        X_input = pd.DataFrame([{f: encoded[f] for f in features}])
        pred_num = model.predict(X_input)[0]
        pred_label = le_target.inverse_transform([pred_num])[0]

        max_range_km = 150
        est_range = round((ev['battery_level'] / 100.0) * max_range_km, 2)
        can_reach = est_range >= route_distance_km

        st.session_state.results = {
            "route_distance_km": route_distance_km,
            "nearest": nearest,
            "ev": ev,
            "est_range": est_range,
            "can_reach": can_reach,
            "pred_label": pred_label,
            "start_coords": start_coords,
            "end_coords": end_coords,
            "route_coords": route_coords,
            "charging_stations": charging_stations
        }

# -------------------- Display Results --------------------
if st.session_state.results:
    r = st.session_state.results
    st.subheader("📊 EcoNav Trip Summary")
    st.markdown(f"""
    - 🔋 **Battery:** {r['ev']['battery_level']}%
    - 🌤️ **Weather:** {r['ev']['weather_condition']}
    - 🛣️ **Road Type:** {r['ev']['road_type']}
    - 🚗 **Speed:** {r['ev']['speed_kmh']} km/h
    - 📏 **Distance:** {r['route_distance_km']} km
    - ⚙️ **ADAS Output:** {r['pred_label']}
    - 🔧 **Est. Range:** {r['est_range']} km
    - ✅ **Reachable:** {r['can_reach']}
    """)

    st.subheader("🗺️ Route & Charging Stations Map")
    center_lat = (r['start_coords'][0] + r['end_coords'][0]) / 2
    center_lon = (r['start_coords'][1] + r['end_coords'][1]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    # Add route, start, end, and stations
    folium.Marker(r['start_coords'], popup=f"Start: {start}", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(r['end_coords'], popup=f"End: {end}", icon=folium.Icon(color="red")).add_to(m)

    if len(r['route_coords']) > 1:
        folium.PolyLine(r['route_coords'], color="blue", weight=4, opacity=0.7).add_to(m)

    for s in r['charging_stations']:
        color = "orange" if r['nearest'] and s['name'] == r['nearest']['name'] else "blue"
        folium.Marker(
            location=(s['lat'], s['lon']),
            popup=s['name'],
            icon=folium.Icon(color=color, icon="bolt")
        ).add_to(m)

    st_folium(m, width=800, height=600)

st.markdown("---")
st.caption("⚡ Developed by Satz | EcoNav — Smart EV Dashboard (with OpenChargeMap)")
