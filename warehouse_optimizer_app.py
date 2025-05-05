import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
from io import StringIO
import base64
import time
from math import radians, cos, sin, asin, sqrt
import random

# Set page configuration
st.set_page_config(
    page_title="Warehouse Location Optimizer",
    page_icon="🏭",
    layout="wide"
)

# Title and description
st.title("Warehouse Location Optimizer")
st.markdown("""
This app helps you determine the optimal locations for warehouses based on store demand and transportation costs.
Upload your store data with locations and demand information to get started.
""")

# Continental US boundaries
CONTINENTAL_US = {
    "min_lat": 24.396308,  # Southern tip of Florida
    "max_lat": 49.384358,  # Northern border with Canada
    "min_lon": -124.848974,  # Western coast
    "max_lon": -66.885444   # Eastern coast
}

# Function to calculate distance between two points using Haversine formula
@st.cache_data
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in miles
    return c * r

# Function to check if point is within continental US
def is_in_continental_us(lat, lon):
    return (CONTINENTAL_US["min_lat"] <= lat <= CONTINENTAL_US["max_lat"] and 
            CONTINENTAL_US["min_lon"] <= lon <= CONTINENTAL_US["max_lon"])

# Function to calculate transportation cost
def calculate_transportation_cost(distance, weight, rate):
    return distance * weight * rate

# Function to generate example data
def generate_example_data(num_stores=100):
    # Generate random points within continental US
    data = []
    for _ in range(num_stores):
        lat = random.uniform(CONTINENTAL_US["min_lat"], CONTINENTAL_US["max_lat"])
        lon = random.uniform(CONTINENTAL_US["min_lon"], CONTINENTAL_US["max_lon"])
        # Generate random yearly demand between 10,000 and 500,000 pounds
        yearly_demand = round(random.uniform(10000, 500000))
        data.append({"store_id": f"Store_{_+1}", "latitude": lat, "longitude": lon, "yearly_demand_lbs": yearly_demand})
    
    return pd.DataFrame(data)

# Function to download dataframe as CSV
def download_link(dataframe, filename, link_text):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Sidebar for uploading data and parameters
st.sidebar.header("Upload Store Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with store locations and demand data", type="csv")

# Option to use example data
use_example_data = st.sidebar.checkbox("Use example data instead", value=False)

# Sample data format explanation
with st.sidebar.expander("CSV Format Requirements"):
    st.write("""
    Your CSV file should include the following columns:
    - `store_id`: Unique identifier for each store
    - `latitude`: Store latitude
    - `longitude`: Store longitude
    - `yearly_demand_lbs`: Annual demand in pounds
    """)
    
    # Display sample data
    sample_df = pd.DataFrame({
        "store_id": ["Store_1", "Store_2", "Store_3"],
        "latitude": [40.7128, 34.0522, 41.8781],
        "longitude": [-74.0060, -118.2437, -87.6298],
        "yearly_demand_lbs": [250000, 175000, 320000]
    })
    
    st.dataframe(sample_df)
    st.markdown(download_link(sample_df, "sample_store_data.csv", "Download Sample CSV"), unsafe_allow_html=True)

# Optimization parameters
st.sidebar.header("Optimization Parameters")
num_warehouses = st.sidebar.slider("Number of Warehouses", min_value=1, max_value=20, value=3)
cost_per_pound_mile = st.sidebar.number_input("Transportation Cost Rate ($ per pound-mile)", min_value=0.0001, max_value=1.0, value=0.001, format="%.5f")

# Main app logic
if uploaded_file is not None:
    # Load the uploaded data
    df = pd.read_csv(uploaded_file)
    data_source = "uploaded"
elif use_example_data:
    # Generate example data
    df = generate_example_data()
    data_source = "example"
else:
    st.info("Please upload a CSV file or use example data to get started.")
    st.stop()

# Check if required columns exist
required_cols = ["store_id", "latitude", "longitude", "yearly_demand_lbs"]
if not all(col in df.columns for col in required_cols):
    st.error(f"The data must contain these columns: {', '.join(required_cols)}")
    st.stop()

# Display the data
st.subheader("Store Data")
st.dataframe(df)

# K-means clustering for warehouse locations
@st.cache_data(show_spinner=False)
def optimize_warehouse_locations(stores_df, n_warehouses, max_iterations=100):
    # Initialize random warehouse locations within continental US boundaries
    warehouses = []
    
    while len(warehouses) < n_warehouses:
        lat = random.uniform(CONTINENTAL_US["min_lat"], CONTINENTAL_US["max_lat"])
        lon = random.uniform(CONTINENTAL_US["min_lon"], CONTINENTAL_US["max_lon"])
        if is_in_continental_us(lat, lon):
            warehouses.append({
                "warehouse_id": f"WH_{len(warehouses)+1}",
                "latitude": lat,
                "longitude": lon
            })
    
    warehouses_df = pd.DataFrame(warehouses)
    
    # Iterative optimization
    prev_cost = float('inf')
    for iteration in range(max_iterations):
        # Assign each store to closest warehouse
        assignments = []
        total_cost = 0
        
        for _, store in stores_df.iterrows():
            min_cost = float('inf')
            assigned_wh = None
            
            for _, wh in warehouses_df.iterrows():
                distance = haversine(store["longitude"], store["latitude"], 
                                    wh["longitude"], wh["latitude"])
                cost = calculate_transportation_cost(distance, store["yearly_demand_lbs"], cost_per_pound_mile)
                
                if cost < min_cost:
                    min_cost = cost
                    assigned_wh = wh["warehouse_id"]
            
            assignments.append({
                "store_id": store["store_id"],
                "warehouse_id": assigned_wh,
                "distance": min_cost / (store["yearly_demand_lbs"] * cost_per_pound_mile),
                "transportation_cost": min_cost
            })
            
            total_cost += min_cost
        
        assignments_df = pd.DataFrame(assignments)
        
        # Check convergence
        if abs(prev_cost - total_cost) < 1:
            break
        
        prev_cost = total_cost
        
        # Update warehouse locations to center of assigned stores
        for _, wh in warehouses_df.iterrows():
            assigned_stores = stores_df[assignments_df["warehouse_id"] == wh["warehouse_id"]]
            
            if len(assigned_stores) > 0:
                # Calculate weighted centroid based on demand
                total_demand = assigned_stores["yearly_demand_lbs"].sum()
                
                if total_demand > 0:
                    weighted_lat = (assigned_stores["latitude"] * assigned_stores["yearly_demand_lbs"]).sum() / total_demand
                    weighted_lon = (assigned_stores["longitude"] * assigned_stores["yearly_demand_lbs"]).sum() / total_demand
                    
                    # Ensure the warehouse is within continental US
                    if is_in_continental_us(weighted_lat, weighted_lon):
                        warehouses_df.loc[warehouses_df["warehouse_id"] == wh["warehouse_id"], "latitude"] = weighted_lat
                        warehouses_df.loc[warehouses_df["warehouse_id"] == wh["warehouse_id"], "longitude"] = weighted_lon
    
    return warehouses_df, assignments_df, total_cost

# Run optimization with a progress bar
with st.spinner(f"Optimizing locations for {num_warehouses} warehouses..."):
    progress_bar = st.progress(0)
    
    # Simulating progress
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    # Actual optimization
    optimized_warehouses, store_assignments, total_transportation_cost = optimize_warehouse_locations(df, num_warehouses)

# Display metrics
st.subheader("Optimization Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Number of Warehouses", num_warehouses)

with col2:
    st.metric("Total Transportation Cost", f"${total_transportation_cost:,.2f}")

with col3:
    avg_cost_per_store = total_transportation_cost / len(df)
    st.metric("Avg. Cost per Store", f"${avg_cost_per_store:,.2f}")

# Calculate additional metrics
warehouse_metrics = store_assignments.groupby("warehouse_id").agg(
    num_stores=("store_id", "count"),
    total_cost=("transportation_cost", "sum"),
    avg_distance=("distance", "mean")
).reset_index()

# Join with warehouse locations
warehouse_metrics = warehouse_metrics.merge(optimized_warehouses, on="warehouse_id")

# Calculate store details with distances and costs
store_details = df.merge(store_assignments, on="store_id")
store_details = store_details.merge(
    optimized_warehouses[["warehouse_id", "latitude", "longitude"]], 
    on="warehouse_id", 
    suffixes=("_store", "_warehouse")
)

# Visualizations
st.subheader("Visualization")

# Map showing stores and warehouses
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=np.mean(df["latitude"]),
        longitude=np.mean(df["longitude"]),
        zoom=3,
        pitch=0,
    ),
    layers=[
        # Store layer
        pdk.Layer(
            "ScatterplotLayer",
            data=store_details,
            get_position=["longitude_store", "latitude_store"],
            get_color=[100, 150, 255, 160],
            get_radius=["yearly_demand_lbs / 50000", 1],  # Size based on demand
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
        ),
        # Warehouse layer
        pdk.Layer(
            "ScatterplotLayer",
            data=warehouse_metrics,
            get_position=["longitude", "latitude"],
            get_color=[255, 0, 0, 200],
            get_radius=["num_stores * 1.5", 10],  # Size based on number of stores
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
        ),
        # Lines connecting stores to warehouses
        pdk.Layer(
            "LineLayer",
            data=store_details,
            get_source_position=["longitude_store", "latitude_store"],
            get_target_position=["longitude_warehouse", "latitude_warehouse"],
            get_color=[200, 200, 200, 50],
            get_width=1,
            pickable=True,
        ),
    ],
    tooltip={
        "html": "<b>ID:</b> {store_id}<br><b>Demand:</b> {yearly_demand_lbs} lbs<br><b>Assigned to:</b> {warehouse_id}<br><b>Distance:</b> {distance:.2f} miles<br><b>Cost:</b> ${transportation_cost:.2f}",
        "style": {"background": "white", "color": "black", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
    },
))

# Show detailed metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Warehouse Details")
    st.dataframe(warehouse_metrics)

with col2:
    # Create warehouse comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart for number of stores and transportation cost
    x = range(len(warehouse_metrics))
    width = 0.35
    
    # Normalize values for better visualization
    norm_stores = warehouse_metrics["num_stores"] / warehouse_metrics["num_stores"].max()
    norm_cost = warehouse_metrics["total_cost"] / warehouse_metrics["total_cost"].max()
    
    bar1 = ax.bar([i - width/2 for i in x], norm_stores, width, label='Stores Served (Normalized)')
    bar2 = ax.bar([i + width/2 for i in x], norm_cost, width, label='Transportation Cost (Normalized)')
    
    ax.set_xlabel('Warehouse')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Warehouse Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(warehouse_metrics["warehouse_id"])
    ax.legend()
    
    # Add the actual values as text
    def add_labels(bars, values, offset):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{value:.1f}' if isinstance(value, float) else f'{value}',
                   ha='center', va='bottom', rotation=0, fontsize=8)
    
    add_labels(bar1, warehouse_metrics["num_stores"], 0.01)
    add_labels(bar2, [f"${cost:,.0f}" for cost in warehouse_metrics["total_cost"]], 0.01)
    
    st.pyplot(fig)

# Download results
st.subheader("Download Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown(download_link(warehouse_metrics, "optimized_warehouses.csv", "Download Warehouse Data"), unsafe_allow_html=True)

with col2:
    st.markdown(download_link(store_details, "store_assignments.csv", "Download Store Assignments"), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Warehouse Location Optimizer - Powered by Streamlit")
