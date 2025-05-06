import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import math
import random
from io import StringIO
import base64

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
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
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
    for i in range(num_stores):
        lat = random.uniform(CONTINENTAL_US["min_lat"], CONTINENTAL_US["max_lat"])
        lon = random.uniform(CONTINENTAL_US["min_lon"], CONTINENTAL_US["max_lon"])
        # Generate random yearly demand between 10,000 and 500,000 pounds
        yearly_demand = round(random.uniform(10000, 500000))
        data.append({"store_id": f"Store_{i+1}", "latitude": lat, "longitude": lon, "yearly_demand_lbs": yearly_demand})
    
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
max_iterations = st.sidebar.slider("Max Optimization Iterations", min_value=10, max_value=100, value=50)

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
    
    # Show initial progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Iterative optimization
    prev_cost = float('inf')
    for iteration in range(max_iterations):
        # Update progress
        progress = int((iteration + 1) / max_iterations * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Optimizing: Iteration {iteration + 1}/{max_iterations}")
        
        # Assign each store to closest warehouse
        assignments = []
        total_cost = 0
        
        for _, store in stores_df.iterrows():
            min_cost = float('inf')
            assigned_wh = None
            min_distance = 0
            
            for _, wh in warehouses_df.iterrows():
                distance = haversine(store["longitude"], store["latitude"], 
                                    wh["longitude"], wh["latitude"])
                cost = calculate_transportation_cost(distance, store["yearly_demand_lbs"], cost_per_pound_mile)
                
                if cost < min_cost:
                    min_cost = cost
                    assigned_wh = wh["warehouse_id"]
                    min_distance = distance
            
            assignments.append({
                "store_id": store["store_id"],
                "warehouse_id": assigned_wh,
                "distance_miles": min_distance,
                "transportation_cost": min_cost
            })
            
            total_cost += min_cost
        
        assignments_df = pd.DataFrame(assignments)
        
        # Check convergence
        if abs(prev_cost - total_cost) < 1:
            progress_bar.progress(100)
            progress_text.text(f"Optimization completed in {iteration + 1} iterations")
            break
        
        prev_cost = total_cost
        
        # Update warehouse locations to center of assigned stores
        for _, wh in warehouses_df.iterrows():
            wh_id = wh["warehouse_id"]
            assigned_stores_indices = assignments_df.index[assignments_df["warehouse_id"] == wh_id].tolist()
            assigned_stores = stores_df.iloc[assigned_stores_indices]
            
            if len(assigned_stores) > 0:
                # Calculate weighted centroid based on demand
                total_demand = assigned_stores["yearly_demand_lbs"].sum()
                
                if total_demand > 0:
                    weighted_lat = (assigned_stores["latitude"] * assigned_stores["yearly_demand_lbs"]).sum() / total_demand
                    weighted_lon = (assigned_stores["longitude"] * assigned_stores["yearly_demand_lbs"]).sum() / total_demand
                    
                    # Ensure the warehouse is within continental US
                    if is_in_continental_us(weighted_lat, weighted_lon):
                        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "latitude"] = weighted_lat
                        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "longitude"] = weighted_lon
    
    # If we've reached max iterations without convergence
    if iteration == max_iterations - 1:
        progress_bar.progress(100)
        progress_text.text(f"Optimization completed after maximum {max_iterations} iterations")
    
    return warehouses_df, assignments_df, total_cost

# Run optimization
if st.button("Run Optimization"):
    optimized_warehouses, store_assignments, total_transportation_cost = optimize_warehouse_locations(
        df, num_warehouses, max_iterations)
    
    # Store results in session state
    st.session_state.optimized_warehouses = optimized_warehouses
    st.session_state.store_assignments = store_assignments
    st.session_state.total_transportation_cost = total_transportation_cost
    st.session_state.optimization_complete = True
else:
    if 'optimization_complete' not in st.session_state:
        st.session_state.optimization_complete = False

# Display results if optimization is complete
if st.session_state.optimization_complete:
    optimized_warehouses = st.session_state.optimized_warehouses
    store_assignments = st.session_state.store_assignments
    total_transportation_cost = st.session_state.total_transportation_cost
    
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
        avg_distance=("distance_miles", "mean")
    ).reset_index()
    
    # Join with warehouse locations
    warehouse_metrics = warehouse_metrics.merge(optimized_warehouses, on="warehouse_id")
    
    # Calculate store details with distances and costs
    store_details = df.copy()
    store_details = store_details.merge(store_assignments, on="store_id")
    
    # Create a DataFrame that includes both warehouse and store info for visualization
    warehouse_data_for_map = optimized_warehouses.copy()
    warehouse_data_for_map["type"] = "warehouse"
    warehouse_data_for_map = warehouse_data_for_map.merge(warehouse_metrics[["warehouse_id", "num_stores", "total_cost"]], on="warehouse_id")
    
    store_data_for_map = df.copy()
    store_data_for_map["type"] = "store"
    store_data_for_map = store_data_for_map.merge(store_assignments[["store_id", "warehouse_id", "distance_miles", "transportation_cost"]], on="store_id")
    
    # Combine for map visualization
    map_data = pd.concat([warehouse_data_for_map, store_data_for_map], ignore_index=True)
    
    # Create a list of lines connecting stores to warehouses for the map
    lines = []
    for _, store in store_data_for_map.iterrows():
        warehouse = warehouse_data_for_map[warehouse_data_for_map["warehouse_id"] == store["warehouse_id"]].iloc[0]
        lines.append({
            "start_lat": store["latitude"],
            "start_lon": store["longitude"],
            "end_lat": warehouse["latitude"],
            "end_lon": warehouse["longitude"],
            "store_id": store["store_id"],
            "warehouse_id": warehouse["warehouse_id"]
        })
    
    lines_df = pd.DataFrame(lines)
    
    # Map showing stores and warehouses
    st.subheader("Map Visualization")
    
    # Create layers for the map
    store_layer = pdk.Layer(
        "ScatterplotLayer",
        data=store_data_for_map,
        get_position=["longitude", "latitude"],
        get_radius=[50, 200],  # Fixed radius for visibility
        get_fill_color=[50, 150, 250, 160],  # Blue for stores
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    warehouse_layer = pdk.Layer(
        "ScatterplotLayer",
        data=warehouse_data_for_map,
        get_position=["longitude", "latitude"],
        get_radius=[500, 1000],  # Larger radius for warehouses
        get_fill_color=[250, 0, 0, 200],  # Red for warehouses
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    line_layer = pdk.Layer(
        "LineLayer",
        data=lines_df,
        get_source_position=["start_lon", "start_lat"],
        get_target_position=["end_lon", "end_lat"],
        get_color=[200, 200, 200, 50],
        get_width=1,
        pickable=True,
    )
    
    # Create the map
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=np.mean(df["latitude"]),
            longitude=np.mean(df["longitude"]),
            zoom=3,
            pitch=0,
        ),
        layers=[line_layer, store_layer, warehouse_layer],
        tooltip={
            "html": "<b>ID:</b> {store_id or warehouse_id}<br><b>Type:</b> {type}<br><b>Demand:</b> {yearly_demand_lbs} lbs<br><b>Cost:</b> ${transportation_cost}",
            "style": {"background": "white", "color": "black", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
        },
    ))
    
    # Show detailed metrics
    st.subheader("Warehouse Details")
    st.dataframe(warehouse_metrics)
    
    # Basic bar chart for Streamlit using st.bar_chart
    st.subheader("Warehouse Comparison")
    
    chart_data = warehouse_metrics[["warehouse_id", "num_stores", "total_cost"]]
    chart_data = chart_data.set_index("warehouse_id")
    
    # Normalize values for better visualization
    chart_data["num_stores_scaled"] = chart_data["num_stores"] / chart_data["num_stores"].max() * 100
    chart_data["total_cost_scaled"] = chart_data["total_cost"] / chart_data["total_cost"].max() * 100
    
    st.bar_chart(chart_data[["num_stores_scaled", "total_cost_scaled"]])
    
    st.caption("Blue: Number of Stores (scaled) | Orange: Total Cost (scaled)")
    
    # Additional metrics table
    st.subheader("Store Assignments")
    st.dataframe(store_details)
    
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
