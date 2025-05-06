# This code replaces the visualization section in warehouse_optimizer_app.py

# Add this after you have calculated store_assignments

# Create a color palette for warehouses
def generate_colors(n):
    """Generate n distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB (simplified version)
        h = hue * 6
        c = 255
        x = 255 * (1 - abs(h % 2 - 1))
        
        if h < 1:
            rgb = [c, x, 0]
        elif h < 2:
            rgb = [x, c, 0]
        elif h < 3:
            rgb = [0, c, x]
        elif h < 4:
            rgb = [0, x, c]
        elif h < 5:
            rgb = [x, 0, c]
        else:
            rgb = [c, 0, x]
            
        colors.append(rgb)
    return colors

# Generate colors for warehouses
warehouse_colors = generate_colors(len(optimized_warehouses))
warehouse_color_map = {wh: color for wh, color in zip(optimized_warehouses['warehouse_id'], warehouse_colors)}

# Create a DataFrame that includes both warehouse and store info for visualization
warehouse_data_for_map = optimized_warehouses.copy()
warehouse_data_for_map["type"] = "warehouse"
warehouse_data_for_map = warehouse_data_for_map.merge(warehouse_metrics[["warehouse_id", "num_stores", "total_cost"]], on="warehouse_id")

# Add color for each warehouse
for i, wh_id in enumerate(warehouse_data_for_map['warehouse_id']):
    warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_r'] = warehouse_colors[i][0]
    warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_g'] = warehouse_colors[i][1]
    warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_b'] = warehouse_colors[i][2]

store_data_for_map = df.copy()
store_data_for_map["type"] = "store"
store_data_for_map = store_data_for_map.merge(store_assignments[["store_id", "warehouse_id", "distance_miles", "transportation_cost"]], on="store_id")

# Add color for each store based on its assigned warehouse
for wh_id in warehouse_data_for_map['warehouse_id']:
    color = warehouse_color_map[wh_id]
    store_data_for_map.loc[store_data_for_map['warehouse_id'] == wh_id, 'color_r'] = color[0]
    store_data_for_map.loc[store_data_for_map['warehouse_id'] == wh_id, 'color_g'] = color[1]
    store_data_for_map.loc[store_data_for_map['warehouse_id'] == wh_id, 'color_b'] = color[2]

# Create a list of lines connecting stores to warehouses for the map
lines = []
for _, store in store_data_for_map.iterrows():
    warehouse = warehouse_data_for_map[warehouse_data_for_map["warehouse_id"] == store["warehouse_id"]].iloc[0]
    # Get the color from the warehouse
    color = [
        warehouse['color_r'],
        warehouse['color_g'],
        warehouse['color_b']
    ]
    
    lines.append({
        "start_lat": store["latitude"],
        "start_lon": store["longitude"],
        "end_lat": warehouse["latitude"],
        "end_lon": warehouse["longitude"],
        "store_id": store["store_id"],
        "warehouse_id": warehouse["warehouse_id"],
        "color_r": color[0],
        "color_g": color[1],
        "color_b": color[2]
    })

lines_df = pd.DataFrame(lines)

# Map showing stores and warehouses
st.subheader("Map Visualization")

# Create layers for the map
store_layer = pdk.Layer(
    "ScatterplotLayer",
    data=store_data_for_map,
    get_position=["longitude", "latitude"],
    get_radius=[100, 300],  # Increased radius for better visibility
    get_fill_color=["color_r", "color_g", "color_b", 200],  # Color based on warehouse assignment
    pickable=True,
    opacity=0.8,
    stroked=True,
    filled=True,
)

warehouse_layer = pdk.Layer(
    "ScatterplotLayer",
    data=warehouse_data_for_map,
    get_position=["longitude", "latitude"],
    get_radius=[800, 1500],  # Larger radius for warehouses
    get_fill_color=["color_r", "color_g", "color_b", 250],  # Warehouse colors
    pickable=True,
    opacity=0.9,
    stroked=True,
    filled=True,
)

line_layer = pdk.Layer(
    "LineLayer",
    data=lines_df,
    get_source_position=["start_lon", "start_lat"],
    get_target_position=["end_lon", "end_lat"],
    get_color=["color_r", "color_g", "color_b", 150],  # Increased opacity, colors match warehouse
    get_width=2,  # Increased line width
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
