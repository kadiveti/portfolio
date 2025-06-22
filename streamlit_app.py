import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import requests
import time
from datetime import datetime
import random

# --- 1. Page Configuration & Initial Setup ---
st.set_page_config(
    page_title="Unified Network Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Configuration, Constants & Styling ---
# Backend-Managed API Endpoint
API_ENDPOINT_URL = "https://your-backend-api.com/network-data"

SEVERITY_COLORS = {
    'healthy': '#2ECC71',   # Green
    'minor': '#F1C40F',     # Yellow
    'major': '#F39C12',     # Orange
    'critical': '#E74C3C',  # Red
    'down': '#34495E',      # Dark Blue/Grey
}
SEVERITY_ICONS = {
    'healthy': '✅', 'minor': '⚠️', 'major': '🟠', 'critical': '🔥', 'down': '💀'
}
SEVERITY_SIZE = {
    'healthy': 15, 'minor': 18, 'major': 22, 'critical': 26, 'down': 24
}

# SLA Thresholds
RESPONSE_TIME_SLA_MS = 500
ERROR_RATE_SLA_PER_MIN = 2.0

# --- 3. Sidebar Controls ---
st.sidebar.header("⚙️ Dashboard Controls")

theme = st.sidebar.radio("Select Theme", ["Dark", "Light"], index=0, help="Change the visual theme of the dashboard.")

st.sidebar.subheader("Data Source")
use_mock_data = st.sidebar.toggle(
    "Use Mock Data", value=True,
    help="Use simulated data. Uncheck to use the live backend API."
)

st.sidebar.subheader("Refresh Settings")
auto_refresh = st.sidebar.toggle("Enable Auto Refresh", value=True)
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)", 5, 60, 10,
    disabled=not auto_refresh
)

st.sidebar.subheader("Graph Display")
layout_option = st.sidebar.selectbox(
    "Graph Layout",
    ["kamada_kawai", "spring", "circular", "shell", "random"],
    help="Choose the algorithm for positioning nodes."
)
node_size_multiplier = st.sidebar.slider(
    "Node Size Multiplier", 0.5, 3.0, 1.5,
    help="Increase or decrease the size of all nodes on the graph."
)

if st.sidebar.button("🔄 Refresh Data Now", use_container_width=True):
    st.session_state.force_refresh = True

# --- 4. Data Generation and Fetching Functions ---

def get_node_abbreviation(name):
    """Creates a short, clean abbreviation for a node name."""
    parts = name.replace('-', ' ').split()
    if len(parts) >= 2:
        return (parts[0][:1] + parts[1][:1]).upper()
    return name[:2].upper()

def generate_mock_data():
    """Generates realistic mock network data for both nodes and edges."""
    services = [
        "api-gateway", "load-balancer", "auth-service", "user-service",
        "order-service", "payment-gateway", "inventory-service", "database",
        "notification-service", "file-storage", "cache-redis", "search-engine"
    ]
    nodes = []
    for service in services:
        severity = random.choices(['healthy', 'minor', 'major', 'critical', 'down'], [70, 15, 8, 5, 2])[0]
        nodes.append({
            'id': service,
            'name': service,
            'abbreviation': get_node_abbreviation(service),
            'severity': severity,
            'status': 'UP' if severity != 'down' else 'DOWN',
            'last_updated': datetime.now().strftime("%H:%M:%S")
        })

    # Define a realistic topology
    g = nx.Graph()
    g.add_nodes_from([node['id'] for node in nodes])
    connections = [
        ("api-gateway", "load-balancer"), ("load-balancer", "user-service"),
        ("load-balancer", "order-service"), ("load-balancer", "search-engine"),
        ("user-service", "auth-service"), ("user-service", "database"),
        ("order-service", "payment-gateway"), ("order-service", "inventory-service"),
        ("order-service", "database"), ("payment-gateway", "notification-service"),
        ("inventory-service", "database"), ("user-service", "cache-redis"),
        ("auth-service", "database"), ("search-engine", "database")
    ]
    g.add_edges_from(connections)

    edges = []
    for u, v in g.edges():
        node_u = next(n for n in nodes if n['id'] == u)
        node_v = next(n for n in nodes if n['id'] == v)
        is_critical_connection = any(n in ['critical', 'down'] for n in [node_u['severity'], node_v['severity']])
        violates_sla = random.random() < 0.15

        if is_critical_connection or violates_sla:
            rt = random.randint(RESPONSE_TIME_SLA_MS, RESPONSE_TIME_SLA_MS + 800)
            err = round(random.uniform(ERROR_RATE_SLA_PER_MIN, ERROR_RATE_SLA_PER_MIN + 5), 2)
        else:
            rt = random.randint(50, RESPONSE_TIME_SLA_MS - 50)
            err = round(random.uniform(0, ERROR_RATE_SLA_PER_MIN - 0.5), 2)

        edges.append({'source': u, 'target': v, 'response_time_ms': rt, 'error_rate_per_min': err})

    return {'nodes': nodes, 'edges': edges}

def fetch_api_data():
    """Fetches data from the hardcoded API_ENDPOINT_URL. Falls back to mock data on failure."""
    try:
        response = requests.get(API_ENDPOINT_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch API data: {e}. Falling back to mock data.")
        return generate_mock_data()

# --- 5. Plotly Graph Creation Function ---
def create_plotly_graph(G, theme, node_size_multiplier):
    """Creates an interactive Plotly graph from a NetworkX graph."""
    pos = getattr(nx, f"{layout_option}_layout")(G)

    # Edge Traces
    edge_x_normal, edge_y_normal, edge_x_high, edge_y_high = [], [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        is_high_latency = edge[2]['response_time_ms'] > RESPONSE_TIME_SLA_MS
        (edge_x_high if is_high_latency else edge_x_normal).extend([x0, x1, None])
        (edge_y_high if is_high_latency else edge_y_normal).extend([y0, y1, None])

    edge_trace_normal = go.Scatter(x=edge_x_normal, y=edge_y_normal, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    edge_trace_high = go.Scatter(x=edge_x_high, y=edge_y_high, line=dict(width=2.5, color='#E74C3C', dash='dash'), hoverinfo='none', mode='lines')

    # Node Trace
    node_x, node_y, node_text, node_hover_text, node_colors, node_sizes = [], [], [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x); node_y.append(y); node_text.append(data['abbreviation'])
        hover_text = f"<b>{data['name']}</b><br>Severity: {data['severity'].title()}<br>Status: {data['status']}"
        node_hover_text.append(hover_text)
        node_colors.append(SEVERITY_COLORS[data['severity']])
        node_sizes.append(SEVERITY_SIZE[data['severity']] * node_size_multiplier)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="middle center", textfont=dict(color='white', size=10, family="Arial Black"),
        hoverinfo='text', hovertext=node_hover_text,
        marker=dict(showscale=False, color=node_colors, size=node_sizes, line=dict(width=2, color='#FFF')))

    # Edge Hover Trace
    middle_node_x, middle_node_y, middle_node_hover_text = [], [], []
    for u, v, data in G.edges(data=True):
        middle_node_x.append((pos[u][0] + pos[v][0]) / 2)
        middle_node_y.append((pos[u][1] + pos[v][1]) / 2)
        hover_text = (f"<b>{u} ↔ {v}</b><br>"
                      f"Response Time: {data['response_time_ms']}ms<br>"
                      f"Errors/Min: {data['error_rate_per_min']}<br>"
                      f"<span style='color: #bbb;'>SLA: <{RESPONSE_TIME_SLA_MS}ms, <{ERROR_RATE_SLA_PER_MIN} err/min</span>")
        middle_node_hover_text.append(hover_text)

    middle_trace = go.Scatter(x=middle_node_x, y=middle_node_y, mode='markers', hoverinfo='text',
                              text=middle_node_hover_text, marker=dict(size=20, color='rgba(0,0,0,0)'))

    # Define Layout and Figure
    bgcolor = '#1E1E1E' if theme == 'Dark' else '#F5F5F5'
    fig = go.Figure(data=[edge_trace_normal, edge_trace_high, node_trace, middle_trace],
                    layout=go.Layout(
                        showlegend=False, hovermode='closest', margin=dict(b=10, l=5, r=5, t=10),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor=bgcolor, paper_bgcolor=bgcolor,
                        font=dict(color='white' if theme == 'Dark' else 'black')
                    ))
    fig.update_layout(transition_duration=500)
    return fig

# --- 6. Main Application Logic ---
st.title("🚀 Unified Network Monitoring Dashboard")

# Fetch data if needed
if 'data' not in st.session_state or st.session_state.get('force_refresh', False):
    with st.spinner("Fetching network data..."):
        st.session_state.data = generate_mock_data() if use_mock_data else fetch_api_data()
    st.session_state.force_refresh = False
    st.session_state.last_update_time = datetime.now().strftime('%H:%M:%S')

# Load data from session state
network_data = st.session_state.data
node_data = network_data['nodes']
edge_data = network_data['edges']

# Display KPIs with Custom Styling
st.subheader("📊 Key Performance Indicators")
# --- CSS Injection for KPI Styling ---
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 2.5rem;
}
[data-testid="stMetricLabel"] {
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

df_edges = pd.DataFrame(edge_data)
avg_response_time = df_edges['response_time_ms'].mean() if not df_edges.empty else 0
total_errors = df_edges['error_rate_per_min'].sum() if not df_edges.empty else 0
critical_nodes = sum(1 for n in node_data if n['severity'] in ['critical', 'down'])
sla_violations = sum(1 for e in edge_data if e['response_time_ms'] > RESPONSE_TIME_SLA_MS)

cols = st.columns(4)
cols[0].metric("Avg. Response Time", f"{avg_response_time:.0f} ms", delta=f"{avg_response_time - RESPONSE_TIME_SLA_MS:.0f} ms vs SLA", delta_color="inverse", help=f"SLA: < {RESPONSE_TIME_SLA_MS}ms")
cols[1].metric("Total Errors/Min", f"{total_errors:.1f}", help="Sum of error rates across all connections.")
cols[2].metric("Critical/Down Nodes", f"{critical_nodes}", delta=critical_nodes, delta_color="inverse")
cols[3].metric("Latency SLA Violations", f"{sla_violations}", delta=sla_violations, delta_color="inverse", help="Connections with response time > SLA.")

# Create and Display Graph and Legend
st.subheader("📍 Service Interconnectivity Map")
graph_col, legend_col = st.columns([4, 1])

with graph_col:
    G = nx.Graph()
    for node in node_data: G.add_node(node['id'], **node)
    for edge in edge_data: G.add_edge(edge['source'], edge['target'], **edge)
    fig = create_plotly_graph(G, theme, node_size_multiplier)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with legend_col:
    st.markdown("**Node Severity**")
    for severity, color in SEVERITY_COLORS.items():
        st.markdown(f"<span style='display:inline-block; width:12px; height:12px; background-color:{color}; border-radius:50%; margin-right: 8px; vertical-align: middle;'></span> {severity.title()}", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Edge Status**")
    st.markdown("<span style='display:inline-block; width:20px; height:2px; background-color:#888; margin-right: 8px; vertical-align: middle;'></span> Normal Edge", unsafe_allow_html=True)
    st.markdown("<span style='display:inline-block; border-top: 3px dashed #E74C3C; width: 20px; margin-right: 8px; vertical-align: middle;'></span> High Latency Edge", unsafe_allow_html=True)

# Display Alerts
st.subheader("🚨 Active Alerts")
alerts = []
for u, v, data in G.edges(data=True):
    if data['response_time_ms'] > RESPONSE_TIME_SLA_MS:
        alerts.append(f"🔴 **High Latency:** Connection `{u} ↔ {v}` is at **{data['response_time_ms']}ms** (SLA: <{RESPONSE_TIME_SLA_MS}ms)")
    if data['error_rate_per_min'] > ERROR_RATE_SLA_PER_MIN:
        alerts.append(f"🟠 **High Error Rate:** Connection `{u} ↔ {v}` has **{data['error_rate_per_min']:.1f} errors/min** (SLA: <{ERROR_RATE_SLA_PER_MIN})")
for node in node_data:
    if node['severity'] == 'critical':
        alerts.append(f"🔥 **Critical Node:** `{node['name']}` requires immediate attention.")
    if node['status'] == 'DOWN':
        alerts.append(f"💀 **Service Down:** `{node['name']}` is unresponsive.")

if alerts:
    for alert in sorted(alerts): st.warning(alert)
else:
    st.success("✅ All services and connections are operating within defined SLAs.")

# Display Detailed Node Information Table with Custom Styling
st.subheader("📋 Node Details")
df_nodes = pd.DataFrame(node_data)
df_nodes['Status Icon'] = df_nodes['severity'].map(SEVERITY_ICONS)
df_display = df_nodes[['Status Icon', 'name', 'severity', 'status', 'last_updated']]

def highlight_severity(row):
    color = SEVERITY_COLORS.get(row.severity, '')
    return [f'background-color: {color}33' for _ in row]

styler = df_display.style.apply(highlight_severity, axis=1)
# --- Increase size of the status icon cell ---
styler.set_properties(subset=['Status Icon'], **{'font-size': '1.8rem', 'text-align': 'center'})

st.dataframe(styler, use_container_width=True, hide_index=True)

# Display "How to Use" Expander
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    - **Interactive Map:** Pan by clicking and dragging. Zoom with your mouse wheel. Hover over nodes and connections for detailed metrics.
    - **KPIs:** The top cards show a high-level summary of the network's current health.
    - **Alerts:** Critical issues that violate predefined Service Level Agreements (SLAs) are automatically flagged.
    - **Sidebar Controls:** Customize the view, switch data sources, and adjust refresh rates.
    - **Node Details:** The table provides a sortable, detailed view of every service being monitored.
    """)

# --- 7. Auto-Refresh Logic ---
st.caption(f"Dashboard last updated at: {st.session_state.last_update_time}")
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()