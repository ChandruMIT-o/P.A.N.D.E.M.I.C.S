import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
import pickle
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import os
import folium
from matplotlib.animation import FuncAnimation
from io import BytesIO
import os
import tempfile

import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'


st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    RoadNetwork = pd.read_csv('derived_data/downsampled_road_network.csv')
    airports = pd.read_csv('derived_data/processed_airports.csv')
    
    GRAPH_FILE = 'derived_data/road_airport_graph.pkl'
    with open(GRAPH_FILE, 'rb') as f:
        G = pickle.load(f)
    return RoadNetwork, airports, G

RoadNetwork, airports, G = load_data()


def run_seir_simulation(params):
    time_steps = params['time_steps']
    R0 = params['R0']
    INCUBATION_PERIOD = params['incubation_period']
    INFECTION_PERIOD = params['infection_period']
    LOCKDOWN_EFFECT = 1 - params['lockdown_effect']
    HANDWASH_EFFECT = 1 - params['handwash_effect']
    MASK_EFFECT = 1 - params['mask_effect']
    LOCKDOWN_DAY = params['lockdown_day']
    INITIAL_INFECTED = params['initial_infected']

    beta, sigma, gamma = R0 / INFECTION_PERIOD, 1 / INCUBATION_PERIOD, 1 / INFECTION_PERIOD
    population = RoadNetwork['population_density'].values
    num_nodes = len(G.nodes)

    mobility_matrix = nx.to_numpy_array(G, weight='weight')
    airport_population = np.full(num_nodes - len(population), population.mean() * 0.1)
    population = np.concatenate([population, airport_population])
    population = np.maximum(population, 1e-3)

    S = population.copy()
    E = np.zeros(num_nodes)
    I = np.zeros(num_nodes)
    R = np.zeros(num_nodes)

    top_nodes = np.argsort(population)[-params['top_k_initial_nodes']:]
    for node in top_nodes:
        I[node] = INITIAL_INFECTED
        S[node] -= INITIAL_INFECTED

    results = {'S': [], 'E': [], 'I': [], 'R': []}
    I_per_timestep = []

    progress_bar = st.progress(0)
    for t in range(time_steps):
        progress_bar.progress((t + 1) / time_steps)

        current_beta = beta
        if t >= LOCKDOWN_DAY:
            current_beta *= LOCKDOWN_EFFECT
        current_beta *= HANDWASH_EFFECT
        current_beta *= MASK_EFFECT

        new_exposed = current_beta * S * (np.dot(mobility_matrix, I / population))
        new_exposed = np.minimum(new_exposed, S)
        new_infectious = sigma * E
        new_recovered = gamma * I

        S -= new_exposed
        E += new_exposed - new_infectious
        I += new_infectious - new_recovered
        R += new_recovered

        results['S'].append(S.sum())
        results['E'].append(E.sum())
        results['I'].append(I.sum())
        results['R'].append(R.sum())
        I_per_timestep.append(I.copy())
    
    progress_bar.empty()  
    return results, I_per_timestep

css = """
<style>
[data-testid = "stAppViewContainer"]{
background-color: #000000;
opacity: 1;
background: linear-gradient(135deg, #19191955 25%, transparent 25%) -16px 0/ 32px 32px, linear-gradient(225deg, #191919 25%, transparent 25%) -16px 0/ 32px 32px, linear-gradient(315deg, #19191955 25%, transparent 25%) 0px 0/ 32px 32px, linear-gradient(45deg, #191919 25%, #000000 25%) 0px 0/ 32px 32px;
}

[data-testid = "stSidebar"]{
background-color: #000000;
opacity: 1;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #000000 16px ), repeating-linear-gradient( #19191955, #191919 );
}
</style>
"""

# st.markdown(css, unsafe_allow_html=True)

st.title("P.A.N.D.E.M.I.C.S")
st.text(body="Pandemic Analysis by Networked Disease Emergence, Mitigation, Investigation, and Control Simulator.")

st.sidebar.header("Control Parameters")
st.sidebar.markdown("Adjust the parameters below to tune the simulation:")

params = {
    'time_steps': st.sidebar.slider("Simulation Duration (Days)", 50, 500, 100, help="Total days to simulate."),
    'R0': st.sidebar.slider("Basic Reproduction Rate (R0)", 1.0, 15.0, 5.0, help="Average number of people infected by one person."),
    'incubation_period': st.sidebar.slider("Incubation Period (Days)", 2, 50, 5, help="Time for exposed individuals to become infectious."),
    'infection_period': st.sidebar.slider("Infection Period (Days)", 5, 50, 10, help="Duration of infectiousness."),
    'lockdown_day': st.sidebar.slider("Lockdown Start Day", 1, 200, 20, help="Day lockdown starts."),
    'lockdown_effect': st.sidebar.slider("Lockdown Effectiveness (%)", 0.0, 1.0, 0.3, help="Reduction in transmission rate."),
    'handwash_effect': st.sidebar.slider("Handwash Effectiveness (%)", 0.0, 1.0, 0.1, help="Reduction in transmission rate."),
    'mask_effect': st.sidebar.slider("Mask Effectiveness (%)", 0.0, 1.0, 0.2, help="Reduction in transmission due to masks."),
    'initial_infected': st.sidebar.number_input("Initial Infections per Node", 10, 1000, 100),
    'top_k_initial_nodes': st.sidebar.slider("Top K Populated Nodes to Infect", 1, 20, 3, help="Number of top populated nodes initially infected."),
}

results, I_per_timestep = run_seir_simulation(params)


st.header("SEIR Interactive Plot")
time_range = list(range(params['time_steps']))

fig = go.Figure()
fig.add_trace(go.Scatter(x=time_range, y=results['S'], name='Susceptible'))
fig.add_trace(go.Scatter(x=time_range, y=results['E'], name='Exposed'))
fig.add_trace(go.Scatter(x=time_range, y=results['I'], name='Infectious'))
fig.add_trace(go.Scatter(x=time_range, y=results['R'], name='Recovered'))
fig.update_layout(xaxis_title="Days", yaxis_title="Population")
st.plotly_chart(fig)


st.header("Infection Spread Visualization")

if st.button("Generate Video"):
    progress_bar = st.progress(0, text="Generating Video Sequences...")

    airports_selected = airports[['lon', 'lat', 'population_density']]
    airports_selected.columns = ['X', 'Y', 'population_density']  

    all_nodes = pd.concat([
        RoadNetwork[['X', 'Y', 'population_density']],
        airports_selected
    ], ignore_index=True)

    all_nodes.reset_index(drop=True, inplace=True)

    def update_map(I, t):
        map_center = [all_nodes['Y'].mean(), all_nodes['X'].mean()]
        disease_map = folium.Map(location=map_center, zoom_start=6)
        max_infections = max(I) if max(I) > 0 else 1
        normalized_infections = I / max_infections

        for i, node in all_nodes.iterrows():
            intensity = min(1, normalized_infections[i])
            folium.CircleMarker(
                location=(node['Y'], node['X']),
                radius=5 + 15 * intensity,
                color='red',
                fill=True,
                fill_opacity=0.6,
                popup=f"Day {t}: Infections {int(I[i])}",
            ).add_to(disease_map)

    for t, I in enumerate(I_per_timestep):
        progress_bar.progress((t + 1) / len(I_per_timestep), text=f"Generating Video Sequences: {t}/{params['time_steps']}")
        update_map(I, t)

    progress_bar.empty() 
    progress_bar2 = st.progress(0, text="Compiling and Exporting Video...")
    
    def animate_infection_spread(I_per_timestep):
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(all_nodes['X'], all_nodes['Y'], c=I_per_timestep[0], cmap='Reds', s=20, alpha=0.8)

        def update(frame):
            scatter.set_array(I_per_timestep[frame])
            ax.set_title(f"Infection Spread (Day {frame})")
            return scatter,
        
        progress_bar2.progress(30, text=f"Compiling and Exporting Video [Few More Seconds]...")

        # Create a temporary file for storing the video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            ani = FuncAnimation(fig, update, frames=len(I_per_timestep), blit=False)
            temp_file_path = temp_file.name
            ani.save(temp_file_path, fps=5)

        # Read the video file back into a BytesIO object for streaming
        with open(temp_file_path, 'rb') as f:
            video_data = f.read()

        # Clean up the temporary file
        os.remove(temp_file_path)

        return video_data

    video_data = animate_infection_spread(I_per_timestep)
    st.success("Video generation complete!")
    progress_bar2.empty()

    progress_bar.empty()
    st.video(video_data)


