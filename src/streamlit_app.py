import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import re
from PIL import Image
import sys
import base64
from io import BytesIO
import subprocess
import pkg_resources

sys.path.append(".")
from models.llm_interface import LLMInterface
from utils.data_processor import DataProcessor
from visualization.matplotlib_visualizer import MatplotlibVisualizer
from visualization.utils import extract_scenario_description, extract_enhancement_strategy, extract_vehicle_trajectories
from agents import AgentManager
from evalution import ScenarioEvaluator, AblationStudy
from utils.dev_tools import compare_llm_scenarios, LLMScenarioComparator

# Set page configuration
st.set_page_config(
    page_title="LLMScenario Framework",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme
COLORS = {
    'primary': '#4C72B0',
    'secondary': '#55A868',
    'accent': '#C44E52',
    'background': '#F8F9FA',
    'text': '#333333',
    'gpt4': '#21A179',
    'claude': '#8E44AD',
    'gemini': '#F39C12'
}


# Cache functions for better performance
@st.cache_data
def load_sample_data():
    """Load sample data if real data is not available."""
    # Create a sample dataframe similar to HighD format
    df = pd.DataFrame({
        'id': np.repeat(range(1, 6), 50),
        'frame': np.tile(range(1, 51), 5),
        'x': np.random.uniform(0, 500, 250),
        'y': np.tile(np.array([9.39, 13.5, 21.68, 25.0, 17.5]), 50) + np.random.normal(0, 0.1, 250),
        'xVelocity': np.random.uniform(20, 40, 250),
        'yVelocity': np.random.normal(0, 0.5, 250),
        'laneId': np.tile(np.array([2, 3, 5, 6, 4]), 50)
    })
    return df


@st.cache_data
def load_existing_scenario(llm_type):
    """Load existing scenario from file."""
    file_path = f'output/{llm_type}_scenario.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    return None


@st.cache_resource
def get_llm_interface():
    """Get LLM interface."""
    return LLMInterface()


@st.cache_resource
def get_visualizer():
    """Get visualizer."""
    return MatplotlibVisualizer()


@st.cache_resource
def get_agent_manager():
    """Get agent manager."""
    return AgentManager()


def generate_scenario_with_llm(llm_type, road_env, vehicle_states, tasks_interactions):
    """Generate scenario with specified LLM."""
    llm_interface = get_llm_interface()

    if llm_type == "gpt4":
        return llm_interface.generate_with_gpt4(road_env, vehicle_states, tasks_interactions)
    elif llm_type == "claude":
        return llm_interface.generate_with_claude(road_env, vehicle_states, tasks_interactions)
    elif llm_type == "gemini":
        return llm_interface.generate_with_gemini(road_env, vehicle_states, tasks_interactions)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def visualize_scenario(scenario_text, llm_type):
    """Visualize scenario."""
    visualizer = get_visualizer()
    try:
        # First try to generate animated visualization
        video_path = visualizer.visualize(scenario_text, f"{llm_type}_scenario", llm_type)
        return video_path, "video"
    except Exception as e:
        st.warning(f"Error with animated visualization: {str(e)}")
        st.info("Falling back to static visualization...")
        try:
            # Fall back to static visualization
            image_path = visualizer.generate_static_visualization(scenario_text, f"{llm_type}_scenario", llm_type)
            return image_path, "image"
        except Exception as e2:
            st.error(f"Error with static visualization: {str(e2)}")
            return None, None


def enhance_scenario_with_agents(scenario_description):
    """Enhance scenario with agents."""
    agent_manager = get_agent_manager()

    try:
        # Create an example event
        event_description = "A vehicle ahead suddenly brakes hard, reducing speed by 50% in 2 seconds."

        # Get agent reactions
        reactions_output = agent_manager.react_to_scenario_event(scenario_description, event_description)

        # Process CrewOutput to extract actual reactions
        reactions = []
        if hasattr(reactions_output, 'values') and callable(getattr(reactions_output, 'values', None)):
            # If it's a CrewOutput with values() method
            reactions = list(reactions_output.values())
        elif hasattr(reactions_output, 'items') and callable(getattr(reactions_output, 'items', None)):
            # If it's a dictionary
            reactions = list(reactions_output.items())
        elif isinstance(reactions_output, list):
            # If it's already a list
            reactions = reactions_output
        elif isinstance(reactions_output, str):
            # If it's a string
            reactions = [reactions_output]
        else:
            # If we can't determine type, convert to string
            reactions = [str(reactions_output)]

        # Populate the scenario with different driver types
        trajectories_output = agent_manager.populate_scenario_with_agents(scenario_description)

        # Process CrewOutput to extract actual trajectories
        trajectories = []
        if hasattr(trajectories_output, 'values') and callable(getattr(trajectories_output, 'values', None)):
            # If it's a CrewOutput with values() method
            trajectories = list(trajectories_output.values())
        elif hasattr(trajectories_output, 'items') and callable(getattr(trajectories_output, 'items', None)):
            # If it's a dictionary
            trajectories = list(trajectories_output.items())
        elif isinstance(trajectories_output, list):
            # If it's already a list
            trajectories = trajectories_output
        elif isinstance(trajectories_output, str):
            # If it's a string
            trajectories = [trajectories_output]
        else:
            # If we can't determine type, convert to string
            trajectories = [str(trajectories_output)]

        return event_description, reactions, trajectories
    except Exception as e:
        st.error(f"Error in agent enhancement: {str(e)}")
        # Return empty defaults in case of error
        return "Example event", [], []


def evaluate_scenarios(scenarios):
    """Evaluate scenarios."""
    evaluator = ScenarioEvaluator(output_dir='output/evaluation')

    # Evaluate scenarios
    results = []
    for llm_type, scenario_text in scenarios.items():
        if scenario_text:
            result = evaluator.evaluate_scenario(scenario_text, llm_type)
            results.append(result)

    # Generate comparison chart
    if results:
        chart_path = evaluator.generate_comparison_charts(results)
        return results, chart_path

    return [], None


def create_radar_chart(results):
    """Create a radar chart for scenario comparison."""
    if not results:
        return None

    # Extract data for radar chart
    metrics = ['vehicle_count', 'interaction_count', 'trajectory_completeness',
               'physical_realism', 'complexity']
    scenario_names = [result['scenario_name'] for result in results]

    # Normalize values
    max_values = {}
    for metric in metrics:
        max_values[metric] = max(result.get(metric, 0) for result in results) or 1

    normalized_values = []
    for result in results:
        normalized = [result.get(metric, 0) / max_values[metric] for metric in metrics]
        normalized_values.append(normalized)

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, values in enumerate(normalized_values):
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=scenario_names[i])
        ax.fill(angles, values, alpha=0.1)

    # Set labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], [metric.replace('_', ' ').title() for metric in metrics])

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf


def render_metrics_table(results):
    """Render a metrics table."""
    if not results:
        return

    df = pd.DataFrame(results)
    df = df.set_index('scenario_name')
    df = df[['vehicle_count', 'interaction_count', 'lane_change_count',
             'trajectory_completeness', 'physical_realism', 'complexity']]
    df.columns = [col.replace('_', ' ').title() for col in df.columns]

    return df


def main():
    """Main Streamlit app."""
    # Custom CSS
    st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3 {{
        color: {COLORS['primary']};
    }}
    .stProgress > div > div {{
        background-color: {COLORS['secondary']};
    }}
    .highlight {{
        background-color: {COLORS['background']};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid {COLORS['accent']};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("🚗 LLMScenario Framework")
        st.markdown("---")

        # Add option to force new scenario generation
        st.subheader("Scenario Generation Options")
        force_new_generation = st.checkbox("Force new scenario generation (overwrite existing)", value=False)
        if force_new_generation:
            st.session_state.force_new_generation = True
        else:
            st.session_state.force_new_generation = False

        st.subheader("Configuration")
        run_data_processing = st.checkbox("Process Data", value=True)
        run_llm_generation = st.checkbox("Generate Scenarios", value=True)
        run_visualization = st.checkbox("Visualize Scenarios", value=True)
        run_agent_enhancement = st.checkbox("Agent Enhancement", value=False)
        run_evaluation = st.checkbox("Evaluation & Analysis", value=True)

        st.markdown("---")
        st.subheader("LLM Selection")
        use_gpt4 = st.checkbox("GPT-4", value=True)
        use_claude = st.checkbox("Claude 3.7", value=True)
        use_gemini = st.checkbox("Gemini", value=True)

        st.markdown("---")
        if st.button("🚀 Run Framework", type="primary"):
            st.session_state.run_framework = True
            st.session_state.progress = 0

        if st.button("♻️ Reset App"):
            st.session_state.clear()
            st.experimental_rerun()

    # Main content
    st.title("LLMScenario Framework")
    st.markdown("Advanced scenario generation for autonomous driving research")

    # Initialize session state variables
    if 'run_framework' not in st.session_state:
        st.session_state.run_framework = False
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
    if 'scenario_descriptions' not in st.session_state:
        st.session_state.scenario_descriptions = {}
    if 'enhancement_strategies' not in st.session_state:
        st.session_state.enhancement_strategies = {}
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = {}
    if 'agent_enhancements' not in st.session_state:
        st.session_state.agent_enhancements = {}
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'road_env' not in st.session_state:
        st.session_state.road_env = None
    if 'vehicle_states' not in st.session_state:
        st.session_state.vehicle_states = None
    if 'tasks_interactions' not in st.session_state:
        st.session_state.tasks_interactions = None

    # If not running the framework, show the initial information
    if not st.session_state.run_framework:
        st.info("Configure the options in the sidebar and click 'Run Framework' to start.")

        # Button to show existing videos
        if st.button("📺 Show Existing Visualizations"):
            st.session_state.show_existing_videos = True

        # Check if videos already exist and should be shown
        if 'show_existing_videos' in st.session_state and st.session_state.show_existing_videos:
            st.subheader("Existing Scenario Visualizations")

            # Check for videos in the output/videos directory
            video_path = "output/videos"
            if os.path.exists(video_path):
                video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi'))]
                image_files = [f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

                if video_files or image_files:
                    # Create tabs for videos and images
                    video_tab, image_tab = st.tabs(["Videos", "Images"])

                    with video_tab:
                        if video_files:
                            for video_file in video_files:
                                full_path = os.path.join(video_path, video_file)
                                st.video(full_path)
                                st.caption(f"Visualization: {video_file}")
                        else:
                            st.info("No video visualizations found.")

                    with image_tab:
                        if image_files:
                            columns = st.columns(min(3, len(image_files)))
                            for i, image_file in enumerate(image_files):
                                with columns[i % 3]:
                                    full_path = os.path.join(video_path, image_file)
                                    st.image(full_path, caption=image_file)
                        else:
                            st.info("No image visualizations found.")
                else:
                    st.warning("No visualization files found in the output/videos directory.")
            else:
                st.warning("The output/videos directory does not exist.")

        # Show overview of the framework
        st.subheader("Framework Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Data Processing**")
            st.markdown("Extract road environment, vehicle states, and interactions from driving data.")

        with col2:
            st.markdown("**LLM Generation**")
            st.markdown("Generate realistic driving scenarios using different LLMs.")

        with col3:
            st.markdown("**Visualization**")
            st.markdown("Visualize the generated scenarios with animated or static representations.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Agent Enhancement**")
            st.markdown("Enhance scenarios with agent-based behaviors of different driver types.")

        with col2:
            st.markdown("**Evaluation**")
            st.markdown("Evaluate scenarios for realism, complexity, and completeness.")

        with col3:
            st.markdown("**Analysis**")
            st.markdown("Compare scenarios across different LLMs and enhancement techniques.")

        return

    # Create a progress bar
    progress_bar = st.progress(st.session_state.progress)

    # Show progress steps
    prog_col1, prog_col2, prog_col3, prog_col4, prog_col5 = st.columns(5)

    with prog_col1:
        if st.session_state.progress >= 20:
            st.success("Data Processing")
        else:
            st.markdown("Data Processing")

    with prog_col2:
        if st.session_state.progress >= 40:
            st.success("LLM Generation")
        else:
            st.markdown("LLM Generation")

    with prog_col3:
        if st.session_state.progress >= 60:
            st.success("Visualization")
        else:
            st.markdown("Visualization")

    with prog_col4:
        if st.session_state.progress >= 80:
            st.success("Agent Enhancement")
        else:
            st.markdown("Agent Enhancement")

    with prog_col5:
        if st.session_state.progress >= 100:
            st.success("Evaluation")
        else:
            st.markdown("Evaluation")

    # Main process log
    st.subheader("📋 Process Log")
    log_container = st.container()

    # 1. Data Processing
    if run_data_processing and st.session_state.progress < 20:
        with log_container:
            st.subheader("1. Data Processing")
            st.write("Loading and processing driving data...")

            # Load sample data or real data
            try:
                tracks_df = pd.read_csv('/Users/shreyanshkhaitan/Downloads/pjt2_final/LLMScenario_legacy/data/processed/tracks3.csv')
                st.success(f"Loaded {len(tracks_df)} records from real data.")
            except:
                st.warning("Real data not found. Using synthetic sample data.")
                tracks_df = load_sample_data()

            # Extract scenario information
            st.write("Extracting scenario information...")
            try:
                road_env, vehicle_states, tasks_interactions = DataProcessor.extract_scenario_from_tracks(tracks_df)
                st.session_state.road_env = road_env
                st.session_state.vehicle_states = vehicle_states
                st.session_state.tasks_interactions = tasks_interactions
            except Exception as e:
                st.error(f"Error in scenario extraction: {str(e)}")
                st.write("Using default scenario information...")

                # Default scenario information if extraction fails
                st.session_state.road_env = """This scenario takes place on a highway with 4 lanes. 
The speed limit is 120 km/h.
Lane IDs present: 2, 3, 5, 6."""

                st.session_state.vehicle_states = """Vehicle 1 (Car):
- Initial position: (362.26, 21.68)
- Final position: (403.33, 21.68)
- Average speed: 41.07 m/s
- Lanes used: 5

Vehicle 2 (Car):
- Initial position: (162.75, 9.39)
- Final position: (130.27, 9.39)
- Average speed: 32.48 m/s
- Lanes used: 2"""

                st.session_state.tasks_interactions = """Identified interactions:
- Vehicle 1 follows vehicle 3
- Vehicle 2 changes from lane 2 to lane 3
- Vehicle 4 changes from lane 5 to lane 6"""

            # Display scenario information
            st.subheader("Scenario Information")
            st.markdown("**Road Environment:**")
            st.code(st.session_state.road_env)

            st.markdown("**Vehicle States:**")
            st.code(st.session_state.vehicle_states)

            st.markdown("**Tasks and Interactions:**")
            st.code(st.session_state.tasks_interactions)

            st.success("Data processing complete!")
            st.session_state.progress = 20
            progress_bar.progress(st.session_state.progress)
            time.sleep(1)

    # 2. LLM Generation
    if run_llm_generation and st.session_state.progress < 40:
        with log_container:
            st.subheader("2. LLM Generation")

            # Ensure output directory exists
            os.makedirs('output', exist_ok=True)

            llm_types = []
            if use_gpt4:
                llm_types.append("gpt4")
            if use_claude:
                llm_types.append("claude")
            if use_gemini:
                llm_types.append("gemini")

            if not llm_types:
                st.warning("No LLMs selected. Skipping generation.")
                st.session_state.progress = 40
                progress_bar.progress(st.session_state.progress)
            else:
                # Generate scenarios or load existing ones
                for llm_type in llm_types:
                    st.write(f"Processing {llm_type.upper()} scenario...")

                    # Check if we have an existing scenario
                    existing_scenario = load_existing_scenario(llm_type)

                    if existing_scenario and not st.session_state.get('force_new_generation', False):
                        st.info(f"Found existing {llm_type.upper()} scenario. Using it.")
                        st.session_state.scenarios[llm_type] = existing_scenario
                    else:
                        if existing_scenario and st.session_state.get('force_new_generation', False):
                            st.info(
                                f"Found existing {llm_type.upper()} scenario, but generating a new one as requested.")
                        else:
                            st.info(f"No existing {llm_type.upper()} scenario found. Generating a new one.")

                        st.write(f"Generating new {llm_type.upper()} scenario...")
                        try:
                            # Generate scenario
                            scenario_text = generate_scenario_with_llm(
                                llm_type,
                                st.session_state.road_env,
                                st.session_state.vehicle_states,
                                st.session_state.tasks_interactions
                            )

                            st.session_state.scenarios[llm_type] = scenario_text

                            # Save scenario
                            with open(f'output/{llm_type}_scenario.txt', 'w') as f:
                                f.write(scenario_text)

                            st.success(f"{llm_type.upper()} scenario generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating {llm_type.upper()} scenario: {str(e)}")

                    # Extract scenario sections
                    if llm_type in st.session_state.scenarios:
                        description = extract_scenario_description(st.session_state.scenarios[llm_type], llm_type)
                        enhancement = extract_enhancement_strategy(st.session_state.scenarios[llm_type], llm_type)

                        st.session_state.scenario_descriptions[llm_type] = description
                        st.session_state.enhancement_strategies[llm_type] = enhancement

                        with st.expander(f"{llm_type.upper()} Scenario Description"):
                            st.markdown(description)

                        with st.expander(f"{llm_type.upper()} Enhancement Strategy"):
                            st.markdown(enhancement)

                    # Add a delay to avoid rate limiting
                    time.sleep(1)

                st.success("LLM Generation complete!")
                st.session_state.progress = 40
                progress_bar.progress(st.session_state.progress)

    # 3. Visualization
    if run_visualization and st.session_state.progress < 60:
        with log_container:
            st.subheader("3. Visualization")

            # Ensure visualization output directories exist
            os.makedirs('output/videos', exist_ok=True)

            if not st.session_state.scenarios:
                st.warning("No scenarios available for visualization.")
                st.session_state.progress = 60
                progress_bar.progress(st.session_state.progress)
            else:
                for llm_type, scenario_text in st.session_state.scenarios.items():
                    st.write(f"Visualizing {llm_type.upper()} scenario...")

                    try:
                        # Visualize scenario
                        path, viz_type = visualize_scenario(scenario_text, llm_type)

                        if path and os.path.exists(path):
                            st.session_state.visualizations[llm_type] = (path, viz_type)
                            st.success(f"{llm_type.upper()} visualization created successfully!")
                        else:
                            st.warning(f"No visualization created for {llm_type.upper()}.")
                    except Exception as e:
                        st.error(f"Error visualizing {llm_type.upper()} scenario: {str(e)}")

                st.success("Visualization complete!")
                st.session_state.progress = 60
                progress_bar.progress(st.session_state.progress)

    # 4. Agent Enhancement
    if run_agent_enhancement and st.session_state.progress < 80:
        with log_container:
            st.subheader("4. Agent Enhancement")

            # Load scenarios if they're not already in session state
            if not st.session_state.scenarios:
                for llm_type in ['gpt4', 'claude', 'gemini']:
                    existing_scenario = load_existing_scenario(llm_type)
                    if existing_scenario:
                        st.session_state.scenarios[llm_type] = existing_scenario

                        # Also extract descriptions if needed
                        if llm_type not in st.session_state.scenario_descriptions:
                            description = extract_scenario_description(existing_scenario, llm_type)
                            st.session_state.scenario_descriptions[llm_type] = description

            if not st.session_state.scenarios:
                st.warning("No scenarios available for agent enhancement. Please run LLM generation first.")
                st.session_state.progress = 80
                progress_bar.progress(st.session_state.progress)
            else:
                # Select the best scenario for agent enhancement
                best_scenario = list(st.session_state.scenarios.keys())[0]  # Default to first available
                for scenario_type in ['claude', 'gpt4', 'gemini']:  # Prioritize Claude if available
                    if scenario_type in st.session_state.scenarios:
                        best_scenario = scenario_type
                        break

                st.write(f"Enhancing {best_scenario.upper()} scenario with agent reactions...")

                try:
                    # Enhance scenario with agents
                    event_description, reactions, trajectories = enhance_scenario_with_agents(
                        st.session_state.scenario_descriptions[best_scenario]
                    )

                    st.session_state.agent_enhancements[best_scenario] = {
                        'event': event_description,
                        'reactions': reactions,
                        'trajectories': trajectories
                    }

                    # Save agent reactions and trajectories
                    with open(f'output/{best_scenario}_agent_reactions.txt', 'w') as f:
                        f.write(f"Event: {event_description}\n\n")
                        for i, reaction in enumerate(reactions):
                            f.write(f"Agent {i + 1} Reaction:\n{reaction}\n\n")

                    with open(f'output/{best_scenario}_agent_trajectories.txt', 'w') as f:
                        for i, trajectory in enumerate(trajectories):
                            f.write(f"Agent {i + 1} Trajectory:\n{trajectory}\n\n")

                    # Display agent reactions
                    st.write("Agent Reactions:")
                    for i, reaction in enumerate(reactions[:2]):  # Show first 2 reactions
                        with st.expander(f"Agent {i + 1} Reaction"):
                            st.write(reaction[:500] + "..." if len(reaction) > 500 else reaction)

                    st.success("Agent enhancement complete!")
                except Exception as e:
                    st.error(f"Error in agent enhancement: {str(e)}")

                st.session_state.progress = 80
                progress_bar.progress(st.session_state.progress)

    # 5. Evaluation and Analysis
    if run_evaluation and st.session_state.progress < 100:
        with log_container:
            st.subheader("5. Evaluation & Analysis")

            if not st.session_state.scenarios:
                st.warning("No scenarios available for evaluation.")
                st.session_state.progress = 100
                progress_bar.progress(st.session_state.progress)
            else:
                # Ensure evaluation output directory exists
                os.makedirs('output/evaluation', exist_ok=True)

                st.write("Evaluating scenarios...")

                # Evaluate scenarios
                results, chart_path = evaluate_scenarios(st.session_state.scenarios)
                st.session_state.evaluation_results = results

                if results:
                    st.success("Evaluation complete!")

                    # Create metrics table
                    metrics_df = render_metrics_table(results)
                    if metrics_df is not None:
                        st.write("Evaluation Metrics:")
                        st.dataframe(metrics_df)

                    # Create radar chart
                    radar_buf = create_radar_chart(results)
                    if radar_buf:
                        st.write("Comparative Analysis:")
                        st.image(radar_buf, caption="Scenario Comparison Radar Chart", use_container_width=True)

                    # Display comparison report
                    st.write("Scenario Comparison Report:")
                    comparator = LLMScenarioComparator(st.session_state.scenarios)
                    report = comparator.generate_comparative_report()
                    st.code(report, language='markdown')
                else:
                    st.warning("No evaluation results available.")

                st.session_state.progress = 100
                progress_bar.progress(st.session_state.progress)

    # Show results after all steps are complete
    if st.session_state.progress >= 100:
        st.success("Framework execution complete!")

        # Results tabs
        tab1, tab2, tab3 = st.tabs(["Generated Scenarios", "Visualizations", "Evaluation"])

        with tab1:
            st.subheader("Generated Scenarios")

            for llm_type in st.session_state.scenarios:
                st.markdown(f"### {llm_type.upper()} Scenario")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Scenario Description**")
                    st.markdown(st.session_state.scenario_descriptions.get(llm_type, "No description available"))

                with col2:
                    st.markdown("**Enhancement Strategy**")
                    st.markdown(st.session_state.enhancement_strategies.get(llm_type, "No strategy available"))

        with tab2:
            st.subheader("Scenario Visualizations")

            if not st.session_state.visualizations:
                st.info("No visualizations available.")
            else:
                cols = st.columns(len(st.session_state.visualizations))

                for i, (llm_type, (path, viz_type)) in enumerate(st.session_state.visualizations.items()):
                    with cols[i]:
                        st.markdown(f"**{llm_type.upper()} Scenario**")

                        if viz_type == "video" and path.endswith(".mp4"):
                            video_file = open(path, 'rb')
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                        elif viz_type == "image" and path.endswith((".png", ".jpg")):
                            st.image(path, caption=f"{llm_type.upper()} Scenario")
                        else:
                            st.warning(f"Visualization for {llm_type} is not available or in an unsupported format.")

        with tab3:
            st.subheader("Evaluation Results")

            if not st.session_state.evaluation_results:
                st.info("No evaluation results available.")
            else:
                # Create metrics table
                metrics_df = render_metrics_table(st.session_state.evaluation_results)
                if metrics_df is not None:
                    st.write("Evaluation Metrics:")
                    st.dataframe(metrics_df)

                # Show radar chart
                radar_buf = create_radar_chart(st.session_state.evaluation_results)
                if radar_buf:
                    st.image(radar_buf, caption="Scenario Comparison Radar Chart", use_container_width=False)

                # Create a bar chart comparing key metrics
                st.write("Key Metrics Comparison:")
                results = st.session_state.evaluation_results
                scenario_names = [result['scenario_name'] for result in results]

                # Select key metrics
                key_metrics = ['physical_realism', 'complexity', 'trajectory_completeness']
                key_metric_values = {metric: [] for metric in key_metrics}

                for result in results:
                    for metric in key_metrics:
                        key_metric_values[metric].append(result.get(metric, 0))

                # Convert to dataframe for plotting
                plot_df = pd.DataFrame({
                    'Scenario': scenario_names * len(key_metrics),
                    'Metric': [metric.replace('_', ' ').title() for metric in key_metrics for _ in scenario_names],
                    'Value': [value for metric in key_metrics for value in key_metric_values[metric]]
                })

                # Plot using Streamlit bar chart
                st.bar_chart(plot_df, x='Scenario', y='Value', color='Metric')


if __name__ == "__main__":
    main()