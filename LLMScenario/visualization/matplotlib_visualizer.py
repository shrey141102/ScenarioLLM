"""
Matplotlib-based visualizer for scenario simulation.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms


class MatplotlibVisualizer:
    def __init__(self, output_dir='output/videos'):
        """
        Initialize the Matplotlib visualizer.

        Args:
            output_dir (str): Directory to save output videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_trajectory_data(self, scenario_text, llm_type):
        """Parse vehicle trajectory data from LLM-generated scenario."""
        # Extract trajectories section based on LLM type
        if llm_type == 'gpt4':
            trajectories_match = re.search(r'\[VEHICLE_TRAJECTORIES\](.*?)\[/VEHICLE_TRAJECTORIES\]', scenario_text,
                                           re.DOTALL)
        elif llm_type == 'claude':
            trajectories_match = re.search(r'<vehicle_trajectories>(.*?)</vehicle_trajectories>', scenario_text,
                                           re.DOTALL)
        elif llm_type == 'gemini':
            trajectories_match = re.search(r'\[VEHICLE_TRAJECTORIES\](.*?)\[/VEHICLE_TRAJECTORIES\]', scenario_text,
                                           re.DOTALL)
        else:
            # Try a more generic pattern if the LLM type is unknown
            trajectories_match = re.search(
                r'(Vehicle ID: \d+.*?Time \(s\), X \(m\), Y \(m\), Velocity \(m/s\), Heading \(rad\).*?)(?=\n\s*$|\Z)',
                scenario_text, re.DOTALL)

        if not trajectories_match:
            print(f"No trajectory data found for {llm_type} scenario")
            return {}

        trajectories_text = trajectories_match.group(1).strip()

        # Extract individual vehicle trajectories
        vehicle_pattern = r"Vehicle ID: (\d+).*?\n(Time \(s\), X \(m\), Y \(m\), Velocity \(m/s\), Heading \(rad\)\n(?:\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\n)+)"
        vehicle_matches = re.findall(vehicle_pattern, trajectories_text, re.DOTALL)

        trajectories = {}

        for vehicle_id, trajectory_text in vehicle_matches:
            # Extract each line of trajectory data
            trajectory_lines = trajectory_text.strip().split('\n')[1:]  # Skip header

            vehicle_trajectory = []
            for line in trajectory_lines:
                try:
                    time, x, y, velocity, heading = map(float, line.split(', '))
                    vehicle_trajectory.append({
                        'time': time,
                        'x': x,
                        'y': y,
                        'velocity': velocity,
                        'heading': heading
                    })
                except ValueError:
                    continue  # Skip invalid lines

            if vehicle_trajectory:  # Only add if we have valid trajectory points
                trajectories[vehicle_id] = vehicle_trajectory

        return trajectories

    def _parse_raw_trajectory_data(self, trajectory_text):
        """Parse raw trajectory text into structured data."""
        trajectories = {}

        # Extract individual vehicle trajectories
        vehicle_pattern = r"Vehicle ID: (\d+).*?\n(Time \(s\), X \(m\), Y \(m\), Velocity \(m/s\), Heading \(rad\)\n(?:\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\n)+)"
        vehicle_matches = re.findall(vehicle_pattern, trajectory_text, re.DOTALL)

        for vehicle_id, trajectory_text in vehicle_matches:
            # Extract each line of trajectory data
            trajectory_lines = trajectory_text.strip().split('\n')[1:]  # Skip header

            vehicle_trajectory = []
            for line in trajectory_lines:
                try:
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        time, x, y, velocity, heading = map(float, parts[:5])
                        vehicle_trajectory.append({
                            'time': time,
                            'x': x,
                            'y': y,
                            'velocity': velocity,
                            'heading': heading
                        })
                except ValueError as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue  # Skip invalid lines

            if vehicle_trajectory:  # Only add if we have valid trajectory points
                trajectories[vehicle_id] = vehicle_trajectory

        return trajectories

    def _direct_trajectory_extraction(self, scenario_text):
        """Try to extract trajectory data directly from the scenario text."""
        trajectories = {}

        # Look for trajectory data in any format
        matches = re.findall(
            r'Vehicle ID: (\d+).*?Time \(s\), X \(m\), Y \(m\), Velocity \(m/s\), Heading \(rad\)(.*?)(?=Vehicle ID:|$)',
            scenario_text, re.DOTALL)

        for vehicle_id, trajectory_block in matches:
            vehicle_trajectory = []

            # Extract numeric lines
            numeric_lines = re.findall(r'(\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+)', trajectory_block)

            for line in numeric_lines:
                try:
                    time, x, y, velocity, heading = map(float, line.split(', '))
                    vehicle_trajectory.append({
                        'time': time,
                        'x': x,
                        'y': y,
                        'velocity': velocity,
                        'heading': heading
                    })
                except ValueError:
                    continue

            if vehicle_trajectory:
                trajectories[vehicle_id] = vehicle_trajectory

        return trajectories

    def visualize(self, scenario_text, scenario_name, llm_type=None, fps=10, duration=10):
        """
        Visualize the scenario using Matplotlib.

        Args:
            scenario_text (str): LLM-generated scenario text
            scenario_name (str): Name of the scenario
            llm_type (str, optional): Type of LLM that generated the scenario
            fps (int): Frames per second for the animation
            duration (int): Duration of the animation in seconds

        Returns:
            str: Path to the output video file
        """
        print(f"Parsing trajectory data for {scenario_name}...")

        # Try different methods to extract trajectory data
        trajectories = {}

        # Method 1: Use our extract_vehicle_trajectories function
        if llm_type:
            try:
                from visualization.utils import extract_vehicle_trajectories
            except ImportError:
                from utils import extract_vehicle_trajectories

            trajectory_text = extract_vehicle_trajectories(scenario_text, llm_type)
            if trajectory_text and trajectory_text != "No vehicle trajectories found":
                # Now parse the trajectory data from the extracted text
                trajectories = self._parse_raw_trajectory_data(trajectory_text)

        # Method 2: Try direct regex extraction if method 1 failed
        if not trajectories:
            trajectories = self._direct_trajectory_extraction(scenario_text)

        # Check if we have any valid trajectory data
        if not trajectories:
            print(f"No valid trajectory data found for {scenario_name}")
            raise ValueError("No valid trajectory data found in the scenario text")

        print(f"Found trajectories for {len(trajectories)} vehicles")

        # Determine time range
        all_times = []
        for vehicle_id, trajectory in trajectories.items():
            for point in trajectory:
                all_times.append(point['time'])

        min_time = min(all_times) if all_times else 0
        max_time = max(all_times) if all_times else 10

        # Determine spatial range
        all_x = []
        all_y = []
        for vehicle_id, trajectory in trajectories.items():
            for point in trajectory:
                all_x.append(point['x'])
                all_y.append(point['y'])

        min_x = min(all_x) - 50 if all_x else 0
        max_x = max(all_x) + 50 if all_x else 1000
        min_y = min(all_y) - 20 if all_y else 0
        max_y = max(all_y) + 20 if all_y else 100

        # Create a colormap for vehicles
        # colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
        colors = plt.colormaps['jet'](np.linspace(0, 1, len(trajectories)))

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Vehicle size (length, width) in meters
        vehicle_size = (5, 2)

        # List to store vehicle patches and texts
        vehicles = {}
        vehicle_labels = {}

        # Initialize vehicles
        for i, (vehicle_id, trajectory) in enumerate(trajectories.items()):
            # Create a rectangle for the vehicle
            rect = Rectangle((0, 0), vehicle_size[0], vehicle_size[1],
                             color=colors[i], alpha=0.7)
            vehicles[vehicle_id] = rect

            # Create a text label for the vehicle
            label = ax.text(0, 0, f"V{vehicle_id}", fontsize=8,
                            ha="center", va="center", color="white")
            vehicle_labels[vehicle_id] = label

            # Add the rectangle to the plot
            ax.add_patch(rect)

        # Set the axis limits
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # Add title and labels
        ax.set_title(f"Scenario: {scenario_name}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # Add grid
        ax.grid(False)

        # Draw lanes (simplified)
        # lane_y_values = np.linspace(min_y + 5, max_y - 5, 4)
        # for y in lane_y_values:
        #     ax.axhline(y, color='k', linestyle='--', alpha=0.3)

        # Draw road with lanes
        road_width = 20  # meters
        lane_width = 3.5  # meters
        num_lanes = 4

        # Draw road background
        road_rect = Rectangle((min_x, min_y + 5), max_x - min_x, road_width,
                              facecolor='gray', alpha=0.3, zorder=1)
        ax.add_patch(road_rect)

        # Draw lane markings
        for i in range(num_lanes + 1):
            lane_y = min_y + 5 + i * lane_width
            if i == 0 or i == num_lanes:  # Solid lines for road edges
                ax.axhline(lane_y, color='white', linestyle='-', linewidth=2, alpha=0.7, zorder=2)
            else:  # Dashed lines for lane dividers
                ax.axhline(lane_y, color='white', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

        # Add shoulder areas
        shoulder_width = 2  # meters
        shoulder_top = Rectangle((min_x, min_y + 5 + road_width), max_x - min_x, shoulder_width,
                                 facecolor='green', alpha=0.2, zorder=1)
        shoulder_bottom = Rectangle((min_x, min_y + 5 - shoulder_width), max_x - min_x, shoulder_width,
                                    facecolor='green', alpha=0.2, zorder=1)
        ax.add_patch(shoulder_top)
        ax.add_patch(shoulder_bottom)

        # Function to interpolate vehicle position at a given time
        def interpolate_position(trajectory, time):
            # Find the two closest time points
            times = [point['time'] for point in trajectory]
            if time <= times[0]:
                return trajectory[0]['x'], trajectory[0]['y'], trajectory[0]['heading']
            if time >= times[-1]:
                return trajectory[-1]['x'], trajectory[-1]['y'], trajectory[-1]['heading']

            # Find the index of the first time point greater than the requested time
            next_idx = next(i for i, t in enumerate(times) if t >= time)
            prev_idx = next_idx - 1

            # Get the two surrounding time points
            t1, t2 = times[prev_idx], times[next_idx]

            # Calculate interpolation factor
            factor = (time - t1) / (t2 - t1) if t2 != t1 else 0

            # Interpolate position
            x1, x2 = trajectory[prev_idx]['x'], trajectory[next_idx]['x']
            y1, y2 = trajectory[prev_idx]['y'], trajectory[next_idx]['y']
            h1, h2 = trajectory[prev_idx]['heading'], trajectory[next_idx]['heading']

            x = x1 + factor * (x2 - x1)
            y = y1 + factor * (y2 - y1)
            heading = h1 + factor * (h2 - h1)

            return x, y, heading

        # Animation update function
        def update(frame):
            # Calculate current time
            time = min_time + frame * (max_time - min_time) / (fps * duration)

            # Update title with current time
            ax.set_title(f"Scenario: {scenario_name} - Time: {time:.2f}s")

            # Update each vehicle's position
            for vehicle_id, trajectory in trajectories.items():
                if time < trajectory[0]['time'] or time > trajectory[-1]['time']:
                    # Vehicle not yet on scene or has left
                    vehicles[vehicle_id].set_visible(False)
                    vehicle_labels[vehicle_id].set_visible(False)
                    continue

                # Make vehicle visible
                vehicles[vehicle_id].set_visible(True)
                vehicle_labels[vehicle_id].set_visible(True)

                # Get interpolated position
                x, y, heading = interpolate_position(trajectory, time)

                # Update vehicle position and rotation
                transform = transforms.Affine2D().rotate(heading).translate(x, y - vehicle_size[1] / 2)
                vehicles[vehicle_id].set_transform(transform + ax.transData)

                # Update label position
                vehicle_labels[vehicle_id].set_position((x, y))

            return list(vehicles.values()) + list(vehicle_labels.values())

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=fps * duration,
                                      interval=1000 / fps, blit=True)

        # Save animation as MP4
        output_path = os.path.join(self.output_dir, f"{scenario_name}.mp4")
        ani.save(output_path, writer='ffmpeg', fps=fps)

        # Close the figure
        plt.close(fig)

        return output_path

    def generate_static_visualization(self, scenario_text, scenario_name, llm_type=None):
        """
        Generate a static visualization of the scenario.

        Args:
            scenario_text (str): LLM-generated scenario text
            scenario_name (str): Name of the scenario
            llm_type (str, optional): Type of LLM that generated the scenario

        Returns:
            str: Path to the output image file
        """
        # Try to extract trajectories using both methods
        trajectories = {}

        # Method 1: Use our extract_vehicle_trajectories function
        if llm_type:
            try:
                from visualization.utils import extract_vehicle_trajectories
            except ImportError:
                from utils import extract_vehicle_trajectories

            trajectory_text = extract_vehicle_trajectories(scenario_text, llm_type)
            if trajectory_text and trajectory_text != "No vehicle trajectories found":
                # Now parse the trajectory data from the extracted text
                trajectories = self._parse_raw_trajectory_data(trajectory_text)

        # Method 2: Try direct regex extraction if method 1 failed
        if not trajectories:
            trajectories = self._direct_trajectory_extraction(scenario_text)

        # Check if we have any valid trajectory data
        if not trajectories:
            print(f"No valid trajectory data found for {scenario_name}")
            raise ValueError("No valid trajectory data found in the scenario text")

        # Create a figure for the static visualization
        plt.figure(figsize=(12, 8))

        # Define colors for different vehicles
        # colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
        colors = plt.colormaps['jet'](np.linspace(0, 1, len(trajectories)))

        # Track min/max coordinates for setting axis limits
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        # Plot each vehicle's trajectory
        for i, (vehicle_id, trajectory) in enumerate(trajectories.items()):
            # Extract x, y coordinates
            x_coords = [point['x'] for point in trajectory]
            y_coords = [point['y'] for point in trajectory]

            # Update min/max coordinates
            min_x = min(min_x, min(x_coords))
            max_x = max(max_x, max(x_coords))
            min_y = min(min_y, min(y_coords))
            max_y = max(max_y, max(y_coords))

            # Plot the trajectory
            plt.plot(x_coords, y_coords, 'o-', color=colors[i], label=f"Vehicle {vehicle_id}")

            # Mark start and end points
            plt.plot(x_coords[0], y_coords[0], 'o', color=colors[i], markersize=10, markeredgecolor='black')
            plt.plot(x_coords[-1], y_coords[-1], 's', color=colors[i], markersize=10, markeredgecolor='black')

        # Add some padding to the axis limits
        padding = 50
        plt.xlim(min_x - padding, max_x + padding)
        plt.ylim(min_y - padding, max_y + padding)

        # Add title and labels
        plt.title(f"Scenario: {scenario_name}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(False)
        plt.legend(loc='upper right')

        # Draw lane markers (assuming lanes are roughly horizontal)
        # lane_y_values = [9.39, 13.5, 21.68, 25.0]  # Approximate y-values for lanes 2, 3, 5, 6
        # for y in lane_y_values:
        #     plt.axhline(y, color='black', linestyle='--', alpha=0.3)

        # Draw road with lanes
        road_width = 20  # meters
        lane_width = 3.5  # meters
        num_lanes = 4

        # Calculate road position based on trajectory data
        road_y_center = (min_y + max_y) / 2
        road_y_bottom = road_y_center - road_width / 2

        # Draw road background
        road_rect = plt.Rectangle((min_x - padding, road_y_bottom), max_x - min_x + 2 * padding, road_width,
                                  facecolor='gray', alpha=0.3, zorder=1)
        plt.gca().add_patch(road_rect)

        # Draw lane markings
        for i in range(num_lanes + 1):
            lane_y = road_y_bottom + i * lane_width
            if i == 0 or i == num_lanes:  # Solid lines for road edges
                plt.axhline(lane_y, color='white', linestyle='-', linewidth=2, alpha=0.7, zorder=2)
            else:  # Dashed lines for lane dividers
                plt.axhline(lane_y, color='white', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

        # Add shoulder areas
        shoulder_width = 2  # meters
        shoulder_top_rect = plt.Rectangle((min_x - padding, road_y_bottom + road_width),
                                          max_x - min_x + 2 * padding, shoulder_width,
                                          facecolor='green', alpha=0.2, zorder=1)
        shoulder_bottom_rect = plt.Rectangle((min_x - padding, road_y_bottom - shoulder_width),
                                             max_x - min_x + 2 * padding, shoulder_width,
                                             facecolor='green', alpha=0.2, zorder=1)
        plt.gca().add_patch(shoulder_top_rect)
        plt.gca().add_patch(shoulder_bottom_rect)

        # Save the figure
        output_path = os.path.join(self.output_dir, f"{scenario_name}_static.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Static visualization saved to {output_path}")
        return output_path