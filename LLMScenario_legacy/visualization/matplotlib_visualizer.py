import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle, Wedge
import matplotlib.transforms as transforms
from matplotlib.path import Path
import matplotlib.patches as patches


class MatplotlibVisualizer:
    def __init__(self, output_dir='output/videos'):
        """
        Initialize the Matplotlib visualizer.

        Args:
            output_dir (str): Directory to save output videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define vehicle colors by type
        self.vehicle_types = {
            'car': '#3498db',     # Blue
            'truck': '#e74c3c',   # Red
            'bus': '#f39c12',     # Orange
            'motorcycle': '#9b59b6'  # Purple
        }

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

    def _create_car_shape(self, length=5, width=2, car_type='car'):
        """Create a more realistic car shape using Path and Patches."""
        # Determine vehicle appearance based on type
        if car_type == 'truck':
            return self._create_truck_shape(length, width)
        elif car_type == 'bus':
            return self._create_bus_shape(length, width)
        elif car_type == 'motorcycle':
            return self._create_motorcycle_shape(length, width)

        # Default car shape
        # Define car body outline
        car_color = self.vehicle_types.get(car_type, '#3498db')

        # Calculate car dimensions and features
        body_length = length * 0.75  # Main body is 75% of total length
        hood_length = length * 0.25  # Hood is 25% of total length

        # Create the car body path
        verts = [
            (0, -width/2),                     # Rear left corner
            (0, width/2),                      # Rear right corner
            (body_length, width/2),            # Front right corner of main body
            (body_length, width/2 - width*0.1),  # Start of hood, right side
            (length, 0),                       # Front center point
            (body_length, -width/2 + width*0.1), # Start of hood, left side
            (body_length, -width/2),           # Front left corner of main body
            (0, -width/2)                      # Back to rear left to close the path
        ]

        codes = [Path.MOVETO] + [Path.LINETO] * 6 + [Path.CLOSEPOLY]
        car_path = Path(verts, codes)
        car_patch = patches.PathPatch(car_path, facecolor=car_color, edgecolor='black', lw=1)

        # Add windows
        window_patch = Rectangle((body_length*0.3, -width/2 + width*0.15), body_length*0.5, width*0.7,
                                 facecolor='lightblue', edgecolor='black', alpha=0.7)

        # Add wheels
        wheel_radius = width * 0.2
        front_wheel_x = body_length * 0.8
        rear_wheel_x = body_length * 0.2
        wheel_y = width/2 + wheel_radius*0.3

        front_wheel_left = Circle((front_wheel_x, -wheel_y), wheel_radius, facecolor='black')
        front_wheel_right = Circle((front_wheel_x, wheel_y), wheel_radius, facecolor='black')
        rear_wheel_left = Circle((rear_wheel_x, -wheel_y), wheel_radius, facecolor='black')
        rear_wheel_right = Circle((rear_wheel_x, wheel_y), wheel_radius, facecolor='black')

        # Add headlights and taillights
        headlight_left = Circle((length-0.1, -width/4), width*0.1, facecolor='yellow', alpha=0.9)
        headlight_right = Circle((length-0.1, width/4), width*0.1, facecolor='yellow', alpha=0.9)

        taillight_left = Rectangle((0.1, -width/2 + 0.1), width*0.1, width*0.1, facecolor='red', alpha=0.9)
        taillight_right = Rectangle((0.1, width/2 - 0.2), width*0.1, width*0.1, facecolor='red', alpha=0.9)

        return [car_patch, window_patch,
                front_wheel_left, front_wheel_right, rear_wheel_left, rear_wheel_right,
                headlight_left, headlight_right, taillight_left, taillight_right]

    def _create_truck_shape(self, length=7, width=2.5):
        """Create a truck shape."""
        # Define colors
        truck_color = self.vehicle_types['truck']

        # Calculate truck dimensions
        cabin_length = length * 0.3
        cargo_length = length * 0.7

        # Create the truck cabin
        cabin = Rectangle((cargo_length, -width/2), cabin_length, width,
                         facecolor=truck_color, edgecolor='black')

        # Create the cargo area
        cargo = Rectangle((0, -width/2), cargo_length, width,
                         facecolor=truck_color, edgecolor='black', alpha=0.9)

        # Add windows
        window = Rectangle((cargo_length + 0.1, -width/3), cabin_length * 0.6, width*0.6,
                          facecolor='lightblue', edgecolor='black', alpha=0.7)

        # Add wheels (trucks have more wheels)
        wheel_radius = width * 0.2
        wheel_y = width/2 + wheel_radius*0.3

        wheels = []
        # Front wheels
        wheels.append(Circle((cargo_length + cabin_length*0.7, -wheel_y), wheel_radius, facecolor='black'))
        wheels.append(Circle((cargo_length + cabin_length*0.7, wheel_y), wheel_radius, facecolor='black'))

        # Rear wheels (two pairs)
        wheels.append(Circle((cargo_length*0.3, -wheel_y), wheel_radius, facecolor='black'))
        wheels.append(Circle((cargo_length*0.3, wheel_y), wheel_radius, facecolor='black'))
        wheels.append(Circle((cargo_length*0.6, -wheel_y), wheel_radius, facecolor='black'))
        wheels.append(Circle((cargo_length*0.6, wheel_y), wheel_radius, facecolor='black'))

        # Add headlights and taillights
        headlight_left = Circle((length-0.1, -width/3), width*0.1, facecolor='yellow', alpha=0.9)
        headlight_right = Circle((length-0.1, width/3), width*0.1, facecolor='yellow', alpha=0.9)

        taillight_left = Rectangle((0.1, -width/2 + 0.1), width*0.1, width*0.1, facecolor='red', alpha=0.9)
        taillight_right = Rectangle((0.1, width/2 - 0.2), width*0.1, width*0.1, facecolor='red', alpha=0.9)

        return [cabin, cargo, window] + wheels + [headlight_left, headlight_right, taillight_left, taillight_right]

    def _create_bus_shape(self, length=12, width=2.5):
        """Create a bus shape."""
        # Define colors
        bus_color = self.vehicle_types['bus']

        # Create the bus body
        body = Rectangle((0, -width/2), length, width,
                         facecolor=bus_color, edgecolor='black')

        # Add windows (multiple windows along the side)
        windows = []
        window_width = length * 0.1
        window_height = width * 0.4
        window_y = -width/2 + width*0.2

        for i in range(1, 9):
            x_pos = i * window_width
            if x_pos < length - 2*window_width:  # Avoid windows too close to the front
                window = Rectangle((x_pos, window_y), window_width*0.8, window_height,
                                   facecolor='lightblue', edgecolor='black', alpha=0.7)
                windows.append(window)

        # Windshield (larger front window)
        windshield = Rectangle((length-2*window_width, -width/3), window_width*1.5, width*0.6,
                               facecolor='lightblue', edgecolor='black', alpha=0.7)

        # Add wheels
        wheel_radius = width * 0.2
        wheel_y = width/2 + wheel_radius*0.3

        wheels = []
        # Front wheels
        wheels.append(Circle((length*0.8, -wheel_y), wheel_radius, facecolor='black'))
        wheels.append(Circle((length*0.8, wheel_y), wheel_radius, facecolor='black'))

        # Rear wheels
        wheels.append(Circle((length*0.2, -wheel_y), wheel_radius, facecolor='black'))
        wheels.append(Circle((length*0.2, wheel_y), wheel_radius, facecolor='black'))

        # Add headlights and taillights
        headlight_left = Circle((length-0.1, -width/3), width*0.1, facecolor='yellow', alpha=0.9)
        headlight_right = Circle((length-0.1, width/3), width*0.1, facecolor='yellow', alpha=0.9)

        taillight_left = Rectangle((0.1, -width/2 + 0.1), width*0.1, width*0.1, facecolor='red', alpha=0.9)
        taillight_right = Rectangle((0.1, width/2 - 0.2), width*0.1, width*0.1, facecolor='red', alpha=0.9)

        return [body, windshield] + windows + wheels + [headlight_left, headlight_right, taillight_left, taillight_right]

    def _create_motorcycle_shape(self, length=2, width=1):
        """Create a motorcycle shape."""
        # Define colors
        bike_color = self.vehicle_types['motorcycle']

        # Create the bike body (simplified)
        body = Polygon([
            (0, 0),                # Rear
            (length*0.4, width*0.1),  # Seat
            (length*0.7, width*0.3),  # Handlebars
            (length, 0),           # Front
            (length*0.7, -width*0.3), # Lower front
            (length*0.4, -width*0.1)  # Lower body
        ], facecolor=bike_color, edgecolor='black')

        # Add wheels
        wheel_radius = width * 0.4
        front_wheel = Circle((length*0.8, 0), wheel_radius, facecolor='black', fill=True, alpha=0.7)
        rear_wheel = Circle((length*0.2, 0), wheel_radius, facecolor='black', fill=True, alpha=0.7)

        # Add rider (simplified)
        head_radius = width * 0.15
        head = Circle((length*0.5, width*0.5), head_radius, facecolor='tan')
        body_upper = Polygon([
            (length*0.4, width*0.2),   # Bottom
            (length*0.4, width*0.4),   # Shoulder level
            (length*0.5, width*0.4),   # Shoulder width
            (length*0.5, width*0.2)    # Bottom width
        ], facecolor='darkblue', edgecolor='black')

        # Add headlight
        headlight = Circle((length-0.1, 0), width*0.1, facecolor='yellow', alpha=0.9)

        # Add taillight
        taillight = Circle((0.1, 0), width*0.08, facecolor='red', alpha=0.9)

        return [body, front_wheel, rear_wheel, head, body_upper, headlight, taillight]

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

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Add a legend for vehicle types
        legend_patches = []
        for vehicle_type, color in self.vehicle_types.items():
            legend_patches.append(patches.Patch(color=color, label=vehicle_type.capitalize()))
        ax.legend(handles=legend_patches, loc='upper right')

        # Assign vehicle types based on their ID or other characteristics
        vehicle_types = {}
        for i, vehicle_id in enumerate(trajectories.keys()):
            # Assign types based on some logic or randomly for demonstration
            vtype = list(self.vehicle_types.keys())[i % len(self.vehicle_types.keys())]
            vehicle_types[vehicle_id] = vtype

        # List to store vehicle components and labels
        vehicles = {}
        vehicle_components = {}
        vehicle_labels = {}

        # Initialize vehicles
        for vehicle_id, trajectory in trajectories.items():
            # Determine vehicle type and size
            vtype = vehicle_types[vehicle_id]

            # Set vehicle size based on type
            if vtype == 'car':
                vehicle_size = (5, 2)
            elif vtype == 'truck':
                vehicle_size = (7, 2.5)
            elif vtype == 'bus':
                vehicle_size = (12, 2.5)
            elif vtype == 'motorcycle':
                vehicle_size = (2, 1)
            else:
                vehicle_size = (5, 2)  # Default

            # Create vehicle shape components
            components = self._create_car_shape(vehicle_size[0], vehicle_size[1], vtype)

            # Add all components to the plot and store references
            vehicle_components[vehicle_id] = []
            for component in components:
                ax.add_patch(component)
                vehicle_components[vehicle_id].append(component)

            # Create a text label for the vehicle
            label = ax.text(0, 0, f"{vtype} {vehicle_id}", fontsize=8,
                            ha="center", va="center", color="white",
                            bbox=dict(facecolor='black', alpha=0.5))
            vehicle_labels[vehicle_id] = label

        # Set the axis limits
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # Add title and labels
        ax.set_title(f"Scenario: {scenario_name}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Draw lanes (simplified)
        lane_y_values = np.linspace(min_y + 5, max_y - 5, 4)
        for y in lane_y_values:
            lane = ax.axhline(y, color='k', linestyle='--', alpha=0.3, linewidth=2)
            # Add lane markings
            for x in np.arange(min_x, max_x, 20):
                ax.plot([x, x+10], [y, y], 'w-', linewidth=1, alpha=0.7)

        # Add road edges
        ax.axhline(min_y + 2, color='white', linewidth=3)
        ax.axhline(max_y - 2, color='white', linewidth=3)

        # Add background color for road
        road_background = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                   facecolor='#444444', alpha=0.3, zorder=-1)
        ax.add_patch(road_background)

        # Function to interpolate vehicle position at a given time
        def interpolate_position(trajectory, time):
            # Find the two closest time points
            times = [point['time'] for point in trajectory]
            if time <= times[0]:
                return trajectory[0]['x'], trajectory[0]['y'], trajectory[0]['heading'], trajectory[0]['velocity']
            if time >= times[-1]:
                return trajectory[-1]['x'], trajectory[-1]['y'], trajectory[-1]['heading'], trajectory[-1]['velocity']

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
            v1, v2 = trajectory[prev_idx]['velocity'], trajectory[next_idx]['velocity']

            x = x1 + factor * (x2 - x1)
            y = y1 + factor * (y2 - y1)
            heading = h1 + factor * (h2 - h1)
            velocity = v1 + factor * (v2 - v1)

            return x, y, heading, velocity

        # Animation update function
        def update(frame):
            # Calculate current time
            time = min_time + frame * (max_time - min_time) / (fps * duration)

            # Update title with current time
            ax.set_title(f"Scenario: {scenario_name} - Time: {time:.2f}s")

            # List to collect all artists that need to be updated
            artists = []

            # Update each vehicle's position
            for vehicle_id, trajectory in trajectories.items():
                if time < trajectory[0]['time'] or time > trajectory[-1]['time']:
                    # Vehicle not yet on scene or has left
                    for component in vehicle_components[vehicle_id]:
                        component.set_visible(False)
                    vehicle_labels[vehicle_id].set_visible(False)
                    continue

                # Make vehicle visible
                for component in vehicle_components[vehicle_id]:
                    component.set_visible(True)
                vehicle_labels[vehicle_id].set_visible(True)

                # Get interpolated position
                x, y, heading, velocity = interpolate_position(trajectory, time)

                # Determine vehicle type and size
                vtype = vehicle_types[vehicle_id]

                # Get vehicle size
                if vtype == 'car':
                    vehicle_size = (5, 2)
                elif vtype == 'truck':
                    vehicle_size = (7, 2.5)
                elif vtype == 'bus':
                    vehicle_size = (12, 2.5)
                elif vtype == 'motorcycle':
                    vehicle_size = (2, 1)
                else:
                    vehicle_size = (5, 2)  # Default

                # Update vehicle position and rotation for each component
                for component in vehicle_components[vehicle_id]:
                    # Create a transform that rotates and then translates
                    transform = transforms.Affine2D().rotate(heading).translate(x, y)
                    component.set_transform(transform + ax.transData)
                    artists.append(component)

                # Update label position
                vehicle_labels[vehicle_id].set_position((x, y - vehicle_size[1]))
                vehicle_labels[vehicle_id].set_text(f"{vtype} {vehicle_id} ({velocity:.1f} m/s)")
                artists.append(vehicle_labels[vehicle_id])

                # Add velocity indicator (arrow showing direction and speed)
                if velocity > 0.5:  # Only show for moving vehicles
                    arrow_length = min(velocity * 0.5, 10)  # Scale but cap the length
                    dx = arrow_length * np.cos(heading)
                    dy = arrow_length * np.sin(heading)

                    # Create a new arrow each frame to show direction of movement
                    arrow = ax.arrow(x, y, dx, dy, head_width=0.5, head_length=1,
                                    fc='white', ec='black', alpha=0.7)
                    artists.append(arrow)

            return artists

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

        # Set the figure background color to represent a road
        plt.gca().set_facecolor('#444444')

        # Define colors for different vehicles
        colors = plt.colormaps['jet'](np.linspace(0, 1, len(trajectories)))

        # Track min/max coordinates for setting axis limits
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        # Assign vehicle types based on their ID for variety
        vehicle_types = {}
        for i, vehicle_id in enumerate(trajectories.keys()):
            vtype = list(self.vehicle_types.keys())[i % len(self.vehicle_types.keys())]
            vehicle_types[vehicle_id] = vtype

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

            # Determine vehicle type and color
            vtype = vehicle_types[vehicle_id]
            vcolor = self.vehicle_types[vtype]

            # Plot the trajectory with dynamic transparency
            # Start more transparent and end more opaque
            alphas = np.linspace(0.2, 0.8, len(x_coords))
            for j in range(len(x_coords)-1):
                plt.plot(x_coords[j:j+2], y_coords[j:j+2], '-',
                        color=vcolor, alpha=alphas[j], linewidth=2)

            # Draw the vehicle at start and end positions
            vehicle_size = (5, 2) if vtype == 'car' else (7, 2.5) if vtype == 'truck' else (12, 2.5) if vtype == 'bus' else (2, 1)

            # Draw vehicle at start (with lighter opacity)
            start_heading = trajectory[0]['heading']
            start_components = self._create_car_shape(vehicle_size[0], vehicle_size[1], vtype)
            transform = transforms.Affine2D().rotate(start_heading).translate(x_coords[0], y_coords[0])
            for component in start_components:
                component.set_alpha(0.5)  # Semi-transparent for start position
                component.set_transform(transform + plt.gca().transData)
                plt.gca().add_patch(component)

            # Draw vehicle at end (with full opacity)
            end_heading = trajectory[-1]['heading']
            end_components = self._create_car_shape(vehicle_size[0], vehicle_size[1], vtype)
            transform = transforms.Affine2D().rotate(end_heading).translate(x_coords[-1], y_coords[-1])
            for component in end_components:
                component.set_transform(transform + plt.gca().transData)
                plt.gca().add_patch(component)

            # Add vehicle labels at start and end
            plt.text(x_coords[0], y_coords[0] - vehicle_size[1],
                    f"Start: {vtype} {vehicle_id}", fontsize=8,
                    ha="center", va="center", color="white",
                    bbox=dict(facecolor='black', alpha=0.5))

            plt.text(x_coords[-1], y_coords[-1] - vehicle_size[1],
                    f"End: {vtype} {vehicle_id}", fontsize=8,
                    ha="center", va="center", color="white",
                    bbox=dict(facecolor='black', alpha=0.5))

            # Add arrows to show direction at various points
            arrow_indices = np.linspace(0, len(x_coords)-1, min(5, len(x_coords))).astype(int)
            for idx in arrow_indices:
                if idx+1 < len(x_coords):
                    dx = x_coords[idx+1] - x_coords[idx]
                    dy = y_coords[idx+1] - y_coords[idx]
                    if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only draw if there's enough movement
                        arrow_len = np.sqrt(dx**2 + dy**2)
                        plt.arrow(x_coords[idx], y_coords[idx],
                                dx*0.8, dy*0.8,  # Slightly shorter than actual movement
                                head_width=arrow_len*0.1, head_length=arrow_len*0.2,
                                fc=vcolor, ec='black', alpha=0.6)

        # Add some padding to the axis limits
        padding = 50
        plt.xlim(min_x - padding, max_x + padding)
        plt.ylim(min_y - padding, max_y + padding)

        # Add title and labels
        plt.title(f"Scenario: {scenario_name}", fontsize=16)
        plt.xlabel("X (m)", fontsize=12)
        plt.ylabel("Y (m)", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Create legend for vehicle types
        legend_patches = []
        for vtype, color in self.vehicle_types.items():
            legend_patches.append(patches.Patch(color=color, label=vtype.capitalize()))
        plt.legend(handles=legend_patches, loc='upper right')

        # Draw lane markers (assuming lanes are roughly horizontal)
        lane_y_values = [9.39, 13.5, 21.68, 25.0]  # Approximate y-values for lanes 2, 3, 5, 6
        for y in lane_y_values:
            plt.axhline(y, color='white', linestyle='--', alpha=0.5, linewidth=2)
            # Add lane markings
            for x in np.arange(min_x, max_x, 20):
                plt.plot([x, x+10], [y, y], 'w-', linewidth=1, alpha=0.7)

        # Add road edges
        plt.axhline(min(lane_y_values) - 5, color='white', linewidth=3)
        plt.axhline(max(lane_y_values) + 5, color='white', linewidth=3)

        # Set background color for road
        ax = plt.gca()
        ax.set_facecolor('#444444')  # Dark gray for road surface

        # Add time indicators
        if all_times:
            time_range = f"Time Range: {min(all_times):.1f}s - {max(all_times):.1f}s"
            plt.figtext(0.5, 0.01, time_range, ha='center', fontsize=10,
                      bbox=dict(facecolor='white', alpha=0.7))

        # Save the figure
        output_path = os.path.join(self.output_dir, f"{scenario_name}_static.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Static visualization saved to {output_path}")
        return output_path