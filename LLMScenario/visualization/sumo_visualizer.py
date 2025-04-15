"""
SUMO visualization tools for scenario simulation.
"""

import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


class SUMOVisualizer:
    def __init__(self, output_dir='output/videos'):
        """
        Initialize the SUMO visualizer.

        Args:
            output_dir (str): Directory to save output videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_sumo_gui(self, config_file):
        """
        Run SUMO GUI with the given configuration.

        Args:
            config_file (str): Path to SUMO config file
        """
        try:
            # Command to run SUMO GUI
            command = ["sumo-gui", "-c", config_file]

            # Run SUMO GUI
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running SUMO GUI: {e}")
        except FileNotFoundError:
            print("SUMO GUI not found. Make sure SUMO is installed and in your PATH.")

    def run_sumo_headless(self, config_file, scenario_name, num_frames=100, fps=10):
        """
        Run SUMO in headless mode and generate a video.

        Args:
            config_file (str): Path to SUMO config file
            scenario_name (str): Name of the scenario
            num_frames (int): Number of frames to capture
            fps (int): Frames per second for the output video

        Returns:
            str: Path to the output video
        """
        try:
            import traci

            # Start SUMO in headless mode
            traci.start(["sumo", "-c", config_file])

            # Create temporary directory for frames
            temp_dir = os.path.join(self.output_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)

            # Capture frames
            for i in range(num_frames):
                # Step the simulation
                if not traci.simulation.getMinExpectedNumber() > 0:
                    break

                traci.simulation.step()

                # Get vehicle positions
                vehicles = traci.vehicle.getIDList()

                # Create a plot
                plt.figure(figsize=(10, 6))

                # Plot all vehicles
                for veh_id in vehicles:
                    x, y = traci.vehicle.getPosition(veh_id)
                    plt.plot(x, y, 'ro', markersize=10)
                    plt.text(x, y + 5, veh_id, fontsize=8)

                # Set plot limits
                plt.xlim([0, 1000])
                plt.ylim([0, 50])
                plt.grid(True)
                plt.title(f"Scenario: {scenario_name} - Frame {i}")

                # Save the plot as an image
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                plt.savefig(frame_path)
                plt.close()

            # Close SUMO
            traci.close()

            # Create video from frames
            video_path = os.path.join(self.output_dir, f"{scenario_name}.mp4")
            self._create_video_from_frames(temp_dir, video_path, fps)

            # Clean up temporary frames
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

            return video_path

        except ImportError:
            print("TraCI not found. Make sure SUMO is installed correctly.")
            return None
        except Exception as e:
            print(f"Error running SUMO: {e}")
            return None

    def _create_video_from_frames(self, frames_dir, output_path, fps=10):
        """
        Create a video from a sequence of frames.

        Args:
            frames_dir (str): Directory containing frame images
            output_path (str): Path to save the output video
            fps (int): Frames per second for the output video
        """
        # Get all frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])

        if not frame_files:
            print("No frames found to create video")
            return

        # Get dimensions from the first frame
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width, _ = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Add frames to video
        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(frames_dir, frame_file))
            video.write(frame)

        # Release the video writer
        video.release()
        print(f"Video saved to {output_path}")