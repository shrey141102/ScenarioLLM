"""
Utility functions for processing HighD data.
"""

import pandas as pd
import numpy as np


class DataProcessor:
    @staticmethod
    def extract_scenario_from_tracks(tracks_df, recording_meta=None, track_meta=None, scenario_id=None):
        """
        Extract scenario information from tracks data.

        Args:
            tracks_df (pd.DataFrame): DataFrame containing track data
            recording_meta (pd.DataFrame, optional): Recording metadata
            track_meta (pd.DataFrame, optional): Track metadata
            scenario_id (int, optional): Specific scenario ID to extract

        Returns:
            tuple: (road_env, vehicle_states, tasks_interactions)
        """
        # If scenario_id is specified, filter data for that scenario
        if scenario_id is not None:
            # Implement filtering logic based on your data structure
            pass

        # Extract road environment information
        road_env = DataProcessor._extract_road_environment(tracks_df, recording_meta)

        # Extract vehicle states and trajectories
        vehicle_states = DataProcessor._extract_vehicle_states(tracks_df, track_meta)

        # Extract tasks and interactions
        tasks_interactions = DataProcessor._extract_tasks_interactions(tracks_df, track_meta)

        return road_env, vehicle_states, tasks_interactions

    @staticmethod
    def _extract_road_environment(tracks_df, recording_meta):
        """Extract road environment description."""
        # Get unique lane IDs
        lanes = tracks_df['laneId'].unique()

        # If recording metadata is available, include more details
        speed_limit = "-1.00 km/h"  # Default value
        if recording_meta is not None and not recording_meta.empty:
            speed_limit = f"{recording_meta.iloc[0]['speedLimit']} km/h"

        road_env = f"""This scenario takes place on a highway with {len(lanes)} lanes. 
The speed limit is {speed_limit}.
Lane IDs present: {', '.join(map(str, lanes))}.
"""
        return road_env

    @staticmethod
    def _extract_vehicle_states(tracks_df, track_meta):
        """Extract vehicle states and trajectories."""
        # Get unique vehicle IDs
        vehicle_ids = tracks_df['id'].unique()

        vehicle_descriptions = []
        for vehicle_id in vehicle_ids[:5]:  # Limit to first 5 vehicles for brevity
            vehicle_data = tracks_df[tracks_df['id'] == vehicle_id]

            # Get vehicle class if available
            vehicle_class = "Car"  # Default
            if track_meta is not None and not track_meta.empty:
                v_meta = track_meta[track_meta['id'] == vehicle_id]
                if not v_meta.empty:
                    vehicle_class = v_meta.iloc[0]['class']

            # Get initial and final positions
            initial_frame = vehicle_data.iloc[0]
            final_frame = vehicle_data.iloc[-1]

            # Calculate average speed
            avg_speed = vehicle_data['xVelocity'].mean()

            description = f"""Vehicle {vehicle_id} ({vehicle_class}):
- Initial position: ({initial_frame['x']:.2f}, {initial_frame['y']:.2f})
- Final position: ({final_frame['x']:.2f}, {final_frame['y']:.2f})
- Average speed: {avg_speed:.2f} m/s
- Lanes used: {', '.join(map(str, vehicle_data['laneId'].unique()))}
"""
            vehicle_descriptions.append(description)

        return "\n".join(vehicle_descriptions)

    @staticmethod
    def _extract_tasks_interactions(tracks_df, track_meta):
        """Extract tasks and interactions between vehicles."""
        # Identify lane changes
        lane_changes = []
        vehicle_ids = tracks_df['id'].unique()

        for vehicle_id in vehicle_ids:
            vehicle_data = tracks_df[tracks_df['id'] == vehicle_id]
            lane_ids = vehicle_data['laneId'].values

            # Check for lane changes
            for i in range(1, len(lane_ids)):
                if lane_ids[i] != lane_ids[i - 1]:
                    lane_changes.append(
                        f"Vehicle {vehicle_id} changes from lane {lane_ids[i - 1]} to lane {lane_ids[i]}")

        # Identify following interactions
        following_interactions = []
        for vehicle_id in vehicle_ids:
            vehicle_data = tracks_df[tracks_df['id'] == vehicle_id]

            # Check preceding vehicle IDs
            preceding_ids = vehicle_data['precedingId'].unique()
            for preceding_id in preceding_ids:
                if preceding_id > 0:  # Valid preceding vehicle
                    following_interactions.append(f"Vehicle {vehicle_id} follows vehicle {preceding_id}")

        # Combine all interactions
        all_interactions = lane_changes + following_interactions

        # Limit to first 10 interactions for brevity
        interaction_text = "\n- ".join(all_interactions[:10])
        if interaction_text:
            interaction_text = "- " + interaction_text

        tasks_interactions = f"""Identified interactions:
{interaction_text}
"""
        return tasks_interactions