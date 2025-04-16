"""
Metrics for evaluating scenario generation quality.
"""

import re
import numpy as np
from collections import defaultdict


class ScenarioMetrics:
    """Metrics for evaluating scenario generation quality."""

    @staticmethod
    def count_vehicles(scenario_text):
        """
        Count the number of vehicles in a scenario.

        Args:
            scenario_text (str): The scenario text

        Returns:
            int: Number of vehicles
        """
        # Use regex to find vehicle IDs
        vehicle_pattern = r"Vehicle ID: (\d+)"
        vehicle_matches = re.findall(vehicle_pattern, scenario_text)

        # Count unique vehicle IDs
        unique_vehicles = set(vehicle_matches)
        return len(unique_vehicles)

    @staticmethod
    def count_interactions(scenario_text):
        """
        Count the number of interactions in a scenario.

        Args:
            scenario_text (str): The scenario text

        Returns:
            int: Number of interactions
        """
        # Keywords indicating interactions
        interaction_keywords = [
            "lane change", "overtake", "cut in", "merge", "follow",
            "brake", "accelerate", "slow down", "speed up",
            "collision", "near miss", "close call", "emergency",
            "yield", "give way", "interact"
        ]

        count = 0
        for keyword in interaction_keywords:
            count += len(re.findall(r"\b" + keyword + r"\w*\b", scenario_text.lower()))

        return count

    @staticmethod
    def count_lane_changes(scenario_text):
        """
        Count the number of lane changes in a scenario.

        Args:
            scenario_text (str): The scenario text

        Returns:
            int: Number of lane changes
        """
        # Keywords indicating lane changes
        lane_change_keywords = [
            "lane change", "change lane", "change to lane",
            "shift to lane", "move to lane", "cut in", "merge"
        ]

        count = 0
        for keyword in lane_change_keywords:
            count += len(re.findall(r"\b" + keyword + r"\w*\b", scenario_text.lower()))

        return count

    @staticmethod
    def parse_trajectories(scenario_text):
        """
        Parse trajectory data from scenario text.

        Args:
            scenario_text (str): The scenario text

        Returns:
            dict: Dictionary of vehicle trajectories
        """
        # Extract trajectory data using regex
        vehicle_pattern = r"Vehicle ID: (\d+).*?\n(Time \(s\), X \(m\), Y \(m\), Velocity \(m/s\), Heading \(rad\)\n(?:\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\n)+)"
        vehicle_matches = re.findall(vehicle_pattern, scenario_text, re.DOTALL)

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

    @staticmethod
    def trajectory_completeness(scenario_text):
        """
        Calculate the completeness of trajectory data.

        Args:
            scenario_text (str): The scenario text

        Returns:
            float: Completeness score (0-1)
        """
        # Parse trajectories
        trajectories = ScenarioMetrics.parse_trajectories(scenario_text)

        if not trajectories:
            return 0.0

        # Calculate average number of time steps per vehicle
        total_steps = sum(len(trajectory) for trajectory in trajectories.values())
        avg_steps = total_steps / len(trajectories)

        # Normalize to a 0-1 scale (assuming 20 steps is complete)
        completeness = min(avg_steps / 20, 1.0)

        return completeness

    @staticmethod
    def calculate_acceleration(trajectory):
        """
        Calculate acceleration values from a vehicle trajectory.

        Args:
            trajectory (list): List of trajectory points

        Returns:
            list: List of acceleration values
        """
        accelerations = []

        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]

            time_diff = curr['time'] - prev['time']
            if time_diff > 0:
                velocity_diff = curr['velocity'] - prev['velocity']
                acceleration = velocity_diff / time_diff
                accelerations.append(acceleration)

        return accelerations

    @staticmethod
    def physical_realism(scenario_text, max_accel=5.0, max_decel=10.0):
        """
        Evaluate physical realism of trajectory data.

        Args:
            scenario_text (str): The scenario text
            max_accel (float): Maximum realistic acceleration (m/s^2)
            max_decel (float): Maximum realistic deceleration (m/s^2)

        Returns:
            float: Realism score (0-1)
        """
        # Parse trajectories
        trajectories = ScenarioMetrics.parse_trajectories(scenario_text)

        if not trajectories:
            return 0.0

        # Check acceleration/deceleration for realism
        unrealistic_count = 0
        total_accel_points = 0

        for vehicle_id, trajectory in trajectories.items():
            accelerations = ScenarioMetrics.calculate_acceleration(trajectory)

            for accel in accelerations:
                total_accel_points += 1
                if accel > max_accel or accel < -max_decel:
                    unrealistic_count += 1

        if total_accel_points == 0:
            return 0.0

        # Calculate realism score (higher is better)
        realism_score = 1.0 - (unrealistic_count / total_accel_points)

        return realism_score

    @staticmethod
    def scenario_complexity(scenario_text):
        """
        Calculate a complexity score for the scenario.

        Args:
            scenario_text (str): The scenario text

        Returns:
            float: Complexity score
        """
        # Component scores
        vehicle_count = ScenarioMetrics.count_vehicles(scenario_text)
        interaction_count = ScenarioMetrics.count_interactions(scenario_text)
        lane_change_count = ScenarioMetrics.count_lane_changes(scenario_text)

        # Base complexity score
        complexity = (
                0.3 * min(vehicle_count / 10, 1.0) +  # Normalize to max of 10 vehicles
                0.4 * min(interaction_count / 20, 1.0) +  # Normalize to max of 20 interactions
                0.3 * min(lane_change_count / 10, 1.0)  # Normalize to max of 10 lane changes
        )

        return complexity

    @staticmethod
    def calculate_llmscenario_metrics(scenario_text):
        """
        Calculate metrics comparable to those in the original LLMScenario paper.

        Args:
            scenario_text (str): The scenario text

        Returns:
            dict: LLMScenario compatible metrics
        """
        # Parse trajectories
        trajectories = ScenarioMetrics.parse_trajectories(scenario_text)

        # Original LLMScenario metrics:
        # 1. Reality (checking for collisions, unrealistic dynamics)
        # 2. Rarity (distance from normal scenarios)

        # 1. Reality Score (based on physical feasibility)
        reality_score = ScenarioMetrics.physical_realism(scenario_text)

        # Check for collisions (simplified version)
        collisions_detected = 0

        # Calculate time steps
        time_steps = set()
        for vehicle_id, traj in trajectories.items():
            for point in traj:
                time_steps.add(point['time'])
        time_steps = sorted(list(time_steps))

        # For each time step, check if any two vehicles are too close
        for time_step in time_steps:
            vehicle_positions = {}

            for vehicle_id, traj in trajectories.items():
                # Find position at this time step
                positions = [point for point in traj if point['time'] == time_step]
                if positions:
                    vehicle_positions[vehicle_id] = (positions[0]['x'], positions[0]['y'])

            # Check distances between all vehicle pairs
            vehicle_ids = list(vehicle_positions.keys())
            for i in range(len(vehicle_ids)):
                for j in range(i + 1, len(vehicle_ids)):
                    id1, id2 = vehicle_ids[i], vehicle_ids[j]
                    pos1, pos2 = vehicle_positions[id1], vehicle_positions[id2]

                    # Calculate distance
                    distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

                    # If vehicles are too close (less than 2 meters), count as potential collision
                    if distance < 2.0:
                        collisions_detected += 1

        # Adjust reality score based on collisions
        if collisions_detected > 0:
            reality_score *= (1.0 / (1.0 + collisions_detected))

        # 2. Rarity Score (based on complexity and uniqueness)
        # Use scenario complexity as a proxy for rarity
        rarity_score = ScenarioMetrics.scenario_complexity(scenario_text)

        # Combine for final score similar to the LLMScenario paper approach
        final_score = 0
        if reality_score > 0:  # Only valid scenarios get rarity points
            final_score = reality_score + rarity_score

        return {
            'reality_score': reality_score,
            'rarity_score': rarity_score,
            'collisions_detected': collisions_detected,
            'final_score': final_score
        }


