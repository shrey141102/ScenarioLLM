# """
# Agent-based enhancements for the LLMScenario framework.
# """
#
# from .driver_agents import DriverAgentManager
# from .scenario_agents import ScenarioAgentManager
# from crewai import Crew
#
#
# class AgentManager:
#     """Main interface for working with agents."""
#
#     def __init__(self, openai_api_key=None):
#         """
#         Initialize the agent manager.
#
#         Args:
#             openai_api_key (str, optional): OpenAI API key
#         """
#         self.driver_manager = DriverAgentManager(openai_api_key)
#         self.scenario_manager = ScenarioAgentManager(openai_api_key)
#
#     def enhance_scenario(self, scenario_description, vehicle_states):
#         """
#         Enhance a scenario using the scenario agents.
#
#         Args:
#             scenario_description (str): Description of the scenario
#             vehicle_states (str): Description of vehicle states
#
#         Returns:
#             str: Enhanced scenario with trajectories
#         """
#         crew = self.scenario_manager.create_scenario_crew(scenario_description, vehicle_states)
#         result = crew.kickoff()
#         return result
#
#     # def populate_scenario_with_agents(self, scenario_description, num_aggressive=1, num_normal=2, num_cautious=1):
#     #     """
#     #     Populate a scenario with different types of driver agents.
#     #
#     #     Args:
#     #         scenario_description (str): Description of the scenario
#     #         num_aggressive (int): Number of aggressive drivers
#     #         num_normal (int): Number of normal drivers
#     #         num_cautious (int): Number of cautious drivers
#     #
#     #     Returns:
#     #         list: List of agent trajectories
#     #     """
#     #     trajectories = []
#     #
#     #     # Base positions and destinations
#     #     # These would typically be derived from the scenario
#     #     base_positions = [(0, 0), (50, 0), (100, 0), (150, 0)]
#     #     destinations = [(500, 0), (450, 0), (550, 0), (600, 0)]
#     #
#     #     # Create tasks for aggressive drivers
#     #     for i in range(num_aggressive):
#     #         task = self.driver_manager.create_driving_task(
#     #             self.driver_manager.aggressive_driver,
#     #             scenario_description,
#     #             f"A{i + 1}",
#     #             base_positions[i % len(base_positions)],
#     #             destinations[i % len(destinations)]
#     #         )
#     #         result = task.execute()
#     #         trajectories.append(result)
#     #
#     #     # Create tasks for normal drivers
#     #     for i in range(num_normal):
#     #         task = self.driver_manager.create_driving_task(
#     #             self.driver_manager.normal_driver,
#     #             scenario_description,
#     #             f"N{i + 1}",
#     #             base_positions[(i + num_aggressive) % len(base_positions)],
#     #             destinations[(i + num_aggressive) % len(destinations)]
#     #         )
#     #         result = task.execute()
#     #         trajectories.append(result)
#     #
#     #     # Create tasks for cautious drivers
#     #     for i in range(num_cautious):
#     #         task = self.driver_manager.create_driving_task(
#     #             self.driver_manager.cautious_driver,
#     #             scenario_description,
#     #             f"C{i + 1}",
#     #             base_positions[(i + num_aggressive + num_normal) % len(base_positions)],
#     #             destinations[(i + num_aggressive + num_normal) % len(destinations)]
#     #         )
#     #         result = task.execute()
#     #         trajectories.append(result)
#     #
#     #     return trajectories
#
#     def populate_scenario_with_agents(self, scenario_description, num_aggressive=1, num_normal=2, num_cautious=1):
#         """
#         Populate a scenario with different types of driver agents.
#
#         Args:
#             scenario_description (str): Description of the scenario
#             num_aggressive (int): Number of aggressive drivers
#             num_normal (int): Number of normal drivers
#             num_cautious (int): Number of cautious drivers
#
#         Returns:
#             list: List of agent trajectories
#         """
#         tasks = []
#         agents = []
#
#         # Base positions and destinations
#         base_positions = [(0, 0), (50, 0), (100, 0), (150, 0)]
#         destinations = [(500, 0), (450, 0), (550, 0), (600, 0)]
#
#         # Create tasks for aggressive drivers
#         for i in range(num_aggressive):
#             agents.append(self.driver_manager.aggressive_driver)
#             tasks.append(self.driver_manager.create_driving_task(
#                 self.driver_manager.aggressive_driver,
#                 scenario_description,
#                 f"A{i + 1}",
#                 base_positions[i % len(base_positions)],
#                 destinations[i % len(destinations)]
#             ))
#
#         # Create tasks for normal drivers
#         for i in range(num_normal):
#             agents.append(self.driver_manager.normal_driver)
#             tasks.append(self.driver_manager.create_driving_task(
#                 self.driver_manager.normal_driver,
#                 scenario_description,
#                 f"N{i + 1}",
#                 base_positions[(i + num_aggressive) % len(base_positions)],
#                 destinations[(i + num_aggressive) % len(destinations)]
#             ))
#
#         # Create tasks for cautious drivers
#         for i in range(num_cautious):
#             agents.append(self.driver_manager.cautious_driver)
#             tasks.append(self.driver_manager.create_driving_task(
#                 self.driver_manager.cautious_driver,
#                 scenario_description,
#                 f"C{i + 1}",
#                 base_positions[(i + num_aggressive + num_normal) % len(base_positions)],
#                 destinations[(i + num_aggressive + num_normal) % len(destinations)]
#             ))
#
#         # Create and run a crew with these tasks
#         crew = Crew(
#             agents=agents,
#             tasks=tasks,
#             verbose=True
#         )
#
#         # Kickoff the crew and get results
#         results = crew.kickoff()
#
#         # Extract results
#         trajectories = []
#         if isinstance(results, dict):
#             # If results is a dictionary of task outputs
#             for task_id, output in results.items():
#                 trajectories.append(output)
#         elif isinstance(results, list):
#             # If results is a list of task outputs
#             trajectories = results
#         else:
#             # If results is a single string output
#             trajectories = [results]
#
#         return trajectories
#
#     # def react_to_scenario_event(self, scenario_description, event_description):
#     #     """
#     #     Have driver agents react to an event in a scenario.
#     #
#     #     Args:
#     #         scenario_description (str): Description of the scenario
#     #         event_description (str): Description of the event
#     #
#     #     Returns:
#     #         list: List of agent reactions
#     #     """
#     #     reactions = []
#     #
#     #     # Example current positions and velocities
#     #     positions = [(100, 0), (150, 0), (200, 0)]
#     #     velocities = [30, 25, 20]
#     #
#     #     # Have the aggressive driver react
#     #     aggressive_reaction = self.driver_manager.react_to_event(
#     #         self.driver_manager.aggressive_driver,
#     #         scenario_description,
#     #         "A1",
#     #         positions[0],
#     #         velocities[0],
#     #         event_description
#     #     ).execute()
#     #     reactions.append(aggressive_reaction)
#     #
#     #     # Have the normal driver react
#     #     normal_reaction = self.driver_manager.react_to_event(
#     #         self.driver_manager.normal_driver,
#     #         scenario_description,
#     #         "N1",
#     #         positions[1],
#     #         velocities[1],
#     #         event_description
#     #     ).execute()
#     #     reactions.append(normal_reaction)
#     #
#     #     # Have the cautious driver react
#     #     cautious_reaction = self.driver_manager.react_to_event(
#     #         self.driver_manager.cautious_driver,
#     #         scenario_description,
#     #         "C1",
#     #         positions[2],
#     #         velocities[2],
#     #         event_description
#     #     ).execute()
#     #     reactions.append(cautious_reaction)
#     #
#     #     return reactions
#
#     def react_to_scenario_event(self, scenario_description, event_description):
#         """
#         Have driver agents react to an event in a scenario.
#
#         Args:
#             scenario_description (str): Description of the scenario
#             event_description (str): Description of the event
#
#         Returns:
#             list: List of agent reactions
#         """
#         reactions = []
#
#         # Example current positions and velocities
#         positions = [(100, 0), (150, 0), (200, 0)]
#         velocities = [30, 25, 20]
#
#         # Create the tasks
#         aggressive_task = self.driver_manager.react_to_event(
#             self.driver_manager.aggressive_driver,
#             scenario_description,
#             "A1",
#             positions[0],
#             velocities[0],
#             event_description
#         )
#
#         normal_task = self.driver_manager.react_to_event(
#             self.driver_manager.normal_driver,
#             scenario_description,
#             "N1",
#             positions[1],
#             velocities[1],
#             event_description
#         )
#
#         cautious_task = self.driver_manager.react_to_event(
#             self.driver_manager.cautious_driver,
#             scenario_description,
#             "C1",
#             positions[2],
#             velocities[2],
#             event_description
#         )
#
#         # Create and run a crew with these tasks
#         crew = Crew(
#             agents=[
#                 self.driver_manager.aggressive_driver,
#                 self.driver_manager.normal_driver,
#                 self.driver_manager.cautious_driver
#             ],
#             tasks=[aggressive_task, normal_task, cautious_task],
#             verbose=True  # Changed to boolean
#         )
#
#         # Kickoff the crew and get results
#         results = crew.kickoff()
#
#         # Extract results
#         if isinstance(results, dict):
#             # If results is a dictionary of task outputs
#             for task_id, output in results.items():
#                 reactions.append(output)
#         elif isinstance(results, list):
#             # If results is a list of task outputs
#             reactions = results
#         else:
#             # If results is a single string output
#             reactions = [results]
#
#         return reactions


"""
Agent-based enhancements for the LLMScenario framework with output parsing and validation.
"""

from .driver_agents import DriverAgentManager
from .scenario_agents import ScenarioAgentManager
from crewai import Crew
import re
import json


class AgentManager:
    """Main interface for working with agents with trajectory validation and parsing."""

    def __init__(self, openai_api_key=None):
        """
        Initialize the agent manager.

        Args:
            openai_api_key (str, optional): OpenAI API key
        """
        self.driver_manager = DriverAgentManager(openai_api_key)
        self.scenario_manager = ScenarioAgentManager(openai_api_key)

    def enhance_scenario(self, scenario_description, vehicle_states):
        """
        Enhance a scenario using the scenario agents.

        Args:
            scenario_description (str): Description of the scenario
            vehicle_states (str): Description of vehicle states

        Returns:
            dict: Enhanced scenario with structured trajectory data
        """
        crew = self.scenario_manager.create_scenario_crew(scenario_description, vehicle_states)
        raw_result = crew.kickoff()

        # Process the result to extract structured trajectory data
        structured_result = self._parse_scenario_output(raw_result)
        return structured_result

    def _parse_scenario_output(self, raw_output):
        """
        Parse the raw output from the scenario enhancement to extract structured data.

        Args:
            raw_output (str): Raw output from the crew

        Returns:
            dict: Structured data containing trajectories and validation
        """
        result = {
            "raw_output": raw_output,
            "trajectories": {},
            "validation": None,
            "analysis": None
        }

        # Extract trajectories using regex
        vehicle_pattern = r"Vehicle ID: ([A-Za-z0-9]+)\s+(Time\(s\), X\(m\), Y\(m\), Velocity\(m/s\), Heading\(rad\)[\s\S]+?)(?=Vehicle ID:|$)"
        vehicle_matches = re.finditer(vehicle_pattern, raw_output)

        for match in vehicle_matches:
            vehicle_id = match.group(1)
            trajectory_data = match.group(2).strip()

            # Parse trajectory data into structured format
            trajectory_lines = trajectory_data.split('\n')
            header = trajectory_lines[0].strip()  # Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)

            data_points = []
            for line in trajectory_lines[1:]:
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse comma-separated values
                    values = [float(val.strip()) for val in line.split(',')]
                    if len(values) == 5:  # Ensure we have all 5 expected values
                        data_point = {
                            "time": values[0],
                            "x": values[1],
                            "y": values[2],
                            "velocity": values[3],
                            "heading": values[4]
                        }
                        data_points.append(data_point)
                except (ValueError, IndexError) as e:
                    # Handle parsing errors
                    print(f"Error parsing trajectory data for vehicle {vehicle_id}: {e}")
                    print(f"Problematic line: {line}")

            if data_points:
                result["trajectories"][vehicle_id] = data_points

        # Extract validation results if present
        validation_pattern = r"\[VALIDATION RESULT: ([A-Z]+)\]([\s\S]+?)(?=\[|$)"
        validation_match = re.search(validation_pattern, raw_output)
        if validation_match:
            validation_status = validation_match.group(1)
            validation_details = validation_match.group(2).strip()
            result["validation"] = {
                "status": validation_status,
                "details": validation_details
            }

        # Extract analysis if present
        analysis_pattern = r"\[INTERACTION POINTS\]([\s\S]+?)(?=\[COMPLEXITY METRICS\])"
        analysis_match = re.search(analysis_pattern, raw_output)
        if analysis_match:
            result["analysis"] = {
                "interaction_points": analysis_match.group(1).strip()
            }

            # Extract complexity metrics if present
            complexity_pattern = r"\[COMPLEXITY METRICS\]([\s\S]+?)(?=\[SAFETY ANALYSIS\])"
            complexity_match = re.search(complexity_pattern, raw_output)
            if complexity_match:
                result["analysis"]["complexity_metrics"] = complexity_match.group(1).strip()

            # Extract safety analysis if present
            safety_pattern = r"\[SAFETY ANALYSIS\]([\s\S]+?)(?=\[CHALLENGE ASSESSMENT\]|$)"
            safety_match = re.search(safety_pattern, raw_output)
            if safety_match:
                result["analysis"]["safety_analysis"] = safety_match.group(1).strip()

            # Extract challenge assessment if present
            challenge_pattern = r"\[CHALLENGE ASSESSMENT\]([\s\S]+?)(?=\[|$)"
            challenge_match = re.search(challenge_pattern, raw_output)
            if challenge_match:
                result["analysis"]["challenge_assessment"] = challenge_match.group(1).strip()

        return result

    def populate_scenario_with_agents(self, scenario_description, num_aggressive=1, num_normal=2, num_cautious=1):
        """
        Populate a scenario with different types of driver agents.

        Args:
            scenario_description (str): Description of the scenario
            num_aggressive (int): Number of aggressive drivers
            num_normal (int): Number of normal drivers
            num_cautious (int): Number of cautious drivers

        Returns:
            dict: Dictionary of parsed agent trajectories
        """
        tasks = []
        agents = []

        # Base positions and destinations
        base_positions = [(0, 0), (50, 0), (100, 0), (150, 0)]
        destinations = [(500, 0), (450, 0), (550, 0), (600, 0)]

        # Create tasks for aggressive drivers
        for i in range(num_aggressive):
            agents.append(self.driver_manager.aggressive_driver)
            tasks.append(self.driver_manager.create_driving_task(
                self.driver_manager.aggressive_driver,
                scenario_description,
                f"A{i + 1}",
                base_positions[i % len(base_positions)],
                destinations[i % len(destinations)]
            ))

        # Create tasks for normal drivers
        for i in range(num_normal):
            agents.append(self.driver_manager.normal_driver)
            tasks.append(self.driver_manager.create_driving_task(
                self.driver_manager.normal_driver,
                scenario_description,
                f"N{i + 1}",
                base_positions[(i + num_aggressive) % len(base_positions)],
                destinations[(i + num_aggressive) % len(destinations)]
            ))

        # Create tasks for cautious drivers
        for i in range(num_cautious):
            agents.append(self.driver_manager.cautious_driver)
            tasks.append(self.driver_manager.create_driving_task(
                self.driver_manager.cautious_driver,
                scenario_description,
                f"C{i + 1}",
                base_positions[(i + num_aggressive + num_normal) % len(base_positions)],
                destinations[(i + num_aggressive + num_normal) % len(destinations)]
            ))

        # Create and run a crew with these tasks
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )

        # Kickoff the crew and get results
        raw_results = crew.kickoff()

        # Process the results into a structured format
        return self._process_trajectory_results(raw_results)

    def _process_trajectory_results(self, raw_results):
        """
        Process raw trajectory results into structured data.

        Args:
            raw_results: Raw results from crew execution

        Returns:
            dict: Dictionary of vehicle IDs mapping to trajectory data
        """
        structured_trajectories = {}

        # Handle different result formats
        if isinstance(raw_results, dict):
            # If results is a dictionary of task outputs
            results_list = list(raw_results.values())
        elif isinstance(raw_results, list):
            # If results is a list of task outputs
            results_list = raw_results
        else:
            # If results is a single string output
            results_list = [raw_results]

        # Process each result
        for result in results_list:
            if not isinstance(result, str):
                continue

            # Extract vehicle ID and trajectory data using regex
            vehicle_match = re.search(r"Vehicle ID: ([A-Za-z0-9]+)", result)
            if not vehicle_match:
                continue

            vehicle_id = vehicle_match.group(1)

            # Extract trajectory data
            trajectory_pattern = r"Time\(s\), X\(m\), Y\(m\), Velocity\(m/s\), Heading\(rad\)([\s\S]+?)(?=Vehicle ID:|$)"
            trajectory_match = re.search(trajectory_pattern, result)

            if not trajectory_match:
                continue

            trajectory_text = trajectory_match.group(1).strip()
            trajectory_data = []

            # Parse each line of trajectory data
            for line in trajectory_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse comma-separated values
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 5:
                        time, x, y, velocity, heading = [float(p) for p in parts]
                        trajectory_data.append({
                            "time": time,
                            "x": x,
                            "y": y,
                            "velocity": velocity,
                            "heading": heading
                        })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing trajectory data: {e}")
                    print(f"Problematic line: {line}")

            if trajectory_data:
                structured_trajectories[vehicle_id] = trajectory_data

        return structured_trajectories

    def react_to_scenario_event(self, scenario_description, event_description):
        """
        Have driver agents react to an event in a scenario.

        Args:
            scenario_description (str): Description of the scenario
            event_description (str): Description of the event

        Returns:
            dict: Dictionary of vehicle IDs mapping to reaction trajectory data
        """
        # Example current positions and velocities
        positions = [(100, 0), (150, 0), (200, 0)]
        velocities = [30, 25, 20]

        # Create the tasks
        aggressive_task = self.driver_manager.react_to_event(
            self.driver_manager.aggressive_driver,
            scenario_description,
            "A1",
            positions[0],
            velocities[0],
            event_description
        )

        normal_task = self.driver_manager.react_to_event(
            self.driver_manager.normal_driver,
            scenario_description,
            "N1",
            positions[1],
            velocities[1],
            event_description
        )

        cautious_task = self.driver_manager.react_to_event(
            self.driver_manager.cautious_driver,
            scenario_description,
            "C1",
            positions[2],
            velocities[2],
            event_description
        )

        # Create and run a crew with these tasks
        crew = Crew(
            agents=[
                self.driver_manager.aggressive_driver,
                self.driver_manager.normal_driver,
                self.driver_manager.cautious_driver
            ],
            tasks=[aggressive_task, normal_task, cautious_task],
            verbose=True
        )

        # Kickoff the crew and get results
        raw_results = crew.kickoff()

        # Process the results into a structured format
        return self._process_trajectory_results(raw_results)

    def export_trajectories_to_json(self, trajectories, output_file):
        """
        Export structured trajectories to a JSON file.

        Args:
            trajectories (dict): Structured trajectory data
            output_file (str): Path to output JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(trajectories, f, indent=2)

    def export_trajectories_to_csv(self, trajectories, output_dir):
        """
        Export structured trajectories to CSV files (one per vehicle).

        Args:
            trajectories (dict): Structured trajectory data
            output_dir (str): Directory to output CSV files
        """
        import os
        import csv

        os.makedirs(output_dir, exist_ok=True)

        for vehicle_id, trajectory_data in trajectories.items():
            output_file = os.path.join(output_dir, f"{vehicle_id}_trajectory.csv")

            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time(s)", "X(m)", "Y(m)", "Velocity(m/s)", "Heading(rad)"])

                for point in trajectory_data:
                    writer.writerow([
                        point["time"],
                        point["x"],
                        point["y"],
                        point["velocity"],
                        point["velocity"],
                        point["heading"]
                    ])

