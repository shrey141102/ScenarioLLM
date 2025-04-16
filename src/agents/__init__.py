"""
Agent-based enhancements for the LLMScenario framework.
"""

from .driver_agents import DriverAgentManager
from .scenario_agents import ScenarioAgentManager
from crewai import Crew


class AgentManager:
    """Main interface for working with agents."""

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
            str: Enhanced scenario with trajectories
        """
        crew = self.scenario_manager.create_scenario_crew(scenario_description, vehicle_states)
        result = crew.kickoff()
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
            list: List of agent trajectories
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
        results = crew.kickoff()

        # Extract results
        trajectories = []
        if isinstance(results, dict):
            # If results is a dictionary of task outputs
            for task_id, output in results.items():
                trajectories.append(output)
        elif isinstance(results, list):
            # If results is a list of task outputs
            trajectories = results
        else:
            # If results is a single string output
            trajectories = [results]

        return trajectories

    def react_to_scenario_event(self, scenario_description, event_description):
        """
        Have driver agents react to an event in a scenario.

        Args:
            scenario_description (str): Description of the scenario
            event_description (str): Description of the event

        Returns:
            list: List of agent reactions
        """
        reactions = []

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
            verbose=True  # Changed to boolean
        )

        # Kickoff the crew and get results
        results = crew.kickoff()

        # Extract results
        if isinstance(results, dict):
            # If results is a dictionary of task outputs
            for task_id, output in results.items():
                reactions.append(output)
        elif isinstance(results, list):
            # If results is a list of task outputs
            reactions = results
        else:
            # If results is a single string output
            reactions = [results]

        return reactions