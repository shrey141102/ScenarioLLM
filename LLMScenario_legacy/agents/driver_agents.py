"""
Driver agents with different driving styles.
"""

from crewai import Task
from .agent_models import AgentFactory


class DriverAgentManager:
    """Manager for driver agents with different behaviors."""

    def __init__(self, openai_api_key=None):
        """
        Initialize the driver agent manager.

        Args:
            openai_api_key (str, optional): OpenAI API key
        """
        self.factory = AgentFactory(openai_api_key)

        # Create driver agents with different styles
        self.aggressive_driver = self.factory.create_driver_agent("aggressive")
        self.normal_driver = self.factory.create_driver_agent("normal")
        self.cautious_driver = self.factory.create_driver_agent("cautious")

    def create_driving_task(self, agent, scenario_description, vehicle_id, initial_position, destination):
        """
        Create a driving task for an agent.

        Args:
            agent: The CrewAI agent
            scenario_description (str): Description of the scenario
            vehicle_id (str): ID of the vehicle
            initial_position (tuple): Initial position (x, y)
            destination (tuple): Destination position (x, y)

        Returns:
            Task: A CrewAI task
        """
        return Task(
            description=f"""
            You are driving vehicle {vehicle_id} in the following scenario:

            {scenario_description}

            Your initial position is {initial_position} and your destination is {destination}.
            Based on your driving style and the scenario, describe how you would navigate and
            generate the trajectory data in the following format:

            ```
            Vehicle ID: {vehicle_id}
            Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
            0.0, x0, y0, v0, h0
            0.1, x1, y1, v1, h1
            ...
            ```

            Include at least 20 time steps, calculating the position, velocity, and heading at each step.
            Your driving style should be reflected in your trajectory decisions.
            """,
            agent=agent,
            expected_output="A detailed trajectory for the vehicle showing positions, velocities and headings over time."
        )

    def react_to_event(self, agent, scenario_description, vehicle_id, current_position, current_velocity,
                       event_description):
        """
        Create a task for an agent to react to a traffic event.

        Args:
            agent: The CrewAI agent
            scenario_description (str): Description of the scenario
            vehicle_id (str): ID of the vehicle
            current_position (tuple): Current position (x, y)
            current_velocity (float): Current velocity
            event_description (str): Description of the event to react to

        Returns:
            Task: A CrewAI task
        """
        return Task(
            description=f"""
            You are driving vehicle {vehicle_id} in the following scenario:

            {scenario_description}

            Your current position is {current_position} and your current velocity is {current_velocity} m/s.

            The following event has just occurred:
            {event_description}

            Based on your driving style, describe how you would react to this event and
            generate the trajectory data for the next 5 seconds in the following format:

            ```
            Vehicle ID: {vehicle_id}
            Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
            0.0, x0, y0, v0, h0
            0.2, x1, y1, v1, h1
            ...
            ```

            Include at least 10 time steps, calculating the position, velocity, and heading at each step.
            Your driving style should be reflected in your reaction.
            """,
            agent=agent,
            expected_output="A trajectory showing how the vehicle reacts to the event over the next 5 seconds."
        )