# """
# Driver agents with different driving styles.
# """
#
# from crewai import Task
# from .agent_models import AgentFactory
#
#
# class DriverAgentManager:
#     """Manager for driver agents with different behaviors."""
#
#     def __init__(self, openai_api_key=None):
#         """
#         Initialize the driver agent manager.
#
#         Args:
#             openai_api_key (str, optional): OpenAI API key
#         """
#         self.factory = AgentFactory(openai_api_key)
#
#         # Create driver agents with different styles
#         self.aggressive_driver = self.factory.create_driver_agent("aggressive")
#         self.normal_driver = self.factory.create_driver_agent("normal")
#         self.cautious_driver = self.factory.create_driver_agent("cautious")
#
#     def create_driving_task(self, agent, scenario_description, vehicle_id, initial_position, destination):
#         """
#         Create a driving task for an agent.
#
#         Args:
#             agent: The CrewAI agent
#             scenario_description (str): Description of the scenario
#             vehicle_id (str): ID of the vehicle
#             initial_position (tuple): Initial position (x, y)
#             destination (tuple): Destination position (x, y)
#
#         Returns:
#             Task: A CrewAI task
#         """
#         return Task(
#             description=f"""
#             You are driving vehicle {vehicle_id} in the following scenario:
#
#             {scenario_description}
#
#             Your initial position is {initial_position} and your destination is {destination}.
#             Based on your driving style and the scenario, describe how you would navigate and
#             generate the trajectory data in the following format:
#
#             ```
#             Vehicle ID: {vehicle_id}
#             Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
#             0.0, x0, y0, v0, h0
#             0.1, x1, y1, v1, h1
#             ...
#             ```
#
#             Include at least 20 time steps, calculating the position, velocity, and heading at each step.
#             Your driving style should be reflected in your trajectory decisions.
#             """,
#             agent=agent,
#             expected_output="A detailed trajectory for the vehicle showing positions, velocities and headings over time."
#         )
#
#     def react_to_event(self, agent, scenario_description, vehicle_id, current_position, current_velocity,
#                        event_description):
#         """
#         Create a task for an agent to react to a traffic event.
#
#         Args:
#             agent: The CrewAI agent
#             scenario_description (str): Description of the scenario
#             vehicle_id (str): ID of the vehicle
#             current_position (tuple): Current position (x, y)
#             current_velocity (float): Current velocity
#             event_description (str): Description of the event to react to
#
#         Returns:
#             Task: A CrewAI task
#         """
#         return Task(
#             description=f"""
#             You are driving vehicle {vehicle_id} in the following scenario:
#
#             {scenario_description}
#
#             Your current position is {current_position} and your current velocity is {current_velocity} m/s.
#
#             The following event has just occurred:
#             {event_description}
#
#             Based on your driving style, describe how you would react to this event and
#             generate the trajectory data for the next 5 seconds in the following format:
#
#             ```
#             Vehicle ID: {vehicle_id}
#             Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
#             0.0, x0, y0, v0, h0
#             0.2, x1, y1, v1, h1
#             ...
#             ```
#
#             Include at least 10 time steps, calculating the position, velocity, and heading at each step.
#             Your driving style should be reflected in your reaction.
#             """,
#             agent=agent,
#             expected_output="A trajectory showing how the vehicle reacts to the event over the next 5 seconds."
#         )

"""
Driver agents with different driving styles for generating trajectory data.
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
        # Sample trajectory for example
        sample_trajectory = """
        Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
        0.0, 100.0, 10.0, 30.0, 0.0
        0.5, 115.0, 10.0, 31.2, 0.0
        1.0, 130.8, 10.2, 32.5, 0.02
        1.5, 147.0, 10.8, 33.1, 0.04
        """

        return Task(
            description=f"""
            ## TRAJECTORY GENERATION TASK

            ### REQUIRED OUTPUT FORMAT
            Your task is to generate ONLY trajectory data in this EXACT format:
            ```
            Vehicle ID: {vehicle_id}
            Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
            0.0, x0, y0, v0, h0
            0.5, x1, y1, v1, h1
            ... (at least 20 time steps with 0.5s intervals)
            ```

            ### SCENARIO INFORMATION
            {scenario_description}
            
            ### LANE INFORMATION
            - The highway has 4 lanes (numbered 1-4 from bottom to top)
            - Each lane is 3.5 meters wide with the following y-coordinate ranges:
              * Lane 1: y = 5.0-8.5 meters
              * Lane 2: y = 8.5-12.0 meters
              * Lane 3: y = 12.0-15.5 meters
              * Lane 4: y = 15.5-19.0 meters
            
            IMPORTANT: Unless a vehicle is performing a lane change, its y-coordinate should remain within the range of its current lane.

            
            ### VEHICLE PARAMETERS
            - Vehicle ID: {vehicle_id}
            - Initial position (x,y): {initial_position}
            - Destination position (x,y): {destination}
            - Initial velocity: 30 m/s (unless specified otherwise)
            - Initial heading: 0 rad (east direction) unless specified otherwise

            ### SAMPLE TRAJECTORY
            Here is an example of the output format expected:
            ```
            {sample_trajectory}
            ```

            ### IMPORTANT RULES
            1. DO NOT include any explanations or descriptions outside the trajectory data
            2. Calculate realistic physics: velocity changes must be physically possible
            3. Heading is in radians (0 = east, π/2 = north, π = west, 3π/2 = south)
            4. Your trajectory should reflect your driving style through numerical patterns
            5. DO NOT switch to a philosophical discussion about driving
            6. ONLY output the trajectory data in the exact format specified

            ### IMPORTANT 
            Your entire response should ONLY be the trajectory data in the exact format requested.
            Do not write anything else.
            """,
            agent=agent,
            expected_output="A trajectory table with 20+ rows of numerical position, velocity and heading data."
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
        # Sample trajectory reaction for example
        sample_reaction = """
        Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
        0.0, 100.0, 10.0, 30.0, 0.0
        0.2, 106.0, 10.0, 29.0, 0.0
        0.4, 111.6, 10.2, 27.5, 0.05
        0.6, 117.0, 10.8, 26.0, 0.08
        """

        return Task(
            description=f"""
            ## TRAJECTORY REACTION GENERATION TASK

            ### REQUIRED OUTPUT FORMAT
            Your task is to generate ONLY trajectory data in this EXACT format:
            ```
            Vehicle ID: {vehicle_id}
            Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
            0.0, x0, y0, v0, h0
            0.2, x1, y1, v1, h1
            ... (at least 10 time steps with 0.2s intervals)
            ```

            ### SCENARIO INFORMATION
            {scenario_description}

            ### CURRENT VEHICLE STATE
            - Vehicle ID: {vehicle_id}
            - Current position (x,y): {current_position}
            - Current velocity: {current_velocity} m/s
            - Current heading: 0 rad (east direction) unless specified otherwise

            ### EVENT TO REACT TO
            {event_description}

            ### SAMPLE REACTION TRAJECTORY
            Here is an example of the output format expected:
            ```
            {sample_reaction}
            ```

            ### IMPORTANT RULES
            1. DO NOT include any explanations or descriptions outside the trajectory data
            2. Calculate realistic physics: maximum deceleration is 7 m/s², maximum acceleration is 3 m/s²
            3. Heading is in radians (0 = east, π/2 = north, π = west, 3π/2 = south)
            4. Your trajectory should reflect your driving style through numerical patterns
            5. DO NOT switch to a philosophical discussion about driving
            6. ONLY output the trajectory data in the exact format specified

            ### IMPORTANT 
            Your entire response should ONLY be the trajectory data in the exact format requested.
            Do not write anything else.
            """,
            agent=agent,
            expected_output="A trajectory table with 10+ rows of numerical position, velocity and heading data."
        )