# """
# Scenario generation and enhancement agents.
# """
#
# from crewai import Task, Crew
# from .agent_models import AgentFactory
#
# class ScenarioAgentManager:
#     """Manager for scenario-related agents."""
#
#     def __init__(self, openai_api_key=None):
#         """
#         Initialize the scenario agent manager.
#
#         Args:
#             openai_api_key (str, optional): OpenAI API key
#         """
#         self.factory = AgentFactory(openai_api_key)
#
#         # Create scenario agents
#         self.analyzer = self.factory.create_scenario_analyzer()
#         self.enhancer = self.factory.create_scenario_enhancer()
#         self.validator = self.factory.create_safety_validator()
#
#     def analyze_scenario_task(self, scenario_description, vehicle_states):
#         """
#         Create a task to analyze a scenario.
#
#         Args:
#             scenario_description (str): Description of the scenario
#             vehicle_states (str): Description of vehicle states
#
#         Returns:
#             Task: A CrewAI task
#         """
#         return Task(
#             description=f"""
#             Analyze the following driving scenario:
#
#             Scenario:
#             {scenario_description}
#
#             Vehicle States:
#             {vehicle_states}
#
#             Your task is to:
#             1. Identify all key interactions between vehicles
#             2. Assess the complexity level of the scenario
#             3. Identify potential safety concerns or collision risks
#             4. Determine what makes this scenario challenging for autonomous vehicles
#
#             Provide a detailed analysis that covers these aspects.
#             """,
#             agent=self.analyzer,
#             expected_output="A detailed analysis of the scenario highlighting key interactions, complexity, safety concerns, and challenges for autonomous vehicles."
#         )
#
#     def enhance_scenario_task(self, scenario_description, vehicle_states, analyzer_output):
#         """
#         Create a task to enhance a scenario.
#
#         Args:
#             scenario_description (str): Description of the scenario
#             vehicle_states (str): Description of vehicle states
#             analyzer_output (str): Output from the analyzer agent
#
#         Returns:
#             Task: A CrewAI task
#         """
#         return Task(
#             description=f"""
#             Enhance the following driving scenario to make it more challenging:
#
#             Scenario:
#             {scenario_description}
#
#             Vehicle States:
#             {vehicle_states}
#
#             Analysis:
#             {analyzer_output}
#
#             Your task is to:
#             1. Create a more challenging version of this scenario
#             2. Introduce complex interactions that test autonomous driving capabilities
#             3. Add at least 2 new vehicles with specific behaviors
#             4. Make the scenario realistic but push the boundaries of complexity
#
#             Provide a detailed description of the enhanced scenario and the trajectories of all vehicles involved.
#             The trajectories should be in the following format:
#
#             ```
#             Vehicle ID: [id]
#             Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
#             0.0, x0, y0, v0, h0
#             0.1, x1, y1, v1, h1
#             ...
#             ```
#             """,
#             agent=self.enhancer,
#             expected_output="An enhanced scenario with detailed vehicle trajectories that presents more challenges for autonomous vehicles."
#         )
#
#     def validate_scenario_task(self, enhanced_scenario):
#         """
#         Create a task to validate an enhanced scenario.
#
#         Args:
#             enhanced_scenario (str): The enhanced scenario from the enhancer agent
#
#         Returns:
#             Task: A CrewAI task
#         """
#         return Task(
#             description=f"""
#             Validate the following enhanced driving scenario:
#
#             {enhanced_scenario}
#
#             Your task is to:
#             1. Check if all vehicle maneuvers are physically realistic (acceleration, deceleration, turning radius)
#             2. Verify that vehicles don't occupy the same space at the same time (no collisions)
#             3. Ensure all trajectories follow basic physics (no teleportation, reasonable speeds)
#             4. Assess if the scenario is challenging but still realistic
#
#             Provide a validation report with any issues found and suggestions for improvements.
#             If you find any problems, provide specific corrections to the trajectories.
#             """,
#             agent=self.validator,
#             expected_output="A validation report identifying any physical inconsistencies, collisions, or unrealistic behavior in the scenario, with suggested corrections."
#         )
#
#     def create_scenario_crew(self, scenario_description, vehicle_states):
#         """
#         Create a crew of agents to analyze, enhance, and validate a scenario.
#
#         Args:
#             scenario_description (str): Description of the scenario
#             vehicle_states (str): Description of vehicle states
#
#         Returns:
#             Crew: A CrewAI crew
#         """
#         # Create tasks
#         analysis_task = self.analyze_scenario_task(scenario_description, vehicle_states)
#         enhance_task = self.enhance_scenario_task(scenario_description, vehicle_states, "{{ analysis_task.output }}")
#         validate_task = self.validate_scenario_task("{{ enhance_task.output }}")
#
#         # Create crew
#         crew = Crew(
#             agents=[self.analyzer, self.enhancer, self.validator],
#             tasks=[analysis_task, enhance_task, validate_task],
#             verbose=True
#         )
#
#         return crew


"""
Scenario generation and enhancement agents for producing structured trajectory data.
"""

from crewai import Task, Crew
from .agent_models import AgentFactory


class ScenarioAgentManager:
    """Manager for scenario-related agents."""

    def __init__(self, openai_api_key=None):
        """
        Initialize the scenario agent manager.

        Args:
            openai_api_key (str, optional): OpenAI API key
        """
        self.factory = AgentFactory(openai_api_key)

        # Create scenario agents
        self.analyzer = self.factory.create_scenario_analyzer()
        self.enhancer = self.factory.create_scenario_enhancer()
        self.validator = self.factory.create_safety_validator()

    def analyze_scenario_task(self, scenario_description, vehicle_states):
        """
        Create a task to analyze a scenario.

        Args:
            scenario_description (str): Description of the scenario
            vehicle_states (str): Description of vehicle states

        Returns:
            Task: A CrewAI task
        """
        return Task(
            description=f"""
            ## TRAJECTORY ANALYSIS TASK

            ### SCENARIO TO ANALYZE
            {scenario_description}

            ### VEHICLE STATES
            {vehicle_states}

            ### REQUIRED OUTPUT FORMAT
            Your output must contain these sections with concrete numerical analysis:

            ```
            [INTERACTION POINTS]
            - Time: X.X seconds - Vehicle A (X,Y) and Vehicle B (X,Y) - Distance: X.X meters
            - Time: X.X seconds - Vehicle C (X,Y) and Vehicle D (X,Y) - Distance: X.X meters
            (list all key interaction points with specific coordinates and times)

            [COMPLEXITY METRICS]
            - Number of vehicles: X
            - Number of lane changes: X
            - Minimum vehicle separation: X.X meters at time X.X seconds
            - Maximum vehicle speed: X.X m/s by Vehicle X
            - Acceleration range: X.X to X.X m/s²

            [SAFETY ANALYSIS]
            - Minimum time-to-collision: X.X seconds at time X.X
            - Critical points: (list specific coordinates where safety margins are minimal)
            - Potential collision coordinates: (X,Y) at time X.X seconds if no intervention

            [CHALLENGE ASSESSMENT]
            - Specific challenging coordinates: (X,Y) at time X.X seconds
            - Key decision points: (list specific times and positions requiring decisions)
            ```

            ### IMPORTANT RULES
            1. Use ONLY numerical data and specific coordinates in your analysis
            2. DO NOT provide general descriptions or philosophical discussions
            3. Focus on extracting concrete trajectory patterns with precise measurements
            4. If exact values aren't available, make reasonable estimates based on the scenario
            5. All positions must be in (X,Y) coordinates, times in seconds, velocities in m/s
            """,
            agent=self.analyzer,
            expected_output="A detailed numerical analysis of vehicle interactions, complexity metrics, safety assessment, and challenge points."
        )

    def enhance_scenario_task(self, scenario_description, vehicle_states, analyzer_output):
        """
        Create a task to enhance a scenario with specific trajectory data.

        Args:
            scenario_description (str): Description of the scenario
            vehicle_states (str): Description of vehicle states
            analyzer_output (str): Output from the analyzer agent

        Returns:
            Task: A CrewAI task
        """
        # Sample trajectory for example
        sample_trajectory = """
        Vehicle ID: V1
        Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
        0.0, 100.0, 10.0, 30.0, 0.0
        0.5, 115.0, 10.0, 31.2, 0.0
        1.0, 130.8, 10.2, 32.5, 0.02

        Vehicle ID: V2
        Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
        0.0, 150.0, 13.5, 28.0, 0.0
        0.5, 164.0, 13.5, 28.0, 0.0
        1.0, 178.0, 13.4, 28.0, -0.01
        """

        return Task(
            description=f"""
            ## TRAJECTORY GENERATION TASK

            ### SCENARIO TO ENHANCE
            {scenario_description}

            ### VEHICLE STATES
            {vehicle_states}

            ### SCENARIO ANALYSIS
            {analyzer_output}

            ### REQUIRED OUTPUT FORMAT
            You must generate trajectory data for ALL vehicles in EXACTLY this format:

            ```
            Vehicle ID: [id]
            Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
            0.0, x0, y0, v0, h0
            0.5, x1, y1, v1, h1
            ...
            (at least 20 time steps with 0.5s intervals for each vehicle)

            Vehicle ID: [next_id]
            Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
            0.0, x0, y0, v0, h0
            ...
            ```

            ### SAMPLE TRAJECTORY DATA
            Here is an example of the expected format:
            ```
            {sample_trajectory}
            ```

            ### ENHANCEMENT REQUIREMENTS
            1. Add at least 2 new vehicles with complete trajectory data
            2. Create challenging interaction points at specific coordinates
            3. Maintain physical realism: max acceleration 3 m/s², max deceleration 7 m/s²
            4. Ensure vehicles maintain minimum 2m separation except during intentional near-miss events
            5. Include at least one lane change maneuver with specific coordinates
            6. Vehicles should maintain consistent lane positions with the following y-coordinate ranges:
               * Lane 1: y = 5.0-8.5 meters
               * Lane 2: y = 8.5-12.0 meters
               * Lane 3: y = 12.0-15.5 meters
               * Lane 4: y = 15.5-19.0 meters
            7. Lane changes should be smooth transitions between these y-coordinate ranges

            ### IMPORTANT RULES
            1. ONLY output the trajectory data in the exact format specified
            2. DO NOT include explanations, descriptions, or analysis outside the trajectory data
            3. Your entire response should be ONLY the vehicle trajectory data tables
            4. Heading is in radians (0 = east, π/2 = north, π = west, 3π/2 = south)
            5. Time steps must be at 0.5 second intervals
            6. All vehicles must have the same number of time steps covering the same time period
            """,
            agent=self.enhancer,
            expected_output="Complete trajectory data tables for all vehicles in the exact specified format."
        )

    def validate_scenario_task(self, enhanced_scenario):
        """
        Create a task to validate an enhanced scenario.

        Args:
            enhanced_scenario (str): The enhanced scenario from the enhancer agent

        Returns:
            Task: A CrewAI task
        """
        return Task(
            description=f"""
            ## TRAJECTORY VALIDATION TASK

            ### TRAJECTORY DATA TO VALIDATE
            {enhanced_scenario}

            ### REQUIRED OUTPUT FORMAT
            Your validation must provide either:

            1. If the trajectories are valid:
            ```
            [VALIDATION RESULT: PASSED]
            All trajectories are physically realistic and properly formatted.
            ```

            2. If there are issues, provide corrections in this format:
            ```
            [VALIDATION RESULT: ISSUES FOUND]

            [FORMATTING ISSUES]
            - Vehicle X: Missing columns in row Y
            - Vehicle Z: Incorrect format at time T

            [PHYSICAL INCONSISTENCIES]
            - Vehicle X: Impossible acceleration (X m/s²) at time T
            - Vehicle Y: Teleportation detected between time T1 and T2

            [COLLISION DETECTION]
            - Vehicles X and Y: Collision at time T, position (X,Y)
            - Vehicles Z and W: Proximity violation at time T (X.X meters, minimum is 2m)

            [CORRECTED TRAJECTORIES]
            Vehicle ID: X
            Time(s), X(m), Y(m), Velocity(m/s), Heading(rad)
            ... (corrected data only for problematic vehicles/timestamps)
            ```

            ### VALIDATION RULES
            1. Maximum acceleration: 3 m/s²
            2. Maximum deceleration: 7 m/s²
            3. Maximum speed: 50 m/s (~180 km/h)
            4. Minimum vehicle separation: 2 meters (except for intentional near-misses)
            5. Heading changes must be physically possible (max 0.1 rad/s for highway)
            6. All trajectory data must follow the exact required format
            7. Lane positioning: Vehicles should stay within lane bounds except during explicit lane changes
               * Lane 1: y = 5.0-8.5 meters
               * Lane 2: y = 8.5-12.0 meters
               * Lane 3: y = 12.0-15.5 meters
               * Lane 4: y = 15.5-19.0 meters

            ### IMPORTANT
            1. DO NOT provide general advice or philosophical discussion
            2. Only flag actual problems with specific numerical evidence
            3. If providing corrected trajectories, maintain the same format
            4. Be precise about times and positions where issues occur
            """,
            agent=self.validator,
            expected_output="A validation report with specific issues found or confirmation that all trajectories are valid."
        )

    def create_scenario_crew(self, scenario_description, vehicle_states):
        """
        Create a crew of agents to analyze, enhance, and validate a scenario.

        Args:
            scenario_description (str): Description of the scenario
            vehicle_states (str): Description of vehicle states

        Returns:
            Crew: A CrewAI crew
        """
        # Create tasks
        analysis_task = self.analyze_scenario_task(scenario_description, vehicle_states)
        enhance_task = self.enhance_scenario_task(scenario_description, vehicle_states, "{{ analysis_task.output }}")
        validate_task = self.validate_scenario_task("{{ enhance_task.output }}")

        # Create crew
        crew = Crew(
            agents=[self.analyzer, self.enhancer, self.validator],
            tasks=[analysis_task, enhance_task, validate_task],
            verbose=True
        )

        return crew
