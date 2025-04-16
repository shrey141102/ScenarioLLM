"""
Scenario generation and enhancement agents.
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
            Analyze the following driving scenario:

            Scenario:
            {scenario_description}

            Vehicle States:
            {vehicle_states}

            Your task is to:
            1. Identify all key interactions between vehicles
            2. Assess the complexity level of the scenario
            3. Identify potential safety concerns or collision risks
            4. Determine what makes this scenario challenging for autonomous vehicles

            Provide a detailed analysis that covers these aspects.
            """,
            agent=self.analyzer,
            expected_output="A detailed analysis of the scenario highlighting key interactions, complexity, safety concerns, and challenges for autonomous vehicles."
        )

    def enhance_scenario_task(self, scenario_description, vehicle_states, analyzer_output):
        """
        Create a task to enhance a scenario.

        Args:
            scenario_description (str): Description of the scenario
            vehicle_states (str): Description of vehicle states
            analyzer_output (str): Output from the analyzer agent

        Returns:
            Task: A CrewAI task
        """
        return Task(
            description=f"""
            Enhance the following driving scenario to make it more challenging:

            Scenario:
            {scenario_description}

            Vehicle States:
            {vehicle_states}

            Analysis:
            {analyzer_output}

            Your task is to:
            1. Create a more challenging version of this scenario
            2. Introduce complex interactions that test autonomous driving capabilities
            3. Add at least 2 new vehicles with specific behaviors
            4. Make the scenario realistic but push the boundaries of complexity

            Provide a detailed description of the enhanced scenario and the trajectories of all vehicles involved.
            The trajectories should be in the following format:

            ```
            Vehicle ID: [id]
            Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
            0.0, x0, y0, v0, h0
            0.1, x1, y1, v1, h1
            ...
            ```
            """,
            agent=self.enhancer,
            expected_output="An enhanced scenario with detailed vehicle trajectories that presents more challenges for autonomous vehicles."
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
            Validate the following enhanced driving scenario:

            {enhanced_scenario}

            Your task is to:
            1. Check if all vehicle maneuvers are physically realistic (acceleration, deceleration, turning radius)
            2. Verify that vehicles don't occupy the same space at the same time (no collisions)
            3. Ensure all trajectories follow basic physics (no teleportation, reasonable speeds)
            4. Assess if the scenario is challenging but still realistic

            Provide a validation report with any issues found and suggestions for improvements.
            If you find any problems, provide specific corrections to the trajectories.
            """,
            agent=self.validator,
            expected_output="A validation report identifying any physical inconsistencies, collisions, or unrealistic behavior in the scenario, with suggested corrections."
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
