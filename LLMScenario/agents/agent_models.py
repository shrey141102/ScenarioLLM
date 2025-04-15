# """
# Base agent models for the LLMScenario framework.
# """
#
# from crewai import Agent, Task, Crew
# from langchain_openai import OpenAI
# # from langchain import Anthropic
# import os
#
#
# class AgentFactory:
#     """Factory class for creating different types of agents."""
#
#     def __init__(self, openai_api_key=None):
#         """
#         Initialize the agent factory.
#
#         Args:
#             openai_api_key (str, optional): OpenAI API key
#         """
#         self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
#         self.llm = OpenAI(model="gpt-4o-2024-08-06",temperature=0, openai_api_key=self.openai_api_key, max_tokens=5000)
#
#     def create_scenario_analyzer(self):
#         """
#         Create a scenario analyzer agent.
#
#         Returns:
#             Agent: A CrewAI agent
#         """
#         return Agent(
#             role="Scenario Analyzer",
#             goal="Analyze driving scenarios to identify key interaction patterns and safety concerns",
#             backstory="You are an expert traffic analyst with years of experience studying driving behaviors. Your specialty is identifying complex interactions and potential safety risks in traffic scenarios.",
#             verbose=True,
#             llm=self.llm
#         )
#
#     def create_scenario_enhancer(self):
#         """
#         Create a scenario enhancer agent.
#
#         Returns:
#             Agent: A CrewAI agent
#         """
#         return Agent(
#             role="Scenario Enhancer",
#             goal="Enhance driving scenarios to make them more challenging and realistic",
#             backstory="You are a creative traffic scenario designer who specializes in creating challenging but realistic driving situations. You work with autonomous vehicle companies to test their systems in edge cases.",
#             verbose=True,
#             llm=self.llm
#         )
#
#     def create_safety_validator(self):
#         """
#         Create a safety validator agent.
#
#         Returns:
#             Agent: A CrewAI agent
#         """
#         return Agent(
#             role="Safety Validator",
#             goal="Ensure all generated scenarios are physically realistic and follow basic traffic rules",
#             backstory="You are a traffic safety expert with a background in physics and automotive engineering. Your job is to ensure generated scenarios are physically realistic and don't contain impossible maneuvers.",
#             verbose=True,
#             llm=self.llm
#         )
#
#     def create_driver_agent(self, driver_type="normal"):
#         """
#         Create a driver agent with specific characteristics.
#
#         Args:
#             driver_type (str): Type of driver (normal, aggressive, cautious)
#
#         Returns:
#             Agent: A CrewAI agent
#         """
#         if driver_type == "aggressive":
#             return Agent(
#                 role="Aggressive Driver",
#                 goal="Simulate an aggressive driving style within physical limits",
#                 backstory="You are an impatient driver who tends to drive faster than others, change lanes frequently, and maintain shorter following distances.",
#                 verbose=True,
#                 llm=self.llm
#             )
#         elif driver_type == "cautious":
#             return Agent(
#                 role="Cautious Driver",
#                 goal="Simulate a cautious driving style with safety as the priority",
#                 backstory="You are a very careful driver who prioritizes safety above all else. You maintain large following distances, drive at or below the speed limit, and rarely change lanes.",
#                 verbose=True,
#                 llm=self.llm
#             )
#         else:  # normal
#             return Agent(
#                 role="Normal Driver",
#                 goal="Simulate a typical driving style balancing efficiency and safety",
#                 backstory="You are an average driver who follows most traffic rules but occasionally takes calculated risks to reach your destination efficiently.",
#                 verbose=True,
#                 llm=self.llm
#             )


"""
Base agent models for the LLMScenario framework.
"""

from crewai import Agent, Task, Crew
from langchain_openai import OpenAI
# from langchain import Anthropic
import os


class AgentFactory:
    """Factory class for creating different types of agents."""

    def __init__(self, openai_api_key=None):
        """
        Initialize the agent factory.

        Args:
            openai_api_key (str, optional): OpenAI API key
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        # Add a more specific system prompt to emphasize trajectory generation
        self.llm = OpenAI(
            model="gpt-4o", #  gpt-4o-2024-08-06
            temperature=0,
            openai_api_key=self.openai_api_key,
            max_tokens=8000
        )

    def create_scenario_analyzer(self):
        """
        Create a scenario analyzer agent.

        Returns:
            Agent: A CrewAI agent
        """
        return Agent(
            role="Scenario Analyzer",
            goal="Extract key vehicle trajectories and identify concrete interaction patterns",
            backstory="""You are a trajectory data analyst who specializes in processing vehicle movement data. 
            Your expertise is specifically in extracting numerical patterns from traffic data and identifying 
            precise coordinates, velocities, and headings. You always present your analysis in structured 
            numerical formats and avoid general descriptions.""",
            verbose=True,
            llm=self.llm
        )

    def create_scenario_enhancer(self):
        """
        Create a scenario enhancer agent.

        Returns:
            Agent: A CrewAI agent
        """
        return Agent(
            role="Scenario Enhancer",
            goal="Generate concrete vehicle trajectory data with specific coordinates and velocities",
            backstory="""You are a trajectory generation expert who specializes in creating numerical 
            vehicle path data. You always output exact coordinates, velocities, and headings over time. 
            You avoid general descriptions and philosophical statements about driving behaviors. 
            Instead, you focus on generating precise numerical time-series data that can be directly 
            plotted on a graph.""",
            verbose=True,
            llm=self.llm
        )

    def create_safety_validator(self):
        """
        Create a safety validator agent.

        Returns:
            Agent: A CrewAI agent
        """
        return Agent(
            role="Safety Validator",
            goal="Verify physical validity of trajectory data and ensure proper numerical formats",
            backstory="""You are a trajectory validation specialist who focuses on ensuring trajectory data 
            adheres to physical laws and proper formatting. You check that acceleration, velocity, and position 
            data follow realistic physics and that all data is presented in the correct numerical format for 
            direct use in simulation software. You flag any trajectory data that doesn't match the required 
            Time, X, Y, Velocity, Heading format.""",
            verbose=True,
            llm=self.llm
        )

    def create_driver_agent(self, driver_type="normal"):
        """
        Create a driver agent with specific characteristics.

        Args:
            driver_type (str): Type of driver (normal, aggressive, cautious)

        Returns:
            Agent: A CrewAI agent
        """
        if driver_type == "aggressive":
            return Agent(
                role="Trajectory Generator - Aggressive Pattern",
                goal="Generate numerical trajectory data reflecting aggressive driving patterns",
                backstory="""You are a trajectory data generator specializing in creating realistic numerical 
                vehicle trajectories for aggressive driving patterns. You output precise time series data with 
                coordinates, velocities, and headings. Your trajectories numerically show: faster acceleration/deceleration, 
                shorter following distances, and more frequent lane changes. You always output data in the exact 
                format: Time(s), X(m), Y(m), Velocity(m/s), Heading(rad) with specific numbers.""",
                verbose=True,
                llm=self.llm
            )
        elif driver_type == "cautious":
            return Agent(
                role="Trajectory Generator - Cautious Pattern",
                goal="Generate numerical trajectory data reflecting cautious driving patterns",
                backstory="""You are a trajectory data generator specializing in creating realistic numerical 
                vehicle trajectories for cautious driving patterns. You output precise time series data with 
                coordinates, velocities, and headings. Your trajectories numerically show: gentler acceleration/deceleration, 
                larger following distances, and fewer lane changes. You always output data in the exact 
                format: Time(s), X(m), Y(m), Velocity(m/s), Heading(rad) with specific numbers.""",
                verbose=True,
                llm=self.llm
            )
        else:  # normal
            return Agent(
                role="Trajectory Generator - Standard Pattern",
                goal="Generate numerical trajectory data reflecting standard driving patterns",
                backstory="""You are a trajectory data generator specializing in creating realistic numerical 
                vehicle trajectories for standard driving patterns. You output precise time series data with 
                coordinates, velocities, and headings. Your trajectories numerically show balanced acceleration/deceleration, 
                standard following distances, and occasional lane changes. You always output data in the exact 
                format: Time(s), X(m), Y(m), Velocity(m/s), Heading(rad) with specific numbers.""",
                verbose=True,
                llm=self.llm
            )