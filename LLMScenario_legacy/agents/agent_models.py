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
        self.llm = OpenAI(model="gpt-4o-2024-08-06",temperature=0, openai_api_key=self.openai_api_key, max_tokens=5000)

    def create_scenario_analyzer(self):
        """
        Create a scenario analyzer agent.

        Returns:
            Agent: A CrewAI agent
        """
        return Agent(
            role="Scenario Analyzer",
            goal="Analyze driving scenarios to identify key interaction patterns and safety concerns",
            backstory="You are an expert traffic analyst with years of experience studying driving behaviors. Your specialty is identifying complex interactions and potential safety risks in traffic scenarios.",
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
            goal="Enhance driving scenarios to make them more challenging and realistic",
            backstory="You are a creative traffic scenario designer who specializes in creating challenging but realistic driving situations. You work with autonomous vehicle companies to test their systems in edge cases.",
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
            goal="Ensure all generated scenarios are physically realistic and follow basic traffic rules",
            backstory="You are a traffic safety expert with a background in physics and automotive engineering. Your job is to ensure generated scenarios are physically realistic and don't contain impossible maneuvers.",
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
                role="Aggressive Driver",
                goal="Simulate an aggressive driving style within physical limits",
                backstory="You are an impatient driver who tends to drive faster than others, change lanes frequently, and maintain shorter following distances.",
                verbose=True,
                llm=self.llm
            )
        elif driver_type == "cautious":
            return Agent(
                role="Cautious Driver",
                goal="Simulate a cautious driving style with safety as the priority",
                backstory="You are a very careful driver who prioritizes safety above all else. You maintain large following distances, drive at or below the speed limit, and rarely change lanes.",
                verbose=True,
                llm=self.llm
            )
        else:  # normal
            return Agent(
                role="Normal Driver",
                goal="Simulate a typical driving style balancing efficiency and safety",
                backstory="You are an average driver who follows most traffic rules but occasionally takes calculated risks to reach your destination efficiently.",
                verbose=True,
                llm=self.llm
            )