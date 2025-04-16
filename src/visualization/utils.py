"""
Utility functions for visualization.
"""

import os
import re


def extract_scenario_description(scenario_text, llm_type):
    """Extract scenario description from LLM-generated text."""
    if llm_type == 'gpt4':
        match = re.search(r'\[SCENARIO_DESCRIPTION\](.*?)\[/SCENARIO_DESCRIPTION\]', scenario_text, re.DOTALL)
    elif llm_type == 'claude':
        match = re.search(r'<scenario_description>(.*?)</scenario_description>', scenario_text, re.DOTALL)
    elif llm_type == 'gemini':
        match = re.search(r'\[SCENARIO_DESCRIPTION\](.*?)\[/SCENARIO_DESCRIPTION\]', scenario_text, re.DOTALL)
    else:
        return "Unknown LLM type"

    if match:
        return match.group(1).strip()
    else:
        return "No scenario description found"


def extract_enhancement_strategy(scenario_text, llm_type):
    """Extract enhancement strategy from LLM-generated text."""
    if llm_type == 'gpt4':
        match = re.search(r'\[ENHANCEMENT_STRATEGY\](.*?)\[/ENHANCEMENT_STRATEGY\]', scenario_text, re.DOTALL)
    elif llm_type == 'claude':
        match = re.search(r'<enhancement_strategy>(.*?)</enhancement_strategy>', scenario_text, re.DOTALL)
    elif llm_type == 'gemini':
        match = re.search(r'\[ENHANCEMENT_STRATEGY\](.*?)\[/ENHANCEMENT_STRATEGY\]', scenario_text, re.DOTALL)
    else:
        return "Unknown LLM type"

    if match:
        return match.group(1).strip()
    else:
        return "No enhancement strategy found"


def extract_vehicle_trajectories(scenario_text, llm_type):
    """Extract vehicle trajectories from LLM-generated text."""
    if llm_type == 'gpt4':
        match = re.search(r'\[VEHICLE_TRAJECTORIES\](.*?)\[/VEHICLE_TRAJECTORIES\]', scenario_text, re.DOTALL)
    elif llm_type == 'claude':
        match = re.search(r'<vehicle_trajectories>(.*?)</vehicle_trajectories>', scenario_text, re.DOTALL)
    elif llm_type == 'gemini':
        match = re.search(r'\[VEHICLE_TRAJECTORIES\](.*?)\[/VEHICLE_TRAJECTORIES\]', scenario_text, re.DOTALL)
    else:
        return "Unknown LLM type"

    if match:
        return match.group(1).strip()
    else:
        return "No vehicle trajectories found"