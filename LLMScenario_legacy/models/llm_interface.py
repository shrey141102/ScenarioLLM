"""
Interface for different LLM models.
"""

import openai
import anthropic
from google import genai
from google.genai import types
from config.api_config import *
from models.prompt_templates import PromptTemplates

class LLMInterface:
    def __init__(self):
        """Initialize the LLM interfaces."""
        # Set up OpenAI client
        openai.api_key = OPENAI_API_KEY

        # Set up Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Set up Google Generative AI
        self.gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

    def generate_with_gpt4(self, road_env, vehicle_states, tasks_interactions):
        """
        Generate scenario using GPT-4.

        Args:
            road_env (str): Description of road environment
            vehicle_states (str): Description of vehicle states
            tasks_interactions (str): Description of tasks and interactions

        Returns:
            str: Generated scenario
        """
        base_prompt = PromptTemplates.get_base_scenario_prompt(
            road_env, vehicle_states, tasks_interactions
        )
        formatted_prompt = PromptTemplates.format_for_gpt4(base_prompt)

        response = openai.chat.completions.create(
            model=GPT4_MODEL,
            messages=[{"role": "system", "content": "You are an advanced driving scenario generator for autonomous vehicle research."},
                      {"role": "user", "content": formatted_prompt}],
            temperature=0,
            max_tokens=MAX_TOKENS
        )

        return response.choices[0].message.content

    def generate_with_claude(self, road_env, vehicle_states, tasks_interactions):
        """
        Generate scenario using Claude 3.7.

        Args:
            road_env (str): Description of road environment
            vehicle_states (str): Description of vehicle states
            tasks_interactions (str): Description of tasks and interactions

        Returns:
            str: Generated scenario
        """
        base_prompt = PromptTemplates.get_base_scenario_prompt(
            road_env, vehicle_states, tasks_interactions
        )
        formatted_prompt = PromptTemplates.format_for_claude(base_prompt)

        response = self.anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system="You are an advanced driving scenario generator for autonomous vehicle research.",
            messages=[{"role": "user", "content": formatted_prompt}]
        )

        return response.content[0].text

    def generate_with_gemini(self, road_env, vehicle_states, tasks_interactions):
        """
        Generate scenario using Gemini.
        """
        base_prompt = PromptTemplates.get_base_scenario_prompt(
            road_env, vehicle_states, tasks_interactions
        )
        formatted_prompt = PromptTemplates.format_for_gemini(base_prompt)

        try:
            response = self.gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[formatted_prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_TOKENS,
                    temperature=0
                )
            )

            if response is None or not hasattr(response, 'text') or response.text is None:
                raise ValueError("Received empty response from Gemini API")

            return response.text
        except Exception as e:
            print(f"Error from Gemini API: {e}")
            # Return a fallback or error message instead of None
            return f"ERROR: Could not generate scenario with Gemini - {str(e)}"