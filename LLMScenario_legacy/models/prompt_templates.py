class PromptTemplates:
    @staticmethod
    def get_base_scenario_prompt(road_env, vehicle_states, tasks_interactions):
        """
        Create a base scenario prompt that works for all models.
        """
        return f"""You are a scenario generator for autonomous driving research. 
I will provide you with a description of a driving scenario from a real-world dataset, and I'd like you to generate a new, challenging variation of this scenario.

# Road Environment:
{road_env}

# Vehicle States and Trajectories:
{vehicle_states}

# Tasks and Interactions:
{tasks_interactions}

Please generate a new, more challenging scenario based on this input. Your generated scenario should:
1. Be realistic and physically feasible
2. Maintain the same road environment
3. Include more complex interactions between vehicles
4. Present challenging situations for autonomous vehicles
5. Output vehicle trajectories in a STRICTLY STRUCTURED FORMAT as specified below
"""

    @staticmethod
    def format_for_gpt4(base_prompt):
        """Format the base prompt specifically for GPT-4."""
        return base_prompt + """
Your response MUST follow this EXACT structure:

[SCENARIO_DESCRIPTION]
Write a detailed description of the enhanced scenario here.
[/SCENARIO_DESCRIPTION]

[ENHANCEMENT_STRATEGY]
Explain how you're making the scenario more challenging.
[/ENHANCEMENT_STRATEGY]

[VEHICLE_TRAJECTORIES]
Vehicle ID: 1 (Car - Brief description)
Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
0.0, 362.26, 21.68, 41.07, 0.0
0.5, 382.80, 21.68, 41.07, 0.0
1.0, 403.33, 21.68, 41.07, 0.0
... (continue with more timesteps and add more vehicles)

Vehicle ID: 2 (Car - Brief description)
Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
0.0, 162.75, 9.39, -32.48, 3.14
0.5, 146.51, 9.39, -32.48, 3.14
1.0, 130.27, 9.39, -32.48, 3.14
... (continue with more timesteps)
[/VEHICLE_TRAJECTORIES]

Follow this structure EXACTLY without deviations. Each vehicle trajectory must have at least 10 timesteps with 0.5 second intervals.
"""

    @staticmethod
    def format_for_claude(base_prompt):
        """Format the base prompt specifically for Claude 3.7."""
        return base_prompt + """
Your response MUST follow this EXACT structure:

<scenario_description>
Write a detailed description of the enhanced scenario here.
</scenario_description>

<enhancement_strategy>
Explain how you're making the scenario more challenging.
</enhancement_strategy>

<vehicle_trajectories>
Vehicle ID: 1 (Car - Brief description)
Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
0.0, 362.26, 21.68, 41.07, 0.0
0.5, 382.80, 21.68, 41.07, 0.0
1.0, 403.33, 21.68, 41.07, 0.0
... (continue with more timesteps and add more vehicles)

Vehicle ID: 2 (Car - Brief description)
Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
0.0, 162.75, 9.39, -32.48, 3.14
0.5, 146.51, 9.39, -32.48, 3.14
1.0, 130.27, 9.39, -32.48, 3.14
... (continue with more timesteps)
</vehicle_trajectories>

Follow this structure EXACTLY without deviations. Each vehicle trajectory must have at least 10 timesteps with 0.5 second intervals.
"""

    @staticmethod
    def format_for_gemini(base_prompt):
        """Format the base prompt specifically for Gemini."""
        return base_prompt + """
Your response MUST follow this EXACT structure:

[SCENARIO_DESCRIPTION]
Write a detailed description of the enhanced scenario here.
[/SCENARIO_DESCRIPTION]

[ENHANCEMENT_STRATEGY]
Explain how you're making the scenario more challenging.
[/ENHANCEMENT_STRATEGY]

[VEHICLE_TRAJECTORIES]
Vehicle ID: 1 (Car - Brief description)
Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
0.0, 362.26, 21.68, 41.07, 0.0
0.5, 382.80, 21.68, 41.07, 0.0
1.0, 403.33, 21.68, 41.07, 0.0
... (continue with more timesteps and add more vehicles)

Vehicle ID: 2 (Car - Brief description)
Time (s), X (m), Y (m), Velocity (m/s), Heading (rad)
0.0, 162.75, 9.39, -32.48, 3.14
0.5, 146.51, 9.39, -32.48, 3.14
1.0, 130.27, 9.39, -32.48, 3.14
... (continue with more timesteps)
[/VEHICLE_TRAJECTORIES]

Follow this structure EXACTLY without deviations. Each vehicle trajectory must have at least 10 timesteps with 0.5 second intervals.
"""