# # """
# # Main module for running the LLMScenario enhanced framework.
# # """
# #
# # import os
# # import pandas as pd
# # import time
# # from utils.dev_tools import compare_llm_scenarios
# # from models.llm_interface import LLMInterface
# # from utils.data_processor import DataProcessor
# # from visualization.matplotlib_visualizer import MatplotlibVisualizer
# # from visualization.utils import extract_scenario_description, extract_enhancement_strategy, extract_vehicle_trajectories
# #
# #
# # def main():
# #     print("LLMScenario Enhanced Framework")
# #     print("==============================")
# #
# #     # Load pre-processed data
# #     print("Loading pre-processed data...")
# #     try:
# #         tracks_df = pd.read_csv('data/processed/tracks1.csv')
# #         print(f"Loaded {len(tracks_df)} records from tracks1.csv")
# #     except Exception as e:
# #         print(f"Error loading data: {e}")
# #         return
# #
# #     # Initialize LLM interface
# #     llm_interface = LLMInterface()
# #
# #     # Extract scenario information
# #     print("Extracting scenario information...")
# #     road_env, vehicle_states, tasks_interactions = DataProcessor.extract_scenario_from_tracks(tracks_df)
# #
# #     print("\nRoad Environment:")
# #     print("----------------")
# #     print(road_env)
# #
# #     print("\nVehicle States:")
# #     print("--------------")
# #     print(vehicle_states)
# #
# #     print("\nTasks and Interactions:")
# #     print("----------------------")
# #     print(tasks_interactions)
# #
# #     # Generate scenarios with each LLM
# #     generate_new_scenarios = False  # Set to False if you already have generated scenarios
# #
# #     if generate_new_scenarios:
# #         print("\nGenerating scenarios with different LLMs...")
# #
# #         # Generate with GPT-4
# #         print("\nGenerating with GPT-4...")
# #         try:
# #             gpt4_scenario = llm_interface.generate_with_gpt4(road_env, vehicle_states, tasks_interactions)
# #             print("GPT-4 scenario generated successfully!")
# #             with open('output/gpt4_scenario.txt', 'w') as f:
# #                 f.write(gpt4_scenario)
# #         except Exception as e:
# #             print(f"Error generating GPT-4 scenario: {e}")
# #
# #         # Add a delay to avoid rate limiting
# #         time.sleep(2)
# #
# #         # Generate with Claude
# #         print("\nGenerating with Claude 3.7...")
# #         try:
# #             claude_scenario = llm_interface.generate_with_claude(road_env, vehicle_states, tasks_interactions)
# #             print("Claude 3.7 scenario generated successfully!")
# #             with open('output/claude_scenario.txt', 'w') as f:
# #                 f.write(claude_scenario)
# #         except Exception as e:
# #             print(f"Error generating Claude scenario: {e}")
# #
# #         # Add a delay to avoid rate limiting
# #         time.sleep(2)
# #
# #         # Generate with Gemini
# #         print("\nGenerating with Gemini...")
# #         try:
# #             gemini_scenario = llm_interface.generate_with_gemini(road_env, vehicle_states, tasks_interactions)
# #             print("Gemini scenario generated successfully!")
# #             with open('output/gemini_scenario.txt', 'w') as f:
# #                 f.write(gemini_scenario)
# #         except Exception as e:
# #             print(f"Error generating Gemini scenario: {e}")
# #
# #     # Load generated scenario texts
# #     print("\nLoading generated scenarios...")
# #     generated_scenarios = {
# #         'gpt4': open('output/gpt4_scenario.txt', 'r').read() if os.path.exists('output/gpt4_scenario.txt') else None,
# #         'claude': open('output/claude_scenario.txt', 'r').read() if os.path.exists(
# #             'output/claude_scenario.txt') else None,
# #         'gemini': open('output/gemini_scenario.txt', 'r').read() if os.path.exists(
# #             'output/gemini_scenario.txt') else None
# #     }
# #
# #     # Initialize Matplotlib visualizer
# #     matplotlib_visualizer = MatplotlibVisualizer()
# #
# #     # Process each generated scenario
# #     for llm_type, scenario_text in generated_scenarios.items():
# #         if not scenario_text:
# #             print(f"\nNo {llm_type} scenario available")
# #             continue
# #
# #         print(f"\nProcessing {llm_type} scenario...")
# #
# #         # Extract scenario sections
# #         description = extract_scenario_description(scenario_text, llm_type)
# #         enhancement = extract_enhancement_strategy(scenario_text, llm_type)
# #         trajectories = extract_vehicle_trajectories(scenario_text, llm_type)
# #
# #         print(f"\nScenario Description ({llm_type}):")
# #         print("-" * (23 + len(llm_type)))
# #         print(description[:300] + "..." if len(description) > 300 else description)
# #
# #         print(f"\nEnhancement Strategy ({llm_type}):")
# #         print("-" * (23 + len(llm_type)))
# #         print(enhancement[:300] + "..." if len(enhancement) > 300 else enhancement)
# #
# #         print(f"\nTrajectories found: {bool(trajectories)}")
# #
# #         # Visualize using Matplotlib
# #         # Visualize using Matplotlib
# #         try:
# #             print(f"\nGenerating visualization for {llm_type} scenario...")
# #             try:
# #                 # Try animated visualization first
# #                 video_path = matplotlib_visualizer.visualize(scenario_text, f"{llm_type}_scenario", llm_type)
# #                 print(f"Video generated: {video_path}")
# #             except Exception as e:
# #                 print(f"Error with animated visualization: {e}")
# #                 print("Falling back to static visualization...")
# #                 # Fall back to static visualization
# #                 image_path = matplotlib_visualizer.generate_static_visualization(scenario_text, f"{llm_type}_scenario",
# #                                                                                  llm_type)
# #                 print(f"Static image generated: {image_path}")
# #         except Exception as e:
# #             print(f"Error visualizing {llm_type} scenario: {e}")
# #
# #     print("\nPerforming comparisons between LLMs...")
# #     # After generating scenarios
# #     compare_llm_scenarios(generated_scenarios)
# #     # Add code to compare outputs between different LLMs
# #     # For example:
# #     # - Compare the number of vehicles generated
# #     # - Compare the complexity of interactions
# #     # - Compare the types of challenges introduced
# #
# #     print("\nDone!")
# #
# #
# # if __name__ == "__main__":
# #     # Create output directories if they don't exist
# #     os.makedirs('output', exist_ok=True)
# #     os.makedirs('output/videos', exist_ok=True)
# #     main()
# #
#
#
# # ----------
# """
# Main module for running the LLMScenario enhanced framework.
# """
#
# import os
# import pandas as pd
# from models.llm_interface import LLMInterface
# from utils.data_processor import DataProcessor
# from visualization.matplotlib_visualizer import MatplotlibVisualizer
# from visualization.utils import extract_scenario_description, extract_enhancement_strategy
# from agents import AgentManager
#
#
# def main():
#     print("LLMScenario Enhanced Framework")
#     print("==============================")
#
#     # Load pre-processed data
#     print("Loading pre-processed data...")
#     try:
#         tracks_df = pd.read_csv('data/processed/tracks1.csv')
#         print(f"Loaded {len(tracks_df)} records from tracks1.csv")
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return
#
#     # Initialize LLM interface
#     llm_interface = LLMInterface()
#
#     # Extract scenario information
#     print("Extracting scenario information...")
#     road_env, vehicle_states, tasks_interactions = DataProcessor.extract_scenario_from_tracks(tracks_df)
#
#     print("\nRoad Environment:")
#     print("----------------")
#     print(road_env)
#
#     print("\nVehicle States:")
#     print("--------------")
#     print(vehicle_states)
#
#     print("\nTasks and Interactions:")
#     print("----------------------")
#     print(tasks_interactions)
#
#     # Load generated scenario texts
#     generated_scenarios = {
#         'gpt4': open('output/gpt4_scenario.txt', 'r').read() if os.path.exists('output/gpt4_scenario.txt') else None,
#         'claude': open('output/claude_scenario.txt', 'r').read() if os.path.exists(
#             'output/claude_scenario.txt') else None,
#         'gemini': open('output/gemini_scenario.txt', 'r').read() if os.path.exists(
#             'output/gemini_scenario.txt') else None
#     }
#
#     # Initialize Matplotlib visualizer
#     matplotlib_visualizer = MatplotlibVisualizer()
#
#     # Process each generated scenario
#     for llm_type, scenario_text in generated_scenarios.items():
#         if not scenario_text:
#             print(f"\nNo {llm_type} scenario available")
#             continue
#
#         print(f"\nProcessing {llm_type} scenario...")
#
#         # Extract scenario description and enhancement strategy
#         description = extract_scenario_description(scenario_text, llm_type)
#         enhancement = extract_enhancement_strategy(scenario_text, llm_type)
#
#         print(f"\nScenario Description ({llm_type}):")
#         print("-" * (23 + len(llm_type)))
#         print(description[:300] + "..." if len(description) > 300 else description)
#
#         print(f"\nEnhancement Strategy ({llm_type}):")
#         print("-" * (23 + len(llm_type)))
#         print(enhancement[:300] + "..." if len(enhancement) > 300 else enhancement)
#
#         # Visualize using Matplotlib
#         try:
#             print(f"\nGenerating visualization for {llm_type} scenario...")
#             video_path = matplotlib_visualizer.visualize(scenario_text, f"{llm_type}_scenario")
#             if video_path:
#                 print(f"Video generated: {video_path}")
#         except Exception as e:
#             print(f"Error visualizing {llm_type} scenario: {e}")
#
#     # Initialize Agent Manager
#     print("\nInitializing Agent-Based Enhancement...")
#     agent_manager = AgentManager()
#
#     # Select the best scenario for agent enhancement
#     best_scenario = "claude"  # You can use a heuristic to determine the best one
#     best_description = extract_scenario_description(generated_scenarios[best_scenario], best_scenario)
#
#     # Enhance the scenario with agent reactions
#     print(f"\nEnhancing {best_scenario} scenario with agent reactions...")
#
#     try:
#         # Create an example event
#         event_description = "A vehicle ahead suddenly brakes hard, reducing speed by 50% in 2 seconds."
#
#         # Get agent reactions
#         reactions = agent_manager.react_to_scenario_event(best_description, event_description)
#
#         with open(f"output/{best_scenario}_agent_reactions.txt", "w") as f:
#             f.write(f"Event: {event_description}\n\n")
#             for i, reaction in enumerate(reactions):
#                 f.write(f"Agent {i + 1} Reaction:\n{reaction}\n\n")
#
#         print(f"Agent reactions saved to output/{best_scenario}_agent_reactions.txt")
#
#         # Optionally, populate the scenario with different driver types
#         print("\nPopulating scenario with different driver agents...")
#         agent_trajectories = agent_manager.populate_scenario_with_agents(best_description)
#
#         with open(f"output/{best_scenario}_agent_trajectories.txt", "w") as f:
#             for i, trajectory in enumerate(agent_trajectories):
#                 f.write(f"Agent {i + 1} Trajectory:\n{trajectory}\n\n")
#
#         print(f"Agent trajectories saved to output/{best_scenario}_agent_trajectories.txt")
#
#     except Exception as e:
#         print(f"Error in agent-based enhancement: {e}")
#
#     print("\nDone!")
#
#
# if __name__ == "__main__":
#     os.makedirs('output', exist_ok=True)
#     os.makedirs('output/videos', exist_ok=True)
#     main()


"""
Main module for running the LLMScenario enhanced framework.
"""

import os
import pandas as pd
import time
from models.llm_interface import LLMInterface
from utils.data_processor import DataProcessor
from visualization.matplotlib_visualizer import MatplotlibVisualizer
from visualization.utils import extract_scenario_description, extract_enhancement_strategy, extract_vehicle_trajectories
from agents import AgentManager
from evalution import ScenarioEvaluator, AblationStudy
from utils.dev_tools import compare_llm_scenarios


def main():
    print("LLMScenario Enhanced Framework")
    print("==============================")

    # Configuration options to skip completed stages
    RUN_DATA_PROCESSING = True  # Process the HighD data
    RUN_LLM_GENERATION = True  # Generate new scenarios with LLMs
    RUN_VISUALIZATION = True  # Generate visualizations for scenarios
    RUN_AGENT_ENHANCEMENT = True  # Enhance scenarios with CrewAI agents
    RUN_EVALUATION = True  # Run evaluation metrics and reports

    # Load pre-processed data
    if RUN_DATA_PROCESSING:
        print("Loading pre-processed data...")
        try:
            tracks_df = pd.read_csv('data/processed/tracks1.csv')
            print(f"Loaded {len(tracks_df)} records from tracks1.csv")

            # Extract scenario information
            print("Extracting scenario information...")
            road_env, vehicle_states, tasks_interactions = DataProcessor.extract_scenario_from_tracks(tracks_df)

            print("\nRoad Environment:")
            print("----------------")
            print(road_env)

            print("\nVehicle States:")
            print("--------------")
            print(vehicle_states)

            print("\nTasks and Interactions:")
            print("----------------------")
            print(tasks_interactions)
        except Exception as e:
            print(f"Error loading or processing data: {e}")
            return
    else:
        print("Skipping data processing...")
        # Define these variables even when skipping, as they might be needed later
        road_env, vehicle_states, tasks_interactions = None, None, None

    # Initialize LLM interface
    llm_interface = LLMInterface()

    # Generate scenarios with each LLM
    if RUN_LLM_GENERATION:
        print("\nGenerating scenarios with different LLMs...")

        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)

        # Generate with GPT-4
        print("\nGenerating with GPT-4...")
        try:
            gpt4_scenario = llm_interface.generate_with_gpt4(road_env, vehicle_states, tasks_interactions)
            print("GPT-4 scenario generated successfully!")
            with open('output/gpt4_scenario.txt', 'w') as f:
                f.write(gpt4_scenario)
        except Exception as e:
            print(f"Error generating GPT-4 scenario: {e}")

        # Add a delay to avoid rate limiting
        time.sleep(2)

        # Generate with Claude
        print("\nGenerating with Claude 3.7...")
        try:
            claude_scenario = llm_interface.generate_with_claude(road_env, vehicle_states, tasks_interactions)
            print("Claude 3.7 scenario generated successfully!")
            with open('output/claude_scenario.txt', 'w') as f:
                f.write(claude_scenario)
        except Exception as e:
            print(f"Error generating Claude scenario: {e}")

        # Add a delay to avoid rate limiting
        time.sleep(2)

        # Generate with Gemini
        print("\nGenerating with Gemini...")
        try:
            gemini_scenario = llm_interface.generate_with_gemini(road_env, vehicle_states, tasks_interactions)
            print("Gemini scenario generated successfully!")
            with open('output/gemini_scenario.txt', 'w') as f:
                f.write(gemini_scenario)
        except Exception as e:
            print(f"Error generating Gemini scenario: {e}")
    else:
        print("\nSkipping LLM scenario generation...")

    # Load generated scenario texts
    print("\nLoading generated scenarios...")
    generated_scenarios = {
        'gpt4': open('output/gpt4_scenario.txt', 'r').read() if os.path.exists('output/gpt4_scenario.txt') else None,
        'claude': open('output/claude_scenario.txt', 'r').read() if os.path.exists(
            'output/claude_scenario.txt') else None,
        'gemini': open('output/gemini_scenario.txt', 'r').read() if os.path.exists(
            'output/gemini_scenario.txt') else None
    }

    # Filter out None values
    generated_scenarios = {k: v for k, v in generated_scenarios.items() if v is not None}

    if not generated_scenarios:
        print("No generated scenarios found. Please run LLM generation first.")
        return

    # Process and visualize scenarios
    if RUN_VISUALIZATION:
        print("\nVisualizing generated scenarios...")
        # Initialize Matplotlib visualizer
        matplotlib_visualizer = MatplotlibVisualizer()

        # Ensure visualization output directories exist
        os.makedirs('output/videos', exist_ok=True)

        # Process each generated scenario
        for llm_type, scenario_text in generated_scenarios.items():
            print(f"\nProcessing {llm_type} scenario...")

            # Extract scenario sections
            description = extract_scenario_description(scenario_text, llm_type)
            enhancement = extract_enhancement_strategy(scenario_text, llm_type)
            trajectories = extract_vehicle_trajectories(scenario_text, llm_type)

            print(f"\nScenario Description ({llm_type}):")
            print("-" * (23 + len(llm_type)))
            print(description[:300] + "..." if len(description) > 300 else description)

            print(f"\nEnhancement Strategy ({llm_type}):")
            print("-" * (23 + len(llm_type)))
            print(enhancement[:300] + "..." if len(enhancement) > 300 else enhancement)

            print(f"\nTrajectories found: {bool(trajectories)}")

            # Visualize using Matplotlib
            try:
                print(f"\nGenerating visualization for {llm_type} scenario...")
                try:
                    # Try animated visualization first
                    video_path = matplotlib_visualizer.visualize(scenario_text, f"{llm_type}_scenario", llm_type)
                    print(f"Video generated: {video_path}")
                except Exception as e:
                    print(f"Error with animated visualization: {e}")
                    print("Falling back to static visualization...")
                    # Fall back to static visualization
                    image_path = matplotlib_visualizer.generate_static_visualization(
                        scenario_text, f"{llm_type}_scenario", llm_type)
                    print(f"Static image generated: {image_path}")
            except Exception as e:
                print(f"Error visualizing {llm_type} scenario: {e}")
    else:
        print("\nSkipping visualization...")

    # Agent-based enhancement
    if RUN_AGENT_ENHANCEMENT:
        print("\nRunning agent-based scenario enhancement...")

        # Initialize Agent Manager
        agent_manager = AgentManager()

        # Select the best scenario for agent enhancement
        best_scenario = list(generated_scenarios.keys())[0]  # Default to first available
        for scenario_type in ['claude', 'gpt4', 'gemini']:  # Prioritize Claude if available
            if scenario_type in generated_scenarios:
                best_scenario = scenario_type
                break

        best_description = extract_scenario_description(generated_scenarios[best_scenario], best_scenario)

        # Enhance the scenario with agent reactions
        print(f"\nEnhancing {best_scenario} scenario with agent reactions...")

        try:
            # Create an example event
            event_description = "A vehicle ahead suddenly brakes hard, reducing speed by 50% in 2 seconds."

            # Get agent reactions
            reactions = agent_manager.react_to_scenario_event(best_description, event_description)

            with open(f"output/{best_scenario}_agent_reactions.txt", "w") as f:
                f.write(f"Event: {event_description}\n\n")
                for i, reaction in enumerate(reactions):
                    f.write(f"Agent {i + 1} Reaction:\n{reaction}\n\n")

            print(f"Agent reactions saved to output/{best_scenario}_agent_reactions.txt")

            # Optionally, populate the scenario with different driver types
            print("\nPopulating scenario with different driver agents...")
            agent_trajectories = agent_manager.populate_scenario_with_agents(best_description)

            with open(f"output/{best_scenario}_agent_trajectories.txt", "w") as f:
                for i, trajectory in enumerate(agent_trajectories):
                    f.write(f"Agent {i + 1} Trajectory:\n{trajectory}\n\n")

            print(f"Agent trajectories saved to output/{best_scenario}_agent_trajectories.txt")

        except Exception as e:
            print(f"Error in agent-based enhancement: {e}")
    else:
        print("\nSkipping agent-based enhancement...")

    # Evaluation and comparative analysis
    if RUN_EVALUATION:
        print("\nRunning evaluation and comparative analysis...")

        # Ensure evaluation output directory exists
        os.makedirs('output/evaluation', exist_ok=True)

        # Get agent reactions if available
        agent_reactions = {}
        if os.path.exists('output/claude_agent_reactions.txt'):
            with open('output/claude_agent_reactions.txt', 'r') as f:
                agent_reactions['claude'] = f.read()

        agent_trajectories = {}
        if os.path.exists('output/claude_agent_trajectories.txt'):
            with open('output/claude_agent_trajectories.txt', 'r') as f:
                agent_trajectories['claude'] = f.read()

        # Initialize evaluation
        print("\nConducting comparative evaluations...")
        evaluator = ScenarioEvaluator(output_dir='output/evaluation')

        # Evaluate scenarios
        evaluation_results = []
        for llm_type, scenario_text in generated_scenarios.items():
            result = evaluator.evaluate_scenario(scenario_text, llm_type)
            evaluation_results.append(result)
            print(f"Evaluated {llm_type} scenario:")
            for metric, value in result.items():
                if metric != 'scenario_name':
                    print(f"  - {metric.replace('_', ' ').title()}: {value:.2f}")

        # Generate comparison charts
        if evaluation_results:
            chart_path = evaluator.generate_comparison_charts(evaluation_results)
            print(f"Comparison chart generated: {chart_path}")

            report_path = evaluator.generate_detailed_report(evaluation_results)
            print(f"Detailed evaluation report generated: {report_path}")

            # Generate LLMScenario comparison report
            llmscenario_report_path = evaluator.generate_llmscenario_comparison_report(evaluation_results)
            print(f"LLMScenario comparison report generated: {llmscenario_report_path}")

        # Run inter-LLM comparison
        print("\nPerforming detailed comparisons between LLMs...")
        compare_llm_scenarios(generated_scenarios)

        # Conduct ablation study
        print("\nConducting ablation study...")
        ablation = AblationStudy(output_dir='output/evaluation')

        # Compare LLM models
        llm_comparison = ablation.compare_llm_models(generated_scenarios)

        # Compare agent enhancement (if available)
        agent_comparison = None
        if 'claude' in generated_scenarios and 'claude' in agent_trajectories:
            print("Comparing baseline vs agent-enhanced scenarios...")
            agent_comparison = ablation.compare_agent_enhancement(
                generated_scenarios['claude'],
                agent_trajectories['claude']
            )

        # Generate ablation charts
        if llm_comparison:
            chart_path = ablation.generate_ablation_charts(llm_comparison, agent_comparison)
            print(f"Ablation study chart generated: {chart_path}")

        # Analyze agent behavior limitations
        if agent_trajectories:
            print("\nAnalyzing agent behavior limitations...")
            print("The current agent implementation has the following limitations:")
            print("1. Agents generate philosophical descriptions instead of trajectory data")
            print("2. The output format does not match the requested format")
            print("3. The agent responses lack specific scenario-related details")
            print("These limitations highlight the challenges of using CrewAI for structured outputs")

            # Write limitations to file
            limitations_path = os.path.join('output/evaluation', 'agent_limitations.txt')
            with open(limitations_path, 'w') as f:
                f.write("# Agent Implementation Limitations\n\n")
                f.write("The current agent implementation has the following limitations:\n\n")
                f.write(
                    "1. **Format Mismatch**: Agents generate philosophical descriptions instead of trajectory data\n")
                f.write(
                    "2. **Task Misinterpretation**: Agents focus on their role definition rather than the specific task\n")
                f.write(
                    "3. **Lack of Structure**: The output lacks the precise structure needed for scenario visualization\n")
                f.write(
                    "4. **Abstract vs. Concrete**: Agents provide abstract driving philosophy rather than concrete coordinates\n\n")
                f.write("## Potential Improvements\n\n")
                f.write(
                    "1. Modify the CrewAI task prompt to explicitly request trajectory data in the required format\n")
                f.write("2. Use more structured prompts with clearer formatting examples\n")
                f.write("3. Implement a post-processing step to convert agent outputs to valid trajectory formats\n")
                f.write(
                    "4. Consider a hybrid approach where LLMs generate trajectories and agents enhance them with behavioral nuances\n")
    else:
        print("\nSkipping evaluation and comparative analysis...")

    print("\nAll requested stages completed!")


if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs('output', exist_ok=True)
    main()