"""
Ablation study for scenario generation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import ScenarioMetrics


class AblationStudy:
    """Ablation study for scenario generation."""

    def __init__(self, output_dir='output/evaluation'):
        """
        Initialize the ablation study.

        Args:
            output_dir (str): Directory to save ablation outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def compare_llm_models(self, generated_scenarios):
        """
        Compare the performance of different LLM models.

        Args:
            generated_scenarios (dict): Dictionary of scenario texts by LLM type

        Returns:
            dict: Comparison results
        """
        results = {}

        for llm_type, scenario_text in generated_scenarios.items():
            if scenario_text:
                results[llm_type] = {
                    'vehicle_count': ScenarioMetrics.count_vehicles(scenario_text),
                    'interaction_count': ScenarioMetrics.count_interactions(scenario_text),
                    'lane_change_count': ScenarioMetrics.count_lane_changes(scenario_text),
                    'trajectory_completeness': ScenarioMetrics.trajectory_completeness(scenario_text),
                    'physical_realism': ScenarioMetrics.physical_realism(scenario_text),
                    'complexity': ScenarioMetrics.scenario_complexity(scenario_text)
                }

        return results

    def compare_agent_enhancement(self, baseline_scenario, agent_enhanced_scenario):
        """
        Compare baseline scenario with agent-enhanced scenario.

        Args:
            baseline_scenario (str): Baseline scenario text
            agent_enhanced_scenario (str): Agent-enhanced scenario text

        Returns:
            dict: Comparison results
        """
        baseline_metrics = {
            'vehicle_count': ScenarioMetrics.count_vehicles(baseline_scenario),
            'interaction_count': ScenarioMetrics.count_interactions(baseline_scenario),
            'lane_change_count': ScenarioMetrics.count_lane_changes(baseline_scenario),
            'trajectory_completeness': ScenarioMetrics.trajectory_completeness(baseline_scenario),
            'physical_realism': ScenarioMetrics.physical_realism(baseline_scenario),
            'complexity': ScenarioMetrics.scenario_complexity(baseline_scenario)
        }

        enhanced_metrics = {
            'vehicle_count': ScenarioMetrics.count_vehicles(agent_enhanced_scenario),
            'interaction_count': ScenarioMetrics.count_interactions(agent_enhanced_scenario),
            'lane_change_count': ScenarioMetrics.count_lane_changes(agent_enhanced_scenario),
            'trajectory_completeness': ScenarioMetrics.trajectory_completeness(agent_enhanced_scenario),
            'physical_realism': ScenarioMetrics.physical_realism(agent_enhanced_scenario),
            'complexity': ScenarioMetrics.scenario_complexity(agent_enhanced_scenario)
        }

        comparison = {
            'baseline': baseline_metrics,
            'agent_enhanced': enhanced_metrics,
            'differences': {}
        }

        # Calculate differences
        for metric in baseline_metrics:
            comparison['differences'][metric] = enhanced_metrics[metric] - baseline_metrics[metric]

        return comparison

    def analyze_agent_behavior(self, agent_trajectories):
        """
        Analyze the behavior patterns of different agent types.

        Args:
            agent_trajectories (dict): Dictionary of agent trajectories by agent type

        Returns:
            dict: Analysis results
        """
        results = {}

        for agent_type, trajectories in agent_trajectories.items():
            # Parse the trajectories
            parsed_trajectories = ScenarioMetrics.parse_trajectories(trajectories)

            if not parsed_trajectories:
                results[agent_type] = {
                    'status': 'No valid trajectory data found'
                }
                continue

            # Analyze driving behavior
            accelerations = []
            velocities = []
            lane_changes = []

            for vehicle_id, trajectory in parsed_trajectories.items():
                # Calculate accelerations
                vehicle_accels = ScenarioMetrics.calculate_acceleration(trajectory)
                accelerations.extend(vehicle_accels)

                # Extract velocities
                vehicle_velocities = [point['velocity'] for point in trajectory]
                velocities.extend(vehicle_velocities)

                # Estimate lane changes (based on y-coordinate changes)
                y_positions = [point['y'] for point in trajectory]
                for i in range(1, len(y_positions)):
                    if abs(y_positions[i] - y_positions[i - 1]) >= 3.0:  # Threshold for lane change
                        lane_changes.append(i)

            # Calculate statistics
            results[agent_type] = {
                'status': 'Valid trajectory data',
                'avg_velocity': np.mean(velocities) if velocities else 0,
                'max_velocity': max(velocities) if velocities else 0,
                'avg_acceleration': np.mean(accelerations) if accelerations else 0,
                'max_acceleration': max(accelerations) if accelerations else 0,
                'min_acceleration': min(accelerations) if accelerations else 0,  # Deceleration
                'lane_change_count': len(lane_changes)
            }

        return results

    def generate_ablation_charts(self, llm_comparison, agent_comparison):
        """
        Generate charts for ablation study.

        Args:
            llm_comparison (dict): LLM comparison results
            agent_comparison (dict): Agent comparison results

        Returns:
            str: Path to the saved chart
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot LLM comparison
        if llm_comparison:
            llm_names = list(llm_comparison.keys())
            complexity_values = [llm_comparison[llm]['complexity'] for llm in llm_names]
            realism_values = [llm_comparison[llm]['physical_realism'] for llm in llm_names]

            # Complexity by LLM
            axes[0, 0].bar(llm_names, complexity_values)
            axes[0, 0].set_title('Scenario Complexity by LLM')
            axes[0, 0].set_ylim(0, max(complexity_values) * 1.2 if complexity_values else 1)
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Realism by LLM
            axes[0, 1].bar(llm_names, realism_values)
            axes[0, 1].set_title('Physical Realism by LLM')
            axes[0, 1].set_ylim(0, max(realism_values) * 1.2 if realism_values else 1)
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot agent comparison
        if agent_comparison and 'differences' in agent_comparison:
            metrics = list(agent_comparison['differences'].keys())
            diff_values = [agent_comparison['differences'][metric] for metric in metrics]

            # Metric improvements
            bars = axes[1, 0].bar(metrics, diff_values)
            axes[1, 0].set_title('Improvement from Agent Enhancement')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Color bars based on positive/negative
            for i, bar in enumerate(bars):
                if diff_values[i] < 0:
                    bar.set_color('red')
                else:
                    bar.set_color('green')

            # Baseline vs enhanced
            if 'baseline' in agent_comparison and 'agent_enhanced' in agent_comparison:
                # Select key metrics
                key_metrics = ['complexity', 'physical_realism', 'trajectory_completeness']
                baseline_values = [agent_comparison['baseline'][metric] for metric in key_metrics]
                enhanced_values = [agent_comparison['agent_enhanced'][metric] for metric in key_metrics]

                x = np.arange(len(key_metrics))
                width = 0.35

                axes[1, 1].bar(x - width / 2, baseline_values, width, label='Baseline')
                axes[1, 1].bar(x + width / 2, enhanced_values, width, label='Agent Enhanced')
                axes[1, 1].set_title('Baseline vs Agent Enhanced')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in key_metrics])
                axes[1, 1].legend()

        # Adjust layout
        plt.tight_layout()

        # Save figure
        chart_path = os.path.join(self.output_dir, 'ablation_study.png')
        plt.savefig(chart_path)
        plt.close(fig)

        return chart_path