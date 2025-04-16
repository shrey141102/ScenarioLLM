"""
Evaluator for scenario generation outputs.
"""

from .metrics import ScenarioMetrics
import matplotlib.pyplot as plt
import numpy as np
import os


class ScenarioEvaluator:
    """Evaluator for scenario generation outputs."""

    def __init__(self, output_dir='output/evaluation'):
        """
        Initialize the scenario evaluator.

        Args:
            output_dir (str): Directory to save evaluation outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = ScenarioMetrics()

    def evaluate_scenario(self, scenario_text, scenario_name):
        """
        Evaluate a generated scenario.

        Args:
            scenario_text (str): The scenario text
            scenario_name (str): Name of the scenario

        Returns:
            dict: Evaluation metrics
        """
        results = {
            'scenario_name': scenario_name,
            'vehicle_count': ScenarioMetrics.count_vehicles(scenario_text),
            'interaction_count': ScenarioMetrics.count_interactions(scenario_text),
            'lane_change_count': ScenarioMetrics.count_lane_changes(scenario_text),
            'trajectory_completeness': ScenarioMetrics.trajectory_completeness(scenario_text),
            'physical_realism': ScenarioMetrics.physical_realism(scenario_text),
            'complexity': ScenarioMetrics.scenario_complexity(scenario_text)
        }

        llm_scenario_metrics = ScenarioMetrics.calculate_llmscenario_metrics(scenario_text)
        results.update(llm_scenario_metrics)

        return results

    def compare_scenarios(self, scenarios, scenario_names):
        """
        Compare multiple scenarios.

        Args:
            scenarios (list): List of scenario texts
            scenario_names (list): List of scenario names

        Returns:
            list: List of evaluation results
        """
        results = []

        for scenario_text, scenario_name in zip(scenarios, scenario_names):
            results.append(self.evaluate_scenario(scenario_text, scenario_name))

        return results

    def generate_comparison_charts(self, results):
        """
        Generate comparison charts for scenario evaluations.

        Args:
            results (list): List of evaluation results

        Returns:
            str: Path to the saved chart
        """
        # Extract data for charts
        scenario_names = [result['scenario_name'] for result in results]
        metrics = ['vehicle_count', 'interaction_count', 'lane_change_count',
                   'trajectory_completeness', 'physical_realism', 'complexity']

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Plot each metric
        for i, metric in enumerate(metrics):
            values = [result[metric] for result in results]
            axes[i].bar(scenario_names, values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylim(0, max(values) * 1.2 if values else 1)

            # Add values above bars
            for j, value in enumerate(values):
                axes[i].text(j, value, f'{value:.2f}', ha='center', va='bottom')

            # Rotate x-axis labels if needed
            axes[i].tick_params(axis='x', rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        chart_path = os.path.join(self.output_dir, 'scenario_comparison.png')
        plt.savefig(chart_path)
        plt.close(fig)

        return chart_path

    def generate_detailed_report(self, results):
        """
        Generate a detailed evaluation report.

        Args:
            results (list): List of evaluation results

        Returns:
            str: Path to the saved report
        """
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')

        with open(report_path, 'w') as f:
            f.write("# Scenario Evaluation Report\n\n")

            # Overall summary
            f.write("## Overall Summary\n\n")
            f.write("| Metric | " + " | ".join(result['scenario_name'] for result in results) + " |\n")
            f.write("|" + "-" * 10 + "|" + "".join("-" * 12 + "|" for _ in results) + "\n")

            metrics = ['vehicle_count', 'interaction_count', 'lane_change_count',
                       'trajectory_completeness', 'physical_realism', 'complexity']

            for metric in metrics:
                f.write("| " + metric.replace('_', ' ').title() + " | " +
                        " | ".join(f"{result[metric]:.2f}" for result in results) + " |\n")

            # Individual scenario details
            f.write("\n## Individual Scenario Details\n\n")

            for result in results:
                f.write(f"### {result['scenario_name']}\n\n")

                for metric, value in result.items():
                    if metric != 'scenario_name':
                        f.write(f"- {metric.replace('_', ' ').title()}: {value:.2f}\n")

                f.write("\n")

            # Conclusions
            f.write("## Conclusions\n\n")

            # Find best performing scenario for each metric
            for metric in metrics:
                if metric == 'vehicle_count':
                    continue  # Skip vehicle count as it's not a quality indicator

                best_value = max(result[metric] for result in results)
                best_scenarios = [result['scenario_name'] for result in results if result[metric] == best_value]

                f.write(f"- Best {metric.replace('_', ' ').title()}: {', '.join(best_scenarios)} ({best_value:.2f})\n")

            # Overall ranking
            f.write("\n### Overall Ranking\n\n")

            # Calculate overall score (weighted average of metrics)
            for result in results:
                result['overall_score'] = (
                        0.2 * result['trajectory_completeness'] +
                        0.3 * result['physical_realism'] +
                        0.5 * result['complexity']
                )

            # Sort by overall score
            sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)

            for i, result in enumerate(sorted_results):
                f.write(f"{i + 1}. {result['scenario_name']} (Score: {result['overall_score']:.2f})\n")

        return report_path

    # Add to ScenarioEvaluator class
    def generate_llmscenario_comparison_report(self, results):
        """
        Generate a report comparing results with LLMScenario paper metrics.

        Args:
            results (list): List of evaluation results

        Returns:
            str: Path to the saved report
        """
        report_path = os.path.join(self.output_dir, 'llmscenario_comparison.txt')

        with open(report_path, 'w') as f:
            f.write("# Comparison with LLMScenario Paper Metrics\n\n")

            f.write("## Original LLMScenario Metrics\n\n")
            f.write("The original LLMScenario paper used two primary metrics:\n\n")
            f.write("1. **Reality Score**: Evaluates if scenarios are realistic, checking for:\n")
            f.write("   - Vehicle collisions\n")
            f.write("   - Disobedience of traffic rules\n")
            f.write("   - Violation of vehicle dynamics constraints\n")
            f.write("   - Vehicles out of drivable area\n\n")

            f.write("2. **Rarity Score**: Measures how different generated scenarios are from:\n")
            f.write("   - Original prompt scenario\n")
            f.write("   - Normal safe scenarios\n")
            f.write("   - Previously generated scenarios\n\n")

            f.write("## Our Comparable Metrics\n\n")
            f.write("We've implemented similar metrics:\n\n")

            f.write("1. **Reality Score**: Evaluates physical realism by checking:\n")
            f.write("   - Realistic acceleration/deceleration\n")
            f.write("   - Absence of vehicle collisions\n")
            f.write("   - Proper trajectory continuity\n\n")

            f.write("2. **Rarity Score**: Evaluates complexity and uniqueness through:\n")
            f.write("   - Number of interactions\n")
            f.write("   - Lane change frequency\n")
            f.write("   - Overall scenario complexity\n\n")

            f.write("## Results Comparison\n\n")
            f.write("| Model | Reality Score | Rarity Score | Combined Score | Collisions |\n")
            f.write("|-------|--------------|--------------|----------------|------------|\n")

            for result in results:
                f.write(f"| {result['scenario_name']} | {result.get('reality_score', 0):.2f} | ")
                f.write(f"{result.get('rarity_score', 0):.2f} | {result.get('final_score', 0):.2f} | ")
                f.write(f"{result.get('collisions_detected', 0)} |\n")

            f.write("\n## Key Differences\n\n")
            f.write("Our metrics differ from the original paper in the following ways:\n\n")
            f.write("1. **Reality checking approach**: While the original paper uses a more comprehensive approach\n")
            f.write(
                "   with traffic rule checking, our implementation focuses on physical dynamics and collisions.\n\n")

            f.write("2. **Rarity calculation**: The original paper uses a distance metric between scenario graphs,\n")
            f.write("   while our approach uses complexity as a proxy for rarity.\n\n")

            f.write(
                "3. **Score combination**: The original uses a weighted linear combination of scores with thresholds,\n")
            f.write("   while our approach adds reality and rarity scores directly (when reality > 0).\n\n")

            f.write("## Conclusion\n\n")
            f.write("Our evaluation system captures the same core principles as the LLMScenario paper:\n")
            f.write("balancing realism with challenge/rarity. Given the implementation differences,\n")
            f.write("the absolute values are not directly comparable, but the relative rankings\n")
            f.write("between models should provide similar insights.\n")

        return report_path

