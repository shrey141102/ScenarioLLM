import re
from typing import Dict, Any, List


class LLMScenarioComparator:
    def __init__(self, scenario_texts: Dict[str, str]):
        """
        Initialize the comparator with scenario texts from different LLMs

        :param scenario_texts: Dictionary with LLM names as keys and scenario texts as values
        """
        self.scenario_texts = scenario_texts
        self.comparison_results = {}

    def count_vehicles(self) -> Dict[str, int]:
        """
        Count the number and types of vehicles in each scenario

        :return: Dictionary with LLM names and total vehicle counts
        """
        vehicle_counts = {}
        for llm, scenario in self.scenario_texts.items():
            # More sophisticated vehicle identification
            car_count = len(re.findall(r'\b(Car|car|passenger vehicle|passenger car)\b', scenario, re.IGNORECASE))
            truck_count = len(re.findall(r'\b(Truck|truck|commercial vehicle)\b', scenario, re.IGNORECASE))

            # Total vehicle count
            total_vehicles = car_count + truck_count

            # Additional details
            vehicle_info = {
                'total_vehicles': total_vehicles,
                'cars': car_count,
                'trucks': truck_count
            }

            vehicle_counts[llm] = vehicle_info

        self.comparison_results['vehicle_counts'] = vehicle_counts
        return {llm: info['total_vehicles'] for llm, info in vehicle_counts.items()}

    def analyze_interaction_complexity(self) -> Dict[str, float]:
        """
        Analyze the complexity of interactions in each scenario

        Scoring based on:
        - Number of unique interaction types
        - Complexity of described interactions

        :return: Dictionary with LLM names and interaction complexity scores
        """
        interaction_complexity = {}
        interaction_patterns = [
            r'(lane change|overtake|merge|conflict|cooperative|competitive)',
            r'(unexpected|challenging|risky|complex)',
            r'(multiple vehicles|simultaneous interactions)'
        ]

        for llm, scenario in self.scenario_texts.items():
            # Convert to lowercase for case-insensitive matching
            scenario_lower = scenario.lower()

            # Calculate complexity score
            complexity_score = 0
            for pattern in interaction_patterns:
                matches = re.findall(pattern, scenario_lower)
                complexity_score += len(matches)

            interaction_complexity[llm] = complexity_score

        self.comparison_results['interaction_complexity'] = interaction_complexity
        return interaction_complexity

    def identify_challenge_types(self) -> Dict[str, List[str]]:
        """
        Identify and categorize challenges introduced in each scenario

        :return: Dictionary with LLM names and lists of challenge types
        """
        challenge_categories = {
            'geometric': [
                r'(curved road|intersection|merge|diverge|complex geometry)',
                r'(road curvature|road shape|road topology)'
            ],
            'behavioral': [
                r'(unexpected maneuver|aggressive driving|hesitation|unpredictable)',
                r'(sudden brake|rapid acceleration|erratic movement)'
            ],
            'environmental': [
                r'(adverse weather|low visibility|slippery surface|challenging conditions)',
                r'(rain|fog|ice|wind)'
            ],
            'interaction': [
                r'(right-of-way conflict|close proximity|potential collision)',
                r'(vehicle interference|competing objectives)'
            ]
        }

        challenge_types = {}

        for llm, scenario in self.scenario_texts.items():
            # Convert to lowercase for case-insensitive matching
            scenario_lower = scenario.lower()

            # Identify challenges
            llm_challenges = []
            for category, patterns in challenge_categories.items():
                for pattern in patterns:
                    if re.search(pattern, scenario_lower):
                        llm_challenges.append(category)
                        break  # Add category only once

            challenge_types[llm] = list(set(llm_challenges))

        self.comparison_results['challenge_types'] = challenge_types
        return challenge_types

    def generate_comparative_report(self) -> str:
        """
        Generate a comprehensive comparative report of scenario generations

        :return: Formatted report string
        """
        # Ensure all analyses have been run
        if not self.comparison_results:
            self.count_vehicles()
            self.analyze_interaction_complexity()
            self.identify_challenge_types()

        report = "LLM Scenario Generation Comparative Report\n"
        report += "=======================================\n\n"

        # Vehicle Count Comparison
        report += "1. Vehicle Count Comparison:\n"
        for llm, counts in self.comparison_results['vehicle_counts'].items():
            report += f"   - {llm}:\n"
            report += f"     * Total Vehicles: {counts['total_vehicles']}\n"
            report += f"     * Cars: {counts['cars']}\n"
            report += f"     * Trucks: {counts['trucks']}\n"

        # Interaction Complexity
        report += "\n2. Interaction Complexity:\n"
        for llm, complexity in self.comparison_results['interaction_complexity'].items():
            report += f"   - {llm}: Complexity Score = {complexity}\n"

        # Challenge Types
        report += "\n3. Challenge Types:\n"
        for llm, challenges in self.comparison_results['challenge_types'].items():
            report += f"   - {llm}: {', '.join(challenges) if challenges else 'No specific challenges identified'}\n"

        # Comparative Analysis
        report += "\n4. Comparative Insights:\n"

        # Vehicle Comparison
        vehicle_counts = self.comparison_results['vehicle_counts']
        max_vehicles = max(
            counts['total_vehicles'] for counts in vehicle_counts.values()
        )
        llms_with_max = [
            llm for llm, counts in vehicle_counts.items()
            if counts['total_vehicles'] == max_vehicles
        ]
        report += f"   - Most comprehensive vehicle representation: {', '.join(llms_with_max)}\n"

        # Complexity Comparison
        complexity_scores = self.comparison_results['interaction_complexity']
        max_complexity = max(complexity_scores.values())
        llms_with_max_complexity = [
            llm for llm, complexity in complexity_scores.items()
            if complexity == max_complexity
        ]
        report += f"   - Highest interaction complexity: {', '.join(llms_with_max_complexity)}\n"

        # Challenge Diversity
        challenge_types = self.comparison_results['challenge_types']
        challenge_diversity = {
            llm: len(challenges) for llm, challenges in challenge_types.items()
        }
        max_diversity = max(challenge_diversity.values())
        llms_with_max_diversity = [
            llm for llm, diversity in challenge_diversity.items()
            if diversity == max_diversity
        ]
        report += f"   - Most diverse challenge types: {', '.join(llms_with_max_diversity)}\n"

        return report


def compare_llm_scenarios(generated_scenarios: Dict[str, str]) -> None:
    """
    Main function to compare scenarios generated by different LLMs

    :param generated_scenarios: Dictionary of scenarios from different LLMs
    """
    # Filter out None or empty scenarios
    valid_scenarios = {k: v for k, v in generated_scenarios.items() if v}

    if not valid_scenarios:
        print("No valid scenarios to compare.")
        return

    # Create comparator
    comparator = LLMScenarioComparator(valid_scenarios)

    # Generate and print comparative report
    comparative_report = comparator.generate_comparative_report()
    print(comparative_report)

    # Optionally, save the report to a file
    with open('output/llm_scenario_comparison_report.txt', 'w') as f:
        f.write(comparative_report)