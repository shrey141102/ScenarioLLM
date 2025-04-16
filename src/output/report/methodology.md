# Methodology

Our research builds upon the LLMScenario framework, extending it with three key improvements:

## 1. Advanced LLM Integration

We compared three state-of-the-art large language models:
- **GPT-4**: OpenAI's latest model with enhanced reasoning capabilities
- **Claude 3.7**: Anthropic's model with detailed instruction following
- **Gemini 2 Flash**: Google's efficient model with strong factual accuracy

Each model received the same prompts structured with:
- Road environment descriptions
- Vehicle states and trajectories
- Tasks and interaction information

## 2. Visualization Framework

We implemented a comprehensive visualization system using:
- **Matplotlib**: For trajectory visualization and animation
- **Custom rendering pipeline**: For vehicle representation with dynamic positions
- **Static and animated outputs**: To support different analysis needs

## 3. Agent-Based Enhancement Approach

We explored enhancing scenarios using a multi-agent framework:
- **Agent roles**: Different driver types (conservative, aggressive, etc.)
- **Scenario reactions**: Agent responses to critical events
- **Limitations analysis**: Evaluation of the agent-based approach challenges

## Dataset

We utilized the HighD dataset, which contains:
- **Real-world highway driving data**: Recorded from an aerial perspective
- **Vehicle trajectories**: Position, velocity, and acceleration data
- **Lane change and interaction information**: For realistic scenario modeling

## Evaluation Metrics

Our evaluation framework assessed scenarios based on:
- **Reality Score**: Measuring physical realism and collision avoidance
- **Rarity Score**: Evaluating complexity and uniqueness
- **Trajectory Completeness**: Assessing the completeness of trajectory data
- **Vehicle Interactions**: Analyzing the number and types of interactions

## Ablation Studies

We conducted ablation studies to isolate the contribution of:
- Different LLM models
- Agent-based enhancements
- Visualization components
