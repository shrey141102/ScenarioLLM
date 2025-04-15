## Step 1: Prepare the LLM Prompting Framework

1. **Create a standardized prompt template** based on the original LLMScenario paper but adapted for newer models:
   - Extract relevant scenario information from your pre-processed data
   - Format it into a prompt that includes:
     - Road environment description
     - Vehicle states and trajectories
     - Tasks and interactions

2. **Set up API connections** to the three LLMs you want to test:
   - GPT-4 via OpenAI API
   - Claude 3.7 via Anthropic API
   - Gemini 2 Flash via Google AI API

## Step 2: Generate Scenarios Using Multiple LLMs

1. **Select representative seed scenarios** from your preprocessed data (5-10 diverse scenarios)

2. **Generate variations** using each LLM:
   - Feed the seed scenarios to each model
   - Request each model to generate challenging variations
   - Save the outputs in a structured format that captures:
     - Vehicle trajectories
     - Interaction descriptions
     - LLM's reasoning process

3. **Process the results** into a common format for comparison:
   - Extract key trajectory data
   - Format vehicle states and behaviors consistently
   - Prepare metadata for visualization

## Step 3: Implement the SUMO Visualization Pipeline

1. **Create a converter** to transform LLM-generated scenarios into SUMO format:
   - Write a script to convert vehicle trajectories to SUMO XML format
   - Handle road geometry and lane information
   - Set up visualization parameters

2. **Build a visualization workflow**:
   - Create a SUMO configuration file for your scenarios
   - Configure visualization settings (camera angle, colors, etc.)
   - Implement recording capabilities to save animations

3. **Test the pipeline** with one sample scenario before scaling up

## Step 4: Implement Agent-Based Enhancement with CrewAI

1. **Define agent roles and behaviors**:
   - Create different driver profiles (aggressive, cautious, etc.)
   - Define how agents make decisions in different scenarios

2. **Integrate CrewAI with your scenario generation**:
   - Have the LLM generate scenarios with specific agent behaviors
   - Allow agents to interact and influence scenario outcomes

3. **Develop an interface** between CrewAI agents and SUMO visualization

## Step 5: Conduct Comparative Evaluations

1. **Define evaluation metrics**:
   - Realism metrics (comparing to HighD statistics)
   - Diversity metrics (scenario types, interaction patterns)
   - Complexity metrics (number of interactions, challenge level)

2. **Perform automated evaluations**:
   - Calculate statistics for each model's generated scenarios
   - Compare with ground truth data from HighD
   - Measure differences between the three LLMs

3. **Conduct human evaluations** (if possible):
   - Have domain experts rate scenario realism
   - Compare ratings across models

## Step 6: Perform Ablation Studies

1. **Design your ablation experiments**:
   - Test different LLMs independently
   - Remove CrewAI agent enhancement
   - Vary prompt strategies

2. **Run experiments systematically**:
   - Generate scenarios under each condition
   - Apply your evaluation metrics
   - Document changes in performance

## Step 7: Analyze Results and Document Findings

1. **Compile all results** from your evaluations and ablation studies

2. **Analyze the differences** between:
   - Original LLMScenario approach and your enhanced version
   - Different LLM models
   - With and without agent-based enhancements

3. **Document your findings** in paper format:
   - Methodology
   - Implementation details
   - Results and analysis
   - Visualizations and examples
