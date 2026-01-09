import streamlit as st
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="centered")
st.title("Multi-Agent AI Framework")

st.markdown("""
### Getting Started

1. **Configure API Key**: Go to Settings → General and enter your Google Gemini API key
2. **Start with Hypothesis**: Navigate to the Hypothesis page and enter your research question
3. **Refine Your Thinking**: Select from generated options and iterate on your reasoning
4. **Generate Hypothesis**: Once satisfied, generate your hypothesis
5. **Plan Experiments**: Use the Experiment Agent to create experimental protocols
6. **Analyze Data**: Upload data files to the Curve Fitting Agent for analysis
7. **Review History**: Check the History page to review all interactions and export data

#### Navigation

- **Home**: This introduction page
- **Dashboard**: System analytics, performance metrics, and agent usage statistics
- **Hypothesis**: Interactive hypothesis generation and refinement
- **Experiment**: Experimental planning and protocol generation
- **Curve Fitting**: Data analysis and curve fitting
- **Settings**: Configuration for API keys, workflows, and agent parameters
- **History**: Complete interaction history with per-agent tabs and export options
- **Export Data**: Download your interactions and results for documentation and sharing

---

### System Overview

This application consists of several interconnected agents that work together to support the research process:

#### Hypothesis Agent
The Hypothesis Agent uses Socratic questioning and tree-of-thought (TOT) reasoning to help you:
- **Clarify your research questions** through AI-powered question refinement
- **Explore multiple lines of reasoning** by generating distinct thought paths
- **Synthesize hypotheses** from your selected reasoning paths
- **Analyze and evaluate** your hypotheses for scientific rigor

**Workflow:**
1. Enter your initial research question
2. Review the AI-clarified question and probing questions
3. Select from three distinct lines of thought
4. Iteratively refine your thinking through continuation options
5. Generate a comprehensive hypothesis
6. Receive analysis and evaluation of your hypothesis

#### Experiment Agent
The Experiment Agent transforms your hypotheses into actionable experimental plans:
- **Generates experimental protocols** based on your hypothesis and constraints
- **Creates worklists** for automated liquid handling systems (Opentrons, Tecan)
- **Designs plate layouts** for multi-well experiments
- **Produces executable protocols** for robotic systems

**Features:**
- Configurable experimental constraints (techniques, equipment, parameters)
- Support for multiple plate formats (96-well, 384-well, 24-well)
- Custom material and parameter specifications
- Integration with Jupyter notebooks for protocol execution

#### Curve Fitting Agent
The Curve Fitting Agent analyzes experimental data:
- **Processes luminescence data** from CSV files
- **Fits multi-peak Gaussian models** to spectral data
- **Generates visualization plots** of data and fits
- **Provides quantitative analysis** with R² scores and peak parameters

**Capabilities:**
- Automatic peak detection and parameter optimization
- Configurable wavelength ranges and read numbers
- Support for multiple wells and time-resolved measurements
- Export of fitting results and visualizations

#### Router Agent
The Router Agent intelligently manages workflow transitions:
- **Confidence-based routing** selects the most appropriate agent based on context
- **LLM-based routing** uses AI analysis to determine next steps
- **Manual workflow control** allows you to define custom agent sequences
- **Automatic progression** through hypothesis → experiment → analysis workflows

#### Watcher Agent
The Watcher Agent monitors your filesystem for changes:
- **Detects new output files** from agent executions
- **Triggers next agents** automatically when files are created
- **Supervises workflows** by monitoring file system events
- **Cross-platform support** (Windows, macOS, Linux)
""")
