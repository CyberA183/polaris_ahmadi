# POLARIS Hypothesis Agent - Workflow Transcript

## Overview

The POLARIS Hypothesis Agent is an AI-powered research assistant that guides users through a structured process of developing scientific hypotheses. The system uses Socratic questioning, Tree of Thoughts (TOT) reasoning, and iterative refinement to help researchers formulate testable, scientifically grounded hypotheses.

---

## Workflow Stages

### Stage 1: Initial Question Entry (`initial`)

**Purpose:** User enters their research question to begin the hypothesis generation process.

**Process:**
1. User enters a research question in the chat interface
2. System processes the question through three sequential steps:
   - **Question Clarification:** The system analyzes the question to identify hidden assumptions, key terms, and scientific context. It rewrites the question to be precise and testable, explicitly listing assumptions that contain scientific reasoning, material properties, or mechanistic insights.
   - **Socratic Pass:** The system generates 3-5 probing questions that explore:
     - Scientific principles or mechanisms relevant to the question
     - Required material properties or structural features
     - Theoretical foundations or known examples from literature
     - Specific criteria or constraints for material/compound selection
   - **Tree of Thoughts (TOT) Generation:** The system produces exactly 3 distinct lines of thought, each structured as a concise mini-hypothesis or assumption (1-2 sentences) that:
     - Presents a specific, testable idea
     - Includes specific materials/compounds when applicable (using real names, not placeholders)
     - Demonstrates scientific reasoning based on hidden assumptions and probing questions
     - Can be easily selected and explored further

**Output:**
- Clarified question with hidden assumptions
- Socratic probing questions with reasoning
- 3 distinct mini-hypotheses/assumptions (thoughts) for user selection

**User Action:** Select one of the 3 thoughts to continue, or proceed to hypothesis generation.

---

### Stage 2: Hypothesis Generation (`hypothesis`)

**Purpose:** Iteratively refine the research direction and synthesize a comprehensive scientific hypothesis.

**Two Modes Available:**

#### A. Standard Hypothesis Mode

**Iterative Refinement Process:**
1. **Option Selection:** When user selects a thought/option:
   - System analyzes the selected thought using the initial clarified question (including hidden assumptions)
   - Generates a new Socratic question and exactly 3 new next-step options
   - Each option is structured as a mini-hypothesis (1-2 sentences) that can be explored further
   - Options are based on scientific reasoning derived from the analysis

2. **Additional Questions:** User can ask follow-up questions:
   - System processes through: clarify → socratic → TOT
   - Generates new options based on the new question and conversation context

3. **Hypothesis Synthesis:** When user clicks "Generate Hypothesis":
   - System synthesizes a comprehensive hypothesis from the entire conversation
   - Uses full conversation context including all questions, thoughts, and options discussed
   - Generates a detailed scientific hypothesis with:
     - **Hypothesis:** Detailed statement with scientific reasoning, underlying mechanisms, theoretical basis, and explanation of WHY the expected outcome is predicted
     - **Predictions:** Specific, quantifiable predictions with scientific justification
     - **Tests:** Comprehensive experimental approaches focusing on WHAT will be measured and WHY it matters scientifically

**Key Features:**
- Iterative refinement allows users to explore different reasoning pathways
- Each iteration builds on previous context
- System maintains scientific rigor while providing concrete suggestions

#### B. Experimental Planning Mode

**Purpose:** Generate complete experimental plans with protocols, worklists, and plate layouts for automated liquid handling.

**Process:**
1. User configures experimental constraints:
   - Experimental techniques (e.g., XRD, PL, SEM)
   - Available equipment
   - Liquid handling setup (instruments, plate format, max volumes)
   - Available materials
   - Parameters to optimize
   - Focus areas

2. System generates 3 distinct experimental plans, each containing:
   - **Hypothesis:** Specific, measurable hypothesis with exact parameters
   - **Protocol:** Step-by-step procedure with precise materials, ratios, and volumes
   - **Worklist:** Complete plate layout with exact well assignments and volumes
   - **Expected Results:** Quantifiable measurements using specified techniques

3. User can:
   - Select a plan to generate detailed outputs
   - Revise the question with new constraints
   - Export worklists and protocols

---

### Stage 3: Hypothesis Analysis (`analysis`)

**Purpose:** Evaluate the generated hypothesis for scientific quality.

**Process:**
1. System analyzes the hypothesis, predictions, and tests
2. Evaluates based on:
   - Novelty
   - Plausibility
   - Testability
   - Scientific rigor
3. Generates a comprehensive analysis report

**Output:** Detailed report assessing the hypothesis quality and providing feedback.

---

### Stage 4: Experimental Outputs (`experimental_outputs`)

**Purpose:** Display and manage generated experimental plans and protocols.

**Features:**
- View complete experimental plans
- Download worklists (CSV format)
- Upload to Jupyter server (if configured)
- Export protocols
- Visualize plate layouts

---

## Key Workflow Principles

### 1. Scientific Reasoning Throughout
- All stages emphasize scientific reasoning, mechanisms, and theoretical foundations
- Hidden assumptions are explicitly identified and used to guide reasoning
- Specific materials/compounds are suggested based on scientific principles, not placeholders

### 2. Iterative Refinement
- Users can iteratively refine their research direction
- Each iteration builds on previous context
- System maintains conversation history to inform future steps

### 3. Concrete Suggestions
- System provides specific, actionable suggestions (e.g., "PEA (phenethylammonium)" not "[specific organic spacer]")
- Suggestions are justified by scientific reasoning
- Options are structured as testable mini-hypotheses

### 4. Two-Path System
- **Standard Hypothesis Mode:** Focuses on developing comprehensive scientific hypotheses
- **Experimental Planning Mode:** Focuses on generating implementable experimental plans with protocols

### 5. Context Preservation
- Full conversation context is maintained throughout
- Previous questions, thoughts, and options inform future generation
- Hypothesis synthesis considers entire conversation flow

---

## Example Workflow

**User Question:** "What organic spacer can work in stabilizing FAPbI3 alpha phase using BDA2PbI4 as a cospacer?"

**Stage 1 - Initial Processing:**
1. **Clarification:** Extracts hidden assumptions about phase stability, crystal structure requirements, molecular interactions
2. **Socratic Questions:** 
   - "What molecular size constraints are required for organic spacers to fit within the FAPbI3 crystal lattice?"
   - "How does charge distribution affect the stability of the alpha phase?"
   - "What structural compatibility is needed between the spacer and BDA2PbI4?"
3. **TOT Thoughts (3 options):**
   - "PEA (phenethylammonium) could stabilize FAPbI3 alpha phase when combined with BDA2PbI4 due to its optimal molecular size matching the crystal lattice requirements."
   - "BA (butylammonium) might work as a cospacer because its charge distribution complements BDA2PbI4's structure, reducing phase transitions."
   - "MA (methylammonium) combined with BDA2PbI4 could provide enhanced stability through synergistic hydrogen bonding interactions."

**Stage 2 - Iterative Refinement:**
- User selects option 1 (PEA)
- System generates new Socratic question and 3 new options exploring PEA further
- User can continue refining or generate hypothesis

**Stage 2 - Hypothesis Synthesis:**
- System synthesizes comprehensive hypothesis from all conversation context
- Includes detailed scientific reasoning, predictions, and experimental tests

**Stage 3 - Analysis:**
- System evaluates hypothesis quality and provides feedback

---

## Technical Implementation

### Core Components

1. **Question Clarification (`clarify_question`)**
   - Identifies hidden assumptions with scientific context
   - Rewrites question to be precise and testable

2. **Socratic Pass (`socratic_pass`)**
   - Generates probing questions using Socratic principles
   - Explores scientific mechanisms and theoretical foundations

3. **Tree of Thoughts (`tot_generation`)**
   - Produces exactly 3 distinct mini-hypotheses
   - Uses scientific reasoning to suggest specific materials/compounds

4. **Iterative Refinement (`retry_thinking_deepen_thoughts`)**
   - Analyzes selected thought
   - Generates new Socratic question and 3 next-step options

5. **Hypothesis Synthesis (`hypothesis_synthesis`)**
   - Synthesizes comprehensive hypothesis from conversation context
   - Includes scientific reasoning, predictions, and tests

6. **Hypothesis Analysis (`analyze_hypothesis`)**
   - Evaluates hypothesis quality
   - Generates analysis report

### State Management

The system uses Streamlit session state to track:
- Current stage (`initial`, `hypothesis`, `analysis`, etc.)
- Conversation history
- Selected thoughts/options
- Experimental constraints (if in experimental mode)
- Generated outputs

---

## User Interface Features

- **Chat Interface:** Natural conversation flow
- **Option Selection:** Easy-to-pick mini-hypotheses (exactly 3 options)
- **Iterative Refinement:** Continue exploring different pathways
- **Context Preservation:** Full conversation history visible
- **Export Capabilities:** Download worklists, protocols, and analysis reports
- **Experimental Planning:** Configure constraints and generate implementable plans

---

## Summary

The POLARIS Hypothesis Agent provides a structured, scientifically rigorous approach to hypothesis development. Through iterative refinement, Socratic questioning, and scientific reasoning, it helps researchers:

1. Identify hidden assumptions and scientific context
2. Explore different reasoning pathways
3. Generate specific, actionable suggestions
4. Synthesize comprehensive, testable hypotheses
5. Evaluate hypothesis quality

The workflow emphasizes scientific depth while maintaining usability, ensuring that generated hypotheses are both scientifically sound and practically testable.

