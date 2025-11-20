# Performance optimized imports
import streamlit as st
import json
import os
import base64
import logging
import re
from datetime import datetime

# Lazy import heavy modules
try:
    import requests
except ImportError:
    requests = None

# Lazy import AI modules (only when needed)
_socratic = None
_hypothesis_synthesis = None

def _lazy_import_socratic():
    """Lazy import of socratic module"""
    global _socratic
    if _socratic is None:
        import tools.socratic as socratic
        _socratic = socratic
    return _socratic

def _lazy_import_hypothesis_synthesis():
    """Lazy import of hypothesis synthesis instructions"""
    global _hypothesis_synthesis
    if _hypothesis_synthesis is None:
        from tools.instruct import HYPOTHESIS_SYNTHESIS
        _hypothesis_synthesis = HYPOTHESIS_SYNTHESIS
    return _hypothesis_synthesis

def _get_socratic():
    """Get socratic module with lazy loading"""
    return _lazy_import_socratic()

def initial_process(question: str, experimental_mode=False, experimental_constraints=None):
    try:
        # Lazy import socratic module
        socratic = _lazy_import_socratic()

        # Check API key first
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or getattr(socratic, 'GOOGLE_API_KEY', None)
        if not api_key:
            error_msg = "API key not found. Please set it in the UI (⚙️ Options) or as environment variable."
            logging.error(error_msg)
            return error_msg, error_msg, ["Error: API key not configured", "Set API key in Options menu", "Check environment variables"]

        clarified_question = socratic.clarify_question(question)
        if not clarified_question:
            error_msg = "Error: Could not clarify question. Please check your API key and try again."
            logging.error(error_msg)
            clarified_question = error_msg

        print(clarified_question)

        # Only proceed if clarification succeeded
        if clarified_question and not clarified_question.startswith("Error:"):
            socratic_questions = socratic.socratic_pass(clarified_question)
            if not socratic_questions:
                error_msg = "Error: Could not generate socratic questions. Please check your API key and try again."
                logging.error(error_msg)
                socratic_questions = error_msg
        else:
            socratic_questions = "Error: Could not generate socratic questions (previous step failed)."

        # CRITICAL CHECK: Are we even getting here?
        print("\n" + "="*80)
        print("DEBUG: Checking if we should generate socratic answers")
        print(f"  socratic_questions exists: {socratic_questions is not None}")
        print(f"  socratic_questions type: {type(socratic_questions)}")
        if socratic_questions:
            print(f"  socratic_questions length: {len(socratic_questions)}")
            print(f"  starts with Error: {socratic_questions.startswith('Error:')}")
        print("="*80 + "\n")
        
        logging.info("CHECK: About to check if we should generate socratic answers")
        logging.info(f"  socratic_questions exists: {socratic_questions is not None}")
        logging.info(f"  socratic_questions type: {type(socratic_questions)}")
        if socratic_questions:
            logging.info(f"  socratic_questions length: {len(socratic_questions)}")
            logging.info(f"  starts with Error: {socratic_questions.startswith('Error:')}")
        
        # Only proceed if socratic pass succeeded
        if socratic_questions and not socratic_questions.startswith("Error:"):
            print("\n" + "="*80)
            print("DEBUG: Condition passed - WILL generate socratic answers")
            print("="*80 + "\n")
            logging.info("CHECK: Condition passed - will generate socratic answers")
            # CRITICAL: Answer the socratic questions to build deeper reasoning
            # This is where the LLM answers its own questions
            socratic_answers = None
            try:
                logging.info("=" * 80)
                logging.info("STEP 3: Answering Socratic Questions")
                logging.info("=" * 80)
                logging.info("Calling socratic_answer_questions...")
                logging.info(f"  Input - Clarified question: {clarified_question[:200]}...")
                logging.info(f"  Input - Socratic questions length: {len(socratic_questions)}")
                
                socratic_answers = socratic.socratic_answer_questions(clarified_question, socratic_questions)
                
                logging.info(f"socratic_answer_questions returned: {type(socratic_answers)}")
                if socratic_answers:
                    logging.info(f"SUCCESS: Generated socratic answers (length: {len(socratic_answers)})")
                    logging.info(f"  First 500 chars: {socratic_answers[:500]}...")
                    logging.info(f"  Last 200 chars: ...{socratic_answers[-200:]}")
                else:
                    logging.error("FAILED: socratic_answer_questions returned None or empty")
                    logging.error("  This means socratic answers will NOT be displayed in UI")
                    logging.error("  TOT generation will use questions instead of answers")
            except Exception as e:
                logging.error(f"EXCEPTION in socratic_answer_questions: {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
                socratic_answers = None
                # Continue with questions if answers fail
            
            # Use experimental plan TOT if in experimental mode
            if experimental_mode and experimental_constraints:
                thoughts = socratic.tot_generation_experimental_plan(
                    socratic_questions, clarified_question, experimental_constraints
                )
            else:
                # Pass answers to TOT generation if available
                logging.info(f"Calling tot_generation with socratic_answers: {socratic_answers is not None}")
                thoughts = socratic.tot_generation(socratic_questions, clarified_question, socratic_answers)
                if thoughts:
                    logging.info(f"Successfully generated {len(thoughts)} thoughts")
                else:
                    logging.error("tot_generation returned None or empty")
        else:
            print("\n" + "="*80)
            print("DEBUG: Condition FAILED - socratic answers will NOT be generated")
            print(f"  socratic_questions: {socratic_questions[:200] if socratic_questions else 'None'}")
            print("="*80 + "\n")
            logging.error("CHECK: Condition FAILED - socratic answers will NOT be generated")
            logging.error(f"  socratic_questions: {socratic_questions[:200] if socratic_questions else 'None'}")
            thoughts = None
            socratic_answers = None

        # Ensure thoughts is always a list with at least 3 items
        if not thoughts:
            thoughts = ["Error generating thoughts", "Please check API key", "Retry with different question"]
        elif len(thoughts) < 3:
            thoughts = list(thoughts) + [""] * (3 - len(thoughts))

        return clarified_question, socratic_questions, thoughts[:3], socratic_answers

    except ValueError as e:
        # API key related error
        error_msg = f"API Key Error: {str(e)}"
        logging.error(error_msg)
        return error_msg, error_msg, ["Error: API key issue", "Set API key in Options menu", "Check environment variables"], None
    except Exception as e:
        logging.error(f"Error in initial_process: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        error_msg = f"Error processing question: {str(e)}"
        return error_msg, error_msg, ["Error occurred", "Check logs for details", "Try again"], None


@st.cache_resource
def init_session():
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    if "stage" not in st.session_state:
        st.session_state.stage = "initial"
    if "stop_hypothesis" not in st.session_state:
        st.session_state.stop_hypothesis = False
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "allow_followup" not in st.session_state:
        st.session_state.allow_followup = False
    if "experimental_mode" not in st.session_state:
        st.session_state.experimental_mode = False
    if "experimental_constraints" not in st.session_state:
        st.session_state.experimental_constraints = {
            "techniques": [],
            "equipment": [],
            "parameters": [],
            "focus_areas": [],
            "liquid_handling": {
                "max_volume_per_mixture": 50,  # microliters
                "instruments": [],
                "plate_format": "96-well",
                "materials": [],
                "csv_path": "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"
            }
        }
    if "jupyter_config" not in st.session_state:
        st.session_state.jupyter_config = {
            "server_url": "http://10.140.141.160:48888/",
            "token": "",
            "upload_enabled": False,
            "notebook_path": "Dual GP 5AVA BDA"
        }
    # Always try to get API key from environment first (set by run_streamlit.ps1)
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    # If we have an API key from environment, use it automatically
    if env_key:
        # Set in session state
        st.session_state.api_key = env_key
        # Ensure environment variables are set
        os.environ["GOOGLE_API_KEY"] = env_key
        os.environ["GEMINI_API_KEY"] = env_key
        # Update the socratic module's API key directly (lazy load if needed)
        socratic = _lazy_import_socratic()
        socratic.GOOGLE_API_KEY = env_key
        # Cache it in the function for immediate access
        if hasattr(socratic, 'generate_text_with_llm'):
            socratic.generate_text_with_llm._cached_api_key = env_key
        logging.info(f"API key automatically loaded from environment (starts with: {env_key[:10]}...)")
    elif "api_key" not in st.session_state:
        # No environment key and no session state - initialize as empty
        st.session_state.api_key = ""
        # Check if socratic module has a key (in case it was set elsewhere)
        socratic = _lazy_import_socratic()
        if hasattr(socratic, 'GOOGLE_API_KEY') and socratic.GOOGLE_API_KEY:
            st.session_state.api_key = socratic.GOOGLE_API_KEY
            os.environ["GOOGLE_API_KEY"] = socratic.GOOGLE_API_KEY
            os.environ["GEMINI_API_KEY"] = socratic.GOOGLE_API_KEY
            # Cache it in the function
            if hasattr(socratic, 'generate_text_with_llm'):
                socratic.generate_text_with_llm._cached_api_key = socratic.GOOGLE_API_KEY
            logging.info("API key loaded from socratic module")
    else:
        # Session state exists but no env key - sync session state to environment
        if st.session_state.api_key:
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
            os.environ["GEMINI_API_KEY"] = st.session_state.api_key
            socratic = _lazy_import_socratic()
            socratic.GOOGLE_API_KEY = st.session_state.api_key
            if hasattr(socratic, 'generate_text_with_llm'):
                socratic.generate_text_with_llm._cached_api_key = st.session_state.api_key
    
    return st.session_state

def insert_interaction(role, message, component):
    # Ensure interactions list exists
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    st.session_state.interactions.append({
        "role": role,
        "message": message,
        "component": component,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def view_component(component):
    # Ensure interactions list exists
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    for i in st.session_state.interactions:
        if i["component"] == component:
            return i["message"]

def clear_conversation():
    # Ensure all session state attributes exist
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    else:
        st.session_state.interactions = []
    st.session_state.stage = "initial"
    st.session_state.conversation_history = []
    st.session_state.allow_followup = False
    st.toast("Conversation restarted")

def save_to_history(question, hypothesis=None, thoughts=None):
    """Save key conversation elements to history for follow-up questions"""
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "hypothesis": hypothesis,
        "thoughts": thoughts
    }
    st.session_state.conversation_history.append(history_entry)

def get_context_for_followup():
    """Get context from previous conversations for follow-up questions"""
    if st.session_state.conversation_history:
        latest = st.session_state.conversation_history[-1]
        context = f"Previous question: {latest['question']}\n"
        if latest['hypothesis']:
            context += f"Previous hypothesis: {latest['hypothesis']}\n"
        if latest['thoughts']:
            context += f"Previous thoughts: {latest['thoughts'][:200]}...\n"  # Truncate for brevity
        return context
    return ""

def build_conversation_context():
    """Build full conversation context from all interactions"""
    context_parts = []
    
    # Get initial question
    initial_q = view_component("initial_question")
    if initial_q:
        context_parts.append(f"Initial Question: {initial_q}")
    
    # Get clarified question
    clarified = view_component("clarified_question")
    if clarified:
        context_parts.append(f"Clarified Question: {clarified}")
    
    # Get socratic pass
    socratic_pass = view_component("socratic_pass")
    if socratic_pass:
        context_parts.append(f"Socratic Analysis: {socratic_pass}")
    
    # Get all thoughts
    thought1 = view_component("first_thought_1")
    thought2 = view_component("second_thought_1")
    thought3 = view_component("third_thought_1")
    if thought1 or thought2 or thought3:
        thoughts = [t for t in [thought1, thought2, thought3] if t]
        context_parts.append(f"Initial Thoughts: {'; '.join(thoughts)}")
    
    # Get all selected options and their responses
    selected_options = []
    for i in st.session_state.interactions:
        if i.get("component") == "option_choice":
            selected_options.append(f"Selected: {i.get('message')}")
        elif i.get("component") == "next_step_option_1":
            selected_options.append(f"Option 1: {i.get('message')}")
        elif i.get("component") == "next_step_option_2":
            selected_options.append(f"Option 2: {i.get('message')}")
        elif i.get("component") == "next_step_option_3":
            selected_options.append(f"Option 3: {i.get('message')}")
        elif i.get("component") == "additional_question":
            selected_options.append(f"Additional Question: {i.get('message')}")
    
    if selected_options:
        context_parts.append(f"Conversation Flow: {'; '.join(selected_options[-10:])}")  # Last 10 interactions
    
    # Get socratic questions from iterations
    retry_q = view_component("retry_thinking_question")
    if retry_q:
        context_parts.append(f"Latest Socratic Question: {retry_q}")
    
    return "\n\n".join(context_parts) if context_parts else "No previous context available."

def generate_hypothesis_with_context(socratic_question, next_step_option, previous_option_1, previous_option_2, conversation_context):
    """Generate hypothesis with full conversation context"""
    try:
        # Lazy import hypothesis synthesis instructions
        HYPOTHESIS_SYNTHESIS = _lazy_import_hypothesis_synthesis()

        # Add experimental constraints to prompt if in experimental mode (not displayed in UI)
        exp_constraints_text = ""
        if st.session_state.get("experimental_mode", False):
            exp_constraints = get_experimental_context()
            if exp_constraints:
                exp_constraints_text = f"\n\nExperimental Constraints (STRICT - Only use these):\n{exp_constraints}\n"

        # Build enhanced prompt with conversation context
        enhanced_prompt = f"""
        {HYPOTHESIS_SYNTHESIS}

        Socratic Question: {socratic_question}
        Next-Step Option: {next_step_option}
        Previous Step Options:
            Previous Option 1: {previous_option_1}
            Previous Option 2: {previous_option_2}

        Full Conversation Context:
        {conversation_context}
        {exp_constraints_text}
        IMPORTANT: Please synthesize a hypothesis that considers the entire conversation flow above and builds upon all the questions, thoughts, and options discussed throughout the conversation. The hypothesis should integrate insights from the full discussion, not just the most recent option.

        **ADDITIONAL REQUIREMENTS FOR SCIENTIFIC DETAIL:**
        - The hypothesis MUST include scientific reasoning, underlying mechanisms, and theoretical background explaining WHY the expected outcome is predicted
        - The hypothesis MUST include specific materials (e.g., chemical names, formulas, concentrations) with scientific justification
        - The predictions MUST include quantifiable outcomes (e.g., specific values, ranges, or measurable changes) with scientific justification for why these specific predictions follow from the hypothesis
        - The tests MUST specify characterization techniques and measurement methods, focusing on WHAT will be measured and WHY it matters scientifically, not just procedural steps
        - The hypothesis should connect to broader scientific understanding and include clear logical connections between hypothesis, predictions, and tests
        - Avoid vague terms like "improve", "optimize", "enhance" - use specific, measurable criteria with scientific rationale
        - Each section should be comprehensive and detailed, providing sufficient scientific depth and context
        """
        
        # Use the socratic module's generate_text_with_llm directly
        from tools.socratic import generate_text_with_llm
        hypothesis = generate_text_with_llm(enhanced_prompt)
        
        if hypothesis is None or not hypothesis.strip():
            # Fallback to standard synthesis
            hypothesis = socratic.hypothesis_synthesis(socratic_question, next_step_option, previous_option_1, previous_option_2)
        
        return hypothesis
    except Exception as e:
        logging.error(f"Hypothesis generation with context failed: {e}")
        # Fallback to standard synthesis
        try:
            return socratic.hypothesis_synthesis(socratic_question, next_step_option, previous_option_1, previous_option_2)
        except Exception as e2:
            logging.error(f"Fallback hypothesis synthesis also failed: {e2}")
            return f"Error generating hypothesis: {str(e2)}. Please check your API key and try again."

def parse_worklist_from_plan(experimental_plan, materials):
    """Parse specific worklist details from experimental plan"""
    import re

    if not experimental_plan:
        return None

    # Look for worklist patterns in the experimental plan
    worklist_data = []

    # Pattern 1: CSV-style format like "A1,Cs_uL=20,BDA_uL=15,Solvent_uL=15"
    csv_pattern = r'([A-Z]\d{1,2}),([^;]+)'
    matches = re.findall(csv_pattern, experimental_plan)

    if matches:
        for well, mixture_str in matches:
            well_data = {'Well': well}

            # Parse mixture components like "Cs_uL=20,BDA_uL=15,Solvent_uL=15"
            components = mixture_str.split(',')
            for component in components:
                component = component.strip()
                if '=' in component:
                    mat, vol = component.split('=', 1)
                    mat = mat.strip()
                    try:
                        vol = float(vol.strip())
                        if mat in materials:
                            well_data[mat] = vol
                    except ValueError:
                        continue

            if len(well_data) > 1:  # Has well + at least one material
                worklist_data.append(well_data)

    # If we found specific worklist data, return it
    if worklist_data:
        return worklist_data

    return None

def start_followup_question():
    """Switch to follow-up question mode"""
    st.session_state.stage = "followup"
    st.session_state.allow_followup = True

def generate_worklist(experimental_plan, plate_format="96-well", materials=None):
    """Generate a worklist CSV based on experimental plan - uses specific worklist if provided, otherwise creates varied mixing ratios"""
    import csv
    import io
    import re

    # Default materials if not provided
    if materials is None:
        materials = ["Cs_uL", "BDA_uL", "BDA_2_uL"]

    # Generate well IDs (Opentrons format: A01, A02, etc.)
    wells = []
    if plate_format == "96-well":
        for row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            for col in range(1, 13):
                wells.append(f"{row}{col:02d}")
    elif plate_format == "384-well":
        for row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
            for col in range(1, 25):
                wells.append(f"{row}{col:02d}")
    else:  # 24-well
        for row in ['A', 'B', 'C', 'D']:
            for col in range(1, 7):
                wells.append(f"{row}{col:02d}")

    # Generate CSV content
    output = io.StringIO()
    fieldnames = ['Well'] + materials
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    # Get max volume from constraints
    max_vol = st.session_state.experimental_constraints["liquid_handling"]["max_volume_per_mixture"]

    # First, try to parse specific worklist from experimental plan
    parsed_worklist = parse_worklist_from_plan(experimental_plan, materials)
    if parsed_worklist:
        # Use the parsed specific worklist
        for well_data in parsed_worklist:
            writer.writerow(well_data)
        return output.getvalue()
    
    # Parse experimental plan to understand what to vary
    plan_lower = experimental_plan.lower() if experimental_plan else ""
    
    # Determine number of conditions to test based on plan
    # Look for keywords like "varying ratios", "different mixing ratios", "range of", etc.
    num_conditions = 24  # Default
    if "mixing ratio" in plan_lower or "ratio" in plan_lower:
        # Create a grid of mixing ratios
        if len(materials) == 2:
            # For 2 materials, create ratio variations (e.g., 90:10, 80:20, ..., 10:90)
            num_conditions = min(len(wells), 24)
            ratios = []
            for i in range(num_conditions):
                # Create ratios from 5:95 to 95:5
                ratio1 = 5 + (i * (90 // max(1, num_conditions - 1)))
                ratio2 = 100 - ratio1
                ratios.append((ratio1, ratio2))
        elif len(materials) >= 2:
            # For multiple materials, create systematic variations
            num_conditions = min(len(wells), 24)
        else:
            num_conditions = min(len(wells), 8)
    
    # Generate varied compositions based on experimental design
    num_wells = min(len(wells), num_conditions)
    
    for i, well in enumerate(wells[:num_wells]):
        row_data = {'Well': well}
        
        if len(materials) == 2:
            # Two materials: vary ratios systematically
            # Create ratios from 10:90 to 90:10 (varying material 1)
            if num_wells > 1:
                # Create range from 10% to 90% in equal steps
                ratio1_pct = 10 + (i * (80.0 / max(1, num_wells - 1)))
            else:
                ratio1_pct = 50
            ratio2_pct = 100 - ratio1_pct
            
            # Calculate volumes with proper rounding
            vol1 = round((ratio1_pct / 100.0) * max_vol)
            vol2 = max_vol - vol1  # Ensure exact total
            
            # Make sure volumes are valid
            vol1 = max(0, min(max_vol, vol1))
            vol2 = max(0, min(max_vol, vol2))
            
            # Final check: ensure total is exactly max_vol
            if vol1 + vol2 != max_vol:
                vol2 = max_vol - vol1
            
            row_data[materials[0]] = vol1
            row_data[materials[1]] = vol2
            
        elif len(materials) == 3:
            # Three materials: create systematic variations
            # Divide into groups: vary material 1, vary material 2, vary material 3
            groups = num_wells // 3
            if groups == 0:
                groups = 1
            
            if i < groups:
                # Vary material 1 (10-90% of max_vol)
                mat1_vol = int(10 + (i * (80 / max(1, groups - 1))))
                remaining = max_vol - mat1_vol
                mat2_vol = remaining // 2
                mat3_vol = remaining - mat2_vol
            elif i < groups * 2:
                # Vary material 2
                mat2_vol = int(10 + ((i - groups) * (80 / max(1, groups - 1))))
                remaining = max_vol - mat2_vol
                mat1_vol = remaining // 2
                mat3_vol = remaining - mat1_vol
            else:
                # Vary material 3
                mat3_vol = int(10 + ((i - groups * 2) * (80 / max(1, groups - 1))))
                remaining = max_vol - mat3_vol
                mat1_vol = remaining // 2
                mat2_vol = remaining - mat1_vol
            
            row_data[materials[0]] = max(0, mat1_vol)
            row_data[materials[1]] = max(0, mat2_vol)
            row_data[materials[2]] = max(0, mat3_vol)
        else:
            # For 4+ materials, create systematic grid
            # Distribute volumes with variations
            base_vol = max_vol // len(materials)
            variation = base_vol // 4
            
            for idx, mat in enumerate(materials):
                # Create slight variations per material
                vol = base_vol + (variation * (i % 3 - 1))  # -1, 0, or +1 variation
                if idx == len(materials) - 1:
                    # Last material gets remainder to ensure total = max_vol
                    total_so_far = sum(row_data.get(m, 0) for m in materials[:-1])
                    vol = max_vol - total_so_far
                row_data[mat] = max(0, min(max_vol, vol))
        
        writer.writerow(row_data)
    
    return output.getvalue()

def upload_to_jupyter(server_url, token, file_content, filename, notebook_path):
    """Upload file to Jupyter server using Jupyter API"""
    try:
        # Clean up URL
        server_url = server_url.rstrip('/')
        if not server_url.startswith('http'):
            server_url = f"http://{server_url}"
        
        # Construct API endpoint
        api_path = f"{notebook_path}/{filename}"
        api_url = f"{server_url}/api/contents/{api_path}"
        
        # Prepare headers
        headers = {
            "Authorization": f"token {token}" if token else None
        }
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Prepare file content (base64 encoded for binary, plain text for text files)
        if filename.endswith('.csv') or filename.endswith('.py') or filename.endswith('.txt'):
            # Text files
            content_data = {
                "type": "file",
                "format": "text",
                "content": file_content
            }
        else:
            # Binary files (base64 encoded)
            content_data = {
                "type": "file",
                "format": "base64",
                "content": base64.b64encode(file_content.encode() if isinstance(file_content, str) else file_content).decode()
            }
        
        # Make PUT request to create/update file
        response = requests.put(
            api_url,
            json=content_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code in [200, 201]:
            return True, f"Successfully uploaded {filename} to {api_path}"
        else:
            return False, f"Failed to upload: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Error uploading file: {str(e)}"

def generate_opentrons_protocol(csv_filename, materials=None, csv_path="/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"):
    """Generate Opentrons Python protocol file that reads CSV and executes transfers
    Each material is in a separate tube, and materials are mixed in the destination well based on ratios"""
    if materials is None:
        materials = ["Cs_uL", "BDA_uL", "BDA_2_uL"]
    
    # Map materials to tube indices (one tube per material, no reservoirs)
    material_tube_mapping = []
    for idx, material in enumerate(materials):
        material_tube_mapping.append(f"        '{material}': {idx},  # Material in tube {idx}")
    
    protocol_code = f"""from opentrons import protocol_api
import csv

metadata = {{
    'protocolName': 'POLARIS Hypothesis Agent - Automated Liquid Handling',
    'author': 'POLARIS Hypothesis Agent',
    'description': 'Generated protocol for automated material mixing from separate tubes',
    'apiLevel': '2.13'
}}

def run(protocol: protocol_api.ProtocolContext):
    # Load labware
    left_tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', '1')
    tuberack = protocol.load_labware('opentrons_15_tuberack_nest_15ml_conical', '2')
    triple_cation_plate = protocol.load_labware('corning_96_wellplate_360ul_flat', '3')

    # Load instruments
    left_pipette = protocol.load_instrument('p300_single', 'left', tip_racks=[left_tiprack])

    # Set flow rates
    left_pipette.flow_rate.aspirate = 30
    left_pipette.flow_rate.dispense = 150

    # Define z heights for tubes (all tubes use same z height)
    z_height = 117  # Standard z height for all tubes

    # Map each material to its tube index (one material per tube)
    material_to_tube = {{
{chr(10).join(material_tube_mapping)}
    }}

    # Read CSV file
    csv_file_path = f'{csv_path}{csv_filename}'
    
    with open(csv_file_path, mode='r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        # Process each well (row) in the CSV
        for row in csvreader:
            try:
                well_id = row['Well']
                
                # Collect all materials and volumes for this well
                material_volumes = []
                for material in material_to_tube.keys():
                    if material in row and row[material]:
                        try:
                            volume = float(row[material])
                            if volume > 0:
                                material_volumes.append((material, volume))
                        except (ValueError, TypeError):
                            protocol.comment(f"Skipping {{material}} for well {{well_id}}: invalid volume")
                            continue
                
                # If we have materials to transfer, mix them all into the same well
                if material_volumes:
                    # Pick up a fresh tip for this well
                    left_pipette.pick_up_tip()
                    
                    # Transfer each material to the same destination well
                    # This will mix them together in the correct ratios
                    for material, volume in material_volumes:
                        tube_index = material_to_tube[material]
                        
                        # Transfer from source tube to destination well
                        left_pipette.transfer(
                            volume,
                            tuberack.wells()[tube_index].top(z=-z_height),
                            triple_cation_plate.wells_by_name()[well_id],
                            blowout_location='destination well',
                            blow_out=True,
                            new_tip='never',  # Use same tip for all materials in this well (mixing)
                            touch_tip=True
                        )
                    
                    # Drop tip after mixing all materials for this well
                    left_pipette.drop_tip()
                    
            except KeyError as e:
                protocol.comment(f"Error processing row: missing key {{str(e)}}")
                continue
            except Exception as e:
                protocol.comment(f"Error processing row: {{str(e)}}")
                continue
"""
    
    return protocol_code

def generate_plate_layout(experimental_plan, plate_format="96-well", worklist_content=None):
    """Generate a visual plate layout based on actual worklist data"""
    import csv
    import io
    
    layout = []
    layout.append("# Well Plate Layout")
    layout.append(f"# Format: {plate_format}")
    layout.append("")
    
    # Parse worklist to get actual compositions
    well_data = {}
    if worklist_content:
        try:
            reader = csv.DictReader(io.StringIO(worklist_content))
            for row in reader:
                well_id = row.get('Well', '')
                if well_id:
                    # Extract material volumes
                    mat_vols = []
                    for key, val in row.items():
                        if key != 'Well' and val:
                            try:
                                vol = float(val)
                                if vol > 0:
                                    mat_name = key.replace('_uL', '')
                                    mat_vols.append((mat_name, vol))
                            except:
                                pass
                    if mat_vols:
                        well_data[well_id] = mat_vols
        except Exception as e:
            pass
    
    if plate_format == "96-well":
        layout.append("    01     02     03     04     05     06     07     08     09     10     11     12")
        for row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            row_layout = [f"{row} "]
            for col in range(1, 13):
                well_id = f"{row}{col:02d}"
                if well_id in well_data and well_data[well_id]:
                    # Show actual composition
                    mat_vols = well_data[well_id]
                    total_vol = sum(v for _, v in mat_vols)
                    
                    if len(mat_vols) == 2 and total_vol > 0:
                        # Show ratio for 2 materials (e.g., "70:30")
                        ratio1 = int((mat_vols[0][1] / total_vol) * 100)
                        ratio2 = 100 - ratio1
                        content = f"{ratio1}:{ratio2}"
                    elif len(mat_vols) == 3 and total_vol > 0:
                        # Show ratios for 3 materials (e.g., "50:30:20")
                        ratios = [int((v / total_vol) * 100) for _, v in mat_vols]
                        content = ":".join([str(r) for r in ratios])
                    else:
                        # Show volumes for multiple materials
                        content = "/".join([f"{int(v)}" for _, v in mat_vols[:2]])
                        if len(mat_vols) > 2:
                            content += "+"
                else:
                    content = "---"
                row_layout.append(f"{content:>7}")
            layout.append(" ".join(row_layout))
    elif plate_format == "384-well":
        # Similar for 384-well
        layout.append("# 384-well plate layout (showing first 24 wells as example)")
        for row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
            row_layout = [f"{row} "]
            for col in range(1, 25):
                well_id = f"{row}{col:02d}"
                if well_id in well_data and well_data[well_id]:
                    mat_vols = well_data[well_id]
                    total_vol = sum(v for _, v in mat_vols)
                    if len(mat_vols) == 2 and total_vol > 0:
                        ratio1 = int((mat_vols[0][1] / total_vol) * 100)
                        ratio2 = 100 - ratio1
                        content = f"{ratio1}:{ratio2}"
                    else:
                        content = "/".join([f"{int(v)}" for _, v in mat_vols[:2]])
                else:
                    content = "---"
                row_layout.append(f"{content:>6}")
            layout.append(" ".join(row_layout))
    else:  # 24-well
        layout.append("    01     02     03     04     05     06")
        for row in ['A', 'B', 'C', 'D']:
            row_layout = [f"{row} "]
            for col in range(1, 7):
                well_id = f"{row}{col:02d}"
                if well_id in well_data and well_data[well_id]:
                    mat_vols = well_data[well_id]
                    total_vol = sum(v for _, v in mat_vols)
                    if len(mat_vols) == 2 and total_vol > 0:
                        ratio1 = int((mat_vols[0][1] / total_vol) * 100)
                        ratio2 = 100 - ratio1
                        content = f"{ratio1}:{ratio2}"
                    else:
                        content = "/".join([f"{int(v)}" for _, v in mat_vols[:2]])
                else:
                    content = "---"
                row_layout.append(f"{content:>7}")
            layout.append(" ".join(row_layout))

    layout.append("")
    layout.append("# Legend:")
    layout.append("# Numbers show mixing ratios (e.g., 70:30 = 70% Material 1, 30% Material 2)")
    layout.append("# Volumes shown as Material1:Material2 or Material1:Material2:Material3")

    return "\n".join(layout)

def toggle_experimental_mode():
    """Toggle experimental planning mode"""
    st.session_state.experimental_mode = not st.session_state.experimental_mode
    if st.session_state.experimental_mode:
        st.success("Experimental Planning Mode Activated!")
    else:
        st.info("Standard Hypothesis Mode Activated!")

def get_experimental_context():
    """Get experimental constraints as context for hypothesis generation"""
    constraints = st.session_state.experimental_constraints
    context = ""
    
    # Add explicit constraint header
    context += "=== EXPERIMENTAL CONSTRAINTS (STRICT - DO NOT DEVIATE) ===\n\n"
    
    if constraints["techniques"]:
        context += f"ONLY USE THESE EXPERIMENTAL TECHNIQUES (DO NOT MENTION OTHERS): {', '.join(constraints['techniques'])}\n"
        context += f"CRITICAL: Do NOT suggest, mention, or reference any techniques NOT in this list.\n"
        context += f"For example, if only 'time-resolved PL' is listed, DO NOT mention XRD, DFT, SEM, TEM, or any other technique.\n\n"
    else:
        context += "WARNING: No experimental techniques specified. Use only basic characterization methods.\n\n"
    
    if constraints["equipment"]:
        context += f"ONLY USE THIS EQUIPMENT (DO NOT SUGGEST ALTERNATIVES): {', '.join(constraints['equipment'])}\n\n"
    else:
        context += "No specific equipment constraints.\n\n"
    
    if constraints["parameters"]:
        context += f"ONLY FOCUS ON THESE PARAMETERS: {', '.join(constraints['parameters'])}\n"
        context += f"Do NOT introduce additional parameters not listed here.\n\n"
    else:
        context += "No specific parameter constraints.\n\n"
    
    if constraints["focus_areas"]:
        context += f"Primary focus areas: {', '.join(constraints['focus_areas'])}\n\n"

    # Add liquid handling context
    context += "=== LIQUID HANDLING CONSTRAINTS ===\n"
    lh = constraints["liquid_handling"]
    if lh["instruments"]:
        context += f"Liquid handling instruments: {', '.join(lh['instruments'])}\n"
    if lh["materials"]:
        context += f"Available materials: {', '.join(lh['materials'])}\n"
    context += f"Plate format: {lh['plate_format']}\n"
    context += f"Maximum volume per mixture: {lh['max_volume_per_mixture']} µL\n"
    context += f"Generate worklists and well plate layouts for automated liquid handling\n\n"
    
    # Add final reminder
    context += "=== REMINDER ===\n"
    context += "STRICTLY adhere to the listed techniques and equipment. Do NOT suggest alternatives or additional methods.\n"

    return context

def go_back_stage():
    # Ensure interactions list exists
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    if st.session_state.interactions:
        st.session_state.interactions.pop()
        st.toast("Returned to previous stage")
    else:
        st.warning("No previous stage to go back to.")

# --- FIX: Stop button now triggers a controlled rerun into hypothesis synthesis ---
def stop_and_create_hypothesis():
    st.session_state.stop_hypothesis = True
    st.session_state.stage = "hypothesis"
    st.toast("🧠 Generating hypothesis...")
    st.rerun()

def export_message_history():
    # Ensure interactions list exists
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    if not st.session_state.interactions:
        st.warning("No messages to export.")
        return None, None

    json_data = json.dumps(st.session_state.interactions, indent=2)
    text_data = "\n\n".join([
        f"[{i['timestamp']}] {i['role'].upper()} - {i['component'].upper()}: {i['message']}"
        for i in st.session_state.interactions
    ])
    return json_data, text_data


# ====== STREAMLIT UI ======
st.set_page_config(page_title="AI Hypothesis Agent", page_icon="✨", layout="centered")

# Add custom CSS for better visual distinction between chat sections
st.markdown("""
    <style>
    /* User messages - light blue background */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageUser"]) {
        background-color: #E3F2FD !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        border-left: 4px solid #2196F3 !important;
    }
    
    /* Assistant messages - light green background */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAssistant"]) {
        background-color: #F1F8E9 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        border-left: 4px solid #4CAF50 !important;
    }
    
    /* Add extra spacing between chat messages */
    div[data-testid="stChatMessage"] {
        margin-bottom: 1.5rem !important;
    }
    
    /* Style for section headers (bold text) */
    div[data-testid="stChatMessage"] strong {
        color: #1565C0 !important;
        font-size: 1.05em;
    }
    
    /* Visual separator for horizontal rules */
    div[data-testid="stChatMessage"] hr {
        border: none;
        border-top: 2px solid #BDBDBD;
        margin: 1rem 0;
    }
    
    /* Style for option numbers to make them stand out */
    div[data-testid="stChatMessage"] p strong:first-child {
        color: #7B1FA2 !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("AI Hypothesis Agent ✨")

init_session()

# --- Layout styling ---
st.markdown("""
<style>
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {
        align-items: flex-end;
    }
    .bottom-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        border-top: 1px solid #ddd;
        padding: 0.8rem 1.5rem;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.05);
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# --- Display existing chat ---
chat_container = st.container()
with chat_container:
    # Ensure interactions list exists
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    for i in st.session_state.interactions:
        with st.chat_message(i["role"]):
            # Add section headers for specific components to make them clear
            if i.get("component") == "socratic_answers":
                st.markdown("**Socratic Reasoning (LLM Answers to Its Own Questions):**")
            st.markdown(i["message"])

st.markdown("<br>", unsafe_allow_html=True)

# --- Bottom Controls ---
bottom = st.container()
with bottom:
    st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
    chat_col, options_col = st.columns([0.8, 0.2])

    with options_col:
        with st.popover("Options"):
            st.markdown("### API Configuration")
            api_key_input = st.text_input(
                "Google Gemini API Key:",
                value=st.session_state.api_key,
                type="password",
                help="Enter your Google Gemini API key",
                key="api_key_input"
            )
            
            if st.button("💾 Save API Key", use_container_width=True):
                if api_key_input and api_key_input.strip():
                    api_key = api_key_input.strip()
                    st.session_state.api_key = api_key
                    # Update environment variables for the socratic module
                    os.environ["GOOGLE_API_KEY"] = api_key
                    os.environ["GEMINI_API_KEY"] = api_key
                    # Update the socratic module's API key directly (this is the module-level variable)
                    socratic.GOOGLE_API_KEY = api_key
                    # Also cache it in the function for immediate access
                    if hasattr(socratic, 'generate_text_with_llm'):
                        socratic.generate_text_with_llm._cached_api_key = api_key
                    st.success("✅ API key updated! The key is now active.")
                    st.rerun()  # Rerun to ensure the key is loaded
                else:
                    st.error("Please enter a valid API key")
            
            if not st.session_state.api_key:
                st.warning("API key not set. Please enter your Google Gemini API key above.")
            else:
                # Show confirmation without revealing any part of the key
                st.info("✅ API key configured and loaded")
            
            st.markdown("---")
            st.markdown("### Conversation Controls")
            st.button("Restart", use_container_width=True, on_click=clear_conversation)
            st.button("Stop & Create Hypothesis", use_container_width=True, on_click=stop_and_create_hypothesis)
            st.button("Go Back", use_container_width=True, on_click=go_back_stage)
            st.markdown("---")
            st.markdown("### Export Data")

            json_data, text_data = export_message_history()
            if json_data and text_data:
                try:
                    st.download_button(
                        label="Export as JSON",
                        data=json_data,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                        use_container_width=True,
                        key=f"export_json_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                except Exception as e:
                    logging.error(f"Error creating JSON download button: {e}")
                
                try:
                    st.download_button(
                        label="Export as Text",
                        data=text_data,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                        use_container_width=True,
                        key=f"export_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                except Exception as e:
                    logging.error(f"Error creating text download button: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ====== MAIN LOGIC FLOW ======
# --- FIX: If stop button was pressed, jump straight to hypothesis ---
if st.session_state.stop_hypothesis and st.session_state.stage != "analysis":
    with st.chat_message("assistant"):
        with st.spinner("Synthesizing hypothesis from current context..."):
            try:
                # Extract available data with fallbacks
                soc_q = view_component("retry_thinking_question") or view_component("clarified_question") or "How can we continue exploring this hypothesis?"
                picked = view_component("next_step_option_1") or view_component("first_thought_1") or "Selected option"
                prev1 = view_component("next_step_option_2") or view_component("second_thought_1") or "Previous option 1"
                prev2 = view_component("next_step_option_3") or view_component("third_thought_1") or "Previous option 2"

                # Ensure we have valid values
                if not soc_q or not picked:
                    st.error("Insufficient context to generate hypothesis. Please go through the conversation flow first.")
                    st.session_state.stop_hypothesis = False
                    st.rerun()

                # Experimental constraints are already in the instructions, not displayed
                hypothesis = socratic.hypothesis_synthesis(soc_q, picked, prev1, prev2)

                # Ensure hypothesis is never None
                if hypothesis is None or not str(hypothesis).strip():
                    hypothesis = "Error generating hypothesis. Please check your API key and try again."
            except Exception as e:
                logging.error(f"Error in stop_hypothesis handler: {e}")
                hypothesis = f"Error generating hypothesis: {str(e)}. Please check your API key and try again."

        st.markdown(hypothesis if hypothesis else "Error generating hypothesis. Please try again.")
        st.success("🎉 Hypothesis generation complete (forced stop).")

    insert_interaction("assistant", hypothesis, "hypothesis")
    st.session_state.stop_hypothesis = False
    st.session_state.stage = "analysis"
    st.rerun()


# --- NORMAL STAGES ---
if st.session_state.stage == "initial":
    st.write("Welcome to the hypothesis agent! Please enter a question that you would like to explore further.")

    # Experimental Planning Controls
    with st.expander("Experimental Planning Mode", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Toggle Experimental Mode", type="secondary"):
                toggle_experimental_mode()
                st.rerun()

            st.markdown(f"**Mode:** {'Experimental Planning' if st.session_state.experimental_mode else 'Standard Hypothesis'}")

        with col2:
            if st.button("🔧 Configure Constraints", type="secondary"):
                st.session_state.show_constraints = not st.session_state.get("show_constraints", False)
                st.rerun()

        if st.session_state.get("show_constraints", False):
            st.markdown("### Experimental Constraints")

            # Techniques
            techniques = st.multiselect(
                "Experimental Techniques:",
                ["in-situ PL", "spin coating", "absorbance spectroscopy", "XRD", "SEM", "TEM", "UV-Vis", "photoluminescence", "time-resolved PL", "impedance spectroscopy"],
                default=st.session_state.experimental_constraints["techniques"]
            )

            # Equipment
            equipment = st.multiselect(
                "Available Equipment:",
                ["spin bot", "pipetting robot", "glove box", "solar simulator", "spectrometer", "microscope", "thermal evaporator", "spin coater", "Tecan liquid handler", "Opentrons liquid handler"],
                default=st.session_state.experimental_constraints["equipment"]
            )

            # Liquid Handling Configuration
            st.markdown("#### Liquid Handling Setup")
            col_lh1, col_lh2 = st.columns(2)

            with col_lh1:
                # Instruments multiselect - ensure no duplicates
                available_instruments = ["Tecan", "Opentrons", "manual pipettes", "multichannel pipettes"]
                current_instruments = st.session_state.experimental_constraints["liquid_handling"]["instruments"]
                # Remove duplicates from current instruments
                current_instruments = list(dict.fromkeys(current_instruments))  # Preserves order, removes duplicates
                
                lh_instruments = st.multiselect(
                    "Liquid Handling Instruments:",
                    available_instruments,
                    default=current_instruments
                )
                # Ensure no duplicates in the selected list
                lh_instruments = list(dict.fromkeys(lh_instruments))

                plate_format = st.selectbox(
                    "Plate Format:",
                    ["96-well", "384-well", "24-well"],
                    index=["96-well", "384-well", "24-well"].index(st.session_state.experimental_constraints["liquid_handling"]["plate_format"])
                )

            with col_lh2:
                max_volume = st.slider(
                    "Max Volume per Mixture (µL):",
                    min_value=10,
                    max_value=200,
                    value=st.session_state.experimental_constraints["liquid_handling"]["max_volume_per_mixture"]
                )

                # Materials input - allow typing and adding custom materials
                st.markdown("**Available Materials:**")
                
                # Show current materials as chips
                current_materials = st.session_state.experimental_constraints["liquid_handling"]["materials"]
                if current_materials:
                    # Display as chips/badges
                    material_chips = " ".join([f"`{mat}`" for mat in current_materials])
                    st.markdown(f"Current: {material_chips}")
                
                # Text input for adding new material
                st.markdown("**Add New Material:**")
                col_input, col_btn = st.columns([3, 1])
                with col_input:
                    new_material = st.text_input(
                        "Material name:",
                        placeholder="e.g., Cs, BDA, 5AVA, or custom name",
                        key="new_material_input",
                        label_visibility="visible"
                    )
                with col_btn:
                    st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
                    add_btn = st.button("➕ Add", key="add_material_btn", use_container_width=True)
                
                # Process new material input
                materials = list(current_materials) if current_materials else []
                
                if add_btn and new_material and new_material.strip():
                    material_to_add = new_material.strip()
                    # Add if not already in list (case-insensitive check)
                    if material_to_add.lower() not in [m.lower() for m in materials]:
                        materials.append(material_to_add)
                        st.session_state.experimental_constraints["liquid_handling"]["materials"] = materials
                        st.success(f"Added: {material_to_add}")
                        st.rerun()
                    else:
                        st.warning(f"Material '{material_to_add}' already exists")
                
                # Preset materials for quick selection
                st.markdown("**Quick Add:**")
                preset_cols = st.columns(4)
                preset_materials = ["Cs", "BDA", "BDA_2", "5AVA", "FAPbI3", "Material 1", "Material 2", "Material 3"]
                for i, preset in enumerate(preset_materials):
                    col_idx = i % 4
                    with preset_cols[col_idx]:
                        if st.button(f"+ {preset}", key=f"add_preset_{preset}", use_container_width=True):
                            if preset.lower() not in [m.lower() for m in materials]:
                                materials.append(preset)
                                st.session_state.experimental_constraints["liquid_handling"]["materials"] = materials
                                st.rerun()
                            else:
                                st.warning(f"'{preset}' already added")
                
                # Remove materials option
                if materials:
                    st.markdown("**Remove Materials:**")
                    remove_cols = st.columns(min(len(materials), 4))
                    for i, mat in enumerate(materials):
                        col_idx = i % 4
                        with remove_cols[col_idx]:
                            if st.button(f"❌ {mat}", key=f"remove_{mat}", use_container_width=True):
                                materials.remove(mat)
                                st.session_state.experimental_constraints["liquid_handling"]["materials"] = materials
                                st.rerun()

                csv_path = st.text_input(
                    "CSV File Path (for Opentrons):",
                    value=st.session_state.experimental_constraints["liquid_handling"].get("csv_path", "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"),
                    help="Path where CSV file will be stored on Opentrons robot"
                )

            # Jupyter Server Configuration
            st.markdown("#### Jupyter Server Upload")
            col_jup1, col_jup2 = st.columns(2)
            
            with col_jup1:
                jupyter_url = st.text_input(
                    "Jupyter Server URL:",
                    value=st.session_state.jupyter_config["server_url"],
                    help="URL of Jupyter server (e.g., http://10.140.141.160:48888/)"
                )
                
                jupyter_token = st.text_input(
                    "Jupyter Token (optional):",
                    value=st.session_state.jupyter_config["token"],
                    type="password",
                    help="Authentication token for Jupyter server"
                )
            
            with col_jup2:
                jupyter_notebook_path = st.text_input(
                    "Notebook Path:",
                    value=st.session_state.jupyter_config["notebook_path"],
                    help="Directory path in Jupyter (e.g., 'Dual GP 5AVA BDA')"
                )
                
                jupyter_upload_enabled = st.checkbox(
                    "Enable Auto-Upload to Jupyter",
                    value=st.session_state.jupyter_config["upload_enabled"],
                    help="Automatically upload generated files to Jupyter server"
                )

            # Parameters
            parameters = st.multiselect(
                "Key Parameters to Optimize:",
                ["spin speed", "concentration", "temperature", "humidity", "annealing time", "layer thickness", "mixing ratio", "deposition rate"],
                default=st.session_state.experimental_constraints["parameters"]
            )

            # Focus Areas
            focus_areas = st.multiselect(
                "Primary Focus Areas:",
                ["device performance", "material stability", "process optimization", "characterization", "scaling", "cost reduction"],
                default=st.session_state.experimental_constraints["focus_areas"]
            )

            if st.button("💾 Save Constraints"):
                st.session_state.experimental_constraints = {
                    "techniques": techniques,
                    "equipment": equipment,
                    "parameters": parameters,
                    "focus_areas": focus_areas,
                    "liquid_handling": {
                        "max_volume_per_mixture": max_volume,
                        "instruments": list(dict.fromkeys(lh_instruments)),  # Ensure no duplicates
                        "plate_format": plate_format,
                        "materials": list(dict.fromkeys(materials)),  # Ensure no duplicates
                        "csv_path": csv_path if 'csv_path' in locals() else "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"
                    }
                }
                st.session_state.jupyter_config = {
                    "server_url": jupyter_url if 'jupyter_url' in locals() else "http://10.140.141.160:48888/",
                    "token": jupyter_token if 'jupyter_token' in locals() else "",
                    "upload_enabled": jupyter_upload_enabled if 'jupyter_upload_enabled' in locals() else False,
                    "notebook_path": jupyter_notebook_path if 'jupyter_notebook_path' in locals() else "Dual GP 5AVA BDA"
                }
                st.success("Constraints saved!")
                st.rerun()

    with chat_col:
        question = st.chat_input("Ask a question...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get experimental constraints if in experimental mode (for LLM only, not displayed)
                exp_constraints_str = None
                if st.session_state.experimental_mode:
                    exp_constraints_str = get_experimental_context()
                    # Note: Constraints are passed to LLM but not displayed in UI
                
                # DEBUG: Show we're calling initial_process
                st.write("DEBUG: Calling initial_process...")
                
                cl_question, soc_pass, thoughts_gen, soc_answers = initial_process(
                    question, 
                    experimental_mode=st.session_state.experimental_mode,
                    experimental_constraints=exp_constraints_str
                )
                
                # DEBUG: Show what was returned
                st.write(f"DEBUG: initial_process returned soc_answers type: {type(soc_answers)}")
                if soc_answers:
                    st.write(f"DEBUG: soc_answers length: {len(soc_answers)}")
                    st.write(f"DEBUG: First 200 chars: {soc_answers[:200]}")
                else:
                    st.write("DEBUG: soc_answers is None or empty!")
                
                # Ensure we have exactly 3 thoughts (initial_process should handle this, but double-check)
                if not thoughts_gen or len(thoughts_gen) == 0:
                    st.error("Failed to generate thoughts. Please try again or check your API key.")
                    thoughts_gen = ["No thoughts generated", "Please retry", "Error occurred"]
                elif len(thoughts_gen) < 3:
                    # Pad with empty strings if we have fewer than 3 thoughts
                    thoughts_gen = list(thoughts_gen) + [""] * (3 - len(thoughts_gen))
                
                # CRITICAL: Ensure thoughts are always displayed - never skip this step
                # Safe unpacking - always get exactly 3 items
                first_thought = thoughts_gen[0] if len(thoughts_gen) > 0 else "Option 1: Continue exploring"
                second_thought = thoughts_gen[1] if len(thoughts_gen) > 1 else "Option 2: Continue exploring"
                third_thought = thoughts_gen[2] if len(thoughts_gen) > 2 else "Option 3: Continue exploring"
                
                # Validate that thoughts are not empty
                if not first_thought or not first_thought.strip():
                    first_thought = "Option 1: Continue exploring"
                if not second_thought or not second_thought.strip():
                    second_thought = "Option 2: Continue exploring"
                if not third_thought or not third_thought.strip():
                    third_thought = "Option 3: Continue exploring"
                
                # DISPLAY EVERYTHING INSIDE THE CHAT MESSAGE BLOCK
                st.markdown("**Clarified Question:**")
                st.markdown(cl_question)
                st.markdown("**Socratic Pass (Probing Questions):**")
                st.markdown(soc_pass)
        
                # DEBUG: Show what we received
                st.markdown(f"**DEBUG: soc_answers status: {type(soc_answers)} - None={soc_answers is None} - Empty={not soc_answers if soc_answers else 'N/A'}**")
        
                # CRITICAL: Display socratic answers - this is where LLM answers its own questions
                print("\n" + "="*80)
                print("DEBUG: DISPLAYING SOCRATIC ANSWERS IN UI")
                print(f"  soc_answers is None: {soc_answers is None}")
                print(f"  soc_answers type: {type(soc_answers)}")
                if soc_answers:
                    print(f"  soc_answers length: {len(soc_answers)}")
                    print(f"  soc_answers stripped empty: {not soc_answers.strip()}")
                    print(f"  First 200 chars: {soc_answers[:200]}")
                print("="*80 + "\n")
                
                if soc_answers and soc_answers.strip():
                    print("DEBUG: DISPLAYING SOCRATIC ANSWERS NOW\n")
                    st.markdown("---")  # Visual separator
                    st.markdown("**Socratic Reasoning (LLM Answers to Its Own Questions):**")
                    st.markdown(soc_answers)
                    print(f"DEBUG: st.markdown called with {len(soc_answers)} characters\n")
                else:
                    print("DEBUG: NOT displaying socratic answers - condition failed\n")
                    if soc_answers is None:
                        st.error("ERROR: Socratic answers were not generated.")
                    elif not soc_answers.strip():
                        st.error("ERROR: Socratic answers were empty.")
                
                # Show different labels for experimental mode
                if st.session_state.experimental_mode:
                    st.markdown("**Experimental Plans:**")
                else:
                    st.markdown("**Generated Thoughts:**")
                
                # Display thoughts with clear numbering for easy selection
                st.markdown(f"**1.** {first_thought}")
                st.markdown(f"**2.** {second_thought}")
                st.markdown(f"**3.** {third_thought}")

        insert_interaction("user", question, "initial_question")
        insert_interaction("assistant", cl_question, "clarified_question")
        insert_interaction("assistant", soc_pass, "socratic_pass")
        # CRITICAL: Save socratic answers so they persist after rerun
        if soc_answers and soc_answers.strip():
            insert_interaction("assistant", soc_answers, "socratic_answers")
        insert_interaction("assistant", first_thought, "first_thought_1")
        insert_interaction("assistant", second_thought, "second_thought_1")
        insert_interaction("assistant", third_thought, "third_thought_1")

        st.session_state.stage = "refine"
        st.rerun()

elif st.session_state.stage == "refine":
    if st.session_state.experimental_mode:
        st.write("**You are presented with three experimental plans.** Choose one to refine, regenerate with assumptions, or ask a question to revise.")
    else:
        st.write("You are presented with three lines of distinct thoughts. Please choose the option that explores your initial question best.")
    
    with chat_col:
        user_choice = st.chat_input("Make a choice 1, 2, or 3...")
    
    # Experimental planning specific controls
    if st.session_state.experimental_mode:
        with st.expander("Experimental Plan Actions", expanded=True):
            col_reg, col_ask = st.columns(2)
            with col_reg:
                regenerate_assumptions = st.text_area(
                    "Regenerate with assumptions/thoughts:",
                    placeholder="e.g., 'Assume high temperature increases stability...'",
                    key="regenerate_assumptions",
                    height=100
                )
                if st.button("Regenerate Plans", key="regenerate_btn"):
                    if regenerate_assumptions and regenerate_assumptions.strip():
                        st.session_state.regenerate_with_assumptions = regenerate_assumptions
                        st.rerun()
            
            with col_ask:
                revise_question = st.text_input(
                    "Ask a question to revise:",
                    placeholder="e.g., 'What if we focus on stability instead?'",
                    key="revise_question_input"
                )
                if st.button("❓ Revise Plans", key="revise_btn"):
                    if revise_question and revise_question.strip():
                        st.session_state.revise_experimental_plan = revise_question
                        st.rerun()
    if user_choice:
        if user_choice not in ["1", "2", "3"]:
            st.warning("Please enter 1, 2, or 3.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(user_choice)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if user_choice == "1":
                    picked = view_component("first_thought_1")
                    prev1 = view_component("second_thought_1")
                    prev2 = view_component("third_thought_1")
                elif user_choice == "2":
                    picked = view_component("second_thought_1")
                    prev1 = view_component("first_thought_1")
                    prev2 = view_component("third_thought_1")
                else:
                    picked = view_component("third_thought_1")
                    prev1 = view_component("first_thought_1")
                    prev2 = view_component("second_thought_1")

                # Show brief analysis of the selected option (not a full hypothesis report)
                # This is just to help the user understand what they selected
                st.markdown("**Selected Option:**")
                st.markdown(picked)

                st.markdown("---")

                # Lazy import socratic module
                socratic = _lazy_import_socratic()
                
                initial_question = view_component("clarified_question")
                result = socratic.retry_thinking_deepen_thoughts(
                    picked, prev1, prev2, initial_question
                )

                # Handle None or unexpected return values
                if result is None:
                    soc_q = "How can we continue exploring this hypothesis?"
                    options = ["Why is this approach theoretically sound?", "What are the mechanistic advantages?", "How does this compare to alternatives?"]
                elif len(result) != 2:
                    soc_q = "How can we continue exploring this hypothesis?"
                    options = ["Why is this approach theoretically sound?", "What are the mechanistic advantages?", "How does this compare to alternatives?"]
                else:
                    soc_q, options = result
                    # Ensure options is a list with exactly 3 items, filtering out None/empty
                    if not isinstance(options, list):
                        options = [str(o) for o in options] if options else []
                    # Filter out None, empty strings, and "None" strings
                    options = [opt for opt in options if opt and str(opt).strip() and str(opt).strip().lower() != "none"]
                    if len(options) < 3:
                        options = list(options) + [""] * (3 - len(options))
                    elif len(options) > 3:
                        options = options[:3]  # Take only first 3
                    # Ensure no None values
                    options = [opt if opt and str(opt).strip() != "None" else f"Option {i+1}: Continue exploring" for i, opt in enumerate(options)]

            # Ensure we only display exactly 3 options, and filter out None/empty values
            valid_options = [opt for opt in options[:3] if opt and str(opt).strip() and str(opt).strip().lower() != "none"]
            # Pad to 3 if needed
            while len(valid_options) < 3:
                valid_options.append(f"Option {len(valid_options) + 1}: Continue exploring this line of reasoning")
            # Take only first 3
            valid_options = valid_options[:3]

            st.markdown("**Continuation Question:**")
            st.markdown(soc_q if soc_q and soc_q != "None" else "How can we continue exploring this hypothesis?")
            st.markdown("**Next-Step Options:**")
            for i, opt in enumerate(valid_options, 1):
                opt_str = str(opt).strip()
                if opt_str and opt_str.lower() != "none":
                    st.markdown(f"{i}. {opt_str}")
                else:
                    st.markdown(f"{i}. Option {i}: Continue exploring this line of reasoning")

        insert_interaction("user", user_choice, "tot_choice")
        insert_interaction("assistant", soc_q if soc_q and str(soc_q).strip().lower() != "none" else "How can we continue exploring this hypothesis?", "retry_thinking_question")
        # Ensure we only save 3 options, using valid_options (defined above)
        valid_options_to_save = valid_options[:3] if len(valid_options) >= 3 else valid_options + [f"Option {len(valid_options) + i + 1}: Continue exploring" for i in range(3 - len(valid_options))]
        insert_interaction("assistant", valid_options_to_save[0] if len(valid_options_to_save) > 0 else "Option 1: Continue exploring", "next_step_option_1")
        insert_interaction("assistant", valid_options_to_save[1] if len(valid_options_to_save) > 1 else "Option 2: Continue exploring", "next_step_option_2")
        insert_interaction("assistant", valid_options_to_save[2] if len(valid_options_to_save) > 2 else "Option 3: Continue exploring", "next_step_option_3")

        # For experimental mode, show "Set Experimental Plan" button instead of going to hypothesis stage
        if st.session_state.experimental_mode:
            st.session_state.selected_experimental_plan = picked
            st.session_state.stage = "experimental_plan_ready"
        else:
            st.session_state.stage = "hypothesis"
            st.rerun()
    
    # Handle regeneration with assumptions
    if st.session_state.get("regenerate_with_assumptions"):
        assumptions = st.session_state.regenerate_with_assumptions
        st.session_state.regenerate_with_assumptions = None
        
        with st.chat_message("user"):
            st.markdown(f"Regenerate with assumptions: {assumptions}")
        
        with st.chat_message("assistant"):
            with st.spinner("Regenerating experimental plans with your assumptions..."):
                initial_question = view_component("clarified_question") or view_component("initial_question") or ""
                socratic_questions = view_component("socratic_pass") or ""
                exp_constraints = get_experimental_context()
                
                # Add assumptions to constraints
                enhanced_constraints = f"{exp_constraints}\n\nAdditional Assumptions/Thoughts:\n{assumptions}"
                
                new_plans = socratic.tot_generation_experimental_plan(
                    socratic_questions, initial_question, enhanced_constraints
                )
                
                if not new_plans or len(new_plans) == 0:
                    new_plans = ["Error generating plans", "Please check API key", "Retry"]
                elif len(new_plans) < 3:
                    new_plans = list(new_plans) + [""] * (3 - len(new_plans))
                
                st.markdown("**Regenerated Experimental Plans:**")
                st.markdown(new_plans[0] if len(new_plans) > 0 else "")
                st.markdown(new_plans[1] if len(new_plans) > 1 else "")
                st.markdown(new_plans[2] if len(new_plans) > 2 else "")
        
        insert_interaction("assistant", new_plans[0] if len(new_plans) > 0 else "", "first_thought_1")
        insert_interaction("assistant", new_plans[1] if len(new_plans) > 1 else "", "second_thought_1")
        insert_interaction("assistant", new_plans[2] if len(new_plans) > 2 else "", "third_thought_1")
        st.rerun()
    
    # Handle revise question
    if st.session_state.get("revise_experimental_plan"):
        revise_q = st.session_state.revise_experimental_plan
        st.session_state.revise_experimental_plan = None
        
        with st.chat_message("user"):
            st.markdown(revise_q)
        
        # Process revised question through the flow
        context = build_conversation_context()
        contextual_question = f"{context}\n\nRevision question: {revise_q}"
        
        with st.chat_message("assistant"):
            with st.spinner("Revising experimental plans based on your question..."):
                clarified_revise = socratic.clarify_question(contextual_question)
                socratic_revise = socratic.socratic_pass(clarified_revise)
                exp_constraints = get_experimental_context()
                
                revised_plans = socratic.tot_generation_experimental_plan(
                    socratic_revise, clarified_revise, exp_constraints
                )
                
                if not revised_plans or len(revised_plans) == 0:
                    revised_plans = ["Error generating plans", "Please check API key", "Retry"]
                elif len(revised_plans) < 3:
                    revised_plans = list(revised_plans) + [""] * (3 - len(revised_plans))
                
                st.markdown("**Revised Experimental Plans:**")
                st.markdown(revised_plans[0] if len(revised_plans) > 0 else "")
                st.markdown(revised_plans[1] if len(revised_plans) > 1 else "")
                st.markdown(revised_plans[2] if len(revised_plans) > 2 else "")
        
        insert_interaction("assistant", clarified_revise, "clarified_question")
        insert_interaction("assistant", socratic_revise, "socratic_pass")
        insert_interaction("assistant", revised_plans[0] if len(revised_plans) > 0 else "", "first_thought_1")
        insert_interaction("assistant", revised_plans[1] if len(revised_plans) > 1 else "", "second_thought_1")
        insert_interaction("assistant", revised_plans[2] if len(revised_plans) > 2 else "", "third_thought_1")
        st.rerun()

elif st.session_state.stage == "experimental_plan_ready":
    st.header("Experimental Plan Selected")

    selected_plan = st.session_state.get("selected_experimental_plan", view_component("first_thought_1"))

    # Parse the experimental plan to extract components
    hypothesis = ""
    protocol = ""
    worklist = ""
    expected_results = ""

    # Try to extract structured components from the plan
    lines = selected_plan.split('\n')
    current_section = ""
    for line in lines:
        line = line.strip()
        if line.startswith('**Hypothesis:**') or line.startswith('Hypothesis:'):
            current_section = "hypothesis"
            hypothesis = line.replace('**Hypothesis:**', '').replace('Hypothesis:', '').strip()
        elif line.startswith('**Protocol:**') or line.startswith('Protocol:'):
            current_section = "protocol"
            protocol = line.replace('**Protocol:**', '').replace('Protocol:', '').strip()
        elif line.startswith('**Worklist:**') or line.startswith('Worklist:'):
            current_section = "worklist"
            worklist = line.replace('**Worklist:**', '').replace('Worklist:', '').strip()
        elif line.startswith('**Expected Results:**') or line.startswith('Expected Results:'):
            current_section = "expected_results"
            expected_results = line.replace('**Expected Results:**', '').replace('Expected Results:', '').strip()
        elif current_section and line:
            if current_section == "hypothesis":
                hypothesis += " " + line
            elif current_section == "protocol":
                protocol += " " + line
            elif current_section == "worklist":
                worklist += " " + line
            elif current_section == "expected_results":
                expected_results += " " + line

    # Display the parsed components
    if hypothesis:
        st.markdown("### 📋 Hypothesis")
        st.markdown(hypothesis.strip())

    if protocol:
        st.markdown("### 🔬 Protocol")
        st.markdown(protocol.strip())

    if worklist:
        st.markdown("### 💧 Worklist Details")
        # Try to format worklist as a table if it's CSV-like
        if ',' in worklist and '_uL=' in worklist:
            st.markdown("**Mixture Compositions:**")
            # Parse CSV-like worklist format
            lines = worklist.strip().split(';')
            if lines:
                st.code(worklist.strip(), language="text")
        else:
            st.markdown(worklist.strip())

    if expected_results:
        st.markdown("### 📊 Expected Results")
        st.markdown(expected_results.strip())

    st.markdown("---")

    if st.button("🚀 Generate Protocol, Worklist & Plate Layout", type="primary", use_container_width=True):
        with st.spinner("Generating protocol, worklist, and plate visualization..."):
            # Get constraints
            lh = st.session_state.experimental_constraints["liquid_handling"]
            plate_format = lh["plate_format"]
            materials = lh["materials"] if lh["materials"] else ["Cs_uL", "BDA_uL", "BDA_2_uL"]
            
            # Format materials for CSV
            csv_materials = []
            for mat in materials:
                if mat.endswith("_uL"):
                    csv_materials.append(mat)
                else:
                    csv_materials.append(f"{mat}_uL")
            
            # Generate worklist using the specific worklist details from the experimental plan
            # If worklist details are available, use them; otherwise fall back to generating from the full plan
            worklist_input = worklist if worklist.strip() else selected_plan
            worklist_content = generate_worklist(worklist_input, plate_format, csv_materials)
            csv_filename = f"worklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # Generate plate layout (pass worklist so it shows actual compositions)
            layout_input = worklist if worklist.strip() else selected_plan
            layout_content = generate_plate_layout(layout_input, plate_format, worklist_content)
            layout_filename = f"plate_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            # Generate protocol if Opentrons is selected
            protocol_content = None
            protocol_filename = None
            if "Opentrons" in lh["instruments"]:
                csv_path = lh.get("csv_path", "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/")
                protocol_content = generate_opentrons_protocol(csv_filename, csv_materials, csv_path)
                protocol_filename = f"opentrons_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            
            # Save to session state for display
            st.session_state.experimental_outputs = {
                "worklist": worklist_content,
                "worklist_filename": csv_filename,
                "layout": layout_content,
                "layout_filename": layout_filename,
                "protocol": protocol_content,
                "protocol_filename": protocol_filename,
                "selected_plan": selected_plan,
                "hypothesis": hypothesis,
                "protocol_description": protocol,
                "expected_results": expected_results
            }
            
            st.session_state.stage = "experimental_outputs"
            st.rerun()

elif st.session_state.stage == "experimental_outputs":
    st.header("Experimental Outputs")

    outputs = st.session_state.get("experimental_outputs", {})

    if outputs:
        # Display hypothesis if available
        hypothesis = outputs.get("hypothesis", "")
        if hypothesis:
            st.markdown("### 📋 Hypothesis")
            st.markdown(hypothesis)
            st.markdown("---")

        # Display expected results if available
        expected_results = outputs.get("expected_results", "")
        if expected_results:
            st.markdown("### 📊 Expected Results")
            st.markdown(expected_results)
            st.markdown("---")
        # Display worklist
        st.markdown("### Worklist (CSV)")
        worklist_data = outputs.get("worklist", "")
        if worklist_data:
            st.code(worklist_data, language="csv")
            try:
                worklist_filename = outputs.get("worklist_filename", "worklist.csv")
                # Ensure filename is valid
                if not worklist_filename or worklist_filename.strip() == "":
                    worklist_filename = f"worklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    "💾 Download Worklist CSV",
                    data=worklist_data,
                    file_name=worklist_filename,
                    mime="text/csv",
                    key=f"dl_worklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except Exception as e:
                logging.error(f"Error creating worklist download: {e}")
                st.error(f"Error creating download: {str(e)}")
        else:
            st.warning("No worklist data available")
        
        st.markdown("---")
        
        # Display plate layout
        st.markdown("### 📊 Plate Visualization")
        layout_data = outputs.get("layout", "")
        if layout_data:
            st.code(layout_data, language="text")
            try:
                layout_filename = outputs.get("layout_filename", "plate_layout.txt")
                # Ensure filename is valid
                if not layout_filename or layout_filename.strip() == "":
                    layout_filename = f"plate_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                st.download_button(
                    "💾 Download Plate Layout",
                    data=layout_data,
                    file_name=layout_filename,
                    mime="text/plain",
                    key=f"dl_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except Exception as e:
                logging.error(f"Error creating layout download: {e}")
                st.error(f"Error creating download: {str(e)}")
        else:
            st.warning("No layout data available")
        
        st.markdown("---")
        
        # Display protocol if available
        protocol_data = outputs.get("protocol", "")
        if protocol_data:
            st.markdown("### 🤖 Opentrons Protocol")
            st.code(protocol_data, language="python")
            try:
                protocol_filename = outputs.get("protocol_filename", "protocol.py")
                # Ensure filename is valid
                if not protocol_filename or protocol_filename.strip() == "":
                    protocol_filename = f"opentrons_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                st.download_button(
                    "💾 Download Protocol",
                    data=protocol_data,
                    file_name=protocol_filename,
                    mime="text/x-python",
                    key=f"dl_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except Exception as e:
                logging.error(f"Error creating protocol download: {e}")
                st.error(f"Error creating download: {str(e)}")
        else:
            if outputs.get("protocol_filename"):
                st.warning("Protocol filename exists but no protocol data available")
        
        st.markdown("---")
        
        if st.button("Start New Experimental Plan", type="primary"):
            st.session_state.stage = "initial"
            st.session_state.experimental_outputs = None
        st.rerun()

elif st.session_state.stage == "hypothesis":
    # REGULAR HYPOTHESIS MODE ONLY - Experimental mode should not reach here
    if st.session_state.experimental_mode:
        st.error("Error: Experimental planning mode should not reach hypothesis stage. Redirecting to experimental planning...")
        st.session_state.stage = "refine"
        st.rerun()
    
    # Check if we just generated new options (from iterative refinement)
    # This runs AFTER rerun, so options saved before rerun should be available here
    newly_generated_soc_q = st.session_state.get("_newly_generated_soc_q", None)
    newly_generated_options = st.session_state.get("_newly_generated_options", None)
    
    # Debug: Log what we retrieved after rerun
    if newly_generated_options:
        logging.info(f"✓ After rerun: Found {len(newly_generated_options)} newly generated options in session state")
        logging.info(f"  Options: {[str(opt)[:80] for opt in newly_generated_options[:3]]}")
        # Verify they're the right options (not fallback/generic options)
        first_opt = str(newly_generated_options[0]) if len(newly_generated_options) > 0 else ""
        if "Explore how Incorporating" in first_opt or "Option 1: Continue exploring" in first_opt:
            logging.warning(f"⚠ WARNING: Retrieved options appear to be fallback/generic options, not the actual generated options!")
            logging.warning(f"  First option: {first_opt[:100]}")
    else:
        logging.warning(f"✗ After rerun: No newly generated options found in session state")
        logging.warning(f"  This means options were either not saved or were cleared during rerun")
        # Check if they're in conversation history instead
        hist_opt1 = view_component("next_step_option_1")
        if hist_opt1:
            logging.info(f"  Found options in conversation history: {hist_opt1[:80]}...")
    
    # Display the next-step options (REGULAR HYPOTHESIS MODE ONLY)
    st.write("**Standard Hypothesis Agent - Iterative Refinement Mode**")
    st.write("You can:")
    st.write("1. **Choose an option (1, 2, or 3)** → generates 3 new continuation options")
    st.write("2. **Ask an additional question** → triggers socratic questioning and TOT thinking")
    st.write("3. **Click 'Generate Hypothesis'** → synthesizes your hypothesis from all conversation")
    
    # CRITICAL: Log session state contents at the VERY START
    logging.info("=" * 80)
    logging.info("START OF CURRENT NEXT-STEP OPTIONS - CHECKING SESSION STATE")
    logging.info(f"  _newly_generated_options in session_state: {'_newly_generated_options' in st.session_state}")
    if '_newly_generated_options' in st.session_state:
        val = st.session_state._newly_generated_options
        logging.info(f"  _newly_generated_options value: {val}")
        logging.info(f"  Type: {type(val)}")
        if isinstance(val, list):
            logging.info(f"  Length: {len(val)}")
            for i, opt in enumerate(val[:3], 1):
                logging.info(f"    Opt {i}: {str(opt)[:120]}...")
    
    # Get the value - try multiple keys
    newly_generated_options = st.session_state.get("_newly_generated_options", None)
    if not newly_generated_options:
        # Try backup key
        newly_generated_options = st.session_state.get("_last_generated_options_backup", None)
        if newly_generated_options:
            logging.info(f"  Retrieved from BACKUP key instead!")

    
    # Log what we're checking
    logging.info(f"Retrieved newly_generated_options: {newly_generated_options is not None}")
    if newly_generated_options:
        logging.info(f"  Type: {type(newly_generated_options)}")
        logging.info(f"  Length: {len(newly_generated_options) if isinstance(newly_generated_options, list) else 'N/A'}")
        if isinstance(newly_generated_options, list):
            logging.info(f"  Content: {[str(opt)[:80] for opt in newly_generated_options[:3]]}")
    logging.info("=" * 80)
    
    if newly_generated_options and isinstance(newly_generated_options, list) and len(newly_generated_options) >= 3:
        logging.info(f"🔵 FOUND newly generated options! Using them instead of conversation history.")
        logging.info(f"  Options count: {len(newly_generated_options)}")
        logging.info(f"  First option preview: {str(newly_generated_options[0])[:100] if len(newly_generated_options) > 0 else 'None'}...")
        # Make sure each option is properly split and cleaned (safety check)
        opt1_raw = str(newly_generated_options[0]).strip() if len(newly_generated_options) > 0 else "Option 1"
        opt2_raw = str(newly_generated_options[1]).strip() if len(newly_generated_options) > 1 else "Option 2"
        opt3_raw = str(newly_generated_options[2]).strip() if len(newly_generated_options) > 2 else "Option 3"
        
        # Clean up any "Distinct Line of Thought" markers
        thought_pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"
        opt1 = re.sub(thought_pattern, "", opt1_raw, flags=re.IGNORECASE).strip()
        opt2 = re.sub(thought_pattern, "", opt2_raw, flags=re.IGNORECASE).strip()
        opt3 = re.sub(thought_pattern, "", opt3_raw, flags=re.IGNORECASE).strip()
        
        # Ensure they're not empty - NO FALLBACKS, just log error
        if not opt1 or len(opt1) < 5:
            logging.error(f"❌ opt1 is empty or too short!")
            opt1 = opt1_raw if opt1_raw and len(opt1_raw) > 5 else "[ERROR: Empty option 1]"
        if not opt2 or len(opt2) < 5:
            logging.error(f"❌ opt2 is empty or too short!")
            opt2 = opt2_raw if opt2_raw and len(opt2_raw) > 5 else "[ERROR: Empty option 2]"
        if not opt3 or len(opt3) < 5:
            logging.error(f"❌ opt3 is empty or too short!")
            opt3 = opt3_raw if opt3_raw and len(opt3_raw) > 5 else "[ERROR: Empty option 3]"
        
        logging.info(f"✓ Using newly generated options for display (found {len(newly_generated_options)} options):")
        logging.info(f"  Option 1: {opt1[:100]}...")
        logging.info(f"  Option 2: {opt2[:100]}...")
        logging.info(f"  Option 3: {opt3[:100]}...")
        
        # Check if options look like placeholders or incomplete
        placeholder_words = ["Incorporating", "Utilizing", "Introducing", "Exploring", "Analyzing"]
        found_placeholders = [word for word in placeholder_words if word in opt2 or word in opt3]
        if found_placeholders:
            logging.warning(f"⚠ WARNING: Options contain placeholder text ({found_placeholders}) - these may be incomplete!")
            logging.warning(f"  This suggests the LLM generated incomplete options or parsing failed")
            # Try to fix incomplete options by extracting from the selected option
            selected_opt = view_component("last_selected_option") or ""
            if selected_opt:
                # Try to find material names in selected option
                material_pattern = r'([A-Z][a-z]+[⁺²⁺]?|[A-Z][a-z]+ammonium|\d+-[A-Z][a-z]+[⁺²⁺]?|[A-Z][A-Z][A-Z][A-Z]?[⁺²⁺]?)'
                materials = re.findall(material_pattern, selected_opt)
                if materials:
                    mat = materials[0]
                    for placeholder in found_placeholders:
                        if placeholder in opt2:
                            opt2 = opt2.replace(placeholder, mat)
                        if placeholder in opt3:
                            opt3 = opt3.replace(placeholder, mat)
                    logging.info(f"  Fixed placeholder by replacing {found_placeholders} with '{mat}'")
    else:
        # No newly generated options found in _newly_generated_options
        # Try to get the MOST RECENT SET of 3 continuation options from interactions
        logging.warning(f"❌ No _newly_generated_options found, searching interactions...")
        logging.warning(f"Total interactions: {len(st.session_state.interactions)}")
        
        # Log the LAST 10 interactions to see what's there
        logging.info("Last 10 interactions (most recent at bottom):")
        for i in range(max(0, len(st.session_state.interactions) - 10), len(st.session_state.interactions)):
            comp = st.session_state.interactions[i].get("component", "")
            content = st.session_state.interactions[i].get("content", "")[:100]
            logging.info(f"  [{i}] {comp}: {content}...")
        
        # Find the LAST occurrence of each next_step_option type (most recent)
        # This ensures we get the newest set, not an older one
        opt1, opt2, opt3 = None, None, None
        opt1_idx, opt2_idx, opt3_idx = -1, -1, -1
        
        # Search backwards to find the LAST (most recent) occurrence of each
        for i in range(len(st.session_state.interactions) - 1, -1, -1):
            comp = st.session_state.interactions[i].get("component", "")
            content = st.session_state.interactions[i].get("content", "")
            
            # Only set if we haven't found this component yet (since we're going backwards, first find is the most recent)
            if comp == "next_step_option_1" and opt1 is None:
                opt1 = content
                opt1_idx = i
                logging.info(f"✓ Found next_step_option_1 at index {i}: {content[:120]}...")
            elif comp == "next_step_option_2" and opt2 is None:
                opt2 = content
                opt2_idx = i
                logging.info(f"✓ Found next_step_option_2 at index {i}: {content[:120]}...")
            elif comp == "next_step_option_3" and opt3 is None:
                opt3 = content
                opt3_idx = i
                logging.info(f"✓ Found next_step_option_3 at index {i}: {content[:120]}...")
            
            # Stop once we've found all 3
            if opt1 and opt2 and opt3:
                break
        
        # Check if all 3 were found and they're reasonably close together (within last 10 interactions)
        found_set = opt1 and opt2 and opt3
        if found_set:
            max_idx = max(opt1_idx, opt2_idx, opt3_idx)
            min_idx = min(opt1_idx, opt2_idx, opt3_idx)
            if max_idx - min_idx > 10:
                logging.warning(f"Found options are spread far apart (indices {min_idx}-{max_idx}), might be from different sets")
            
            logging.info(f"✓ Retrieved most recent options from indices {min_idx}-{max_idx}:")
            logging.info(f"  Option 1 (idx {opt1_idx}): {opt1[:100] if opt1 else 'None'}...")
            logging.info(f"  Option 2 (idx {opt2_idx}): {opt2[:100] if opt2 else 'None'}...")
            logging.info(f"  Option 3 (idx {opt3_idx}): {opt3[:100] if opt3 else 'None'}...")
        
        # Check if we actually found all 3 options (regardless of found_set flag)
        if not opt1 or not opt2 or not opt3:
            logging.error(f"❌ CRITICAL: Could not find all 3 options!")
            logging.error(f"  opt1 is None/empty: {not opt1}")
            logging.error(f"  opt2 is None/empty: {not opt2}")
            logging.error(f"  opt3 is None/empty: {not opt3}")
            # Only set error messages for options that are actually missing
            if not opt1:
                opt1 = "[ERROR: Option 1 not found in interactions]"
            if not opt2:
                opt2 = "[ERROR: Option 2 not found in interactions]"
            if not opt3:
                opt3 = "[ERROR: Option 3 not found in interactions]"
        else:
            # All 3 options were found
            logging.info(f"✓ Successfully retrieved all 3 options from interactions")
            logging.info(f"  Indices: opt1={opt1_idx}, opt2={opt2_idx}, opt3={opt3_idx}")
    
    # Always show "Current Next-Step Options" with the options (newly generated or from history)
    # Final validation: check for corrupted or outdated options
    logging.info("=" * 80)
    logging.info("FINAL VALIDATION before displaying Current Next-Step Options:")
    logging.info(f"  opt1: {opt1[:120] if opt1 else 'None'}...")
    logging.info(f"  opt2: {opt2[:120] if opt2 else 'None'}...")
    logging.info(f"  opt3: {opt3[:120] if opt3 else 'None'}...")
    
    # Check for corrupted options (containing old materials or generic fallbacks)
    # Log errors but DO NOT replace - user wants to see actual retrieved options, not fallbacks
    bad_indicators = ['Cl₂', 'Examine the specific mechanism', 'preventing water ingress', 'How does Examine', 
                     'PPD', 'XBDA', 'para-phenylenediamine', '1,4-bis(aminomethyl)benzene']
    
    # Just LOG if corrupted, don't replace
    if any(bad in str(opt1) for bad in bad_indicators):
        logging.error(f"⚠️ CORRUPTED option 1 detected (old material): {opt1[:150]}")
    if any(bad in str(opt2) for bad in bad_indicators):
        logging.error(f"⚠️ CORRUPTED option 2 detected (old material): {opt2[:150]}")
    if any(bad in str(opt3) for bad in bad_indicators):
        logging.error(f"⚠️ CORRUPTED option 3 detected (old material): {opt3[:150]}")
    
    # DO NOT replace - show what was actually retrieved so we can debug
    
    logging.info("After corruption check:")
    logging.info(f"  opt1: {opt1[:120] if opt1 else 'None'}...")
    logging.info(f"  opt2: {opt2[:120] if opt2 else 'None'}...")
    logging.info(f"  opt3: {opt3[:120] if opt3 else 'None'}...")
    logging.info("=" * 80)
    
    with st.expander("Current Next-Step Options", expanded=True):
        st.markdown(f"**1.** {opt1}")
        st.markdown(f"**2.** {opt2}")
        st.markdown(f"**3.** {opt3}")
    
    # Single input that can handle either option choice or additional question
    user_input = st.chat_input("Pick option 1, 2, or 3, or ask an additional question...")
    
    # Separate text input for additional questions
    with st.expander("💬 Ask Additional Question", expanded=False):
        additional_question_input = st.text_input(
            "Ask a question to refine your thinking:",
            placeholder="e.g., 'How would this apply to...?' or 'What if we consider...?'",
            key="additional_question_input"
        )
        if st.button("Ask Question", key="ask_additional_btn"):
            if additional_question_input and additional_question_input.strip():
                st.session_state.pending_additional_question = additional_question_input
                st.rerun()
    
    # Check for pending additional question from button click
    additional_question = st.session_state.get("pending_additional_question", None)
    if additional_question:
        # Clear it so it doesn't process again
        st.session_state.pending_additional_question = None
        # DON'T clear newly generated options here - they should persist until new ones replace them
        # Clearing too early causes the "Current Next-Step Options" to show old options
        # st.session_state._newly_generated_soc_q = None
        # st.session_state._newly_generated_options = None
        logging.info("Additional question asked, but NOT clearing _newly_generated_options yet")
    
    # Handle option selection FIRST (before hypothesis generation button and additional questions)
    # This ensures iterative refinement works properly
    user_choice_2 = user_input if user_input and user_input.strip() in ["1", "2", "3"] else None
    if user_choice_2:
        if user_choice_2 not in ["1", "2", "3"]:
            st.warning("Please enter 1, 2, or 3.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(f"Selected option {user_choice_2}")

        # DON'T clear newly generated options here - we need them to determine which option was selected
        # They will be cleared after we generate new options

        # Save selected option
        if user_choice_2 == "1":
            picked = opt1
            prev1 = opt2
            prev2 = opt3
        elif user_choice_2 == "2":
            picked = opt2
            prev1 = opt1
            prev2 = opt3
        else:
            picked = opt3
            prev1 = opt1
            prev2 = opt2

        insert_interaction("user", user_choice_2, "option_choice")
        insert_interaction("assistant", picked, "last_selected_option")
        insert_interaction("assistant", prev1, "last_prev1")
        insert_interaction("assistant", prev2, "last_prev2")

        # Generate continuation options based on selected option (iterative refinement)
        # CRITICAL: Initialize new_options variable before the chat message block
        # so it's available after the block ends
        new_options = None
        soc_q = ""
        
        with st.chat_message("assistant"):
            with st.spinner("Generating continuation options based on your selection..."):

                # Show socratic analysis for the selected option during iterative refinement
                # Show brief display of the selected option (not a full hypothesis report)
                # This is just to help the user understand what they selected
                st.markdown("**Selected Option:**")
                st.markdown(picked)

                st.markdown("---")
                try:
                    initial_question = view_component("clarified_question") or view_component("initial_question") or ""
                    
                    # Build conversation context to help the LLM understand the flow
                    conversation_context = build_conversation_context()
                    
                    # Log the full selected option for debugging
                    logging.info(f"Calling retry_thinking_deepen_thoughts with:")
                    logging.info(f"  picked (SELECTED - MUST CONTINUE FROM THIS): '{picked[:200]}...'")
                    logging.info(f"  prev1 (NOT selected): '{prev1[:100] if prev1 else ''}...'")
                    logging.info(f"  prev2 (NOT selected): '{prev2[:100] if prev2 else ''}...'")
                    logging.info(f"  initial_question: '{initial_question[:100] if initial_question else ''}...'")
                    
                    # REGULAR HYPOTHESIS MODE ONLY - Don't use experimental constraints here
                    # Pass conversation context to help it understand the flow
                    # CRITICAL: The 'picked' variable contains the option the user selected - we MUST continue from this
                    
                    # Ensure socratic module is imported
                    socratic = _lazy_import_socratic()
                    
                    result = socratic.retry_thinking_deepen_thoughts(picked, prev1, prev2, initial_question, conversation_context)
                    
                    if result is None:
                        logging.error("retry_thinking_deepen_thoughts returned None")
                        soc_q = ""  # No socratic question for continuations
                        new_options = ["Option 1: Continue exploring", "Option 2: Analyze deeper", "Option 3: Consider alternatives"]
                    elif not isinstance(result, (list, tuple)) or len(result) != 2:
                        logging.error(f"retry_thinking_deepen_thoughts returned unexpected format: {type(result)}, value: {result}")
                        soc_q = ""  # No socratic question for continuations
                        new_options = ["Option 1: Continue exploring", "Option 2: Analyze deeper", "Option 3: Consider alternatives"]
                    else:
                        soc_q, new_options = result
                        logging.info(f"Got result: {len(new_options) if isinstance(new_options, list) else 0} continuation options")
                        
                        # For continuations, we don't want socratic questions - always set to empty
                        soc_q = ""
                        
                        # Clean and ensure 3 options
                        if not isinstance(new_options, list):
                            new_options = [str(new_options)] if new_options else []

                        new_options = [str(opt).strip() for opt in new_options if opt and str(opt).strip() and str(opt).strip().lower() != "none"]
                        
                        # Check if any option contains multiple "Distinct Line of Thought" markers (parsing issue)
                        # If so, split them properly
                        expanded_options = []
                        thought_pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"
                        for opt in new_options:
                            # Check if this option contains multiple "Distinct Line of Thought" markers
                            matches = list(re.finditer(thought_pattern, opt, flags=re.IGNORECASE))
                            if len(matches) >= 2:
                                # Split this option into multiple options
                                for i, match in enumerate(matches):
                                    start_pos = match.end()
                                    if i + 1 < len(matches):
                                        end_pos = matches[i + 1].start()
                                    else:
                                        end_pos = len(opt)
                                    split_opt = opt[start_pos:end_pos].strip()
                                    split_opt = re.sub(thought_pattern, "", split_opt, flags=re.IGNORECASE).strip()
                                    if split_opt and len(split_opt) > 10:
                                        expanded_options.append(split_opt)
                            else:
                                # Single option, clean it up
                                clean_opt = re.sub(thought_pattern, "", opt, flags=re.IGNORECASE).strip()
                                if clean_opt and len(clean_opt) > 10:
                                    expanded_options.append(clean_opt)
                        
                        # Use expanded options if we got better results
                        if len(expanded_options) >= 3:
                            new_options = expanded_options[:3]
                        elif len(expanded_options) > 0:
                            new_options = expanded_options

                        # Final check: ensure each option is separate and doesn't contain multiple "Distinct Line of Thought" markers
                        # This is a safety check in case the parsing didn't work correctly
                        final_split_options = []
                        thought_pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"
                        for opt in new_options:
                            opt_str = str(opt).strip()
                            # Check if this option contains multiple "Distinct Line of Thought" markers
                            matches = list(re.finditer(thought_pattern, opt_str, flags=re.IGNORECASE))
                            if len(matches) >= 2:
                                # Split this option into multiple options
                                for i, match in enumerate(matches):
                                    start_pos = match.end()
                                    if i + 1 < len(matches):
                                        end_pos = matches[i + 1].start()
                                    else:
                                        end_pos = len(opt_str)
                                    split_opt = opt_str[start_pos:end_pos].strip()
                                    split_opt = re.sub(thought_pattern, "", split_opt, flags=re.IGNORECASE).strip()
                                    if split_opt and len(split_opt) > 10:
                                        final_split_options.append(split_opt)
                            else:
                                # Single option, clean it up
                                clean_opt = re.sub(thought_pattern, "", opt_str, flags=re.IGNORECASE).strip()
                                if clean_opt and len(clean_opt) > 10:
                                    final_split_options.append(clean_opt)
                        
                        # Use the properly split options
                        if len(final_split_options) >= 3:
                            new_options = final_split_options[:3]
                        elif len(final_split_options) > 0:
                            new_options = final_split_options
                        
                        # Ensure we have exactly 3 distinct options (don't include the selected option)
                        if len(new_options) < 3:
                            # Pad with meaningful fallback options based on selected option
                            fallback_base = picked[:50] if picked else "this line of reasoning"
                            fallback_options = [
                                f"Option {len(new_options) + 1}: Analyze {fallback_base}...",
                                f"Option {len(new_options) + 2}: Explore implications",
                                f"Option {len(new_options) + 3}: Consider alternatives"
                            ]
                            new_options = list(new_options) + fallback_options[:3 - len(new_options)]
                        elif len(new_options) > 3:
                            new_options = new_options[:3]
                        
                        logging.info(f"Final new_options after parsing: {len(new_options)} options")
                        for i, opt in enumerate(new_options, 1):
                            logging.info(f"  Final option {i}: {opt[:100]}...")
                        
                        # Display the 3 continuation options in the chat message
                        st.markdown("**How can we continue exploring this hypothesis?**")
                        st.markdown("**Probing Questions to Deepen Understanding:**")
                        for i, opt in enumerate(new_options[:3], 1):
                            opt_display = str(opt).strip()
                            if opt_display and len(opt_display) > 5:
                                st.markdown(f"**{i}.** {opt_display}")
                            else:
                                st.markdown(f"**{i}.** Option {i}: Continue exploring")
                        
                        # CRITICAL: Save options to session state IMMEDIATELY after generating them
                        # This ensures they're available for "Current Next-Step Options" after rerun
                        # Make a copy to prevent modification
                        final_options_to_save = new_options[:3] if len(new_options) >= 3 else new_options
                        # Convert to list and make a deep copy to ensure they're not modified
                        options_list = list(final_options_to_save)
                        st.session_state._newly_generated_options = options_list
                        logging.info(f"✓ Saved {len(options_list)} options to session state immediately after generation")
                        logging.info(f"  Saved options: {[str(opt)[:80] for opt in options_list]}")
                        
                        # Verify they're actually saved
                        verify = st.session_state.get("_newly_generated_options", None)
                        if verify and len(verify) >= 3:
                            logging.info(f"  ✓ Verification: Options confirmed saved ({len(verify)} options)")
                        else:
                            logging.error(f"  ✗ Verification FAILED: Options not saved! verify={verify}")
                    
                except Exception as e:
                    logging.error(f"Error in iterative refinement: {e}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    st.error(f"Error generating new thoughts: {str(e)}")
                    # Fallback options (no socratic question for continuations)
                    soc_q = ""  # No socratic question for continuations
                    new_options = ["Option 1: Continue exploring", "Option 2: Analyze deeper", "Option 3: Consider alternatives"]
                    # Save fallback options
                    st.session_state._newly_generated_options = list(new_options)
                    logging.info(f"Saved fallback options due to error: {new_options}")
        
        # Ensure we have valid values
        if 'soc_q' not in locals():
            soc_q = ""  # No socratic question for continuations
        
        # CRITICAL: Check if new_options was set inside the chat message block
        # If not, try to get it from session state (it might have been saved there)
        if 'new_options' not in locals() or not new_options:
            logging.warning(f"new_options not in locals after chat message block, checking session state...")
            new_options = st.session_state.get("_newly_generated_options", None)
            if new_options:
                logging.info(f"  ✓ Found options in session state: {len(new_options)} options")
                logging.info(f"  Options: {[str(opt)[:80] for opt in new_options[:3]]}")
            else:
                logging.warning(f"  ✗ Options not in session state either, using fallback")
                # Include the selected option even in fallback case
                selected_fallback = picked if picked and str(picked).strip() else "Continue exploring this option"
                new_options = [selected_fallback, "Option 2: Analyze deeper", "Option 3: Consider alternatives"]
                # Save fallback options
                st.session_state._newly_generated_options = list(new_options)
        
        # Final validation: make sure we have exactly 3 separate options
        if len(new_options) != 3:
            logging.warning(f"new_options has {len(new_options)} items, expected 3. Options: {new_options}")
            # Try to split any combined options one more time
            final_validated_options = []
            thought_pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"
            for opt in new_options:
                opt_str = str(opt).strip()
                matches = list(re.finditer(thought_pattern, opt_str, flags=re.IGNORECASE))
                if len(matches) >= 2:
                    for i, match in enumerate(matches):
                        start_pos = match.end()
                        if i + 1 < len(matches):
                            end_pos = matches[i + 1].start()
                        else:
                            end_pos = len(opt_str)
                        split_opt = opt_str[start_pos:end_pos].strip()
                        split_opt = re.sub(thought_pattern, "", split_opt, flags=re.IGNORECASE).strip()
                        if split_opt and len(split_opt) > 10:
                            final_validated_options.append(split_opt)
                else:
                    clean_opt = re.sub(thought_pattern, "", opt_str, flags=re.IGNORECASE).strip()
                    if clean_opt and len(clean_opt) > 10:
                        final_validated_options.append(clean_opt)
            
            if len(final_validated_options) >= 3:
                new_options = final_validated_options[:3]
            elif len(final_validated_options) > 0:
                # Pad if needed
                while len(final_validated_options) < 3:
                    final_validated_options.append(f"Option {len(final_validated_options) + 1}: Continue exploring")
                new_options = final_validated_options[:3]
        
        # Save new options to session state for display after rerun
        # For continuations, we don't save socratic questions - only the options
        st.session_state._newly_generated_soc_q = None  # No socratic question for continuations
        
        # Ensure we have exactly 3 options before saving
        if len(new_options) < 3:
            logging.warning(f"Only {len(new_options)} options to save, expected 3. Padding with fallback options.")
            while len(new_options) < 3:
                new_options.append(f"Option {len(new_options) + 1}: Continue exploring")
        elif len(new_options) > 3:
            new_options = new_options[:3]
        
        # Log what we're saving
        logging.info(f"Saving {len(new_options)} options to session state:")
        for i, opt in enumerate(new_options, 1):
            logging.info(f"  Saved option {i}: {opt[:100]}...")
        
        # CRITICAL: Save new options to session state BEFORE rerun so they're available on next render
        # Note: Options may have already been saved immediately after generation (line 2066), but we save again here
        # to ensure they're the final validated options
        # Make a copy to prevent any modifications
        final_options_copy = list(new_options) if isinstance(new_options, list) else [str(opt) for opt in new_options]
        
        # Ensure we have exactly 3 options
        if len(final_options_copy) < 3:
            logging.warning(f"Only {len(final_options_copy)} options in final copy, padding to 3")
            while len(final_options_copy) < 3:
                final_options_copy.append(f"Option {len(final_options_copy) + 1}: Continue exploring")
        elif len(final_options_copy) > 3:
            final_options_copy = final_options_copy[:3]
        
        # Save to session state
        st.session_state._newly_generated_options = final_options_copy
        logging.info(f"Final save: Saved {len(final_options_copy)} options to _newly_generated_options before rerun")
        logging.info(f"Final options content: {[str(opt)[:80] for opt in final_options_copy]}")
        
        # Double-check they're actually saved
        verify_saved = st.session_state.get("_newly_generated_options", None)
        if verify_saved and len(verify_saved) >= 3:
            logging.info(f"✓ Verification: Options confirmed in session state: {len(verify_saved)} options")
            # Check if they match what we just saved
            if verify_saved == final_options_copy or all(str(v) == str(f) for v, f in zip(verify_saved[:3], final_options_copy[:3])):
                logging.info(f"  ✓ Options match what we saved")
            else:
                logging.warning(f"  ⚠ Options in session state don't match what we just saved!")
        else:
            logging.error(f"✗ Verification FAILED: Options NOT in session state after save!")
            # Last resort: try saving one more time
            st.session_state._newly_generated_options = final_options_copy
            logging.info(f"  Last resort: Re-saved options to session state")
        
        # CRITICAL: Ensure we have new_options before proceeding
        if 'new_options' not in locals() or not new_options:
            logging.error(f"✗ CRITICAL ERROR: new_options not defined after generation!")
            # Try to get them from session state if they were saved
            new_options = st.session_state.get("_newly_generated_options", None)
            if not new_options:
                logging.error(f"  Options not in session state either! Using fallback.")
                selected_fallback = picked if picked and str(picked).strip() else "Continue exploring this option"
                new_options = [selected_fallback, "Option 2: Analyze deeper", "Option 3: Consider alternatives"]
        
        # Ensure we have exactly 3 options
        if len(new_options) < 3:
            logging.warning(f"Only {len(new_options)} options, padding to 3")
            while len(new_options) < 3:
                new_options.append(f"Option {len(new_options) + 1}: Continue exploring")
        elif len(new_options) > 3:
            new_options = new_options[:3]
        
        # CRITICAL: Save to session state ONE FINAL TIME before rerun
        # Make absolutely sure they're saved with multiple redundant saves
        final_save_list = [str(opt) for opt in new_options[:3]]
        
        # Save to multiple keys for redundancy
        st.session_state._newly_generated_options = final_save_list
        st.session_state['_newly_generated_options'] = final_save_list  # Alternative syntax
        st.session_state._last_generated_options_backup = final_save_list  # Backup key
        
        logging.info(f"🔵 FINAL SAVE before rerun: {len(final_save_list)} options")
        logging.info(f"  Options: {[str(opt)[:80] for opt in final_save_list]}")
        
        # Verify with all three keys
        verify_final_1 = st.session_state.get("_newly_generated_options", None)
        verify_final_2 = st.session_state.get("_last_generated_options_backup", None)
        if verify_final_1 and len(verify_final_1) >= 3:
            logging.info(f"  ✓ VERIFICATION 1: Options confirmed in _newly_generated_options")
        else:
            logging.error(f"  ✗ VERIFICATION 1 FAILED! _newly_generated_options={verify_final_1}")
        if verify_final_2 and len(verify_final_2) >= 3:
            logging.info(f"  ✓ VERIFICATION 2: Options confirmed in _last_generated_options_backup")
        else:
            logging.error(f"  ✗ VERIFICATION 2 FAILED! _last_generated_options_backup={verify_final_2}")
        
        # Save new options to conversation history BEFORE rerun
        # This ensures they're persisted and can be retrieved after rerun
        logging.info("💾 Saving new options to interactions before rerun...")
        insert_interaction("assistant", new_options[0] if len(new_options) > 0 else "Option 1", "next_step_option_1")
        insert_interaction("assistant", new_options[1] if len(new_options) > 1 else "Option 2", "next_step_option_2")
        insert_interaction("assistant", new_options[2] if len(new_options) > 2 else "Option 3", "next_step_option_3")
        logging.info(f"✓ Saved {len(new_options)} options to interactions")
        
        # Verify they're in interactions now
        last_3_components = [st.session_state.interactions[i].get("component") for i in range(-3, 0) if i >= -len(st.session_state.interactions)]
        logging.info(f"  Last 3 components in interactions: {last_3_components}")
        
        # CRITICAL: Ensure stage is set
        st.session_state.stage = "hypothesis"  # Stay in hypothesis stage to show new options
        
        # Rerun to update the "Current Next-Step Options" section with the newly generated options
        logging.info("🔄 Calling st.rerun() to update display...")
        st.rerun()
    
    # Button to generate hypothesis (only if no option was selected and no additional question was asked)
    if st.button("Generate Hypothesis", type="primary", use_container_width=True):
        # Clear newly generated options when generating hypothesis
        st.session_state._newly_generated_soc_q = None
        st.session_state._newly_generated_options = None
        with st.spinner("Synthesizing hypothesis from conversation..."):
            try:
                # Build conversation context from all interactions
                conversation_context = build_conversation_context()

                # Lazy import socratic module
                socratic = _lazy_import_socratic()

                # Get the most recent selected option with fallbacks
                last_choice = view_component("last_selected_option")
                if not last_choice:
                    last_choice = opt1
                if not last_choice:
                    last_choice = view_component("next_step_option_1") or "Selected option"

                last_prev1 = view_component("last_prev1")
                if not last_prev1:
                    last_prev1 = opt2
                if not last_prev1:
                    last_prev1 = view_component("next_step_option_2") or "Previous option 1"

                last_prev2 = view_component("last_prev2")
                if not last_prev2:
                    last_prev2 = opt3
                if not last_prev2:
                    last_prev2 = view_component("next_step_option_3") or "Previous option 2"

                last_soc_q = view_component("retry_thinking_question")
                if not last_soc_q:
                    last_soc_q = view_component("socratic_pass")
                if not last_soc_q:
                    last_soc_q = view_component("clarified_question") or "How can we continue exploring this hypothesis?"

                # Ensure we have valid values
                if not last_choice or not last_soc_q:
                    st.error("Insufficient context to generate hypothesis. Please go through the conversation flow first.")
                    st.stop()

                # Experimental constraints are already included in conversation_context
                # Generate hypothesis with full conversation context
                hypothesis = generate_hypothesis_with_context(
                    last_soc_q, last_choice, last_prev1, last_prev2, conversation_context
                )

                # Ensure hypothesis is never None
                if hypothesis is None or not str(hypothesis).strip():
                    hypothesis = "Error generating hypothesis. Please check your API key and try again."
            except Exception as e:
                logging.error(f"Error in hypothesis generation button handler: {e}")
                hypothesis = f"Error generating hypothesis: {str(e)}. Please check your API key and try again."
        
        with st.chat_message("assistant"):
            st.markdown("**Hypothesis:**")
            st.markdown(hypothesis if hypothesis else "Error generating hypothesis. Please try again.")

        insert_interaction("assistant", hypothesis, "hypothesis")

        st.success("🎉 Hypothesis generation complete!")
        st.session_state.stage = "analysis"
        st.rerun()
    
    
    # Handle additional question (triggers socratic + TOT, stays in hypothesis stage)
    # REGULAR HYPOTHESIS MODE ONLY - Experimental mode should not use this
    # Process additional question ONLY if no option was selected (to avoid conflicts)
    if additional_question and additional_question.strip() and not user_choice_2:
        if st.session_state.experimental_mode:
            st.error("Error: Additional questions in hypothesis stage are only for regular hypothesis mode. Use experimental planning mode instead.")
            st.session_state.stage = "refine"
            st.rerun()
        
        with st.chat_message("user"):
            st.markdown(additional_question)
        
        # Build context from conversation so far
        context = build_conversation_context()
        contextual_question = f"{context}\n\nAdditional question: {additional_question}"
        
        with st.chat_message("assistant"):
            with st.spinner("Processing your question through socratic questioning and TOT thinking..."):
                try:
                    # Lazy import socratic module
                    socratic = _lazy_import_socratic()
                    
                    # Process through normal flow: clarify → socratic → answer → TOT
                    # REGULAR HYPOTHESIS MODE ONLY - Don't use experimental TOT here
                    clarified_followup = socratic.clarify_question(contextual_question)
                    if not clarified_followup:
                        clarified_followup = contextual_question
                    
                    socratic_questions = socratic.socratic_pass(clarified_followup)
                    if not socratic_questions:
                        socratic_questions = "How can we explore this further?"
                    
                    # NEW: Answer the socratic questions to build deeper reasoning
                    socratic_answers = None
                    try:
                        socratic_answers = socratic.socratic_answer_questions(clarified_followup, socratic_questions)
                        if not socratic_answers:
                            logging.warning("Could not generate socratic answers, will use questions instead")
                    except Exception as e:
                        logging.error(f"Error generating socratic answers: {e}")
                        # Continue with questions if answers fail
                    
                    # REGULAR HYPOTHESIS MODE ONLY - Standard TOT generation (with answers if available)
                    thoughts = socratic.tot_generation(socratic_questions, clarified_followup, socratic_answers)
                    
                    # Ensure thoughts is a list with exactly 3 items
                    if not thoughts or len(thoughts) == 0:
                        thoughts = ["No thoughts generated", "Please check API key", "Retry with different question"]
                    elif len(thoughts) < 3:
                        thoughts = list(thoughts) + [""] * (3 - len(thoughts))
                    elif len(thoughts) > 3:
                        thoughts = thoughts[:3]
                    
                    # Convert thoughts to next-step options for hypothesis stage
                    first_thought = thoughts[0] if len(thoughts) > 0 else "Option 1: Continue exploring"
                    second_thought = thoughts[1] if len(thoughts) > 1 else "Option 2: Continue exploring"
                    third_thought = thoughts[2] if len(thoughts) > 2 else "Option 3: Continue exploring"
                    
                    # Display the new options
                    st.markdown("**Clarified Question:**")
                    st.markdown(clarified_followup if clarified_followup else "No clarified question generated")
                    st.markdown("**Socratic Pass:**")
                    st.markdown(socratic_questions if socratic_questions else "How can we explore this further?")
                    
                    # Display socratic answers if available
                    if socratic_answers and socratic_answers.strip():
                        st.markdown("**Socratic Reasoning (Answers to Probing Questions):**")
                        st.markdown(socratic_answers)
                    
                    st.markdown("**New Next-Step Options:**")
                    st.markdown(f"**1.** {first_thought}")
                    st.markdown(f"**2.** {second_thought}")
                    st.markdown(f"**3.** {third_thought}")
                except Exception as e:
                    logging.error(f"Error processing additional question: {e}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    st.error(f"Error processing question: {str(e)}")
                    # Fallback options
                    first_thought = "Option 1: Continue exploring"
                    second_thought = "Option 2: Continue exploring"
                    third_thought = "Option 3: Continue exploring"
        
        # Save to interactions and update next-step options
        insert_interaction("user", additional_question, "additional_question")
        insert_interaction("assistant", clarified_followup if 'clarified_followup' in locals() else additional_question, "clarified_question")
        insert_interaction("assistant", socratic_questions if 'socratic_questions' in locals() else "How can we explore this further?", "socratic_pass")
        # CRITICAL: Save socratic answers so they persist after rerun
        if 'socratic_answers' in locals() and socratic_answers and socratic_answers.strip():
            insert_interaction("assistant", socratic_answers, "socratic_answers")
        insert_interaction("assistant", socratic_questions if 'socratic_questions' in locals() else "How can we explore this further?", "retry_thinking_question")
        insert_interaction("assistant", first_thought if 'first_thought' in locals() else "Option 1: Continue exploring", "next_step_option_1")
        insert_interaction("assistant", second_thought if 'second_thought' in locals() else "Option 2: Continue exploring", "next_step_option_2")
        insert_interaction("assistant", third_thought if 'third_thought' in locals() else "Option 3: Continue exploring", "next_step_option_3")
        
        # Save new options to session state for immediate display
        st.session_state._newly_generated_soc_q = socratic_questions if 'socratic_questions' in locals() else "How can we explore this further?"
        st.session_state._newly_generated_options = [first_thought if 'first_thought' in locals() else "Option 1: Continue exploring",
                                                      second_thought if 'second_thought' in locals() else "Option 2: Continue exploring",
                                                      third_thought if 'third_thought' in locals() else "Option 3: Continue exploring"]
        
        # Stay in hypothesis stage to show new options (iterative refinement)
        st.session_state.stage = "hypothesis"
        st.rerun()

elif st.session_state.stage == "analysis":
    # REGULAR HYPOTHESIS MODE ANALYSIS - Experimental mode should not reach here
    if st.session_state.experimental_mode:
        st.error("Error: Experimental planning mode should not reach analysis stage. Redirecting to experimental outputs...")
        st.session_state.stage = "experimental_outputs"
        st.rerun()
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Hypothesis and Producing Report..."):
            try:
                # Lazy import socratic module
                socratic = _lazy_import_socratic()

                socratic_question = view_component("retry_thinking_question") or view_component("socratic_pass") or ""
                hypothesis = view_component("hypothesis")

                # CRITICAL: Validate that we actually have a hypothesis, not socratic questions/answers
                if not hypothesis:
                    st.error("No hypothesis found. Please generate a hypothesis first.")
                    st.session_state.stage = "hypothesis"
                    st.rerun()
                
                # CRITICAL: Check if hypothesis is actually a hypothesis report or socratic content
                if "Hypothesis Report" in str(hypothesis) or "Hypothesis Evaluation Report" in str(hypothesis):
                    logging.error("❌ Found hypothesis report in 'hypothesis' field - this should not happen!")
                    logging.error("  Hypothesis content starts with: " + str(hypothesis)[:200])
                    st.error("Error: Found hypothesis report instead of hypothesis. Please generate a hypothesis first.")
                    st.session_state.stage = "hypothesis"
                    st.rerun()
                
                # Check if it's actually socratic questions/answers
                if "What specific" in str(hypothesis) and "Reasoning:" in str(hypothesis):
                    logging.error("❌ Found socratic questions in 'hypothesis' field - this should not happen!")
                    st.error("Error: Found socratic questions instead of hypothesis. Please generate a hypothesis first.")
                    st.session_state.stage = "hypothesis"
                    st.rerun()

                # REGULAR HYPOTHESIS MODE - Standard analysis without experimental constraints
                analysis_rubric = socratic.local_hypothesis_analysis_fallback(hypothesis, socratic_question)

                if not analysis_rubric or not str(analysis_rubric).strip():
                    analysis_rubric = "Analysis generated but content is empty. Please check your API key and try again."
            except Exception as e:
                logging.error(f"Error in analysis stage: {e}")
                analysis_rubric = f"Error analyzing hypothesis: {str(e)}. Please check your API key and try again."

        st.markdown(analysis_rubric if analysis_rubric else "Error generating analysis. Please try again.")

    insert_interaction("assistant", analysis_rubric, "analysis_rubric")
    st.success("Analysis complete!")

elif st.session_state.stage == "followup":
    st.header("Follow-up Question")
    lh = st.session_state.experimental_constraints["liquid_handling"]
    plate_format = lh["plate_format"]
    materials = lh["materials"] if lh["materials"] else ["Cs_uL", "BDA_uL", "BDA_2_uL"]
    
    # Format materials for CSV (add _uL suffix if not present)
    csv_materials = []
    for mat in materials:
        if mat.endswith("_uL"):
            csv_materials.append(mat)
        else:
            csv_materials.append(f"{mat}_uL")

    # Initialize variables
    worklist_content = ""
    csv_filename = "worklist.csv"
    layout_content = ""
    layout_filename = "plate_layout.txt"
    protocol_content = None
    protocol_filename = None
    
    if hypothesis_text:
        try:
            # Generate worklist with actual materials
            worklist_content = generate_worklist(hypothesis_text, plate_format, csv_materials)
            csv_filename = f"worklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            layout_filename = f"plate_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            # Pass worklist_content to generate_plate_layout so it shows actual compositions
            layout_content = generate_plate_layout(hypothesis_text, plate_format, worklist_content)
        except Exception as e:
            logging.error(f"Error generating experimental outputs: {e}")
            st.error(f"Error generating experimental outputs: {str(e)}")
        
        # Generate protocol if needed
        if "Opentrons" in lh["instruments"]:
            csv_path = lh.get("csv_path", "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/")
            protocol_content = generate_opentrons_protocol(csv_filename, csv_materials, csv_path)
            protocol_filename = f"opentrons_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        
        col1, col2, col3 = st.columns(3)

        with col1:
            try:
                if worklist_content:
                    st.download_button(
                            label="Download Worklist (.csv)",
                        data=worklist_content,
                        file_name=csv_filename,
                        mime="text/csv",
                        help="Download CSV worklist for Tecan/Opentrons liquid handlers",
                        key=f"download_worklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                else:
                    st.warning("No worklist content available")
            except Exception as e:
                logging.error(f"Error creating worklist download button: {e}")
                st.error(f"Error creating download: {str(e)}")
            
            # Upload to Jupyter button
            jup_config = st.session_state.jupyter_config
            if jup_config["upload_enabled"]:
                if st.button("Upload CSV to Jupyter", key="upload_csv"):
                    with st.spinner("Uploading to Jupyter..."):
                        success, message = upload_to_jupyter(
                            jup_config["server_url"],
                            jup_config["token"],
                            worklist_content,
                            csv_filename,
                            jup_config["notebook_path"]
                        )
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

            with col2:
                try:
                    if layout_content:
                        st.download_button(
                            label="Download Plate Layout (.txt)",
                            data=layout_content,
                            file_name=layout_filename,
                            mime="text/plain",
                            help="Download visual plate layout for experiment planning",
                            key=f"download_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
                    else:
                        st.warning("No plate layout content available")
                except Exception as e:
                    logging.error(f"Error creating layout download button: {e}")
                    st.error(f"Error creating download: {str(e)}")
                
                # Upload to Jupyter button
                if jup_config["upload_enabled"]:
                    if st.button("Upload Layout to Jupyter", key="upload_layout"):
                        with st.spinner("Uploading to Jupyter..."):
                            success, message = upload_to_jupyter(
                                jup_config["server_url"],
                                jup_config["token"],
                                layout_content,
                                layout_filename,
                                jup_config["notebook_path"]
                            )
                            if success:
                                st.success(message)
                            else:
                                st.error(message)

            with col3:
                # Generate Opentrons protocol if Opentrons is selected
                if "Opentrons" in lh["instruments"] and protocol_content:
                    try:
                        st.download_button(
                            label="🤖 Download Opentrons Protocol (.py)",
                            data=protocol_content,
                            file_name=protocol_filename,
                            mime="text/x-python",
                            help="Download Python protocol file for Opentrons robot",
                            key=f"download_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
                    except Exception as e:
                        logging.error(f"Error creating protocol download button: {e}")
                        st.error(f"Error creating download: {str(e)}")
                    
                    # Upload to Jupyter button
                    if jup_config["upload_enabled"]:
                        if st.button("Upload Protocol to Jupyter", key="upload_protocol"):
                            with st.spinner("Uploading to Jupyter..."):
                                success, message = upload_to_jupyter(
                                    jup_config["server_url"],
                                    jup_config["token"],
                                    protocol_content,
                                    protocol_filename,
                                    jup_config["notebook_path"]
                                )
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                else:
                    st.info("💡 Enable Opentrons in constraints to generate protocol")
            
            # Auto-upload if enabled
            if jup_config["upload_enabled"] and "auto_uploaded" not in st.session_state:
                st.markdown("---")
                st.markdown("### Auto-Uploading to Jupyter...")
                upload_results = []
                
                # Upload CSV
                success, message = upload_to_jupyter(
                    jup_config["server_url"],
                    jup_config["token"],
                    worklist_content,
                    csv_filename,
                    jup_config["notebook_path"]
                )
                upload_results.append(("CSV Worklist", success, message))
                
                # Upload Layout
                success, message = upload_to_jupyter(
                    jup_config["server_url"],
                    jup_config["token"],
                    layout_content,
                    layout_filename,
                    jup_config["notebook_path"]
                )
                upload_results.append(("Plate Layout", success, message))
                
                # Upload Protocol if Opentrons is enabled
                if "Opentrons" in lh["instruments"] and protocol_content:
                    success, message = upload_to_jupyter(
                        jup_config["server_url"],
                        jup_config["token"],
                        protocol_content,
                        protocol_filename,
                        jup_config["notebook_path"]
                    )
                    upload_results.append(("Opentrons Protocol", success, message))
                
                # Display results
                for file_type, success, message in upload_results:
                    if success:
                        st.success(f"✅ {file_type}: {message}")
                    else:
                        st.warning(f"{file_type}: {message}")
                
                st.session_state.auto_uploaded = True

        # Show volume constraints info
        st.info(f"Volume Constraint: Max {lh['max_volume_per_mixture']} µL per mixture | Plate: {lh['plate_format']} | Materials: {', '.join(csv_materials)} | Instruments: {', '.join(lh['instruments']) if lh['instruments'] else 'None specified'}")

    # Add follow-up option
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ask Follow-up Question", type="primary"):
            start_followup_question()
            st.rerun()
    with col2:
        if st.button("Start New Question"):
            save_to_history(view_component("original_question"), view_component("hypothesis"))
            clear_conversation()
    st.rerun()

elif st.session_state.stage == "followup":
    st.header("Follow-up Question")

    # Show previous context
    if st.session_state.conversation_history:
        with st.expander("Previous Conversation Context", expanded=True):
            latest = st.session_state.conversation_history[-1]
            st.markdown(f"**Previous Question:** {latest['question']}")
            if latest['hypothesis']:
                st.markdown(f"**Previous Hypothesis:** {latest['hypothesis'][:300]}..." if len(latest['hypothesis']) > 300 else f"**Previous Hypothesis:** {latest['hypothesis']}")

    followup_question = st.text_input(
        "Ask a follow-up question based on the previous hypothesis:",
        placeholder="e.g., 'What experimental methods would validate this hypothesis?'"
    )

    if st.button("🚀 Process Follow-up", type="primary"):
        if followup_question.strip():
            # Get context from previous conversation
            context = get_context_for_followup()

            # Process the follow-up question with context - same flow as initial question
            with st.spinner("Processing follow-up question..."):
                # Create a contextualized question
                contextual_question = f"{context}\n\nFollow-up question: {followup_question}"

                # Process through the normal flow but with context
                clarified_followup = socratic.clarify_question(contextual_question)
                socratic_questions = socratic.socratic_pass(clarified_followup)
                
                # NEW: Answer the socratic questions to build deeper reasoning
                socratic_answers = None
                try:
                    socratic_answers = socratic.socratic_answer_questions(clarified_followup, socratic_questions)
                    if not socratic_answers:
                        logging.warning("Could not generate socratic answers, will use questions instead")
                except Exception as e:
                    logging.error(f"Error generating socratic answers: {e}")
                    # Continue with questions if answers fail
                
                thoughts = socratic.tot_generation(socratic_questions, clarified_followup, socratic_answers)
                
                # Ensure thoughts is a list with exactly 3 items
                if not thoughts or len(thoughts) == 0:
                    thoughts = ["No thoughts generated", "Please check API key", "Retry with different question"]
                elif len(thoughts) < 3:
                    thoughts = list(thoughts) + [""] * (3 - len(thoughts))
                elif len(thoughts) > 3:
                    thoughts = thoughts[:3]  # Take only first 3

            # Display results in chat format (same as initial question)
            with st.chat_message("user"):
                st.markdown(followup_question)

            with st.chat_message("assistant"):
                st.markdown("**Clarified Question:**")
                st.markdown(clarified_followup if clarified_followup else "No clarified question generated")
                st.markdown("**Socratic Pass:**")
                st.markdown(socratic_questions if socratic_questions else "No socratic questions generated")
                
                # Display socratic answers if available
                if socratic_answers and socratic_answers.strip():
                    st.markdown("**Socratic Reasoning (Answers to Probing Questions):**")
                    st.markdown(socratic_answers)
                
                st.markdown("**Generated Thoughts:**")
                first_thought = thoughts[0] if len(thoughts) > 0 else ""
                second_thought = thoughts[1] if len(thoughts) > 1 else ""
                third_thought = thoughts[2] if len(thoughts) > 2 else ""
                st.markdown(first_thought)
                st.markdown(second_thought)
                st.markdown(third_thought)

            # Save to history for follow-up workflow
            insert_interaction("user", followup_question, "original_question")
            insert_interaction("assistant", clarified_followup, "clarified_question")
            insert_interaction("assistant", socratic_questions, "socratic_pass")
            # CRITICAL: Save socratic answers so they persist after rerun
            if socratic_answers and socratic_answers.strip():
                insert_interaction("assistant", socratic_answers, "socratic_answers")
            insert_interaction("assistant", first_thought, "first_thought_1")
            insert_interaction("assistant", second_thought, "second_thought_1")
            insert_interaction("assistant", third_thought, "third_thought_1")

            # Move to refine stage so user can choose thoughts (same workflow as initial question)
            st.session_state.stage = "refine"
            st.rerun()
        else:
            st.warning("Please enter a follow-up question.")
