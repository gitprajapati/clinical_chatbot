import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Set, Tuple, Optional
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv  

load_dotenv() 

# Set page config
st.set_page_config(
    page_title="Clinical Trial Chatbot",
    page_icon="🏥",
    layout="wide"
)

# Your existing classes here
class ClinicalTrialTerminologyMapper:
    """Handles all terminology variations and mappings for clinical trials"""
    
    def __init__(self):
        # Trial name variations
        self.trial_patterns = {
            'CHECKMATE': ['CM', 'CHECKMATE', 'CHECK MATE', 'CHECK-MATE'],
            'COMBI': ['COMBI', 'COMBINATION'],
            'DREAMSEQ': ['DREAMSEQ', 'DREAM SEQ', 'DREAM-SEQ'],
            'IMSPIRE': ['IMSPIRE', 'IM SPIRE', 'IM-SPIRE'],
            'COBRIM': ['COBRIM', 'CO BRIM', 'CO-BRIM']
        }
        
        # Efficacy metrics mappings
        self.efficacy_terms = {
            'ORR': ['ORR', 'objective response rate', 'overall response rate', 'response rate'],
            'CR': ['CR', 'complete response', 'complete responses'],
            'PR': ['PR', 'partial response', 'partial responses'],
            'mPFS': ['mPFS', 'median PFS', 'median progression free survival', 'median progression-free survival', 'PFS'],
            'mOS': ['mOS', 'median OS', 'median overall survival', 'OS', 'overall survival'],
            'mDOR': ['mDOR', 'mDoR', 'median DOR', 'median DoR', 'median duration of response', 'DOR', 'DoR', 'duration of response'],
            'DCR': ['DCR', 'disease control rate']
        }
        
        # Safety metrics mappings
        self.safety_terms = {
            'Gr ≥3 TRAEs': [
                'Gr ≥3 TRAEs', 'Gr 3+ TRAEs', 'Grade 3 and above TRAEs', 'High grade TRAEs',
                'Gr ≥3 treatment related adverse events', 'Gr ≥3 treatment related AEs',
                'Gr ≥3 treatment-related adverse events', 'Gr ≥3 treatment-related AEs',
                'Gr 3+ treatment related adverse events', 'Gr 3+ treatment related AEs',
                'Gr 3+ treatment-related adverse events', 'Gr 3+ treatment-related AEs',
                'Grade 3 and above treatment related adverse events', 'Grade 3 and above treatment related AEs',
                'Grade 3 and above treatment-related adverse events', 'Grade 3 and above treatment-related AEs',
                'High grade treatment-related AEs', 'High grade treatment-related adverse events',
                'Grade ≥3 TRAEs', 'severe treatment related adverse events'
            ],
            'Gr 3/4 TRAEs': [
                'Gr 3/4 TRAEs', 'Grade 3/4 TRAEs', 'Grade 3-4 TRAEs', 'Gr 3-4 TRAEs',
                'Grade 3 or 4 treatment related adverse events', 'Gr 3/4 treatment-related AEs'
            ],
            'Gr ≥3 TEAEs': [
                'Gr ≥3 TEAEs', 'Gr 3+ TEAEs', 'Grade 3 and above TEAEs', 'High grade TEAEs',
                'Gr ≥3 treatment emergent adverse events', 'Gr ≥3 treatment emergent AEs',
                'Grade ≥3 TEAEs', 'severe treatment emergent adverse events'
            ],
            'Gr ≥3 irAEs': [
                'Gr ≥3 irAEs', 'Gr 3+ irAEs', 'Grade 3 and above irAEs', 'High grade irAEs',
                'Gr ≥3 immune related adverse events', 'Gr ≥3 immune-related adverse events',
                'Grade ≥3 irAEs', 'severe immune related adverse events'
            ],
            'Deaths': [
                'deaths', 'mortality', 'fatalities', 'death count', 'patients died',
                'treatment related deaths', 'treatment-related deaths', 'Gr 5 TRAEs',
                'Grade 5 TRAEs', 'Tx-related Deaths', 'All Deaths', 'Gr 5 AEs'
            ]
        }
        
        # Phase variations
        self.phase_terms = {
            'Ph3': ['Ph3', 'Phase 3', 'Phase III', 'Phase3', 'Ph 3', 'phase 3'],
            'Ph2': ['Ph2', 'Phase 2', 'Phase II', 'Phase2', 'Ph 2', 'phase 2'],
            'Ph1': ['Ph1', 'Phase 1', 'Phase I', 'Phase1', 'Ph 1', 'phase 1']
        }
        
        # Clinical outcome groupings
        self.outcome_groups = {
            'clinical outcomes': [
                'ORR', 'CR', 'PR', 'mPFS', 'mOS', 'mDOR', 'DCR',
                'Gr ≥3 TRAEs', 'Gr ≥3 TEAEs', 'Gr ≥3 irAEs', 'Gr ≥3 AEs',
                'Gr 3/4 TRAEs', 'Gr 3/4 TEAEs', 'Gr 3/4 irAEs', 'Gr 3/4 AEs'
            ],
            'efficacy outcomes': ['ORR', 'CR', 'PR', 'mPFS', 'mOS', 'mDOR', 'DCR'],
            'safety outcomes': [
                'Gr ≥3 TRAEs', 'Gr ≥3 TEAEs', 'Gr ≥3 irAEs', 'Gr ≥3 AEs',
                'Gr 3/4 TRAEs', 'Gr 3/4 TEAEs', 'Gr 3/4 irAEs', 'Gr 3/4 AEs',
                'Deaths', 'Tx-related Deaths'
            ]
        }
        
        # Drug/regimen terms
        self.drug_terms = {
            'nivo': ['nivo', 'nivolumab', 'opdivo'],
            'ipi': ['ipi', 'ipilimumab', 'yervoy'],
            'atezo': ['atezo', 'atezolizumab', 'tecentriq'],
            'cobi': ['cobi', 'cobimetinib', 'cotellic'],
            'vemu': ['vemu', 'vemurafenib', 'zelboraf'],
            'dab': ['dab', 'dabrafenib', 'tafinlar'],
            'tram': ['tram', 'trametinib', 'mekinist']
        }

    def normalize_trial_name(self, trial_name: str) -> List[str]:
        """Generate all possible variations of a trial name"""
        variations = set()
        trial_upper = trial_name.upper()
        
        # Extract numbers
        numbers = re.findall(r'\d{3,}', trial_upper)
        
        # Check for known trial prefixes
        for prefix, variants in self.trial_patterns.items():
            if any(v in trial_upper for v in variants):
                for num in numbers:
                    for v in variants:
                        variations.add(f"{v}-{num}")
                        variations.add(f"{v}{num}")
                        variations.add(f"{v} {num}")
        
        # Add the original and number-only variations
        variations.add(trial_upper)
        variations.update(numbers)
        
        return list(variations)
    
    def map_term_to_column(self, term: str, available_columns: List[str]) -> Optional[str]:
        """Map a user term to actual column name in the dataframe"""
        term_lower = term.lower()
        
        # Check all mappings
        all_mappings = {
            **self.efficacy_terms,
            **self.safety_terms,
            **self.phase_terms
        }
        
        for column, variations in all_mappings.items():
            if any(v.lower() == term_lower for v in variations):
                # Find matching column in available columns
                for col in available_columns:
                    if column.lower() in col.lower() or col.lower() in column.lower():
                        return col
        
        return None
    
    def extract_requested_metrics(self, query: str, available_columns: List[str]) -> List[str]:
        """Extract all metrics mentioned in the query"""
        query_lower = query.lower()
        requested_metrics = []
        
        # Check for outcome group mentions
        for group, metrics in self.outcome_groups.items():
            if group in query_lower:
                for metric in metrics:
                    col = self.map_term_to_column(metric, available_columns)
                    if col and col not in requested_metrics:
                        requested_metrics.append(col)
        
        # Check for individual metric mentions
        for metric_group in [self.efficacy_terms, self.safety_terms]:
            for metric, variations in metric_group.items():
                if any(v.lower() in query_lower for v in variations):
                    col = self.map_term_to_column(metric, available_columns)
                    if col and col not in requested_metrics:
                        requested_metrics.append(col)
        
        return requested_metrics


class GeneralPurposeClinicalChatbot:
    """General purpose chatbot for clinical trial data analysis"""
    
    def __init__(self, df: pd.DataFrame, model: str = "gpt-4o"):
        self.df = df
        self.model = model
        self.mapper = ClinicalTrialTerminologyMapper()
        self.trial_column = self._identify_trial_column()
        
    def _identify_trial_column(self) -> Optional[str]:
        """Identify the column containing trial names"""
        possible_names = ['Trial Acronym/ID', 'Trial Name', 'Trial ID', 'Study Name', 'Study ID']
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None
    
    def _extract_trials_from_query(self, query: str) -> List[str]:
        """Extract all trial references from the query"""
        found_trials = set()
        query_upper = query.upper()
        
        # Look for trial patterns
        patterns = [
            r'(?:CM|CHECKMATE|CHECK-MATE)[-\s]?\d{3,}',
            r'(?:COMBI|COMBINATION)[-\s]?[A-Z]',
            r'DREAMSEQ|DREAM[-\s]?SEQ',
            r'IMSPIRE\d{3,}|IM[-\s]?SPIRE\d{3,}',
            r'[A-Z]+\d{3,}',  # Generic pattern
            r'\b\d{3}\b'  # Just numbers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            found_trials.update(matches)
        
        # Map to actual trial names in data
        if self.trial_column:
            actual_trials = []
            for found in found_trials:
                variations = self.mapper.normalize_trial_name(found)
                for trial in self.df[self.trial_column].unique():
                    if pd.notna(trial):
                        trial_str = str(trial).upper()
                        if any(v in trial_str or trial_str in v for v in variations):
                            actual_trials.append(trial)
            
            return list(set(actual_trials))
        
        return list(found_trials)
    
    def _create_context_aware_agent(self, filtered_df: Optional[pd.DataFrame] = None) -> any:
        """Create an agent with comprehensive context"""
        df_to_use = filtered_df if filtered_df is not None else self.df
        
        # Build comprehensive context
        context_parts = []
        
        # Available trials
        if self.trial_column:
            trials = df_to_use[self.trial_column].unique()
            trial_list = [str(t) for t in trials if pd.notna(t)]
            context_parts.append(f"Available trials: {', '.join(trial_list)}")
        
        # Available columns with categories
        efficacy_cols = [col for col in df_to_use.columns if any(
            term in col.upper() for term in ['ORR', 'CR', 'PR', 'PFS', 'OS', 'DOR', 'DCR']
        )]
        safety_cols = [col for col in df_to_use.columns if any(
            term in col.upper() for term in ['TRAE', 'TEAE', 'IRAE', 'AE', 'DEATH', 'GRADE', 'GR']
        )]
        
        if efficacy_cols:
            context_parts.append(f"Efficacy metrics: {', '.join(efficacy_cols)}")
        if safety_cols:
            context_parts.append(f"Safety metrics: {', '.join(safety_cols)}")
        
        context = "\n".join(context_parts)
        
        prefix = f"""You are an expert clinical trial data analyst with deep knowledge of oncology trials.

{context}

IMPORTANT INSTRUCTIONS:

1. TRIAL NAME MATCHING:
   - Be flexible with trial names (e.g., CM-067 = CHECKMATE-067 = CHECKMATE067)
   - Use .str.contains() with case=False for searching
   - Always show ALL rows/arms for each trial

2. TERMINOLOGY UNDERSTANDING:
   - "Clinical outcomes" includes both efficacy (ORR, CR, PFS, OS, DOR) and safety metrics
   - "Efficacy outcomes" = ORR, CR, PFS, OS, DOR
   - "Safety outcomes" = adverse events (TRAEs, TEAEs, irAEs) and deaths
   - "High grade" = Grade ≥3 or Grade 3/4
   - "Deaths" can be called mortality, fatalities, or treatment-related deaths

3. SEARCHING STRATEGY:
   - For trials: df[df['{self.trial_column}'].str.contains('pattern', case=False, na=False)]
   - For multiple trials: df[df['{self.trial_column}'].str.contains('pattern1|pattern2', case=False, na=False)]
   - Always check if requested data exists before analyzing

4. COMPARISON INSTRUCTIONS:
   - When comparing trials, create side-by-side comparisons
   - Calculate differences when appropriate
   - Highlight which trial performs better on each metric
   - Note any missing data as "NR" or "Not Reported"

5. OUTPUT FORMAT:
   - Use clear headers for each trial
   - Present data in an organized, easy-to-read format
   - Include units (%, months, etc.) where applicable
   - For missing data, explicitly state "Not Reported" or "NR"

Remember: Each trial may have multiple treatment arms - analyze them all unless specified otherwise.
"""
        
        return create_pandas_dataframe_agent(
            ChatOpenAI(model=self.model, temperature=0),
            df_to_use,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=prefix,
            allow_dangerous_code=True,
            max_iterations=15,
            handle_parsing_errors=True
        )
    
    def query(self, question: str) -> Dict[str, any]:
        """Process any clinical trial question"""
        # Extract trials and metrics from query
        mentioned_trials = self._extract_trials_from_query(question)
        requested_metrics = self.mapper.extract_requested_metrics(question, self.df.columns)
        
        # Create info dict for UI
        info = {
            "trials_detected": mentioned_trials,
            "metrics_requested": requested_metrics
        }
        
        # Filter dataframe if specific trials mentioned
        if mentioned_trials and self.trial_column:
            pattern = '|'.join([re.escape(str(trial)) for trial in mentioned_trials])
            filtered_df = self.df[self.df[self.trial_column].str.contains(pattern, case=False, na=False, regex=True)]
            
            if filtered_df.empty:
                info["warning"] = "Could not find the mentioned trials in the dataset"
                agent = self._create_context_aware_agent()
            else:
                info["rows_found"] = len(filtered_df)
                agent = self._create_context_aware_agent(filtered_df)
        else:
            agent = self._create_context_aware_agent()
        
        # Execute query
        try:
            result = agent.invoke({"input": question})
            return {"output": result['output'], "info": info}
        except Exception as e:
            return {"output": f"Error: {str(e)}", "info": info}


# Streamlit App
def main():
    st.title("🏥 Clinical Trial Chatbot")
    st.markdown("Ask questions about clinical trial data in natural language")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        
         
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        
        
    default_file_path = "Sample Data.xlsx"
    data_source = None
    data_source_name = None

    # Determine which data source to use
    if uploaded_file is not None:
        data_source = uploaded_file
        data_source_name = uploaded_file.name
    elif os.path.exists(default_file_path):
        data_source = default_file_path
        data_source_name = default_file_path      
            
    

    if data_source is not None:
        if 'data_source_name' not in st.session_state or st.session_state.data_source_name != data_source_name:
            with st.spinner(f"Loading data from {data_source_name}..."):
                try:
                    df = pd.read_excel(data_source)
                    # Clean column names
                    df.columns = df.columns.str.strip().str.replace("\n", " ").str.replace("  ", " ")
                    
                    # Store data and chatbot in session state
                    st.session_state.df = df
                    st.session_state.chatbot = GeneralPurposeClinicalChatbot(df)
                    st.session_state.data_source_name = data_source_name
                    st.session_state.messages = []  # Clear chat history on new data load
                    st.success(f"Data from **{data_source_name}** loaded successfully!")
                    st.rerun() # Rerun to update the UI immediately
                except Exception as e:
                    st.error(f"Error loading data file: {e}")
                    # Clear session state on failure
                    st.session_state.clear()
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "info" in message:
                    with st.expander("🔍 Query Analysis"):
                        if message["info"].get("trials_detected"):
                            st.write("**Trials detected:**", ", ".join(message["info"]["trials_detected"]))
                        if message["info"].get("metrics_requested"):
                            st.write("**Metrics requested:**", ", ".join(message["info"]["metrics_requested"]))
                        if message["info"].get("rows_found"):
                            st.write("**Rows analyzed:**", message["info"]["rows_found"])
                        if message["info"].get("warning"):
                            st.warning(message["info"]["warning"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the clinical trials..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    result = st.session_state.chatbot.query(prompt)
                    st.markdown(result["output"])
                    
                    # Show analysis info
                    if result.get("info"):
                        with st.expander("🔍 Query Analysis"):
                            if result["info"].get("trials_detected"):
                                st.write("**Trials detected:**", ", ".join(result["info"]["trials_detected"]))
                            if result["info"].get("metrics_requested"):
                                st.write("**Metrics requested:**", ", ".join(result["info"]["metrics_requested"]))
                            if result["info"].get("rows_found"):
                                st.write("**Rows analyzed:**", result["info"]["rows_found"])
                            if result["info"].get("warning"):
                                st.warning(result["info"]["warning"])
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["output"],
                        "info": result.get("info", {})
                    })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
    else:
        # Welcome message when no file is uploaded
        st.info("👈 Please upload an Excel file with clinical trial data to get started")
        
    

if __name__ == "__main__":
    main()