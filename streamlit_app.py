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

class ClinicalTrialKeywordMatcher:
    """Intelligent, scalable keyword-based matching for any clinical trial data"""
    
    def __init__(self):
        # Base drug/treatment keywords - easily expandable
        self.base_drugs = {
            'pembrolizumab': ['pembrolizumab', 'pembro', 'keytruda'],
            'nivolumab': ['nivolumab', 'nivo', 'opdivo'],
            'ipilimumab': ['ipilimumab', 'ipi', 'yervoy'],
            'relatlimab': ['relatlimab', 'opdualag'],
            'atezolizumab': ['atezolizumab', 'atezo', 'tecentriq'],
            'durvalumab': ['durvalumab', 'imfinzi'],
            'avelumab': ['avelumab', 'bavencio'],
            'cemiplimab': ['cemiplimab', 'libtayo'],
            'dostarlimab': ['dostarlimab', 'jemperli'],
            'spartalizumab': ['spartalizumab'],
            'vemurafenib': ['vemurafenib', 'zelboraf'],
            'dabrafenib': ['dabrafenib', 'tafinlar'],
            'trametinib': ['trametinib', 'mekinist'],
            'cobimetinib': ['cobimetinib', 'cotellic'],
            'binimetinib': ['binimetinib', 'mektovi'],
            'encorafenib': ['encorafenib', 'braftovi'],
            'lenvatinib': ['lenvatinib', 'lenvima'],
            'axitinib': ['axitinib', 'inlyta'],
            'bevacizumab': ['bevacizumab', 'avastin'],
            'temozolomide': ['temozolomide', 'temodar'],
            'dacarbazine': ['dacarbazine', 'dtic'],
            'til': ['til', 'tils', 'tumor infiltrating lymphocytes', 'lifileucel'],
            'il-2': ['il-2', 'interleukin-2', 'aldesleukin'],
            'interferon': ['interferon', 'ifn'],
            'ipilimumab': ['ipilimumab', 'ipi', 'yervoy']
        }
        
        # Combination indicators
        self.combination_indicators = ['+', '→', 'plus', 'with', 'and', 'combination', 'combo', '/', ' + ', ' and ']
        
        # Monotherapy indicators
        self.monotherapy_indicators = ['mono', 'monotherapy', 'single agent', 'alone', 'single-agent']
        
        # Trial name patterns (expandable)
        self.trial_patterns = {
            'checkmate': ['checkmate', 'cm-', 'cm'],
            'keynote': ['keynote', 'kn-', 'kn'],
            'relativity': ['relativity', 'rel'],
            'imspire': ['imspire', 'im-spire'],
            'combi': ['combi-', 'combi'],
            'dreamseq': ['dreamseq', 'dream-seq'],
            'columbus': ['columbus'],
            'echo': ['echo-'],
            'leap': ['leap-'],
            'cosmic': ['cosmic-'],
            'illuminate': ['illuminate-']
        }
        
        # Outcome categories (expandable)
        self.outcome_categories = {
            'efficacy': ['orr', 'cr', 'pr', 'mpfs', 'mos', 'mdor', 'dcr', 'pfs', 'os', 'response', 'survival'],
            'safety': ['traes', 'teaes', 'iraes', 'aes', 'adverse', 'toxicity', 'deaths', 'grade', 'gr'],
            'biomarker': ['pd-l1', 'pdl1', 'microsatellite', 'msi', 'tmb', 'tumor mutation burden'],
            'quality_of_life': ['qol', 'quality of life', 'patient reported', 'pro']
        }

    def extract_drugs_from_text(self, text: str) -> List[str]:
        """Extract all mentioned drugs from text using intelligent pattern matching"""
        text_lower = text.lower().strip()
        found_drugs = []
        
        for drug, variants in self.base_drugs.items():
            if any(variant.lower() in text_lower for variant in variants):
                found_drugs.append(drug)
        
        return found_drugs

    def determine_treatment_type(self, text: str, found_drugs: List[str]) -> str:
        """Intelligently determine if treatment is mono, combo, or specific combination"""
        text_lower = text.lower().strip()
        
        # Check for explicit monotherapy indicators
        is_mono_explicit = any(indicator in text_lower for indicator in self.monotherapy_indicators)
        
        # Check for combination indicators
        has_combo_indicators = any(indicator in text_lower for indicator in self.combination_indicators)
        
        if len(found_drugs) == 1 and (is_mono_explicit or not has_combo_indicators):
            return f"{found_drugs[0]}_monotherapy"
        elif len(found_drugs) > 1:
            # Sort drugs alphabetically for consistent naming
            sorted_drugs = sorted(found_drugs)
            return " + ".join(sorted_drugs)
        elif len(found_drugs) == 1 and has_combo_indicators:
            # Single drug mentioned but has combo indicators - might be part of combination
            return f"{found_drugs[0]}_combination"
        else:
            return "unknown_treatment"

    def extract_keywords_from_query(self, query: str) -> Dict[str, List[str]]:
        """Extract all relevant keywords from user query using intelligent pattern matching"""
        query_lower = query.lower()
        
        # Extract drugs mentioned in query
        query_drugs = self.extract_drugs_from_text(query)
        
        # Determine treatment types from query
        treatments_found = []
        
        # Look for specific treatment patterns in the query
        for phrase in query.split(','):
            phrase = phrase.strip()
            phrase_drugs = self.extract_drugs_from_text(phrase)
            if phrase_drugs:
                treatment_type = self.determine_treatment_type(phrase, phrase_drugs)
                if treatment_type not in treatments_found:
                    treatments_found.append(treatment_type)
        
        # If no specific treatments found but drugs mentioned, add general drug searches
        if not treatments_found and query_drugs:
            treatments_found = [f"{drug}_any" for drug in query_drugs]
        
        # Find trial patterns
        trials_found = []
        for trial_type, patterns in self.trial_patterns.items():
            if any(pattern.lower() in query_lower for pattern in patterns):
                trials_found.append(trial_type)
        
        # Find outcome categories
        outcomes_found = []
        for category, keywords in self.outcome_categories.items():
            if any(keyword.lower() in query_lower for keyword in keywords):
                outcomes_found.append(category)
        
        # Default to efficacy if comparing treatments
        if not outcomes_found and any(term in query_lower for term in ['compare', 'comparison', 'versus', 'vs']):
            outcomes_found = ['efficacy', 'safety']
        
        return {
            'treatments': treatments_found,
            'drugs': query_drugs,
            'trials': trials_found,
            'outcomes': outcomes_found
        }

    def create_filter_mask(self, df: pd.DataFrame, keywords: Dict[str, List[str]]) -> pd.Series:
        """Create boolean mask using intelligent, scalable filtering logic"""
        mask = pd.Series([False] * len(df))
        
        # Primary filtering by treatments/drugs
        if keywords['treatments'] or keywords['drugs']:
            treatment_mask = pd.Series([False] * len(df))
            
            # Process specific treatments
            for treatment in keywords['treatments']:
                if '_monotherapy' in treatment:
                    # Handle monotherapy: find drug without combination indicators
                    drug = treatment.replace('_monotherapy', '')
                    treatment_mask = treatment_mask | self._filter_monotherapy(df, drug)
                elif '_combination' in treatment:
                    # Handle general combination mentions
                    drug = treatment.replace('_combination', '')
                    treatment_mask = treatment_mask | self._filter_combination(df, drug)
                elif '_any' in treatment:
                    # Handle any mention of drug
                    drug = treatment.replace('_any', '')
                    treatment_mask = treatment_mask | self._filter_any_drug(df, drug)
                elif '+' in treatment:
                    # Handle specific combinations
                    drugs = treatment.split(' + ')
                    treatment_mask = treatment_mask | self._filter_specific_combination(df, drugs)
                else:
                    # General search
                    treatment_mask = treatment_mask | self._filter_general(df, treatment)
            
            # Process general drug mentions
            for drug in keywords['drugs']:
                treatment_mask = treatment_mask | self._filter_any_drug(df, drug)
            
            mask = mask | treatment_mask
        
        # Filter by trial patterns if specified
        if keywords['trials']:
            trial_mask = pd.Series([False] * len(df))
            for trial_type in keywords['trials']:
                patterns = self.trial_patterns.get(trial_type, [trial_type])
                for pattern in patterns:
                    if 'Trial Acronym/ID' in df.columns:
                        pattern_mask = df['Trial Acronym/ID'].str.contains(
                            pattern, case=False, na=False, regex=False
                        )
                        trial_mask = trial_mask | pattern_mask
            mask = mask | trial_mask
        
        # If no filters matched, show registrational trials
        if not mask.any():
            if 'R/P' in df.columns:
                mask = df['R/P'].str.contains('Registrational', case=False, na=False)
            else:
                mask = pd.Series([True] * len(df))
        
        return mask

    def _filter_monotherapy(self, df: pd.DataFrame, drug: str) -> pd.Series:
        """Filter for monotherapy of specific drug"""
        if 'Product/Regimen Name' not in df.columns:
            return pd.Series([False] * len(df))
        
        # Find rows containing the drug
        drug_variants = self.base_drugs.get(drug, [drug])
        drug_mask = pd.Series([False] * len(df))
        
        for variant in drug_variants:
            variant_mask = df['Product/Regimen Name'].str.contains(
                variant, case=False, na=False, regex=False
            )
            drug_mask = drug_mask | variant_mask
        
        # Exclude rows with combination indicators
        combo_mask = pd.Series([False] * len(df))
        for indicator in self.combination_indicators:
            if indicator in ['+', '→', '/']:  # Special characters need escaping
                indicator_escaped = '\\' + indicator
                indicator_mask = df['Product/Regimen Name'].str.contains(
                    indicator_escaped, case=False, na=False, regex=True
                )
            else:
                indicator_mask = df['Product/Regimen Name'].str.contains(
                    indicator, case=False, na=False, regex=False
                )
            combo_mask = combo_mask | indicator_mask
        
        return drug_mask & ~combo_mask

    def _filter_combination(self, df: pd.DataFrame, drug: str) -> pd.Series:
        """Filter for combinations containing specific drug"""
        if 'Product/Regimen Name' not in df.columns:
            return pd.Series([False] * len(df))
        
        drug_variants = self.base_drugs.get(drug, [drug])
        drug_mask = pd.Series([False] * len(df))
        
        for variant in drug_variants:
            variant_mask = df['Product/Regimen Name'].str.contains(
                variant, case=False, na=False, regex=False
            )
            drug_mask = drug_mask | variant_mask
        
        # Only include rows WITH combination indicators
        combo_mask = pd.Series([False] * len(df))
        for indicator in self.combination_indicators:
            if indicator in ['+', '→', '/']:
                indicator_escaped = '\\' + indicator
                indicator_mask = df['Product/Regimen Name'].str.contains(
                    indicator_escaped, case=False, na=False, regex=True
                )
            else:
                indicator_mask = df['Product/Regimen Name'].str.contains(
                    indicator, case=False, na=False, regex=False
                )
            combo_mask = combo_mask | indicator_mask
        
        return drug_mask & combo_mask

    def _filter_specific_combination(self, df: pd.DataFrame, drugs: List[str]) -> pd.Series:
        """Filter for specific drug combinations"""
        if 'Product/Regimen Name' not in df.columns:
            return pd.Series([False] * len(df))
        
        # All drugs must be present
        combined_mask = pd.Series([True] * len(df))
        
        for drug in drugs:
            drug_variants = self.base_drugs.get(drug.strip(), [drug.strip()])
            drug_mask = pd.Series([False] * len(df))
            
            for variant in drug_variants:
                variant_mask = df['Product/Regimen Name'].str.contains(
                    variant, case=False, na=False, regex=False
                )
                drug_mask = drug_mask | variant_mask
            
            combined_mask = combined_mask & drug_mask
        
        return combined_mask

    def _filter_any_drug(self, df: pd.DataFrame, drug: str) -> pd.Series:
        """Filter for any mention of drug (mono or combo)"""
        if 'Product/Regimen Name' not in df.columns:
            return pd.Series([False] * len(df))
        
        drug_variants = self.base_drugs.get(drug, [drug])
        drug_mask = pd.Series([False] * len(df))
        
        for variant in drug_variants:
            variant_mask = df['Product/Regimen Name'].str.contains(
                variant, case=False, na=False, regex=False
            )
            drug_mask = drug_mask | variant_mask
        
        return drug_mask

    def _filter_general(self, df: pd.DataFrame, term: str) -> pd.Series:
        """General filter for any term"""
        if 'Product/Regimen Name' not in df.columns:
            return pd.Series([False] * len(df))
        
        return df['Product/Regimen Name'].str.contains(
            term, case=False, na=False, regex=False
        )
        
        # Filter by trial names if specified
        if keywords['trials']:
            trial_mask = pd.Series([False] * len(df))
            
            for trial_type in keywords['trials']:
                trial_patterns = self.trial_keywords.get(trial_type, [trial_type])
                
                for pattern in trial_patterns:
                    if 'Trial Acronym/ID' in df.columns:
                        pattern_mask = df['Trial Acronym/ID'].str.contains(
                            pattern, case=False, na=False, regex=False
                        )
                        trial_mask = trial_mask | pattern_mask
            
            mask = mask | trial_mask
        
        # Filter by phase if mentioned (Ph3, Ph2, etc.)
        query_text = ' '.join(keywords.get('treatments', []) + keywords.get('trials', []) + keywords.get('outcomes', []))
        if 'ph3' in query_text.lower() or 'phase 3' in query_text.lower() or 'phase iii' in query_text.lower():
            if 'Trial Phase' in df.columns:
                phase_mask = df['Trial Phase'].str.contains('Ph3|Phase 3|Phase III', case=False, na=False, regex=True)
                mask = mask | phase_mask
            elif 'Highest Phase' in df.columns:
                phase_mask = df['Highest Phase'].str.contains('Ph3|Phase 3|Phase III', case=False, na=False, regex=True)
                mask = mask | phase_mask
        
        # If no specific filters match, show ALL registrational/relevant rows
        if not mask.any():
            # Show all registrational trials or trials with actual data
            if 'R/P' in df.columns:
                mask = df['R/P'].str.contains('Registrational', case=False, na=False)
            else:
                # Fallback: show rows that have trial data
                if 'Trial Acronym/ID' in df.columns:
                    mask = df['Trial Acronym/ID'].notna() & (df['Trial Acronym/ID'] != '')
                else:
                    mask = pd.Series([True] * len(df))  # Show all if no filtering possible
        
        return mask


class StreamlinedClinicalChatbot:
    """Streamlined chatbot with keyword-based filtering and ID-based retrieval - NO COLUMN FILTERING"""
    
    def __init__(self, df: pd.DataFrame, model: str = "gpt-4o-mini"):
        self.df = self._prepare_dataframe(df)
        self.model = model
        self.matcher = ClinicalTrialKeywordMatcher()
        
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe with ID column and clean data"""
        df_clean = df.copy()
        
        # Add ID column if not exists
        if 'ID' not in df_clean.columns:
            df_clean.insert(0, 'ID', range(1, len(df_clean) + 1))
        
        # Clean column names
        df_clean.columns = df_clean.columns.str.strip().str.replace("\n", " ").str.replace("  ", " ")
        
        return df_clean
    
    def filter_relevant_rows(self, query: str) -> Tuple[pd.DataFrame, Dict]:
        """Filter dataframe to only relevant rows based on query keywords - PRESERVES ALL COLUMNS"""
        # Extract keywords from query
        keywords = self.matcher.extract_keywords_from_query(query)
        
        # Create filter mask
        mask = self.matcher.create_filter_mask(self.df, keywords)
        
        # Apply filter - KEEP ALL COLUMNS, only filter rows
        filtered_df = self.df[mask].copy()
        
        # Create info for debugging
        filter_info = {
            'keywords_found': keywords,
            'total_rows': len(self.df),
            'filtered_rows': len(filtered_df),
            'row_ids': filtered_df['ID'].tolist() if 'ID' in filtered_df.columns else [],
            'total_columns': len(self.df.columns),
            'available_columns': list(filtered_df.columns)  # Show all available columns
        }
        
        return filtered_df, filter_info
    
    def create_simple_agent(self, filtered_df: pd.DataFrame) -> any:
        """Create a simple agent with filtered data and clear instructions - ALL COLUMNS AVAILABLE"""
        
        prefix = f"""You are a clinical trial data analyst working with a PRE-FILTERED dataset.

CRITICAL DATA ACCESS INSTRUCTIONS:
- You have access to ALL {len(filtered_df.columns)} columns in the dataset
- The dataset has {len(filtered_df)} relevant rows already filtered for this query
- ALWAYS check the actual dataframe values using df.iloc[], df.loc[], or direct column access
- DO NOT assume data is missing - check the actual cell values first

CRITICAL: SHOW ALL MATCHING ROWS - NOT JUST ONE PER TREATMENT TYPE
- If there are multiple rows for "Nivolumab mono", show ALL of them
- If there are multiple rows for "Pembrolizumab mono", show ALL of them  
- If there are multiple rows for any treatment, show ALL matching rows
- Do NOT summarize or aggregate - show each individual row's data

DATA EXTRACTION RULES:
1. Use direct dataframe access: df['Column_Name'].iloc[row_index] or df.loc[row_index, 'Column_Name']
2. Check for actual values before reporting "Not Reported"
3. Only report "Not Reported" if the cell is truly NaN, None, empty string, or contains "Not Reported"
4. Report exact values as they appear (e.g., "65.0%" should be reported as "65.0%")
5. For percentages, numbers, and text - report the exact cell content

CRITICAL: DO NOT SHOW DATAFRAME SHAPE OR STRUCTURE
- Never display: print(f"Dataframe shape: {{df.shape}}")
- Never display: print(f"Columns: {{df.columns.tolist()}}")
- Go directly to data extraction and analysis

DATA EXTRACTION PROCESS:
1. Examine EVERY ROW systematically to identify treatments and trials
2. Extract values for ALL relevant columns from EVERY matching row
3. Create comprehensive comparison tables showing ALL rows
4. Always verify values exist before creating tables

EXAMPLE - SHOW ALL ROWS FOR EACH TREATMENT:
```python
# Process EVERY row in the filtered dataset
for idx in df.index:
    treatment = df.loc[idx, 'Product/Regimen Name']
    trial = df.loc[idx, 'Trial Acronym/ID'] 
    orr = df.loc[idx, 'ORR']
    cr = df.loc[idx, 'CR']
    mpfs = df.loc[idx, 'mPFS']
    mos = df.loc[idx, 'mOS']
    # Show each individual row
    print(f"Row {{idx}}: {{treatment}} ({{trial}}) - ORR: {{orr}}, CR: {{cr}}, mPFS: {{mpfs}}, mOS: {{mos}}")
```

COMPARISON TABLE REQUIREMENTS:
- Include ALL relevant columns that have data
- Show actual values from the dataframe, not assumptions
- Create comprehensive tables with precise data extraction
- Include trial names, treatment names, and all outcome measures
- Show EVERY SINGLE ROW that matches the query criteria
- If there are 5 nivolumab monotherapy rows, show all 5 rows individually

Remember: ALWAYS access actual dataframe values, not assumptions. DO NOT show dataframe structure. SHOW ALL MATCHING ROWS INDIVIDUALLY.
"""
        
        return create_pandas_dataframe_agent(
            ChatOpenAI(model=self.model, temperature=0),
            filtered_df,
            verbose=True,  
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=prefix,
            allow_dangerous_code=True,
            max_iterations=15  # Increase iterations to handle more comprehensive analysis
        )
    
    def query(self, question: str) -> Dict[str, any]:
        """Process query with streamlined approach - NO COLUMN FILTERING"""
        # Step 1: Filter relevant rows (but keep ALL columns)
        filtered_df, filter_info = self.filter_relevant_rows(question)
        
        if filtered_df.empty:
            return {
                "output": "No relevant data found for your query. Please try rephrasing or check if the treatments/trials exist in the dataset.",
                "info": filter_info
            }
        
        # Step 2: Create simple agent with filtered data (ALL COLUMNS AVAILABLE)
        agent = self.create_simple_agent(filtered_df)
        
        # Step 3: Create simplified, direct question emphasizing actual data access
        simple_question = f"""
        {question}
        
        CRITICAL INSTRUCTIONS FOR DATA EXTRACTION:
        
        IMPORTANT: DO NOT show dataframe shape, structure, or column lists. Go directly to analysis.
        
        CRITICAL: SHOW ALL MATCHING ROWS INDIVIDUALLY
        - The dataset may have MULTIPLE rows for each treatment type
        - For example: there may be multiple "Nivolumab mono" rows from different trials
        - Show EVERY SINGLE row that matches the criteria
        - Do NOT aggregate or summarize - show each row individually
        
        1. EXAMINE EVERY ROW systematically:
           - For EVERY row index, identify: Product/Regimen Name, Trial Acronym/ID  
           - Extract actual values from each relevant column using df.loc[index, 'column']
           - Verify values before reporting - check if they're truly missing or contain actual data
           - Show ALL rows that match the criteria individually
        
        2. DATA ACCESS EXAMPLE:
        ```python
        # Process EVERY row in the filtered dataset - don't skip any
        for idx in df.index:
            treatment = df.loc[idx, 'Product/Regimen Name']
            trial = df.loc[idx, 'Trial Acronym/ID']
            orr_val = df.loc[idx, 'ORR']
            cr_val = df.loc[idx, 'CR'] 
            mpfs_val = df.loc[idx, 'mPFS']
            mos_val = df.loc[idx, 'mOS']
            traes_val = df.loc[idx, 'Gr ≥3 TRAEs']
            # Print details for EVERY row
            print(f"Row {{idx}}: {{treatment}} ({{trial}}) - ORR: {{orr_val}}, CR: {{cr_val}}, mPFS: {{mpfs_val}}, mOS: {{mos_val}}")
        ```
        
        3. CREATE COMPREHENSIVE COMPARISON TABLE:
        - Include ALL rows that match the query criteria as separate entries
        - Only use actual extracted values from the dataframe
        - Include ALL available data columns that are relevant
        - Do not assume "Not Reported" - verify each cell value
        - If there are multiple trials for the same treatment type, show them all separately
        - Each row should be a separate entry in your comparison table
        
        CRITICAL: Process ALL rows in the filtered dataset. Show every matching row individually in your analysis and comparison table.
        """
        
        # Step 4: Execute query
        try:
            result = agent.invoke({"input": simple_question})
            
            # Extract the output properly
            if isinstance(result, dict):
                output = result.get('output', str(result))
            else:
                output = str(result)
                
            return {
                "output": output, 
                "info": filter_info
            }
        except Exception as e:
            return {
                "output": f"Error processing query: {str(e)}", 
                "info": filter_info
            }


# Streamlit App
def main():
    st.title("🏥 Clinical Trial Chatbot")
    st.markdown("Fast and accurate clinical trial data analysis")
    
    # Load default data
    default_file_path = "Sample Data.xlsx"
    
    if os.path.exists(default_file_path):
        if 'data_loaded' not in st.session_state:
            with st.spinner(f"Loading data from {default_file_path}..."):
                try:
                    df = pd.read_excel(default_file_path)
                    
                    # Store data and chatbot in session state
                    st.session_state.df = df
                    st.session_state.chatbot = StreamlinedClinicalChatbot(df)
                    st.session_state.data_loaded = True
                    st.session_state.messages = []  # Clear chat history
                    st.success(f"Data loaded successfully!")
                    st.info(f"📊 Dataset: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Show available columns (OPTIONAL - you can remove this too if you don't want any column info)
                    with st.expander("📋 Available Columns"):
                        st.write("**All columns are available for analysis:**")
                        for i, col in enumerate(df.columns, 1):
                            st.write(f"{i}. {col}")
                            
                except Exception as e:
                    st.error(f"Error loading data file: {e}")
                    return
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "info" in message:
                    with st.expander("🔍 Query Analysis"):
                        info = message["info"]
                        if info.get("keywords_found"):
                            keywords = info["keywords_found"]
                            if keywords.get("treatments"):
                                st.write("**Treatments detected:**", ", ".join(keywords["treatments"]))
                            if keywords.get("trials"):
                                st.write("**Trials detected:**", ", ".join(keywords["trials"]))
                            if keywords.get("outcomes"):
                                st.write("**Outcomes requested:**", ", ".join(keywords["outcomes"]))
                        
                        st.write(f"**Rows filtered:** {info.get('filtered_rows', 0)} out of {info.get('total_rows', 0)}")
                        
                        if info.get("row_ids"):
                            st.write(f"**Row IDs analyzed:** {', '.join(map(str, info['row_ids'][:10]))}")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the clinical trials..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        result = st.session_state.chatbot.query(prompt)
                        
                        # Display the result
                        if result and result.get("output"):
                            st.markdown(result["output"])
                        else:
                            st.error("No response generated. Please try rephrasing your question.")
                        
                        # Show analysis info
                        if result and result.get("info"):
                            with st.expander("🔍 Query Analysis"):
                                info = result["info"]
                                if info.get("keywords_found"):
                                    keywords = info["keywords_found"]
                                    if keywords.get("treatments"):
                                        st.write("**Treatments detected:**", ", ".join(keywords["treatments"]))
                                    if keywords.get("trials"):
                                        st.write("**Trials detected:**", ", ".join(keywords["trials"]))
                                    if keywords.get("outcomes"):
                                        st.write("**Outcomes requested:**", ", ".join(keywords["outcomes"]))
                                
                                st.write(f"**Rows filtered:** {info.get('filtered_rows', 0)} out of {info.get('total_rows', 0)}")
                                
                                if info.get("row_ids"):
                                    st.write(f"**Row IDs analyzed:** {', '.join(map(str, info['row_ids'][:10]))}")
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result.get("output", "No response generated") if result else "Error occurred",
                            "info": result.get("info", {}) if result else {}
                        })
                        
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg,
                            "info": {}
                        })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
    else:
        # Error message when default file is not found
        st.error(f"❌ Default data file '{default_file_path}' not found!")
        st.info("Please ensure the Sample Data.xlsx file is in the same directory as this script.")
        
        # Show example queries
        st.markdown("### 💡 Example Queries:")
        st.markdown("""
        - *Compare clinical outcomes of pembrolizumab monotherapy, nivolumab monotherapy, nivolumab + ipilimumab, and nivolumab + relatlimab*
        - *What are the ORR and mPFS for CHECKMATE-067?*
        - *Show safety data for nivolumab combinations*
        - *Compare KEYNOTE vs CHECKMATE trials*
        - *Show me all available data for pembrolizumab trials*
        """)

if __name__ == "__main__":
    main()