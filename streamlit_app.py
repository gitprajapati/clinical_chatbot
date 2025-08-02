import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json
import numpy as np
import plotly.express as px

load_dotenv() 

# Set page config
st.set_page_config(
    page_title="Clinical Trial Chatbot",
    page_icon="🏥",
    layout="wide"
)

class SuperFastClinicalChatbot:
    """Chatbot with comprehensive alias mappings and optimized search"""
    
    def __init__(self, df: pd.DataFrame, model: str = "gpt-4o-mini"):
        self.df = self._prepare_dataframe(df)
        self.model = model
        self.llm = ChatOpenAI(model=self.model, temperature=0)
        self._create_search_index()
        self._build_alias_mappings()
        
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe with ID column and clean data"""
        df_clean = df.copy()
        
        # Add ID column if not exists
        if 'ID' not in df_clean.columns:
            df_clean.insert(0, 'ID', range(1, len(df_clean) + 1))
        
        # Clean column names
        df_clean.columns = df_clean.columns.str.strip().str.replace("\n", " ").str.replace("  ", " ")
        
        # Convert all object columns to string and handle NaN
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('N/A', '').replace('NA', '')
        
        return df_clean
    
    def _create_search_index(self):
        """Create search indexes for faster lookups"""
        self.text_columns = []
        self.numeric_columns = []
        
        for col in self.df.columns:
            if self.df[col].dtype in ['object', 'string']:
                self.text_columns.append(col)
            elif self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self.numeric_columns.append(col)
    
    def _build_alias_mappings(self):
        """Build comprehensive alias mappings for faster search"""
        
        # Trial acronym aliases
        self.trial_aliases = {
            # CHECKMATE variations
            'checkmate-511': ['cm-511', 'cm511', 'cm 511', 'checkmate511', 'checkmate 511'],
            'checkmate-067': ['cm-067', 'cm067', 'cm 067', 'checkmate067', 'checkmate 067'],
            'checkmate-066': ['cm-066', 'cm066', 'cm 066', 'checkmate066', 'checkmate 066'],
            
            # RELATIVITY variations
            'relativity-047': ['rel-047', 'rel047', 'rel 047', 'relativity047', 'relativity 047'],
            
            # KEYNOTE variations
            'keynote-006': ['kn-006', 'kn006', 'kn 006', 'keynote006', 'keynote 006'],
            'keynote-252': ['kn-252', 'kn252', 'kn 252', 'keynote252', 'keynote 252'],
            
            # Add more as needed
            'combi-d': ['combid', 'combi d'],
            'combi-v': ['combiv', 'combi v'],
            'combi-i': ['combii', 'combi i'],
            'columbus': ['col', 'columbus trial'],
            'dreamseq': ['dream-seq', 'dream seq'],
            'inspire150': ['imspire-150', 'imspire 150'],
            'cobrim': ['co-brim', 'co brim']
        }
        
        # Phase aliases
        self.phase_aliases = {
            'ph3': ['phase 3', 'phase iii', 'phase3', 'phase-3', 'phase-iii'],
            'ph2': ['phase 2', 'phase ii', 'phase2', 'phase-2', 'phase-ii'],
            'ph1': ['phase 1', 'phase i', 'phase1', 'phase-1', 'phase-i'],
            'ph4': ['phase 4', 'phase iv', 'phase4', 'phase-4', 'phase-iv'],
        }
        
        # Company aliases
        self.company_aliases = {
            'bms': ['bristol myers squibb', 'bristol-myers squibb', 'bristol myers', 'bristol-myers'],
            'merck': ['merck/mds', 'merck us', 'us-based merck', 'merck & co', 'merck & co.'],
            'roche': ['genentech', 'roche genentech'],
            'novartis': ['novartis pharma', 'novartis ag'],
            'pfizer': ['pfizer inc', 'pfizer inc.'],
        }
        
        # Clinical outcomes aliases - comprehensive mapping
        self.outcome_aliases = {
            # Efficacy outcomes
            'orr': ['overall response rate', 'response rate', 'objective response rate'],
            'cr': ['complete response', 'complete response rate'],
            'pfs': ['progression free survival', 'progression-free survival', 'median pfs', 'mpfs'],
            'os': ['overall survival', 'median os', 'mos'],
            'dor': ['duration of response', 'median dor', 'mdor'],
            
            # Safety outcomes with extensive aliases
            'gr ≥3 traes': [
                'gr 3+ traes', 'grade 3 and above traes', 'high grade traes',
                'gr ≥3 treatment related adverse events', 'gr ≥3 treatment related aes',
                'gr ≥3 treatment-related adverse events', 'gr ≥3 treatment-related aes',
                'gr 3+ treatment related adverse events', 'gr 3+ treatment related aes',
                'gr 3+ treatment-related adverse events', 'gr 3+ treatment-related aes',
                'grade 3 and above treatment related adverse events',
                'grade 3 and above treatment related aes',
                'grade 3 and above treatment-related adverse events',
                'grade 3 and above treatment-related aes',
                'high grade treatment-related aes', 'high grade treatment-related adverse events'
            ],
            'gr 3/4 traes': [
                'grade 3/4 traes', 'grade 3 or 4 traes', 'gr 3 or 4 traes',
                'grade 3/4 treatment related adverse events',
                'grade 3/4 treatment-related adverse events'
            ],
            'gr ≥3 teaes': [
                'gr 3+ teaes', 'grade 3 and above teaes',
                'gr ≥3 treatment emergent adverse events',
                'grade 3 and above treatment emergent adverse events'
            ],
            'gr 3/4 teaes': [
                'grade 3/4 teaes', 'grade 3 or 4 teaes',
                'grade 3/4 treatment emergent adverse events'
            ],
            'gr ≥3 iraes': [
                'gr 3+ iraes', 'grade 3 and above iraes',
                'gr ≥3 immune related adverse events',
                'grade 3 and above immune related adverse events'
            ],
            'gr 3/4 iraes': [
                'grade 3/4 iraes', 'grade 3 or 4 iraes',
                'grade 3/4 immune related adverse events'
            ],
            'gr ≥3 aes': [
                'gr 3+ aes', 'grade 3 and above aes', 'high grade aes',
                'gr ≥3 adverse events', 'grade 3 and above adverse events'
            ],
            'gr 3/4 aes': [
                'grade 3/4 aes', 'grade 3 or 4 aes',
                'grade 3/4 adverse events'
            ]
        }
        
        # Default table columns - what should be shown by default
        self.default_table_columns = [
            'Product/Regimen Name', 'Active Developers (Companies Names)', 
            'Highest Phase', 'Line of therapy (LoT)', 'Therapeutic Indication'
        ]
        
        # If LoT doesn't exist, use alternatives
        if 'Line of therapy (LoT)' not in self.df.columns:
            lot_alternatives = [col for col in self.df.columns if 'line' in col.lower() or 'lot' in col.lower()]
            if lot_alternatives:
                self.default_table_columns[3] = lot_alternatives[0]
            else:
                self.default_table_columns[3] = 'Therapeutic Area'  # fallback
    
    def _expand_query_terms(self, query: str) -> List[str]:
        """Expand query terms using alias mappings"""
        query_lower = query.lower()
        expanded_terms = [query_lower]
        
        # Expand trial aliases
        for canonical, aliases in self.trial_aliases.items():
            if canonical in query_lower:
                expanded_terms.extend(aliases)
            for alias in aliases:
                if alias in query_lower:
                    expanded_terms.append(canonical)
                    expanded_terms.extend([a for a in aliases if a != alias])
        
        # Expand phase aliases
        for canonical, aliases in self.phase_aliases.items():
            if canonical in query_lower:
                expanded_terms.extend(aliases)
            for alias in aliases:
                if alias in query_lower:
                    expanded_terms.append(canonical)
        
        # Expand company aliases
        for canonical, aliases in self.company_aliases.items():
            if canonical in query_lower:
                expanded_terms.extend(aliases)
            for alias in aliases:
                if alias in query_lower:
                    expanded_terms.append(canonical)
        
        # Expand outcome aliases
        for canonical, aliases in self.outcome_aliases.items():
            if canonical in query_lower:
                expanded_terms.extend(aliases)
            for alias in aliases:
                if alias in query_lower:
                    expanded_terms.append(canonical)
        
        return list(set(expanded_terms)) 
    
    def extract_search_terms(self, query: str) -> Dict[str, List[str]]:
        """Extract and expand search terms from query"""
        # Get expanded terms
        expanded_terms = self._expand_query_terms(query)
        
        # Common clinical trial terms
        trial_terms = ['checkmate', 'relativity', 'keynote', 'opdivo', 'yervoy', 'nivolumab', 'ipilimumab', 'relatlimab']
        drug_terms = ['nivolumab', 'ipilimumab', 'relatlimab', 'pembrolizumab', 'atezolizumab', 'cobimetinib', 'vemurafenib']
        outcome_terms = ['orr', 'pfs', 'os', 'response', 'survival', 'progression', 'overall', 'cr', 'dor', 'traes', 'teaes', 'iraes']
        
        # Extract terms from expanded list
        found_trials = []
        found_drugs = []
        found_outcomes = []
        
        for term in expanded_terms:
            # Check for trial terms
            for trial_term in trial_terms:
                if trial_term in term:
                    found_trials.append(trial_term)
            
            # Check for drug terms
            for drug_term in drug_terms:
                if drug_term in term:
                    found_drugs.append(drug_term)
            
            # Check for outcome terms
            for outcome_term in outcome_terms:
                if outcome_term in term:
                    found_outcomes.append(outcome_term)
        
        # Extract quoted terms and numbers
        quoted_terms = re.findall(r'"([^"]*)"', query)
        number_terms = re.findall(r'\b\d+\b', query)
        
        return {
            'trials': list(set(found_trials)),
            'drugs': list(set(found_drugs)),
            'outcomes': list(set(found_outcomes)),
            'quoted': quoted_terms,
            'numbers': number_terms,
            'expanded_terms': expanded_terms,
            'all_terms': query.lower().split()
        }
    
    def search_dataframe_optimized(self, search_terms: Dict[str, List[str]]) -> pd.DataFrame:
        """pandas-based search with alias expansion"""
        
        # Start with all rows
        mask = pd.Series(True, index=self.df.index)
        search_performed = False
        
        # Search using expanded terms for better matching
        all_search_terms = (
            search_terms.get('trials', []) +
            search_terms.get('drugs', []) +
            search_terms.get('outcomes', []) +
            search_terms.get('quoted', []) +
            search_terms.get('numbers', []) +
            search_terms.get('expanded_terms', [])
        )
        
        if all_search_terms:
            combined_mask = pd.Series(False, index=self.df.index)
            
            for term in all_search_terms:
                if not term or len(term) < 2:
                    continue
                    
                term_mask = pd.Series(False, index=self.df.index)
                
                # Search in specific high-priority columns first
                priority_columns = [
                    'Trial Acronym/ID', 'Product/Regimen Name', 
                    'Active Developers (Companies Names)', 'Highest Phase'
                ]
                
                for col in priority_columns:
                    if col in self.df.columns:
                        try:
                            col_mask = self.df[col].str.contains(term, case=False, na=False, regex=False)
                            term_mask |= col_mask
                        except:
                            continue
                
                # If not found in priority columns, search all text columns
                if not term_mask.any():
                    for col in self.text_columns:
                        try:
                            col_mask = self.df[col].str.contains(term, case=False, na=False, regex=False)
                            term_mask |= col_mask
                        except:
                            continue
                
                combined_mask |= term_mask
            
            if combined_mask.any():
                mask = combined_mask
                search_performed = True
        
        # If no specific search performed, return top 20 rows
        if not search_performed:
            return self.df.head(20)
        
        return self.df[mask] if mask.any() else self.df.head(10)
    
    def retrieve_data_ultra_fast(self, query: str) -> Dict[str, Any]:
        """Data retrieval with comprehensive search"""
        try:
            # Extract and expand search terms
            search_terms = self.extract_search_terms(query)
            
            # Search dataframe
            filtered_df = self.search_dataframe_optimized(search_terms)
            
            # Limit results for performance
            max_records = 50
            if len(filtered_df) > max_records:
                filtered_df = filtered_df.head(max_records)
            
            # Convert to records
            records = filtered_df.to_dict('records')
            
            # Get relevant columns
            relevant_columns = self._identify_relevant_columns_fast(query, filtered_df)
            
            return {
                "success": True,
                "retrieved_data": records,
                "columns_found": relevant_columns,
                "total_records": len(records),
                "search_terms_used": search_terms,
                "filtered_df": filtered_df,
                "filtered_df_shape": filtered_df.shape
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "retrieved_data": []
            }
    
    def _identify_relevant_columns_fast(self, query: str, df: pd.DataFrame) -> List[str]:
        """Fast identification of relevant columns"""
        query_lower = query.lower()
        
        # Start with default table columns
        relevant_cols = [col for col in self.default_table_columns if col in df.columns]
        
        # Add outcome-specific columns based on query
        outcome_column_mappings = {
            'orr': ['ORR', 'ORR Notes'],
            'cr': ['CR', 'CR Notes'],
            'pfs': ['mPFS', 'PFS Notes', '1-yr PFS Rate', '2-yr PFS Rate'],
            'os': ['mOS', 'OS Notes', '1-yr OS Rate', '2-yr OS Rate'],
            'dor': ['mDoR'],
            'safety': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs', 'Gr 3/4 AEs', 'Gr ≥3 AEs'],
            'traes': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs'],
            'adverse': ['Key AEs', 'Gr 3/4 AEs', 'Gr ≥3 AEs']
        }
        
        for keyword, columns in outcome_column_mappings.items():
            if keyword in query_lower:
                for col_pattern in columns:
                    matching_cols = [col for col in df.columns if col_pattern in col]
                    relevant_cols.extend(matching_cols)
        
        # If comparing trials, include trial identification columns
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            comparison_cols = ['Trial Acronym/ID', 'Product/Regimen Name', 'ORR', 'mPFS', 'mOS']
            relevant_cols.extend([col for col in comparison_cols if col in df.columns])
        
        return list(set(relevant_cols))
    
    def extract_metrics_from_question(self, question: str, metric_list: List[str]) -> List[str]:
        """Extracts metrics mentioned in the user's question."""
        q_lower = question.lower()
        return [metric for metric in metric_list if any(tok in q_lower for tok in re.split(r'[\s\-/()]+', metric.lower()) if len(tok) >= 3)]
    
    def generate_response_ultra_fast(self, original_query: str, retrieved_data: Dict[str, Any]) -> str:
        """Generate response with optimized prompting"""
        
        if not retrieved_data["success"]:
            return f"Error retrieving data: {retrieved_data.get('error', 'Unknown error')}"
        
        # Limit data for faster processing
        data_subset = retrieved_data["retrieved_data"][:15]
        
        # Shorter, more focused prompt
        response_prompt = f"""Answer this clinical trial question: {original_query}

Data available ({len(data_subset)} records):
{json.dumps(data_subset, indent=1)}

Instructions:
- Use ONLY the provided data
- Create comparison tables if requested
- Be concise and accurate
- State "Not available" for missing data

Answer directly:"""
        
        try:
            from langchain.schema import HumanMessage
            response = self.llm.invoke([HumanMessage(content=response_prompt)])
            return response.content
            
        except Exception as e:
            return self._create_fallback_response(original_query, retrieved_data)
    
    def _create_fallback_response(self, query: str, data: Dict[str, Any]) -> str:
        """Create basic response if LLM fails"""
        if not data["retrieved_data"]:
            return "No relevant data found for your query."
        
        records = data["retrieved_data"][:10]
        response = f"## Results for: {query}\n\nFound {len(records)} relevant records:\n\n"
        
        if records:
            # Use default columns for table
            available_cols = [col for col in self.default_table_columns if col in records[0]]
            
            if available_cols:
                response += "| " + " | ".join(available_cols) + " |\n"
                response += "| " + " | ".join(["---"] * len(available_cols)) + " |\n"
                
                for record in records[:5]:
                    row = []
                    for col in available_cols:
                        value = str(record.get(col, "N/A"))[:50]  # Truncate long values
                        row.append(value)
                    response += "| " + " | ".join(row) + " |\n"
        
        return response
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query with maximum optimization"""
        
        # Step 1: Ultra-fast data retrieval
        start_time = pd.Timestamp.now()
        retrieved_data = self.retrieve_data_ultra_fast(question)
        retrieval_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Step 2: Fast response generation
        start_time = pd.Timestamp.now()
        response = self.generate_response_ultra_fast(question, retrieved_data)
        response_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        return {
            "output": response,
            "retrieved_data": retrieved_data,
            "timing": {
                "retrieval_seconds": round(retrieval_time, 2),
                "response_seconds": round(response_time, 2),
                "total_seconds": round(retrieval_time + response_time, 2)
            },
            "dataset_info": {
                "total_rows_analyzed": len(self.df),
                "records_found": retrieved_data.get("total_records", 0),
                "query_type": "ultra_fast_search"
            }
        }
    
    def get_column_info(self) -> str:
        """Get streamlined column information"""
        key_cols = ['Product/Regimen Name', 'Trial Acronym/ID', 'Active Developers (Companies Names)', 
                   'Highest Phase', 'ORR', 'mPFS', 'mOS']
        info = []
        
        for col in key_cols:
            if col in self.df.columns:
                unique_count = self.df[col].nunique()
                sample_values = self.df[col].dropna().unique()[:3]
                info.append(f"**{col}**: {unique_count} unique values")
        
        return "\n".join(info)


def display_bar_charts(df, arm_col, metric_cols, base_key=""):
    """Display bar charts for selected metrics"""
    st.markdown("### 📊 Arm-wise Metrics Breakdown")

    df = df.copy()

    # Ensure unique arms
    if df[arm_col].duplicated().any():
        df[arm_col] = df[arm_col] + " - arm " + df.groupby(arm_col).cumcount().add(1).astype(str)

    # Melt to long format
    melted_df = pd.melt(df[[arm_col] + metric_cols], id_vars=arm_col,
                        var_name="Metric", value_name="RawValue")
    melted_df["RawValue"] = melted_df["RawValue"].astype(str).str.strip()

    missing_keywords = {"", "na", "n/a", "nr", "nan", "none", "null", "not reported"}
    melted_df["IsMissing"] = melted_df["RawValue"].str.lower().isin(missing_keywords)

    # Extract numeric values
    melted_df["Value"] = (
        melted_df["RawValue"]
        .str.replace('%', '', regex=False)
        .str.replace(r'[^\d\.\-]+', '', regex=True)
    )
    melted_df["Value"] = pd.to_numeric(melted_df["Value"], errors='coerce')
    melted_df["PlotValue"] = melted_df["Value"]
    melted_df.loc[melted_df["IsMissing"], "PlotValue"] = 0.3  # thin bar
    melted_df.loc[melted_df["IsMissing"], "TextPosX"] = 2.5  # fixed text location for missing values
    melted_df.loc[~melted_df["IsMissing"], "TextPosX"] = melted_df["PlotValue"]
    melted_df["Text"] = melted_df["RawValue"].str.upper()

    color_map = {metric: px.colors.qualitative.Set3[i % 10] for i, metric in enumerate(melted_df["Metric"].unique())}
    melted_df["Color"] = melted_df["Metric"].map(color_map)

    fig = px.bar(
        melted_df,
        x="PlotValue",
        y=arm_col,
        color="Metric",
        facet_col="Metric",
        facet_col_wrap=5,
        orientation="h",
        text="Text",
        color_discrete_map=color_map
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=14),
        cliponaxis=False
    )

    fig.update_layout(
        height=450 + 60 * len(df),
        font=dict(size=14),
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=60),
        plot_bgcolor="rgba(0,0,0,0)"
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].strip()))
    fig.for_each_xaxis(lambda x: x.update(title='', showticklabels=False))
    fig.update_yaxes(title='', automargin=True)

    if melted_df["IsMissing"].any():
        st.warning("Some metric values were not reported: 'NR', 'NA', or similar placeholders.")

    st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_bar_chart")


# Streamlit App
def main():
    st.title("🏥 Clinical Trial Chatbot")
    
    # Load default data
    default_file_path = "Sample Data.xlsx"
    
    if os.path.exists(default_file_path):
        if 'data_loaded' not in st.session_state:
            with st.spinner("Loading data..."):
                try:
                    df = pd.read_excel(default_file_path)
                    
                    # Store data and chatbot in session state
                    st.session_state.df = df
                    st.session_state.chatbot = SuperFastClinicalChatbot(df)
                    st.session_state.data_loaded = True
                    st.session_state.messages = []
                    
                    st.success(f"✅ Data loaded! ({len(df)} rows, {len(df.columns)} columns)")
                            
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
                    return
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about clinical trials..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                try:
                    result = st.session_state.chatbot.query(prompt)
                    
                    # Display result
                    if result and result.get("output"):
                        st.markdown(result["output"])
                        
                        # Check if we should show visualization
                        retrieved_data = result.get("retrieved_data", {})
                        if retrieved_data.get("success") and retrieved_data.get("filtered_df") is not None:
                            filtered_df = retrieved_data["filtered_df"]
                            
                            # Identify trial ID column
                            trial_id_col = next((col for col in [
                                "Trial Acronym/ID", "Trial Name", "Trial ID", 
                                "Product/Regimen Name", "Product/Regimen"
                            ] if col in filtered_df.columns), None)
                            
                            if trial_id_col and len(filtered_df) > 0:
                                # Get all available metric columns
                                all_metrics = [
                                    "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
                                    "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate",
                                    "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate",
                                    "Gr 3/4 TRAEs", "Gr 3/4 TEAEs", "Gr 3/4 irAEs",
                                    "Tx-related Deaths (Gr 5 TRAEs)", "All Deaths (Gr 5 AEs)"
                                ]
                                
                                # Find which metrics are in the filtered dataframe
                                available_metrics = [m for m in all_metrics if m in filtered_df.columns]
                                
                                if available_metrics:
                                    # Extract mentioned metrics from the query
                                    mentioned_metrics = st.session_state.chatbot.extract_metrics_from_question(
                                        prompt, available_metrics
                                    )
                                    
                                    # Default to mentioned metrics or first 3 available
                                    default_metrics = mentioned_metrics if mentioned_metrics else available_metrics[:3]
                                    
                                    # Create unique key for this visualization
                                    viz_key = f"viz_{len(st.session_state.messages)}"
                                    
                                    # Metric selection
                                    st.markdown(f"#### Select Metrics to Visualize")
                                    selected_metrics = st.multiselect(
                                        "Metrics",
                                        options=available_metrics,
                                        default=default_metrics,
                                        key=f"{viz_key}_metrics"
                                    )
                                    
                                    # Display charts if metrics selected
                                    if selected_metrics:
                                        display_bar_charts(
                                            filtered_df,
                                            trial_id_col,
                                            selected_metrics,
                                            base_key=viz_key
                                        )
                    else:
                        st.error("No response generated.")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result.get("output", "No response") if result else "Error"
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    else:
        st.error(f"❌ File '{default_file_path}' not found!")
        st.info("Ensure Sample Data.xlsx is in the same directory.")

if __name__ == "__main__":
    main()
