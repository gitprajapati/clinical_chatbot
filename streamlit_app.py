import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import pickle
import os
from typing import List, Dict, Tuple, Any, Optional
import json
import plotly.express as px
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Clinical Trial AI Assistant",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better UI with dark/light mode support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f0f2f6;
        margin-right: 20%;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class ClinicalChatWithFAISS:
    def __init__(self):
        """Initialize the chat system with pre-built FAISS index and embeddings"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = self.api_key
        self.embedding_model = "text-embedding-3-large"
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Load pre-built components
        self.index, self.df, self.embeddings = self._load_saved_components()
        self._build_comprehensive_mappings()
        
    def _load_saved_components(self) -> Tuple[faiss.Index, pd.DataFrame, np.ndarray]:
        """Load saved FAISS index, dataframe, and embeddings"""
        try:
            index = faiss.read_index("clinical_data_index.faiss")
            df = pd.read_pickle("clinical_data_df.pkl")
            embeddings = np.load("clinical_data_embeddings.npy")
            return index, df, embeddings
        except Exception as e:
            st.error(f"Error loading saved components: {str(e)}")
            raise
    
    def _build_comprehensive_mappings(self):
        """Build comprehensive alias mappings from the pasted code"""
        # Trial acronym aliases
        self.trial_aliases = {
            'checkmate-511': ['cm-511', 'cm511', 'cm 511', 'checkmate511', 'checkmate 511'],
            'checkmate-067': ['cm-067', 'cm067', 'cm 067', 'checkmate067', 'checkmate 067'],
            'checkmate-066': ['cm-066', 'cm066', 'cm 066', 'checkmate066', 'checkmate 066'],
            'checkmate-238': ['cm-238', 'cm238', 'cm 238', 'checkmate238', 'checkmate 238'],
            'keynote-006': ['kn-006', 'kn006', 'kn 006', 'keynote006', 'keynote 006'],
            'keynote-252': ['kn-252', 'kn252', 'kn 252', 'keynote252', 'keynote 252'],
            'dreamseq': ['dream-seq', 'dream seq', 'dream sequence', 'dreamsequence'],
            'combi-d': ['combid', 'combi d', 'combination d'],
            'columbus': ['col', 'columbus trial', 'columbus study'],
            'cobrim': ['co-brim', 'co brim', 'cobrim trial'],
            'relativity-047': ['relativity047', 'relativity 047', 'rel-047', 'rel047'],
        }
        
        # Phase aliases
        self.phase_aliases = {
            'ph3': ['phase 3', 'phase iii', 'phase3', 'phase-3', 'p3'],
            'ph2': ['phase 2', 'phase ii', 'phase2', 'phase-2', 'p2'],
            'ph1': ['phase 1', 'phase i', 'phase1', 'phase-1', 'p1'],
        }
        
        # Company aliases
        self.company_aliases = {
            'bms': ['bristol myers squibb', 'bristol-myers squibb', 'bristol myers'],
            'merck': ['merck/mds', 'merck us', 'us-based merck', 'msd'],
            'roche': ['genentech', 'roche genentech'],
            'novartis': ['novartis pharma', 'novartis ag'],
            'pfizer': ['pfizer inc', 'pfizer inc.'],
        }
        
        # Drug aliases
        self.drug_aliases = {
            'nivolumab': ['nivo', 'opdivo', 'bms-936558'],
            'ipilimumab': ['ipi', 'yervoy', 'mdx-010'],
            'pembrolizumab': ['pembro', 'keytruda', 'mk-3475'],
            'vemurafenib': ['vemu', 'zelboraf', 'plx4032'],
            'dabrafenib': ['dabra', 'tafinlar'],
            'trametinib': ['trame', 'mekinist'],
            'relatlimab': ['relatl', 'lag-3', 'bms-986016']
        }
        
        # Outcome aliases
        self.outcome_aliases = {
            'orr': ['overall response rate', 'response rate', 'objective response rate'],
            'cr': ['complete response', 'complete response rate'],
            'pr': ['partial response', 'partial response rate'],
            'pfs': ['progression free survival', 'progression-free survival', 'median pfs'],
            'os': ['overall survival', 'median os', 'mos'],
            'aes': ['adverse events', 'safety', 'toxicity'],
            'traes': ['treatment related adverse events', 'treatment-related adverse events'],
        }
        
        # Text columns for search
        self.text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        self.numeric_columns = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
    
    def _expand_query_with_aliases(self, query: str) -> str:
        """Expand query using alias mappings"""
        expanded_terms = []
        query_lower = query.lower()
        
        # Check all alias dictionaries
        all_aliases = [
            self.trial_aliases, self.phase_aliases, 
            self.company_aliases, self.drug_aliases, self.outcome_aliases
        ]
        
        for alias_dict in all_aliases:
            for canonical, aliases in alias_dict.items():
                if canonical in query_lower:
                    expanded_terms.extend(aliases)
                for alias in aliases:
                    if alias in query_lower:
                        expanded_terms.append(canonical)
        
        # Combine original query with expanded terms
        if expanded_terms:
            expanded_query = query + " " + " ".join(expanded_terms)
            return expanded_query
        return query
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text using OpenAI API"""
        try:
            response = openai.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            return None
    
    def semantic_search(self, query: str, k: int = 10) -> Tuple[pd.DataFrame, List[float]]:
        """Perform semantic search using FAISS"""
        # Expand query with aliases
        expanded_query = self._expand_query_with_aliases(query)
        
        # Generate embedding
        query_embedding = self.generate_embedding(expanded_query)
        if not query_embedding:
            return pd.DataFrame(), []
        
        # Normalize for cosine similarity
        query_vec = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_vec)
        
        # Search
        distances, indices = self.index.search(query_vec, k)
        
        # Get results
        valid_indices = indices[0][indices[0] != -1]
        valid_distances = distances[0][:len(valid_indices)]
        
        if len(valid_indices) > 0:
            result_df = self.df.iloc[valid_indices].copy()
            result_df['similarity_score'] = valid_distances
            return result_df, valid_distances.tolist()
        
        return pd.DataFrame(), []
    
    def hybrid_search(self, query: str, k: int = 20) -> pd.DataFrame:
        """Combine semantic search with keyword filtering"""
        # First, get semantic search results
        semantic_results, scores = self.semantic_search(query, k=k)
        
        if semantic_results.empty:
            return pd.DataFrame()
        
        # Extract specific terms from query for additional filtering
        query_lower = query.lower()
        
        # Look for specific trial names, drugs, etc.
        specific_filters = []
        
        # Check for trial names
        trial_patterns = [
            r'checkmate[-\s]?\d+', r'keynote[-\s]?\d+', r'dreamseq',
            r'combi[-\s][div]', r'columbus', r'cobrim', r'relativity[-\s]?\d+'
        ]
        
        for pattern in trial_patterns:
            matches = re.findall(pattern, query_lower)
            specific_filters.extend(matches)
        
        # If we have specific filters, apply them
        if specific_filters and 'Trial Acronym/ID' in semantic_results.columns:
            filter_mask = pd.Series(False, index=semantic_results.index)
            for term in specific_filters:
                mask = semantic_results['Trial Acronym/ID'].str.contains(term, case=False, na=False)
                filter_mask |= mask
            
            if filter_mask.any():
                # Prioritize exact matches
                exact_matches = semantic_results[filter_mask]
                other_matches = semantic_results[~filter_mask]
                semantic_results = pd.concat([exact_matches, other_matches])
        
        return semantic_results
    
    def should_create_visualization(self, query: str, response: str) -> bool:
        """Use LLM to determine if visualization is needed based on query and response"""
        decision_prompt = f"""
        You are an expert at determining when clinical data visualizations are needed.
        
        Analyze this query and response to determine if a visualization would be helpful:
        
        Query: {query}
        Response: {response}
        
        Return ONLY "YES" or "NO" based on these criteria:
        
        YES if the query/response involves:
        - Comparisons between treatments, trials, or outcomes
        - Multiple numerical metrics (ORR, PFS, OS, response rates, etc.)
        - Performance analysis across different regimens
        - Efficacy or safety comparisons
        - Questions asking about "best", "better", "compare", "versus", "difference"
        - Multiple trial arms or treatments mentioned with metrics
        
        NO if the query/response involves:
        - Simple factual questions (who, what company, which indication)
        - Single piece of information lookup
        - Developer/manufacturer information
        - Trial design details without metrics
        - Regulatory or approval information
        - General background information
        - Questions about trial logistics, enrollment, or study design
        
        Answer: """
        
        try:
            from langchain.schema import HumanMessage
            decision_response = self.llm.invoke([HumanMessage(content=decision_prompt)])
            decision = decision_response.content.strip().upper()
            return decision == "YES"
        except Exception as e:
            # Default to showing visualization if there's an error in decision making
            # Check for basic comparison keywords as fallback
            comparison_keywords = ['compare', 'versus', 'vs', 'better', 'best', 'difference', 'efficacy', 'response rate', 'survival']
            return any(keyword in query.lower() or keyword in response.lower() for keyword in comparison_keywords)
    
    def generate_response(self, query: str, search_results: pd.DataFrame) -> Dict[str, Any]:
        """Generate response using LLM and return both response and used trial arms"""
        if search_results.empty:
            return {
                'response': "I couldn't find any relevant clinical trials matching your query. Please try different search terms.",
                'used_trial_arms': []
            }
        
        # Prepare context from search results with unique identifiers
        context_data = []
        for idx, row in search_results.head(15).iterrows():
            row_dict = row.to_dict()
            # Create unique identifier using multiple columns since no single unique column exists
            trial_id = str(row_dict.get('Trial Acronym/ID', 'Unknown'))
            product = str(row_dict.get('Product/Regimen Name', 'Unknown'))
            comparator = str(row_dict.get('Comparator', 'Unknown'))
            
            # Create a more robust unique identifier
            row_dict['UNIQUE_ARM_ID'] = f"{trial_id}||{product}||{comparator}||{idx}"
            context_data.append(row_dict)
        
        # Create enhanced prompt that asks LLM to specify which trial arms it used
        prompt = f"""You are a clinical trial expert assistant. Answer the following question based on the provided clinical trial data.

Question: {query}

Available Data:
{json.dumps(context_data, indent=2)}

Instructions:
1. Provide a clear, accurate answer using ONLY the provided data
2. If comparing trials, create a structured comparison table
3. Include specific metrics (ORR, PFS, OS, etc.) when relevant
4. State "Not available" for any missing data
5. Be concise but thorough
6. Use markdown formatting for better readability
7. **CRITICAL**: At the end of your response, include a section called "**TRIAL_ARMS_USED:**" followed by a comma-separated list of the exact "UNIQUE_ARM_ID" values for ALL specific trial arms/treatments you referenced in your answer. Each UNIQUE_ARM_ID contains the trial ID, treatment regimen, comparator, and row index.
8. **VERY IMPORTANT**: When comparing specific treatments (like "nivolumab monotherapy"), make sure you identify the exact treatment arms that match that description. For example:
   - If asked about "nivolumab monotherapy" vs "combination therapy", only include arms where the Product/Regimen Name contains "Nivolumab" alone, not "Nivolumab + other drugs"
   - Be precise about which specific treatment regimen you're discussing
9. Provide only the direct answer to the asked question. Do not generate your own questions, add extra information, or give irrelevant answers.

Format for trial arms used:
**TRIAL_ARMS_USED:** UNIQUE_ARM_ID1, UNIQUE_ARM_ID2, UNIQUE_ARM_ID3

Answer:"""

        try:
            from langchain.schema import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            full_response = response.content
            
            # Extract used trial arms from response
            used_trial_arms = self._extract_used_trial_arms(full_response, search_results, query)
            
            # Clean the response by removing the TRIAL_ARMS_USED section
            clean_response = re.sub(r'\*\*TRIAL_ARMS_USED:\*\*.*$', '', full_response, flags=re.MULTILINE | re.DOTALL).strip()
            
            return {
                'response': clean_response,
                'used_trial_arms': used_trial_arms
            }
            
        except Exception as e:
            # Fallback response
            fallback_response = self._create_fallback_response(query, search_results)
            # For fallback, use top results with unique identifiers
            used_trial_arms = []
            for idx, row in search_results.head(3).iterrows():
                trial_id = str(row.get('Trial Acronym/ID', 'Unknown'))
                product = str(row.get('Product/Regimen Name', 'Unknown'))
                comparator = str(row.get('Comparator', 'Unknown'))
                used_trial_arms.append(f"{trial_id}||{product}||{comparator}||{idx}")
            
            return {
                'response': fallback_response,
                'used_trial_arms': used_trial_arms
            }
    
    def identify_relevant_metrics_from_context(self, query: str, response: str, df: pd.DataFrame) -> List[str]:
        """Use LLM to identify which metrics are most relevant based on query and response context"""
        
        # All possible metric columns
        all_metrics = [
            "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
            "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate",
            "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate",
            "Gr 3/4 TRAEs", "Gr ≥3 TRAEs", "Gr 3/4 TEAEs", "Gr ≥3 TEAEs"
        ]
        
        # Filter to available metrics in the dataframe
        available_metrics = []
        for metric in all_metrics:
            if metric in df.columns:
                # Check if metric has any non-null, non-empty values
                non_empty_values = df[metric].dropna()
                non_empty_values = non_empty_values[non_empty_values.astype(str).str.strip() != '']
                non_empty_values = non_empty_values[non_empty_values.astype(str).str.lower() != 'na']
                if len(non_empty_values) > 0:
                    available_metrics.append(metric)
        
        if not available_metrics:
            return []
        
        metrics_prompt = f"""
        Based on this clinical query and response, identify the 3-5 most relevant metrics to visualize.
        
        Query: {query}
        Response: {response}
        
        Available metrics: {', '.join(available_metrics)}
        
        Rules:
        1. Choose metrics that are specifically mentioned or most relevant to the query/response
        2. Prioritize metrics that would help compare treatments if comparison is involved
        3. Include safety metrics if adverse events are discussed
        4. Include survival metrics if survival outcomes are discussed  
        5. Include response metrics if efficacy is discussed
        6. Return 3-5 metrics maximum for clear visualization
        7. If efficacy is the focus, prioritize: ORR, mPFS, mOS
        8. If safety is the focus, prioritize: Gr 3/4 TRAEs, Gr ≥3 TRAEs
  
        Return ONLY a comma-separated list of metric names from the available metrics.
        Example: ORR, mPFS, mOS, Gr 3/4 TRAEs
        
        Answer: """
        
        try:
            from langchain.schema import HumanMessage
            metrics_response = self.llm.invoke([HumanMessage(content=metrics_prompt)])
            suggested_metrics_text = metrics_response.content.strip()
            
            # Parse the response and validate metrics
            suggested_metrics = [m.strip() for m in suggested_metrics_text.split(',')]
            valid_metrics = [m for m in suggested_metrics if m in available_metrics]
            
            # Limit to 5 metrics maximum
            return valid_metrics[:5]
            
        except Exception as e:
            # Fallback: use query-based heuristics
            query_lower = query.lower()
            response_lower = response.lower()
            
            mentioned_metrics = []
            
            # Check for specific metric mentions
            metric_keywords = {
                'orr': ['ORR'],
                'response': ['ORR', 'CR', 'PR'],
                'complete response': ['CR'],
                'partial response': ['PR'],
                'pfs': ['mPFS', '1-yr PFS Rate', '2-yr PFS Rate'],
                'progression': ['mPFS'],
                'os': ['mOS', '1-yr OS Rate', '2-yr OS Rate'],
                'survival': ['mOS', 'mPFS'],
                'safety': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs'],
                'adverse': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs'],
                'efficacy': ['ORR', 'mPFS', 'mOS']
            }
            
            combined_text = query_lower + " " + response_lower
            for keyword, metrics in metric_keywords.items():
                if keyword in combined_text:
                    for metric in metrics:
                        if metric in available_metrics and metric not in mentioned_metrics:
                            mentioned_metrics.append(metric)
            
            # Default metrics if none mentioned
            if not mentioned_metrics:
                default_metrics = ['ORR', 'mPFS', 'mOS', 'Gr 3/4 TRAEs']
                mentioned_metrics = [m for m in default_metrics if m in available_metrics][:4]
            
            return mentioned_metrics[:5]  # Limit to 5 metrics
    
    def _extract_used_trial_arms(self, response: str, search_results: pd.DataFrame, query: str = "") -> List[str]:
        """Extract trial arm IDs that the LLM mentioned it used with improved accuracy"""
        # Look for the TRIAL_ARMS_USED section
        arms_used_match = re.search(r'\*\*TRIAL_ARMS_USED:\*\*\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        
        if arms_used_match:
            arms_text = arms_used_match.group(1).strip()
            # Split by comma and clean up
            mentioned_arms = [arm.strip() for arm in arms_text.split(',')]
            
            # Validate that these trial arms exist in our search results
            valid_arms = []
            
            for mentioned_arm in mentioned_arms:
                # Parse the unique arm ID (format: trial_id||product_name||comparator||row_index)
                if '||' in mentioned_arm:
                    parts = mentioned_arm.split('||')
                    if len(parts) >= 4:
                        trial_part, product_part, comparator_part, row_idx = parts[0], parts[1], parts[2], parts[3]
                        
                        try:
                            # Try to get by row index first (most accurate)
                            row_idx = int(row_idx)
                            if row_idx in search_results.index:
                                # Validate that this row actually matches what we expect
                                row = search_results.loc[row_idx]
                                
                                # Additional validation: check if this arm matches the query intent
                                if self._validate_arm_matches_query(row, query, response):
                                    valid_arms.append(mentioned_arm)
                                    continue
                        except (ValueError, KeyError):
                            pass
                        
                        # Fallback to matching by content with validation
                        matching_rows = search_results[
                            (search_results['Trial Acronym/ID'].astype(str).str.contains(str(trial_part), case=False, na=False)) &
                            (search_results['Product/Regimen Name'].astype(str).str.contains(str(product_part), case=False, na=False)) &
                            (search_results['Comparator'].astype(str).str.contains(str(comparator_part), case=False, na=False))
                        ]
                        
                        for _, row in matching_rows.iterrows():
                            if self._validate_arm_matches_query(row, query, response):
                                actual_trial = str(row['Trial Acronym/ID'])
                                actual_product = str(row['Product/Regimen Name'])
                                actual_comparator = str(row['Comparator'])
                                actual_row_idx = row.name
                                valid_arms.append(f"{actual_trial}||{actual_product}||{actual_comparator}||{actual_row_idx}")
                                break  # Only take the first matching valid arm
            
            if valid_arms:
                return valid_arms
        
        # Fallback: try to extract from response content if TRIAL_ARMS_USED section not found
        return self._fallback_trial_arm_extraction(response, search_results, query)
    
    def _validate_arm_matches_query(self, row: pd.Series, query: str, response: str) -> bool:
        """Validate that a trial arm matches the intent of the query and response"""
        query_lower = query.lower()
        response_lower = response.lower()
        product_name = str(row.get('Product/Regimen Name', '')).lower()
        
        # Check for specific treatment intent in the query
        if 'monotherapy' in query_lower or 'mono' in query_lower:
            # If query asks for monotherapy, ensure we're not including combination arms
            combination_indicators = ['+', 'combination', 'combo', 'plus']
            if any(indicator in product_name for indicator in combination_indicators):
                return False
        
        if 'combination' in query_lower or 'combo' in query_lower:
            # If query asks for combination, ensure we're including combination arms
            combination_indicators = ['+', 'combination', 'combo', 'plus']
            if not any(indicator in product_name for indicator in combination_indicators):
                return False
        
        # For nivolumab monotherapy specifically
        if 'nivolumab' in query_lower and ('monotherapy' in query_lower or 'mono' in query_lower):
            if 'nivolumab' in product_name:
                # Make sure it's not a combination (no +, no other drug names)
                other_drugs = ['relatlimab', 'ipilimumab', 'pembrolizumab', 'dabrafenib', 'trametinib']
                if any(drug in product_name for drug in other_drugs) or '+' in product_name:
                    return False
                return True
            return False
        
        # For specific drug combinations mentioned in query
        drug_combinations = [
            (['nivolumab', 'ipilimumab'], ['nivolumab', 'ipilimumab']),
            (['nivolumab', 'relatlimab'], ['nivolumab', 'relatlimab']),
            (['dabrafenib', 'trametinib'], ['dabrafenib', 'trametinib']),
            (['pembrolizumab'], ['pembrolizumab'])
        ]
        
        # Check if query mentions specific drug combinations
        for query_drugs, product_drugs in drug_combinations:
            if all(drug in query_lower for drug in query_drugs):
                # Check if this product contains all the required drugs
                if all(drug in product_name for drug in product_drugs):
                    return True
        
        # For relatlimab + nivolumab combinations (legacy check)
        if ('relatlimab' in query_lower and 'nivolumab' in query_lower):
            if 'relatlimab' in product_name and 'nivolumab' in product_name:
                return True
        
        # For ipilimumab + nivolumab combinations
        if ('ipilimumab' in query_lower and 'nivolumab' in query_lower):
            if 'ipilimumab' in product_name and 'nivolumab' in product_name:
                return True
        
        return True  # Default to true if no specific validation needed
    
    def _fallback_trial_arm_extraction(self, response: str, search_results: pd.DataFrame, query: str = "") -> List[str]:
        """Fallback method to extract trial arms from response content with better logic"""
        required_cols = ['Trial Acronym/ID', 'Product/Regimen Name', 'Comparator']
        if not all(col in search_results.columns for col in required_cols):
            return []
        
        mentioned_arms = []
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Look for specific trials mentioned in response
        trial_patterns = {
            'checkmate-067': ['checkmate-067', 'checkmate067', 'cm-067', 'cm067'],
            'relativity-047': ['relativity-047', 'relativity047', 'rel-047', 'rel047'],
            'keynote-006': ['keynote-006', 'keynote006', 'kn-006', 'kn006'],
        }
        
        mentioned_trials = []
        for trial, patterns in trial_patterns.items():
            if any(pattern in response_lower or pattern in query_lower for pattern in patterns):
                mentioned_trials.append(trial)
        
        # Filter search results to only mentioned trials
        if mentioned_trials:
            trial_filtered_results = search_results[
                search_results['Trial Acronym/ID'].str.lower().str.contains('|'.join(mentioned_trials), case=False, na=False)
            ]
        else:
            trial_filtered_results = search_results.head(10)  # Fallback to top results
        
        # Now apply treatment-specific filtering based on query intent
        for idx, row in trial_filtered_results.iterrows():
            if self._validate_arm_matches_query(row, query, response):
                trial_id = str(row['Trial Acronym/ID'])
                product = str(row['Product/Regimen Name'])
                comparator = str(row['Comparator'])
                arm_id = f"{trial_id}||{product}||{comparator}||{idx}"
                if arm_id not in mentioned_arms:
                    mentioned_arms.append(arm_id)
        
        # If still no specific matches, return top results based on similarity scores but validate them
        if not mentioned_arms:
            for idx, row in search_results.head(4).iterrows():
                if self._validate_arm_matches_query(row, query, response):
                    trial_id = str(row['Trial Acronym/ID'])
                    product = str(row['Product/Regimen Name'])
                    comparator = str(row['Comparator'])
                    mentioned_arms.append(f"{trial_id}||{product}||{comparator}||{idx}")
        
        return mentioned_arms
    
    def _create_fallback_response(self, query: str, results: pd.DataFrame) -> str:
        """Create a structured response if LLM fails"""
        response = f"## Results for: {query}\n\n"
        response += f"Found {len(results)} relevant clinical trials:\n\n"
        
        # Key columns to display
        key_cols = ['Product/Regimen Name', 'Trial Acronym/ID', 'Therapeutic Area', 
                   'ORR', 'mPFS', 'mOS', 'Highest Phase']
        available_cols = [col for col in key_cols if col in results.columns]
        
        if available_cols:
            # Create a simple table
            response += "| " + " | ".join(available_cols) + " |\n"
            response += "| " + " | ".join(["---"] * len(available_cols)) + " |\n"
            
            for _, row in results.head(5).iterrows():
                row_values = []
                for col in available_cols:
                    value = str(row.get(col, "N/A"))[:30]
                    row_values.append(value)
                response += "| " + " | ".join(row_values) + " |\n"
        
        return response
    
    def get_trial_arms_for_visualization(self, used_trial_arms: List[str], search_results: pd.DataFrame) -> pd.DataFrame:
        """Get the exact trial arms used by LLM for visualization"""
        if not used_trial_arms or search_results.empty:
            return pd.DataFrame()
        
        viz_rows = []
        
        for arm_id in used_trial_arms:
            if '||' in arm_id:
                parts = arm_id.split('||')
                
                # Handle different formats
                if len(parts) >= 4:
                    # Format: trial||product||comparator||row_index
                    trial_part, product_part, comparator_part, row_idx = parts[0], parts[1], parts[2], parts[3]
                    
                    try:
                        # Try to get by row index first (most accurate)
                        row_idx = int(row_idx)
                        if row_idx in search_results.index:
                            viz_rows.append(search_results.loc[row_idx])
                            continue
                    except (ValueError, KeyError):
                        pass
                    
                    # Fallback to matching by content
                    matching_rows = search_results[
                        (search_results['Trial Acronym/ID'].astype(str) == trial_part) &
                        (search_results['Product/Regimen Name'].astype(str) == product_part) &
                        (search_results['Comparator'].astype(str) == comparator_part)
                    ]
                    
                    if not matching_rows.empty:
                        viz_rows.append(matching_rows.iloc[0])
                        continue
                    
                    # Try partial matching
                    partial_match = search_results[
                        (search_results['Trial Acronym/ID'].astype(str).str.contains(str(trial_part), case=False, na=False)) &
                        (search_results['Product/Regimen Name'].astype(str).str.contains(str(product_part), case=False, na=False))
                    ]
                    if not partial_match.empty:
                        viz_rows.append(partial_match.iloc[0])
                
                elif len(parts) >= 2:
                    # Legacy format: trial||product
                    trial_part, product_part = parts[0], parts[1]
                    
                    matching_rows = search_results[
                        (search_results['Trial Acronym/ID'].astype(str) == trial_part) &
                        (search_results['Product/Regimen Name'].astype(str) == product_part)
                    ]
                    
                    if not matching_rows.empty:
                        viz_rows.append(matching_rows.iloc[0])
                    else:
                        # Try partial matching
                        partial_match = search_results[
                            (search_results['Trial Acronym/ID'].astype(str).str.contains(str(trial_part), case=False, na=False)) &
                            (search_results['Product/Regimen Name'].astype(str).str.contains(str(product_part), case=False, na=False))
                        ]
                        if not partial_match.empty:
                            viz_rows.append(partial_match.iloc[0])
        
        if viz_rows:
            viz_df = pd.DataFrame(viz_rows)
            # Remove any duplicates and reset index
            viz_df = viz_df.drop_duplicates().reset_index(drop=True)
            return viz_df
        
        return pd.DataFrame()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return results with visualization data"""
        # Perform hybrid search
        search_results = self.hybrid_search(query)
        
        # Generate response and get used trial arms
        response_data = self.generate_response(query, search_results)
        response = response_data['response']
        used_trial_arms = response_data['used_trial_arms']
        
        # Determine if visualization should be created
        should_visualize = self.should_create_visualization(query, response)
        
        # Debug information - can be removed in production
        print(f"DEBUG - should_visualize: {should_visualize}")
        print(f"DEBUG - used_trial_arms count: {len(used_trial_arms)}")
        print(f"DEBUG - used_trial_arms: {used_trial_arms}")
        
        # Prepare visualization data if applicable
        viz_data = None
        if should_visualize and not search_results.empty:
            # Get trial arms that LLM actually used
            viz_df = self.get_trial_arms_for_visualization(used_trial_arms, search_results)
            
            print(f"DEBUG - viz_df shape: {viz_df.shape if not viz_df.empty else 'Empty'}")
            
            # Lower the threshold - show visualization even with 1 arm if it's a comparison query
            comparison_keywords = ['compare', 'comparison', 'versus', 'vs', 'and', 'between']
            is_comparison = any(keyword in query.lower() for keyword in comparison_keywords)
            
            min_arms_needed = 1 if is_comparison else 2
            
            if not viz_df.empty and len(viz_df) >= min_arms_needed:
                # Get metrics for visualization using LLM
                viz_metrics = self.identify_relevant_metrics_from_context(query, response, viz_df)
                
                print(f"DEBUG - viz_metrics: {viz_metrics}")
                
                if viz_metrics:
                    viz_data = {
                        'df': viz_df,
                        'trial_col': 'Product/Regimen Name',
                        'metrics': viz_metrics,
                        'query': query,
                        'used_trial_arms': used_trial_arms
                    }
                    print("DEBUG - viz_data created successfully")
                else:
                    print("DEBUG - No viz_metrics found")
            else:
                print(f"DEBUG - Not enough arms for visualization. Need >= {min_arms_needed}, got {len(viz_df) if not viz_df.empty else 0}")
        
        return {
            'response': response,
            'search_results': search_results,
            'viz_data': viz_data,
            'used_trial_arms': used_trial_arms,
            'num_results': len(search_results),
            'should_visualize': should_visualize
        }


def get_text_color_for_theme():
    """Get appropriate text color based on Streamlit theme"""
    # Check if we're in dark mode by examining CSS
    # This is a workaround since we can't directly detect theme
    return "white"  # Default to white, will be overridden by CSS


def display_bar_charts(df: pd.DataFrame, trial_col: str, metric_cols: List[str], key_prefix: str = ""):
    """Display bar charts for metrics comparison with dynamic height and improved layout"""
    if df.empty:
        st.info("No data available for visualization")
        return
    
    # Create display names combining trial and product for clarity - with better truncation
    if 'Trial Acronym/ID' in df.columns and 'Product/Regimen Name' in df.columns:
        df['Display_Name'] = df.apply(lambda row: 
            f"{str(row['Trial Acronym/ID']).split('/')[0]} - {str(row['Product/Regimen Name'])[:60]}", axis=1)
    else:
        df['Display_Name'] = df[trial_col].apply(lambda x: str(x)[:70])  # Increased character limit
    
    df_viz = df.copy()
    
    # Prepare data for melting - only include available metrics
    available_metrics = [col for col in metric_cols if col in df_viz.columns]
    if not available_metrics:
        st.info("No metrics available for visualization")
        return
    
    # Melt data for plotting
    melted = pd.melt(df_viz[['Display_Name'] + available_metrics], 
                     id_vars='Display_Name',
                     var_name="Metric", 
                     value_name="RawValue")
    
    # Clean and process values
    melted["RawValue"] = melted["RawValue"].astype(str).str.strip()
    
    # Define missing values - EXCLUDE "NR" and "Not Reached" from missing values
    missing_values = {"", "na", "n/a", "nan", "none", "null", "not available"}
    # Define "Not Reached" values separately - more comprehensive list
    not_reached_values = {"nr", "not reached", "not_reached", "notreached", "not-reached"}
    
    # Create cleaned lowercase version for comparison
    melted["RawValue_lower"] = melted["RawValue"].str.lower().str.strip()
    
    melted["IsMissing"] = melted["RawValue_lower"].isin(missing_values)
    melted["IsNotReached"] = melted["RawValue_lower"].isin(not_reached_values)
    
    # Extract numeric values more robustly
    melted["Value"] = melted["RawValue"].str.replace('%', '', regex=False)
    melted["Value"] = melted["Value"].str.replace('months', '', regex=False)
    melted["Value"] = melted["Value"].str.extract(r'([\d.]+)', expand=False)
    melted["Value"] = pd.to_numeric(melted["Value"], errors='coerce')
    
    # Set plot values (use small value for missing data and NR)
    melted["PlotValue"] = melted["Value"].fillna(0)
    melted.loc[melted["IsMissing"], "PlotValue"] = 0.1
    melted.loc[melted["IsNotReached"], "PlotValue"] = 0.1
    
    # Create display text with more robust NR handling
    def create_display_text(row):
        raw_val = str(row["RawValue"]).strip()
        raw_val_lower = raw_val.lower().strip()
        
        # First check for explicit NR patterns (most comprehensive)
        if (raw_val_lower in ['nr', 'not reached', 'not_reached', 'notreached', 'not-reached'] or
            'not reached' in raw_val_lower or 
            raw_val_lower == 'nr' or
            raw_val.upper() == 'NR'):
            return "NR"
        
        # Check for missing/empty values
        elif (raw_val_lower in ['', 'na', 'n/a', 'nan', 'none', 'null', 'not available'] or
              raw_val_lower == 'nan' or 
              pd.isna(raw_val) or
              raw_val == '' or
              raw_val == 'None'):
            return "N/A"
        
        # Regular numeric or text values
        else:
            return raw_val.upper().replace('MONTHS', '').replace('MONTH', '').strip()
    
    melted["DisplayText"] = melted.apply(create_display_text, axis=1)
    
    # Additional safety check for survival metrics that might have been missed
    survival_metrics = melted["Metric"].str.contains("OS|PFS", case=False, na=False)
    potentially_nr = (melted["DisplayText"] == "N/A") & survival_metrics
    
    # If we find survival metrics showing N/A, check if raw value could be NR
    for idx in melted[potentially_nr].index:
        raw_val = str(melted.loc[idx, "RawValue"]).strip().upper()
        if raw_val in ['NR', 'NOT REACHED', 'NOTREACHED', 'NOT_REACHED']:
            melted.loc[idx, "DisplayText"] = "NR"
    
    # Create distinct colors for each trial arm (expand color palette for more arms)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    arm_names = melted['Display_Name'].unique()
    
    # Extend color palette if needed
    while len(colors) < len(arm_names):
        colors.extend(colors)
    
    arm_color_map = {name: colors[i % len(colors)] for i, name in enumerate(arm_names)}
    
    # Calculate appropriate dimensions - IMPROVED HEIGHT CALCULATION
    num_metrics = len(available_metrics)
    num_arms = len(df_viz)
    
    # More dynamic height calculation based on number of arms
    if num_arms <= 2:
        # For 2 or fewer arms, make it more compact (50% thinner as requested)
        base_height_per_arm = 60  # Reduced from 100
        min_height = 300
    elif num_arms <= 4:
        base_height_per_arm = 80
        min_height = 400
    else:
        base_height_per_arm = 100
        min_height = 500
    
    chart_height = max(min_height, base_height_per_arm * num_arms + 150)
    
    # Create bar chart
    fig = px.bar(
        melted,
        x="PlotValue",
        y="Display_Name",
        color="Display_Name",
        facet_col="Metric",
        facet_col_wrap=min(num_metrics, 4),  # Max 4 columns
        orientation="h",
        text="DisplayText",
        color_discrete_map=arm_color_map,
        title=f"Clinical Metrics Comparison"
    )
    
    # Update traces for better text positioning
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=10, color="black"),
        cliponaxis=False,
        # Make bars thinner for fewer rows
        marker=dict(
            line=dict(width=1, color='rgba(0,0,0,0.2)')
        )
    )
    
    # IMPROVED LAYOUT with better margins for long labels
    fig.update_layout(
        height=chart_height,
        showlegend=False,
        # Increased left margin significantly for longer product names
        margin=dict(l=400, r=150, t=100, b=50),  # Increased from l=250 to l=400
        plot_bgcolor="white",
        paper_bgcolor="white",
        # Adjust bar spacing for thinner appearance when few rows
        bargap=0.3 if num_arms <= 2 else 0.2  # More gap between bars for fewer arms
    )
    
    # Clean facet titles and axes
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1], 
        font=dict(size=12, color="black")
    ))
    
    fig.for_each_xaxis(lambda x: x.update(
        title='', 
        showticklabels=False,
        gridcolor="rgba(0,0,0,0.1)"
    ))
    
    # IMPROVED Y-AXIS formatting for better label display
    fig.for_each_yaxis(lambda y: y.update(
        title='',
        tickfont=dict(size=9, color="black"),
        # Ensure labels don't get cut off
        tickmode='linear',
        # Add some padding
        automargin=True,
        # Wrap long text by adjusting tick text
        tickangle=0  # Keep horizontal
    ))
    
    # For very long labels, we can also adjust the tick text directly
    # This ensures product names are more visible
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'y') and trace.y is not None:
            # Trace y values correspond to Display_Name
            updated_labels = []
            for label in trace.y:
                if isinstance(label, str) and len(label) > 50:
                    # Break long labels into multiple lines for better readability
                    words = label.split(' - ')
                    if len(words) >= 2 and len(words[1]) > 40:
                        # Split the product name part if it's too long
                        trial_part = words[0]
                        product_part = words[1]
                        if len(product_part) > 40:
                            # Find a good breaking point
                            mid_point = len(product_part) // 2
                            space_near_mid = product_part.find(' ', mid_point)
                            if space_near_mid != -1 and space_near_mid < len(product_part) - 10:
                                product_part = product_part[:space_near_mid] + '<br>' + product_part[space_near_mid+1:]
                        updated_labels.append(f"{trial_part} - {product_part}")
                    else:
                        updated_labels.append(label)
                else:
                    updated_labels.append(label)
    
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")


def create_visualization(viz_data: Dict[str, Any], message_idx: int):
    """Create interactive visualization for search results with metric selection"""
    if not viz_data:
        return
    
    df = viz_data['df']
    trial_col = viz_data['trial_col']
    default_metrics = viz_data['metrics']
    used_trial_arms = viz_data.get('used_trial_arms', [])
    
    if not df.empty:

        all_possible_metrics = [
            "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
            "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate", "4-yr PFS Rate", "5-yr PFS Rate",
            "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate", "4-yr OS Rate", "5-yr OS Rate",
            "Gr 3/4 TRAEs", "Gr ≥3 TRAEs", "Gr 3/4 TEAEs", "Gr ≥3 TEAEs",
            "Gr 3/4 AEs", "Gr ≥3 AEs", "Gr 3/4 irAEs", "Gr ≥3 irAEs",
            "Tx-related Deaths (Gr 5 TRAEs)", "All Deaths (Gr 5 AEs)",
            "PFS HR (p-value)", "OS HR (p-value)", "Median Follow-up (mFU)",
            "N (No of All Enrolled pts)", "n (Enrolled pts in arm)"
        ]
        
        # Filter to only metrics that exist in the dataframe and have some data
        available_metrics = []
        for metric in all_possible_metrics:
            if metric in df.columns:
                # Check if metric has any non-null, non-empty values
                non_empty_values = df[metric].dropna()
                non_empty_values = non_empty_values[non_empty_values.astype(str).str.strip() != '']
                non_empty_values = non_empty_values[non_empty_values.astype(str).str.lower() != 'na']
                if len(non_empty_values) > 0:
                    available_metrics.append(metric)
        
        if available_metrics:
            # Ensure default metrics are in available metrics
            valid_default_metrics = [m for m in default_metrics if m in available_metrics]
            if not valid_default_metrics:
                valid_default_metrics = available_metrics[:4]  # Use first 4 if no defaults are valid
            
            # Multi-select widget for metrics
            selected_metrics = st.multiselect(
                "📈 **Select additional metrics to compare:**",
                options=available_metrics,
                default=valid_default_metrics,
                help="The chart shows the most relevant metrics based on your query. You can add or remove metrics using this selector.",
                key=f"metrics_selector_{message_idx}"
            )
            
            # Create the visualization with selected metrics
            if selected_metrics:
                display_bar_charts(df, trial_col, selected_metrics, f"viz_{message_idx}")
            else:
                st.warning("⚠️ Please select at least one metric to visualize.")
        
        else:
            st.warning("⚠️ No metrics with valid data are available for visualization.")


def main():
    st.markdown('<h1 class="main-header">Clinical Trial Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_system' not in st.session_state:
        try:
            st.session_state.chat_system = ClinicalChatWithFAISS()
            st.success("✅ System ready! Ask me anything about clinical trials.")
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            st.stop()
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Re-display visualizations
            if message["role"] == "assistant" and "viz_data" in message:
                create_visualization(message["viz_data"], i)
    
    # Chat input
    if prompt := st.chat_input("Ask about clinical trials..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing..."):
                try:
                    # Get results
                    results = st.session_state.chat_system.process_query(prompt)
                    
                    # Display response
                    st.markdown(results['response'])
                    
                    # Create visualization if applicable
                    if results['viz_data']:
                        create_visualization(results['viz_data'], len(st.session_state.messages))
                    elif results['should_visualize'] and not results['viz_data']:
                        st.info("💡 This query could benefit from visualization, but insufficient comparative data was found.")
                    
                    # Add to history
                    message_data = {
                        "role": "assistant",
                        "content": results['response']
                    }
                    if results['viz_data']:
                        message_data['viz_data'] = results['viz_data']
                    
                    st.session_state.messages.append(message_data)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
