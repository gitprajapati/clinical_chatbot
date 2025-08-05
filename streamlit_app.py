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
    
    def __init__(self, df: pd.DataFrame, model: str = "gpt-4o"):
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
        
        # Enhanced trial acronym aliases with actual data analysis
        self.trial_aliases = {
            # CHECKMATE variations
            'checkmate-511': ['cm-511', 'cm511', 'cm 511', 'checkmate511', 'checkmate 511', 'check mate 511'],
            'checkmate-067': ['cm-067', 'cm067', 'cm 067', 'checkmate067', 'checkmate 067', 'check mate 067'],
            'checkmate-066': ['cm-066', 'cm066', 'cm 066', 'checkmate066', 'checkmate 066', 'check mate 066'],
            'checkmate-238': ['cm-238', 'cm238', 'cm 238', 'checkmate238', 'checkmate 238', 'check mate 238'],
            'checkmate-76k': ['cm-76k', 'cm76k', 'cm 76k', 'checkmate76k', 'checkmate 76k', 'check mate 76k'],
            
            # RELATIVITY variations
            'relativity-047': ['rel-047', 'rel047', 'rel 047', 'relativity047', 'relativity 047', 'relativity047'],
            
            # KEYNOTE variations
            'keynote-006': ['kn-006', 'kn006', 'kn 006', 'keynote006', 'keynote 006', 'key note 006'],
            'keynote-252': ['kn-252', 'kn252', 'kn 252', 'keynote252', 'keynote 252', 'key note 252'],
            'keynote-716': ['kn-716', 'kn716', 'kn 716', 'keynote716', 'keynote 716', 'key note 716'],
            
            # DREAMSEQ variations - ENHANCED
            'dreamseq': ['dream-seq', 'dream seq', 'dream sequence', 'dreamsequence', 'dream_seq', 'dreams eq'],
            
            # Other trials
            'combi-d': ['combid', 'combi d', 'combination d', 'combi-d trial'],
            'combi-v': ['combiv', 'combi v', 'combination v', 'combi-v trial'],
            'combi-i': ['combii', 'combi i', 'combination i', 'combi-i trial'],
            'columbus': ['col', 'columbus trial', 'columbus study'],
            'inspire150': ['imspire-150', 'imspire 150', 'inspire 150', 'imspire150'],
            'cobrim': ['co-brim', 'co brim', 'cobrim trial'],
            
            # Add more based on actual data
            'brim-2': ['brim2', 'brim 2'],
            'brim-3': ['brim3', 'brim 3'],
            'brim-7': ['brim7', 'brim 7'],
        }
        
        # Enhanced phase aliases
        self.phase_aliases = {
            'ph3': ['phase 3', 'phase iii', 'phase3', 'phase-3', 'phase-iii', 'p3', 'piii'],
            'ph2': ['phase 2', 'phase ii', 'phase2', 'phase-2', 'phase-ii', 'p2', 'pii'],
            'ph1': ['phase 1', 'phase i', 'phase1', 'phase-1', 'phase-i', 'p1', 'pi'],
            'ph4': ['phase 4', 'phase iv', 'phase4', 'phase-4', 'phase-iv', 'p4', 'piv'],
            'ph1/2': ['phase 1/2', 'phase i/ii', 'phase1/2', 'phase-1/2', 'p1/2'],
            'ph2/3': ['phase 2/3', 'phase ii/iii', 'phase2/3', 'phase-2/3', 'p2/3'],
        }
        
        # Enhanced company aliases
        self.company_aliases = {
            'bms': ['bristol myers squibb', 'bristol-myers squibb', 'bristol myers', 'bristol-myers', 'bms pharma'],
            'merck': ['merck/mds', 'merck us', 'us-based merck', 'merck & co', 'merck & co.', 'msd'],
            'roche': ['genentech', 'roche genentech', 'f. hoffmann-la roche'],
            'novartis': ['novartis pharma', 'novartis ag', 'novartis oncology'],
            'pfizer': ['pfizer inc', 'pfizer inc.', 'pfizer oncology'],
            'array': ['array biopharma', 'array pharmaceuticals'],
            'pierre fabre': ['pierre fabre medicament', 'pierre fabre pharmaceuticals'],
        }
        
        # Enhanced drug aliases
        self.drug_aliases = {
            'nivolumab': ['nivo', 'opdivo', 'bms-936558'],
            'ipilimumab': ['ipi', 'yervoy', 'mdx-010'],
            'pembrolizumab': ['pembro', 'keytruda', 'mk-3475'],
            'atezolizumab': ['atezo', 'tecentriq', 'mpdl3280a'],
            'vemurafenib': ['vemu', 'zelboraf', 'plx4032'],
            'dabrafenib': ['dabra', 'tafinlar', 'gsk2118436'],
            'trametinib': ['trame', 'mekinist', 'gsk1120212'],
            'cobimetinib': ['cobi', 'cotellic', 'gdc-0973'],
            'encorafenib': ['enco', 'braftovi', 'lgx818'],
            'binimetinib': ['bini', 'mektovi', 'arry-162'],
            'relatlimab': ['rela', 'bms-986016'],
        }
        
        # Clinical outcomes aliases - comprehensive mapping
        self.outcome_aliases = {
            # Efficacy outcomes
            'orr': ['overall response rate', 'response rate', 'objective response rate', 'complete + partial response'],
            'cr': ['complete response', 'complete response rate', 'complete remission'],
            'pr': ['partial response', 'partial response rate', 'partial remission'],
            'pfs': ['progression free survival', 'progression-free survival', 'median pfs', 'mpfs', 'pfs rate'],
            'os': ['overall survival', 'median os', 'mos', 'os rate'],
            'dor': ['duration of response', 'median dor', 'mdor', 'response duration'],
            'dcr': ['disease control rate', 'clinical benefit rate'],
            
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
                'grade 3/4 treatment-related adverse events',
                'gr 3/4 treatment related aes'
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
        """Extract and expand search terms from query with enhanced logic"""
        # Get expanded terms
        expanded_terms = self._expand_query_terms(query)
        
        # Enhanced pattern matching for trials
        trial_patterns = [
            r'checkmate[-\s]?\d+', r'keynote[-\s]?\d+', r'relativity[-\s]?\d+',
            r'dreamseq', r'dream[-\s]?seq', r'combi[-\s][div]', r'columbus',
            r'inspire\d+', r'imspire[-\s]?\d+', r'cobrim', r'brim[-\s]?\d+'
        ]
        
        found_trials = []
        for pattern in trial_patterns:
            matches = re.findall(pattern, query.lower())
            found_trials.extend(matches)
        
        # Enhanced drug detection
        drug_terms = ['nivolumab', 'ipilimumab', 'relatlimab', 'pembrolizumab', 'atezolizumab', 
                     'vemurafenib', 'dabrafenib', 'trametinib', 'cobimetinib', 'encorafenib', 'binimetinib',
                     'opdivo', 'yervoy', 'keytruda', 'tecentriq', 'zelboraf', 'tafinlar', 'mekinist', 'cotellic']
        
        found_drugs = []
        for term in expanded_terms:
            for drug_term in drug_terms:
                if drug_term in term or term in drug_term:
                    found_drugs.append(drug_term)
        
        # Enhanced outcome detection
        outcome_terms = ['orr', 'pfs', 'os', 'response', 'survival', 'progression', 'overall', 'cr', 'pr', 'dor', 
                        'traes', 'teaes', 'iraes', 'adverse', 'safety', 'efficacy', 'dcr']
        
        found_outcomes = []
        for term in expanded_terms:
            for outcome_term in outcome_terms:
                if outcome_term in term:
                    found_outcomes.append(outcome_term)
        
        # Extract quoted terms and numbers
        quoted_terms = re.findall(r'"([^"]*)"', query)
        number_terms = re.findall(r'\b\d+\b', query)
        
        # Extract comparison indicators
        comparison_words = ['vs', 'versus', 'compare', 'comparison', 'against', 'compared to']
        is_comparison = any(word in query.lower() for word in comparison_words)
        
        return {
            'trials': list(set(found_trials)),
            'drugs': list(set(found_drugs)),
            'outcomes': list(set(found_outcomes)),
            'quoted': quoted_terms,
            'numbers': number_terms,
            'expanded_terms': expanded_terms,
            'all_terms': query.lower().split(),
            'is_comparison': is_comparison
        }
    
    def search_dataframe_optimized(self, search_terms: Dict[str, List[str]]) -> pd.DataFrame:
        """Enhanced pandas-based search with comprehensive alias expansion"""
        
        # Start with all rows
        mask = pd.Series(True, index=self.df.index)
        search_performed = False
        
        # Get all search terms including expanded ones
        all_search_terms = []
        for key in ['trials', 'drugs', 'outcomes', 'quoted', 'numbers', 'expanded_terms']:
            if key in search_terms:
                all_search_terms.extend(search_terms[key])
        
        # Remove duplicates and filter out very short terms
        all_search_terms = list(set([term for term in all_search_terms if term and len(term) >= 2]))
        
        if all_search_terms:
            combined_mask = pd.Series(False, index=self.df.index)
            
            # Define search priority: high-priority columns first
            priority_columns = [
                'Trial Acronym/ID', 'Product/Regimen Name', 'Active Developers (Companies Names)', 
                'Highest Phase', 'Therapeutic Indication', 'Therapeutic Area'
            ]
            
            # Search with different strategies
            for term in all_search_terms:
                if len(term) < 2:
                    continue
                
                term_mask = pd.Series(False, index=self.df.index)
                
                # Strategy 1: Exact match in priority columns
                for col in priority_columns:
                    if col in self.df.columns:
                        try:
                            col_mask = self.df[col].str.contains(re.escape(term), case=False, na=False, regex=True)
                            term_mask |= col_mask
                        except:
                            continue
                
                # Strategy 2: Partial match in all text columns
                if not term_mask.any():
                    for col in self.text_columns:
                        if col not in priority_columns:  # Skip already searched columns
                            try:
                                col_mask = self.df[col].str.contains(re.escape(term), case=False, na=False, regex=True)
                                term_mask |= col_mask
                            except:
                                continue
                
                # Strategy 3: Word boundary search for better accuracy
                if not term_mask.any() and len(term) > 3:
                    for col in self.text_columns:
                        try:
                            # Use word boundary regex for more precise matching
                            pattern = r'\b' + re.escape(term) + r'\b'
                            col_mask = self.df[col].str.contains(pattern, case=False, na=False, regex=True)
                            term_mask |= col_mask
                        except:
                            continue
                
                combined_mask |= term_mask
            
            if combined_mask.any():
                mask = combined_mask
                search_performed = True
        
        # If no specific search performed or no results, return relevant subset
        if not search_performed or not mask.any():
            return self.df.head(20)
        
        # For comparison queries, try to ensure we get multiple trials
        if search_terms.get('is_comparison', False):
            result_df = self.df[mask]
            # If we have trial names mentioned, try to get at least 2 different trials
            trial_col = self._get_trial_id_column()
            if trial_col and len(result_df[trial_col].unique()) == 1 and len(all_search_terms) > 1:
                # Try a broader search to find the second trial
                broader_mask = pd.Series(False, index=self.df.index)
                for term in all_search_terms:
                    if len(term) >= 3:
                        for col in self.text_columns:
                            try:
                                col_mask = self.df[col].str.contains(re.escape(term), case=False, na=False, regex=True)
                                broader_mask |= col_mask
                            except:
                                continue
                result_df = self.df[broader_mask] if broader_mask.any() else result_df
        else:
            result_df = self.df[mask]
        
        return result_df
    
    def _get_trial_id_column(self) -> Optional[str]:
        """Get the primary trial identification column"""
        trial_columns = ['Trial Acronym/ID', 'Trial Name', 'Trial ID', 'Product/Regimen Name']
        for col in trial_columns:
            if col in self.df.columns:
                return col
        return None
    
    def retrieve_data_ultra_fast(self, query: str) -> Dict[str, Any]:
        """Enhanced data retrieval with comprehensive search"""
        try:
            # Extract and expand search terms
            search_terms = self.extract_search_terms(query)
            
            # Search dataframe with enhanced logic
            filtered_df = self.search_dataframe_optimized(search_terms)
            
            # Limit results for performance but ensure comparison queries get enough data
            max_records = 100 if search_terms.get('is_comparison', False) else 50
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
        """Enhanced identification of relevant columns"""
        query_lower = query.lower()
        
        # Start with default table columns
        relevant_cols = [col for col in self.default_table_columns if col in df.columns]
        
        # Add trial identification columns for comparisons
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus', 'against']):
            comparison_cols = ['Trial Acronym/ID', 'Product/Regimen Name']
            relevant_cols.extend([col for col in comparison_cols if col in df.columns])
        
        # Add outcome-specific columns based on query
        outcome_column_mappings = {
            'orr': ['ORR', 'ORR Notes'],
            'cr': ['CR', 'CR Notes'],
            'pr': ['PR', 'PR Notes'],
            'pfs': ['mPFS', 'PFS Notes', '1-yr PFS Rate', '2-yr PFS Rate', '3-yr PFS Rate'],
            'os': ['mOS', 'OS Notes', '1-yr OS Rate', '2-yr OS Rate', '3-yr OS Rate'],
            'dor': ['mDoR', 'DoR Notes'],
            'dcr': ['DCR', 'DCR Notes'],
            'safety': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs', 'Gr 3/4 AEs', 'Gr ≥3 AEs', 'Key AEs'],
            'traes': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs', 'Tx-related Deaths (Gr 5 TRAEs)'],
            'teaes': ['Gr 3/4 TEAEs', 'Gr ≥3 TEAEs'],  
            'iraes': ['Gr 3/4 irAEs', 'Gr ≥3 irAEs'],
            'adverse': ['Key AEs', 'Gr 3/4 AEs', 'Gr ≥3 AEs', 'All Deaths (Gr 5 AEs)'],
            'efficacy': ['ORR', 'mPFS', 'mOS', 'CR', 'PR', 'DCR', 'mDoR'],
            'survival': ['mPFS', 'mOS', '1-yr PFS Rate', '2-yr PFS Rate', '1-yr OS Rate', '2-yr OS Rate'],
            'response': ['ORR', 'CR', 'PR', 'DCR', 'mDoR']
        }
        
        for keyword, columns in outcome_column_mappings.items():
            if keyword in query_lower:
                for col_pattern in columns:
                    matching_cols = [col for col in df.columns if col_pattern in col]
                    relevant_cols.extend(matching_cols)
        
        # Add phase and development info for general queries
        if any(word in query_lower for word in ['phase', 'development', 'company', 'developer']):
            dev_cols = ['Highest Phase', 'Active Developers (Companies Names)', 'Therapeutic Area']
            relevant_cols.extend([col for col in dev_cols if col in df.columns])
        
        return list(set(relevant_cols))
    
    def extract_metrics_from_question(self, question: str, metric_list: List[str]) -> List[str]:
        """Extracts metrics mentioned in the user's question."""
        q_lower = question.lower()
        mentioned_metrics = []
        
        for metric in metric_list:
            # Split metric name into parts and check if any part is in the question
            metric_parts = re.split(r'[\s\-/()]+', metric.lower())
            if any(part in q_lower for part in metric_parts if len(part) >= 3):
                mentioned_metrics.append(metric)
        
        return mentioned_metrics

    def identify_specific_trials_for_visualization(self, query: str, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently identify specific trials mentioned in the query for visualization.
        This handles comparison queries and filters the dataframe to show only relevant trials.
        """
        query_lower = query.lower()
        
        # Get trial identification column
        trial_col = self._get_trial_id_column()
        if not trial_col or trial_col not in filtered_df.columns:
            return filtered_df
        
        # For comparison queries, try to identify specific trials mentioned
        comparison_indicators = ['vs', 'versus', 'compare', 'comparison', 'against', 'compared to', 'with']
        is_comparison = any(word in query_lower for word in comparison_indicators)
        
        if is_comparison:
            # Extract specific trial/regimen names mentioned in the query
            specific_trials = []
            
            # Look for specific trial patterns in query
            trial_patterns = [
                r'checkmate[-\s]?\d+[a-z]*', r'keynote[-\s]?\d+[a-z]*', r'relativity[-\s]?\d+[a-z]*',
                r'dreamseq', r'dream[-\s]?seq', r'combi[-\s][div]', r'columbus',
                r'inspire\d+[a-z]*', r'imspire[-\s]?\d+[a-z]*', r'cobrim', r'brim[-\s]?\d+[a-z]*'
            ]
            
            for pattern in trial_patterns:
                matches = re.findall(pattern, query_lower)
                specific_trials.extend(matches)
            
            # Look for drug combinations mentioned
            drug_combinations = []
            if 'nivolumab' in query_lower:
                if 'ipilimumab' in query_lower:
                    drug_combinations.append('nivolumab + ipilimumab')
                    drug_combinations.append('nivolumab+ipilimumab')
                    drug_combinations.append('opdivo + yervoy')
                if 'relatlimab' in query_lower:
                    drug_combinations.append('nivolumab + relatlimab')
                    drug_combinations.append('nivolumab+relatlimab')
                    drug_combinations.append('opdivo + relatlimab')
            
            if 'pembrolizumab' in query_lower:
                drug_combinations.append('pembrolizumab')
                drug_combinations.append('keytruda')
            
            if 'dabrafenib' in query_lower and 'trametinib' in query_lower:
                drug_combinations.append('dabrafenib + trametinib')
                drug_combinations.append('dabrafenib+trametinib')
                drug_combinations.append('tafinlar + mekinist')
            
            # Create filter based on specific mentions
            specific_filter = pd.Series(False, index=filtered_df.index)
            
            # Filter by specific trials
            for trial in specific_trials:
                trial_mask = filtered_df[trial_col].str.contains(trial, case=False, na=False)
                specific_filter |= trial_mask
            
            # Filter by drug combinations in Product/Regimen Name
            if 'Product/Regimen Name' in filtered_df.columns:
                for combo in drug_combinations:
                    combo_mask = filtered_df['Product/Regimen Name'].str.contains(combo, case=False, na=False)
                    specific_filter |= combo_mask
            
            # If we found specific matches, use them
            if specific_filter.any():
                result_df = filtered_df[specific_filter]
                
                # For comparison queries, ensure we have at least 2 different trials
                if len(result_df[trial_col].unique()) >= 2:
                    return result_df
                else:
                    # If only one trial found, try to find the comparison trial with broader search
                    remaining_terms = [term for term in specific_trials + drug_combinations 
                                     if not filtered_df[trial_col].str.contains(term, case=False, na=False).any()]
                    
                    for term in remaining_terms:
                        broader_mask = filtered_df[trial_col].str.contains(term, case=False, na=False)
                        if 'Product/Regimen Name' in filtered_df.columns:
                            broader_mask |= filtered_df['Product/Regimen Name'].str.contains(term, case=False, na=False)
                        
                        if broader_mask.any():
                            result_df = pd.concat([result_df, filtered_df[broader_mask]]).drop_duplicates()
                            break
                    
                    return result_df
            
            # If no specific trials found but it's a comparison, limit to top relevant results
            else:
                # For comparison queries without specific trial names, show top results but limit them
                unique_trials = filtered_df[trial_col].unique()
                if len(unique_trials) > 6:  # Limit to 6 different trials for comparison
                    top_trials = unique_trials[:6]
                    trial_filter = filtered_df[trial_col].isin(top_trials)
                    return filtered_df[trial_filter]
                else:
                    return filtered_df
        
        # For non-comparison queries
        else:
            # Look for specific trial mentions even in non-comparison queries
            specific_trials = []
            
            # Extract trial names from query
            trial_patterns = [
                r'checkmate[-\s]?\d+[a-z]*', r'keynote[-\s]?\d+[a-z]*', r'relativity[-\s]?\d+[a-z]*',
                r'dreamseq', r'dream[-\s]?seq', r'combi[-\s][div]', r'columbus',
                r'inspire\d+[a-z]*', r'imspire[-\s]?\d+[a-z]*', r'cobrim', r'brim[-\s]?\d+[a-z]*'
            ]
            
            for pattern in trial_patterns:
                matches = re.findall(pattern, query_lower)
                specific_trials.extend(matches)
            
            if specific_trials:
                # Filter by specific trials mentioned
                specific_filter = pd.Series(False, index=filtered_df.index)
                for trial in specific_trials:
                    trial_mask = filtered_df[trial_col].str.contains(trial, case=False, na=False)
                    specific_filter |= trial_mask
                
                if specific_filter.any():
                    return filtered_df[specific_filter]
            
            # For general queries, limit the number of trials shown
            unique_trials = filtered_df[trial_col].unique()
            if len(unique_trials) > 10:  # Limit to 10 different trials for general queries
                top_trials = unique_trials[:10]
                trial_filter = filtered_df[trial_col].isin(top_trials)
                return filtered_df[trial_filter]
            else:
                return filtered_df
    
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
    """Display bar charts for selected metrics with improved visualization"""
    st.markdown("### 📊 Arm-wise Metrics Breakdown")

    df = df.copy()

    # Ensure unique arms by adding suffix if duplicates exist
    if df[arm_col].duplicated().any():
        df[arm_col] = df[arm_col] + " - arm " + df.groupby(arm_col).cumcount().add(1).astype(str)

    # Melt to long format for visualization
    melted_df = pd.melt(df[[arm_col] + metric_cols], id_vars=arm_col,
                        var_name="Metric", value_name="RawValue")
    melted_df["RawValue"] = melted_df["RawValue"].astype(str).str.strip()

    # Identify missing values
    missing_keywords = {"", "na", "n/a", "nr", "nan", "none", "null", "not reported", "not available"}
    melted_df["IsMissing"] = melted_df["RawValue"].str.lower().isin(missing_keywords)

    # Extract numeric values for plotting
    melted_df["Value"] = (
        melted_df["RawValue"]
        .str.replace('%', '', regex=False)
        .str.replace(r'[^\d\.\-]+', '', regex=True)
    )
    melted_df["Value"] = pd.to_numeric(melted_df["Value"], errors='coerce')
    
    # Set plot values - use small bar for missing data
    melted_df["PlotValue"] = melted_df["Value"]
    melted_df.loc[melted_df["IsMissing"], "PlotValue"] = 0.3  # Small bar for missing values
    melted_df.loc[melted_df["IsMissing"], "TextPosX"] = 2.5  # Fixed text position for missing values
    melted_df.loc[~melted_df["IsMissing"], "TextPosX"] = melted_df["PlotValue"]
    melted_df["Text"] = melted_df["RawValue"].str.upper()

    # Create color mapping for metrics
    color_map = {metric: px.colors.qualitative.Set3[i % 10] for i, metric in enumerate(melted_df["Metric"].unique())}
    melted_df["Color"] = melted_df["Metric"].map(color_map)

    # Create the bar chart
    fig = px.bar(
        melted_df,
        x="PlotValue",
        y=arm_col,
        color="Metric",
        facet_col="Metric",
        facet_col_wrap=5,
        orientation="h",
        text="Text",
        color_discrete_map=color_map,
        title="Clinical Trial Outcomes Comparison"
    )

    # Update traces for better appearance
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12, color="black"),
        cliponaxis=False
    )

    # Update layout
    fig.update_layout(
        height=max(400, 80 * len(df)),  # Dynamic height based on number of trials
        font=dict(size=12),
        showlegend=False,
        margin=dict(l=80, r=30, t=80, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        title_font_size=16
    )

    # Clean up facet titles
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].strip()))
    fig.for_each_xaxis(lambda x: x.update(title='', showticklabels=False))
    fig.update_yaxes(title='', automargin=True)

    # Show warning for missing data
    if melted_df["IsMissing"].any():
        st.warning("⚠️ Some metric values were not reported (shown as 'NR', 'NA', or similar placeholders).")

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
                
                # Re-display visualizations for assistant messages that have viz data
                if message["role"] == "assistant" and "viz_data" in message:
                    viz_data = message["viz_data"]
                    display_visualization_from_state(viz_data)
        
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
                        
                        # Enhanced visualization logic
                        retrieved_data = result.get("retrieved_data", {})
                        viz_data = None  # Initialize viz_data
                        
                        if retrieved_data.get("success") and retrieved_data.get("filtered_df") is not None:
                            original_filtered_df = retrieved_data["filtered_df"]
                            
                            # Apply intelligent filtering for visualization
                            viz_filtered_df = st.session_state.chatbot.identify_specific_trials_for_visualization(
                                prompt, original_filtered_df
                            )
                            
                            # Identify trial ID column
                            trial_id_col = next((col for col in [
                                "Trial Acronym/ID", "Trial Name", "Trial ID", 
                                "Product/Regimen Name", "Product/Regimen"
                            ] if col in viz_filtered_df.columns), None)
                            
                            if trial_id_col and len(viz_filtered_df) > 0:
                                # Show info about filtering
                                if len(viz_filtered_df) != len(original_filtered_df):
                                    st.info(f"📊 Showing visualization for {len(viz_filtered_df)} most relevant trials out of {len(original_filtered_df)} found.")
                                
                                # Get all available metric columns
                                all_metrics = [
                                    "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
                                    "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate",
                                    "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate",
                                    "Gr 3/4 TRAEs", "Gr ≥3 TRAEs", "Gr 3/4 TEAEs", "Gr ≥3 TEAEs", 
                                    "Gr 3/4 irAEs", "Gr ≥3 irAEs",
                                    "Tx-related Deaths (Gr 5 TRAEs)", "All Deaths (Gr 5 AEs)",
                                    "Key AEs"
                                ]
                                
                                # Find which metrics are in the filtered dataframe
                                available_metrics = [m for m in all_metrics if m in viz_filtered_df.columns]
                                
                                if available_metrics:
                                    # Extract mentioned metrics from the query
                                    mentioned_metrics = st.session_state.chatbot.extract_metrics_from_question(
                                        prompt, available_metrics
                                    )
                                    
                                    # For comparison queries, prioritize key efficacy and safety metrics
                                    comparison_indicators = ['vs', 'versus', 'compare', 'comparison', 'against', 'compared to']
                                    is_comparison = any(word in prompt.lower() for word in comparison_indicators)
                                    
                                    if is_comparison and not mentioned_metrics:
                                        # Default comparison metrics - prioritize most common outcomes
                                        priority_metrics = ["ORR", "mPFS", "mOS", "Gr 3/4 TRAEs", "CR", "PR"]
                                        mentioned_metrics = [m for m in priority_metrics if m in available_metrics][:4]
                                    
                                    # Default to mentioned metrics or first few available
                                    default_metrics = mentioned_metrics if mentioned_metrics else available_metrics[:3]
                                    
                                    # Store visualization data for persistence
                                    viz_data = {
                                        "df": viz_filtered_df,
                                        "trial_id_col": trial_id_col,
                                        "available_metrics": available_metrics,
                                        "default_metrics": default_metrics,
                                        "prompt": prompt,
                                        "message_index": len(st.session_state.messages)
                                    }
                                    
                                    # Create visualization
                                    create_persistent_visualization(viz_data)
                                
                                else:
                                    st.info("No visualization metrics available in the filtered data.")
                            
                            else:
                                st.info("No trials found for visualization.")
                    
                    else:
                        st.error("No response generated.")
                    
                    # Add to chat history with viz data if available
                    message_data = {
                        "role": "assistant", 
                        "content": result.get("output", "No response") if result else "Error"
                    }
                    if viz_data:
                        message_data["viz_data"] = viz_data
                    
                    st.session_state.messages.append(message_data)
                    
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


def create_persistent_visualization(viz_data):
    """Create visualization with persistent state"""
    df = viz_data["df"]
    trial_id_col = viz_data["trial_id_col"]
    available_metrics = viz_data["available_metrics"]
    default_metrics = viz_data["default_metrics"]
    message_index = viz_data["message_index"]
    
    # Create unique key for this visualization
    viz_key = f"viz_{message_index}"
    
    # Organize metrics by category for better UX
    efficacy_metrics = [m for m in available_metrics if any(keyword in m.lower() 
                      for keyword in ['orr', 'cr', 'pr', 'pfs', 'os', 'dcr', 'dor', 'rate'])]
    safety_metrics = [m for m in available_metrics if any(keyword in m.lower() 
                    for keyword in ['traes', 'teaes', 'iraes', 'aes', 'deaths', 'adverse'])]
    
    st.markdown("#### 📊 Select Metrics to Visualize")
    
    col1, col2 = st.columns(2)
    with col1:
        if efficacy_metrics:
            st.write("**Efficacy Metrics:**")
            for metric in efficacy_metrics[:5]:  # Show first 5
                st.write(f"• {metric}")
    
    with col2:
        if safety_metrics:
            st.write("**Safety Metrics:**")
            for metric in safety_metrics[:5]:  # Show first 5
                st.write(f"• {metric}")
    
    # Initialize session state for this visualization if not exists
    state_key = f"selected_metrics_{viz_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_metrics
    
    selected_metrics = st.multiselect(
        "Choose metrics to compare:",
        options=available_metrics,
        default=st.session_state[state_key],
        key=f"{viz_key}_metrics",
        help="Select the clinical outcomes you want to compare between trials"
    )
    
    # Update session state
    st.session_state[state_key] = selected_metrics
    
    # Display charts if metrics selected
    if selected_metrics:
        display_bar_charts(
            df,
            trial_id_col,
            selected_metrics,
            base_key=viz_key
        )
        
        # Show summary table for selected trials and metrics
        st.markdown("#### 📋 Summary Table")
        summary_cols = [trial_id_col] + selected_metrics
        summary_df = df[summary_cols].copy()
        
        # Clean up the display
        for col in selected_metrics:
            summary_df[col] = summary_df[col].astype(str).replace('nan', 'Not Available')
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("Please select at least one metric to visualize the comparison.")


def display_visualization_from_state(viz_data):
    """Re-display visualization from stored state data"""
    if viz_data:
        create_persistent_visualization(viz_data)


if __name__ == "__main__":
    main()