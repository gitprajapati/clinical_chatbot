# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import faiss
# # # import openai
# # # import pickle
# # # import os
# # # from typing import List, Dict, Tuple
# # # import json
# # # import time

# # # # Page configuration
# # # st.set_page_config(
# # #     page_title="Clinical Dataset Search",
# # #     page_icon="🔬",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )

# # # # Custom CSS for better UI
# # # st.markdown("""
# # # <style>
# # #     .main-header {
# # #         font-size: 2.5rem;
# # #         color: #1f77b4;
# # #         text-align: center;
# # #         margin-bottom: 2rem;
# # #     }
# # #     .search-section {
# # #         background-color: #f0f2f6;
# # #         padding: 2rem;
# # #         border-radius: 10px;
# # #         margin-bottom: 2rem;
# # #     }
# # #     .result-card {
# # #         background-color: white;
# # #         padding: 1.5rem;
# # #         border-radius: 8px;
# # #         margin-bottom: 1rem;
# # #         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
# # #     }
# # #     .metric-value {
# # #         font-size: 1.2rem;
# # #         font-weight: bold;
# # #         color: #1f77b4;
# # #     }
# # #     .stProgress > div > div > div > div {
# # #         background-color: #1f77b4;
# # #     }
# # # </style>
# # # """, unsafe_allow_html=True)

# # # # Initialize session state
# # # if 'embeddings_created' not in st.session_state:
# # #     st.session_state.embeddings_created = False
# # # if 'index' not in st.session_state:
# # #     st.session_state.index = None
# # # if 'df' not in st.session_state:
# # #     st.session_state.df = None
# # # if 'embeddings' not in st.session_state:
# # #     st.session_state.embeddings = None

# # # class ClinicalDatasetEmbeddings:
# # #     def __init__(self, api_key: str):
# # #         """Initialize the embeddings manager with OpenAI API key."""
# # #         self.api_key = api_key
# # #         openai.api_key = api_key
# # #         self.embedding_model = "text-embedding-3-large"
# # #         self.embedding_dim = 3072  # dimension for text-embedding-3-large
        
# # #     def load_data(self, file_path: str) -> pd.DataFrame:
# # #         """Load the Excel file and return as DataFrame."""
# # #         df = pd.read_excel(file_path, sheet_name='First-line Outcomes')
# # #         return df
    
# # #     def create_text_representation(self, row: pd.Series) -> str:
# # #         """Create a comprehensive text representation of a row for embedding."""
# # #         # Key fields for comprehensive representation
# # #         key_fields = {
# # #             'Product/Regimen Name': 'Product',
# # #             'Comparator': 'Comparator',
# # #             'Trial Acronym/ID': 'Trial ID',
# # #             'Regimen MoAs': 'Mechanism of Action',
# # #             'Product/Regimen Target': 'Target',
# # #             'Description': 'Description',
# # #             'Active Developers (Companies Names)': 'Developer',
# # #             'Therapeutic Area': 'Therapeutic Area',
# # #             'Therapeutic Indication': 'Indication',
# # #             'Additional TA Details': 'Additional Details',
# # #             'Highest Phase': 'Phase',
# # #             'Biomarkers': 'Biomarkers',
# # #             'ORR': 'Overall Response Rate',
# # #             'CR': 'Complete Response',
# # #             'PR': 'Partial Response',
# # #             'mPFS': 'Median PFS',
# # #             'mOS': 'Median OS',
# # #             'Key AEs': 'Key Adverse Events',
# # #             'Comments (Safety)': 'Safety Comments'
# # #         }
        
# # #         text_parts = []
        
# # #         for field, label in key_fields.items():
# # #             if field in row and pd.notna(row[field]) and str(row[field]).strip() != 'NA':
# # #                 value = str(row[field]).strip()
# # #                 text_parts.append(f"{label}: {value}")
        
# # #         # Add additional numerical data if available
# # #         numerical_fields = ['N (No of All Enrolled pts)', 'n (Enrolled pts in arm)', 
# # #                           'Median Follow-up (mFU)', 'DCR', 'mDoR']
        
# # #         for field in numerical_fields:
# # #             if field in row and pd.notna(row[field]) and str(row[field]).strip() != 'NA':
# # #                 text_parts.append(f"{field}: {row[field]}")
        
# # #         # Combine all parts into a comprehensive text
# # #         full_text = " | ".join(text_parts)
# # #         return full_text
    
# # #     def generate_embedding(self, text: str) -> List[float]:
# # #         """Generate embedding for a single text using OpenAI API."""
# # #         try:
# # #             response = openai.embeddings.create(
# # #                 input=text,
# # #                 model=self.embedding_model
# # #             )
# # #             return response.data[0].embedding
# # #         except Exception as e:
# # #             st.error(f"Error generating embedding: {str(e)}")
# # #             return None
    
# # #     def create_embeddings_for_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
# # #         """Create embeddings for all rows in the dataset."""
# # #         embeddings = []
# # #         texts = []
        
# # #         progress_bar = st.progress(0)
# # #         status_text = st.empty()
        
# # #         for idx, row in df.iterrows():
# # #             # Create text representation
# # #             text = self.create_text_representation(row)
# # #             texts.append(text)
            
# # #             # Generate embedding
# # #             status_text.text(f"Processing row {idx + 1}/{len(df)}: {row.get('Product/Regimen Name', 'N/A')[:50]}...")
# # #             embedding = self.generate_embedding(text)
            
# # #             if embedding:
# # #                 embeddings.append(embedding)
# # #             else:
# # #                 # Use zero vector if embedding fails
# # #                 embeddings.append([0.0] * self.embedding_dim)
            
# # #             progress_bar.progress((idx + 1) / len(df))
            
# # #             # Add small delay to avoid rate limiting
# # #             time.sleep(0.1)
        
# # #         progress_bar.empty()
# # #         status_text.empty()
        
# # #         return np.array(embeddings, dtype='float32'), texts
    
# # #     def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
# # #         """Create and return a FAISS index from embeddings."""
# # #         # Normalize embeddings for cosine similarity
# # #         faiss.normalize_L2(embeddings)
        
# # #         # Create index
# # #         index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
# # #         index.add(embeddings)
        
# # #         return index
    
# # #     def save_index_and_data(self, index: faiss.Index, df: pd.DataFrame, embeddings: np.ndarray):
# # #         """Save FAISS index and associated data."""
# # #         # Save FAISS index
# # #         faiss.write_index(index, "clinical_data_index.faiss")
        
# # #         # Save DataFrame
# # #         df.to_pickle("clinical_data_df.pkl")
        
# # #         # Save embeddings
# # #         np.save("clinical_data_embeddings.npy", embeddings)
        
# # #     def load_index_and_data(self) -> Tuple[faiss.Index, pd.DataFrame, np.ndarray]:
# # #         """Load saved FAISS index and associated data."""
# # #         if os.path.exists("clinical_data_index.faiss"):
# # #             index = faiss.read_index("clinical_data_index.faiss")
# # #             df = pd.read_pickle("clinical_data_df.pkl")
# # #             embeddings = np.load("clinical_data_embeddings.npy")
# # #             return index, df, embeddings
# # #         return None, None, None
    
# # #     def search(self, query: str, index: faiss.Index, df: pd.DataFrame, k: int = 5) -> List[Dict]:
# # #         """Search for similar entries based on query."""
# # #         # Generate embedding for query
# # #         query_embedding = self.generate_embedding(query)
# # #         if not query_embedding:
# # #             return []
        
# # #         # Normalize query embedding
# # #         query_vec = np.array([query_embedding], dtype='float32')
# # #         faiss.normalize_L2(query_vec)
        
# # #         # Search
# # #         distances, indices = index.search(query_vec, k)
        
# # #         # Prepare results
# # #         results = []
# # #         for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
# # #             if idx != -1:  # Valid result
# # #                 row = df.iloc[idx]
# # #                 result = {
# # #                     'rank': i + 1,
# # #                     'similarity_score': float(dist),
# # #                     'data': row.to_dict()
# # #                 }
# # #                 results.append(result)
        
# # #         return results

# # # def display_search_results(results: List[Dict]):
# # #     """Display search results in a clean format."""
# # #     if not results:
# # #         st.warning("No results found. Try a different query.")
# # #         return
    
# # #     for result in results:
# # #         with st.container():
# # #             col1, col2 = st.columns([3, 1])
            
# # #             with col1:
# # #                 st.markdown(f"### {result['data'].get('Product/Regimen Name', 'N/A')}")
                
# # #                 # Display key information
# # #                 st.markdown(f"**Trial ID:** {result['data'].get('Trial Acronym/ID', 'N/A')}")
# # #                 st.markdown(f"**Therapeutic Area:** {result['data'].get('Therapeutic Area', 'N/A')}")
# # #                 st.markdown(f"**Phase:** {result['data'].get('Highest Phase', 'N/A')}")
                
# # #                 # Efficacy metrics
# # #                 col_eff1, col_eff2, col_eff3, col_eff4 = st.columns(4)
# # #                 with col_eff1:
# # #                     st.metric("ORR", result['data'].get('ORR', 'N/A'))
# # #                 with col_eff2:
# # #                     st.metric("CR", result['data'].get('CR', 'N/A'))
# # #                 with col_eff3:
# # #                     st.metric("mPFS", result['data'].get('mPFS', 'N/A'))
# # #                 with col_eff4:
# # #                     st.metric("mOS", result['data'].get('mOS', 'N/A'))
                
# # #                 # Additional details in expander
# # #                 with st.expander("View Full Details"):
# # #                     # Group related fields
# # #                     st.markdown("#### Trial Information")
# # #                     st.write(f"- **Comparator:** {result['data'].get('Comparator', 'N/A')}")
# # #                     st.write(f"- **Indication:** {result['data'].get('Therapeutic Indication', 'N/A')}")
# # #                     st.write(f"- **Biomarkers:** {result['data'].get('Biomarkers', 'N/A')}")
# # #                     st.write(f"- **Enrolled Patients:** {result['data'].get('n (Enrolled pts in arm)', 'N/A')}")
                    
# # #                     st.markdown("#### Mechanism of Action")
# # #                     st.write(f"- **MoAs:** {result['data'].get('Regimen MoAs', 'N/A')}")
# # #                     st.write(f"- **Target:** {result['data'].get('Product/Regimen Target', 'N/A')}")
                    
# # #                     st.markdown("#### Safety Information")
# # #                     st.write(f"- **Key AEs:** {result['data'].get('Key AEs', 'N/A')}")
# # #                     st.write(f"- **Gr ≥3 TRAEs:** {result['data'].get('Gr ≥3 TRAEs', 'N/A')}")
                    
# # #                     st.markdown("#### References")
# # #                     if result['data'].get('Trial URL') and result['data']['Trial URL'] != 'NA':
# # #                         st.write(f"- [Clinical Trial Link]({result['data']['Trial URL']})")
# # #                     if result['data'].get('References') and result['data']['References'] != 'NA':
# # #                         st.write(f"- [Additional References]({result['data']['References']})")
            
# # #             with col2:
# # #                 st.metric("Similarity Score", f"{result['similarity_score']:.3f}")
# # #                 st.metric("Rank", f"#{result['rank']}")
            
# # #             st.divider()

# # # def main():
# # #     st.markdown('<h1 class="main-header">🔬 Clinical Dataset Semantic Search</h1>', unsafe_allow_html=True)
    
# # #     # Sidebar for configuration
# # #     with st.sidebar:
# # #         st.header("⚙️ Configuration")
        
# # #         # API Key input
# # #         api_key = st.text_input("OpenAI API Key", type="password", 
# # #                                help="Enter your OpenAI API key for text-embedding-3-large model")
        
# # #         st.divider()
        
# # #         # Data processing section
# # #         st.header("📊 Data Processing")
        
# # #         if api_key:
# # #             embeddings_manager = ClinicalDatasetEmbeddings(api_key)
            
# # #             # Check if embeddings already exist
# # #             if os.path.exists("clinical_data_index.faiss") and not st.session_state.embeddings_created:
# # #                 if st.button("Load Existing Embeddings", type="primary"):
# # #                     with st.spinner("Loading embeddings..."):
# # #                         index, df, embeddings = embeddings_manager.load_index_and_data()
# # #                         if index is not None:
# # #                             st.session_state.index = index
# # #                             st.session_state.df = df
# # #                             st.session_state.embeddings = embeddings
# # #                             st.session_state.embeddings_created = True
# # #                             st.success("✅ Embeddings loaded successfully!")
            
# # #             if st.button("Create New Embeddings", type="secondary"):
# # #                 with st.spinner("Processing dataset..."):
# # #                     # Load data
# # #                     df = embeddings_manager.load_data("Sample Data.xlsx")
# # #                     st.session_state.df = df
                    
# # #                     # Create embeddings
# # #                     st.info("Creating embeddings for each row...")
# # #                     embeddings, texts = embeddings_manager.create_embeddings_for_dataset(df)
# # #                     st.session_state.embeddings = embeddings
                    
# # #                     # Create FAISS index
# # #                     st.info("Building FAISS index...")
# # #                     index = embeddings_manager.create_faiss_index(embeddings)
# # #                     st.session_state.index = index
                    
# # #                     # Save everything
# # #                     st.info("Saving index and data...")
# # #                     embeddings_manager.save_index_and_data(index, df, embeddings)
                    
# # #                     st.session_state.embeddings_created = True
# # #                     st.success("✅ Embeddings created and saved successfully!")
# # #         else:
# # #             st.warning("Please enter your OpenAI API key to proceed.")
        
# # #         # Display dataset info if loaded
# # #         if st.session_state.df is not None:
# # #             st.divider()
# # #             st.header("📈 Dataset Info")
# # #             st.metric("Total Records", len(st.session_state.df))
# # #             st.metric("Total Columns", len(st.session_state.df.columns))
            
# # #             # Show unique values for key fields
# # #             if 'Therapeutic Area' in st.session_state.df.columns:
# # #                 unique_areas = st.session_state.df['Therapeutic Area'].nunique()
# # #                 st.metric("Therapeutic Areas", unique_areas)
    
# # #     # Main search interface
# # #     if st.session_state.embeddings_created and api_key:
# # #         # Search section
# # #         st.markdown('<div class="search-section">', unsafe_allow_html=True)
# # #         st.header("🔍 Search Clinical Trials")
        
# # #         # Search input
# # #         col1, col2 = st.columns([4, 1])
        
# # #         with col1:
# # #             query = st.text_input("Enter your search query", 
# # #                                 placeholder="e.g., melanoma immunotherapy with high response rate",
# # #                                 help="Search for trials by indication, treatment, mechanism, outcomes, etc.")
        
# # #         with col2:
# # #             num_results = st.number_input("Results", min_value=1, max_value=20, value=5)
        
# # #         # Example queries
# # #         st.markdown("**Example queries:**")
# # #         example_cols = st.columns(3)
# # #         with example_cols[0]:
# # #             if st.button("BRAF inhibitors"):
# # #                 query = "BRAF inhibitors melanoma"
# # #         with example_cols[1]:
# # #             if st.button("High ORR immunotherapy"):
# # #                 query = "immunotherapy high overall response rate"
# # #         with example_cols[2]:
# # #             if st.button("Phase 3 trials"):
# # #                 query = "phase 3 registrational trials"
        
# # #         st.markdown('</div>', unsafe_allow_html=True)
        
# # #         # Search execution
# # #         if query:
# # #             embeddings_manager = ClinicalDatasetEmbeddings(api_key)
            
# # #             with st.spinner("Searching..."):
# # #                 results = embeddings_manager.search(
# # #                     query, 
# # #                     st.session_state.index, 
# # #                     st.session_state.df, 
# # #                     k=num_results
# # #                 )
            
# # #             # Display results
# # #             st.header(f"Search Results for: '{query}'")
# # #             display_search_results(results)
    
# # #     elif not api_key:
# # #         st.info("👈 Please enter your OpenAI API key in the sidebar to get started.")
# # #     else:
# # #         st.info("👈 Please create or load embeddings from the sidebar to enable search.")

# # # if __name__ == "__main__":
# # #     main()



# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import faiss
# # import openai
# # import pickle
# # import os
# # from typing import List, Dict, Tuple, Any, Optional
# # import json
# # import plotly.express as px
# # from langchain_openai import ChatOpenAI
# # from dotenv import load_dotenv
# # import re

# # # Load environment variables
# # load_dotenv()

# # # Page configuration
# # st.set_page_config(
# #     page_title="Clinical Trial AI Assistant",
# #     page_icon="🔬",
# #     layout="wide"
# # )

# # # Custom CSS for better UI
# # st.markdown("""
# # <style>
# #     .main-header {
# #         font-size: 2.5rem;
# #         color: #1f77b4;
# #         text-align: center;
# #         margin-bottom: 2rem;
# #     }
# #     .chat-message {
# #         padding: 1.5rem;
# #         border-radius: 10px;
# #         margin-bottom: 1rem;
# #     }
# #     .user-message {
# #         background-color: #e6f3ff;
# #         margin-left: 20%;
# #     }
# #     .assistant-message {
# #         background-color: #f0f2f6;
# #         margin-right: 20%;
# #     }
# #     .metric-card {
# #         background-color: white;
# #         padding: 1rem;
# #         border-radius: 8px;
# #         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
# #         text-align: center;
# #     }
# #     .stProgress > div > div > div > div {
# #         background-color: #1f77b4;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # class ClinicalChatWithFAISS:
# #     def __init__(self):
# #         """Initialize the chat system with pre-built FAISS index and embeddings"""
# #         self.api_key = os.getenv("OPENAI_API_KEY")
# #         if not self.api_key:
# #             raise ValueError("OPENAI_API_KEY not found in environment variables")
        
# #         openai.api_key = self.api_key
# #         self.embedding_model = "text-embedding-3-large"
# #         self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
# #         # Load pre-built components
# #         self.index, self.df, self.embeddings = self._load_saved_components()
# #         self._build_comprehensive_mappings()
        
# #     def _load_saved_components(self) -> Tuple[faiss.Index, pd.DataFrame, np.ndarray]:
# #         """Load saved FAISS index, dataframe, and embeddings"""
# #         try:
# #             index = faiss.read_index("clinical_data_index.faiss")
# #             df = pd.read_pickle("clinical_data_df.pkl")
# #             embeddings = np.load("clinical_data_embeddings.npy")
# #             return index, df, embeddings
# #         except Exception as e:
# #             st.error(f"Error loading saved components: {str(e)}")
# #             raise
    
# #     def _build_comprehensive_mappings(self):
# #         """Build comprehensive alias mappings from the pasted code"""
# #         # Trial acronym aliases
# #         self.trial_aliases = {
# #             'checkmate-511': ['cm-511', 'cm511', 'cm 511', 'checkmate511', 'checkmate 511'],
# #             'checkmate-067': ['cm-067', 'cm067', 'cm 067', 'checkmate067', 'checkmate 067'],
# #             'checkmate-066': ['cm-066', 'cm066', 'cm 066', 'checkmate066', 'checkmate 066'],
# #             'checkmate-238': ['cm-238', 'cm238', 'cm 238', 'checkmate238', 'checkmate 238'],
# #             'keynote-006': ['kn-006', 'kn006', 'kn 006', 'keynote006', 'keynote 006'],
# #             'keynote-252': ['kn-252', 'kn252', 'kn 252', 'keynote252', 'keynote 252'],
# #             'dreamseq': ['dream-seq', 'dream seq', 'dream sequence', 'dreamsequence'],
# #             'combi-d': ['combid', 'combi d', 'combination d'],
# #             'columbus': ['col', 'columbus trial', 'columbus study'],
# #             'cobrim': ['co-brim', 'co brim', 'cobrim trial'],
# #         }
        
# #         # Phase aliases
# #         self.phase_aliases = {
# #             'ph3': ['phase 3', 'phase iii', 'phase3', 'phase-3', 'p3'],
# #             'ph2': ['phase 2', 'phase ii', 'phase2', 'phase-2', 'p2'],
# #             'ph1': ['phase 1', 'phase i', 'phase1', 'phase-1', 'p1'],
# #         }
        
# #         # Company aliases
# #         self.company_aliases = {
# #             'bms': ['bristol myers squibb', 'bristol-myers squibb', 'bristol myers'],
# #             'merck': ['merck/mds', 'merck us', 'us-based merck', 'msd'],
# #             'roche': ['genentech', 'roche genentech'],
# #             'novartis': ['novartis pharma', 'novartis ag'],
# #             'pfizer': ['pfizer inc', 'pfizer inc.'],
# #         }
        
# #         # Drug aliases
# #         self.drug_aliases = {
# #             'nivolumab': ['nivo', 'opdivo', 'bms-936558'],
# #             'ipilimumab': ['ipi', 'yervoy', 'mdx-010'],
# #             'pembrolizumab': ['pembro', 'keytruda', 'mk-3475'],
# #             'vemurafenib': ['vemu', 'zelboraf', 'plx4032'],
# #             'dabrafenib': ['dabra', 'tafinlar'],
# #             'trametinib': ['trame', 'mekinist'],
# #         }
        
# #         # Outcome aliases
# #         self.outcome_aliases = {
# #             'orr': ['overall response rate', 'response rate', 'objective response rate'],
# #             'cr': ['complete response', 'complete response rate'],
# #             'pr': ['partial response', 'partial response rate'],
# #             'pfs': ['progression free survival', 'progression-free survival', 'median pfs'],
# #             'os': ['overall survival', 'median os', 'mos'],
# #             'aes': ['adverse events', 'safety', 'toxicity'],
# #             'traes': ['treatment related adverse events', 'treatment-related adverse events'],
# #         }
        
# #         # Text columns for search
# #         self.text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
# #         self.numeric_columns = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
    
# #     def _expand_query_with_aliases(self, query: str) -> str:
# #         """Expand query using alias mappings"""
# #         expanded_terms = []
# #         query_lower = query.lower()
        
# #         # Check all alias dictionaries
# #         all_aliases = [
# #             self.trial_aliases, self.phase_aliases, 
# #             self.company_aliases, self.drug_aliases, self.outcome_aliases
# #         ]
        
# #         for alias_dict in all_aliases:
# #             for canonical, aliases in alias_dict.items():
# #                 if canonical in query_lower:
# #                     expanded_terms.extend(aliases)
# #                 for alias in aliases:
# #                     if alias in query_lower:
# #                         expanded_terms.append(canonical)
        
# #         # Combine original query with expanded terms
# #         if expanded_terms:
# #             expanded_query = query + " " + " ".join(expanded_terms)
# #             return expanded_query
# #         return query
    
# #     def generate_embedding(self, text: str) -> List[float]:
# #         """Generate embedding for a text using OpenAI API"""
# #         try:
# #             response = openai.embeddings.create(
# #                 input=text,
# #                 model=self.embedding_model
# #             )
# #             return response.data[0].embedding
# #         except Exception as e:
# #             st.error(f"Error generating embedding: {str(e)}")
# #             return None
    
# #     def semantic_search(self, query: str, k: int = 10) -> Tuple[pd.DataFrame, List[float]]:
# #         """Perform semantic search using FAISS"""
# #         # Expand query with aliases
# #         expanded_query = self._expand_query_with_aliases(query)
        
# #         # Generate embedding
# #         query_embedding = self.generate_embedding(expanded_query)
# #         if not query_embedding:
# #             return pd.DataFrame(), []
        
# #         # Normalize for cosine similarity
# #         query_vec = np.array([query_embedding], dtype='float32')
# #         faiss.normalize_L2(query_vec)
        
# #         # Search
# #         distances, indices = self.index.search(query_vec, k)
        
# #         # Get results
# #         valid_indices = indices[0][indices[0] != -1]
# #         valid_distances = distances[0][:len(valid_indices)]
        
# #         if len(valid_indices) > 0:
# #             result_df = self.df.iloc[valid_indices].copy()
# #             result_df['similarity_score'] = valid_distances
# #             return result_df, valid_distances.tolist()
        
# #         return pd.DataFrame(), []
    
# #     def hybrid_search(self, query: str, k: int = 20) -> pd.DataFrame:
# #         """Combine semantic search with keyword filtering"""
# #         # First, get semantic search results
# #         semantic_results, scores = self.semantic_search(query, k=k)
        
# #         if semantic_results.empty:
# #             return pd.DataFrame()
        
# #         # Extract specific terms from query for additional filtering
# #         query_lower = query.lower()
        
# #         # Look for specific trial names, drugs, etc.
# #         specific_filters = []
        
# #         # Check for trial names
# #         trial_patterns = [
# #             r'checkmate[-\s]?\d+', r'keynote[-\s]?\d+', r'dreamseq',
# #             r'combi[-\s][div]', r'columbus', r'cobrim'
# #         ]
        
# #         for pattern in trial_patterns:
# #             matches = re.findall(pattern, query_lower)
# #             specific_filters.extend(matches)
        
# #         # If we have specific filters, apply them
# #         if specific_filters and 'Trial Acronym/ID' in semantic_results.columns:
# #             filter_mask = pd.Series(False, index=semantic_results.index)
# #             for term in specific_filters:
# #                 mask = semantic_results['Trial Acronym/ID'].str.contains(term, case=False, na=False)
# #                 filter_mask |= mask
            
# #             if filter_mask.any():
# #                 # Prioritize exact matches
# #                 exact_matches = semantic_results[filter_mask]
# #                 other_matches = semantic_results[~filter_mask]
# #                 semantic_results = pd.concat([exact_matches, other_matches])
        
# #         return semantic_results
    
# #     def generate_response(self, query: str, search_results: pd.DataFrame) -> str:
# #         """Generate response using LLM based on search results"""
# #         if search_results.empty:
# #             return "I couldn't find any relevant clinical trials matching your query. Please try different search terms."
        
# #         # Prepare context from search results
# #         context_data = search_results.head(10).to_dict('records')
        
# #         # Identify if this is a comparison query
# #         is_comparison = any(word in query.lower() for word in ['vs', 'versus', 'compare', 'comparison'])
        
# #         # Create prompt
# #         prompt = f"""You are a clinical trial expert assistant. Answer the following question based on the provided clinical trial data.

# # Question: {query}

# # Available Data:
# # {json.dumps(context_data, indent=2)}

# # Instructions:
# # 1. Provide a clear, accurate answer using ONLY the provided data
# # 2. If comparing trials, create a structured comparison table
# # 3. Include specific metrics (ORR, PFS, OS, etc.) when relevant
# # 4. State "Not available" for any missing data
# # 5. Be concise but thorough
# # 6. Use markdown formatting for better readability

# # Answer:"""

# #         try:
# #             from langchain.schema import HumanMessage
# #             response = self.llm.invoke([HumanMessage(content=prompt)])
# #             return response.content
# #         except Exception as e:
# #             return self._create_fallback_response(query, search_results)
    
# #     def _create_fallback_response(self, query: str, results: pd.DataFrame) -> str:
# #         """Create a structured response if LLM fails"""
# #         response = f"## Results for: {query}\n\n"
# #         response += f"Found {len(results)} relevant clinical trials:\n\n"
        
# #         # Key columns to display
# #         key_cols = ['Product/Regimen Name', 'Trial Acronym/ID', 'Therapeutic Area', 
# #                    'ORR', 'mPFS', 'mOS', 'Highest Phase']
# #         available_cols = [col for col in key_cols if col in results.columns]
        
# #         if available_cols:
# #             # Create a simple table
# #             response += "| " + " | ".join(available_cols) + " |\n"
# #             response += "| " + " | ".join(["---"] * len(available_cols)) + " |\n"
            
# #             for _, row in results.head(5).iterrows():
# #                 row_values = []
# #                 for col in available_cols:
# #                     value = str(row.get(col, "N/A"))[:30]
# #                     row_values.append(value)
# #                 response += "| " + " | ".join(row_values) + " |\n"
        
# #         return response
    
# #     def identify_metrics_for_visualization(self, query: str, df: pd.DataFrame) -> List[str]:
# #         """Identify which metrics to visualize based on the query"""
# #         query_lower = query.lower()
        
# #         # All possible metric columns
# #         all_metrics = [
# #             "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
# #             "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate",
# #             "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate",
# #             "Gr 3/4 TRAEs", "Gr ≥3 TRAEs", "Gr 3/4 TEAEs", "Gr ≥3 TEAEs"
# #         ]
        
# #         # Filter to available metrics
# #         available_metrics = [m for m in all_metrics if m in df.columns]
        
# #         # Extract mentioned metrics
# #         mentioned_metrics = []
        
# #         # Check for specific metric mentions
# #         metric_keywords = {
# #             'orr': ['ORR'],
# #             'response': ['ORR', 'CR', 'PR'],
# #             'complete response': ['CR'],
# #             'partial response': ['PR'],
# #             'pfs': ['mPFS', '1-yr PFS Rate', '2-yr PFS Rate'],
# #             'progression': ['mPFS'],
# #             'os': ['mOS', '1-yr OS Rate', '2-yr OS Rate'],
# #             'survival': ['mOS', 'mPFS'],
# #             'safety': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs'],
# #             'adverse': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs'],
# #             'efficacy': ['ORR', 'mPFS', 'mOS']
# #         }
        
# #         for keyword, metrics in metric_keywords.items():
# #             if keyword in query_lower:
# #                 for metric in metrics:
# #                     if metric in available_metrics:
# #                         mentioned_metrics.append(metric)
        
# #         # Remove duplicates while preserving order
# #         mentioned_metrics = list(dict.fromkeys(mentioned_metrics))
        
# #         # Default metrics if none mentioned
# #         if not mentioned_metrics:
# #             default_metrics = ['ORR', 'mPFS', 'mOS', 'Gr 3/4 TRAEs']
# #             mentioned_metrics = [m for m in default_metrics if m in available_metrics][:4]
        
# #         return mentioned_metrics[:6]  # Limit to 6 metrics max
    
# #     def extract_trials_from_response(self, response: str, search_results: pd.DataFrame) -> pd.DataFrame:
# #         """Extract trials for visualization using multiple fallback strategies"""
# #         if search_results.empty:
# #             return search_results
        
# #         response_lower = response.lower()
        
# #         # Remove duplicates first to work with clean data
# #         if 'Trial Acronym/ID' in search_results.columns:
# #             unique_results = search_results.drop_duplicates(subset=['Trial Acronym/ID'], keep='first')
# #         else:
# #             unique_results = search_results.drop_duplicates(keep='first')
        
# #         # Strategy 1: Look for explicit trial names in response
# #         trial_names = set()
        
# #         # Extract trial names from various formats
# #         trial_patterns = [
# #             r'checkmate[-\s]?(\d+)',
# #             r'keynote[-\s]?(\d+)', 
# #             r'(nct\d+)',
# #             r'(dreamseq)',
# #             r'(columbus)',
# #             r'(cobrim)'
# #         ]
        
# #         for pattern in trial_patterns:
# #             matches = re.findall(pattern, response_lower, re.IGNORECASE)
# #             for match in matches:
# #                 if match:
# #                     trial_names.add(match.lower())
        
# #         # Strategy 2: If explicit trials found, filter by them
# #         if trial_names:
# #             filtered_results = []
# #             for idx, row in unique_results.iterrows():
# #                 trial_id = str(row.get('Trial Acronym/ID', '')).lower()
                
# #                 for mentioned_trial in trial_names:
# #                     if mentioned_trial in trial_id or trial_id.endswith(mentioned_trial):
# #                         filtered_results.append(idx)
# #                         break
            
# #             if filtered_results:
# #                 return unique_results.loc[filtered_results]
        
# #         # Strategy 3: Drug-based filtering when no trial names found
# #         drug_keywords = {
# #             'nivolumab': ['nivolumab', 'opdivo', 'nivo'],
# #             'ipilimumab': ['ipilimumab', 'yervoy', 'ipi'],
# #             'pembrolizumab': ['pembrolizumab', 'keytruda', 'pembro'],
# #             'dabrafenib': ['dabrafenib', 'tafinlar'],
# #             'trametinib': ['trametinib', 'mekinist'],
# #             'vemurafenib': ['vemurafenib', 'zelboraf']
# #         }
        
# #         mentioned_drugs = []
# #         for drug, aliases in drug_keywords.items():
# #             if any(alias in response_lower for alias in aliases):
# #                 mentioned_drugs.append(drug)
        
# #         # Strategy 4: Filter by drug combinations
# #         if mentioned_drugs:
# #             drug_filtered = []
# #             for idx, row in unique_results.iterrows():
# #                 product_name = str(row.get('Product/Regimen Name', '')).lower()
                
# #                 # Check if product contains mentioned drugs
# #                 drug_match_count = sum(1 for drug in mentioned_drugs if drug in product_name)
                
# #                 if drug_match_count > 0:
# #                     drug_filtered.append((idx, drug_match_count))
            
# #             # Sort by drug match count (higher is better)
# #             if drug_filtered:
# #                 drug_filtered.sort(key=lambda x: x[1], reverse=True)
# #                 filtered_indices = [idx for idx, count in drug_filtered]
# #                 result = unique_results.loc[filtered_indices]
                
# #                 # For comparison queries, try to get contrasting regimens
# #                 if any(word in response_lower for word in ['compare', 'comparison', 'vs', 'versus']):
# #                     return result.head(2)
# #                 else:
# #                     return result.head(3)
        
# #         # Strategy 5: Response content-based filtering
# #         response_keywords = [
# #             'monotherapy', 'combination', 'plus', '+', 'checkpoint inhibitor',
# #             'targeted therapy', 'immunotherapy', 'melanoma', 'nsclc', 'rcc'
# #         ]
        
# #         keyword_filtered = []
# #         for idx, row in unique_results.iterrows():
# #             score = 0
# #             product_name = str(row.get('Product/Regimen Name', '')).lower()
# #             therapeutic_area = str(row.get('Therapeutic Area', '')).lower()
            
# #             # Score based on keywords in response
# #             for keyword in response_keywords:
# #                 if keyword in response_lower:
# #                     if keyword in product_name:
# #                         score += 2
# #                     if keyword in therapeutic_area:
# #                         score += 1
            
# #             if score > 0:
# #                 keyword_filtered.append((idx, score))
        
# #         if keyword_filtered:
# #             keyword_filtered.sort(key=lambda x: x[1], reverse=True)
# #             filtered_indices = [idx for idx, score in keyword_filtered]
# #             result = unique_results.loc[filtered_indices]
            
# #             if any(word in response_lower for word in ['compare', 'comparison', 'vs', 'versus']):
# #                 return result.head(2)
# #             else:
# #                 return result.head(3)
        
# #         # Strategy 6: Fallback - use similarity scores if available
# #         if 'similarity_score' in unique_results.columns:
# #             sorted_results = unique_results.sort_values('similarity_score', ascending=False)
# #         else:
# #             sorted_results = unique_results
        
# #         # Final fallback based on query type
# #         if any(word in response_lower for word in ['compare', 'comparison', 'vs', 'versus']):
# #             return sorted_results.head(2)
# #         elif any(word in response_lower for word in ['show', 'find', 'list', 'trials']):
# #             return sorted_results.head(3)
# #         else:
# #             return sorted_results.head(2)
    
# #     def smart_trial_selection(self, query: str, search_results: pd.DataFrame) -> pd.DataFrame:
# #         """Additional smart selection based on query context"""
# #         if search_results.empty:
# #             return search_results
        
# #         query_lower = query.lower()
        
# #         # Identify query intent
# #         is_comparison = any(word in query_lower for word in ['vs', 'versus', 'compare', 'comparison'])
# #         is_safety_focused = any(word in query_lower for word in ['safety', 'adverse', 'toxicity', 'side effects'])
# #         is_efficacy_focused = any(word in query_lower for word in ['efficacy', 'response', 'survival', 'orr', 'pfs', 'os'])
        
# #         # For comparison queries, try to get diverse regimens
# #         if is_comparison:
# #             # Look for different drug combinations
# #             monotherapy_trials = []
# #             combination_trials = []
            
# #             for idx, row in search_results.iterrows():
# #                 product_name = str(row.get('Product/Regimen Name', '')).lower()
# #                 if '+' in product_name or 'combination' in product_name:
# #                     combination_trials.append(idx)
# #                 else:
# #                     monotherapy_trials.append(idx)
            
# #             # Try to get one of each type for comparison
# #             selected_indices = []
# #             if combination_trials:
# #                 selected_indices.append(combination_trials[0])
# #             if monotherapy_trials and len(selected_indices) < 2:
# #                 selected_indices.append(monotherapy_trials[0])
            
# #             # Fill remaining slots if needed
# #             remaining_slots = 2 - len(selected_indices)
# #             if remaining_slots > 0:
# #                 for idx, row in search_results.iterrows():
# #                     if idx not in selected_indices:
# #                         selected_indices.append(idx)
# #                         remaining_slots -= 1
# #                         if remaining_slots == 0:
# #                             break
            
# #             if selected_indices:
# #                 return search_results.loc[selected_indices]
        
# #         # Default selection
# #         return search_results.head(2 if is_comparison else 3)
    
# #     def process_query(self, query: str) -> Dict[str, Any]:
# #         """Process a query and return results with visualization data"""
# #         # Perform hybrid search
# #         search_results = self.hybrid_search(query)
        
# #         # Generate response
# #         response = self.generate_response(query, search_results)
        
# #         # Prepare visualization data if applicable
# #         viz_data = None
# #         if not search_results.empty:
# #             # Identify trial column
# #             trial_col = next((col for col in ['Trial Acronym/ID', 'Product/Regimen Name'] 
# #                             if col in search_results.columns), None)
            
# #             if trial_col:
# #                 # Get only trials mentioned in response
# #                 viz_df = self.extract_trials_from_response(response, search_results)
                
# #                 # Get metrics for visualization
# #                 viz_metrics = self.identify_metrics_for_visualization(query, viz_df)
                
# #                 if viz_metrics and not viz_df.empty:
# #                     viz_data = {
# #                         'df': viz_df,
# #                         'trial_col': trial_col,
# #                         'metrics': viz_metrics,
# #                         'query': query
# #                     }
        
# #         return {
# #             'response': response,
# #             'search_results': search_results,
# #             'viz_data': viz_data,
# #             'num_results': len(search_results)
# #         }


# # def display_bar_charts(df: pd.DataFrame, trial_col: str, metric_cols: List[str], key_prefix: str = ""):
# #     """Display bar charts for metrics comparison with improved deduplication"""
# #     st.markdown("### 📊 Clinical Metrics Comparison")
    
# #     # First, ensure we have unique trials by removing duplicates
# #     df_unique = df.drop_duplicates(subset=[trial_col], keep='first').copy()
    
# #     # If still empty, return
# #     if df_unique.empty:
# #         st.info("No data available for visualization")
# #         return
    
# #     # Create clean trial names for display
# #     df_unique['Display_Name'] = df_unique[trial_col].apply(lambda x: str(x).split('/')[0] if '/' in str(x) else str(x))
    
# #     # Further ensure uniqueness by display name
# #     df_unique = df_unique.drop_duplicates(subset=['Display_Name'], keep='first')
    
# #     # Limit to maximum 3 trials for better visualization
# #     df_viz = df_unique.head(3).copy()
    
# #     # Prepare data for melting - only include available metrics
# #     available_metrics = [col for col in metric_cols if col in df_viz.columns]
# #     if not available_metrics:
# #         st.info("No metrics available for visualization")
# #         return
    
# #     # Melt data for plotting
# #     melted = pd.melt(df_viz[['Display_Name'] + available_metrics], 
# #                      id_vars='Display_Name',
# #                      var_name="Metric", 
# #                      value_name="RawValue")
    
# #     # Clean and process values
# #     melted["RawValue"] = melted["RawValue"].astype(str).str.strip()
# #     missing_values = {"", "na", "n/a", "nr", "nan", "none", "null", "not available", "not reached"}
# #     melted["IsMissing"] = melted["RawValue"].str.lower().isin(missing_values)
    
# #     # Extract numeric values more robustly
# #     melted["Value"] = melted["RawValue"].str.replace('%', '', regex=False)
# #     melted["Value"] = melted["Value"].str.replace('months', '', regex=False)
# #     melted["Value"] = melted["Value"].str.extract(r'([\d.]+)', expand=False)
# #     melted["Value"] = pd.to_numeric(melted["Value"], errors='coerce')
    
# #     # Set plot values (use small value for missing data)
# #     melted["PlotValue"] = melted["Value"].fillna(0)
# #     melted.loc[melted["IsMissing"], "PlotValue"] = 0.1
    
# #     # Create display text
# #     melted["DisplayText"] = melted.apply(lambda row: 
# #         "N/A" if row["IsMissing"] else 
# #         row["RawValue"].upper().replace('MONTHS', '').strip(), axis=1)
    
# #     # Create distinct colors for each trial
# #     trial_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
# #     trial_names = melted['Display_Name'].unique()
# #     trial_color_map = {name: trial_colors[i % len(trial_colors)] for i, name in enumerate(trial_names)}
    
# #     # Create bar chart
# #     fig = px.bar(
# #         melted,
# #         x="PlotValue",
# #         y="Display_Name",
# #         color="Display_Name",
# #         facet_col="Metric",
# #         facet_col_wrap=min(len(available_metrics), 3),
# #         orientation="h",
# #         text="DisplayText",
# #         color_discrete_map=trial_color_map,
# #         title=f"Clinical Metrics Comparison ({len(df_viz)} Trials)"
# #     )
    
# #     # Update layout for better appearance
# #     fig.update_traces(
# #         textposition="outside",
# #         textfont=dict(size=12, color="white"),
# #         cliponaxis=False
# #     )
    
# #     fig.update_layout(
# #         height=max(400, 120 * len(df_viz)),
# #         showlegend=False,
# #         margin=dict(l=150, r=100, t=100, b=50),
# #         plot_bgcolor="rgba(0,0,0,0)",
# #         paper_bgcolor="rgba(0,0,0,0)"
# #     )
    
# #     # Clean facet titles and axes
# #     fig.for_each_annotation(lambda a: a.update(
# #         text=a.text.split("=")[-1], 
# #         font=dict(size=14, color="white")
# #     ))
# #     fig.for_each_xaxis(lambda x: x.update(
# #         title='', 
# #         showticklabels=False,
# #         gridcolor="rgba(255,255,255,0.2)"
# #     ))
# #     fig.for_each_yaxis(lambda y: y.update(
# #         title='',
# #         tickfont=dict(size=11, color="white")
# #     ))
    
# #     st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")


# # def create_visualization(viz_data: Dict[str, Any], message_idx: int):
# #     """Create interactive visualization for search results"""
# #     if not viz_data:
# #         return
    
# #     df = viz_data['df']
# #     trial_col = viz_data['trial_col']
# #     default_metrics = viz_data['metrics']
    
# #     # Get all available metrics
# #     all_metrics = [
# #         "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
# #         "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate",
# #         "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate",
# #         "Gr 3/4 TRAEs", "Gr ≥3 TRAEs", "Gr 3/4 TEAEs", "Gr ≥3 TEAEs"
# #     ]
# #     available_metrics = [m for m in all_metrics if m in df.columns]
    
# #     if not available_metrics:
# #         return
    
# #     # Directly display the visualization without metric selection UI
# #     if default_metrics:
# #         display_bar_charts(df, trial_col, default_metrics[:4], f"viz_{message_idx}")


# # def main():
# #     st.markdown('<h1 class="main-header">🔬 Clinical Trial AI Assistant</h1>', unsafe_allow_html=True)
    
# #     # Initialize session state
# #     if 'messages' not in st.session_state:
# #         st.session_state.messages = []
# #     if 'chat_system' not in st.session_state:
# #         try:
# #             st.session_state.chat_system = ClinicalChatWithFAISS()
# #             st.success("✅ System ready! Ask me anything about clinical trials.")
# #         except Exception as e:
# #             st.error(f"Failed to initialize system: {str(e)}")
# #             st.stop()
    
# #     # Display chat history
# #     for i, message in enumerate(st.session_state.messages):
# #         with st.chat_message(message["role"]):
# #             st.markdown(message["content"])
            
# #             # Re-display visualizations
# #             if message["role"] == "assistant" and "viz_data" in message:
# #                 create_visualization(message["viz_data"], i)
    
# #     # Chat input
# #     if prompt := st.chat_input("Ask about clinical trials..."):
# #         # Add user message
# #         st.session_state.messages.append({"role": "user", "content": prompt})
# #         with st.chat_message("user"):
# #             st.markdown(prompt)
        
# #         # Process query
# #         with st.chat_message("assistant"):
# #             with st.spinner("Searching and analyzing..."):
# #                 try:
# #                     # Get results
# #                     results = st.session_state.chat_system.process_query(prompt)
                    
# #                     # Display response
# #                     st.markdown(results['response'])
                    
# #                     # Create visualization if available
# #                     if results['viz_data']:
# #                         create_visualization(results['viz_data'], len(st.session_state.messages))
                    
# #                     # Add to history
# #                     message_data = {
# #                         "role": "assistant",
# #                         "content": results['response']
# #                     }
# #                     if results['viz_data']:
# #                         message_data['viz_data'] = results['viz_data']
                    
# #                     st.session_state.messages.append(message_data)
                    
# #                 except Exception as e:
# #                     error_msg = f"❌ Error: {str(e)}"
# #                     st.error(error_msg)
# #                     st.session_state.messages.append({
# #                         "role": "assistant",
# #                         "content": error_msg
# #                     })


# # if __name__ == "__main__":
# #     main()
# import streamlit as st
# import pandas as pd
# import numpy as np
# import faiss
# import openai
# import pickle
# import os
# from typing import List, Dict, Tuple, Any, Optional
# import json
# import plotly.express as px
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import re

# # Load environment variables
# load_dotenv()

# # Page configuration
# st.set_page_config(
#     page_title="Clinical Trial AI Assistant",
#     page_icon="🔬",
#     layout="wide"
# )

# # Custom CSS for better UI
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .chat-message {
#         padding: 1.5rem;
#         border-radius: 10px;
#         margin-bottom: 1rem;
#     }
#     .user-message {
#         background-color: #e6f3ff;
#         margin-left: 20%;
#     }
#     .assistant-message {
#         background-color: #f0f2f6;
#         margin-right: 20%;
#     }
#     .metric-card {
#         background-color: white;
#         padding: 1rem;
#         border-radius: 8px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         text-align: center;
#     }
#     .stProgress > div > div > div > div {
#         background-color: #1f77b4;
#     }
# </style>
# """, unsafe_allow_html=True)

# class ClinicalChatWithFAISS:
#     def __init__(self):
#         """Initialize the chat system with pre-built FAISS index and embeddings"""
#         self.api_key = os.getenv("OPENAI_API_KEY")
#         if not self.api_key:
#             raise ValueError("OPENAI_API_KEY not found in environment variables")
        
#         openai.api_key = self.api_key
#         self.embedding_model = "text-embedding-3-large"
#         self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
#         # Load pre-built components
#         self.index, self.df, self.embeddings = self._load_saved_components()
#         self._build_comprehensive_mappings()
        
#     def _load_saved_components(self) -> Tuple[faiss.Index, pd.DataFrame, np.ndarray]:
#         """Load saved FAISS index, dataframe, and embeddings"""
#         try:
#             index = faiss.read_index("clinical_data_index.faiss")
#             df = pd.read_pickle("clinical_data_df.pkl")
#             embeddings = np.load("clinical_data_embeddings.npy")
#             return index, df, embeddings
#         except Exception as e:
#             st.error(f"Error loading saved components: {str(e)}")
#             raise
    
#     def _build_comprehensive_mappings(self):
#         """Build comprehensive alias mappings from the pasted code"""
#         # Trial acronym aliases
#         self.trial_aliases = {
#             'checkmate-511': ['cm-511', 'cm511', 'cm 511', 'checkmate511', 'checkmate 511'],
#             'checkmate-067': ['cm-067', 'cm067', 'cm 067', 'checkmate067', 'checkmate 067'],
#             'checkmate-066': ['cm-066', 'cm066', 'cm 066', 'checkmate066', 'checkmate 066'],
#             'checkmate-238': ['cm-238', 'cm238', 'cm 238', 'checkmate238', 'checkmate 238'],
#             'keynote-006': ['kn-006', 'kn006', 'kn 006', 'keynote006', 'keynote 006'],
#             'keynote-252': ['kn-252', 'kn252', 'kn 252', 'keynote252', 'keynote 252'],
#             'dreamseq': ['dream-seq', 'dream seq', 'dream sequence', 'dreamsequence'],
#             'combi-d': ['combid', 'combi d', 'combination d'],
#             'columbus': ['col', 'columbus trial', 'columbus study'],
#             'cobrim': ['co-brim', 'co brim', 'cobrim trial'],
#         }
        
#         # Phase aliases
#         self.phase_aliases = {
#             'ph3': ['phase 3', 'phase iii', 'phase3', 'phase-3', 'p3'],
#             'ph2': ['phase 2', 'phase ii', 'phase2', 'phase-2', 'p2'],
#             'ph1': ['phase 1', 'phase i', 'phase1', 'phase-1', 'p1'],
#         }
        
#         # Company aliases
#         self.company_aliases = {
#             'bms': ['bristol myers squibb', 'bristol-myers squibb', 'bristol myers'],
#             'merck': ['merck/mds', 'merck us', 'us-based merck', 'msd'],
#             'roche': ['genentech', 'roche genentech'],
#             'novartis': ['novartis pharma', 'novartis ag'],
#             'pfizer': ['pfizer inc', 'pfizer inc.'],
#         }
        
#         # Drug aliases
#         self.drug_aliases = {
#             'nivolumab': ['nivo', 'opdivo', 'bms-936558'],
#             'ipilimumab': ['ipi', 'yervoy', 'mdx-010'],
#             'pembrolizumab': ['pembro', 'keytruda', 'mk-3475'],
#             'vemurafenib': ['vemu', 'zelboraf', 'plx4032'],
#             'dabrafenib': ['dabra', 'tafinlar'],
#             'trametinib': ['trame', 'mekinist'],
#         }
        
#         # Outcome aliases
#         self.outcome_aliases = {
#             'orr': ['overall response rate', 'response rate', 'objective response rate'],
#             'cr': ['complete response', 'complete response rate'],
#             'pr': ['partial response', 'partial response rate'],
#             'pfs': ['progression free survival', 'progression-free survival', 'median pfs'],
#             'os': ['overall survival', 'median os', 'mos'],
#             'aes': ['adverse events', 'safety', 'toxicity'],
#             'traes': ['treatment related adverse events', 'treatment-related adverse events'],
#         }
        
#         # Text columns for search
#         self.text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
#         self.numeric_columns = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
    
#     def _expand_query_with_aliases(self, query: str) -> str:
#         """Expand query using alias mappings"""
#         expanded_terms = []
#         query_lower = query.lower()
        
#         # Check all alias dictionaries
#         all_aliases = [
#             self.trial_aliases, self.phase_aliases, 
#             self.company_aliases, self.drug_aliases, self.outcome_aliases
#         ]
        
#         for alias_dict in all_aliases:
#             for canonical, aliases in alias_dict.items():
#                 if canonical in query_lower:
#                     expanded_terms.extend(aliases)
#                 for alias in aliases:
#                     if alias in query_lower:
#                         expanded_terms.append(canonical)
        
#         # Combine original query with expanded terms
#         if expanded_terms:
#             expanded_query = query + " " + " ".join(expanded_terms)
#             return expanded_query
#         return query
    
#     def generate_embedding(self, text: str) -> List[float]:
#         """Generate embedding for a text using OpenAI API"""
#         try:
#             response = openai.embeddings.create(
#                 input=text,
#                 model=self.embedding_model
#             )
#             return response.data[0].embedding
#         except Exception as e:
#             st.error(f"Error generating embedding: {str(e)}")
#             return None
    
#     def semantic_search(self, query: str, k: int = 10) -> Tuple[pd.DataFrame, List[float]]:
#         """Perform semantic search using FAISS"""
#         # Expand query with aliases
#         expanded_query = self._expand_query_with_aliases(query)
        
#         # Generate embedding
#         query_embedding = self.generate_embedding(expanded_query)
#         if not query_embedding:
#             return pd.DataFrame(), []
        
#         # Normalize for cosine similarity
#         query_vec = np.array([query_embedding], dtype='float32')
#         faiss.normalize_L2(query_vec)
        
#         # Search
#         distances, indices = self.index.search(query_vec, k)
        
#         # Get results
#         valid_indices = indices[0][indices[0] != -1]
#         valid_distances = distances[0][:len(valid_indices)]
        
#         if len(valid_indices) > 0:
#             result_df = self.df.iloc[valid_indices].copy()
#             result_df['similarity_score'] = valid_distances
#             return result_df, valid_distances.tolist()
        
#         return pd.DataFrame(), []
    
#     def hybrid_search(self, query: str, k: int = 20) -> pd.DataFrame:
#         """Combine semantic search with keyword filtering"""
#         # First, get semantic search results
#         semantic_results, scores = self.semantic_search(query, k=k)
        
#         if semantic_results.empty:
#             return pd.DataFrame()
        
#         # Extract specific terms from query for additional filtering
#         query_lower = query.lower()
        
#         # Look for specific trial names, drugs, etc.
#         specific_filters = []
        
#         # Check for trial names
#         trial_patterns = [
#             r'checkmate[-\s]?\d+', r'keynote[-\s]?\d+', r'dreamseq',
#             r'combi[-\s][div]', r'columbus', r'cobrim'
#         ]
        
#         for pattern in trial_patterns:
#             matches = re.findall(pattern, query_lower)
#             specific_filters.extend(matches)
        
#         # If we have specific filters, apply them
#         if specific_filters and 'Trial Acronym/ID' in semantic_results.columns:
#             filter_mask = pd.Series(False, index=semantic_results.index)
#             for term in specific_filters:
#                 mask = semantic_results['Trial Acronym/ID'].str.contains(term, case=False, na=False)
#                 filter_mask |= mask
            
#             if filter_mask.any():
#                 # Prioritize exact matches
#                 exact_matches = semantic_results[filter_mask]
#                 other_matches = semantic_results[~filter_mask]
#                 semantic_results = pd.concat([exact_matches, other_matches])
        
#         return semantic_results
    
#     def generate_response(self, query: str, search_results: pd.DataFrame) -> Dict[str, Any]:
#         """Generate response using LLM and return both response and used trial arms"""
#         if search_results.empty:
#             return {
#                 'response': "I couldn't find any relevant clinical trials matching your query. Please try different search terms.",
#                 'used_trial_arms': []
#             }
        
#         # Prepare context from search results with unique identifiers
#         context_data = []
#         for idx, row in search_results.head(15).iterrows():
#             row_dict = row.to_dict()
#             # Create unique identifier using multiple columns since no single unique column exists
#             trial_id = str(row_dict.get('Trial Acronym/ID', 'Unknown'))
#             product = str(row_dict.get('Product/Regimen Name', 'Unknown'))
#             comparator = str(row_dict.get('Comparator', 'Unknown'))
            
#             # Create a more robust unique identifier
#             row_dict['UNIQUE_ARM_ID'] = f"{trial_id}||{product}||{comparator}||{idx}"
#             context_data.append(row_dict)
        
#         # Create enhanced prompt that asks LLM to specify which trial arms it used
#         prompt = f"""You are a clinical trial expert assistant. Answer the following question based on the provided clinical trial data.

# Question: {query}

# Available Data:
# {json.dumps(context_data, indent=2)}

# Instructions:
# 1. Provide a clear, accurate answer using ONLY the provided data
# 2. If comparing trials, create a structured comparison table
# 3. Include specific metrics (ORR, PFS, OS, etc.) when relevant
# 4. State "Not available" for any missing data
# 5. Be concise but thorough
# 6. Use markdown formatting for better readability
# 7. **CRITICAL**: At the end of your response, include a section called "**TRIAL_ARMS_USED:**" followed by a comma-separated list of the exact "UNIQUE_ARM_ID" values for ALL specific trial arms/treatments you referenced in your answer. Each UNIQUE_ARM_ID contains the trial ID, treatment regimen, comparator, and row index.

# Format for trial arms used:
# **TRIAL_ARMS_USED:** UNIQUE_ARM_ID1, UNIQUE_ARM_ID2, UNIQUE_ARM_ID3

# Answer:"""

#         try:
#             from langchain.schema import HumanMessage
#             response = self.llm.invoke([HumanMessage(content=prompt)])
#             full_response = response.content
            
#             # Extract used trial arms from response
#             used_trial_arms = self._extract_used_trial_arms(full_response, search_results)
            
#             # Clean the response by removing the TRIAL_ARMS_USED section
#             clean_response = re.sub(r'\*\*TRIAL_ARMS_USED:\*\*.*$', '', full_response, flags=re.MULTILINE | re.DOTALL).strip()
            
#             return {
#                 'response': clean_response,
#                 'used_trial_arms': used_trial_arms
#             }
            
#         except Exception as e:
#             # Fallback response
#             fallback_response = self._create_fallback_response(query, search_results)
#             # For fallback, use top results with unique identifiers
#             used_trial_arms = []
#             for idx, row in search_results.head(3).iterrows():
#                 trial_id = str(row.get('Trial Acronym/ID', 'Unknown'))
#                 product = str(row.get('Product/Regimen Name', 'Unknown'))
#                 comparator = str(row.get('Comparator', 'Unknown'))
#                 used_trial_arms.append(f"{trial_id}||{product}||{comparator}||{idx}")
            
#             return {
#                 'response': fallback_response,
#                 'used_trial_arms': used_trial_arms
#             }
    
#     def _extract_used_trial_arms(self, response: str, search_results: pd.DataFrame) -> List[str]:
#         """Extract trial arm IDs that the LLM mentioned it used"""
#         # Look for the TRIAL_ARMS_USED section
#         arms_used_match = re.search(r'\*\*TRIAL_ARMS_USED:\*\*\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        
#         if arms_used_match:
#             arms_text = arms_used_match.group(1).strip()
#             # Split by comma and clean up
#             mentioned_arms = [arm.strip() for arm in arms_text.split(',')]
            
#             # Validate that these trial arms exist in our search results
#             valid_arms = []
            
#             for mentioned_arm in mentioned_arms:
#                 # Parse the unique arm ID (format: trial_id||product_name||comparator||row_index)
#                 if '||' in mentioned_arm:
#                     parts = mentioned_arm.split('||')
#                     if len(parts) >= 4:
#                         trial_part, product_part, comparator_part, row_idx = parts[0], parts[1], parts[2], parts[3]
                        
#                         try:
#                             # Try to get by row index first (most accurate)
#                             row_idx = int(row_idx)
#                             if row_idx in search_results.index:
#                                 valid_arms.append(mentioned_arm)
#                                 continue
#                         except (ValueError, KeyError):
#                             pass
                        
#                         # Fallback to matching by content
#                         matching_rows = search_results[
#                             (search_results['Trial Acronym/ID'].astype(str).str.contains(str(trial_part), case=False, na=False)) &
#                             (search_results['Product/Regimen Name'].astype(str).str.contains(str(product_part), case=False, na=False)) &
#                             (search_results['Comparator'].astype(str).str.contains(str(comparator_part), case=False, na=False))
#                         ]
                        
#                         if not matching_rows.empty:
#                             # Use the first matching row's actual values
#                             row = matching_rows.iloc[0]
#                             actual_trial = str(row['Trial Acronym/ID'])
#                             actual_product = str(row['Product/Regimen Name'])
#                             actual_comparator = str(row['Comparator'])
#                             actual_row_idx = matching_rows.index[0]
#                             valid_arms.append(f"{actual_trial}||{actual_product}||{actual_comparator}||{actual_row_idx}")
            
#             if valid_arms:
#                 return valid_arms
        
#         # Fallback: try to extract from response content if TRIAL_ARMS_USED section not found
#         return self._fallback_trial_arm_extraction(response, search_results)
    
#     def _fallback_trial_arm_extraction(self, response: str, search_results: pd.DataFrame) -> List[str]:
#         """Fallback method to extract trial arms from response content"""
#         required_cols = ['Trial Acronym/ID', 'Product/Regimen Name', 'Comparator']
#         if not all(col in search_results.columns for col in required_cols):
#             return []
        
#         mentioned_arms = []
#         response_lower = response.lower()
        
#         # Look for specific drug combinations mentioned in response
#         drug_patterns = {
#             'pembrolizumab': ['pembrolizumab', 'keytruda', 'pembro'],
#             'nivolumab': ['nivolumab', 'opdivo', 'nivo'],
#             'ipilimumab': ['ipilimumab', 'yervoy', 'ipi'],
#             'relatlimab': ['relatlimab', 'relatl'],
#             'dabrafenib': ['dabrafenib', 'tafinlar'],
#             'trametinib': ['trametinib', 'mekinist'],
#             'vemurafenib': ['vemurafenib', 'zelboraf'],
#             'tils': ['til', 'tils', 'tumor infiltrating lymphocytes']
#         }
        
#         mentioned_drugs = []
#         for drug, aliases in drug_patterns.items():
#             if any(alias in response_lower for alias in aliases):
#                 mentioned_drugs.append(drug)
        
#         # Find rows that match mentioned drugs
#         for idx, row in search_results.iterrows():
#             product_name = str(row['Product/Regimen Name']).lower()
#             trial_id = str(row['Trial Acronym/ID'])
#             comparator = str(row['Comparator'])
            
#             # Check if this row's product contains any mentioned drugs
#             drug_matches = [drug for drug in mentioned_drugs if drug in product_name]
            
#             if drug_matches:
#                 arm_id = f"{trial_id}||{row['Product/Regimen Name']}||{comparator}||{idx}"
#                 if arm_id not in mentioned_arms:
#                     mentioned_arms.append(arm_id)
        
#         # If no specific matches, return top results based on similarity scores
#         if not mentioned_arms:
#             for idx, row in search_results.head(4).iterrows():
#                 trial_id = str(row['Trial Acronym/ID'])
#                 product = str(row['Product/Regimen Name'])
#                 comparator = str(row['Comparator'])
#                 mentioned_arms.append(f"{trial_id}||{product}||{comparator}||{idx}")
        
#         return mentioned_arms
    
#     def _create_fallback_response(self, query: str, results: pd.DataFrame) -> str:
#         """Create a structured response if LLM fails"""
#         response = f"## Results for: {query}\n\n"
#         response += f"Found {len(results)} relevant clinical trials:\n\n"
        
#         # Key columns to display
#         key_cols = ['Product/Regimen Name', 'Trial Acronym/ID', 'Therapeutic Area', 
#                    'ORR', 'mPFS', 'mOS', 'Highest Phase']
#         available_cols = [col for col in key_cols if col in results.columns]
        
#         if available_cols:
#             # Create a simple table
#             response += "| " + " | ".join(available_cols) + " |\n"
#             response += "| " + " | ".join(["---"] * len(available_cols)) + " |\n"
            
#             for _, row in results.head(5).iterrows():
#                 row_values = []
#                 for col in available_cols:
#                     value = str(row.get(col, "N/A"))[:30]
#                     row_values.append(value)
#                 response += "| " + " | ".join(row_values) + " |\n"
        
#         return response
    
#     def identify_metrics_for_visualization(self, query: str, df: pd.DataFrame) -> List[str]:
#         """Identify which metrics to visualize based on the query"""
#         query_lower = query.lower()
        
#         # All possible metric columns
#         all_metrics = [
#             "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
#             "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate",
#             "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate",
#             "Gr 3/4 TRAEs", "Gr ≥3 TRAEs", "Gr 3/4 TEAEs", "Gr ≥3 TEAEs"
#         ]
        
#         # Filter to available metrics
#         available_metrics = [m for m in all_metrics if m in df.columns]
        
#         # Extract mentioned metrics
#         mentioned_metrics = []
        
#         # Check for specific metric mentions
#         metric_keywords = {
#             'orr': ['ORR'],
#             'response': ['ORR', 'CR', 'PR'],
#             'complete response': ['CR'],
#             'partial response': ['PR'],
#             'pfs': ['mPFS', '1-yr PFS Rate', '2-yr PFS Rate'],
#             'progression': ['mPFS'],
#             'os': ['mOS', '1-yr OS Rate', '2-yr OS Rate'],
#             'survival': ['mOS', 'mPFS'],
#             'safety': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs'],
#             'adverse': ['Gr 3/4 TRAEs', 'Gr ≥3 TRAEs'],
#             'efficacy': ['ORR', 'mPFS', 'mOS']
#         }
        
#         for keyword, metrics in metric_keywords.items():
#             if keyword in query_lower:
#                 for metric in metrics:
#                     if metric in available_metrics:
#                         mentioned_metrics.append(metric)
        
#         # Remove duplicates while preserving order
#         mentioned_metrics = list(dict.fromkeys(mentioned_metrics))
        
#         # Default metrics if none mentioned
#         if not mentioned_metrics:
#             default_metrics = ['ORR', 'mPFS', 'mOS', 'Gr 3/4 TRAEs']
#             mentioned_metrics = [m for m in default_metrics if m in available_metrics][:4]
        
#         return mentioned_metrics  # No limit - return all relevant metrics
    
#     def get_trial_arms_for_visualization(self, used_trial_arms: List[str], search_results: pd.DataFrame) -> pd.DataFrame:
#         """Get the exact trial arms used by LLM for visualization"""
#         if not used_trial_arms or search_results.empty:
#             return pd.DataFrame()
        
#         viz_rows = []
        
#         for arm_id in used_trial_arms:
#             if '||' in arm_id:
#                 parts = arm_id.split('||')
                
#                 # Handle different formats
#                 if len(parts) >= 4:
#                     # Format: trial||product||comparator||row_index
#                     trial_part, product_part, comparator_part, row_idx = parts[0], parts[1], parts[2], parts[3]
                    
#                     try:
#                         # Try to get by row index first (most accurate)
#                         row_idx = int(row_idx)
#                         if row_idx in search_results.index:
#                             viz_rows.append(search_results.loc[row_idx])
#                             continue
#                     except (ValueError, KeyError):
#                         pass
                    
#                     # Fallback to matching by content
#                     matching_rows = search_results[
#                         (search_results['Trial Acronym/ID'].astype(str) == trial_part) &
#                         (search_results['Product/Regimen Name'].astype(str) == product_part) &
#                         (search_results['Comparator'].astype(str) == comparator_part)
#                     ]
                    
#                     if not matching_rows.empty:
#                         viz_rows.append(matching_rows.iloc[0])
#                         continue
                    
#                     # Try partial matching
#                     partial_match = search_results[
#                         (search_results['Trial Acronym/ID'].astype(str).str.contains(str(trial_part), case=False, na=False)) &
#                         (search_results['Product/Regimen Name'].astype(str).str.contains(str(product_part), case=False, na=False))
#                     ]
#                     if not partial_match.empty:
#                         viz_rows.append(partial_match.iloc[0])
                
#                 elif len(parts) >= 2:
#                     # Legacy format: trial||product
#                     trial_part, product_part = parts[0], parts[1]
                    
#                     matching_rows = search_results[
#                         (search_results['Trial Acronym/ID'].astype(str) == trial_part) &
#                         (search_results['Product/Regimen Name'].astype(str) == product_part)
#                     ]
                    
#                     if not matching_rows.empty:
#                         viz_rows.append(matching_rows.iloc[0])
#                     else:
#                         # Try partial matching
#                         partial_match = search_results[
#                             (search_results['Trial Acronym/ID'].astype(str).str.contains(str(trial_part), case=False, na=False)) &
#                             (search_results['Product/Regimen Name'].astype(str).str.contains(str(product_part), case=False, na=False))
#                         ]
#                         if not partial_match.empty:
#                             viz_rows.append(partial_match.iloc[0])
        
#         if viz_rows:
#             viz_df = pd.DataFrame(viz_rows)
#             # Remove any duplicates and reset index
#             viz_df = viz_df.drop_duplicates().reset_index(drop=True)
#             return viz_df
        
#         return pd.DataFrame()
    
#     def process_query(self, query: str) -> Dict[str, Any]:
#         """Process a query and return results with visualization data"""
#         # Perform hybrid search
#         search_results = self.hybrid_search(query)
        
#         # Generate response and get used trial arms
#         response_data = self.generate_response(query, search_results)
#         response = response_data['response']
#         used_trial_arms = response_data['used_trial_arms']
        
#         # Prepare visualization data if applicable
#         viz_data = None
#         if used_trial_arms and not search_results.empty:
#             # Get only trial arms that LLM actually used
#             viz_df = self.get_trial_arms_for_visualization(used_trial_arms, search_results)
            
#             if not viz_df.empty:
#                 # Get metrics for visualization
#                 viz_metrics = self.identify_metrics_for_visualization(query, viz_df)
                
#                 if viz_metrics:
#                     viz_data = {
#                         'df': viz_df,
#                         'trial_col': 'Product/Regimen Name',  # Use product name as primary identifier
#                         'metrics': viz_metrics,
#                         'query': query,
#                         'used_trial_arms': used_trial_arms
#                     }
        
#         return {
#             'response': response,
#             'search_results': search_results,
#             'viz_data': viz_data,
#             'used_trial_arms': used_trial_arms,
#             'num_results': len(search_results)
#         }


# def display_bar_charts(df: pd.DataFrame, trial_col: str, metric_cols: List[str], key_prefix: str = ""):
#     """Display bar charts for metrics comparison - supports any number of trial arms"""
#     st.markdown("### 📊 Clinical Metrics Comparison")
    
#     if df.empty:
#         st.info("No data available for visualization")
#         return
    
#     # Create display names combining trial and product for clarity
#     if 'Trial Acronym/ID' in df.columns and 'Product/Regimen Name' in df.columns:
#         df['Display_Name'] = df.apply(lambda row: 
#             f"{str(row['Trial Acronym/ID']).split('/')[0]} - {str(row['Product/Regimen Name'])[:40]}", axis=1)
#     else:
#         df['Display_Name'] = df[trial_col].apply(lambda x: str(x)[:50])
    
#     # No duplicates should exist now since we're using unique trial arms
#     df_viz = df.copy()
    
#     st.info(f"Visualizing {len(df_viz)} trial arms as referenced in the response")
    
#     # Prepare data for melting - only include available metrics
#     available_metrics = [col for col in metric_cols if col in df_viz.columns]
#     if not available_metrics:
#         st.info("No metrics available for visualization")
#         return
    
#     # Melt data for plotting
#     melted = pd.melt(df_viz[['Display_Name'] + available_metrics], 
#                      id_vars='Display_Name',
#                      var_name="Metric", 
#                      value_name="RawValue")
    
#     # Clean and process values
#     melted["RawValue"] = melted["RawValue"].astype(str).str.strip()
#     missing_values = {"", "na", "n/a", "nr", "nan", "none", "null", "not available", "not reached"}
#     melted["IsMissing"] = melted["RawValue"].str.lower().isin(missing_values)
    
#     # Extract numeric values more robustly
#     melted["Value"] = melted["RawValue"].str.replace('%', '', regex=False)
#     melted["Value"] = melted["Value"].str.replace('months', '', regex=False)
#     melted["Value"] = melted["Value"].str.extract(r'([\d.]+)', expand=False)
#     melted["Value"] = pd.to_numeric(melted["Value"], errors='coerce')
    
#     # Set plot values (use small value for missing data)
#     melted["PlotValue"] = melted["Value"].fillna(0)
#     melted.loc[melted["IsMissing"], "PlotValue"] = 0.1
    
#     # Create display text
#     melted["DisplayText"] = melted.apply(lambda row: 
#         "N/A" if row["IsMissing"] else 
#         row["RawValue"].upper().replace('MONTHS', '').strip(), axis=1)
    
#     # Create distinct colors for each trial arm (expand color palette for more arms)
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#     arm_names = melted['Display_Name'].unique()
    
#     # Extend color palette if needed
#     while len(colors) < len(arm_names):
#         colors.extend(colors)
    
#     arm_color_map = {name: colors[i % len(colors)] for i, name in enumerate(arm_names)}
    
#     # Calculate appropriate dimensions
#     num_metrics = len(available_metrics)
#     num_arms = len(df_viz)
    
#     # Create bar chart
#     fig = px.bar(
#         melted,
#         x="PlotValue",
#         y="Display_Name",
#         color="Display_Name",
#         facet_col="Metric",
#         facet_col_wrap=min(num_metrics, 4),  # Max 4 columns
#         orientation="h",
#         text="DisplayText",
#         color_discrete_map=arm_color_map,
#         title=f"Clinical Metrics Comparison ({num_arms} Trial Arms, {num_metrics} Metrics)"
#     )
    
#     # Update layout for better appearance with dynamic sizing
#     fig.update_traces(
#         textposition="outside",
#         textfont=dict(size=10, color="white"),
#         cliponaxis=False
#     )
    
#     # Dynamic height based on number of trial arms
#     chart_height = max(500, 100 * num_arms + 150)
    
#     fig.update_layout(
#         height=chart_height,
#         showlegend=False,
#         margin=dict(l=250, r=150, t=100, b=50),  # Increased left margin for longer labels
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
    
#     # Clean facet titles and axes
#     fig.for_each_annotation(lambda a: a.update(
#         text=a.text.split("=")[-1], 
#         font=dict(size=12, color="white")
#     ))
#     fig.for_each_xaxis(lambda x: x.update(
#         title='', 
#         showticklabels=False,
#         gridcolor="rgba(255,255,255,0.2)"
#     ))
#     fig.for_each_yaxis(lambda y: y.update(
#         title='',
#         tickfont=dict(size=9, color="white")
#     ))
    
#     st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")


# def create_visualization(viz_data: Dict[str, Any], message_idx: int):
#     """Create interactive visualization for search results"""
#     if not viz_data:
#         return
    
#     df = viz_data['df']
#     trial_col = viz_data['trial_col']
#     metrics = viz_data['metrics']
#     used_trial_arms = viz_data.get('used_trial_arms', [])
    
#     if metrics and not df.empty:
#         # Show which specific trial arms were referenced
#         st.markdown(f"**Trial arms referenced in response:**")
#         for i, arm_id in enumerate(used_trial_arms, 1):
#             if '||' in arm_id:
#                 parts = arm_id.split('||')
#                 if len(parts) >= 4:
#                     trial_part, product_part, comparator_part, row_idx = parts[0], parts[1], parts[2], parts[3]
#                     st.markdown(f"• **{trial_part}** - {product_part} vs {comparator_part}")
#                 elif len(parts) >= 3:
#                     trial_part, product_part, comparator_part = parts[0], parts[1], parts[2]
#                     st.markdown(f"• **{trial_part}** - {product_part} vs {comparator_part}")
#                 else:
#                     trial_part, product_part = parts[0], parts[1]
#                     st.markdown(f"• **{trial_part}** - {product_part}")
#             else:
#                 st.markdown(f"• {arm_id}")
        
#         display_bar_charts(df, trial_col, metrics, f"viz_{message_idx}")


# def main():
#     st.markdown('<h1 class="main-header">🔬 Clinical Trial AI Assistant</h1>', unsafe_allow_html=True)
    
#     # Initialize session state
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'chat_system' not in st.session_state:
#         try:
#             st.session_state.chat_system = ClinicalChatWithFAISS()
#             st.success("✅ System ready! Ask me anything about clinical trials.")
#         except Exception as e:
#             st.error(f"Failed to initialize system: {str(e)}")
#             st.stop()
    
#     # Display chat history
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
            
#             # Re-display visualizations
#             if message["role"] == "assistant" and "viz_data" in message:
#                 create_visualization(message["viz_data"], i)
    
#     # Chat input
#     if prompt := st.chat_input("Ask about clinical trials..."):
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Process query
#         with st.chat_message("assistant"):
#             with st.spinner("Searching and analyzing..."):
#                 try:
#                     # Get results
#                     results = st.session_state.chat_system.process_query(prompt)
                    
#                     # Display response
#                     st.markdown(results['response'])
                    
#                     # Create visualization if available
#                     if results['viz_data']:
#                         create_visualization(results['viz_data'], len(st.session_state.messages))
                    
#                     # Add to history
#                     message_data = {
#                         "role": "assistant",
#                         "content": results['response']
#                     }
#                     if results['viz_data']:
#                         message_data['viz_data'] = results['viz_data']
                    
#                     st.session_state.messages.append(message_data)
                    
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({
#                         "role": "assistant",
#                         "content": error_msg
#                     })


# if __name__ == "__main__":
#     main()

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

# Custom CSS for better UI
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
            r'combi[-\s][div]', r'columbus', r'cobrim'
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

Format for trial arms used:
**TRIAL_ARMS_USED:** UNIQUE_ARM_ID1, UNIQUE_ARM_ID2, UNIQUE_ARM_ID3

Answer:"""

        try:
            from langchain.schema import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            full_response = response.content
            
            # Extract used trial arms from response
            used_trial_arms = self._extract_used_trial_arms(full_response, search_results)
            
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
    
    def _extract_used_trial_arms(self, response: str, search_results: pd.DataFrame) -> List[str]:
        """Extract trial arm IDs that the LLM mentioned it used"""
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
                                valid_arms.append(mentioned_arm)
                                continue
                        except (ValueError, KeyError):
                            pass
                        
                        # Fallback to matching by content
                        matching_rows = search_results[
                            (search_results['Trial Acronym/ID'].astype(str).str.contains(str(trial_part), case=False, na=False)) &
                            (search_results['Product/Regimen Name'].astype(str).str.contains(str(product_part), case=False, na=False)) &
                            (search_results['Comparator'].astype(str).str.contains(str(comparator_part), case=False, na=False))
                        ]
                        
                        if not matching_rows.empty:
                            # Use the first matching row's actual values
                            row = matching_rows.iloc[0]
                            actual_trial = str(row['Trial Acronym/ID'])
                            actual_product = str(row['Product/Regimen Name'])
                            actual_comparator = str(row['Comparator'])
                            actual_row_idx = matching_rows.index[0]
                            valid_arms.append(f"{actual_trial}||{actual_product}||{actual_comparator}||{actual_row_idx}")
            
            if valid_arms:
                return valid_arms
        
        # Fallback: try to extract from response content if TRIAL_ARMS_USED section not found
        return self._fallback_trial_arm_extraction(response, search_results)
    
    def _fallback_trial_arm_extraction(self, response: str, search_results: pd.DataFrame) -> List[str]:
        """Fallback method to extract trial arms from response content"""
        required_cols = ['Trial Acronym/ID', 'Product/Regimen Name', 'Comparator']
        if not all(col in search_results.columns for col in required_cols):
            return []
        
        mentioned_arms = []
        response_lower = response.lower()
        
        # Look for specific drug combinations mentioned in response
        drug_patterns = {
            'pembrolizumab': ['pembrolizumab', 'keytruda', 'pembro'],
            'nivolumab': ['nivolumab', 'opdivo', 'nivo'],
            'ipilimumab': ['ipilimumab', 'yervoy', 'ipi'],
            'relatlimab': ['relatlimab', 'relatl'],
            'dabrafenib': ['dabrafenib', 'tafinlar'],
            'trametinib': ['trametinib', 'mekinist'],
            'vemurafenib': ['vemurafenib', 'zelboraf'],
            'tils': ['til', 'tils', 'tumor infiltrating lymphocytes']
        }
        
        mentioned_drugs = []
        for drug, aliases in drug_patterns.items():
            if any(alias in response_lower for alias in aliases):
                mentioned_drugs.append(drug)
        
        # Find rows that match mentioned drugs
        for idx, row in search_results.iterrows():
            product_name = str(row['Product/Regimen Name']).lower()
            trial_id = str(row['Trial Acronym/ID'])
            comparator = str(row['Comparator'])
            
            # Check if this row's product contains any mentioned drugs
            drug_matches = [drug for drug in mentioned_drugs if drug in product_name]
            
            if drug_matches:
                arm_id = f"{trial_id}||{row['Product/Regimen Name']}||{comparator}||{idx}"
                if arm_id not in mentioned_arms:
                    mentioned_arms.append(arm_id)
        
        # If no specific matches, return top results based on similarity scores
        if not mentioned_arms:
            for idx, row in search_results.head(4).iterrows():
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
    
    def identify_metrics_for_visualization(self, query: str, df: pd.DataFrame) -> List[str]:
        """Identify which metrics to visualize based on the query"""
        query_lower = query.lower()
        
        # All possible metric columns
        all_metrics = [
            "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
            "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate",
            "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate",
            "Gr 3/4 TRAEs", "Gr ≥3 TRAEs", "Gr 3/4 TEAEs", "Gr ≥3 TEAEs"
        ]
        
        # Filter to available metrics
        available_metrics = [m for m in all_metrics if m in df.columns]
        
        # Extract mentioned metrics
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
        
        for keyword, metrics in metric_keywords.items():
            if keyword in query_lower:
                for metric in metrics:
                    if metric in available_metrics:
                        mentioned_metrics.append(metric)
        
        # Remove duplicates while preserving order
        mentioned_metrics = list(dict.fromkeys(mentioned_metrics))
        
        # Default metrics if none mentioned
        if not mentioned_metrics:
            default_metrics = ['ORR', 'mPFS', 'mOS', 'Gr 3/4 TRAEs']
            mentioned_metrics = [m for m in default_metrics if m in available_metrics][:4]
        
        return mentioned_metrics  # No limit - return all relevant metrics
    
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
        
        # Prepare visualization data if applicable
        viz_data = None
        if used_trial_arms and not search_results.empty:
            # Get only trial arms that LLM actually used
            viz_df = self.get_trial_arms_for_visualization(used_trial_arms, search_results)
            
            if not viz_df.empty:
                # Get metrics for visualization
                viz_metrics = self.identify_metrics_for_visualization(query, viz_df)
                
                if viz_metrics:
                    viz_data = {
                        'df': viz_df,
                        'trial_col': 'Product/Regimen Name',  # Use product name as primary identifier
                        'metrics': viz_metrics,
                        'query': query,
                        'used_trial_arms': used_trial_arms
                    }
        
        return {
            'response': response,
            'search_results': search_results,
            'viz_data': viz_data,
            'used_trial_arms': used_trial_arms,
            'num_results': len(search_results)
        }


def display_bar_charts(df: pd.DataFrame, trial_col: str, metric_cols: List[str], key_prefix: str = ""):
    """Display bar charts for metrics comparison - supports any number of trial arms"""
    if df.empty:
        st.info("No data available for visualization")
        return
    
    # Create display names combining trial and product for clarity
    if 'Trial Acronym/ID' in df.columns and 'Product/Regimen Name' in df.columns:
        df['Display_Name'] = df.apply(lambda row: 
            f"{str(row['Trial Acronym/ID']).split('/')[0]} - {str(row['Product/Regimen Name'])[:40]}", axis=1)
    else:
        df['Display_Name'] = df[trial_col].apply(lambda x: str(x)[:50])
    
    # No duplicates should exist now since we're using unique trial arms
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
    missing_values = {"", "na", "n/a", "nr", "nan", "none", "null", "not available", "not reached"}
    melted["IsMissing"] = melted["RawValue"].str.lower().isin(missing_values)
    
    # Extract numeric values more robustly
    melted["Value"] = melted["RawValue"].str.replace('%', '', regex=False)
    melted["Value"] = melted["Value"].str.replace('months', '', regex=False)
    melted["Value"] = melted["Value"].str.extract(r'([\d.]+)', expand=False)
    melted["Value"] = pd.to_numeric(melted["Value"], errors='coerce')
    
    # Set plot values (use small value for missing data)
    melted["PlotValue"] = melted["Value"].fillna(0)
    melted.loc[melted["IsMissing"], "PlotValue"] = 0.1
    
    # Create display text
    melted["DisplayText"] = melted.apply(lambda row: 
        "N/A" if row["IsMissing"] else 
        row["RawValue"].upper().replace('MONTHS', '').strip(), axis=1)
    
    # Create distinct colors for each trial arm (expand color palette for more arms)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    arm_names = melted['Display_Name'].unique()
    
    # Extend color palette if needed
    while len(colors) < len(arm_names):
        colors.extend(colors)
    
    arm_color_map = {name: colors[i % len(colors)] for i, name in enumerate(arm_names)}
    
    # Calculate appropriate dimensions
    num_metrics = len(available_metrics)
    num_arms = len(df_viz)
    
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
    
    # Update layout for better appearance with dynamic sizing
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=10, color="white"),
        cliponaxis=False
    )
    
    # Dynamic height based on number of trial arms
    chart_height = max(500, 100 * num_arms + 150)
    
    fig.update_layout(
        height=chart_height,
        showlegend=False,
        margin=dict(l=250, r=150, t=100, b=50),  # Increased left margin for longer labels
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    # Clean facet titles and axes
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1], 
        font=dict(size=12, color="white")
    ))
    fig.for_each_xaxis(lambda x: x.update(
        title='', 
        showticklabels=False,
        gridcolor="rgba(255,255,255,0.2)"
    ))
    fig.for_each_yaxis(lambda y: y.update(
        title='',
        tickfont=dict(size=9, color="white")
    ))
    
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
        # Show which specific trial arms were referenced
        st.markdown(f"**Trial arms referenced in response:**")
        for i, arm_id in enumerate(used_trial_arms, 1):
            if '||' in arm_id:
                parts = arm_id.split('||')
                if len(parts) >= 4:
                    trial_part, product_part, comparator_part, row_idx = parts[0], parts[1], parts[2], parts[3]
                    st.markdown(f"• **{trial_part}** - {product_part} vs {comparator_part}")
                elif len(parts) >= 3:
                    trial_part, product_part, comparator_part = parts[0], parts[1], parts[2]
                    st.markdown(f"• **{trial_part}** - {product_part} vs {comparator_part}")
                else:
                    trial_part, product_part = parts[0], parts[1]
                    st.markdown(f"• **{trial_part}** - {product_part}")
            else:
                st.markdown(f"• {arm_id}")
        
        # Get all available metrics from the dataframe
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
                "Select metrics to visualize:",
                options=available_metrics,
                default=valid_default_metrics,
                help="Choose which clinical metrics to display in the comparison chart",
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
    st.markdown('<h1 class="main-header">🔬 Clinical Trial AI Assistant</h1>', unsafe_allow_html=True)
    
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
                    
                    # Create visualization if available
                    if results['viz_data']:
                        create_visualization(results['viz_data'], len(st.session_state.messages))
                    
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
