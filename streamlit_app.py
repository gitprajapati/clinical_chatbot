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

class FlexibleClinicalChatbot:
    """Flexible chatbot that can handle any clinical trial question without restrictive filtering"""
    
    def __init__(self, df: pd.DataFrame, model: str = "gpt-4o-mini"):
        self.df = self._prepare_dataframe(df)
        self.model = model
        
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe with ID column and clean data"""
        df_clean = df.copy()
        
        # Add ID column if not exists
        if 'ID' not in df_clean.columns:
            df_clean.insert(0, 'ID', range(1, len(df_clean) + 1))
        
        # Clean column names
        df_clean.columns = df_clean.columns.str.strip().str.replace("\n", " ").str.replace("  ", " ")
        
        return df_clean
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary for context"""
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'key_info': {}
        }
        
        # Identify key columns and their unique values
        key_column_patterns = {
            'companies': ['company', 'sponsor', 'manufacturer', 'developer', 'pharma'],
            'treatments': ['product', 'regimen', 'treatment', 'drug', 'therapy'],
            'trials': ['trial', 'study', 'acronym', 'id'],
            'phases': ['phase'],
            'indications': ['indication', 'disease', 'cancer', 'tumor'],
            'outcomes': ['orr', 'pfs', 'os', 'response', 'survival', 'adverse', 'safety']
        }
        
        for category, patterns in key_column_patterns.items():
            matching_cols = []
            for col in self.df.columns:
                if any(pattern in col.lower() for pattern in patterns):
                    matching_cols.append(col)
            
            if matching_cols:
                summary['key_info'][category] = {
                    'columns': matching_cols,
                    'sample_values': {}
                }
                
                # Get sample unique values for first matching column
                first_col = matching_cols[0]
                unique_vals = self.df[first_col].dropna().unique()
                summary['key_info'][category]['sample_values'][first_col] = unique_vals[:10].tolist()
        
        return summary
    
    def create_comprehensive_agent(self) -> any:
        """Create an agent with the full dataset and comprehensive instructions"""
        
        dataset_summary = self.get_dataset_summary()
        
        prefix = f"""You are an expert clinical trial data analyst with access to a comprehensive clinical trial dataset.

DATASET INFORMATION:
- Total rows: {dataset_summary['total_rows']}
- Total columns: {dataset_summary['total_columns']}
- ALL COLUMNS AVAILABLE: {', '.join(dataset_summary['columns'])}

CORE PRINCIPLES:
1. YOU HAVE COMPLETE ACCESS TO ALL DATA - {dataset_summary['total_rows']} rows and {dataset_summary['total_columns']} columns
2. ALWAYS extract REAL data from the dataset - never use placeholders or "Not specified"
3. Use proper pandas operations to search and filter data
4. Check multiple columns when searching for information
5. Report actual values found in the dataset

SEARCH METHODOLOGY:
- Use df[column].str.contains('term', case=False, na=False) for text searches
- Use df[df['column'] == value] for exact matches
- Use df.loc[] and df.iloc[] to access specific data
- Always verify data exists before making claims
- Check related columns if primary search doesn't find data

DATA PRESENTATION:
1. Extract and display ACTUAL values from the dataset
2. Create accurate tables with real data from the dataframe
3. Provide comprehensive analysis based on available information
4. Include insights and patterns found in the actual data
5. Present findings in clean, professional markdown format

CRITICAL: This dataset contains real information - find it, extract it, and report it accurately. Never guess or use placeholders.
"""
        
        return create_pandas_dataframe_agent(
            ChatOpenAI(model=self.model, temperature=0),
            self.df,
            verbose=False,  # Keep clean for production
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=prefix,
            allow_dangerous_code=True,
            max_iterations=25,  # Sufficient iterations for complex queries
            early_stopping_method="generate"
        )
    
    def _format_key_info(self, key_info: Dict) -> str:
        """Format key dataset information for the prompt"""
        formatted = []
        for category, info in key_info.items():
            if info['columns']:
                formatted.append(f"- {category.upper()}: Columns available: {', '.join(info['columns'])}")
                for col, values in info['sample_values'].items():
                    if values:
                        formatted.append(f"  Sample {col}: {', '.join(map(str, values[:5]))}")
        return '\n'.join(formatted)
    
    def query(self, question: str) -> Dict[str, any]:
        """Process any clinical trial question with full dataset access"""
        
        # Create agent with full dataset
        agent = self.create_comprehensive_agent()
        
        # Create generalized enhancement for any question
        enhanced_question = f"""
        {question}
        
        ANALYSIS REQUIREMENTS:
        1. Access the complete dataset with all {len(self.df)} rows and {len(self.df.columns)} columns
        2. Use proper pandas operations to find and extract real data
        3. Report actual values found in the dataset - never use "Not specified" without verification
        4. Search comprehensively across relevant columns for the requested information
        5. Present findings in professional format with accurate data tables
        
        Provide a complete, data-driven analysis that answers the question thoroughly.
        """
        
        try:
            result = agent.invoke({"input": enhanced_question})
            
            # Extract the output
            if isinstance(result, dict):
                output = result.get('output', str(result))
            else:
                output = str(result)
                
            return {
                "output": output,
                "dataset_info": {
                    "total_rows_analyzed": len(self.df),
                    "total_columns_available": len(self.df.columns),
                    "query_type": "comprehensive_analysis"
                }
            }
            
        except Exception as e:
            # Simplified fallback that still ensures data access
            try:
                simple_agent = create_pandas_dataframe_agent(
                    ChatOpenAI(model=self.model, temperature=0),
                    self.df,
                    verbose=False,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True,
                    max_iterations=15
                )
                
                result = simple_agent.invoke({"input": question})
                output = result.get('output', str(result)) if isinstance(result, dict) else str(result)
                
                return {
                    "output": output,
                    "dataset_info": {
                        "total_rows_analyzed": len(self.df),
                        "total_columns_available": len(self.df.columns),
                        "query_type": "fallback_analysis"
                    }
                }
                
            except Exception as e2:
                return {
                    "output": f"Error processing query: {str(e2)}\n\nDataset contains {len(self.df)} rows and {len(self.df.columns)} columns of clinical trial data.",
                    "dataset_info": {
                        "total_rows_analyzed": len(self.df),
                        "error": str(e2)
                    }
                }


# Streamlit App
def main():
    st.title("🏥 Clinical Trial Chatbot")
    
    # Load default data
    default_file_path = "Sample Data.xlsx"
    
    if os.path.exists(default_file_path):
        if 'data_loaded' not in st.session_state:
            with st.spinner(f"Loading data..."):
                try:
                    df = pd.read_excel(default_file_path)
                    
                    # Store data and chatbot in session state
                    st.session_state.df = df
                    st.session_state.chatbot = FlexibleClinicalChatbot(df)
                    st.session_state.data_loaded = True
                    st.session_state.messages = []
                    
                    st.success(f"✅ Data loaded successfully!")
                            
                except Exception as e:
                    st.error(f"❌ Error loading data file: {e}")
                    return
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask any question about the clinical trials..."):
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
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result.get("output", "No response generated") if result else "Error occurred",
                            "dataset_info": result.get("dataset_info", {}) if result else {}
                        })
                        
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg,
                            "dataset_info": {"error": str(e)}
                        })
            
    else:
        # Error message when default file is not found
        st.error(f"❌ Default data file '{default_file_path}' not found!")
        st.info("Please ensure the Sample Data.xlsx file is in the same directory as this script.")

if __name__ == "__main__":
    main()
