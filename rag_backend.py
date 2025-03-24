import os
import openai
import faiss
import base64
import fitz as pymupdf
import logging
import warnings
import pickle
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import tabula
import json
import pandas as pd
import numpy as np

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ OpenAI API key loaded successfully!")
    openai.api_key = api_key
else:
    print("❌ OpenAI API key not found. Check your .env file.")

# Constants
BASE_DIR = "data"
VECTOR_STORE = "vector_store"
FAISS_INDEX = "faiss.index"
ITEMS_PICKLE = "items.pkl"
QUERY_EMBEDDINGS_CACHE = "query_embeddings.pkl"

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def load_or_initialize_stores():
    """Load FAISS index, document items, and query embeddings cache.
       If missing, initializes new ones.
    """
    if os.path.exists(os.path.join(VECTOR_STORE, FAISS_INDEX)):
        logger.info("🔄 Loading existing FAISS index...")
        index = faiss.read_index(os.path.join(VECTOR_STORE, FAISS_INDEX))
    else:
        logger.warning("⚠️ FAISS index not found, creating a new one...")
        index = faiss.IndexFlatL2(3072)  # Assuming embedding size of 3072

    if os.path.exists(os.path.join(VECTOR_STORE, ITEMS_PICKLE)):
        with open(os.path.join(VECTOR_STORE, ITEMS_PICKLE), "rb") as f:
            all_items = pickle.load(f)
    else:
        all_items = []

    if os.path.exists(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE)):
        with open(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE), "rb") as f:
            query_embeddings_cache = pickle.load(f)
    else:
        query_embeddings_cache = {}

    return index, all_items, query_embeddings_cache


def create_directories():
    """Create necessary directories for storing data"""
    dirs = [BASE_DIR, VECTOR_STORE]
    subdirs = ["images", "text", "tables", "page_images"]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(BASE_DIR, subdir), exist_ok=True)

def process_tables(doc, page_num, items, filepath):
    """Process tables with better table handling"""
    try:
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            # Skip empty tables
            if table.empty:
                continue
                
            # Clean table data
            table = table.fillna('')  # Handle NaN values
            
            # Create a more readable markdown table
            headers = table.columns.tolist()
            markdown_rows = []
            
            # Add headers
            markdown_rows.append("| " + " | ".join(str(h) for h in headers) + " |")
            markdown_rows.append("| " + " | ".join(['---' for _ in headers]) + " |")
            
            # Add data rows
            for _, row in table.iterrows():
                markdown_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
            table_text = f"### Table {table_idx + 1}\n" + "\n".join(markdown_rows)
            
            table_file_name = os.path.join(BASE_DIR, "tables", 
                f"{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt")
                
            with open(table_file_name, 'w', encoding='utf-8') as f:
                f.write(table_text)
                
            items.append({
                "page": page_num,
                "type": "table",
                "text": table_text,
                "path": table_file_name,
                "raw_table": table.to_dict('records')
            })
    except Exception as e:
        logger.warning(f"Table processing error on page {page_num + 1}: {str(e)}")

def process_text_chunks(text, text_splitter, page_num, items, filepath):
    """Enhanced text processing with better structure preservation"""
    import re
    
    # Document structure patterns
    patterns = {
        'heading': re.compile(r'^(?:#{1,6}|\d+\.|[A-Z][^.]+:)\s*(.+)$', re.MULTILINE),
        'bullet_list': re.compile(r'^\s*[-*•]\s+(.+)$', re.MULTILINE),
        'numbered_list': re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),
        'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
        'table': re.compile(r'^\s*\|(?:[^|]+\|)+\s*$', re.MULTILINE)
    }
    
    def extract_structure(text_block):
        """Extract structural elements while preserving their exact format"""
        elements = []
        current_element = {'type': 'text', 'content': []}
        lines = text_block.split('\n')
        code_block = False
        
        for line in lines:
            # Handle code blocks
            if line.startswith('```'):
                if code_block:
                    current_element['content'].append(line)
                    elements.append(current_element)
                    current_element = {'type': 'text', 'content': []}
                    code_block = False
                else:
                    if current_element['content']:
                        elements.append(current_element)
                    current_element = {'type': 'code', 'content': [line]}
                    code_block = True
                continue
                
            if code_block:
                current_element['content'].append(line)
                continue
                
            # Check for structural elements
            for elem_type, pattern in patterns.items():
                if pattern.match(line):
                    if current_element['content']:
                        elements.append(current_element)
                    current_element = {'type': elem_type, 'content': [line]}
                    break
            else:
                if line.strip():
                    current_element['content'].append(line)
                elif current_element['content']:
                    elements.append(current_element)
                    current_element = {'type': 'text', 'content': []}
        
        if current_element['content']:
            elements.append(current_element)
            
        return elements
    
    def save_element(element, section_num):
        """Save a structural element while preserving its format"""
        content = '\n'.join(element['content'])
        file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_{element['type']}_{page_num}_{section_num}.txt"
        
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)
            
        metadata = {
            'type': element['type'],
            'has_code': element['type'] == 'code',
            'has_list': element['type'] in ['bullet_list', 'numbered_list'],
            'has_table': element['type'] == 'table',
            'is_heading': element['type'] == 'heading'
        }
        
        return {
            "page": page_num,
            "type": "text",
            "text": content,
            "path": file_name,
            "metadata": metadata
        }
    
    try:
        # First extract structural elements
        elements = extract_structure(text)
        
        # Save elements while preserving their structure
        for i, element in enumerate(elements):
            item = save_element(element, i)
            items.append(item)
        
        # Process any remaining text traditionally
        remaining_text = text_splitter.split_text(text)
        for i, chunk in enumerate(remaining_text):
            # Only save chunks that aren't part of structural elements
            if not any(chunk in elem['content'] for elem in elements):
                text_file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
                with open(text_file_name, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                items.append({
                    "page": page_num,
                    "type": "text",
                    "text": chunk,
                    "path": text_file_name,
                    "metadata": {'type': 'text'}
                })
                
    except Exception as e:
        logger.error(f"Error processing text chunks on page {page_num}: {str(e)}")
        # Fall back to basic processing
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            text_file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
            with open(text_file_name, 'w', encoding='utf-8') as f:
                f.write(chunk)
            items.append({
                "page": page_num,
                "type": "text",
                "text": chunk,
                "path": text_file_name
            })

def process_images(page, page_num, items, filepath, doc):
    """Process images from PDF pages"""
    images = page.get_images()
    for idx, image in enumerate(images):
        try:
            xref = image[0]
            pix = pymupdf.Pixmap(doc, xref)
            
            # Improve image quality by converting to RGB if needed
            if pix.n - pix.alpha < 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                
            image_name = os.path.join(BASE_DIR, "images", 
                f"{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png")
            
            # Save image without quality parameter
            pix.save(image_name)
            
            with open(image_name, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf8')
            items.append({
                "page": page_num,
                "type": "image",
                "path": image_name,
                "image": encoded_image
            })
        except Exception as e:
            logger.warning(f"Image processing error on page {page_num + 1}, image {idx}: {str(e)}")
            continue

def process_page_images(page, page_num, items, filepath):
    """Process full page images"""
    pix = page.get_pixmap()
    page_path = os.path.join(BASE_DIR, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page", "path": page_path, "image": page_image})

def process_pdf(uploaded_file):
    """Process uploaded PDF file and extract content"""
    if uploaded_file is None:
        return None, None
    
    filepath = os.path.join(BASE_DIR, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc = pymupdf.open(filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
    items = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        process_tables(doc, page_num, items, filepath)
        process_text_chunks(text, text_splitter, page_num, items, filepath)
        process_images(page, page_num, items, filepath, doc)
        process_page_images(page, page_num, items, filepath)

    return items, filepath

def preprocess_excel_data(df):
    """
    Preprocess Excel data for better embeddings, handling various data types
    and structures generically
    """
    processed_items = []
    
    # Get column types for better handling
    column_types = df.dtypes.to_dict()
    
    # Process each row
    for idx, row in df.iterrows():
        fields = []
        
        for col, value in row.items():
            # Skip empty values
            if pd.isna(value):
                continue
                
            # Handle different data types
            if pd.api.types.is_numeric_dtype(column_types[col]):
                # Format numbers without scientific notation and with reasonable precision
                if isinstance(value, (int, np.integer)):
                    formatted_value = str(value)
                else:
                    formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
                fields.append(f"{col}: {formatted_value}")
                
            elif pd.api.types.is_datetime64_any_dtype(column_types[col]):
                # Format datetime consistently
                formatted_value = pd.to_datetime(value).strftime('%Y-%m-%d %H:%M:%S')
                fields.append(f"{col}: {formatted_value}")
                
            elif pd.api.types.is_categorical_dtype(column_types[col]):
                # Handle categorical data
                fields.append(f"{col}: {str(value)}")
                
            else:
                # Handle text and other types
                # Clean and normalize text
                cleaned_value = str(value).strip()
                if cleaned_value:
                    fields.append(f"{col}: {cleaned_value}")
        
        # Create semantic text that preserves column relationships
        semantic_text = ". ".join(fields)
        
        # Add metadata about the data types present in this row
        data_types = {
            col: {
                'type': str(dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(dtype),
                'is_categorical': pd.api.types.is_categorical_dtype(dtype),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(dtype),
                'is_text': pd.api.types.is_string_dtype(dtype)
            }
            for col, dtype in column_types.items()
            if not pd.isna(row[col])
        }
        
        processed_items.append({
            'text': semantic_text,
            'row_index': idx,
            'original_data': row.to_dict(),
            'data_types': data_types
        })
    
    return processed_items

def process_excel(uploaded_file):
    """Process uploaded Excel file with enhanced data handling"""
    if uploaded_file is None:
        return None, None
    
    filepath = os.path.join(BASE_DIR, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    items = []
    excel_file = pd.ExcelFile(filepath)
    
    for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
        # Read the sheet with appropriate data types
        df = pd.read_excel(
            filepath, 
            sheet_name=sheet_name,
            parse_dates=True,  # Automatically parse dates
            na_filter=True     # Handle missing values
        )
        
        # Convert appropriate columns to categorical
        for col in df.select_dtypes(include=['object']):
            # If column has low cardinality (few unique values), make it categorical
            if df[col].nunique() < len(df) * 0.5:  # If unique values are less than 50% of rows
                df[col] = df[col].astype('category')
        
        # Process data with type awareness
        processed_items = preprocess_excel_data(df)
        
        # Create the markdown table format for display
        headers = df.columns.tolist()
        table_data = []
        table_data.append("| " + " | ".join(str(h) for h in headers) + " |")
        table_data.append("| " + " | ".join(['---' for _ in headers]) + " |")
        
        for _, row in df.iterrows():
            formatted_row = []
            for col in headers:
                value = row[col]
                if pd.isna(value):
                    formatted_row.append('')
                elif pd.api.types.is_numeric_dtype(df[col].dtype):
                    if isinstance(value, (int, np.integer)):
                        formatted_row.append(str(value))
                    else:
                        formatted_row.append(f"{value:.4f}".rstrip('0').rstrip('.'))
                else:
                    formatted_row.append(str(value))
            table_data.append("| " + " | ".join(formatted_row) + " |")
        
        table_text = f"### Sheet: {sheet_name}\n" + "\n".join(table_data)
        
        # Save the table view
        table_file_name = os.path.join(BASE_DIR, "tables", 
            f"{os.path.basename(filepath)}_sheet_{sheet_idx}.txt")
            
        with open(table_file_name, 'w', encoding='utf-8') as f:
            f.write(table_text)
        
        # Add the table view
        items.append({
            "page": sheet_idx,
            "type": "table",
            "text": table_text,
            "path": table_file_name,
            "sheet_name": sheet_name
        })
        
        # Add processed items for semantic search
        for processed_item in processed_items:
            items.append({
                "page": sheet_idx,
                "type": "text",
                "text": processed_item['text'],
                "path": table_file_name,
                "metadata": {
                    "type": "excel_row",
                    "row_index": processed_item['row_index'],
                    "original_data": processed_item['original_data'],
                    "data_types": processed_item['data_types']
                }
            })
    
    return items, filepath

def process_document(uploaded_file):
    """Process uploaded document (PDF or Excel)"""
    if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
        return process_excel(uploaded_file)
    else:
        return process_pdf(uploaded_file)

def generate_text_embeddings(prompt):
    """Generate embeddings using OpenAI's text-embedding-3-large model."""
    if not prompt:
        raise ValueError("Please provide a valid text prompt.")

    try:
        response = openai.embeddings.create(
            input=prompt,
            model="text-embedding-3-large"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as err:
        logger.error(f"❌ Error generating embeddings: {str(err)}")
        return None

def save_stores(index, all_items, query_embeddings_cache):
    """Save FAISS vector store and cache"""
    os.makedirs(VECTOR_STORE, exist_ok=True)
    
    faiss.write_index(index, os.path.join(VECTOR_STORE, FAISS_INDEX))
    
    with open(os.path.join(VECTOR_STORE, ITEMS_PICKLE), 'wb') as f:
        pickle.dump(all_items, f)
    
    with open(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE), 'wb') as f:
        pickle.dump(query_embeddings_cache, f)

def summarize_documents(docs):
    """Summarize retrieved documents before sending them to GPT-4."""
    document_objects = [Document(page_content=doc) for doc in docs]

    llm = ChatOpenAI(model="gpt-4-turbo")
    summarizer = load_summarize_chain(llm, chain_type="stuff")  # Summarization
    return summarizer.run(document_objects)

def invoke_gpt_model(prompt, matched_items):
    """Generate response using GPT-4 with optimized document processing."""
    try:
        system_msg = """
        You are a highly intelligent assistant that extracts information ONLY from provided documents. 
        STRICT RULES:
        - Use ONLY document data
        - NO outside knowledge
        - If no info is found, say: "I don't have any information about that."
        """
        
        # ✅ LIMIT RETRIEVED DOCUMENTS TO 5
        matched_items = sorted(matched_items, key=lambda x: x.get('score',0), reverse=True)[:5]
        
        # ✅ SUMMARIZE DOCUMENTS BEFORE SENDING TO GPT-4
        summarized_docs = summarize_documents([item.get('text', '') for item in matched_items if 'text' in item])

        message_content = [
            {"type": "text", "text": summarized_docs},
            {"type": "text", "text": f"Question: {prompt}"}
        ]

        chat = ChatOpenAI(model="gpt-4o", max_tokens=800, temperature=0.7)
        response = chat.invoke([SystemMessage(content=system_msg), HumanMessage(content=message_content)])

        return judge_response(prompt, summarized_docs, response.content)
    except Exception as e:
        logger.error(f"Error invoking GPT-4: {str(e)}")
        return "I encountered an error while processing your request."

    
def clear_vector_store():
    """Clear all stored vectors and caches"""
    try:
        if os.path.exists(VECTOR_STORE):
            import shutil
            shutil.rmtree(VECTOR_STORE)
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")

def clear_history():
    """Clear the query history and cached responses"""
    try:
        if os.path.exists(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE)):
            os.remove(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE))
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")

def check_openai_credentials():
    """Verify OpenAI API key is properly configured"""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return False
        return True
    except Exception as e:
        logger.error(f"OpenAI configuration error: {str(e)}")
        return False

def judge_response(query, summarized_docs, ai_response):
    """Judge AI response for accuracy and relevance, then reformat."""
    judge_prompt = f"""
    You are a Judge Agent evaluating AI responses. Review the answer below for relevance, accuracy, and completeness.
    - **Query:** {query}
    - **Summarized Documents:** {summarized_docs}
    - **AI Response:** {ai_response}
    If correct, reformat it into bullet points. If incorrect, fix it first.
    """
    chat = ChatOpenAI(model="gpt-4-turbo")
    judged_response = chat.invoke([HumanMessage(content=judge_prompt)])
    return judged_response.content
