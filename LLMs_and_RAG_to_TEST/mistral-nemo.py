from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from tabulate import tabulate
import json
import re

# load data
input_dir_path = "RAG_Folder"
loader = SimpleDirectoryReader(
            input_dir = input_dir_path,
            required_exts=[".pdf"],
            recursive=True
        )
docs = loader.load_data()
# print(f"Loaded {len(docs)} documents")


#embedding
embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)


#Vector database
# ====== Create vector store and upload indexed data ======
Settings.embed_model = embed_model # we specify the embedding model to be used
index = VectorStoreIndex.from_documents(docs)

#Query Engine
# setting up the llm
llm = Ollama(model="mistral-nemo", request_timeout=120.0,) 

# ====== Setup a query engine on the index previously created ======
Settings.llm = llm # specifying the llm to be used
query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

# Modified prompt template 
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, provide a structured response in the following JSON format. "
    "Provide estimated price ranges based on typical contractor costs:\n"
    "{\n"
    "  'items': [\n"
    "    {'item': 'item name', 'price_range': 'estimated $min-$max', 'priority': 'high/medium/low'}\n"
    "  ]\n"
    "}\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

def format_json_string(json_str):
    """Convert single quotes to double quotes for valid JSON"""
    return json_str.replace("'", '"')

# Get response and format as table
response = query_engine.query('Create a Table of what needs to be fixed in the Home to improve the property value. Include estimated price ranges.')
try:
    # Extract JSON using regex
    json_match = re.search(r'({[\s\S]*})', str(response))
    if json_match:
        json_str = json_match.group(1)
        json_str = format_json_string(json_str)
        data = json.loads(json_str)
        
        # Sort by priority
        priority_map = {
            'high': 0,
            'high-medium': 1,
            'medium': 2,
            'low-medium': 3,
            'low': 4
        }
        
        sorted_items = sorted(
            data['items'],
            key=lambda x: priority_map.get(x['priority'].lower(), 5)
        )
        
        # Format table
        table = [[
            item['item'],
            item['price_range'],
            item['priority'].upper()
        ] for item in sorted_items]
        
        headers = ['Item', 'Estimated Cost', 'Priority']
        
        print("\nHome Improvement Recommendations:")
        print("================================")
        print(tabulate(table, headers=headers, tablefmt='grid'))
        
        # Extract and print notes if present
        notes = str(response).split(json_str)[-1].strip()
        if notes:
            print("\nNotes:")
            print("------")
            print(notes)
            
except Exception as e:
    print(f"Error formatting response: {e}")
    print("Raw response:")
    print(response)