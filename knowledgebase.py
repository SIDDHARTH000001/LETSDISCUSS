import os
import openai,pickle
import configparser,base64
from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from typing_extensions import List
import tqdm
import asyncio
from llama_index.core.schema import NodeWithScore,QueryBundle
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor,PrevNextNodePostprocessor,KeywordNodePostprocessor
from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import ServiceContext
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser,get_leaf_nodes,get_deeper_nodes,get_root_nodes
from llama_index.core import SimpleDirectoryReader,Document
from llama_index.core import PromptTemplate
from typing_extensions import List
import tqdm
import asyncio
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.schema import NodeWithScore,QueryBundle




# ____________________________________________ reading config
config = configparser.ConfigParser()
config.read('config.properties')
APIKey = config['AzureCredentials']['APIKey']
APIKey=base64.b64decode(APIKey).decode('utf-8')
Endpoint = config['AzureCredentials']['Endpoint']
Deployment = config['AzureCredentials']['Deployment']
version = config['AzureCredentials']['version']
EmbeddingDeployment = config['AzureCredentials']['EmbeddingDeployment']

# ____________________________________________ os enviourment 
os.environ["AZURE_OPENAI_ENDPOINT"] = Endpoint
os.environ["OPENAI_API_KEY"] = APIKey
openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = version
os.environ["azure_endpoint"] = Endpoint
os.environ["azure_endpoint"] = Endpoint


# ______________________________________________ LLM
llm =AzureOpenAI(
        deployment_name=Deployment,
        api_key=APIKey,
        azure_endpoint=Endpoint,
        api_version=version
)
embed_model = AzureOpenAIEmbedding(
    deployment_name=EmbeddingDeployment,
    api_key=APIKey,
    azure_endpoint=Endpoint,
    api_version=version,
)

# _____________________________________________ RAG Details
chunk_sizes=config.get('RAG_Details','chunk_sizes').split(',')
chunk_sizes=[int(i.strip()) for i in chunk_sizes]



# ___________________________________________________________  Create a Index for Document

def create_or_load_ko_dump(file_name):

    dump_path=f"knowledge_dump/{file_name.split('.')[0]}"

    if not os.path.exists(dump_path):
#   ____________________________________________________________________ build node parser 
    
        document=SimpleDirectoryReader(input_files=[f'./Library/{file_name}']).load_data()
        # create the hierarchical node parser w/ default settings
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes
        )
        document=Document(text='\n\n'.join([i.text for i in document]))
        nodes = node_parser.get_nodes_from_documents([document])
        leaf_nodes=get_leaf_nodes(nodes)

#   _______________________________________________________________ buidling service context 
        Service_Context=ServiceContext.from_defaults(llm=llm,embed_model=embed_model,node_parser=node_parser)   

#   _______________________________________________________________ storage context 
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        automerging_index = VectorStoreIndex(
                leaf_nodes,
                storage_context=storage_context,
                service_context=Service_Context
            )

        automerging_index.storage_context.persist(persist_dir=dump_path)
        with open(f'{dump_path}/service_context.pkl','wb') as f:
            pickle.dump(Service_Context,f)
    else:

        with open(f'{dump_path}/service_context.pkl','rb') as f:
            service_context=pickle.load(f)
        
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=dump_path),
            service_context=service_context
        )
        
    return automerging_index


# __________________________________________________ Build/reterive indexing
def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    response_mode='refine'
):
    
    retriever = VectorIndexRetriever(
        index=automerging_index,
        similarity_top_k=similarity_top_k,
        vector_store_query_mode=VectorStoreQueryMode.MMR,

    )
    
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=response_mode,
        streaming=True
        # structured_answer_filtering=True
    )

    '''
            default: "create and refine" an answer by sequentially going through each retrieved Node; 
                      This makes a separate LLM call per Node. Good for more detailed answers.
            
            compact: "compact" the prompt during each LLM call by stuffing as many Node text chunks 
                      that can fit within the maximum prompt size.If there are too many chunks to stuff in one prompt, 
                      "create and refine" an answer by going through multiple prompts.
            
            tree_summarize: Given a set of Node objects and the query, recursively construct a tree and return the root 
                            node as the response. Good for summarization purposes.
            
            no_text: Only runs the retriever to fetch the nodes that would have been sent to the LLM, 
                     without actually sending them. Then can be inspected by checking response.source_nodes. The response object 
                     is covered in more detail in Section 5.
            
            accumulate: Given a set of Node objects and the query, apply the query to each Node text chunk while accumulating 
                        the responses into an array. Returns a concatenated string of all responses. Good for when you need to 
                        run the same query separately against each text chunk.
    
    '''

    # _____________________________ KeywordNodePostprocessor : Filters nodes by required_keywords and exclude_keywords - not good
    # node_postprocessors = [
    #     KeywordNodePostprocessor(
    #         required_keywords=["Combinator"], exclude_keywords=["Italy"]
    #     )
    # ]


    # ______________________________  SimilarityPostprocessor :  filters nodes by setting a threshold on the similarity score
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)],
        response_synthesizer=response_synthesizer
    )

    return query_engine



class MultiQueriesRetriever():
    def __init__(self):
        self.template = PromptTemplate("""You are an AI language model assistant. Your task is to generate four
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by newlines.
    Original question: {question}""")
        
        self.model = llm
    
    def gen_queries(self, query)->str:
        prompt = self.template.format(question=query)
        res = self.model.complete(prompt)
        return '\n'.join(res.text.split("\n"))
    


