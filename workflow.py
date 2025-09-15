import os
from typing import Optional, Any, List, cast


from llama_index.core.schema import  NodeWithScore
from llama_index.core.workflow import (
    Event,
)
from llama_index.core import SummaryIndex

from llama_index.core.schema import Document
from prompt_template import DEFAULT_RELEVANCY_PROMPT_TEMPLATE , DEFAULT_TRANSFORM_QUERY_TEMPLATE


from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    step,
    Workflow,
    Context,
)
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import LLM
from llama_index.llms.azure_openai import AzureOpenAI


from linkup import LinkupClient
from llama_index.tools.linkup_research import LinkupToolSpec
from llama_index.core.base.base_retriever import BaseRetriever



import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


from dotenv import load_dotenv
load_dotenv()


class RetrieveEvent(Event) :
    retrieved_nodes : List[NodeWithScore]


class WebSearchEvent(Event):
    """Web search event."""
    relevant_text: str  


class QueryEvent(Event):
    """Query event. Queries given relevant text and search text."""
    relevant_text: str
    search_text: str





class CorrectiveRAGWorkflow(Workflow):
    def __init__(self, index, linkup_api_key: str , llm : AzureOpenAI , **kwargs: Any ) -> None :

        super().__init__(**kwargs)
        self.index = index
        linkup_api_key = os.environ["LINKUP_API_KEY"]

        self.linkup_tool = LinkupToolSpec(api_key=linkup_api_key,
                                          depth="deep",
                                          output_type="searchResults", # or "sourcedAnswer" or "structured" 
                                        ) 
        self.llm = llm
    

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> Optional[RetrieveEvent]:
        """Retrieve the relevant nodes for the query."""
        query_str = ev.get("query_str")
        retriever_kwargs = ev.get("retriever_kwargs", {})

        if not query_str:
            return None
        
        retriever: BaseRetriever = self.index.as_retriever(**retriever_kwargs)
        result = retriever.retrieve(query_str)

        await ctx.set("retrieved_nodes", result)
        await ctx.set("query_str", query_str)

        return RetrieveEvent(retrieved_nodes=result)
    

    @step
    async def eval_relevance(self, ctx: Context, ev: RetrieveEvent ) -> WebSearchEvent | QueryEvent :
        """Evaluate relevancy of retrieved documents with the query."""

        retrieved_nodes = ev.retrieved_nodes
        query_str = await ctx.get("query_str")

        relevancy_results = []

        for node in retrieved_nodes: 
            prompt = DEFAULT_RELEVANCY_PROMPT_TEMPLATE.format(context_str=node.text, query_str=query_str)

            #ollama_llm = cast(Ollama, self.llm)
            #relevancy = await ollama_llm.acomplete(prompt)


            relevancy = await self.llm.acomplete(prompt)
            relevancy_results.append(relevancy.text.lower().strip())
        
        relevant_texts = [
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if result == "yes"
        ]

        relevant_text = "\n".join(relevant_texts)

        if "no" in relevancy_results:
            return WebSearchEvent(relevant_text=relevant_text)
        else:
            return QueryEvent(relevant_text=relevant_text, search_text="")
    

    @step
    async def web_search(self, ctx: Context, ev: WebSearchEvent) -> QueryEvent: 
        """Search the transformed query with Tavily API."""

        query_str = await ctx.get("query_str")

        prompt = DEFAULT_TRANSFORM_QUERY_TEMPLATE.format(query_str=query_str)
     

        #ollama_llm = cast(Ollama, self.llm)

        result = await self.llm.acomplete(prompt)
        transformed_query_str = result.text
     
        # Perform search with the transformed query string
        search_results = self.linkup_tool.search(transformed_query_str).results
        

        search_text = "\n".join([res.content for res in search_results])

        return QueryEvent(relevant_text=ev.relevant_text, search_text=search_text)
    

    @step
    async def query_result(self, ctx: Context, ev: QueryEvent) -> StopEvent :
        """Get result with relevant text."""

        relevant_text = ev.relevant_text
        search_text = ev.search_text

        query_str = await ctx.get("query_str")

        documents = [Document(text=relevant_text + "\n" + search_text)]
        index = SummaryIndex.from_documents(documents, llm=self.llm)
        
        query_engine = index.as_query_engine(llm=self.llm)
        result = await query_engine.aquery(query_str)

        logging.info("Reached end of workflow")

        return StopEvent(result=result)

    

    










       
       



    

