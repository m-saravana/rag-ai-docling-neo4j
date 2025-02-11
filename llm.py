import logging
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Any

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self, model_name: str = "phi"):
        self.llm = Ollama(
            model=model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question. If you cannot answer the question based on the context, say "I don't have enough information to answer that."

Context: {context}

Question: {question}

Answer: """
        )
        
        self.llm_chain = self.prompt | self.llm

    def generate_response(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        logger.debug(f"Question: {question}")
        logger.debug(f"Number of context chunks: {len(context_chunks)}")
        logger.debug(f"Context chunks similarity scores: {[chunk.get('similarity', 0) for chunk in context_chunks]}")
        
        context = "\n".join([chunk['content'] for chunk in context_chunks])
        logger.debug(f"Combined context: {context}")
        
        response = self.llm_chain.invoke(
            {"context": context, "question": question}
        )
        logger.debug(f"LLM Response: {response}")
        
        return response.content if hasattr(response, 'content') else response