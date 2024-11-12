"""
gpt.py

Copyright © kar7mp5 2024-Present - https://github.com/kar7mp5
"""

from discord.ext import commands
from discord.ext.commands import Context

# LLM model import
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# RAG (Retrieval-Augmented Generation)
import wikipedia
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio

import logging
import os


# Logger 설정
logger = logging.getLogger("gpt")
logger.setLevel(logging.INFO)


def get_prompts(file_list, logger):
    """Fetch prompt text from specified files.

    Args:
        file_list (list): List of relative file paths for prompt files.
        logger (logging.Logger): Logger instance for logging.
    
    Returns:
        dict: Dictionary containing content of each prompt file.
    """
    # Generate full paths based on current directory
    full_file_list = [os.path.join(os.getcwd(), file_path) for file_path in file_list]
    del file_list  # Clean up original list

    # Check if all specified files exist
    file_exist = all(os.path.exists(file_path) for file_path in full_file_list)

    if file_exist:
        prompts = {}

        for file_path in full_file_list:
            # Log the prompt loading instead of printing
            logger.info(f"Loading prompt - {os.path.basename(file_path)}")
            with open(file_path) as f:
                prompts[os.path.basename(file_path).replace('.txt', '')] = f.read()

        return prompts

    else:
        logger.error("FileNotFoundError: One or more prompt files were not found.")
        return None


async def extract_keyword(model, user_prompt, keyword_extract_prompt):
    """Extracts a keyword from the user's prompt using a language model.
    
    Args:
        model (langchain_ollama.OllamaLLM): Language model instance for keyword extraction.
        user_prompt (str): The user’s input to extract keywords from.
        keyword_extract_prompt (str): System-defined prompt for guiding the model in keyword extraction.

    Returns:
        str: The extracted keyword or 'False' if extraction fails.
    """
    # Define the prompt template for system and user inputs
    template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    prompt = PromptTemplate(input_variables=['system_prompt', 'user_prompt'], template=template)
    
    # Use await for asynchronous model invocation
    keyword = model.invoke(prompt.format(system_prompt=keyword_extract_prompt, user_prompt=user_prompt)).strip()
    return keyword


async def get_wikipedia_content(keyword):
    """Asynchronously fetches content from Wikipedia based on a search keyword.
    
    Args:
        keyword (str): Search term to locate the Wikipedia page.

    Returns:
        str: Content from the Wikipedia page if found; otherwise, returns None.
    
    Raises:
        Exception: Logs an error if fetching Wikipedia content fails.
    """
    try:
        loop = asyncio.get_running_loop()
        # Use run_in_executor to make Wikipedia API calls asynchronously
        search_results = await loop.run_in_executor(None, wikipedia.search, keyword)
        if not search_results:
            return None
        page_content = await loop.run_in_executor(None, wikipedia.page, search_results[0])
        return page_content.content
    except Exception as e:
        logger.error(f"Error fetching Wikipedia content: {e}")
        return None


async def generate_response(model, user_prompt, system_prompt, content=None):
    """Generates a response using a language model, optionally incorporating Wikipedia content asynchronously.
    
    Args:
        model (langchain_ollama.OllamaLLM): Language model for generating responses.
        user_prompt (str): User’s input prompt.
        system_prompt (str): System-defined prompt guiding the model.
        content (str, optional): Additional context, e.g., Wikipedia content, to enhance the response.

    Returns:
        str: Generated response from the model.
    """
    template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    prompt = PromptTemplate(input_variables=['system_prompt', 'user_prompt'], template=template)
    
    if content:
        # When context is provided, split it for retrieval-augmented generation
        doc = Document(page_content=content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        all_splits = text_splitter.split_documents([doc])

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)

        qachain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
        
        # Generate a response using the QA chain and context
        response = qachain.invoke(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
        return response['result']
    else:
        # Generate a response without additional context
        response = model.invoke(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)).strip()
        return response


class GPT(commands.Cog, name="gpt"):
    """A Discord cog for GPT-based language model interactions, allowing users to query and receive responses asynchronously."""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger("discord_bot.gpt")
        self.prompts = get_prompts(["prompt/system_prompt.txt", "prompt/keyword_extract_prompt.txt"], self.logger)

    @commands.hybrid_command(name="gpt", description="Interact with GPT for responses.")
    async def gpt(self, context: Context, *, message: str):
        """Handles the Discord command '/gpt' to generate a response from the GPT model based on the user's message.
        
        Args:
            context (Context): The context in which the command was called.
            message (str): User’s message prompting GPT to generate a response.
        
        Steps:
        - Sends an initial loading message to the user.
        - Extracts a keyword from the user's message for possible Wikipedia retrieval.
        - Generates a response using the language model, with or without Wikipedia context.
        """
        try:
            initial_message = await context.send("답변을 생성하는 중입니다. 잠시만 기다려주세요...")
            model = OllamaLLM(model='gemma2:9b', stop=["<|eot_id|>"])

            keyword = await extract_keyword(model=model, user_prompt=message, keyword_extract_prompt=self.prompts["keyword_extract_prompt"])

            if keyword == "false":
                # If no keyword is found, generate response without additional Wikipedia context
                self.logger.info("No keyword found. Generating response without additional context.")
                response = await generate_response(model=model,
                                             user_prompt=message, 
                                             system_prompt=self.prompts["system_prompt"])

                await initial_message.edit(content=f"## 질문: {message}\n\n ## 답변: \n{response}")

            else:
                # Retrieve Wikipedia content if a keyword is extracted
                content = await get_wikipedia_content(keyword)
                if content:
                    self.logger.info(f"Found Wikipedia content for '{keyword}'.")
                    await initial_message.edit(content=f"다음 {keyword} 문서를 검색하겠습니다...")

                    # Generate response using both the user message and the Wikipedia content
                    response = await generate_response(model=model,
                                                 user_prompt=message, 
                                                 system_prompt=self.prompts["system_prompt"], 
                                                 content=content)
                    await initial_message.edit(content=f"## 질문: {message}\n\n## 답변: \n{response}\n### [참고문헌](https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')})")

                else:
                    # Generate response without additional context if no content is found
                    response = await generate_response(model=model, user_prompt=message, system_prompt=self.prompts["system_prompt"])

                    await initial_message.edit(content=f"## 질문: {message}\n\n## 답변: \n{response}")

        except Exception as e:
            logger.error(f"Error in gpt command: {e}")
            await context.send("답변하는데 오류가 발생하였습니다. 일반적으로 답변 단어수가 2,000자를 넘긴 경우에 해당할 것입니다.")

async def setup(bot):
    await bot.add_cog(GPT(bot))