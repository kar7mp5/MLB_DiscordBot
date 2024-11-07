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
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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


def extract_keyword(model, user_prompt, keyword_extract_prompt):
    """Extracts a keyword from the user's prompt using a language model."""

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
    keyword = model.invoke(prompt.format(system_prompt=keyword_extract_prompt, user_prompt=user_prompt)).strip()

    return keyword


def get_wikipedia_content(keyword):
    """Fetches content from Wikipedia based on a search keyword."""

    try:
        search_results = wikipedia.search(keyword)
        if not search_results:
            return None
        page_content = wikipedia.page(search_results[0]).content
        return page_content

    except Exception as e:
        logger.error(f"Error fetching Wikipedia content: {e}")
        return None


def generate_response(model, user_prompt, system_prompt, content=None):
    """Generates a response using an LLM model with optional context from Wikipedia."""

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
        doc = Document(page_content=content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents([doc])
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
        qachain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
        response = qachain.invoke(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
        return response['result']
    else:
        response = model.invoke(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)).strip()
        return response


class GPT(commands.Cog, name="gpt"):

    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger("discord_bot.gpt")
        # Pass logger to get_prompts
        self.prompts = get_prompts(["prompt/system_prompt.txt", "prompt/keyword_extract_prompt.txt"], self.logger)


    @commands.hybrid_command(name="gpt", description="Interact with GPT for responses.")
    async def gpt(self, context: Context, *, message: str):
        """Discord command to generate a response using GPT."""
        try:
            initial_message = await context.send("답변을 생성하는 중입니다. 잠시만 기다려주세요...")
            model = OllamaLLM(model='gemma2:2b', stop=["<|eot_id|>"])
            keyword = extract_keyword(model=model, 
                                      user_prompt=message, 
                                      keyword_extract_prompt=self.prompts["keyword_extract_prompt"])

            if keyword == "false":
                logger.info("No keyword found. Generating response without additional context.")
                response = generate_response(model=model,
                                             user_prompt=message, 
                                             system_prompt=self.prompts["system_prompt"])
            else:
                content = get_wikipedia_content(keyword)
                if content:
                    logger.info(f"Found Wikipedia content for '{keyword}'.")
                    await context.send(f"다음 {keyword} 문서를 검색하겠습니다...")
                    response = generate_response(model=model,
                                                 user_prompt=message, 
                                                 system_prompt=self.prompts["system_prompt"], 
                                                 content=content)
                else:
                    logger.info("No content found. Generating response without additional context.")
                    response = generate_response(model=model,
                                                 user_prompt=message, 
                                                 system_prompt=self.prompts["system_prompt"])

            await initial_message.edit(content=response)

        except Exception as e:
            logger.error(f"Error in gpt command: {e}")
            await context.send("An error occurred while processing your request.")


async def setup(bot):
    await bot.add_cog(GPT(bot))
