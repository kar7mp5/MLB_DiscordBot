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
    """Extracts a keyword from the user's prompt using a language model.
    
    Args:
        model (langchain_ollama.OllamaLLM): Language model instance used for keyword extraction.
        user_prompt (str): The user's input from which the keyword will be extracted.
        keyword_extract_prompt (str): A system-defined prompt that guides the model in extracting the keyword.
    
    Returns:
        str: The extracted keyword as a string. If the model fails to extract a keyword, returns `False`.
    """

    # Define the template for the LLM prompt, formatting it for the system and user messages.
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
    # Create the prompt by filling in system and user prompts
    prompt = PromptTemplate(input_variables=['system_prompt', 'user_prompt'], template=template)
    
    # Invoke the model to generate a response based on the filled prompt
    keyword = model.invoke(prompt.format(system_prompt=keyword_extract_prompt, user_prompt=user_prompt)).strip()

    return keyword


def get_wikipedia_content(keyword):
    """Fetches content from Wikipedia based on a search keyword.
    
    Args:
        keyword (str): The search term extracted from the user prompt, used to find the most relevant Wikipedia page.
    
    Returns:
        str: The main content of the Wikipedia page if found; otherwise, returns `None`.
    
    Raises:
        Exception: Logs an error if fetching Wikipedia content fails, then returns `None`.
    """

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
    """Generates a response using a language model, optionally incorporating contextual information from Wikipedia content.
    
    Args:
        model (langchain_ollama.OllamaLLM): Language model instance used to generate the response.
        user_prompt (str): The user’s input prompt, serving as the primary source of information for the model.
        system_prompt (str): System-defined instructions to guide the model in responding to the user prompt.
        content (str, optional): Additional contextual content (e.g., Wikipedia text) to enhance the model’s response. 
                                 If provided, it is processed and used in conjunction with the model to refine the answer.
    
    Returns:
        str: Generated response from the language model. If `content` is provided, the response is enhanced with retrieval-based QA.
    """

    # Define the LLM prompt structure, formatting for system and user inputs
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
        # When additional context is provided, split it into manageable chunks for retrieval
        doc = Document(page_content=content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        all_splits = text_splitter.split_documents([doc])
        
        # Generate embeddings for each document chunk and create a retrievable vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
        
        # Set up a retrieval-based question-answering chain using the vector store
        qachain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
        
        # Use the QA chain to generate the response with contextual information
        response = qachain.invoke(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
        return response['result']
    else:
        # Generate the response without additional context if `content` is not provided
        response = model.invoke(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)).strip()
        return response


class GPT(commands.Cog, name="gpt"):
    """A Discord cog to interact with GPT-based language models, designed to process user input and generate responses."""
    
    def __init__(self, bot):
        """Initializes the GPT cog with a Discord bot instance and loads prompts for interaction.
        
        Args:
            bot (commands.Bot): The Discord bot instance to which this cog is attached.
        """
        self.bot = bot
        self.logger = logging.getLogger("discord_bot.gpt")  # Initialize logger for tracking cog activities
        # Load system and keyword extraction prompts with logger for error tracking
        self.prompts = get_prompts(["prompt/system_prompt.txt", "prompt/keyword_extract_prompt.txt"], self.logger)

    @commands.hybrid_command(name="gpt", description="Interact with GPT for responses.")
    async def gpt(self, context: Context, *, message: str):
        """Handles the Discord command '/gpt' to generate a response from the GPT model based on the user's message.
        
        Args:
            context (Context): The context in which the command was called.
            message (str): The user’s message that prompts GPT to generate a response.
        
        This method handles the following:
        - Displays a loading message while processing.
        - Extracts a keyword from the user message to optionally fetch additional Wikipedia context.
        - Uses the language model to generate and send a response.
        """
        try:
            # Send initial message to indicate response is being generated
            initial_message = await context.send("답변을 생성하는 중입니다. 잠시만 기다려주세요...")
            
            # Initialize the language model with specific settings
            model = OllamaLLM(model='gemma2:2b', stop=["<|eot_id|>"])
            
            # Attempt to extract a keyword from the user's message
            keyword = extract_keyword(model=model, 
                                      user_prompt=message, 
                                      keyword_extract_prompt=self.prompts["keyword_extract_prompt"])

            if keyword == "false":
                # If no keyword is found, generate response without additional Wikipedia context
                self.logger.info("No keyword found. Generating response without additional context.")
                response = generate_response(model=model,
                                             user_prompt=message, 
                                             system_prompt=self.prompts["system_prompt"])
            else:
                # If a keyword is found, attempt to retrieve content from Wikipedia
                content = get_wikipedia_content(keyword)
                if content:
                    self.logger.info(f"Found Wikipedia content for '{keyword}'.")
                    await initial_message.send(f"다음 {keyword} 문서를 검색하겠습니다...")

                    # Generate response using both the user message and the Wikipedia content
                    response = generate_response(model=model,
                                                 user_prompt=message, 
                                                 system_prompt=self.prompts["system_prompt"], 
                                                 content=content)
                else:
                    # If no Wikipedia content is found, proceed without additional context
                    self.logger.info("No content found. Generating response without additional context.")
                    response = generate_response(model=model,
                                                 user_prompt=message, 
                                                 system_prompt=self.prompts["system_prompt"])

            # Update the initial message with the generated response
            await initial_message.edit(content=response)

        except Exception as e:
            # Log error details and notify the user of an error in response generation
            self.logger.error(f"Error in gpt command: {e}")
            await context.send("An error occurred while processing your request.")


async def setup(bot):
    await bot.add_cog(GPT(bot))
