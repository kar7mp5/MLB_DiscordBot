"""
gpt.py

Copyright © kar7mp5 2024-Present - https://github.com/kar7mp5
"""

from discord.ext import commands
from discord.ext.commands import Context

# LLM model import
from langchain_community.llms import Ollama
from langchain import PromptTemplate

# RAG (Retrieval-Augmented Generation)
import wikipedia
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

import os


def get_prompts(file_list=None):
    """Fetch prompt text from specified files.

    Args:
        file_list (list, optional): List of relative file paths for prompt files.
    
    Returns:
        dict: Dictionary containing content of each prompt file.
    """
    if file_list is not None:
        # Generate full paths based on current directory
        full_file_list = [os.path.join(os.path.dirname(os.getcwd()), file_path) for file_path in file_list]
        del file_list  # Clean up original list

        # Check if all specified files exist
        file_exist = all(os.path.exists(file_path) for file_path in full_file_list)

        if file_exist:
            prompts = {}

            for file_path in full_file_list:
                print(f"Loading prompt:\033[96m{os.path.basename(file_path)}\033[0m")
                with open(file_path) as f:
                    prompts[os.path.basename(file_path).replace('.txt', '')] = f.read()

            return prompts

        else:
            print("FileNotFoundError")
            return None
    else:
        print("FileNotFoundError")
        return None


def extract_keyword(model, user_prompt, keyword_extract_prompt):
    """Extracts a keyword from the user's prompt using a language model.

    Args:
        model (langchain_community.llms.Ollama): Local LLM model (e.g., gemma2:2b).
        user_prompt (str): User's input prompt for question.
        keyword_extract_prompt (str): Prompt for extracting keywords.

    Returns:
        str: Extracted keyword from the user's prompt.
    """
    # Template for the keyword extraction prompt
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
    keyword = model(prompt.format(system_prompt=keyword_extract_prompt, user_prompt=user_prompt)).strip()
    
    return keyword


def get_wikipedia_content(keyword):
    """Fetches content from Wikipedia based on a search keyword.

    Args:
        keyword (str): Keyword to search in Wikipedia.
    
    Returns:
        str: Content of the Wikipedia page related to the keyword, if available.
    """
    try:
        search_results = wikipedia.search(keyword)
        if not search_results:
            return None
        page_content = wikipedia.page(search_results[0]).content
        return page_content

    except Exception as e:
        print(f"Error fetching Wikipedia content: {e}")
        return None


def generate_response(model, user_prompt, system_prompt, content=None):
    """Generates a response using an LLM model with optional context from Wikipedia.

    Args:
        model (langchain_community.llms.Ollama): Local LLM model (e.g., gemma2:2b).
        user_prompt (str): User's input prompt for question.
        system_prompt (str): System-level prompt for guiding the response.
        content (str, optional): Wikipedia content to include for enhanced context.
    
    Returns:
        str: Model's response based on the user's prompt.
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
        # Split content and embed for RAG processing if content is provided
        doc = Document(page_content=content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents([doc])
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
        
        # Retrieval-Augmented Generation (RAG) chain for context-based response
        qachain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
        response = qachain(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
        return response['result']
    
    else:
        # Generate response without additional context
        response = model(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)).strip()
        return response


class GPT(commands.Cog, name="gpt"):
    """Discord bot cog that interacts with an LLM model for generating responses.

    Attributes:
        bot (commands.Bot): The Discord bot instance.
        prompts (dict): Loaded prompts used for interactions with the model.
    """

    def __init__(self, bot):
        self.bot = bot
        self.prompts = get_prompts(["prompt/system_prompt.txt", "prompt/keyword_extract_prompt.txt"])

    @commands.hybrid_command(name="gpt", description="Interact with GPT for responses.")
    async def gpt(self, context: Context, *, message: str):
        """Discord command to generate a response using GPT.

        Args:
            context (Context): The context in which the command was used.
            message (str): User's input message to be processed.
        
        Returns:
            None
        """
        try:
            # Send a message to indicate that processing is ongoing
            initial_message = await context.send("Processing your request, please wait...")

            model = Ollama(model='gemma2:2b', stop=["<|eot_id|>"])
            
            # Extract keyword using the prompt
            keyword = extract_keyword(model=model, 
                                      user_prompt=message, 
                                      keyword_extract_prompt=self.prompts["keyword_extract_prompt"])

            # Determine if a keyword was successfully extracted
            if keyword == "false":
                print("No keyword found. Generating response without additional context.")
                response = generate_response(model=model,
                                             user_prompt=message, 
                                             system_prompt=self.prompts["system_prompt"])

            else:
                # Retrieve Wikipedia content based on keyword, if available
                content = get_wikipedia_content(keyword)
                if content:
                    print(f"Found Wikipedia content for '{keyword}'.")
                    response = generate_response(model=model,
                                                 user_prompt=message, 
                                                 system_prompt=self.prompts["system_prompt"], 
                                                 content=content)
                else:
                    print("No content found. Generating response without additional context.")
                    response = generate_response(model=model,
                                                 user_prompt=message, 
                                                 system_prompt=self.prompts["system_prompt"])

            # Edit the initial message with the GPT response
            await initial_message.edit(content=response)

        except Exception as e:
            print(e)
            # Send an error message if an exception occurs
            await context.send("An error occurred while processing your request.")


async def setup(bot):
    await bot.add_cog(GPT(bot))
