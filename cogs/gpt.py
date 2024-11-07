"""
gpt.py

Copyright © kar7mp5 2024-Present - https://github.com/kar7mp5
"""

from discord.ext import commands
from discord.ext.commands import Context

# llm model
from langchain_community.llms import Ollama
from langchain import PromptTemplate

# RAG
import wikipedia
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings




def extract_keyword(model, user_prompt):
    """Extract keyword from user prompt using LLM model
    
    Args:
        model (langchain_community.llms.Ollama): local llm model; gemma2:2b
        user_prompt (str): user question prompt
        
    Returns:
        keyword (str): keyword extracted from user prompts
    """

    keyword_extract_system_prompt = """
Think and write your step-by-step reasoning before responding.
Please write only the fully spelled-out form of the acronym in English that corresponds to the following user's question, without abbreviations or additional text.
If you don't know how to respond, just say false.
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
    keyword = model(prompt.format(system_prompt=keyword_extract_system_prompt, user_prompt=user_prompt)).strip()
    
    return keyword


def get_wikipedia_content(keyword):
    """Fetch content from Wikipedia based on the keyword
    
    Args:
        keyword (str): search keyword on wikipedia
        
    Returns:
        page_content (str): search keyword that exists on wikipedia database
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


def generate_response(model, user_prompt, content=None):
    """Generate response using GPT model with optional document content
    
    Args:
        model (langchain_community.llms.Ollama): local llm model; gemma2:2b
        user_prompt (str): user question prompt

    Returns:
        response (str): result of user's prompt
    """

    system_prompt = """
Please write all conversations in Korean(한국어).
Think and write your step-by-step reasoning before responding.
Write the article title using ## in Markdown syntax.
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
        # Split and embed content if provided
        doc = Document(page_content=content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents([doc])
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
        
        # Use RAG for response generation
        qachain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
        response = qachain(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
        return response['result']
    else:
        # Generate response without additional document
        response = model(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)).strip()
        return response




class GPT(commands.Cog, name="gpt"):

    def __init__(self, bot):
        self.bot = bot


    @commands.hybrid_command(name="gpt", description="Can communicate with gpt.")
    async def gpt(self, context: Context, *, message: str):
        try:
            # Send an initial response indicating processing
            initial_message = await context.send("Processing your request, please wait...")

            model = Ollama(model='gemma2:2b', stop=["<|eot_id|>"])
            
            keyword = extract_keyword(model, message)

            # if keyword is existed, the model use RAG
            if keyword == "false":
                print("키워드를 찾을 수 없습니다. 검색 없이 응답을 생성합니다.")
                response = generate_response(model, message)
            else:
                content = get_wikipedia_content(keyword)
                if content:
                    print(f"{keyword}에 대한 Wikipedia 문서를 찾았습니다.")
                    response = generate_response(model, message, content=content)
                else:
                    print("문서를 찾을 수 없습니다. 검색 없이 응답을 생성합니다.")
                    response = generate_response(model, message)

            # Edit the initial message with the GPT response
            await initial_message.edit(content=response)

        except Exception as e:
            print(e)
            # Send an error message if an exception occurs
            await context.send("An error occurred while processing your request.")


async def setup(bot):
    # Add the GPT cog to the bot
    await bot.add_cog(GPT(bot))
