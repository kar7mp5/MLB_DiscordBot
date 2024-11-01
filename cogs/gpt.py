# gpt.py

from discord.ext import commands
from discord.ext.commands import Context
from openai import OpenAI
from dotenv import load_dotenv
import os

class GPT(commands.Cog, name="gpt"):

    def __init__(self, bot):
        self.bot = bot
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @commands.hybrid_command(name="gpt", description="Can communicate with gpt.")
    async def gpt(self, context: Context, *, message: str):
        try:
            # Send an initial response indicating processing
            initial_message = await context.send("Processing your request, please wait...")

            # Call the OpenAI API
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Do not exceed 5,000 words."},
                    {"role": "user", "content": message}
                ]
            )

            # Edit the initial message with the GPT response
            await initial_message.edit(content=completion.choices[0].message.content)
        
        except Exception as e:
            print(e)
            # Send an error message if an exception occurs
            await context.send("An error occurred while processing your request.")

async def setup(bot):
    # Add the GPT cog to the bot
    await bot.add_cog(GPT(bot))
