# MLB_DiscordBot
Discord chat bot with llm.  

## Table of contents
- [Development environment setting](#development-environment-setting)
- [Docker setting](#docker-settting)
- [CONTRIBUTING](#contributing)
- [LICENSE](#license)

## Development environment setting
You need to set up the discord bot environment parameters.

1. **Edit `config.json`**  
    ```json
    {
        "prefix": <prefix>, # In orindary, use `!`
        "invite_link": <discord_invite_link>
    }
    ```

2. **Edit `.env`**  
    Create `.env` file and insert the following contents.
    ```
    DISCORD_TOKEN=<discord_bot_token>
    ```

3. **Setting python environment automatically**  
    `install.sh` help to set up virtual environment and install python libraries. automatically on `linux`:  

    <u>_**Warning**: if your kernal is not `zsh`, you need to change `install.sh`'s shebang!_</u> 
    ```bash
    $ chmod 700 install.sh
    $ ./install.sh
    ```

## Docker settting
1. **Build and start server:**  
    ```bash
    $ docker-compose up
    ```

2. **Enter the `Ollama container` (docker container name) and pull llm model:**  
    ```bash
    $ docker exec -it <ollama-docker-container-name> /bin/bash
    ```

    Pull the llm model.
    ```bash
    $ ollama pull <llm-model>(e.g. gemma2:9b)
    ```

## CONTRIBUTING
If you are curious about how to collaborate, please refer to [CONTRIBUTING.md](./CONTRIBUTING.md)  

## LICENSE
This project is licensed under the Apache License 2.0 - see the [LICENSE.md](./LICENSE) file for details.