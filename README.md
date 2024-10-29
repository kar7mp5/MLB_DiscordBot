# MLB_DiscordBot
Discord chat bot with llm.  

### Way to setup
You need to set up the bot environment parameters.

#### `config.json` file
```json
{
    "prefix": <prefix>,
    "invite_link": <discord_invite_link>
}
```

#### `.env` file
```
DISCORD_TOKEN=<discord_bot_token>
```

### Quick Start
`install.sh` help to set up virtual environment and install python libraries automatically on `linux`.  

```bash
$ ./install.sh
```

Install python libararies by yourself with this command:  

```bash
$ pip3 install -r requirements.txt
```

### LICENSE
This project is licensed under the Apache License 2.0 - see the [LICENSE.md](./LICENSE) file for details.