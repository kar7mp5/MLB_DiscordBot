# MLB_DiscordBot
Discord chat bot with llm.  

## Table of contents

- [Development environment setting](#development-environment-setting)
- [Docker setting](#docker-settting)
- [LICENSE](#license)

## Development environment setting
You need to set up the discord bot environment parameters.

#### 1.`config.json` file
```json
{
    "prefix": <prefix>,
    "invite_link": <discord_invite_link>
}
```

#### 2. `.env` file
```
DISCORD_TOKEN=<discord_bot_token>
```

#### 3. Install python virtual environment
`install.sh` help to set up virtual environment and install python libraries automatically on `linux`.  

```bash
./install.sh
```

Install python libararies by yourself with this command:  

```bash
pip3 install -r requirements.txt
```

## Docker Settting
To build:
```bash
docker build -t discord-bot .
```

To run:
```bash
docker run -d --name discord-proj discord-bot
```

To stop:
```bash
docker ps
docker stop <docker-container-name>
```

### LICENSE
This project is licensed under the Apache License 2.0 - see the [LICENSE.md](./LICENSE) file for details.