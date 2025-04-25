# mcp-server-demo
## Installation
1. Install npx (Node Package eXecutor)
* https://nodejs.org/en/download/

Install nvm
```
# Linux
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 20

# macos
brew install node@22
echo 'export PATH="/opt/homebrew/opt/node@22/bin:$PATH"' >> ~/.zshrc
export LDFLAGS="-L/opt/homebrew/opt/node@22/lib"
export CPPFLAGS="-I/opt/homebrew/opt/node@22/include
```

Check installation
```
(base) muhayu@croa01:~/downloads$ node -v
v20.19.1
(base) muhayu@croa01:~/downloads$ nvm current
v20.19.1
```

## Setup

2. Clade Desktop
ex.
'/Users/yrsong/Library/Application Support/Claude/claude_desktop_config.json'

### Resources
* MCP Server Quickstart Guide [[link](https://modelcontextprotocol.io/quickstart/server)]