# Setup Tailscale
https://login.tailscale.com/admin/machines/new-linux

# SSH into the Tailscale machine
ssh ubuntu@{tailscale_ip}

# Install uv and clone the nanochat repository
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/ewjordan/nanochat && cd nanochat && git checkout recurrent-layer-state
uv venv && uv sync --extra gpu &&source .venv/bin/activate
# Login to wandb
wandb login
(Go to https://wandb.ai/authorize?ref=models, copy the authorization token, and paste it into the terminal)

# Pull the latest changes, start the training, tail logs
git pull && cd ~/nanochat && mkdir -p ~/nanochat/local_rls_experiments_full && touch ~/nanochat/local_rls_experiments_full/rls.log && touch ~/nanochat/local_rls_experiments_full/baseline.log && tmux new -d -s training "cd ~/nanochat && ./local_train_rls_full.sh" && tail -f ./local_rls_experiments_full/*

# Kill the training session
tmux kill-session -t training