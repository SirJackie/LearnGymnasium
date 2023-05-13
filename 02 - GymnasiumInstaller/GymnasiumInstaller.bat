pip config set global.index-url https://pypi.douban.com/simple/
pip config set install.trusted-host pypi.douban.com
pip config set install.disable-pip-version-check true
pip config set install.timeout 6000

pip install gymnasium[classic-control]

start https://gymnasium.farama.org/
jupyter-lab
