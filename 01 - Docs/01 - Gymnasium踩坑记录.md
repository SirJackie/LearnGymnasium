# Gymnasium踩坑记录

## 1. 安装时遇到的问题

- 开V2Ray代理后，会导致pip安装失败，报错如下：

  ```
  ValueError: check_hostname requires server_hostname
  ```

  方法：关闭V2Ray，修改pip源为国内豆瓣源

- 使用V2Ray 全局连接，会导致无法访问IP为10开头的内网地址（例如Jupyter）

- 使用V2Ray PAC链接：会导致无法访问ChatGPT（未被添加到PAC白名单）

- 在学校电脑上PyTorch无法初始化：没有GPU导致的，请安装CPU版本的PyTorch

- 无法使用CartPole-v0：网上的代码版本太旧，请使用CartPole-v1

- 代码出错：新版本gymnasium的API有所改变，网上的代码版本太旧，请使用官方文档。

- 使用官方文档的代码还是出错 or PyCharm的API提示有误：gymnasium版本不是最新版导致的。请安装gymnasium而不是gym（新名字vs旧名字）

- pip安装gymnasium，最后时刻出现一堆红字Error，例如：

  ```
  ERROR: Command errored out with exit status...
  ```

  这是由于无法编译Box2D依赖库导致的，不影响CartPole-v1的运行，可以无视。

  但如果你要运行LunarLander，则需要解决这个问题。

  解决方法：安装VS2019和它的C++编译工具套件，然后重新pip安装gymnasium

- 啊啊啊

## 2. 官网

 [GitHub - Farama-Foundation/Gymnasium: A standard API for single-agent reinforcement learning environments, with popular reference environments and related utilities (formerly Gym)](https://github.com/Farama-Foundation/Gymnasium) 

 [Gymnasium Documentation (farama.org)](https://gymnasium.farama.org/) 

## 3. 安装

安装核心库：

```
pip install gymnasium
```

安装全部依赖库：

```
pip install gymnasium[all]
```

可能会遇到这些错误：

```
ValueError: check_hostname requires server_hostname
```

这时，请关闭所有V2Ray或者其他代理再尝试。

