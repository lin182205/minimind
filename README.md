# MiniMind

MiniMind 模型与预训练代码。

## 在服务器上拉取

私有仓库需要服务器具有仓库访问权限。已为 GitHub 账号配置 SSH 密钥时：

```bash
git clone git@github.com:lin182205/minimind.git
cd minimind
```

也可以使用 HTTPS，并通过 Git 凭据管理器或个人访问令牌认证：

```bash
git clone https://github.com/lin182205/minimind.git
cd minimind
```

后续在仓库目录内更新：

```bash
git pull --ff-only origin main
```

## 目录

- `model/`：模型定义。
- `dataset/`：数据集加载代码。
- `trainer/`：预训练脚本、训练工具和分词器文件。
- `requirements.txt`：Python 依赖。

训练数据、模型权重、检查点和本地环境不纳入 Git，需要在服务器上另行准备。此仓库同步代码；训练环境与运行流程尚未在服务器上验证。
