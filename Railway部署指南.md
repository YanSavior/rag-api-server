# Railway RAG API 部署指南

## 关于Railway
Railway是一个现代化的云平台，对国内用户访问友好，免费计划包含：
- 每月500小时免费使用
- 自动部署
- 全球CDN加速
- 支持自定义域名

## 部署步骤

### 1. 准备文件
确保以下文件已准备好：
- `api_server.py` - 主API服务器
- `requirements.txt` - Python依赖
- `Procfile` - 启动命令
- `railway.json` - Railway配置
- `runtime.txt` - Python版本
- `rag_system/` - RAG系统模块
- `data/` - 数据文件

### 2. 创建GitHub仓库
```bash
# 如果还没有GitHub仓库，创建一个新的
git init
git add .
git commit -m "Initial commit for Railway deployment"
git branch -M main
git remote add origin https://github.com/YanSavior/rag-api-server.git
git push -u origin main
```

### 3. 在Railway上部署

#### 方法一：通过GitHub连接（推荐）
1. 访问 https://railway.app/
2. 使用GitHub账号登录
3. 点击 "New Project"
4. 选择 "Deploy from GitHub repo"
5. 选择您的仓库 `YanSavior/rag-api-server`
6. 选择分支 `main`
7. 点击 "Deploy Now"

#### 方法二：通过Railway CLI
```bash
# 安装Railway CLI
npm install -g @railway/cli

# 登录Railway
railway login

# 初始化项目
railway init

# 部署
railway up
```

### 4. 配置环境变量（可选）
在Railway项目设置中可以添加环境变量：
- `PORT` - 端口号（Railway会自动设置）
- `PYTHON_VERSION` - Python版本

### 5. 获取部署URL
部署完成后，Railway会提供一个URL，类似：
`https://your-app-name.railway.app`

### 6. 测试API
访问以下端点测试：
- `https://your-app-name.railway.app/` - 根路径
- `https://your-app-name.railway.app/api/health` - 健康检查
- `https://your-app-name.railway.app/api/query` - 查询接口

## 国内访问说明

Railway的优势：
- ✅ 全球CDN加速，国内访问速度快
- ✅ 自动SSL证书
- ✅ 支持自定义域名
- ✅ 免费计划稳定

## 故障排除

### 常见问题

1. **构建失败**
   - 检查 `requirements.txt` 中的依赖版本
   - 确保Python版本兼容

2. **启动失败**
   - 检查 `Procfile` 中的启动命令
   - 查看Railway日志

3. **模块导入错误**
   - 确保所有Python文件都在正确位置
   - 检查文件路径

### 查看日志
在Railway控制台可以查看实时日志，帮助诊断问题。

## 更新部署
每次推送代码到GitHub，Railway会自动重新部署。

```bash
git add .
git commit -m "Update for Railway deployment"
git push origin main
```

## 成本说明
- 免费计划：每月500小时
- 超出后按使用量计费
- 可以设置使用限制避免意外费用 