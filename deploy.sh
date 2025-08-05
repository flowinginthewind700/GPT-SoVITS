#!/bin/bash

# GPT-SoVITS Docker 部署脚本

set -e

echo "🐳 GPT-SoVITS Docker 部署脚本"
echo "================================"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p models logs configs nginx/ssl

# 检查模型文件
echo "🔍 检查模型文件..."
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "⚠️  模型目录为空，请将GPT-SoVITS模型文件放入models目录"
    echo "   模型文件结构应该如下："
    echo "   models/"
    echo "   ├── pretrained_models/"
    echo "   │   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    echo "   │   ├── s2G2333k.pth"
    echo "   │   └── ..."
    echo "   └── ..."
fi

# 构建Docker镜像
echo "🔨 构建Docker镜像..."
docker build -t gpt-sovits-api .

if [ $? -eq 0 ]; then
    echo "✅ Docker镜像构建成功"
else
    echo "❌ Docker镜像构建失败"
    exit 1
fi

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ 服务启动成功"
    echo "📊 服务状态："
    docker-compose ps
else
    echo "❌ 服务启动失败"
    echo "📋 查看日志："
    docker-compose logs gpt-sovits-api
    exit 1
fi

# 健康检查
echo "🏥 执行健康检查..."
if curl -f http://localhost:9880/docs > /dev/null 2>&1; then
    echo "✅ API服务健康检查通过"
else
    echo "❌ API服务健康检查失败"
    echo "📋 查看日志："
    docker-compose logs gpt-sovits-api
fi

echo ""
echo "🎉 部署完成！"
echo "================================"
echo "📋 服务信息："
echo "   API服务: http://localhost:9880"
echo "   API文档: http://localhost:9880/docs"
echo "   Redis: localhost:6379"
echo "   Nginx: http://localhost:80"
echo ""
echo "📝 常用命令："
echo "   查看日志: docker-compose logs -f gpt-sovits-api"
echo "   停止服务: docker-compose down"
echo "   重启服务: docker-compose restart"
echo "   更新服务: docker-compose pull && docker-compose up -d"
echo ""
echo "🔧 客户端使用示例："
echo "   python gpt_sovits_client/examples/basic_usage.py" 