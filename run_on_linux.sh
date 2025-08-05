#!/bin/bash

# GPT-SoVITS 在Linux服务器上运行脚本

set -e

echo "🚀 GPT-SoVITS Linux服务器部署脚本"
echo "======================================"

# 私有仓库配置
REGISTRY="1.82.180.74:5000"
USERNAME="xueyuan"
PASSWORD="WOaini3344"
IMAGE_NAME="gpt-sovits-api"
TAG="latest"
CONTAINER_NAME="gpt-sovits-api"

echo "📝 配置信息:"
echo "   仓库地址: $REGISTRY"
echo "   镜像名称: $IMAGE_NAME"
echo "   标签: $TAG"
echo "   容器名称: $CONTAINER_NAME"
echo ""

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请启动Docker"
    exit 1
fi

# 登录到私有仓库
echo "🔐 登录到私有仓库..."
echo $PASSWORD | docker login $REGISTRY -u $USERNAME --password-stdin

# 停止并删除旧容器
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "🛑 停止并删除旧容器..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# 拉取最新镜像
echo "📥 拉取最新镜像..."
docker pull $REGISTRY/$IMAGE_NAME:$TAG

# 运行新容器
echo "🚀 启动新容器..."
docker run -d \
    --name $CONTAINER_NAME \
    -p 9880:9880 \
    -v $(pwd):/app \
    --restart unless-stopped \
    $REGISTRY/$IMAGE_NAME:$TAG

echo ""
echo "✅ 部署完成！"
echo "======================================"
echo "📝 服务信息:"
echo "   容器名称: $CONTAINER_NAME"
echo "   服务地址: http://localhost:9880"
echo "   API文档: http://localhost:9880/docs"
echo ""
echo "🔧 常用命令:"
echo "   查看日志: docker logs -f $CONTAINER_NAME"
echo "   停止服务: docker stop $CONTAINER_NAME"
echo "   重启服务: docker restart $CONTAINER_NAME"
echo "   进入容器: docker exec -it $CONTAINER_NAME bash"
echo ""
echo "📊 检查服务状态..."
sleep 5
if curl -f http://localhost:9880/docs > /dev/null 2>&1; then
    echo "✅ 服务运行正常！"
else
    echo "⚠️  服务可能还在启动中，请稍等片刻..."
    echo "   查看日志: docker logs $CONTAINER_NAME"
fi 