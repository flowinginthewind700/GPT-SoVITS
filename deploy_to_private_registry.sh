#!/bin/bash

# GPT-SoVITS 推送到私有Docker仓库脚本

set -e

echo "🚀 GPT-SoVITS 推送到私有Docker仓库"
echo "======================================"

# 私有仓库配置
REGISTRY="1.82.180.74:5000"
USERNAME="xueyuan"
PASSWORD="WOaini3344"
IMAGE_NAME="gpt-sovits-api"
TAG="latest"

echo "📝 配置信息:"
echo "   仓库地址: $REGISTRY"
echo "   镜像名称: $IMAGE_NAME"
echo "   标签: $TAG"
echo ""

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请启动Docker"
    exit 1
fi

# 登录到私有仓库
echo "🔐 登录到私有仓库..."
echo $PASSWORD | docker login $REGISTRY -u $USERNAME --password-stdin

# 构建镜像
echo "🔨 构建Docker镜像..."
docker build -t $IMAGE_NAME:$TAG .

# 给镜像打标签
echo "🏷️  给镜像打标签..."
docker tag $IMAGE_NAME:$TAG $REGISTRY/$IMAGE_NAME:$TAG

# 推送镜像
echo "📤 推送镜像到私有仓库..."
docker push $REGISTRY/$IMAGE_NAME:$TAG

echo ""
echo "✅ 推送完成！"
echo "======================================"
echo "📝 使用说明:"
echo ""
echo "1. 在Linux服务器上拉取镜像:"
echo "   docker pull $REGISTRY/$IMAGE_NAME:$TAG"
echo ""
echo "2. 运行容器:"
echo "   docker run -d \\"
echo "     --name gpt-sovits-api \\"
echo "     -p 9880:9880 \\"
echo "     -v \$(pwd):/app \\"
echo "     $REGISTRY/$IMAGE_NAME:$TAG"
echo ""
echo "3. 或者使用docker-compose:"
echo "   docker-compose -f docker-compose-private.yaml up -d"
echo ""
echo "🌐 服务地址: http://localhost:9880"
echo "📚 API文档: http://localhost:9880/docs" 