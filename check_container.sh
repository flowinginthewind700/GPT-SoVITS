#!/bin/bash

echo "🔍 检查GPT-SoVITS容器状态"
echo "======================================"

# 检查容器状态
echo "📊 容器状态:"
docker ps -a | grep gpt-sovits-api

echo ""
echo "📋 容器日志:"
docker logs gpt-sovits-api --tail 50

echo ""
echo "🔧 进入容器检查:"
echo "docker exec -it gpt-sovits-api bash"
echo ""
echo "📝 手动启动API服务:"
echo "docker exec -it gpt-sovits-api conda run -n GPTSoVits python api_v2.py" 