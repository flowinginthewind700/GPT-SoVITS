#!/usr/bin/env python3
"""
GPT-SoVITS Client SDK Setup
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "GPT-SoVITS Client SDK - 多语言混合TTS客户端"

# 读取requirements文件
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="gpt-sovits-client",
    version="1.0.0",
    author="GPT-SoVITS Team",
    author_email="support@gpt-sovits.com",
    description="GPT-SoVITS Client SDK - 多语言混合TTS客户端",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/RVC-Boss/GPT-SoVITS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpt-sovits-client=gpt_sovits_client.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gpt_sovits_client": ["*.txt", "*.md"],
    },
    keywords="tts, text-to-speech, gpt-sovits, multilingual, voice-synthesis",
    project_urls={
        "Bug Reports": "https://github.com/RVC-Boss/GPT-SoVITS/issues",
        "Source": "https://github.com/RVC-Boss/GPT-SoVITS",
        "Documentation": "https://github.com/RVC-Boss/GPT-SoVITS/wiki",
    },
) 