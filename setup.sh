#!/bin/bash
# Script para criar a estrutura atualizada do projeto
# Execute este script DENTRO do diretório raiz do seu projeto (ex: ./meu_novo_projeto/)

echo "Criando diretórios..."

# Criar as pastas principais
mkdir -p configs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models

# Criar as subpastas de src (com a primeira letra maiúscula conforme o diagrama)
mkdir -p src/Core

echo "Criando arquivos..."

# Arquivos da raiz e configurações
touch requirements.txt
touch configs/params.yml

# Arquivos do módulo Core
touch src/Core/__init__.py
touch src/Core/training.py
touch src/Core/processing.py
touch src/Core/main.py

# Listar a estrutura criada para verificação
echo "Estrutura de pastas e arquivos criada com sucesso:"
ls -R