#!/bin/bash
#=======================================================================
# DIRETIVAS DO SLURM
#=======================================================================

#SBATCH --job-name=Emotions_MFCC_Seeds # Nome para fácil identificação
#SBATCH --partition=amd-512 # Partição correta
#SBATCH --ntasks=1 # Número de tarefas
#SBATCH --cpus-per-task=16 # Aumentado para 8 CPUs (Librosa exige muito processamento)
#SBATCH --mem=32G # Aumentado para 32GB de RAM (Arrays .npy pesam na memória)
#SBATCH --time=1-00:00:00 # Aumentado para 1 dia (para garantir que não seja cancelado no meio)

# --- Salvando Output ---
#SBATCH --output=resultados/resultado_%j.out
#SBATCH --error=erros/erro_%j.err
  
#=======================================================================
# COMANDOS DE EXECUÇÃO
#=======================================================================

# Limpa logs de avisos numéricos do TensorFlow
export TF_ENABLE_ONEDNN_OPTS=0

echo "==============================================================="
echo "Job iniciado em: $(date)"
echo "Executando no no: $(hostname)"
echo "Arquivos de saida serao salvos em 'resultados/'"
echo "Arquivos de erro serao salvos em 'erros/'"
echo "==============================================================="

# Cria as pastas de logs antes do job tentar escrever nelas
mkdir -p resultados erros

# Navegue para o diretório do seu projeto
cd ~/SyncDesk/'crema_d-metodology-test' || { echo "Erro: Diretorio do projeto nao encontrado!"; exit 1; }

# Execute o script Python DIRETAMENTE usando o binário do ambiente Conda
echo "Iniciando execucao do main.py..."

~/.conda/envs/DeepLearning/bin/python -m src.Core.main

echo "Execucao do .py finalizada! :D"

echo "==============================================================="      
echo "Job finalizado em: $(date)"
echo "==============================================================="
