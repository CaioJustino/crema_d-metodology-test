# CREMA-D Emotion Recognition (CNN)

Este presemte projeto implementa um modelo de classificação de emoções em fala utilizando **Redes Neurais Convolucionais (CNN)**. A metodologia de processamento e divisão de dados (splits) é estritamente baseada na implementação oficial do dataset **CREMA-D** para o TensorFlow Datasets (TFDS).

## Visão Geral

O objetivo é classificar áudios em 6 emoções básicas:
* **NEU**: Neutro
* **HAP**: Alegria (Happy)
* **SAD**: Tristeza (Sad)
* **ANG**: Raiva (Anger)
* **FEA**: Medo (Fear)
* **DIS**: Nojo (Disgust)

## Estrutura do Projeto

```text
.
├── data/
│   ├── raw/AudioWAV/               # Áudios originais (.wav)
│   └── processed/
│       ├── processed_results/      # Contém o summaryTable.csv
│       └── AudioMFCC/              # Features extraídas (.npy)
├── models/
│   ├── cnn.py                      # Arquitetura da Rede Neural
│   └── emotion_model.h5            # Modelo treinado
├── src/
│   └── Core/
│       ├── main.py                 # Orquestrador do projeto
│       ├── processing.py           # Filtros, Splits e MFCC
│       └── training.py             # Lógica de treino e validação
└── README.md
```

## Metodologia de Dados

### 1. Fonte de Verdade (`summaryTable.csv`)
Diferente de abordagens que processam todos os arquivos de uma pasta, este projeto utiliza o arquivo de metadados oficial para:
* Ignorar arquivos corrompidos ou com labels inconsistentes (ex: `1040_ITH_SAD_XX`).
* Garantir a integridade dos rótulos utilizados no treinamento.

### 2. Extração de Features (MFCC)
O áudio bruto é convertido em **Coeficientes Cepstrais de Frequência Mel (MFCCs)**. 
* **Taxa de Amostragem**: 16.000 Hz.
* **Duração**: Padronizada em 3.0 segundos (com padding ou crop).
* **Output**: Matrizes de shape `(time_steps, 40)`.

### 3. Split por Locutor (Speaker-based Split)
Para evitar o **Data Leakage** (vazamento de dados), o split não é aleatório por arquivo, mas sim por **grupo de locutores**:
* **Treino (70%) / Validação (10%) / Teste (20%)**.
* Se um ator está no conjunto de treino, ele **nunca** aparecerá no conjunto de teste. Isso força o modelo a aprender padrões emocionais universais em vez de memorizar timbres de vozes específicas.

## Arquitetura do Modelo

A rede é uma **CNN 1D** otimizada para sequências temporais:
1.  **3 Blocos Convolucionais**: Com `BatchNormalization`, `MaxPooling` e `Dropout`.
2.  **Global Average Pooling**: Substitui o `Flatten` para reduzir o overfitting e o número de parâmetros.
3.  **Camada Densa**: Com regularização **L2** e Dropout de 0.5 para garantir generalização.

## Como Executar

1.  **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Prepare os dados**:
    Coloque os áudios em `data/raw/AudioWAV/` e o arquivo `summaryTable.csv` em `data/processed/processed_results/`.
3.  **Rode o projeto**:
    ```bash
    python -m src.Core.main
    ```

## Resultados Esperados
O modelo busca estabilidade entre a perda de treino e validação.