"""
AutoValor — API de Predição de Preços de Veículos (Tabela FIPE)
================================================================

Esta API recebe as características de um veículo e retorna o preço estimado
usando um modelo Random Forest treinado com dados reais da Tabela FIPE.

Analogia com o laboratório:
- A API é como um sistema automatizado de análise
- O endpoint /predict é como submeter uma amostra para análise
- O Pydantic (validação) é o controle de qualidade na entrada
- O modelo .joblib é o "método analítico calibrado" serializado
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import pandas as pd
import os

# ============================================================
# 1. CARREGAR O MODELO (faz isso UMA vez quando a API inicia)
# ============================================================
# É como ligar e estabilizar o equipamento antes de começar as análises

MODELO_PATH = os.path.join(os.path.dirname(__file__), "modelo_fipe.joblib")

try:
    artefatos = joblib.load(MODELO_PATH)
    modelo = artefatos["modelo"]
    features = artefatos["features"]
    mediana_geral = artefatos["mediana_geral"]
    media_log_marca = artefatos["media_log_marca"]
    mediana_marca = artefatos["mediana_marca"]
    print(f"✓ Modelo carregado com sucesso de {MODELO_PATH}")
except FileNotFoundError:
    print(f"✗ ERRO: Modelo não encontrado em {MODELO_PATH}")
    modelo = None

# Lista de marcas válidas (extraída do modelo treinado)
MARCAS_VALIDAS = list(media_log_marca.keys()) if modelo else []


# ============================================================
# 2. DEFINIR O SCHEMA DE ENTRADA (Pydantic)
# ============================================================
# Pydantic valida automaticamente os dados de entrada.
# É como um checklist de recebimento de amostra no laboratório:
# - A amostra veio no frasco correto? (tipo de dado)
# - Está dentro do range aceitável? (validações)
# - Tem todas as informações necessárias? (campos obrigatórios)

class VeiculoInput(BaseModel):
    """Schema de entrada para predição de preço de um veículo."""

    marca: str = Field(
        ...,  # ... significa obrigatório
        description="Marca do veículo (ex: 'Toyota', 'VW - VolksWagen')",
        examples=["Toyota"]
    )
    anoModelo: int = Field(
        ...,
        description="Ano do modelo do veículo",
        ge=1985,  # ge = greater or equal (mínimo no dataset)
        le=2024,  # le = less or equal
        examples=[2015]
    )
    mesReferencia: int = Field(
        ...,
        description="Mês de referência da consulta FIPE (1-12)",
        ge=1,
        le=12,
        examples=[6]
    )
    anoReferencia: int = Field(
        ...,
        description="Ano de referência da consulta FIPE",
        ge=2001,
        le=2024,
        examples=[2022]
    )
    variacao_preco_pct: float = Field(
        default=-20.0,
        description="Variação percentual de preço do modelo (default: -20%)",
        ge=-100.0,
        le=600.0,
        examples=[-20.0]
    )

    # Validador customizado: verificar se a marca existe no dataset
    @field_validator("marca")
    @classmethod
    def validar_marca(cls, v):
        if MARCAS_VALIDAS and v not in MARCAS_VALIDAS:
            raise ValueError(
                f"Marca '{v}' não encontrada. "
                f"Exemplos válidos: {MARCAS_VALIDAS[:10]}..."
            )
        return v


# Schema de saída
class PredicaoOutput(BaseModel):
    """Schema de resposta com o preço estimado."""
    marca: str
    anoModelo: int
    preco_estimado: float = Field(description="Preço estimado em reais (R$)")
    log_preco: float = Field(description="Log do preço predito pelo modelo")
    confianca: str = Field(description="Nível de confiança da predição")


# ============================================================
# 3. CRIAR A APLICAÇÃO FastAPI
# ============================================================

app = FastAPI(
    title="AutoValor — API de Preços FIPE",
    description=(
        "Sistema de predição de preços de veículos baseado em dados reais "
        "da Tabela FIPE. Utiliza um modelo Random Forest com R²=0.978."
    ),
    version="1.0.0",
)


# ============================================================
# 4. ENDPOINTS
# ============================================================

@app.get("/")
def home():
    """Endpoint raiz — verifica se a API está funcionando."""
    return {
        "status": "online",
        "modelo": "Random Forest (R²=0.978)",
        "marcas_disponiveis": len(MARCAS_VALIDAS),
        "docs": "Acesse /docs para a documentação interativa"
    }


@app.get("/marcas")
def listar_marcas():
    """Lista todas as marcas disponíveis no modelo."""
    return {"marcas": sorted(MARCAS_VALIDAS), "total": len(MARCAS_VALIDAS)}


@app.post("/predict", response_model=PredicaoOutput)
def prever_preco(veiculo: VeiculoInput):
    """
    Prediz o preço de referência de um veículo com base em suas características.

    Recebe os dados do veículo em JSON e retorna o preço estimado em reais.
    Internamente, o modelo prediz o log(preço) e a API converte de volta.

    Analogia: é como enviar uma amostra para o laboratório e receber o laudo.
    """
    # Verificar se o modelo está carregado
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não disponível. Verifique o arquivo modelo_fipe.joblib."
        )

    # Calcular features derivadas (o "preparo da amostra")
    idade = veiculo.anoReferencia - veiculo.anoModelo

    # Buscar encoding da marca
    marca_enc = media_log_marca.get(veiculo.marca)
    if marca_enc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Marca '{veiculo.marca}' não encontrada no modelo."
        )

    # Calcular índice da marca
    med_marca = mediana_marca.get(veiculo.marca, mediana_geral)
    idx_marca = med_marca / mediana_geral

    # Montar o vetor de features na mesma ordem do treinamento
    dados = pd.DataFrame([{
        "anoModelo": veiculo.anoModelo,
        "mesReferencia": veiculo.mesReferencia,
        "anoReferencia": veiculo.anoReferencia,
        "idade_veiculo": idade,
        "indice_marca": idx_marca,
        "variacao_preco_pct": veiculo.variacao_preco_pct,
        "marca_encoded": marca_enc,
    }])[features]  # garantir a ordem correta das colunas

    # Fazer a predição
    log_pred = modelo.predict(dados)[0]
    preco = float(np.expm1(log_pred))  # inverso de log1p

    # Avaliar confiança baseada na idade do veículo
    if -1 <= idade <= 30:
        confianca = "alta"
    else:
        confianca = "baixa — veículo fora da faixa típica de treinamento"

    return PredicaoOutput(
        marca=veiculo.marca,
        anoModelo=veiculo.anoModelo,
        preco_estimado=round(preco, 2),
        log_preco=round(float(log_pred), 4),
        confianca=confianca,
    )
