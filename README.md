# AutoValor — Sistema de Previsão de Preço de Veículos (Tabela FIPE)

Sistema de Machine Learning para estimativa automática de preços de veículos com base em dados reais da Tabela FIPE, servido como API REST containerizada com Docker.

## Visão Geral

| Componente | Tecnologia |
|---|---|
| Machine Learning | Random Forest (scikit-learn) — R² = 0.978 |
| API REST | FastAPI + Pydantic |
| Testes | pytest + httpx (7 testes) |
| Containerização | Docker + Docker Compose |
| Dataset | [Tabela FIPE — Histórico de Preços (Kaggle)](https://www.kaggle.com/datasets/franckepeixoto/tabela-fipe) |

## Estrutura do Repositório

```
autovalor/
├── notebooks/
│   └── analise_exploratoria.ipynb   # EDA + treinamento dos modelos (Etapas 1 e 2)
├── app/
│   ├── main.py                      # API REST com FastAPI (Etapa 3)
│   └── modelo_fipe.joblib           # Modelo serializado (gerado pelo notebook)
├── tests/
│   └── test_api.py                  # Testes automatizados (Etapa 3)
├── Dockerfile                       # Imagem Docker (Etapa 4)
├── docker-compose.yml               # Orquestração (Etapa 4)
└── requirements.txt                 # Dependências
```

## Como Executar

### 1. Clonar o repositório e baixar o dataset

```bash
git clone https://github.com/grippleo/autovalor.git
cd autovalor
```

Baixe o dataset da [Tabela FIPE no Kaggle](https://www.kaggle.com/datasets/franckepeixoto/tabela-fipe) e extraia o CSV.

### 2. Gerar o modelo (Etapas 1 e 2)

```bash
pip install -r requirements.txt
```

Abra e execute o notebook `notebooks/analise_exploratoria.ipynb` do início ao fim. Ele realiza a análise exploratória, o feature engineering, o treinamento de 5 modelos e salva o melhor modelo (`modelo_fipe.joblib`). Copie o arquivo gerado para a pasta `app/`.

### 3. Rodar a API localmente (Etapa 3)

```bash
uvicorn app.main:app --reload
```

A API estará disponível em `http://localhost:8000`. Acesse `http://localhost:8000/docs` para a documentação interativa (Swagger UI).

### 4. Rodar os testes

```bash
pytest tests/test_api.py -v
```

### 5. Rodar com Docker (Etapa 4)

```bash
docker compose up --build
```

A API estará acessível em `http://localhost:8000` dentro do container.

## Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | Status da API |
| GET | `/marcas` | Lista todas as marcas disponíveis |
| POST | `/predict` | Prediz o preço de um veículo |

### Exemplo de requisição (`/predict`)

```json
{
  "marca": "Toyota",
  "anoModelo": 2015,
  "mesReferencia": 6,
  "anoReferencia": 2022,
  "variacao_preco_pct": -20.0
}
```

### Exemplo de resposta

```json
{
  "marca": "Toyota",
  "anoModelo": 2015,
  "preco_estimado": 67020.24,
  "log_preco": 11.1128,
  "confianca": "alta"
}
```

## Resultados — Comparação de Modelos

| Ranking | Algoritmo | R² | RMSE | Erro Aproximado |
|---|---|---|---|---|
| 1 | **Random Forest** | **0.9778** | **0.1557** | **~16.8%** |
| 2 | Gradient Boosting | 0.9274 | 0.2812 | ~32.5% |
| 3 | SVR | 0.8568 | 0.3950 | ~48.4% |
| 4 | Regressão Linear | 0.7944 | 0.4733 | ~60.5% |
| 5 | Ridge | 0.7944 | 0.4733 | ~60.5% |

## Feature Engineering

Três features derivadas foram criadas a partir das variáveis originais:

- **idade_veiculo** — diferença entre o ano de referência e o ano do modelo (depreciação temporal)
- **indice_marca** — preço mediano da marca / preço mediano geral (posicionamento premium vs. popular)
- **variacao_preco_pct** — variação percentual do preço do modelo ao longo do tempo (tendência de valorização/depreciação)

As três features derivadas somam ~43% da importância total no modelo final.

## Decisões de Limpeza

- **Idades negativas (< -1):** Removidos ~102 mil registros com idades impossíveis (ex: Alfa Romeo 156 com anoModelo=2022 na FIPE de 2001), identificados como artefatos de encoding incorreto.
- **Outliers de preço:** Removidos registros abaixo do P1 (R$ 4.832) e acima do P99 (R$ 992.452).
- **Resultado:** Dataset reduzido de 466.020 para 356.483 registros (~23% de perda), com ganho em qualidade.

## Tecnologias

- Python 3.11
- FastAPI 0.115 + Uvicorn
- scikit-learn 1.5.2
- pandas + numpy
- pytest + httpx
- Docker + Docker Compose
