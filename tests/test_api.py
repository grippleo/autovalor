"""
Testes automatizados da API AutoValor
=====================================

Usa pytest + httpx para testar a API sem precisar rodá-la manualmente.
O TestClient simula requisições HTTP — é como testar o método analítico
com padrões de concentração conhecida antes de analisar amostras reais.

Para rodar: pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
from app.main import app

# TestClient simula um "cliente" fazendo requisições à API
client = TestClient(app)


# ============================================================
# Teste 1: Requisição válida (padrão positivo)
# ============================================================
def test_predicao_valida():
    """Testa se a API retorna predição para input válido."""
    payload = {
        "marca": "Toyota",
        "anoModelo": 2015,
        "mesReferencia": 6,
        "anoReferencia": 2022,
        "variacao_preco_pct": -20.0
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "preco_estimado" in data
    assert data["preco_estimado"] > 0
    assert data["marca"] == "Toyota"
    assert data["confianca"] == "alta"

    print(f"  Toyota 2015 → R$ {data['preco_estimado']:,.2f}")


# ============================================================
# Teste 2: Input inválido — marca inexistente (branco negativo)
# ============================================================
def test_marca_invalida():
    """Testa se a API rejeita marca que não existe no dataset."""
    payload = {
        "marca": "MarcaInventada",
        "anoModelo": 2015,
        "mesReferencia": 6,
        "anoReferencia": 2022,
    }
    response = client.post("/predict", json=payload)

    # Deve retornar erro 422 (Unprocessable Entity — validação falhou)
    assert response.status_code == 422


# ============================================================
# Teste 3: Input inválido — campo obrigatório ausente
# ============================================================
def test_campo_ausente():
    """Testa se a API rejeita requisição sem campos obrigatórios."""
    payload = {
        "marca": "Toyota",
        # falta anoModelo, mesReferencia, anoReferencia
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ============================================================
# Teste 4: Input inválido — ano fora do range
# ============================================================
def test_ano_invalido():
    """Testa se a API rejeita anoModelo fora do range aceitável."""
    payload = {
        "marca": "Toyota",
        "anoModelo": 1900,  # muito antigo — fora do range do dataset
        "mesReferencia": 6,
        "anoReferencia": 2022,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ============================================================
# Teste 5: Endpoint raiz funciona
# ============================================================
def test_home():
    """Testa se o endpoint raiz retorna status online."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


# ============================================================
# Teste 6: Listagem de marcas
# ============================================================
def test_listar_marcas():
    """Testa se o endpoint /marcas retorna a lista de marcas."""
    response = client.get("/marcas")
    assert response.status_code == 200
    data = response.json()
    assert "marcas" in data
    assert len(data["marcas"]) > 0


# ============================================================
# Teste 7: Mês inválido
# ============================================================
def test_mes_invalido():
    """Testa se a API rejeita mês fora do range 1-12."""
    payload = {
        "marca": "Toyota",
        "anoModelo": 2015,
        "mesReferencia": 13,  # mês inválido
        "anoReferencia": 2022,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
