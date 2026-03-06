# ============================================================
# Dockerfile — AutoValor API
# ============================================================
# O Dockerfile é a "receita" para construir o container.
# Analogia: é como o SOP (procedimento operacional padrão)
# do laboratório — lista todos os passos para reproduzir
# o ambiente de trabalho em qualquer lugar.

# 1. Imagem base: Python 3.11 versão slim (leve)
FROM python:3.11-slim

# 2. Definir o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copiar e instalar dependências primeiro (otimização de cache)
#    Se o requirements.txt não mudar, o Docker reutiliza esta camada
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar o código da aplicação e o modelo
COPY app/ ./app/
COPY tests/ ./tests/

# 5. Expor a porta 8000 (onde a API vai escutar)
EXPOSE 8000

# 6. Comando para iniciar a API quando o container subir
#    uvicorn é o servidor ASGI que roda o FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
