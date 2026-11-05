# Use a PostgreSQL image that already includes dense and sparse pgvector types.
FROM pgvector/pgvector:0.8.7-pg17-bookworm

LABEL org.opencontainers.image.description="Local PostgreSQL 17 with pgvector for the resume RAG engine"

EXPOSE 5432
