"""Neo4j connection from environment variables (see .env.example)."""
import os

from neo4j import GraphDatabase


def driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise SystemExit("NEO4J_PASSWORD is not set. Run:  set -a && source .env && set +a")
    return GraphDatabase.driver(uri, auth=(user, password))
