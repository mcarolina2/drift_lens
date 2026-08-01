"""
github_api.py
Busca o drift_report.json e metadados do commit via API oficial do GitHub
(em vez de raw.githubusercontent.com), permitindo obter autor e data do
commit que gerou o relatório.
"""

import base64
import json
import os

import requests

GITHUB_API_BASE = "https://api.github.com"


def _headers():
    """Usa token se disponível (maior rate limit: 5000/h vs 60/h anônimo)."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_file_content(owner: str, repo: str, path: str, branch: str = "main") -> dict:
    """Busca o conteúdo de um arquivo via API (contents endpoint)."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(url, headers=_headers(), params={"ref": branch}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    conteudo_decodificado = base64.b64decode(data["content"]).decode("utf-8")
    return json.loads(conteudo_decodificado)


def get_last_commit_info(owner: str, repo: str, path: str, branch: str = "main") -> dict:
    """Busca autor e data do último commit que alterou o arquivo."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
    params = {"path": path, "sha": branch, "per_page": 1}
    resp = requests.get(url, headers=_headers(), params=params, timeout=10)
    resp.raise_for_status()
    commits = resp.json()

    if not commits:
        return {"autor": "desconhecido", "data": "desconhecida", "mensagem": "", "sha": ""}

    commit = commits[0]
    return {
        "autor": commit["commit"]["author"]["name"],
        "data": commit["commit"]["author"]["date"],
        "mensagem": commit["commit"]["message"],
        "sha": commit["sha"][:7],
        "url": commit["html_url"],
    }


def get_drift_report_com_metadados(owner: str, repo: str, path: str = "drift_report.json", branch: str = "main") -> dict:
    """Combina conteúdo do relatório + metadados do commit em um único dict."""
    conteudo = get_file_content(owner, repo, path, branch)
    commit_info = get_last_commit_info(owner, repo, path, branch)
    conteudo["_commit"] = commit_info
    return conteudo