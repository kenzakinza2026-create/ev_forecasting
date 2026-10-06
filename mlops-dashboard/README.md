# DCForecaster · Ops Console

Dashboard unifié qui agrège MLflow, Prometheus, MinIO et les rapports Evidently
de la stack MLOps du projet dans une seule page React, sans problème de CORS
(un backend FastAPI fait le pont vers chaque service via le réseau Docker interne).

## Installation dans ton projet existant

1. Copie tout ce dossier `mlops-dashboard/` à la racine de
   `mlops-dcforecaster/` (à côté de `docker-compose.yml`, `drift_reports/`, etc.)

2. Ajoute le service `dashboard` dans ton `docker-compose.yml` principal :
   voir `docker-compose.snippet.yml` pour le bloc à copier-coller.

3. Ajoute la ligne `GF_SECURITY_ALLOW_EMBEDDING: "true"` dans les variables
   d'environnement du service `grafana` existant (sinon l'onglet Grafana du
   dashboard restera vide, bloqué par Grafana lui-même).

4. Lance :
   ```
   docker compose up -d --build dashboard
   ```

5. Ouvre `http://localhost:8080`.

## Pages

- **Vue d'ensemble** : statut up/down de chaque service
- **MLflow · Runs** : derniers runs (statut, métriques, params) lus en direct
- **Prometheus · Métriques** : requêtes PromQL + liste des targets scrapées
  (adapte les requêtes par défaut dans `frontend/src/components/PrometheusPanel.jsx`
  à tes vraies métriques : PR-AUC, NRMSE, drift...)
- **Evidently · Drift** : liste des rapports HTML dans `drift_reports/`,
  affichés directement en iframe
- **MinIO · Artefacts** : contenu du bucket `mlflow-artifacts`
- **Grafana** : un panneau/dashboard Grafana embarqué en iframe (à configurer
  avec ta vraie URL "Share → Embed")

## Développement local (sans Docker, hot-reload)

Terminal 1 (backend, en dehors de Docker il faudra pointer sur localhost) :
```
cd backend
pip install -r requirements.txt
MLFLOW_TRACKING_URI=http://localhost:5000 \
PROMETHEUS_URL=http://localhost:9090 \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
uvicorn main:app --reload --port 8080
```

Terminal 2 (frontend) :
```
cd frontend
npm install
npm run dev
```
→ ouvre `http://localhost:5173` (le proxy Vite redirige `/api` vers `:8080`).

## Ce qui reste à personnaliser
- Les requêtes PromQL par défaut dans `PrometheusPanel.jsx` sont génériques
  (`http_requests_total`) — remplace-les par tes vraies métriques poussées via
  le Pushgateway (PR-AUC, NRMSE, drift score...).
- L'URL Grafana dans `GrafanaPanel.jsx` est un placeholder — remplace-la par
  l'URL réelle de ton dashboard une fois créé dans Grafana.
