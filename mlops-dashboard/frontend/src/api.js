const BASE = "";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  mlflowRuns: (max = 50) => get(`/api/mlflow/runs?max_results=${max}`),
  prometheusQuery: (q) => get(`/api/prometheus/query?query=${encodeURIComponent(q)}`),
  prometheusTargets: () => get(`/api/prometheus/targets`),
  minioArtifacts: () => get(`/api/minio/artifacts`),
  driftReports: () => get(`/api/drift-reports`),
};
