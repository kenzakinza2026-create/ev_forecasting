import { useEffect, useState } from "react";
import { api } from "../api";

const DEFAULT_QUERIES = [
  { label: "HTTP requests (rate 5m)", query: "sum(rate(http_requests_total[5m]))" },
  { label: "Latency p95 (5m)", query: "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))" },
];

export default function PrometheusPanel() {
  const [results, setResults] = useState({});
  const [targets, setTargets] = useState(null);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    DEFAULT_QUERIES.forEach((q) => {
      api.prometheusQuery(q.query)
        .then((data) => setResults((prev) => ({ ...prev, [q.label]: data })))
        .catch((e) => setErrors((prev) => ({ ...prev, [q.label]: e.message })));
    });
    api.prometheusTargets().then(setTargets).catch(() => {});
  }, []);

  return (
    <div>
      <h1 className="page-title">Prometheus · Metrics</h1>
      <p className="page-sub">Common PromQL queries against your Prometheus instance. Adjust the queries in PrometheusPanel.jsx to match your real metrics (PR-AUC, NRMSE, drift...).</p>

      <div className="grid">
        {DEFAULT_QUERIES.map((q) => {
          const res = results[q.label];
          const err = errors[q.label];
          const value = res?.data?.result?.[0]?.value?.[1];
          return (
            <div className="card" key={q.label}>
              <div className="stat-label">{q.label}</div>
              {err && <div className="error-box">{err}</div>}
              {!err && <div className="stat-value">{value !== undefined ? Number(value).toFixed(3) : "…"}</div>}
            </div>
          );
        })}
      </div>

      <div className="card">
        <div className="panel-title">Monitored Targets</div>
        {!targets && <div className="empty">Loading…</div>}
        {targets && (
          <table>
            <thead>
              <tr><th>Job</th><th>Endpoint</th><th>Status</th></tr>
            </thead>
            <tbody>
              {targets.data.activeTargets.map((t, i) => (
                <tr key={i}>
                  <td>{t.labels.job}</td>
                  <td>{t.scrapeUrl}</td>
                  <td><span className={`badge ${t.health === "up" ? "FINISHED" : "FAILED"}`}>{t.health}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}