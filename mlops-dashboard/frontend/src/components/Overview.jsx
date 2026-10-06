import { useEffect, useState } from "react";
import { api } from "../api";

function StatusCard({ label, promise }) {
  const [state, setState] = useState("loading");

  useEffect(() => {
    promise().then(() => setState("ok")).catch(() => setState("down"));
  }, []);

  return (
    <div className="card">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${state === "ok" ? "ok" : state === "down" ? "danger" : ""}`}>
        {state === "loading" ? "…" : state === "ok" ? "active" : "unreachable"}
      </div>
    </div>
  );
}

export default function Overview() {
  return (
    <div>
      <h1 className="page-title">Overview</h1>
      <p className="page-sub">Live status of the DCForecaster MLOps stack services.</p>
      <div className="grid">
        <StatusCard label="MLflow" promise={() => api.mlflowRuns(1)} />
        <StatusCard label="Prometheus" promise={() => api.prometheusTargets()} />
        <StatusCard label="MinIO" promise={() => api.minioArtifacts()} />
        <StatusCard label="Drift Reports" promise={() => api.driftReports()} />
      </div>
      <div className="card">
        <div className="panel-title">Shortcuts</div>
        <p className="page-sub" style={{margin:0}}>
          Use the menu on the left to explore each source in detail: MLflow runs,
          Prometheus metrics, Evidently drift reports, MinIO artifacts, and Grafana.
        </p>
      </div>
    </div>
  );
}