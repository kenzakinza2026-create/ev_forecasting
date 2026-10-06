import { useEffect, useState } from "react";
import { api } from "../api";

export default function MinioPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.minioArtifacts().then(setData).catch((e) => setError(e.message));
  }, []);

  return (
    <div>
      <h1 className="page-title">MinIO · Artifacts</h1>
      <p className="page-sub">Contents of the <code>mlflow-artifacts</code> bucket (models, reports, files logged by MLflow).</p>

      {error && <div className="error-box">Could not reach MinIO: {error}</div>}
      {!error && !data && <div className="empty">Loading…</div>}
      {data && data.count === 0 && <div className="empty">Bucket is empty.</div>}

      {data && data.count > 0 && (
        <div className="card">
          <table>
            <thead>
              <tr><th>Key</th><th>Size</th><th>Modified</th></tr>
            </thead>
            <tbody>
              {data.objects.map((o) => (
                <tr key={o.key}>
                  <td>{o.key}</td>
                  <td>{o.size_kb} KB</td>
                  <td>{new Date(o.last_modified).toLocaleString("en-US")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <a className="link-out" href="http://localhost:9001" target="_blank" rel="noreferrer">
        Open full MinIO console →
      </a>
    </div>
  );
}