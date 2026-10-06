// Replace these two UIDs/slugs with your 2 Grafana dashboards
// (copied from the address bar when you open each dashboard,
// e.g. http://localhost:3000/d/advqdrf/dataset-drift-detected)
const DASHBOARDS = [
  { uid: "advqdrf", slug: "dataset-drift-detected", title: "Dataset Drift Detected" },
  { uid: "dcforecaster-v3", slug: "mlflow-v3-q95", title: "MLflow Monitoring v3 (Q95)" },
];

export default function GrafanaPanel() {
  return (
    <div>
      <h1 className="page-title">Grafana</h1>
      <p className="page-sub">
        Dashboards embedded in kiosk mode (without the Grafana menu around them). Requires{" "}
        <code>GF_SECURITY_ALLOW_EMBEDDING=true</code> on the grafana service.
      </p>

      {DASHBOARDS.map((d) => (
        <div key={d.uid} className="card" style={{ marginBottom: 20, padding: 12 }}>
          <div className="panel-title">{d.title}</div>
          <iframe
            className="iframe-embed"
            style={{ height: "60vh" }}
            src={`http://localhost:3000/d/${d.uid}/${d.slug}?orgId=1&theme=dark&kiosk&refresh=30s`}
            title={d.title}
          />
        </div>
      ))}
    </div>
  );
}