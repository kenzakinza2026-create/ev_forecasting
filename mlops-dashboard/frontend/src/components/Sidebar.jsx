import logo from "../assets/larina-logo.png";

const ITEMS = [
  { id: "overview", label: "Overview" },
  { id: "mlflow", label: "MLflow · Runs" },
  { id: "prometheus", label: "Prometheus · Metrics" },
  { id: "drift", label: "Evidently · Drift" },
  { id: "minio", label: "MinIO · Artifacts" },
  { id: "grafana", label: "Grafana" },
];

export default function Sidebar({ page, setPage }) {
  return (
    <div className="sidebar">
      <div className="brand">
        <img src={logo} alt="LaRiNa" />
        <small>ops console</small>
      </div>
      {ITEMS.map((item) => (
        <div
          key={item.id}
          className={`nav-item ${page === item.id ? "active" : ""}`}
          onClick={() => setPage(item.id)}
        >
          <span className="nav-dot" />
          {item.label}
        </div>
      ))}
    </div>
  );
}