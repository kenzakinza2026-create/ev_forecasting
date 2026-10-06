export default function MlflowPanel() {
  return (
    <div>
      <h1 className="page-title">MLflow</h1>
      <p className="page-sub">
        MLflow 
      </p>

      <iframe
        className="iframe-embed"
        style={{ height: "80vh", background: "#fff" }}
        src="http://localhost:5000"
        title="MLflow"
      />
    </div>
  );
}