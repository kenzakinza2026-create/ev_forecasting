import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Overview from "./components/Overview";
import MlflowPanel from "./components/MlflowPanel";
import PrometheusPanel from "./components/PrometheusPanel";
import DriftPanel from "./components/DriftPanel";
import MinioPanel from "./components/MinioPanel";
import GrafanaPanel from "./components/GrafanaPanel";

const PAGES = {
  overview: Overview,
  mlflow: MlflowPanel,
  prometheus: PrometheusPanel,
  drift: DriftPanel,
  minio: MinioPanel,
  grafana: GrafanaPanel,
};

export default function App() {
  const [page, setPage] = useState("overview");
  const Page = PAGES[page];

  return (
    <div className="layout">
      <Sidebar page={page} setPage={setPage} />
      <div className="main">
        <Page />
      </div>
    </div>
  );
}
