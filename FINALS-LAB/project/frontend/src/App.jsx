import { useEffect, useState } from "react";
import StatsCards from "./components/StatsCards";
import API from "./api";

function App() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    let isMounted = true;

    const loadStats = async () => {
      try {
        if (isMounted) {
          setLoading(true);
          setError("");
        }

        const response = await API.get("/stats/overview", {
          signal: controller.signal,
        });

        if (isMounted) {
          setStats(response.data);
        }
      } catch (err) {
        if (err.name === "CanceledError" || err.code === "ERR_CANCELED") {
          return;
        }

        const message =
          err.response?.data?.detail ||
          err.message ||
          "Unable to load dashboard statistics.";

        if (isMounted) {
          setError(message);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    loadStats();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, []);

  return (
    <main className="page-shell">
      <section className="dashboard">
        <div className="dashboard__eyebrow">Academic Report</div>
        <header className="dashboard__header">
          <div>
            <h1>Placement Dashboard</h1>
            <p>Student Placement Analytics</p>
          </div>
          <div className="dashboard__badge">Live API Data</div>
        </header>

        {loading ? (
          <div className="status-panel">Loading...</div>
        ) : error ? (
          <div className="status-panel status-panel--error">
            Failed to load dashboard statistics.
            <span>{error}</span>
          </div>
        ) : (
          <StatsCards stats={stats} />
        )}
      </section>
    </main>
  );
}

export default App;
