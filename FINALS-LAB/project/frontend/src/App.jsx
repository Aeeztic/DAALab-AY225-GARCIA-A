import { useEffect, useState } from "react";
import API from "./api";

function App() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    API.get("/stats/overview")
      .then((res) => setStats(res.data))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div>
      <h1>Placement Dashboard</h1>

      {stats ? (
        <div>
          <p>Total Students: {stats.total_students}</p>
          <p>Placement Rate: {stats.placement_rate}</p>
          <p>Avg CGPA: {stats.avg_cgpa}</p>
          <p>Avg Internships: {stats.avg_internships}</p>
        </div>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
}

export default App;
