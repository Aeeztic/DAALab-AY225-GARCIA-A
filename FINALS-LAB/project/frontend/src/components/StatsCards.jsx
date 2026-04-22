function formatPlacementRate(value) {
  const numericValue = Number(value ?? 0);
  const percentage = numericValue <= 1 ? numericValue * 100 : numericValue;

  return `${percentage.toFixed(2)}%`;
}

function formatCgpa(value) {
  return Number(value ?? 0).toFixed(2);
}

function formatCount(value) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(Number(value ?? 0));
}

function formatInternships(value) {
  return Number(value ?? 0).toFixed(2);
}

function StatsCards({ stats }) {
  const cards = [
    {
      label: "Total Students",
      value: formatCount(stats?.total_students),
      note: "Students included in the processed dataset",
    },
    {
      label: "Placement Rate",
      value: formatPlacementRate(stats?.placement_rate),
      note: "Share of students marked as placed",
    },
    {
      label: "Avg CGPA",
      value: formatCgpa(stats?.avg_cgpa),
      note: "Average cumulative grade point average",
    },
    {
      label: "Avg Internships",
      value: formatInternships(stats?.avg_internships),
      note: "Mean internship count per student",
    },
  ];

  return (
    <section className="stats-grid" aria-label="Placement statistics">
      {cards.map((card) => (
        <article className="stat-card" key={card.label}>
          <p className="stat-card__label">{card.label}</p>
          <h2 className="stat-card__value">{card.value}</h2>
          <p className="stat-card__note">{card.note}</p>
        </article>
      ))}
    </section>
  );
}

export default StatsCards;
