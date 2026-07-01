/**
 * Home.jsx – Landing page for VeriNews AI.
 * Calls /health and displays the backend status.
 */
import React from "react";
import { useHealth } from "../hooks/useHealth";
import StatusBadge from "../components/StatusBadge";
import "./Home.css";

export default function Home() {
  const { data, loading, error } = useHealth();

  return (
    <main className="home">
      {/* ── Background decoration ──────────────────────────────────────── */}
      <div className="home__orb home__orb--1" aria-hidden="true" />
      <div className="home__orb home__orb--2" aria-hidden="true" />
      <div className="home__orb home__orb--3" aria-hidden="true" />

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section className="home__hero">
        <div className="home__badge-row">
          <span className="home__pill">🧠 Powered by AI</span>
        </div>

        <h1 className="home__title">
          Veri<span className="home__title--accent">News</span>&nbsp;AI
        </h1>

        <p className="home__subtitle">
          Next-generation fake-news detection engine — cut through misinformation
          with state-of-the-art natural-language understanding.
        </p>

        {/* Backend status */}
        <div className="home__status" aria-label="Backend status">
          <StatusBadge
            status={data?.status}
            message={data?.message}
            loading={loading}
            error={error}
          />
        </div>

        {/* CTA buttons */}
        <div className="home__cta">
          <button className="btn btn--primary" id="btn-analyze" disabled>
            Analyse Article
          </button>
          <button className="btn btn--ghost" id="btn-learn">
            Learn More ↓
          </button>
        </div>
      </section>

      {/* ── Feature cards ─────────────────────────────────────────────────── */}
      <section className="home__features" aria-label="Features">
        {FEATURES.map((f) => (
          <article className="feature-card" key={f.title}>
            <span className="feature-card__icon" aria-hidden="true">{f.icon}</span>
            <h2 className="feature-card__title">{f.title}</h2>
            <p className="feature-card__desc">{f.desc}</p>
          </article>
        ))}
      </section>
    </main>
  );
}

const FEATURES = [
  {
    icon: "🔍",
    title: "Deep Text Analysis",
    desc: "Transformer-based NLP models dissect every sentence for statistical and semantic patterns linked to misinformation.",
  },
  {
    icon: "⚡",
    title: "Real-time Results",
    desc: "Sub-second inference via optimised FastAPI endpoints — paste a headline and get an answer instantly.",
  },
  {
    icon: "📊",
    title: "Confidence Scoring",
    desc: "Every prediction comes with a calibrated probability score so you can trust the output, not just the label.",
  },
  {
    icon: "🔗",
    title: "Source Verification",
    desc: "Cross-reference claims against a curated knowledge graph of trusted publishers and fact-checkers.",
  },
];
