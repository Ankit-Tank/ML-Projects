import { useState, useEffect, useMemo } from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer
} from "recharts";

/* ─────────────────────────────────────────────
   DATASET  (150 rows: sepalL, sepalW, petalL, petalW, label)
──────────────────────────────────────────── */

const DATA = [
  [5.1,3.5,1.4,0.2,0],[4.9,3.0,1.4,0.2,0],[4.7,3.2,1.3,0.2,0],[4.6,3.1,1.5,0.2,0],
  [5.0,3.6,1.4,0.2,0],[5.4,3.9,1.7,0.4,0],[4.6,3.4,1.4,0.3,0],[5.0,3.4,1.5,0.2,0],
  [4.4,2.9,1.4,0.2,0],[4.9,3.1,1.5,0.1,0],[5.4,3.7,1.5,0.2,0],[4.8,3.4,1.6,0.2,0],
  [4.8,3.0,1.4,0.1,0],[4.3,3.0,1.1,0.1,0],[5.8,4.0,1.2,0.2,0],[5.7,4.4,1.5,0.4,0],
  [5.4,3.9,1.3,0.4,0],[5.1,3.5,1.4,0.3,0],[5.7,3.8,1.7,0.3,0],[5.1,3.8,1.5,0.3,0],
  [5.4,3.4,1.7,0.2,0],[5.1,3.7,1.5,0.4,0],[4.6,3.6,1.0,0.2,0],[5.1,3.3,1.7,0.5,0],
  [4.8,3.4,1.9,0.2,0],[5.0,3.0,1.6,0.2,0],[5.0,3.4,1.6,0.4,0],[5.2,3.5,1.5,0.2,0],
  [5.2,3.4,1.4,0.2,0],[4.7,3.2,1.6,0.2,0],[4.8,3.1,1.6,0.2,0],[5.4,3.4,1.5,0.4,0],
  [5.2,4.1,1.5,0.1,0],[5.5,4.2,1.4,0.2,0],[4.9,3.1,1.5,0.1,0],[5.0,3.2,1.2,0.2,0],
  [5.5,3.5,1.3,0.2,0],[4.9,3.1,1.5,0.1,0],[4.4,3.0,1.3,0.2,0],[5.1,3.4,1.5,0.2,0],
];

/* ─────────────────────────────────────────────
   ML ALGORITHMS
──────────────────────────────────────────── */

function computeScaler(data) {
  const n = data.length, f = 4;
  const means = Array(f).fill(0);

  data.forEach(r => {
    for (let i = 0; i < f; i++) means[i] += r[i];
  });

  means.forEach((m, i) => (means[i] = m / n));

  const stds = Array(f).fill(0);

  data.forEach(r => {
    for (let i = 0; i < f; i++) {
      stds[i] += (r[i] - means[i]) ** 2;
    }
  });

  stds.forEach((s, i) => (stds[i] = Math.sqrt(s / n) || 1));

  return { means, stds };
}

function scale(row, sc) {
  return row.map((v, i) => (v - sc.means[i]) / sc.stds[i]);
}

function knnPredict(data, input, k = 5) {
  const sc = computeScaler(data);
  const sx = scale(input, sc);

  const dists = data.map(r => ({
    d: Math.sqrt(
      r.slice(0, 4).reduce(
        (s, v, i) =>
          s + (scale(r.slice(0, 4), sc)[i] - sx[i]) ** 2,
        0
      )
    ),
    y: r[4]
  }));

  dists.sort((a, b) => a.d - b.d);

  const votes = [0, 0, 0];

  dists.slice(0, k).forEach(({ y }) => votes[y]++);

  const total = votes.reduce((a, b) => a + b, 0);

  return {
    prediction: votes.indexOf(Math.max(...votes)),
    probabilities: votes.map(v => v / total)
  };
}

/* ─────────────────────────────────────────────
   CONSTANTS
──────────────────────────────────────────── */

const FEATURES = [
  "Sepal Length",
  "Sepal Width",
  "Petal Length",
  "Petal Width"
];

const FEAT_RANGES = [
  [4.0, 8.0],
  [2.0, 4.5],
  [1.0, 7.0],
  [0.1, 2.5]
];

const FEAT_DEFAULTS = [5.8, 3.0, 4.4, 1.4];

/* ─────────────────────────────────────────────
   STYLES
──────────────────────────────────────────── */

const css = `
body {
  background: #0a1a10;
  color: white;
  font-family: sans-serif;
}

.iris-app {
  min-height: 100vh;
  padding: 40px;
}
`;

/* ─────────────────────────────────────────────
   COMPONENT
──────────────────────────────────────────── */

export default function IrisPredictor() {
  const [vals, setVals] = useState(FEAT_DEFAULTS);

  const result = useMemo(() => {
    return knnPredict(DATA, vals);
  }, [vals]);

  const radarData = FEATURES.map((f, i) => ({
    feature: f,
    value:
      ((vals[i] - FEAT_RANGES[i][0]) /
        (FEAT_RANGES[i][1] - FEAT_RANGES[i][0])) *
      100,
    fullMark: 100
  }));

  return (
    <>
      <style>{css}</style>

      <div className="iris-app">
        <h1>Iris Predictor</h1>

        {FEATURES.map((f, i) => (
          <div key={f}>
            <label>{f}</label>

            <input
              type="range"
              min={FEAT_RANGES[i][0]}
              max={FEAT_RANGES[i][1]}
              step="0.1"
              value={vals[i]}
              onChange={e => {
                const nv = [...vals];
                nv[i] = parseFloat(e.target.value);
                setVals(nv);
              }}
            />

            <span>{vals[i]}</span>
          </div>
        ))}

        <div style={{ width: 400, height: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="feature" />
              <PolarRadiusAxis />
              <Radar
                dataKey="value"
                stroke="#3cb86a"
                fill="#3cb86a"
                fillOpacity={0.4}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <h2>Prediction: {result.prediction}</h2>
      </div>
    </>
  );
}