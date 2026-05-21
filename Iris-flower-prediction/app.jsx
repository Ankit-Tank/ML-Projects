import { useState, useEffect, useMemo, useRef } from "react";
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from "recharts";

// ─────────────────────────────────────────────
// DATASET  (150 rows: sepalL, sepalW, petalL, petalW, label)
// ─────────────────────────────────────────────
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
  [5.0,3.5,1.3,0.3,0],[4.5,2.3,1.3,0.3,0],[4.4,3.2,1.3,0.2,0],[5.0,3.5,1.6,0.6,0],
  [5.1,3.8,1.9,0.4,0],[4.8,3.0,1.4,0.3,0],[5.1,3.8,1.6,0.2,0],[4.6,3.2,1.4,0.2,0],
  [5.3,3.7,1.5,0.2,0],[5.0,3.3,1.4,0.2,0],
  [7.0,3.2,4.7,1.4,1],[6.4,3.2,4.5,1.5,1],[6.9,3.1,4.9,1.5,1],[5.5,2.3,4.0,1.3,1],
  [6.5,2.8,4.6,1.5,1],[5.7,2.8,4.5,1.3,1],[6.3,3.3,4.7,1.6,1],[4.9,2.4,3.3,1.0,1],
  [6.6,2.9,4.6,1.3,1],[5.2,2.7,3.9,1.4,1],[5.0,2.0,3.5,1.0,1],[5.9,3.0,4.2,1.5,1],
  [6.0,2.2,4.0,1.0,1],[6.1,2.9,4.7,1.4,1],[5.6,2.9,3.6,1.3,1],[6.7,3.1,4.4,1.4,1],
  [5.6,3.0,4.5,1.5,1],[5.8,2.7,4.1,1.0,1],[6.2,2.2,4.5,1.5,1],[5.6,2.5,3.9,1.1,1],
  [5.9,3.2,4.8,1.8,1],[6.1,2.8,4.0,1.3,1],[6.3,2.5,4.9,1.5,1],[6.1,2.8,4.7,1.2,1],
  [6.4,2.9,4.3,1.3,1],[6.6,3.0,4.4,1.4,1],[6.8,2.8,4.8,1.4,1],[6.7,3.0,5.0,1.7,1],
  [6.0,2.9,4.5,1.5,1],[5.7,2.6,3.5,1.0,1],[5.5,2.4,3.8,1.1,1],[5.5,2.4,3.7,1.0,1],
  [5.8,2.7,3.9,1.2,1],[6.0,2.7,5.1,1.6,1],[5.4,3.0,4.5,1.5,1],[6.0,3.4,4.5,1.6,1],
  [6.7,3.1,4.7,1.5,1],[6.3,2.3,4.4,1.3,1],[5.6,3.0,4.1,1.3,1],[5.5,2.5,4.0,1.3,1],
  [5.5,2.6,4.4,1.2,1],[6.1,3.0,4.6,1.4,1],[5.8,2.6,4.0,1.2,1],[5.0,2.3,3.3,1.0,1],
  [5.6,2.7,4.2,1.3,1],[5.7,3.0,4.2,1.2,1],[5.7,2.9,4.2,1.3,1],[6.2,2.9,4.3,1.3,1],
  [5.1,2.5,3.0,1.1,1],[5.7,2.8,4.1,1.3,1],
  [6.3,3.3,6.0,2.5,2],[5.8,2.7,5.1,1.9,2],[7.1,3.0,5.9,2.1,2],[6.3,2.9,5.6,1.8,2],
  [6.5,3.0,5.8,2.2,2],[7.6,3.0,6.6,2.1,2],[4.9,2.5,4.5,1.7,2],[7.3,2.9,6.3,1.8,2],
  [6.7,2.5,5.8,1.8,2],[7.2,3.6,6.1,2.5,2],[6.5,3.2,5.1,2.0,2],[6.4,2.7,5.3,1.9,2],
  [6.8,3.0,5.5,2.1,2],[5.7,2.5,5.0,2.0,2],[5.8,2.8,5.1,2.4,2],[6.4,3.2,5.3,2.3,2],
  [6.5,3.0,5.5,1.8,2],[7.7,3.8,6.7,2.2,2],[7.7,2.6,6.9,2.3,2],[6.0,2.2,5.0,1.5,2],
  [6.9,3.2,5.7,2.3,2],[5.6,2.8,4.9,2.0,2],[7.7,2.8,6.7,2.0,2],[6.3,2.7,4.9,1.8,2],
  [6.7,3.3,5.7,2.1,2],[7.2,3.2,6.0,1.8,2],[6.2,2.8,4.8,1.8,2],[6.1,3.0,4.9,1.8,2],
  [6.4,2.8,5.6,2.1,2],[7.2,3.0,5.8,1.6,2],[7.4,2.8,6.1,1.9,2],[7.9,3.8,6.4,2.0,2],
  [6.4,2.8,5.6,2.2,2],[6.3,2.8,5.1,1.5,2],[6.1,2.6,5.6,1.4,2],[7.7,3.0,6.1,2.3,2],
  [6.3,3.4,5.6,2.4,2],[6.4,3.1,5.5,1.8,2],[6.0,3.0,4.8,1.8,2],[6.9,3.1,5.4,2.1,2],
  [6.7,3.1,5.6,2.4,2],[6.9,3.1,5.1,2.3,2],[5.8,2.7,5.1,1.9,2],[6.8,3.2,5.9,2.3,2],
  [6.7,3.3,5.7,2.5,2],[6.7,3.0,5.2,2.3,2],[6.3,2.5,5.0,1.9,2],[6.5,3.0,5.2,2.0,2],
  [6.2,3.4,5.4,2.3,2],[5.9,3.0,5.1,1.8,2],
];

// ─────────────────────────────────────────────
// ML ALGORITHMS
// ─────────────────────────────────────────────
function computeScaler(data) {
  const n = data.length, f = 4;
  const means = Array(f).fill(0);
  data.forEach(r => { for (let i = 0; i < f; i++) means[i] += r[i]; });
  means.forEach((m, i) => (means[i] = m / n));
  const stds = Array(f).fill(0);
  data.forEach(r => { for (let i = 0; i < f; i++) stds[i] += (r[i] - means[i]) ** 2; });
  stds.forEach((s, i) => (stds[i] = Math.sqrt(s / n) || 1));
  return { means, stds };
}
function scale(row, sc) { return row.map((v, i) => (v - sc.means[i]) / sc.stds[i]); }

function knnPredict(data, input, k = 5) {
  const sc = computeScaler(data);
  const sx = scale(input, sc);
  const dists = data.map(r => ({ d: Math.sqrt(r.slice(0,4).reduce((s,v,i) => s+(scale(r.slice(0,4),sc)[i]-sx[i])**2, 0)), y: r[4] }));
  dists.sort((a,b) => a.d - b.d);
  const votes = [0,0,0];
  dists.slice(0,k).forEach(({ y }) => votes[y]++);
  const total = votes.reduce((a,b) => a+b, 0);
  return { prediction: votes.indexOf(Math.max(...votes)), probabilities: votes.map(v => v/total) };
}

function buildNB(data) {
  const stats = [0,1,2].map(c => {
    const cd = data.filter(r => r[4] === c);
    const n = cd.length, prior = n / data.length;
    const means = Array(4).fill(0), vars = Array(4).fill(0);
    cd.forEach(r => { for (let i=0;i<4;i++) means[i]+=r[i]; });
    means.forEach((m,i) => (means[i]=m/n));
    cd.forEach(r => { for (let i=0;i<4;i++) vars[i]+=(r[i]-means[i])**2; });
    vars.forEach((v,i) => (vars[i]=v/n+1e-9));
    return { prior, means, vars };
  });
  return input => {
    const logP = stats.map(({ prior, means, vars }) => {
      let lp = Math.log(prior);
      for (let i=0;i<4;i++) lp += -0.5*Math.log(2*Math.PI*vars[i]) - (input[i]-means[i])**2/(2*vars[i]);
      return lp;
    });
    const mx = Math.max(...logP);
    const exps = logP.map(lp => Math.exp(lp-mx));
    const sum = exps.reduce((a,b) => a+b, 0);
    const probabilities = exps.map(p => p/sum);
    return { prediction: probabilities.indexOf(Math.max(...probabilities)), probabilities };
  };
}

function buildLR(data, epochs=600, lr=0.15) {
  const sc = computeScaler(data);
  const softmax = z => { const mx=Math.max(...z), e=z.map(v=>Math.exp(v-mx)), s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s); };
  let W = Array(3).fill(null).map(() => Array(4).fill(0).map(() => (Math.random()-0.5)*0.01));
  let b = Array(3).fill(0);
  const sd = data.map(r => ({ x: scale(r.slice(0,4), sc), y: r[4] }));
  for (let ep=0; ep<epochs; ep++) {
    const dW = Array(3).fill(null).map(() => Array(4).fill(0)), db = Array(3).fill(0);
    sd.forEach(({ x, y }) => {
      const z = W.map((w,c) => w.reduce((s,wi,i) => s+wi*x[i],0)+b[c]);
      const p = softmax(z);
      for (let c=0;c<3;c++) {
        const d = p[c]-(c===y?1:0);
        for (let i=0;i<4;i++) dW[c][i]+=d*x[i];
        db[c]+=d;
      }
    });
    const n=sd.length;
    for (let c=0;c<3;c++) { for (let i=0;i<4;i++) W[c][i]-=lr*dW[c][i]/n; b[c]-=lr*db[c]/n; }
  }
  return input => {
    const sx = scale(input, sc);
    const z = W.map((w,c) => w.reduce((s,wi,i) => s+wi*sx[i],0)+b[c]);
    const p = softmax(z);
    return { prediction: p.indexOf(Math.max(...p)), probabilities: p };
  };
}

function computeAccuracy(data, predictFn, needsScale=false) {
  const sc = needsScale ? computeScaler(data) : null;
  let correct = 0;
  data.forEach(r => {
    const inp = r.slice(0,4);
    const { prediction } = predictFn(inp);
    if (prediction === r[4]) correct++;
  });
  return (correct / data.length * 100).toFixed(1);
}

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
const SPECIES_INFO = [
  { name: "Iris setosa", emoji: "🌼", color: "#f0a0b8", bg: "#2a1520", text: "#f7d0de",
    desc: "Small, hardy species. Distinguished by short, broad petals and grows in arctic/subarctic regions.",
    hint: "Tiny petals (< 2cm), wide sepals" },
  { name: "Iris versicolor", emoji: "🌺", color: "#6ec6e8", bg: "#0f2535", text: "#c8e8f5",
    desc: "Blue flag iris. Native to eastern North America. Medium-sized with striking blue-violet blooms.",
    hint: "Medium petals (3–5cm)" },
  { name: "Iris virginica", emoji: "🌷", color: "#b48fe0", bg: "#1e1030", text: "#dfc8f8",
    desc: "Virginia iris. Largest of the three species, with elegant lavender-purple flowers.",
    hint: "Long petals (> 5cm), large overall" },
];
const FEATURES = ["Sepal Length","Sepal Width","Petal Length","Petal Width"];
const FEAT_RANGES = [[4.0,8.0],[2.0,4.5],[1.0,7.0],[0.1,2.5]];
const FEAT_DEFAULTS = [5.8, 3.0, 4.4, 1.4];
const FEAT_STEP = [0.1,0.1,0.1,0.1];
const MODEL_KEYS = ["Logistic Regression","KNN","Naive Bayes"];
const MODEL_DESC = [
  "Finds a decision boundary by optimizing a softmax linear classifier.",
  "Classifies by majority vote among the 5 nearest training examples.",
  "Uses per-class Gaussian distributions for probabilistic inference.",
];

// ─────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────
const css = `
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0a1a10; }
.iris-app {
  min-height: 100vh;
  background: radial-gradient(ellipse 120% 60% at 50% -10%, #1a3a20 0%, #0a1a10 60%);
  color: #d4e8d0;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 15px;
  padding: 0 0 60px;
}
.hero {
  padding: 48px 32px 36px;
  text-align: center;
  border-bottom: 1px solid rgba(100,180,120,0.12);
  background: linear-gradient(180deg, rgba(20,60,25,0.6) 0%, transparent 100%);
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(80,180,100,0.12); border: 1px solid rgba(80,180,100,0.25);
  border-radius: 100px; padding: 5px 16px;
  font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
  color: #7ecf95; margin-bottom: 20px;
}
.hero-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: clamp(32px, 5vw, 52px); font-weight: 600;
  color: #e8f5e4; line-height: 1.15; margin-bottom: 12px;
  letter-spacing: -0.02em;
}
.hero-title em { font-style: italic; color: #7ecf95; }
.hero-sub { color: #7aab82; font-size: 15px; font-weight: 300; max-width: 500px; margin: 0 auto; line-height: 1.7; }

.content { max-width: 1100px; margin: 0 auto; padding: 40px 24px 0; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 760px) { .grid-2 { grid-template-columns: 1fr; } }
.grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 32px; }
@media (max-width: 760px) { .grid-3 { grid-template-columns: 1fr; } }

.card {
  background: rgba(18,35,20,0.7);
  border: 1px solid rgba(100,170,110,0.14);
  border-radius: 20px; padding: 28px;
  backdrop-filter: blur(8px);
}
.card-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 17px; font-weight: 600; color: #c8e8c0; margin-bottom: 20px;
  display: flex; align-items: center; gap: 10px;
}
.card-title svg { opacity: 0.7; }

.section-label {
  font-size: 10px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase;
  color: #4d8c5a; margin-bottom: 24px;
  display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: rgba(77,140,90,0.25); }

/* Model selector */
.model-tabs { display: flex; gap: 8px; margin-bottom: 28px; flex-wrap: wrap; }
.model-tab {
  flex: 1; min-width: 120px; padding: 10px 14px;
  background: rgba(10,26,14,0.8); border: 1px solid rgba(100,170,110,0.18);
  border-radius: 12px; cursor: pointer; transition: all 0.2s;
  color: #6a9e74; font-family: 'DM Sans', sans-serif; font-size: 13px;
  font-weight: 400; text-align: center;
}
.model-tab:hover { border-color: rgba(100,200,120,0.35); color: #9ecfa8; }
.model-tab.active {
  background: rgba(30,90,40,0.5); border-color: rgba(80,200,100,0.5);
  color: #7ee89a; font-weight: 500;
  box-shadow: 0 0 20px rgba(60,160,80,0.15) inset, 0 0 20px rgba(60,160,80,0.08);
}

/* Sliders */
.slider-wrap { margin-bottom: 22px; }
.slider-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
.slider-label { font-size: 13px; color: #7aab82; font-weight: 400; }
.slider-val {
  font-size: 18px; font-weight: 500; color: #b8e0c0;
  font-family: 'DM Sans', sans-serif; letter-spacing: -0.03em;
}
.slider-unit { font-size: 11px; color: #4d8c5a; margin-left: 2px; }
.slider-track { position: relative; height: 6px; }
.range-input {
  -webkit-appearance: none; appearance: none;
  width: 100%; height: 6px; border-radius: 3px;
  background: rgba(40,80,45,0.8);
  outline: none; cursor: pointer; position: relative;
}
.range-input::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 20px; height: 20px; border-radius: 50%;
  background: #3cb86a; border: 3px solid rgba(10,26,14,0.8);
  box-shadow: 0 0 12px rgba(60,184,106,0.5); cursor: pointer;
  transition: transform 0.15s, box-shadow 0.15s;
}
.range-input::-webkit-slider-thumb:hover {
  transform: scale(1.2); box-shadow: 0 0 18px rgba(60,184,106,0.7);
}
.range-input::-moz-range-thumb {
  width: 20px; height: 20px; border-radius: 50%;
  background: #3cb86a; border: 3px solid rgba(10,26,14,0.8);
  box-shadow: 0 0 12px rgba(60,184,106,0.5); cursor: pointer;
}

/* Prediction card */
.pred-card {
  border-radius: 20px; padding: 32px;
  position: relative; overflow: hidden;
  min-height: 200px;
  transition: all 0.4s ease;
}
.pred-card::before {
  content: ''; position: absolute; top: -60px; right: -60px;
  width: 200px; height: 200px; border-radius: 50%;
  opacity: 0.06;
  background: currentColor;
}
.pred-species-emoji { font-size: 52px; margin-bottom: 8px; line-height: 1; }
.pred-label { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.6; margin-bottom: 6px; }
.pred-name {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 28px; font-weight: 600; line-height: 1.2; margin-bottom: 16px;
}
.pred-name em { font-style: italic; display: block; font-size: 20px; opacity: 0.8; }
.confidence-bar-wrap { margin-bottom: 6px; }
.conf-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; opacity: 0.8; }
.conf-bar {
  height: 8px; border-radius: 4px; background: rgba(255,255,255,0.08);
  overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 4px; transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1); }

/* Prob bars */
.prob-section { margin-top: 20px; }
.prob-row { margin-bottom: 14px; }
.prob-row-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 13px; }
.prob-bar-bg { height: 10px; background: rgba(255,255,255,0.06); border-radius: 5px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 5px; transition: width 0.5s cubic-bezier(0.34,1.56,0.64,1); }

/* Accuracy cards */
.acc-card {
  background: rgba(12,28,15,0.8);
  border: 1px solid rgba(100,170,110,0.14);
  border-radius: 16px; padding: 20px 22px;
  transition: border-color 0.2s;
}
.acc-card.selected { border-color: rgba(80,200,100,0.4); background: rgba(20,50,25,0.6); }
.acc-num { font-size: 30px; font-weight: 500; color: #7ee89a; letter-spacing: -0.04em; margin: 8px 0 4px; font-family: 'DM Sans', sans-serif; }
.acc-label { font-size: 11px; color: #4d8c5a; text-transform: uppercase; letter-spacing: 0.1em; }
.acc-model { font-size: 14px; color: #90c89a; font-weight: 500; margin-bottom: 2px; }
.acc-badge { display: inline-block; padding: 2px 10px; border-radius: 100px; font-size: 11px; font-weight: 500; }

/* Species cards */
.species-cards { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-top: 32px; }
@media (max-width: 760px) { .species-cards { grid-template-columns: 1fr; } }
.sp-card { border-radius: 16px; padding: 22px; border: 1px solid transparent; transition: all 0.3s; }
.sp-card:hover { transform: translateY(-3px); }
.sp-emoji { font-size: 28px; margin-bottom: 10px; }
.sp-name { font-family: 'Playfair Display', Georgia, serif; font-size: 16px; font-weight: 600; margin-bottom: 6px; }
.sp-hint { font-size: 11px; opacity: 0.65; margin-bottom: 10px; letter-spacing: 0.05em; }
.sp-desc { font-size: 13px; line-height: 1.65; opacity: 0.8; font-weight: 300; }

/* Radar */
.radar-wrap { height: 260px; }

/* Footer */
.footer { text-align: center; margin-top: 60px; padding-top: 32px; border-top: 1px solid rgba(100,170,110,0.1); color: #3a6040; font-size: 12px; letter-spacing: 0.05em; }

/* Animate in */
@keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
.anim { animation: fadeUp 0.5s ease both; }
.anim-1 { animation-delay: 0.05s; }
.anim-2 { animation-delay: 0.12s; }
.anim-3 { animation-delay: 0.19s; }
.anim-4 { animation-delay: 0.26s; }
`;

// ─────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────
export default function IrisPredictor() {
  const [model, setModel] = useState("Logistic Regression");
  const [vals, setVals] = useState(FEAT_DEFAULTS);
  const [trained, setTrained] = useState(null);
  const [accuracies, setAccuracies] = useState({});

  // Train once on mount
  useEffect(() => {
    const nbFn = buildNB(DATA);
    const lrFn = buildLR(DATA);
    const knnFn = (inp) => knnPredict(DATA, inp);

    const acc = {
      "Logistic Regression": computeAccuracy(DATA, lrFn),
      "KNN": computeAccuracy(DATA, knnFn),
      "Naive Bayes": computeAccuracy(DATA, nbFn),
    };

    setTrained({ lr: lrFn, knn: knnFn, nb: nbFn });
    setAccuracies(acc);
  }, []);

  // Predict
  const result = useMemo(() => {
    if (!trained) return null;
    const inp = vals;
    if (model === "Logistic Regression") return trained.lr(inp);
    if (model === "KNN") return trained.knn(inp);
    return trained.nb(inp);
  }, [trained, model, vals]);

  const sp = result ? SPECIES_INFO[result.prediction] : SPECIES_INFO[1];

  // Radar data (normalized 0-100)
  const radarData = FEATURES.map((f, i) => ({
    feature: f.replace(" ", "\n"),
    value: Math.round(((vals[i] - FEAT_RANGES[i][0]) / (FEAT_RANGES[i][1] - FEAT_RANGES[i][0])) * 100),
    fullMark: 100,
  }));

  const setosaMeans = [4.9, 3.4, 1.5, 0.2];
  const versMeans = [5.9, 2.8, 4.3, 1.3];
  const virgMeans = [6.6, 2.9, 5.6, 2.0];

  const comparisonRadar = FEATURES.map((f, i) => ({
    feature: f.split(" ")[0],
    setosa: Math.round(((setosaMeans[i]-FEAT_RANGES[i][0])/(FEAT_RANGES[i][1]-FEAT_RANGES[i][0]))*100),
    versicolor: Math.round(((versMeans[i]-FEAT_RANGES[i][0])/(FEAT_RANGES[i][1]-FEAT_RANGES[i][0]))*100),
    virginica: Math.round(((virgMeans[i]-FEAT_RANGES[i][0])/(FEAT_RANGES[i][1]-FEAT_RANGES[i][0]))*100),
  }));

  return (
    <>
      <style>{css}</style>
      <div className="iris-app">
        {/* HERO */}
        <div className="hero anim">
          <div className="hero-badge">
            <span>🌿</span> Supervised Machine Learning
          </div>
          <h1 className="hero-title">
            Iris Flower <em>Species Predictor</em>
          </h1>
          <p className="hero-sub">
            Identify iris species from petal and sepal measurements using three
            classical machine learning algorithms trained on 150 botanical samples.
          </p>
        </div>

        <div className="content">
          {/* ACCURACY SUMMARY */}
          <p className="section-label anim anim-1">Model Performance</p>
          <div className="grid-3 anim anim-2">
            {MODEL_KEYS.map((m, i) => (
              <div
                key={m}
                className={`acc-card${model === m ? " selected" : ""}`}
                style={{ cursor: "pointer" }}
                onClick={() => setModel(m)}
              >
                <div className="acc-model">{m}</div>
                <div className="acc-num">{accuracies[m] ?? "—"}%</div>
                <div className="acc-label">Accuracy</div>
                <div style={{ marginTop: 10, fontSize: 12, color: "#4d8c5a", lineHeight: 1.5, fontWeight: 300 }}>
                  {MODEL_DESC[i]}
                </div>
              </div>
            ))}
          </div>

          {/* MAIN GRID */}
          <div className="grid-2 anim anim-3">
            {/* LEFT: Controls */}
            <div className="card">
              <div className="card-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7ecf95" strokeWidth="2" strokeLinecap="round">
                  <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"/><path d="M12 8v4l3 3"/>
                </svg>
                Flower Measurements
              </div>

              {/* Model tabs */}
              <div style={{ marginBottom: 8, fontSize: 12, color: "#4d8c5a" }}>Select model</div>
              <div className="model-tabs">
                {MODEL_KEYS.map(m => (
                  <button key={m} className={`model-tab${model===m?" active":""}`} onClick={() => setModel(m)}>
                    {m}
                  </button>
                ))}
              </div>

              {/* Sliders */}
              {FEATURES.map((f, i) => {
                const [min, max] = FEAT_RANGES[i];
                const pct = ((vals[i] - min) / (max - min)) * 100;
                const fillStyle = {
                  background: `linear-gradient(to right, #3cb86a ${pct}%, rgba(40,80,45,0.8) ${pct}%)`
                };
                return (
                  <div className="slider-wrap" key={f}>
                    <div className="slider-header">
                      <span className="slider-label">{f}</span>
                      <span className="slider-val">
                        {vals[i].toFixed(1)}<span className="slider-unit">cm</span>
                      </span>
                    </div>
                    <input
                      type="range" className="range-input"
                      min={min} max={max} step={FEAT_STEP[i]} value={vals[i]}
                      style={fillStyle}
                      onChange={e => {
                        const nv = [...vals];
                        nv[i] = parseFloat(e.target.value);
                        setVals(nv);
                      }}
                    />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#3a5e40", marginTop: 4 }}>
                      <span>{min}</span><span>{max}</span>
                    </div>
                  </div>
                );
              })}

              {/* Radar chart of input */}
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 12, color: "#4d8c5a", marginBottom: 4 }}>Measurement profile</div>
                <div className="radar-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="rgba(100,180,110,0.18)" />
                      <PolarAngleAxis dataKey="feature" tick={{ fill: "#6a9e74", fontSize: 11 }} />
                      <PolarRadiusAxis angle={30} domain={[0,100]} tick={false} axisLine={false} />
                      <Radar name="Input" dataKey="value" stroke="#3cb86a" fill="#3cb86a" fillOpacity={0.2} strokeWidth={2} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* RIGHT: Prediction */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {/* Prediction result */}
              <div
                className="pred-card"
                style={{
                  background: sp.bg,
                  border: `1px solid ${sp.color}30`,
                  color: sp.text,
                }}
              >
                <div className="pred-species-emoji">{sp.emoji}</div>
                <div className="pred-label">Predicted Species</div>
                <div className="pred-name">
                  {sp.name.split(" ")[0]}
                  <em>{sp.name.split(" ")[1]}</em>
                </div>

                {/* Confidence */}
                {result && (
                  <div className="confidence-bar-wrap">
                    <div className="conf-header">
                      <span>Confidence</span>
                      <span style={{ fontWeight: 500 }}>{(result.probabilities[result.prediction]*100).toFixed(1)}%</span>
                    </div>
                    <div className="conf-bar">
                      <div
                        className="conf-fill"
                        style={{
                          width: `${(result.probabilities[result.prediction]*100).toFixed(1)}%`,
                          background: sp.color,
                        }}
                      />
                    </div>
                  </div>
                )}

                <div style={{ marginTop: 16, padding: "12px 16px", background: "rgba(0,0,0,0.2)", borderRadius: 12, fontSize: 13, lineHeight: 1.6, fontWeight: 300 }}>
                  {sp.desc}
                </div>
              </div>

              {/* Probability breakdown */}
              <div className="card" style={{ flex: 1 }}>
                <div className="card-title">Class Probabilities</div>
                {result && SPECIES_INFO.map((s, i) => (
                  <div className="prob-row" key={s.name}>
                    <div className="prob-row-header">
                      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 16 }}>{s.emoji}</span>
                        <span style={{ color: "#9ecba4", fontSize: 13 }}>{s.name}</span>
                      </span>
                      <span style={{ fontSize: 14, fontWeight: 500, color: i === result.prediction ? s.color : "#5a8a64" }}>
                        {(result.probabilities[i]*100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="prob-bar-bg">
                      <div
                        className="prob-bar-fill"
                        style={{
                          width: `${(result.probabilities[i]*100).toFixed(1)}%`,
                          background: i === result.prediction ? s.color : `${s.color}55`,
                        }}
                      />
                    </div>
                  </div>
                ))}

                {/* Species comparison radar */}
                <div style={{ marginTop: 20 }}>
                  <div style={{ fontSize: 12, color: "#4d8c5a", marginBottom: 4 }}>Species average profiles</div>
                  <div className="radar-wrap">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={comparisonRadar}>
                        <PolarGrid stroke="rgba(100,180,110,0.15)" />
                        <PolarAngleAxis dataKey="feature" tick={{ fill: "#6a9e74", fontSize: 11 }} />
                        <PolarRadiusAxis angle={30} domain={[0,100]} tick={false} axisLine={false} />
                        <Radar name="Setosa" dataKey="setosa" stroke="#f0a0b8" fill="#f0a0b8" fillOpacity={0.12} strokeWidth={1.5} />
                        <Radar name="Versicolor" dataKey="versicolor" stroke="#6ec6e8" fill="#6ec6e8" fillOpacity={0.12} strokeWidth={1.5} />
                        <Radar name="Virginica" dataKey="virginica" stroke="#b48fe0" fill="#b48fe0" fillOpacity={0.12} strokeWidth={1.5} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: "flex", gap: 16, justifyContent: "center", fontSize: 12, color: "#5a8a64" }}>
                    {SPECIES_INFO.map(s => (
                      <span key={s.name} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                        <span style={{ width: 10, height: 2, background: s.color, display: "inline-block", borderRadius: 1 }} />
                        {s.name.split(" ")[1]}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* SPECIES INFORMATION */}
          <p className="section-label" style={{ marginTop: 48 }}>Species Guide</p>
          <div className="species-cards anim anim-4">
            {SPECIES_INFO.map((s, i) => (
              <div
                key={s.name}
                className="sp-card"
                style={{ background: s.bg, borderColor: `${s.color}25`, color: s.text }}
              >
                <div className="sp-emoji">{s.emoji}</div>
                <div className="sp-name" style={{ color: s.color }}>{s.name}</div>
                <div className="sp-hint">{s.hint}</div>
                <div className="sp-desc">{s.desc}</div>
                <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {[["Sepal L", setosaMeans[0], versMeans[0], virgMeans[0]],
                    ["Petal L", setosaMeans[2], versMeans[2], virgMeans[2]]].map(([lbl, ...means]) => (
                    <div key={lbl} style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "6px 10px", fontSize: 12 }}>
                      <span style={{ opacity: 0.6 }}>{lbl} avg: </span>
                      <span style={{ fontWeight: 500, color: s.color }}>{means[i].toFixed(1)} cm</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* DATASET INFO */}
          <div className="card" style={{ marginTop: 24 }}>
            <div className="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7ecf95" strokeWidth="2" strokeLinecap="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/>
              </svg>
              Dataset & Methodology
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16, marginBottom: 20 }}>
              {[["150","Total Samples"], ["3","Species Classes"], ["4","Feature Dimensions"], ["50","Samples / Class"]].map(([n, l]) => (
                <div key={l} style={{ background: "rgba(10,26,14,0.6)", borderRadius: 12, padding: "14px 18px", border: "1px solid rgba(100,170,110,0.1)" }}>
                  <div style={{ fontSize: 24, fontWeight: 500, color: "#7ee89a", letterSpacing: "-0.04em" }}>{n}</div>
                  <div style={{ fontSize: 12, color: "#4d8c5a", marginTop: 4 }}>{l}</div>
                </div>
              ))}
            </div>
            <p style={{ fontSize: 13, lineHeight: 1.8, color: "#6a9e74", fontWeight: 300 }}>
              The classic Fisher Iris dataset contains measurements for sepal length, sepal width, petal length, and petal width
              across three species. All three ML algorithms are trained directly in-browser on the full 150-sample dataset.
              Models are retrained on every page load. The Iris dataset is a benchmark — near-perfect accuracy reflects
              its clean, well-balanced structure, not necessarily what to expect from real-world botanical data.
            </p>
          </div>

          <div className="footer">
            🌿 Iris Flower Predictor · Logistic Regression · KNN · Naive Bayes · 150 samples · Fisher 1936
          </div>
        </div>
      </div>
    </>
  );
}