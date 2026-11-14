#!/usr/bin/env python3
"""Live 3D Parameter Space & Loss Observatory for MLX LoRA Fine-Tuning.

Reads telemetry emitted by train_visual.py (under ./runs/<run_id>/) and renders:
1. Interactive 3D parameter-space trajectory relative to the frozen regular model
   at origin (0, 0, 0), including Attention/MLP trails and 36-layer constellation.
2. Live Train & Validation loss curve graph with hover inspection and scrubbing.
3. Layer x Module heatmap comparing every LoRA weight shift against the regular model.
4. Top shifting modules & Attention vs MLP parameter-movement breakdown.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LoRA Parameter Space & Loss Observatory</title>
  <style>
    :root {
      --bg: #060911;
      --bg2: #0c1220;
      --panel: rgba(15, 22, 38, 0.92);
      --panel2: rgba(21, 31, 52, 0.92);
      --border: #1e2d4a;
      --text: #e6edf8;
      --muted: #8496b8;
      --cyan: #38bdf8;
      --emerald: #34d399;
      --amber: #fbbf24;
      --coral: #fb7185;
      --violet: #a78bfa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: #060914;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.45;
      overflow-x: hidden;
    }
    body.embedded {
      background: transparent;
    }
    body.embedded .bg-aurora {
      display: none;
    }
    body.embedded .wrap {
      padding-top: 4px;
      padding-bottom: 24px;
    }
    body.embedded header {
      margin-bottom: 10px;
    }
    .bg-aurora {
      position: fixed;
      inset: -18%;
      z-index: 0;
      pointer-events: none;
      background:
        radial-gradient(46% 42% at 20% 20%, rgba(56, 189, 248, 0.25), transparent 70%),
        radial-gradient(45% 45% at 80% 22%, rgba(167, 139, 250, 0.27), transparent 70%),
        radial-gradient(48% 44% at 54% 84%, rgba(59, 130, 246, 0.22), transparent 72%),
        radial-gradient(38% 36% at 15% 78%, rgba(168, 85, 247, 0.19), transparent 70%);
      filter: blur(42px) saturate(1.2);
      animation: auroraDrift 20s ease-in-out infinite alternate;
      will-change: transform;
    }
    .bg-aurora::after {
      content: "";
      position: absolute;
      inset: 8%;
      background:
        radial-gradient(38% 36% at 70% 35%, rgba(34, 211, 238, 0.16), transparent 68%),
        radial-gradient(40% 38% at 28% 68%, rgba(124, 58, 237, 0.19), transparent 68%);
      animation: auroraCounter 26s ease-in-out infinite alternate;
    }
    @keyframes auroraDrift {
      0% { transform: translate3d(-2%, -1.5%, 0) scale(1) rotate(0deg); }
      50% { transform: translate3d(2.5%, 1.5%, 0) scale(1.06) rotate(3deg); }
      100% { transform: translate3d(-1.5%, 2.5%, 0) scale(1.03) rotate(-2deg); }
    }
    @keyframes auroraCounter {
      0% { transform: translate3d(2%, 1%, 0) scale(1.02); }
      100% { transform: translate3d(-3%, -2%, 0) scale(1.08); }
    }
    @media (prefers-reduced-motion: reduce) {
      .bg-aurora, .bg-aurora::after { animation: none; }
    }
    .wrap {
      position: relative;
      z-index: 1;
      max-width: 1540px;
      margin: 0 auto;
      padding: 18px 20px 34px;
    }
    .guide-btn {
      background: rgba(56, 189, 248, 0.14);
      color: #bae6fd;
      border-color: rgba(56, 189, 248, 0.36);
    }
    .guide-btn:hover { background: rgba(56, 189, 248, 0.24); }
    .onboard-backdrop {
      position: fixed;
      inset: 0;
      z-index: 100;
      background: rgba(5, 8, 18, 0.78);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .onboard-backdrop.open { display: flex; }
    .onboard-modal {
      width: min(780px, 100%);
      max-height: 90vh;
      overflow-y: auto;
      background: linear-gradient(155deg, rgba(14, 21, 38, 0.96), rgba(24, 18, 48, 0.95));
      border: 1px solid rgba(167, 139, 250, 0.38);
      border-radius: 16px;
      padding: 22px 24px;
      box-shadow: 0 24px 70px rgba(2, 6, 23, 0.78), 0 0 42px rgba(56, 189, 248, 0.14);
    }
    .onboard-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 12px;
    }
    .onboard-kicker {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #7dd3fc;
      background: rgba(56, 189, 248, 0.14);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 999px;
      padding: 3px 10px;
      margin-bottom: 8px;
    }
    .onboard-modal h2 {
      margin: 0;
      font-size: 1.3rem;
      letter-spacing: -0.015em;
    }
    .onboard-lead {
      color: #cbd5e1;
      font-size: 0.9rem;
      margin: 6px 0 16px;
    }
    .onboard-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    @media (max-width: 720px) {
      .onboard-grid { grid-template-columns: 1fr; }
    }
    .onboard-step {
      background: rgba(9, 14, 27, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 10px;
      padding: 12px;
    }
    .onboard-step h3 {
      margin: 0 0 4px;
      font-size: 0.88rem;
      color: #e0f2fe;
    }
    .onboard-step p {
      margin: 0;
      font-size: 0.82rem;
      color: var(--muted);
    }
    .onboard-cmd {
      background: rgba(8, 13, 25, 0.88);
      border: 1px solid rgba(56, 189, 248, 0.24);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 0.8rem;
      color: #bae6fd;
      margin-bottom: 14px;
      overflow-x: auto;
    }
    .onboard-foot {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      padding-top: 10px;
      border-top: 1px solid rgba(148, 163, 184, 0.16);
    }
    .onboard-cta {
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      color: #06101f;
      border-color: transparent;
      font-weight: 700;
      padding: 8px 16px;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 1.32rem;
      font-weight: 700;
      letter-spacing: -0.015em;
      background: linear-gradient(90deg, #f8fafc, #38bdf8, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .sub {
      color: var(--muted);
      font-size: 0.83rem;
      margin-top: 2px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    select, button {
      background: var(--panel2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 11px;
      font-size: 0.85rem;
      font-family: inherit;
    }
    button { cursor: pointer; font-weight: 550; }
    button:hover { border-color: var(--cyan); }
    button.active {
      background: rgba(56, 189, 248, 0.16);
      border-color: var(--cyan);
      color: #fff;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.78rem;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--muted);
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--muted);
    }
    .dot.live {
      background: var(--emerald);
      box-shadow: 0 0 8px var(--emerald);
      animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.35; }
    }
    .kpis {
      display: grid;
      grid-template-columns: repeat(6, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    @media (max-width: 1120px) {
      .kpis { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 640px) {
      .kpis { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi {
      background: rgba(15, 22, 41, 0.76);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 11px 13px;
    }
    .kpi-label {
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }
    .kpi-val {
      font-size: 1.32rem;
      font-weight: 700;
      margin-top: 2px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .kpi-sub {
      font-size: 0.76rem;
      color: var(--muted);
      margin-top: 2px;
    }
    .scrubber-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      background: rgba(15, 22, 41, 0.76);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 14px;
      margin-bottom: 14px;
    }
    .scrubber-bar input[type="range"] {
      flex: 1;
      min-width: 220px;
      accent-color: var(--cyan);
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1.12fr 1fr;
      gap: 14px;
      margin-bottom: 14px;
    }
    @media (max-width: 1080px) {
      .grid-2 { grid-template-columns: 1fr; }
    }
    .card {
      background: rgba(15, 22, 41, 0.76);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 12px 34px rgba(2, 6, 23, 0.42);
    }
    .card-head {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 10px;
    }
    .card-title {
      font-size: 1rem;
      font-weight: 650;
    }
    .card-sub {
      font-size: 0.78rem;
      color: var(--muted);
    }
    .checks {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 0.78rem;
      color: var(--muted);
    }
    .checks label {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      cursor: pointer;
    }
    canvas {
      display: block;
      width: 100%;
      border-radius: 10px;
      background: radial-gradient(circle at 50% 45%, rgba(24, 36, 62, 0.55), rgba(7, 10, 18, 0.92));
      border: 1px solid rgba(36, 53, 86, 0.75);
    }
    #space-canvas { height: 420px; cursor: grab; }
    #space-canvas:active { cursor: grabbing; }
    #loss-canvas { height: 420px; }
    #heat-canvas { height: 320px; }
    #nn-canvas { height: 310px; }
    .tooltip {
      position: fixed;
      pointer-events: none;
      display: none;
      z-index: 50;
      background: rgba(8, 12, 22, 0.95);
      border: 1px solid var(--cyan);
      border-radius: 8px;
      padding: 7px 10px;
      font-size: 0.78rem;
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
      max-width: 320px;
      white-space: pre-line;
    }
    .bar-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .bar-item {
      display: grid;
      grid-template-columns: 165px 1fr 86px;
      align-items: center;
      gap: 10px;
      font-size: 0.8rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .bar-track {
      height: 10px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--cyan), var(--emerald));
    }
    .bar-fill.attn {
      background: linear-gradient(90deg, #60a5fa, #a78bfa);
    }
    .split-bar {
      display: flex;
      height: 14px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,0.06);
      margin: 10px 0 6px;
    }
    .empty-banner {
      background: rgba(56, 189, 248, 0.08);
      border: 1px dashed rgba(56, 189, 248, 0.45);
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 14px;
      font-size: 0.86rem;
      color: #cfe8ff;
      display: none;
    }
    code {
      background: rgba(255,255,255,0.08);
      padding: 2px 6px;
      border-radius: 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.82rem;
    }
  </style>
</head>
<body>
  <div class="bg-aurora" aria-hidden="true"></div>
  <div class="onboard-backdrop" id="onboard-backdrop" role="dialog" aria-modal="true" aria-labelledby="onboard-title">
    <div class="onboard-modal">
      <div class="onboard-head">
        <div>
          <span class="onboard-kicker">Welcome</span>
          <h2 id="onboard-title">LoRA Parameter Space &amp; Loss Observatory</h2>
        </div>
        <button type="button" id="onboard-close-x" aria-label="Close guide">Close</button>
      </div>
      <p class="onboard-lead">
        Watch real LoRA weights shift away from the frozen regular base model in real time while training and validation loss drop.
      </p>
      <div class="onboard-grid">
        <div class="onboard-step">
          <h3>1. Regular Model Origin (0, 0, 0)</h3>
          <p>The silver sphere at the origin is the frozen base model (ΔW = 0). Drag in 3D or scroll to inspect the trajectory as fine-tuning diverges from it.</p>
        </div>
        <div class="onboard-step">
          <h3>2. Layer &amp; Module Orbits</h3>
          <p>Toggle <b>Layer Sub-Trails</b> and <b>Projection Clusters</b> to see which Transformer layers and attention/MLP modules move fastest in weight space.</p>
        </div>
        <div class="onboard-step">
          <h3>3. Loss &amp; Heatmaps</h3>
          <p>Track train and validation cross-entropy loss with EMA smoothing or log scale alongside the 36×7 Layer×Module displacement matrix and top movers.</p>
        </div>
        <div class="onboard-step">
          <h3>4. Time-Scrub &amp; Replay</h3>
          <p>Pause live polling or drag the step scrubber to replay any snapshot across epochs without touching the training run.</p>
        </div>
      </div>
      <div class="onboard-cmd">
        Quick smoke test: <code>.venv/bin/python train_visual.py --smoke-test</code>
      </div>
      <div class="onboard-foot">
        <span class="sub">Reopen anytime via the Guide button in the top right.</span>
        <button type="button" class="onboard-cta" id="onboard-start-btn">Explore Observatory</button>
      </div>
    </div>
  </div>
  <div class="wrap">
    <header>
      <div>
        <h1>LoRA Parameter Space &amp; Loss Observatory</h1>
        <div class="sub" id="run-subtitle">Watching live parameter displacement vs. frozen regular model</div>
      </div>
      <div class="toolbar">
        <select id="run-select"></select>
        <button id="live-btn" class="active" type="button">Live Poll: ON</button>
        <button id="guide-open-btn" class="guide-btn" type="button">Guide</button>
        <span class="pill"><span class="dot" id="status-dot"></span><span id="status-text">Waiting</span></span>
      </div>
    </header>

    <div class="empty-banner" id="empty-banner">
      No telemetry runs found yet under <code>./runs/</code>. Start a quick 30-step test or full LoRA training in your Terminal:
      <br /><code>.venv/bin/python train_visual.py --smoke-test</code> &nbsp;or&nbsp; <code>.venv/bin/python train_visual.py</code>
    </div>

    <section class="kpis">
      <div class="kpi">
        <div class="kpi-label">Iteration</div>
        <div class="kpi-val" id="kpi-step">0</div>
        <div class="kpi-sub" id="kpi-step-sub">-</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Train Loss</div>
        <div class="kpi-val" id="kpi-train-loss">-</div>
        <div class="kpi-sub" id="kpi-train-sub">-</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Best Val Loss</div>
        <div class="kpi-val" id="kpi-val-loss">-</div>
        <div class="kpi-sub" id="kpi-val-sub">-</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Shift vs Regular Model</div>
        <div class="kpi-val" id="kpi-delta">-</div>
        <div class="kpi-sub" id="kpi-delta-sub">||ΔW||_F from origin</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Throughput</div>
        <div class="kpi-val" id="kpi-tps">-</div>
        <div class="kpi-sub" id="kpi-tps-sub">-</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Peak Unified RAM</div>
        <div class="kpi-val" id="kpi-mem">-</div>
        <div class="kpi-sub">M4 32 GB Unified</div>
      </div>
    </section>

    <section class="scrubber-bar">
      <button id="play-btn" type="button">Play Replay</button>
      <span style="font-size:0.83rem;color:var(--muted);">Timeline Step:</span>
      <input id="step-slider" type="range" min="0" max="0" value="0" />
      <span id="slider-label" style="font-family:ui-monospace,monospace;font-size:0.84rem;min-width:110px;text-align:right;">Step 0</span>
      <button id="follow-btn" class="active" type="button">Follow Latest</button>
    </section>

    <section class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <div class="card-title">3D Parameter Space Trajectory</div>
            <div class="card-sub">Origin (0,0,0) = Regular Model &bull; Drag to orbit, scroll to zoom</div>
          </div>
          <div class="checks">
            <label><input id="chk-subtrails" type="checkbox" checked /> Attn vs MLP trails</label>
            <label><input id="chk-layers" type="checkbox" checked /> 36-Layer Constellation</label>
            <label><input id="chk-orbit" type="checkbox" checked /> Auto-orbit</label>
            <button id="reset-cam" type="button" style="padding:3px 8px;font-size:0.76rem;">Reset View</button>
          </div>
        </div>
        <canvas id="space-canvas"></canvas>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <div class="card-title">Loss Function Over Training</div>
            <div class="card-sub">Raw Train Loss, EMA Smoothed Trend &amp; Held-Out Validation Loss</div>
          </div>
          <div class="checks">
            <label><input id="chk-ema" type="checkbox" checked /> EMA Smooth</label>
            <label><input id="chk-log" type="checkbox" /> Log Scale</label>
          </div>
        </div>
        <canvas id="loss-canvas"></canvas>
      </div>
    </section>

    <section class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <div class="card-title">Parameter Shift Heatmap (Layers &times; Modules)</div>
            <div class="card-sub" id="heat-sub">Comparing live LoRA weights against frozen regular model weights</div>
          </div>
          <div class="checks">
            <label><input id="chk-rel" type="checkbox" checked /> % Shift vs Regular Model</label>
          </div>
        </div>
        <canvas id="heat-canvas"></canvas>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <div class="card-title">Which Parameters Changed Most?</div>
            <div class="card-sub">Top shifting weight matrices &amp; Attention vs MLP share at selected step</div>
          </div>
        </div>
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:var(--muted);">
            <span id="attn-share-label">Attention (q/k/v/o): -</span>
            <span id="mlp-share-label">MLP (gate/up/down): -</span>
          </div>
          <div class="split-bar">
            <div id="attn-bar" style="width:50%;background:linear-gradient(90deg,#60a5fa,#a78bfa);"></div>
            <div id="mlp-bar" style="width:50%;background:linear-gradient(90deg,#10b981,#34d399);"></div>
          </div>
        </div>
        <div class="bar-list" id="top-bars"></div>
      </div>
    </section>

    <section class="card" style="margin-bottom: 14px;">
      <div class="card-head">
        <div>
          <div class="card-title">Transformer Neural Network Layers (Live Weight Shift Flow)</div>
          <div class="card-sub">Input Tokens &rarr; Embedding &rarr; All Transformer Blocks (Self-Attention Q/K/V/O + MLP Gate/Up/Down) &rarr; LM Head &bull; Hover a layer to inspect</div>
        </div>
        <div class="checks">
          <span class="sub" style="font-size:0.75rem;">Top band: Attention &bull; Bottom band: MLP</span>
        </div>
      </div>
      <canvas id="nn-canvas"></canvas>
    </section>
  </div>

  <div class="tooltip" id="tooltip"></div>

  <script>
    const runSelect = document.getElementById("run-select");
    const liveBtn = document.getElementById("live-btn");
    const stepSlider = document.getElementById("step-slider");
    const sliderLabel = document.getElementById("slider-label");
    const followBtn = document.getElementById("follow-btn");
    const playBtn = document.getElementById("play-btn");
    const tooltip = document.getElementById("tooltip");

    const spaceCanvas = document.getElementById("space-canvas");
    const lossCanvas = document.getElementById("loss-canvas");
    const heatCanvas = document.getElementById("heat-canvas");
    const nnCanvas = document.getElementById("nn-canvas");
    const isEmbedded = new URLSearchParams(window.location.search).get("embed") === "1";
    if (isEmbedded) document.body.classList.add("embedded");

    let state = {
      runs: [],
      currentRun: null,
      meta: null,
      trainEvents: [],
      valEvents: [],
      snapshots: [], // merged {iteration, train_loss, param_space, ...}
      selectedIdx: 0,
      followLatest: true,
      livePoll: true,
      playing: false,
      camYaw: 0.65,
      camPitch: 0.42,
      camZoom: 1.0,
      dragging: false,
      lastX: 0,
      lastY: 0,
      heatCells: [],
      lossPoints: [],
      layerDots: [],
      nnNodes: [],
    };

    function showTooltip(x, y, text) {
      tooltip.style.display = "block";
      tooltip.textContent = text;
      const pad = 14;
      const w = tooltip.offsetWidth;
      const h = tooltip.offsetHeight;
      tooltip.style.left = Math.min(window.innerWidth - w - 10, x + pad) + "px";
      tooltip.style.top = Math.min(window.innerHeight - h - 10, y + pad) + "px";
    }
    function hideTooltip() {
      tooltip.style.display = "none";
    }

    function sizeCanvas(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
        canvas.width = Math.round(rect.width * dpr);
        canvas.height = Math.round(rect.height * dpr);
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w: rect.width, h: rect.height };
    }

    async function fetchTelemetry() {
      try {
        const q = state.currentRun ? ("?run=" + encodeURIComponent(state.currentRun)) : "";
        const res = await fetch("/api/telemetry" + q);
        const data = await res.json();

        state.runs = data.runs || [];
        const selVal = runSelect.value;
        runSelect.innerHTML = state.runs.length
          ? state.runs.map(r => `<option value="${r}">${r}</option>`).join("")
          : '<option value="">No runs yet</option>';
        if (data.run_id && state.runs.includes(data.run_id)) {
          runSelect.value = data.run_id;
          state.currentRun = data.run_id;
        } else if (selVal && state.runs.includes(selVal)) {
          runSelect.value = selVal;
        }

        document.getElementById("empty-banner").style.display = state.runs.length ? "none" : "block";
        state.meta = data.meta || null;

        const events = data.events || [];
        state.trainEvents = events.filter(e => e.type === "train");
        state.valEvents = events.filter(e => e.type === "val");
        state.snapshots = events.filter(e => e.param_space && e.param_space.xyz);

        const maxIdx = Math.max(0, state.snapshots.length - 1);
        stepSlider.max = String(maxIdx);
        if (state.followLatest) {
          state.selectedIdx = maxIdx;
          stepSlider.value = String(maxIdx);
        } else {
          state.selectedIdx = Math.min(state.selectedIdx, maxIdx);
          stepSlider.value = String(state.selectedIdx);
        }

        updateStatus(data);
        renderAll();
      } catch (e) {
        document.getElementById("status-text").textContent = "Disconnected";
      }
    }

    function updateStatus(data) {
      const dot = document.getElementById("status-dot");
      const txt = document.getElementById("status-text");
      if (!state.snapshots.length) {
        dot.className = "dot";
        txt.textContent = state.runs.length ? "Initializing..." : "No runs yet";
        return;
      }
      dot.className = state.livePoll ? "dot live" : "dot";
      txt.textContent = state.livePoll ? "Live Telemetry" : "Paused";
      if (state.meta) {
        document.getElementById("run-subtitle").textContent =
          `${state.meta.model || ""}  |  rank=${state.meta.rank ?? "-"}, scale=${state.meta.scale ?? "-"}, ${state.meta.num_lora_modules ?? "-"} LoRA matrices`;
      }
    }

    function updateKPIs() {
      const snap = state.snapshots[state.selectedIdx];
      const latestTrain = state.trainEvents[state.trainEvents.length - 1];
      if (!snap) return;

      const totalIters = state.meta?.total_iters || "-";
      const curIter = snap.iteration ?? 0;
      sliderLabel.textContent = `Step ${curIter}`;
      document.getElementById("kpi-step").textContent = `${curIter} / ${totalIters}`;
      const pct = (typeof totalIters === "number" && totalIters > 0)
        ? ((curIter / totalIters) * 100).toFixed(1) + "% complete" : "";
      document.getElementById("kpi-step-sub").textContent = pct;

      const trainAtIdx = state.trainEvents.filter(e => e.iteration <= curIter);
      const tCur = trainAtIdx[trainAtIdx.length - 1];
      const tFirst = state.trainEvents[0];
      if (tCur) {
        document.getElementById("kpi-train-loss").textContent = tCur.train_loss.toFixed(4);
        const drop = tFirst ? ((tFirst.train_loss - tCur.train_loss) / Math.max(tFirst.train_loss, 1e-6) * 100) : 0;
        document.getElementById("kpi-train-sub").textContent =
          tFirst ? `${drop >= 0 ? "-" : "+"}${Math.abs(drop).toFixed(1)}% from step ${tFirst.iteration}` : "";
        document.getElementById("kpi-tps").textContent = `${Math.round(tCur.tokens_per_second)} tok/s`;
        document.getElementById("kpi-tps-sub").textContent = `${tCur.iterations_per_second.toFixed(2)} it/s | lr ${tCur.learning_rate.toExponential(1)}`;
        document.getElementById("kpi-mem").textContent = `${tCur.peak_memory_gb.toFixed(2)} GB`;
      }

      const valAtIdx = state.valEvents.filter(e => e.iteration <= curIter);
      if (valAtIdx.length) {
        const best = valAtIdx.reduce((a, b) => (b.val_loss < a.val_loss ? b : a), valAtIdx[0]);
        const curV = valAtIdx[valAtIdx.length - 1];
        document.getElementById("kpi-val-loss").textContent = best.val_loss.toFixed(4);
        document.getElementById("kpi-val-sub").textContent = `latest ${curV.val_loss.toFixed(4)} @ step ${best.iteration}`;
      } else {
        document.getElementById("kpi-val-loss").textContent = "-";
        document.getElementById("kpi-val-sub").textContent = "Awaiting eval step";
      }

      const ps = snap.param_space;
      document.getElementById("kpi-delta").textContent = ps.total_delta_norm.toFixed(3);
      const mods = ps.modules || [];
      const meanPct = mods.length ? (mods.reduce((s, m) => s + m.rel_pct, 0) / mods.length) : 0;
      document.getElementById("kpi-delta-sub").textContent = `avg ${meanPct.toFixed(3)}% shift vs regular`;
    }

    function project3D(pt, scale, cx, cy) {
      const cosY = Math.cos(state.camYaw), sinY = Math.sin(state.camYaw);
      const cosP = Math.cos(state.camPitch), sinP = Math.sin(state.camPitch);
      const x1 = pt[0] * cosY - pt[2] * sinY;
      const z1 = pt[0] * sinY + pt[2] * cosY;
      const y2 = pt[1] * cosP - z1 * sinP;
      const z2 = pt[1] * sinP + z1 * cosP;
      const persp = 4.2 / Math.max(1.2, 4.2 + z2 * scale * 0.16);
      return [cx + x1 * scale * persp, cy - y2 * scale * persp, z2];
    }

    function lerpColor(t) {
      // amber (#fbbf24) -> cyan (#38bdf8) -> emerald (#34d399)
      t = Math.max(0, Math.min(1, t));
      if (t < 0.5) {
        const k = t * 2;
        const r = Math.round(251 + (56 - 251) * k);
        const g = Math.round(191 + (189 - 191) * k);
        const b = Math.round(36 + (248 - 36) * k);
        return `rgb(${r},${g},${b})`;
      }
      const k = (t - 0.5) * 2;
      const r = Math.round(56 + (52 - 56) * k);
      const g = Math.round(189 + (211 - 189) * k);
      const b = Math.round(248 + (153 - 248) * k);
      return `rgb(${r},${g},${b})`;
    }

    function renderSpace() {
      const { ctx, w, h } = sizeCanvas(spaceCanvas);
      ctx.clearRect(0, 0, w, h);
      state.layerDots = [];

      const snaps = state.snapshots.slice(0, state.selectedIdx + 1);
      const cx = w * 0.5, cy = h * 0.52;

      // Normalize coordinate scale across all snapshots & layers so motion fills viewport nicely
      let maxCoord = 0.05;
      const showSub = document.getElementById("chk-subtrails").checked;
      const showLayers = document.getElementById("chk-layers").checked;
      for (const s of state.snapshots) {
        const ps = s.param_space;
        for (const v of ps.xyz) maxCoord = Math.max(maxCoord, Math.abs(v));
        if (showSub) {
          for (const v of (ps.attn_xyz || [])) maxCoord = Math.max(maxCoord, Math.abs(v));
          for (const v of (ps.mlp_xyz || [])) maxCoord = Math.max(maxCoord, Math.abs(v));
        }
      }
      const curLayers = snaps.length ? (snaps[snaps.length - 1].param_space.layers || {}) : {};
      if (showLayers) {
        for (const k of Object.keys(curLayers)) {
          for (const v of curLayers[k].xyz) maxCoord = Math.max(maxCoord, Math.abs(v));
        }
      }
      const scale = (Math.min(w, h) * 0.34 / maxCoord) * state.camZoom;

      // Floor grid centered at Regular Model (0,0,0)
      ctx.strokeStyle = "rgba(148, 163, 184, 0.10)";
      ctx.lineWidth = 1;
      const gStep = maxCoord * 0.45;
      for (let i = -3; i <= 3; i++) {
        const a = project3D([i * gStep, 0, -3 * gStep], scale, cx, cy);
        const b = project3D([i * gStep, 0, 3 * gStep], scale, cx, cy);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        const c = project3D([-3 * gStep, 0, i * gStep], scale, cx, cy);
        const d = project3D([3 * gStep, 0, i * gStep], scale, cx, cy);
        ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke();
      }

      // Draw Attention & MLP sub-trails
      if (showSub && snaps.length > 1) {
        for (const [key, color] of [["attn_xyz", "rgba(96, 165, 250, 0.55)"], ["mlp_xyz", "rgba(52, 211, 153, 0.55)"]]) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          snaps.forEach((s, i) => {
            const p = project3D(s.param_space[key] || [0,0,0], scale, cx, cy);
            if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
          });
          ctx.stroke();
          const lastP = project3D(snaps[snaps.length - 1].param_space[key] || [0,0,0], scale, cx, cy);
          ctx.fillStyle = color.replace("0.55", "0.95");
          ctx.beginPath(); ctx.arc(lastP[0], lastP[1], 4, 0, Math.PI * 2); ctx.fill();
        }
      }

      // Draw 36-Layer Constellation at selected step
      if (showLayers && snaps.length) {
        const lyrKeys = Object.keys(curLayers);
        const maxLN = Math.max(1e-6, ...lyrKeys.map(k => curLayers[k].delta_norm || 0));
        const originP = project3D([0, 0, 0], scale, cx, cy);
        for (const k of lyrKeys) {
          const info = curLayers[k];
          const p = project3D(info.xyz, scale, cx, cy);
          const ratio = (info.delta_norm || 0) / maxLN;
          ctx.strokeStyle = `rgba(167, 139, 250, ${0.08 + ratio * 0.25})`;
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(originP[0], originP[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();

          const rad = 2.6 + ratio * 4.4;
          ctx.fillStyle = `rgba(167, 139, 250, ${0.45 + ratio * 0.55})`;
          ctx.beginPath(); ctx.arc(p[0], p[1], rad, 0, Math.PI * 2); ctx.fill();
          state.layerDots.push({ x: p[0], y: p[1], r: rad + 4, layer: k, norm: info.delta_norm });
        }
      }

      // Draw Main Full-Model Trajectory
      if (snaps.length > 0) {
        const losses = snaps.map(s => s.train_loss).filter(v => typeof v === "number");
        const maxL = losses.length ? Math.max(...losses) : 1;
        const minL = losses.length ? Math.min(...losses) : 0;

        for (let i = 1; i < snaps.length; i++) {
          const p0 = project3D(snaps[i - 1].param_space.xyz, scale, cx, cy);
          const p1 = project3D(snaps[i].param_space.xyz, scale, cx, cy);
          const t = i / Math.max(1, snaps.length - 1);
          ctx.strokeStyle = lerpColor(t);
          ctx.lineWidth = 2.6;
          ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
        }

        const head = snaps[snaps.length - 1];
        const hp = project3D(head.param_space.xyz, scale, cx, cy);
        const grad = ctx.createRadialGradient(hp[0], hp[1], 1, hp[0], hp[1], 16);
        grad.addColorStop(0, "rgba(52, 211, 153, 0.95)");
        grad.addColorStop(1, "rgba(52, 211, 153, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(hp[0], hp[1], 16, 0, Math.PI * 2); ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.beginPath(); ctx.arc(hp[0], hp[1], 4.5, 0, Math.PI * 2); ctx.fill();
        ctx.font = "600 11px ui-monospace, monospace";
        ctx.fillStyle = "#a7f3d0";
        ctx.fillText(`Fine-Tuned @ Step ${head.iteration}`, hp[0] + 9, hp[1] - 6);
      }

      // Draw Regular Model Origin Beacon (0,0,0)
      const o = project3D([0, 0, 0], scale, cx, cy);
      const og = ctx.createRadialGradient(o[0], o[1], 1, o[0], o[1], 22);
      og.addColorStop(0, "rgba(251, 191, 36, 0.85)");
      og.addColorStop(1, "rgba(251, 191, 36, 0)");
      ctx.fillStyle = og;
      ctx.beginPath(); ctx.arc(o[0], o[1], 22, 0, Math.PI * 2); ctx.fill();

      ctx.strokeStyle = "rgba(251, 191, 36, 0.7)";
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(o[0], o[1], 9, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#fef08a";
      ctx.beginPath(); ctx.arc(o[0], o[1], 4, 0, Math.PI * 2); ctx.fill();
      ctx.font = "650 11px -apple-system, sans-serif";
      ctx.fillText("Regular Model (0,0,0)", o[0] + 12, o[1] + 4);
    }

    function renderLoss() {
      const { ctx, w, h } = sizeCanvas(lossCanvas);
      ctx.clearRect(0, 0, w, h);
      state.lossPoints = [];

      const trains = state.trainEvents;
      const vals = state.valEvents;
      if (!trains.length && !vals.length) {
        ctx.fillStyle = "#8496b8";
        ctx.font = "13px -apple-system, sans-serif";
        ctx.fillText("Waiting for training loss data...", 24, 40);
        return;
      }

      const pad = { l: 54, r: 22, t: 22, b: 36 };
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;

      const allIters = [...trains.map(e => e.iteration), ...vals.map(e => e.iteration)];
      const maxX = Math.max(10, ...allIters);
      const useLog = document.getElementById("chk-log").checked;
      const useEma = document.getElementById("chk-ema").checked;

      const rawVals = [...trains.map(e => e.train_loss), ...vals.map(e => e.val_loss)].filter(v => v > 0);
      const minY = Math.max(0.01, Math.min(...rawVals) * 0.9);
      const maxY = Math.max(minY + 0.1, Math.max(...rawVals) * 1.06);

      const xOf = it => pad.l + (it / maxX) * pw;
      const yOf = val => {
        const clamped = Math.max(minY, val);
        const norm = useLog
          ? (Math.log(clamped) - Math.log(minY)) / (Math.log(maxY) - Math.log(minY))
          : (clamped - minY) / (maxY - minY);
        return pad.t + ph - norm * ph;
      };

      // Grid lines
      ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
      ctx.lineWidth = 1;
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillStyle = "#8496b8";
      for (let i = 0; i <= 5; i++) {
        const y = pad.t + (ph * i) / 5;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        const v = useLog
          ? Math.exp(Math.log(maxY) - (i / 5) * (Math.log(maxY) - Math.log(minY)))
          : maxY - (i / 5) * (maxY - minY);
        ctx.fillText(v.toFixed(2), 8, y + 4);
      }

      // X axis ticks
      for (let i = 0; i <= 5; i++) {
        const it = Math.round((maxX * i) / 5);
        const x = xOf(it);
        ctx.fillText(String(it), x - 10, h - 12);
      }

      // Raw train loss
      if (trains.length) {
        ctx.strokeStyle = useEma ? "rgba(56, 189, 248, 0.28)" : "#38bdf8";
        ctx.lineWidth = useEma ? 1.3 : 2.2;
        ctx.beginPath();
        trains.forEach((e, i) => {
          const x = xOf(e.iteration), y = yOf(e.train_loss);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          state.lossPoints.push({ x, y, type: "Train", ...e });
        });
        ctx.stroke();

        if (useEma) {
          let ema = trains[0].train_loss;
          ctx.strokeStyle = "#38bdf8";
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          trains.forEach((e, i) => {
            ema = 0.85 * ema + 0.15 * e.train_loss;
            const x = xOf(e.iteration), y = yOf(ema);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          });
          ctx.stroke();
        }
      }

      // Val loss
      if (vals.length) {
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        vals.forEach((e, i) => {
          const x = xOf(e.iteration), y = yOf(e.val_loss);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();

        for (const e of vals) {
          const x = xOf(e.iteration), y = yOf(e.val_loss);
          ctx.fillStyle = "#fbbf24";
          ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
          state.lossPoints.push({ x, y, type: "Val", ...e });
        }
      }

      // Selected scrubber cursor
      const snap = state.snapshots[state.selectedIdx];
      if (snap) {
        const cx = xOf(snap.iteration);
        ctx.strokeStyle = "rgba(52, 211, 153, 0.65)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, pad.t + ph); ctx.stroke();
        ctx.setLineDash([]);
      }

      // Legend
      ctx.font = "11px -apple-system, sans-serif";
      ctx.fillStyle = "#38bdf8"; ctx.fillText("● Train Loss", pad.l + 8, pad.t + 14);
      ctx.fillStyle = "#fbbf24"; ctx.fillText("● Val Loss", pad.l + 96, pad.t + 14);
    }

    function heatColor(ratio) {
      const t = Math.max(0, Math.min(1, ratio));
      if (t < 0.33) {
        const k = t / 0.33;
        return `rgb(${Math.round(15 + 14*k)}, ${Math.round(23 + 55*k)}, ${Math.round(42 + 174*k)})`;
      }
      if (t < 0.66) {
        const k = (t - 0.33) / 0.33;
        return `rgb(${Math.round(29 - 23*k)}, ${Math.round(78 + 130*k)}, ${Math.round(216 - 30*k)})`;
      }
      const k = (t - 0.66) / 0.34;
      return `rgb(${Math.round(6 + 245*k)}, ${Math.round(208 - 17*k)}, ${Math.round(186 - 150*k)})`;
    }

    function renderHeatmapAndBars() {
      const { ctx, w, h } = sizeCanvas(heatCanvas);
      ctx.clearRect(0, 0, w, h);
      state.heatCells = [];

      const snap = state.snapshots[state.selectedIdx];
      const mods = snap?.param_space?.modules || [];
      if (!mods.length) {
        ctx.fillStyle = "#8496b8";
        ctx.font = "13px -apple-system, sans-serif";
        ctx.fillText("Waiting for parameter telemetry...", 24, 40);
        return;
      }

      const useRel = document.getElementById("chk-rel").checked;
      const projOrder = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
      const layers = [...new Set(mods.map(m => m.layer))].filter(l => l >= 0).sort((a, b) => a - b);
      const byKey = new Map(mods.map(m => [`${m.layer}:${m.proj}`, m]));

      const vals = mods.map(m => useRel ? m.rel_pct : m.delta_norm);
      const maxV = Math.max(1e-7, ...vals);

      const pad = { l: 82, r: 16, t: 18, b: 32 };
      const cw = (w - pad.l - pad.r) / Math.max(1, layers.length);
      const ch = (h - pad.t - pad.b) / projOrder.length;

      ctx.font = "11px ui-monospace, monospace";
      projOrder.forEach((proj, row) => {
        const y = pad.t + row * ch;
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(proj, 8, y + ch * 0.62);

        layers.forEach((lyr, col) => {
          const x = pad.l + col * cw;
          const m = byKey.get(`${lyr}:${proj}`);
          const v = m ? (useRel ? m.rel_pct : m.delta_norm) : 0;
          ctx.fillStyle = heatColor(v / maxV);
          ctx.fillRect(x + 1, y + 1, Math.max(1, cw - 2), Math.max(1, ch - 2));
          if (m) {
            state.heatCells.push({ x, y, w: cw, h: ch, m });
          }
        });
      });

      // Layer labels
      ctx.fillStyle = "#8496b8";
      layers.forEach((lyr, col) => {
        if (lyr % 5 === 0 || lyr === layers[layers.length - 1]) {
          ctx.fillText("L" + lyr, pad.l + col * cw + 2, h - 10);
        }
      });

      // Top moving parameters & Attn vs MLP share
      const ps = snap.param_space;
      const attnSq = ps.attn_delta_norm ** 2;
      const mlpSq = ps.mlp_delta_norm ** 2;
      const totSq = Math.max(1e-9, attnSq + mlpSq);
      const attnPct = (attnSq / totSq) * 100;
      const mlpPct = (mlpSq / totSq) * 100;
      document.getElementById("attn-share-label").textContent = `Attention (q/k/v/o): ${attnPct.toFixed(1)}%`;
      document.getElementById("mlp-share-label").textContent = `MLP (gate/up/down): ${mlpPct.toFixed(1)}%`;
      document.getElementById("attn-bar").style.width = `${attnPct}%`;
      document.getElementById("mlp-bar").style.width = `${mlpPct}%`;

      const sorted = [...mods].sort((a, b) => (useRel ? b.rel_pct - a.rel_pct : b.delta_norm - a.delta_norm)).slice(0, 10);
      const topMax = Math.max(1e-7, sorted[0] ? (useRel ? sorted[0].rel_pct : sorted[0].delta_norm) : 1);
      document.getElementById("top-bars").innerHTML = sorted.map(m => {
        const val = useRel ? m.rel_pct : m.delta_norm;
        const pctW = (val / topMax) * 100;
        const label = useRel ? `${m.rel_pct.toFixed(3)}%` : `${m.delta_norm.toFixed(3)}`;
        return `
          <div class="bar-item">
            <span>L${String(m.layer).padStart(2, "0")}.${m.proj}</span>
            <div class="bar-track"><div class="bar-fill ${m.group === "attn" ? "attn" : ""}" style="width:${pctW.toFixed(1)}%"></div></div>
            <span style="text-align:right;color:var(--muted);">${label}</span>
          </div>
        `;
      }).join("");
    }

    function renderNeuralNet() {
      const { ctx, w, h } = sizeCanvas(nnCanvas);
      ctx.clearRect(0, 0, w, h);
      state.nnNodes = [];

      const snap = state.snapshots[state.selectedIdx];
      const mods = snap?.param_space?.modules || [];
      const useRel = document.getElementById("chk-rel").checked;

      const byLayer = new Map();
      mods.forEach(m => {
        if (m.layer < 0) return;
        if (!byLayer.has(m.layer)) {
          byLayer.set(m.layer, { attn: 0, mlp: 0, mods: [], topMod: null, topVal: -1 });
        }
        const entry = byLayer.get(m.layer);
        const val = useRel ? m.rel_pct : m.delta_norm;
        if (m.group === "attn") entry.attn += val;
        else entry.mlp += val;
        entry.mods.push(m);
        if (val > entry.topVal) {
          entry.topVal = val;
          entry.topMod = m;
        }
      });

      const layerNums = byLayer.size
        ? [...byLayer.keys()].sort((a, b) => a - b)
        : Array.from({ length: 36 }, (_, i) => i);

      let maxGroupVal = 1e-7;
      layerNums.forEach(l => {
        const e = byLayer.get(l);
        if (e) maxGroupVal = Math.max(maxGroupVal, e.attn, e.mlp);
      });

      const pad = { l: 74, r: 74, t: 44, b: 40 };
      const usableW = Math.max(200, w - pad.l - pad.r);
      const midY = (pad.t + (h - pad.b)) * 0.52;
      const attnY = midY - 64;
      const mlpY = midY + 14;
      const boxH = 48;

      // Input & Output endpoint pills
      function drawEndpoint(x, label, sub) {
        const ew = 54, eh = 96;
        const ey = midY - eh / 2;
        ctx.fillStyle = "rgba(30, 41, 59, 0.86)";
        ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.roundRect(x - ew / 2, ey, ew, eh, 10);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#e2e8f0";
        ctx.font = "600 11px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(label, x, midY - 4);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(sub, x, midY + 14);
        ctx.textAlign = "left";
      }

      const stepX = usableW / Math.max(1, layerNums.length);
      const boxW = Math.max(6, Math.min(26, stepX * 0.72));

      // Residual spine connecting Embed -> L0..LN -> LM Head
      ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(40 + 27, midY);
      ctx.lineTo(w - 40 - 27, midY);
      ctx.stroke();

      drawEndpoint(38, "Tokens", "Embed");
      drawEndpoint(w - 38, "Logits", "LM Head");

      layerNums.forEach((lyr, idx) => {
        const cx = pad.l + (idx + 0.5) * stepX;
        const x0 = cx - boxW / 2;
        const info = byLayer.get(lyr);
        const attnRatio = info ? Math.min(1, info.attn / maxGroupVal) : 0;
        const mlpRatio = info ? Math.min(1, info.mlp / maxGroupVal) : 0;
        const combined = Math.max(attnRatio, mlpRatio);

        // Inter-layer flow lines (Attn -> MLP vertical link + residual arc)
        ctx.strokeStyle = info
          ? `rgba(125, 211, 252, ${0.14 + 0.55 * combined})`
          : "rgba(100, 116, 139, 0.16)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(cx, attnY + boxH);
        ctx.lineTo(cx, mlpY);
        ctx.stroke();

        // Attention node (top band)
        ctx.fillStyle = info
          ? `rgba(${Math.round(30 + 26 * attnRatio)}, ${Math.round(58 + 131 * attnRatio)}, ${Math.round(138 + 110 * attnRatio)}, ${0.34 + 0.62 * attnRatio})`
          : "rgba(30, 41, 59, 0.45)";
        ctx.strokeStyle = info && attnRatio > 0.45 ? "rgba(56, 189, 248, 0.9)" : "rgba(96, 165, 250, 0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x0, attnY, boxW, boxH, 4);
        ctx.fill();
        ctx.stroke();

        // MLP node (bottom band)
        ctx.fillStyle = info
          ? `rgba(${Math.round(16 + 151 * mlpRatio * 0.55)}, ${Math.round(78 + 133 * mlpRatio)}, ${Math.round(92 + 90 * mlpRatio)}, ${0.34 + 0.62 * mlpRatio})`
          : "rgba(30, 41, 59, 0.45)";
        ctx.strokeStyle = info && mlpRatio > 0.45 ? "rgba(52, 211, 153, 0.9)" : "rgba(167, 139, 250, 0.35)";
        ctx.beginPath();
        ctx.roundRect(x0, mlpY, boxW, boxH, 4);
        ctx.fill();
        ctx.stroke();

        // Glow halo on top movers
        if (info && combined > 0.65) {
          const rg = ctx.createRadialGradient(cx, midY, 1, cx, midY, Math.max(18, boxW * 1.4));
          rg.addColorStop(0, "rgba(56, 189, 248, 0.22)");
          rg.addColorStop(1, "rgba(56, 189, 248, 0)");
          ctx.fillStyle = rg;
          ctx.beginPath();
          ctx.arc(cx, midY, Math.max(18, boxW * 1.4), 0, Math.PI * 2);
          ctx.fill();
        }

        if (lyr % 5 === 0 || lyr === layerNums[layerNums.length - 1] || boxW >= 18) {
          ctx.fillStyle = "#94a3b8";
          ctx.font = "10px ui-monospace, monospace";
          ctx.textAlign = "center";
          ctx.fillText("L" + lyr, cx, h - 14);
          ctx.textAlign = "left";
        }

        state.nnNodes.push({
          x: x0 - 2,
          y: attnY - 4,
          w: boxW + 4,
          h: mlpY + boxH - attnY + 8,
          layer: lyr,
          info,
        });
      });

      // Row subtitles + blueprint hint
      ctx.font = "600 11px -apple-system, sans-serif";
      ctx.fillStyle = "#7dd3fc";
      ctx.fillText("Self-Attention (Q · K · V · O)", pad.l, attnY - 12);
      ctx.fillStyle = "#6ee7b7";
      ctx.fillText("Feed-Forward MLP (Gate · Up · Down)", pad.l, mlpY + boxH + 18);

      if (!mods.length) {
        ctx.fillStyle = "rgba(186, 230, 253, 0.85)";
        ctx.font = "600 12px -apple-system, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText("Architecture Blueprint · Run train_visual.py to watch live layer shifts", w - pad.r, 24);
        ctx.textAlign = "left";
      }
    }

    function notifyEmbedHeight() {
      if (!isEmbedded) return;
      try {
        const h = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
        window.parent.postMessage({ type: "viz-height", height: h }, "*");
      } catch {}
    }

    function renderAll() {
      updateKPIs();
      renderSpace();
      renderLoss();
      renderHeatmapAndBars();
      renderNeuralNet();
      notifyEmbedHeight();
    }

    // 3D Orbit & Zoom controls
    spaceCanvas.addEventListener("mousedown", (e) => {
      state.dragging = true;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
    });
    window.addEventListener("mouseup", () => { state.dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!state.dragging) return;
      state.camYaw += (e.clientX - state.lastX) * 0.008;
      state.camPitch = Math.max(-1.35, Math.min(1.35, state.camPitch + (e.clientY - state.lastY) * 0.008));
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      renderSpace();
    });
    spaceCanvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      state.camZoom = Math.max(0.35, Math.min(4.0, state.camZoom * (e.deltaY > 0 ? 0.92 : 1.08)));
      renderSpace();
    }, { passive: false });
    document.getElementById("reset-cam").addEventListener("click", () => {
      state.camYaw = 0.65;
      state.camPitch = 0.42;
      state.camZoom = 1.0;
      renderSpace();
    });

    // Hover tooltips
    spaceCanvas.addEventListener("mousemove", (e) => {
      if (state.dragging) { hideTooltip(); return; }
      const rect = spaceCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const hit = state.layerDots.find(d => Math.hypot(d.x - mx, d.y - my) <= d.r);
      if (hit) {
        showTooltip(e.clientX, e.clientY, `Layer ${hit.layer}\n||ΔW||_F = ${hit.norm.toFixed(4)}`);
      } else {
        hideTooltip();
      }
    });
    spaceCanvas.addEventListener("mouseleave", hideTooltip);

    heatCanvas.addEventListener("mousemove", (e) => {
      const rect = heatCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const hit = state.heatCells.find(c => mx >= c.x && mx <= c.x + c.w && my >= c.y && my <= c.y + c.h);
      if (hit) {
        const m = hit.m;
        showTooltip(
          e.clientX,
          e.clientY,
          `Layer ${m.layer} · ${m.proj} (${m.group})\nShift vs Regular: ${m.rel_pct.toFixed(4)}%\n||ΔW||_F: ${m.delta_norm.toFixed(5)}`
        );
      } else {
        hideTooltip();
      }
    });
    heatCanvas.addEventListener("mouseleave", hideTooltip);

    lossCanvas.addEventListener("mousemove", (e) => {
      const rect = lossCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let best = null, bestDist = 18;
      for (const p of state.lossPoints) {
        const d = Math.hypot(p.x - mx, p.y - my);
        if (d < bestDist) { bestDist = d; best = p; }
      }
      if (best) {
        const val = best.type === "Val" ? best.val_loss : best.train_loss;
        showTooltip(e.clientX, e.clientY, `${best.type} Loss @ Step ${best.iteration}\nLoss: ${val.toFixed(4)}`);
      } else {
        hideTooltip();
      }
    });
    lossCanvas.addEventListener("mouseleave", hideTooltip);

    nnCanvas.addEventListener("mousemove", (e) => {
      const rect = nnCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const hit = state.nnNodes.find(n => mx >= n.x && mx <= n.x + n.w && my >= n.y && my <= n.y + n.h);
      if (hit) {
        if (hit.info) {
          const top = hit.info.topMod;
          const unit = document.getElementById("chk-rel").checked ? "%" : "";
          const topTxt = top ? `\nTop matrix: ${top.proj} (${hit.info.topVal.toFixed(4)}${unit})` : "";
          showTooltip(
            e.clientX,
            e.clientY,
            `Transformer Layer ${hit.layer}\nAttn (Q/K/V/O) sum: ${hit.info.attn.toFixed(4)}${unit}\nMLP (Gate/Up/Down) sum: ${hit.info.mlp.toFixed(4)}${unit}${topTxt}`
          );
        } else {
          showTooltip(e.clientX, e.clientY, `Transformer Layer ${hit.layer}\nWaiting for training telemetry...`);
        }
      } else {
        hideTooltip();
      }
    });
    nnCanvas.addEventListener("mouseleave", hideTooltip);

    // UI toggles & scrubber
    stepSlider.addEventListener("input", () => {
      state.followLatest = false;
      followBtn.classList.remove("active");
      state.selectedIdx = Number(stepSlider.value);
      renderAll();
    });
    followBtn.addEventListener("click", () => {
      state.followLatest = !state.followLatest;
      followBtn.classList.toggle("active", state.followLatest);
      if (state.followLatest) {
        state.selectedIdx = Math.max(0, state.snapshots.length - 1);
        stepSlider.value = String(state.selectedIdx);
        renderAll();
      }
    });
    playBtn.addEventListener("click", () => {
      state.playing = !state.playing;
      playBtn.textContent = state.playing ? "Pause Replay" : "Play Replay";
      if (state.playing) {
        state.followLatest = false;
        followBtn.classList.remove("active");
        if (state.selectedIdx >= state.snapshots.length - 1) state.selectedIdx = 0;
      }
    });
    liveBtn.addEventListener("click", () => {
      state.livePoll = !state.livePoll;
      liveBtn.textContent = state.livePoll ? "Live Poll: ON" : "Live Poll: OFF";
      liveBtn.classList.toggle("active", state.livePoll);
      updateStatus();
    });
    runSelect.addEventListener("change", () => {
      state.currentRun = runSelect.value;
      state.followLatest = true;
      followBtn.classList.add("active");
      fetchTelemetry();
    });
    ["chk-subtrails", "chk-layers", "chk-ema", "chk-log", "chk-rel"].forEach(id => {
      document.getElementById(id).addEventListener("change", renderAll);
    });

    // Animation loop (auto-orbit + replay playback)
    setInterval(() => {
      if (state.playing && state.snapshots.length > 1) {
        state.selectedIdx = (state.selectedIdx + 1) % state.snapshots.length;
        stepSlider.value = String(state.selectedIdx);
        renderAll();
      }
    }, 220);

    function animLoop() {
      if (!state.dragging && document.getElementById("chk-orbit").checked && state.snapshots.length) {
        state.camYaw += 0.0035;
        renderSpace();
      }
      requestAnimationFrame(animLoop);
    }

    setInterval(() => {
      if (state.livePoll) fetchTelemetry();
    }, 1200);

    const onboardBackdrop = document.getElementById("onboard-backdrop");
    const ONBOARD_KEY = "lora_observatory_onboarded_v1";
    function openOnboarding() {
      onboardBackdrop.classList.add("open");
    }
    function closeOnboarding() {
      onboardBackdrop.classList.remove("open");
      try { localStorage.setItem(ONBOARD_KEY, "1"); } catch {}
    }
    document.getElementById("guide-open-btn").addEventListener("click", openOnboarding);
    document.getElementById("onboard-close-x").addEventListener("click", closeOnboarding);
    document.getElementById("onboard-start-btn").addEventListener("click", closeOnboarding);
    onboardBackdrop.addEventListener("click", (e) => {
      if (e.target === onboardBackdrop) closeOnboarding();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && onboardBackdrop.classList.contains("open")) closeOnboarding();
    });
    try {
      if (!isEmbedded && !localStorage.getItem(ONBOARD_KEY)) openOnboarding();
    } catch {
      if (!isEmbedded) openOnboarding();
    }

    window.renderAll = renderAll;
    window.addEventListener("resize", renderAll);
    fetchTelemetry();
    requestAnimationFrame(animLoop);
  </script>
</body>
</html>
"""


def read_run_telemetry(runs_dir: Path, requested_run: str | None) -> dict[str, Any]:
    if not runs_dir.exists():
        return {"runs": [], "run_id": None, "meta": None, "events": []}

    runs = sorted(
        [p.name for p in runs_dir.iterdir() if p.is_dir() and (p / "telemetry.jsonl").exists()],
        reverse=True,
    )
    active_file = runs_dir / "active_run.txt"
    active_name = active_file.read_text(encoding="utf-8").strip() if active_file.exists() else None

    if requested_run and requested_run in runs:
        run_id = requested_run
    elif active_name and active_name in runs:
        run_id = active_name
    elif runs:
        run_id = runs[0]
    else:
        return {"runs": [], "run_id": None, "meta": None, "events": []}

    run_path = runs_dir / run_id
    meta_path = run_path / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    events: list[dict[str, Any]] = []
    tel_path = run_path / "telemetry.jsonl"
    if tel_path.exists():
        for line in tel_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return {
        "runs": runs,
        "run_id": run_id,
        "meta": meta,
        "events": events,
    }


def make_handler(runs_dir: Path) -> type[BaseHTTPRequestHandler]:
    class VisualizerHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/telemetry":
                qs = parse_qs(parsed.query)
                req_run = qs.get("run", [None])[0]
                payload = read_run_telemetry(runs_dir, req_run)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return VisualizerHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live 3D Parameter Space & Loss Visualizer.")
    parser.add_argument("--runs-dir", default="./runs")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runs_dir))
    server.daemon_threads = True
    print(f"LoRA Parameter Space & Loss Observatory running at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
