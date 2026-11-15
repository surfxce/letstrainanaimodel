#!/usr/bin/env python3
"""Local side-by-side streaming comparison UI for MLX models.

Designed for 32 GB unified-memory Macs: by default, runs Model A then unloads it
before loading Model B (sequential mode) so peak memory stays around ~16 GB
with zero swap. Pass --simultaneous (or toggle in the UI) if you have closed
other apps and want both models kept resident in RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

from visualizer import HTML_PAGE as VISUALIZER_HTML, read_run_telemetry


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MLX Side-by-Side Model Arena</title>
  <style>
    :root {
      --bg: #0b0e14;
      --panel: #121722;
      --panel-2: #181f2e;
      --border: #252f45;
      --text: #e6ecf5;
      --muted: #8b98b0;
      --accent: #60a5fa;
      --accent-2: #34d399;
      --warn: #fbbf24;
      --danger: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: #070a14;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
      overflow-x: hidden;
    }
    .bg-aurora {
      position: fixed;
      inset: -18%;
      z-index: 0;
      pointer-events: none;
      background:
        radial-gradient(46% 42% at 22% 24%, rgba(56, 189, 248, 0.24), transparent 70%),
        radial-gradient(44% 44% at 78% 20%, rgba(139, 92, 246, 0.26), transparent 70%),
        radial-gradient(48% 44% at 58% 82%, rgba(59, 130, 246, 0.22), transparent 72%),
        radial-gradient(38% 36% at 16% 80%, rgba(168, 85, 247, 0.18), transparent 70%);
      filter: blur(42px) saturate(1.18);
      animation: auroraDrift 20s ease-in-out infinite alternate;
      will-change: transform;
    }
    .bg-aurora::after {
      content: "";
      position: absolute;
      inset: 8%;
      background:
        radial-gradient(36% 36% at 68% 34%, rgba(34, 211, 238, 0.16), transparent 68%),
        radial-gradient(40% 38% at 30% 68%, rgba(124, 58, 237, 0.18), transparent 68%);
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
      max-width: 1480px;
      margin: 0 auto;
      padding: 18px 20px 32px;
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
      font-size: 1.25rem;
      margin: 0;
      font-weight: 650;
      letter-spacing: -0.01em;
    }
    .sub {
      color: var(--muted);
      font-size: 0.85rem;
    }
    .card {
      background: rgba(17, 23, 39, 0.76);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 12px 34px rgba(2, 6, 23, 0.42);
    }
    .guide-btn {
      background: rgba(56, 189, 248, 0.12);
      color: #bae6fd;
      border: 1px solid rgba(56, 189, 248, 0.34);
      font-size: 0.8rem;
      padding: 6px 12px;
    }
    .guide-btn:hover { background: rgba(56, 189, 248, 0.22); }
    .onboard-backdrop {
      position: fixed;
      inset: 0;
      z-index: 100;
      background: rgba(5, 8, 18, 0.76);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .onboard-backdrop.open { display: flex; }
    .onboard-modal {
      width: min(760px, 100%);
      max-height: 90vh;
      overflow-y: auto;
      background: linear-gradient(155deg, rgba(16, 23, 42, 0.96), rgba(22, 18, 46, 0.95));
      border: 1px solid rgba(139, 92, 246, 0.38);
      border-radius: 16px;
      padding: 22px 24px;
      box-shadow: 0 24px 70px rgba(2, 6, 23, 0.75), 0 0 40px rgba(56, 189, 248, 0.12);
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
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #7dd3fc;
      background: rgba(56, 189, 248, 0.14);
      border: 1px solid rgba(56, 189, 248, 0.28);
      border-radius: 999px;
      padding: 3px 10px;
      margin-bottom: 8px;
    }
    .onboard-modal h2 {
      margin: 0;
      font-size: 21px;
      letter-spacing: -0.015em;
    }
    .onboard-lead {
      color: #cbd5e1;
      font-size: 13.5px;
      margin: 6px 0 16px;
    }
    .onboard-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    @media (max-width: 720px) {
      .onboard-grid { grid-template-columns: 1fr; }
    }
    .onboard-step {
      background: rgba(11, 16, 30, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 10px;
      padding: 12px;
    }
    .onboard-step h3 {
      margin: 0 0 4px;
      font-size: 13.5px;
      color: #e0f2fe;
    }
    .onboard-step p {
      margin: 0;
      font-size: 12.5px;
      color: var(--muted);
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
    .view-toggle {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      background: rgba(15, 22, 41, 0.78);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid rgba(148, 163, 184, 0.24);
      border-radius: 999px;
    }
    .view-toggle button {
      border: none;
      background: transparent;
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 0.84rem;
      font-weight: 600;
      transition: background 0.18s, color 0.18s, box-shadow 0.18s;
    }
    .view-toggle button:hover:not(:disabled) {
      color: var(--text);
      background: rgba(148, 163, 184, 0.12);
    }
    .view-toggle button.active {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.28), rgba(139, 92, 246, 0.34));
      color: #f8fafc;
      box-shadow: 0 0 0 1px rgba(125, 211, 252, 0.35);
    }
    #viz-view {
      width: 100%;
    }
    #viz-frame {
      width: 100%;
      min-height: 1120px;
      border: none;
      background: transparent;
      display: block;
    }
    .controls {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 14px;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    select, input[type="number"], textarea {
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.9rem;
    }
    select, input[type="number"] {
      padding: 6px 10px;
    }
    select.sample-select {
      flex: 1;
      min-width: 260px;
      max-width: 100%;
    }
    textarea {
      width: 100%;
      padding: 10px 12px;
      resize: vertical;
      min-height: 82px;
      line-height: 1.45;
    }
    textarea.system-box {
      min-height: 52px;
      font-size: 0.84rem;
      color: var(--muted);
    }
    button {
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 13px;
      font-size: 0.88rem;
      font-weight: 550;
      cursor: pointer;
      transition: border-color 0.15s, background 0.15s;
    }
    button:hover:not(:disabled) {
      border-color: var(--accent);
      background: #1d2638;
    }
    button.primary {
      background: #2563eb;
      border-color: #3b82f6;
      color: #fff;
    }
    button.primary:hover:not(:disabled) {
      background: #1d4ed8;
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    label.inline {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.84rem;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    @media (max-width: 960px) {
      .grid { grid-template-columns: 1fr; }
    }
    .pane {
      display: flex;
      flex-direction: column;
      min-height: 440px;
    }
    .pane-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 10px;
    }
    .model-title {
      font-weight: 650;
      font-size: 1rem;
    }
    .model-path {
      font-size: 0.78rem;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-all;
    }
    .badge {
      display: inline-block;
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--muted);
      white-space: nowrap;
    }
    .badge.running { border-color: var(--warn); color: var(--warn); }
    .badge.done { border-color: var(--accent-2); color: var(--accent-2); }
    .badge.err { border-color: var(--danger); color: var(--danger); }
    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
      font-size: 0.78rem;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .metrics span {
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 2px 7px;
    }
    .output {
      flex: 1;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 0.92rem;
      line-height: 1.55;
      color: var(--text);
      overflow-y: auto;
      max-height: 62vh;
      padding-right: 4px;
    }
    button.small {
      padding: 4px 10px;
      font-size: 0.78rem;
    }
    .pane-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .ckpt-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      padding: 6px 8px;
      background: rgba(15, 22, 41, 0.7);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 8px;
    }
    .ckpt-row label {
      font-size: 0.74rem;
      color: var(--muted);
      white-space: nowrap;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .ckpt-select {
      flex: 1;
      min-width: 140px;
      font-size: 0.8rem;
      padding: 4px 8px;
    }
    .ckpt-progress {
      width: 68px;
      height: 7px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.16);
      overflow: hidden;
      flex-shrink: 0;
    }
    .ckpt-progress-fill {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, #38bdf8, #a78bfa, #34d399);
      transition: width 0.2s ease;
    }
    .ckpt-pct {
      font-size: 0.74rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #cbd5e1;
      min-width: 34px;
      text-align: right;
    }
    .undergrad-box {
      display: none;
      flex-direction: column;
      gap: 6px;
      margin-top: 4px;
      padding: 10px 12px;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .undergrad-box.open { display: flex; }
    .undergrad-box label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent-2);
    }
    .save-status {
      font-size: 0.8rem;
      color: var(--accent-2);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    details.ref-box {
      margin-top: 8px;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 12px;
    }
    details.ref-box summary {
      cursor: pointer;
      font-size: 0.85rem;
      color: var(--accent);
      font-weight: 550;
    }
    .ref-body {
      margin-top: 8px;
      white-space: pre-wrap;
      font-size: 0.88rem;
      color: #cbd5e1;
      border-top: 1px solid var(--border);
      padding-top: 8px;
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
          <h2 id="onboard-title">Side-by-Side Local LLM Evaluation Lab</h2>
        </div>
        <button type="button" class="secondary" id="onboard-close-x" aria-label="Close guide">Close</button>
      </div>
      <p class="onboard-lead">
        Compare any two local Apple Silicon MLX models head-to-head on the same prompt, test yourself alongside them, and save structured markdown comparisons.
      </p>
      <div class="onboard-grid">
        <div class="onboard-step">
          <h3>1. Pick or Write a Prompt</h3>
          <p>Load an item from your validation dataset dropdown or write a custom system prompt and user question. Hit Cmd+Enter to run.</p>
        </div>
        <div class="onboard-step">
          <h3>2. Choose Memory &amp; Streaming</h3>
          <p>Keep both models in RAM for instant switching, and toggle <b>Stream both live</b> to watch both models generate side-by-side simultaneously.</p>
        </div>
        <div class="onboard-step">
          <h3>3. Enable Undergrad Mode</h3>
          <p>Toggle <b>Undergrad mode</b> to write your own answer blind before or after viewing the models and dataset reference answer.</p>
        </div>
        <div class="onboard-step">
          <h3>4. Export Comparisons</h3>
          <p>Click <b>Save individual</b> for one markdown report per prompt or <b>Save to log</b> to append entries to a running log file.</p>
        </div>
      </div>
      <div class="onboard-foot">
        <span class="sub">Reopen anytime via the Guide button in the top right.</span>
        <button type="button" id="onboard-start-btn">Start Comparing</button>
      </div>
    </div>
  </div>
  <div class="wrap">
    <header>
      <div>
        <h1>MLX AI Workbench</h1>
        <div class="sub" id="subtitle">Loading configuration...</div>
      </div>
      <div class="view-toggle" role="tablist" aria-label="Workspace View">
        <button type="button" id="tab-arena" class="active" role="tab" aria-selected="true">Compare Outputs</button>
        <button type="button" id="tab-viz" role="tab" aria-selected="false">Training Visualiser</button>
      </div>
      <div class="row">
        <button type="button" class="guide-btn" id="guide-open-btn">Guide</button>
        <span class="badge" id="global-status">Idle</span>
      </div>
    </header>

    <div id="arena-view">

    <section class="card controls">
      <div class="row">
        <label class="inline" for="sample-select">Validation Sample:</label>
        <select id="sample-select" class="sample-select"></select>
        <button id="prev-btn" type="button">Prev</button>
        <button id="next-btn" type="button">Next</button>
        <button id="rand-btn" type="button">Random</button>
      </div>

      <details id="system-details">
        <summary style="cursor:pointer;font-size:0.84rem;color:var(--muted);">System prompt (optional, auto-loaded from sample)</summary>
        <div style="margin-top:6px;">
          <textarea id="system-input" class="system-box" placeholder="Optional system prompt..."></textarea>
        </div>
      </details>

      <textarea id="prompt-input" placeholder="Ask a course question or pick a validation sample above... (Cmd/Ctrl+Enter to run)"></textarea>

      <div class="row" style="justify-content: space-between;">
        <div class="row">
          <label class="inline">Max tokens
            <input id="max-tokens" type="number" min="32" max="4096" step="64" value="1024" style="width:82px;" />
          </label>
          <label class="inline">Temp
            <input id="temp" type="number" min="0" max="2" step="0.1" value="0.0" style="width:68px;" />
          </label>
          <label class="inline">
            <input id="thinking" type="checkbox" /> Enable thinking
          </label>
          <label class="inline" title="Keep both models loaded in RAM at once (~25 GB). Leave unchecked on 32 GB Macs with other apps open.">
            <input id="simultaneous" type="checkbox" /> Keep both in RAM
          </label>
          <label class="inline" title="Interleave token streaming from both models at the same time (automatically keeps both in RAM).">
            <input id="stream-both" type="checkbox" /> Stream both live
          </label>
          <label class="inline" title="Show a box to write your own answer to the question before or alongside the models.">
            <input id="undergrad-mode" type="checkbox" /> Undergrad mode
          </label>
        </div>
        <div class="row">
          <button id="save-ind-btn" type="button">Save Individual</button>
          <button id="save-log-btn" type="button">Save to Log</button>
          <button id="clear-btn" type="button">Clear Outputs</button>
          <button id="run-btn" class="primary" type="button">Compare Side-by-Side</button>
        </div>
      </div>

      <div class="row" style="justify-content: flex-end; min-height: 0;">
        <span class="save-status" id="save-status"></span>
      </div>

      <div class="undergrad-box" id="undergrad-box">
        <label for="undergrad-input">Your Answer (Undergrad Mode)</label>
        <textarea id="undergrad-input" placeholder="Write your own answer to the question here..."></textarea>
      </div>

      <details class="ref-box" id="ref-details" style="display:none;">
        <summary>Reveal Ground-Truth Reference Answer</summary>
        <div class="ref-body" id="ref-body"></div>
      </details>
    </section>

    <section class="grid">
      <div class="card pane">
        <div class="pane-head">
          <div>
            <div class="model-title" id="left-title">Model A</div>
            <div class="model-path" id="left-path"></div>
          </div>
          <div class="pane-actions">
            <button class="small copy-btn" data-side="left" type="button">Copy</button>
            <span class="badge" id="left-status">Idle</span>
          </div>
        </div>
        <div class="ckpt-row">
          <label for="left-adapter">Checkpoint</label>
          <select id="left-adapter" class="ckpt-select"></select>
          <div class="ckpt-progress" title="Training progress at selected checkpoint"><div class="ckpt-progress-fill" id="left-ckpt-bar"></div></div>
          <span class="ckpt-pct" id="left-ckpt-pct">0%</span>
          <button class="small refresh-ckpts-btn" type="button" title="Rescan adapter checkpoints">&#8635;</button>
        </div>
        <div class="metrics" id="left-metrics"></div>
        <div class="output" id="left-output"></div>
      </div>

      <div class="card pane">
        <div class="pane-head">
          <div>
            <div class="model-title" id="right-title">Model B</div>
            <div class="model-path" id="right-path"></div>
          </div>
          <div class="pane-actions">
            <button class="small copy-btn" data-side="right" type="button">Copy</button>
            <span class="badge" id="right-status">Idle</span>
          </div>
        </div>
        <div class="ckpt-row">
          <label for="right-adapter">Checkpoint</label>
          <select id="right-adapter" class="ckpt-select"></select>
          <div class="ckpt-progress" title="Training progress at selected checkpoint"><div class="ckpt-progress-fill" id="right-ckpt-bar"></div></div>
          <span class="ckpt-pct" id="right-ckpt-pct">0%</span>
          <button class="small refresh-ckpts-btn" type="button" title="Rescan adapter checkpoints">&#8635;</button>
        </div>
        <div class="metrics" id="right-metrics"></div>
        <div class="output" id="right-output"></div>
      </div>
    </section>
    </div>

    <div id="viz-view" style="display:none;">
      <iframe id="viz-frame" title="LoRA Parameter Space &amp; Loss Observatory"></iframe>
    </div>
  </div>

  <script>
    const sampleSelect = document.getElementById("sample-select");
    const systemDetails = document.getElementById("system-details");
    const systemInput = document.getElementById("system-input");
    const promptInput = document.getElementById("prompt-input");
    const refDetails = document.getElementById("ref-details");
    const refBody = document.getElementById("ref-body");
    const runBtn = document.getElementById("run-btn");
    const simultaneousBox = document.getElementById("simultaneous");
    const streamBothBox = document.getElementById("stream-both");
    const undergradBox = document.getElementById("undergrad-mode");
    const undergradPanel = document.getElementById("undergrad-box");
    const undergradInput = document.getElementById("undergrad-input");
    const saveStatus = document.getElementById("save-status");
    const globalStatus = document.getElementById("global-status");
    const leftAdapterSel = document.getElementById("left-adapter");
    const rightAdapterSel = document.getElementById("right-adapter");

    let lastMetrics = { left: null, right: null };
    let checkpoints = [];
    let arenaCfg = null;

    streamBothBox.addEventListener("change", () => {
      if (streamBothBox.checked) simultaneousBox.checked = true;
    });
    simultaneousBox.addEventListener("change", () => {
      if (!simultaneousBox.checked) streamBothBox.checked = false;
    });
    undergradBox.addEventListener("change", () => {
      undergradPanel.classList.toggle("open", undergradBox.checked);
    });

    document.querySelectorAll(".copy-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const side = btn.dataset.side;
        const text = document.getElementById(side + "-output").textContent;
        if (!text) return;
        try {
          await navigator.clipboard.writeText(text);
        } catch {
          const ta = document.createElement("textarea");
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          ta.remove();
        }
        const orig = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => { btn.textContent = orig; }, 1400);
      });
    });

    let samples = [];
    let currentSource = null;

    function setBadge(id, text, cls = "") {
      const el = document.getElementById(id);
      el.textContent = text;
      el.className = "badge" + (cls ? " " + cls : "");
    }

    function renderMetrics(side, m) {
      const el = document.getElementById(side + "-metrics");
      if (!m) { el.innerHTML = ""; return; }
      el.innerHTML = [
        `<span>load ${m.load_time_s}s</span>`,
        `<span>prefill ${m.prompt_tps} tok/s</span>`,
        `<span>gen ${m.generation_tps} tok/s</span>`,
        `<span>${m.generation_tokens} tok</span>`,
        `<span>peak ${m.peak_memory_gb} GB</span>`,
        `<span>${m.wall_time_s}s</span>`,
      ].join("");
    }

    function loadSample(idx) {
      undergradInput.value = "";
      saveStatus.textContent = "";
      if (idx < 0 || idx >= samples.length) {
        systemInput.value = "";
        promptInput.value = "";
        refDetails.style.display = "none";
        refBody.textContent = "";
        return;
      }
      const s = samples[idx];
      sampleSelect.value = String(idx);
      systemInput.value = s.system || "";
      systemDetails.open = Boolean(s.system);
      promptInput.value = s.prompt;
      if (s.reference) {
        refBody.textContent = s.reference;
        refDetails.open = false;
        refDetails.style.display = "block";
      } else {
        refDetails.style.display = "none";
      }
    }

    function updateCkptProgress(side) {
      const sel = side === "left" ? leftAdapterSel : rightAdapterSel;
      const bar = document.getElementById(side + "-ckpt-bar");
      const pctEl = document.getElementById(side + "-ckpt-pct");
      const item = checkpoints.find(c => c.id === sel.value);
      const pct = item && typeof item.pct === "number" ? item.pct : 0;
      bar.style.width = pct + "%";
      pctEl.textContent = pct + "%";
      if (arenaCfg) {
        const baseModel = side === "left" ? arenaCfg.left_model : arenaCfg.right_model;
        document.getElementById(side + "-path").textContent = baseModel + (sel.value ? " + " + sel.value : "");
      }
    }

    function populateAdapterSelect(side, modelPath, preferredVal) {
      const sel = side === "left" ? leftAdapterSel : rightAdapterSel;
      const prev = preferredVal !== undefined ? preferredVal : sel.value;
      const modelBase = (modelPath || "").split("/").pop();
      const compatible = checkpoints.filter(c => {
        if (!c.id || !c.model) return true;
        return c.model === modelPath || c.model.split("/").pop() === modelBase;
      });
      sel.innerHTML = compatible.map(c => `<option value="${c.id}">${c.label}</option>`).join("");
      if (compatible.some(c => c.id === prev)) {
        sel.value = prev;
      } else if (prev) {
        const cleanPrev = prev.replace(/^\.\//, "").replace(/\/+$/, "");
        const matched = compatible.find(c => c.id === `${cleanPrev}/adapters.safetensors`);
        sel.value = matched ? matched.id : "";
      } else {
        sel.value = "";
      }
      updateCkptProgress(side);
    }

    async function refreshCheckpoints(defaultLeft = undefined) {
      try {
        const data = await (await fetch("/api/checkpoints")).json();
        checkpoints = data.checkpoints || [];
        if (arenaCfg) {
          populateAdapterSelect("left", arenaCfg.left_model, defaultLeft);
          populateAdapterSelect("right", arenaCfg.right_model);
        }
      } catch {}
    }

    leftAdapterSel.addEventListener("change", () => updateCkptProgress("left"));
    rightAdapterSel.addEventListener("change", () => updateCkptProgress("right"));
    document.querySelectorAll(".refresh-ckpts-btn").forEach(btn => {
      btn.addEventListener("click", () => refreshCheckpoints());
    });

    async function init() {
      const cfg = await (await fetch("/api/config")).json();
      arenaCfg = cfg;
      document.getElementById("left-title").textContent = cfg.left_label;
      document.getElementById("left-path").textContent = cfg.left_model + (cfg.adapter_path ? " + " + cfg.adapter_path : "");
      document.getElementById("right-title").textContent = cfg.right_label;
      document.getElementById("right-path").textContent = cfg.right_model;
      document.getElementById("simultaneous").checked = Boolean(cfg.simultaneous);
      document.getElementById("subtitle").textContent =
        `Dataset: ${cfg.prompts_file || "none"} (${cfg.sample_count} samples)`;

      await refreshCheckpoints(cfg.adapter_path || "");

      const data = await (await fetch("/api/samples")).json();
      samples = data.samples || [];
      sampleSelect.innerHTML = '<option value="-1">-- Custom prompt --</option>' +
        samples.map((s, i) => {
          const preview = s.prompt.replace(/\s+/g, " ").slice(0, 110);
          return `<option value="${i}">#${i + 1}: ${preview}${s.prompt.length > 110 ? "..." : ""}</option>`;
        }).join("");

      if (samples.length > 0) loadSample(0);
    }

    sampleSelect.addEventListener("change", () => loadSample(Number(sampleSelect.value)));
    document.getElementById("prev-btn").addEventListener("click", () => {
      const cur = Number(sampleSelect.value);
      loadSample(cur > 0 ? cur - 1 : 0);
    });
    document.getElementById("next-btn").addEventListener("click", () => {
      const cur = Number(sampleSelect.value);
      if (cur + 1 < samples.length) loadSample(cur + 1);
    });
    document.getElementById("rand-btn").addEventListener("click", () => {
      if (samples.length > 0) loadSample(Math.floor(Math.random() * samples.length));
    });
    document.getElementById("clear-btn").addEventListener("click", () => {
      document.getElementById("left-output").textContent = "";
      document.getElementById("right-output").textContent = "";
      lastMetrics = { left: null, right: null };
      renderMetrics("left", null);
      renderMetrics("right", null);
      setBadge("left-status", "Idle");
      setBadge("right-status", "Idle");
      saveStatus.textContent = "";
    });

    function paneLabelWithCkpt(side) {
      const base = document.getElementById(side + "-title").textContent;
      const sel = side === "left" ? leftAdapterSel : rightAdapterSel;
      if (!sel.value) return base;
      const opt = sel.options[sel.selectedIndex];
      return opt ? `${base} (${opt.textContent})` : base;
    }

    async function saveComparison(mode) {
      const payload = {
        mode,
        sample_index: Number(sampleSelect.value),
        system_prompt: systemInput.value.trim(),
        question: promptInput.value.trim(),
        reference_answer: refBody.textContent.trim(),
        left_label: paneLabelWithCkpt("left"),
        left_output: document.getElementById("left-output").textContent,
        left_metrics: lastMetrics.left,
        right_label: paneLabelWithCkpt("right"),
        right_output: document.getElementById("right-output").textContent,
        right_metrics: lastMetrics.right,
        undergrad_response: undergradInput.value.trim(),
      };
      saveStatus.textContent = "Saving...";
      try {
        const res = await fetch("/api/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        saveStatus.textContent = data.ok ? data.message : ("Save failed: " + data.error);
      } catch (err) {
        saveStatus.textContent = "Save failed: " + err;
      }
    }

    document.getElementById("save-ind-btn").addEventListener("click", () => saveComparison("individual"));
    document.getElementById("save-log-btn").addEventListener("click", () => saveComparison("log"));

    function runComparison() {
      const prompt = promptInput.value.trim();
      if (!prompt) return;
      if (currentSource) {
        currentSource.close();
        currentSource = null;
      }

      document.getElementById("left-output").textContent = "";
      document.getElementById("right-output").textContent = "";
      lastMetrics = { left: null, right: null };
      renderMetrics("left", null);
      renderMetrics("right", null);
      saveStatus.textContent = "";
      setBadge("left-status", "Queued");
      setBadge("right-status", "Queued");
      setBadge("global-status", "Running...", "running");
      runBtn.disabled = true;

      const params = new URLSearchParams({
        prompt,
        system: systemInput.value.trim(),
        max_tokens: document.getElementById("max-tokens").value || "1024",
        temp: document.getElementById("temp").value || "0.0",
        thinking: document.getElementById("thinking").checked ? "1" : "0",
        simultaneous: simultaneousBox.checked ? "1" : "0",
        stream_both: streamBothBox.checked ? "1" : "0",
        left_adapter: leftAdapterSel.value || "",
        right_adapter: rightAdapterSel.value || "",
      });

      const es = new EventSource("/api/stream?" + params.toString());
      currentSource = es;

      es.addEventListener("status", (ev) => {
        const d = JSON.parse(ev.data);
        setBadge(d.side + "-status", d.text, d.state || "running");
      });

      es.addEventListener("token", (ev) => {
        const d = JSON.parse(ev.data);
        const out = document.getElementById(d.side + "-output");
        out.textContent += d.text;
        out.scrollTop = out.scrollHeight;
      });

      es.addEventListener("metrics", (ev) => {
        const d = JSON.parse(ev.data);
        lastMetrics[d.side] = d;
        renderMetrics(d.side, d);
        setBadge(d.side + "-status", "Done", "done");
      });

      es.addEventListener("error_msg", (ev) => {
        const d = JSON.parse(ev.data);
        setBadge(d.side ? d.side + "-status" : "global-status", "Error", "err");
        if (d.side) {
          document.getElementById(d.side + "-output").textContent += "\n[Error] " + d.message;
        }
      });

      es.addEventListener("done", () => {
        setBadge("global-status", "Complete", "done");
        runBtn.disabled = false;
        es.close();
        currentSource = null;
      });

      es.onerror = () => {
        setBadge("global-status", "Disconnected", "err");
        runBtn.disabled = false;
        es.close();
        currentSource = null;
      };
    }

    runBtn.addEventListener("click", runComparison);
    promptInput.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        runComparison();
      }
    });

    const onboardBackdrop = document.getElementById("onboard-backdrop");
    const ONBOARD_KEY = "mlx_compare_onboarded_v1";
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
      if (!localStorage.getItem(ONBOARD_KEY)) openOnboarding();
    } catch {
      openOnboarding();
    }

    const tabArena = document.getElementById("tab-arena");
    const tabViz = document.getElementById("tab-viz");
    const arenaView = document.getElementById("arena-view");
    const vizView = document.getElementById("viz-view");
    const vizFrame = document.getElementById("viz-frame");
    const VIEW_KEY = "mlx_workbench_view_v1";
    let vizLoaded = false;

    function switchMode(mode) {
      const showViz = mode === "viz";
      tabArena.classList.toggle("active", !showViz);
      tabArena.setAttribute("aria-selected", String(!showViz));
      tabViz.classList.toggle("active", showViz);
      tabViz.setAttribute("aria-selected", String(showViz));
      arenaView.style.display = showViz ? "none" : "block";
      vizView.style.display = showViz ? "block" : "none";
      document.title = showViz ? "Training Visualiser — MLX Workbench" : "Compare Outputs — MLX Workbench";
      try {
        localStorage.setItem(VIEW_KEY, mode);
      } catch {}
      if (showViz) {
        if (!vizLoaded) {
          vizFrame.src = "/visualizer?embed=1";
          vizLoaded = true;
        } else {
          try {
            if (vizFrame.contentWindow && typeof vizFrame.contentWindow.renderAll === "function") {
              vizFrame.contentWindow.renderAll();
            }
          } catch {}
        }
      }
    }

    tabArena.addEventListener("click", () => {
      history.replaceState(null, "", "#compare");
      switchMode("arena");
      refreshCheckpoints();
    });
    tabViz.addEventListener("click", () => {
      history.replaceState(null, "", "#visualiser");
      switchMode("viz");
    });
    window.addEventListener("message", (ev) => {
      if (ev.data && ev.data.type === "viz-height" && typeof ev.data.height === "number") {
        vizFrame.style.height = Math.max(960, Math.ceil(ev.data.height) + 12) + "px";
      }
    });

    const initialHash = window.location.hash;
    let initialView = "arena";
    if (initialHash === "#visualiser") initialView = "viz";
    else if (initialHash === "#compare") initialView = "arena";
    else {
      try {
        if (localStorage.getItem(VIEW_KEY) === "viz") initialView = "viz";
      } catch {}
    }
    if (initialView === "viz") switchMode("viz");

    init();
  </script>
</body>
</html>
"""


def load_samples(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    items: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if "messages" in obj:
                sys_msgs = [m["content"] for m in obj["messages"] if m.get("role") == "system"]
                usr_msgs = [m["content"] for m in obj["messages"] if m.get("role") == "user"]
                ast_msgs = [m["content"] for m in obj["messages"] if m.get("role") == "assistant"]
                if not usr_msgs:
                    continue
                items.append(
                    {
                        "system": sys_msgs[-1] if sys_msgs else "",
                        "prompt": usr_msgs[-1],
                        "reference": ast_msgs[-1] if ast_msgs else "",
                    }
                )
            elif "prompt" in obj:
                items.append(
                    {
                        "system": obj.get("system", ""),
                        "prompt": obj["prompt"],
                        "reference": obj.get("reference", obj.get("completion", "")),
                    }
                )
    return items


STEP_CKPT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")


def discover_checkpoints(extra_paths: list[str | None] | None = None) -> list[dict[str, Any]]:
    """Scan local adapter folders and list checkpoints with training step progress."""
    results: list[dict[str, Any]] = [
        {
            "id": "",
            "label": "Regular Base Model — Step 0 (Untrained)",
            "model": "",
            "step": 0,
            "total_iters": None,
            "pct": 0,
        }
    ]

    candidate_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    raw_dirs: list[Path] = [Path("./adapters"), Path("./adapters_smoke")]
    raw_dirs.extend(sorted(Path(".").glob("adapters_*")))
    for extra in extra_paths or []:
        if extra:
            p = Path(extra)
            raw_dirs.append(p.parent if p.is_file() else p)

    for d in raw_dirs:
        if not d.exists() or not d.is_dir():
            continue
        res = d.resolve()
        if res not in seen_dirs:
            seen_dirs.add(res)
            candidate_dirs.append(d)

    for d in candidate_dirs:
        cfg_file = d / "adapter_config.json"
        if not cfg_file.exists():
            continue
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        total_iters = cfg.get("iters") if isinstance(cfg.get("iters"), int) and cfg.get("iters") > 0 else None
        model_hint = str(cfg.get("model") or "")

        step_files: list[tuple[int, Path]] = []
        for f in d.iterdir():
            if not f.is_file():
                continue
            m = STEP_CKPT_RE.match(f.name)
            if m:
                step_files.append((int(m.group(1)), f))
        step_files.sort(key=lambda x: x[0])

        for step, f in step_files:
            if total_iters:
                pct = min(100, round((step / total_iters) * 100))
                label = f"[{d.name}] Step {step:,} / {total_iters:,} ({pct}%)"
            else:
                pct = 100
                label = f"[{d.name}] Step {step:,}"
            results.append(
                {
                    "id": str(f),
                    "label": label,
                    "model": model_hint,
                    "step": step,
                    "total_iters": total_iters,
                    "pct": pct,
                }
            )

        latest_file = d / "adapters.safetensors"
        if latest_file.exists():
            last_step = step_files[-1][0] if step_files else total_iters
            if total_iters and last_step and last_step >= total_iters:
                lbl = f"[{d.name}] Final / Latest — Step {total_iters:,} / {total_iters:,} (100%)"
                pct_val = 100
            elif total_iters and last_step:
                pct_val = min(100, round((last_step / total_iters) * 100))
                lbl = f"[{d.name}] Latest saved — Step {last_step:,} / {total_iters:,} ({pct_val}%)"
            else:
                pct_val = 100
                lbl = f"[{d.name}] Latest (adapters.safetensors)"
            results.append(
                {
                    "id": str(latest_file),
                    "label": lbl,
                    "model": model_hint,
                    "step": last_step,
                    "total_iters": total_iters,
                    "pct": pct_val,
                }
            )

    return results


class ModelManager:
    """Manages loading, unloading, and hot-swapping LoRA checkpoints safely under a lock."""

    def __init__(self, left_spec: dict[str, Any], right_spec: dict[str, Any]) -> None:
        self.lock = threading.Lock()
        self.specs = {"left": left_spec, "right": right_spec}
        self.loaded: dict[str, tuple[Any, Any, str | None, str | None]] = {}

    @staticmethod
    def _normalize_adapter(adapter_spec: str | None) -> tuple[str | None, str | None]:
        if not adapter_spec:
            return None, None
        p = Path(adapter_spec)
        if p.is_dir():
            latest = p / "adapters.safetensors"
            target = str(latest.resolve()) if latest.exists() else str(p.resolve())
            return target, str(p.resolve())
        return str(p.resolve()), str(p.parent.resolve())

    def _load_fresh(self, model_path: str, adapter_spec: str | None) -> tuple[Any, Any, str | None]:
        if not adapter_spec:
            model, tokenizer = load(model_path)
            return model, tokenizer, None
        p = Path(adapter_spec)
        if p.is_dir():
            model, tokenizer = load(model_path, adapter_path=str(p))
            return model, tokenizer, str(p.resolve())

        cfg_path = p.parent / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        fine_tune_type = cfg.get("fine_tune_type", "lora")
        model, tokenizer = load(model_path)
        if fine_tune_type != "full":
            linear_to_lora_layers(
                model,
                cfg["num_layers"],
                cfg["lora_parameters"],
                use_dora=(fine_tune_type == "dora"),
            )
        weights = mx.load(str(p))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        return model, tokenizer, str(p.parent.resolve())

    def get_model(
        self,
        side: str,
        keep_both: bool,
        adapter_spec: str | None = None,
    ) -> tuple[Any, Any, float]:
        req_adapter, req_cfg_dir = self._normalize_adapter(adapter_spec)

        if side in self.loaded:
            model, tokenizer, active_adapter, active_cfg_dir = self.loaded[side]
            if active_adapter == req_adapter:
                return model, tokenizer, 0.0
            # Fast hot-swap when staying on LoRA checkpoints from the same adapter folder
            if (
                req_adapter
                and req_cfg_dir
                and active_cfg_dir == req_cfg_dir
                and Path(req_adapter).is_file()
            ):
                t0 = time.perf_counter()
                weights = mx.load(req_adapter)
                model.load_weights(list(weights.items()), strict=False)
                mx.eval(model.parameters())
                dt = round(time.perf_counter() - t0, 2)
                self.loaded[side] = (model, tokenizer, req_adapter, req_cfg_dir)
                return model, tokenizer, dt
            # Base <-> LoRA or cross-folder change requires clean base reload
            del self.loaded[side]
            gc.collect()
            mx.clear_cache()

        if not keep_both:
            other = "right" if side == "left" else "left"
            if other in self.loaded:
                del self.loaded[other]
                gc.collect()
                mx.clear_cache()

        spec = self.specs[side]
        t0 = time.perf_counter()
        model, tokenizer, cfg_dir = self._load_fresh(spec["model_path"], adapter_spec)
        load_time_s = round(time.perf_counter() - t0, 2)
        self.loaded[side] = (model, tokenizer, req_adapter, cfg_dir)
        return model, tokenizer, load_time_s


def format_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    system_prompt: str,
    enable_thinking: bool,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if not hasattr(tokenizer, "apply_chat_template"):
        return user_prompt
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _build_metrics(
    side: str,
    load_time_s: float,
    last_resp: Any,
    wall_s: float,
) -> dict[str, Any]:
    return {
        "side": side,
        "load_time_s": load_time_s,
        "prompt_tps": round(last_resp.prompt_tps, 2) if last_resp else 0.0,
        "generation_tps": round(last_resp.generation_tps, 2) if last_resp else 0.0,
        "generation_tokens": last_resp.generation_tokens if last_resp else 0,
        "peak_memory_gb": round(last_resp.peak_memory, 2) if last_resp else 0.0,
        "wall_time_s": wall_s,
    }


_SAVE_LOCK = threading.Lock()


def _format_metrics_line(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return ""
    return (
        f"_Metrics: load {metrics.get('load_time_s', 0)}s | "
        f"prefill {metrics.get('prompt_tps', 0)} tok/s | "
        f"gen {metrics.get('generation_tps', 0)} tok/s | "
        f"{metrics.get('generation_tokens', 0)} tok | "
        f"peak {metrics.get('peak_memory_gb', 0)} GB | "
        f"wall {metrics.get('wall_time_s', 0)}s_\n\n"
    )


def format_saved_entry(payload: dict[str, Any], timestamp_str: str) -> str:
    sample_idx = payload.get("sample_index", -1)
    sample_note = f" (Validation Sample #{sample_idx + 1})" if isinstance(sample_idx, int) and sample_idx >= 0 else " (Custom Prompt)"
    lines = [
        f"## Comparison Entry — {timestamp_str}{sample_note}",
        "",
        "### System Prompt",
        payload.get("system_prompt") or "_None_",
        "",
        "### Question",
        payload.get("question") or "_Empty_",
        "",
        "### Ground-Truth Reference Answer",
        payload.get("reference_answer") or "_None_",
        "",
        f"### {payload.get('left_label', 'Model A')} Output",
        _format_metrics_line(payload.get("left_metrics")) + (payload.get("left_output") or "_No output_"),
        "",
        f"### {payload.get('right_label', 'Model B')} Output",
        _format_metrics_line(payload.get("right_metrics")) + (payload.get("right_output") or "_No output_"),
        "",
    ]
    undergrad = (payload.get("undergrad_response") or "").strip()
    if undergrad:
        lines.extend(["### My Response (Undergrad Mode)", undergrad, ""])
    return "\n".join(lines)


def save_comparison_record(payload: dict[str, Any], save_dir: Path = Path("./saved_comparisons")) -> dict[str, Any]:
    mode = payload.get("mode", "individual")
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    entry_md = format_saved_entry(payload, timestamp_str)

    with _SAVE_LOCK:
        save_dir.mkdir(parents=True, exist_ok=True)
        if mode == "log":
            log_path = save_dir / "comparison_log.md"
            is_new = not log_path.exists() or log_path.stat().st_size == 0
            with log_path.open("a", encoding="utf-8") as f:
                if is_new:
                    f.write("# Model Comparison Log\n\n")
                else:
                    f.write("---\n\n")
                f.write(entry_md + "\n")
            return {"ok": True, "path": str(log_path), "message": f"Appended to {log_path}"}

        stamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = save_dir / f"comparison_{stamp}.md"
        counter = 1
        while out_path.exists():
            out_path = save_dir / f"comparison_{stamp}_{counter}.md"
            counter += 1
        out_path.write_text(entry_md, encoding="utf-8")
        return {"ok": True, "path": str(out_path), "message": f"Saved {out_path}"}


def make_handler(
    manager: ModelManager,
    config_payload: dict[str, Any],
    samples: list[dict[str, str]],
    runs_dir: Path = Path("./runs"),
) -> type[BaseHTTPRequestHandler]:
    class ArenaHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse(self, event: str, data: dict[str, Any]) -> None:
            msg = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
            self.wfile.write(msg)
            self.wfile.flush()

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

            if parsed.path == "/visualizer":
                body = VISUALIZER_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/telemetry":
                qs = parse_qs(parsed.query)
                req_run = qs.get("run", [None])[0]
                self._send_json(read_run_telemetry(runs_dir, req_run))
                return

            if parsed.path == "/api/config":
                self._send_json(config_payload)
                return

            if parsed.path == "/api/samples":
                self._send_json({"samples": samples})
                return

            if parsed.path == "/api/checkpoints":
                extra = [config_payload.get("adapter_path")]
                self._send_json({"checkpoints": discover_checkpoints(extra)})
                return

            if parsed.path == "/api/stream":
                qs = parse_qs(parsed.query)
                user_prompt = qs.get("prompt", [""])[0]
                system_prompt = qs.get("system", [""])[0]
                max_tokens = max(32, min(4096, int(qs.get("max_tokens", ["1024"])[0])))
                temp = max(0.0, min(2.0, float(qs.get("temp", ["0.0"])[0])))
                thinking = qs.get("thinking", ["0"])[0] == "1"
                stream_both = qs.get("stream_both", ["0"])[0] == "1"
                simultaneous = (qs.get("simultaneous", ["0"])[0] == "1") or stream_both

                allowed_ckpts = {
                    c["id"]: c["id"]
                    for c in discover_checkpoints([config_payload.get("adapter_path")])
                    if c.get("id")
                }
                cli_left = config_payload.get("adapter_path")
                if cli_left:
                    allowed_ckpts[cli_left] = cli_left
                req_adapters = {
                    "left": allowed_ckpts.get(qs.get("left_adapter", [""])[0], None),
                    "right": allowed_ckpts.get(qs.get("right_adapter", [""])[0], None),
                }

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                sampler = make_sampler(temp=temp)
                try:
                    with manager.lock:
                        if stream_both:
                            gens: dict[str, Any] = {}
                            load_times: dict[str, float] = {}
                            t_starts: dict[str, float] = {}
                            last_resps: dict[str, Any] = {"left": None, "right": None}
                            active: list[str] = []
                            try:
                                mx.reset_peak_memory()
                                for side in ("left", "right"):
                                    try:
                                        self._send_sse("status", {"side": side, "text": "Loading model..."})
                                        model, tokenizer, load_time_s = manager.get_model(
                                            side, True, adapter_spec=req_adapters[side]
                                        )
                                        load_times[side] = load_time_s
                                        formatted = format_chat_prompt(
                                            tokenizer, user_prompt, system_prompt, thinking
                                        )
                                        stream = mx.new_stream(mx.default_device())
                                        gens[side] = stream_generate(
                                            model,
                                            tokenizer,
                                            prompt=formatted,
                                            max_tokens=max_tokens,
                                            sampler=sampler,
                                            stream=stream,
                                        )
                                        t_starts[side] = time.perf_counter()
                                        active.append(side)
                                        self._send_sse("status", {"side": side, "text": "Generating..."})
                                    except Exception as exc:
                                        self._send_sse("error_msg", {"side": side, "message": str(exc)})

                                while active:
                                    for side in list(active):
                                        try:
                                            resp = next(gens[side])
                                            last_resps[side] = resp
                                            if resp.text:
                                                self._send_sse("token", {"side": side, "text": resp.text})
                                        except StopIteration:
                                            wall_s = round(time.perf_counter() - t_starts[side], 2)
                                            self._send_sse(
                                                "metrics",
                                                _build_metrics(side, load_times[side], last_resps[side], wall_s),
                                            )
                                            active.remove(side)
                                        except Exception as exc:
                                            self._send_sse("error_msg", {"side": side, "message": str(exc)})
                                            active.remove(side)
                            finally:
                                for gen in gens.values():
                                    try:
                                        gen.close()
                                    except Exception:
                                        pass
                        else:
                            for side in ("left", "right"):
                                try:
                                    self._send_sse("status", {"side": side, "text": "Loading model..."})
                                    model, tokenizer, load_time_s = manager.get_model(
                                        side, simultaneous, adapter_spec=req_adapters[side]
                                    )
                                    self._send_sse("status", {"side": side, "text": "Generating..."})

                                    formatted = format_chat_prompt(
                                        tokenizer, user_prompt, system_prompt, thinking
                                    )
                                    mx.reset_peak_memory()
                                    t0 = time.perf_counter()
                                    last_resp = None
                                    for resp in stream_generate(
                                        model,
                                        tokenizer,
                                        prompt=formatted,
                                        max_tokens=max_tokens,
                                        sampler=sampler,
                                    ):
                                        last_resp = resp
                                        if resp.text:
                                            self._send_sse("token", {"side": side, "text": resp.text})

                                    wall_s = round(time.perf_counter() - t0, 2)
                                    self._send_sse(
                                        "metrics",
                                        _build_metrics(side, load_time_s, last_resp, wall_s),
                                    )
                                except (BrokenPipeError, ConnectionResetError):
                                    return
                                except Exception as exc:
                                    self._send_sse("error_msg", {"side": side, "message": str(exc)})
                        self._send_sse("done", {})
                except (BrokenPipeError, ConnectionResetError):
                    return
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                result = save_comparison_record(payload)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    return ArenaHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local side-by-side MLX model comparison UI.")
    parser.add_argument("--left-model", default="./models/Qwen3-8B-MLX-8bit")
    parser.add_argument("--left-label", default="Qwen3-8B-8bit")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter for left model.")
    parser.add_argument("--right-model", default="./models/Qwen3.8-27B-4bit")
    parser.add_argument("--right-label", default="Qwen3.8-27B-4bit (Control)")
    default_prompts = None
    if Path("./data/valid.jsonl").exists():
        default_prompts = "./data/valid.jsonl"
    elif Path("./examples/sample_valid.jsonl").exists():
        default_prompts = "./examples/sample_valid.jsonl"

    parser.add_argument(
        "--prompts",
        default=default_prompts,
        help="Path to JSONL prompts file (defaults to ./data/valid.jsonl or ./examples/sample_valid.jsonl if present).",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-dir", default="./runs", help="Directory containing live LoRA training telemetry.")
    parser.add_argument(
        "--simultaneous",
        action="store_true",
        help="Keep both models loaded in RAM simultaneously (~25 GB).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.prompts)
    manager = ModelManager(
        left_spec={"model_path": args.left_model, "adapter_path": args.adapter_path},
        right_spec={"model_path": args.right_model, "adapter_path": None},
    )
    config_payload = {
        "left_model": args.left_model,
        "left_label": args.left_label,
        "adapter_path": args.adapter_path,
        "right_model": args.right_model,
        "right_label": args.right_label,
        "prompts_file": args.prompts,
        "sample_count": len(samples),
        "simultaneous": args.simultaneous,
    }
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(manager, config_payload, samples, Path(args.runs_dir)),
    )
    server.daemon_threads = True
    print(f"Side-by-side comparison UI running at http://127.0.0.1:{args.port}")
    print(f"Loaded {len(samples)} validation samples from {args.prompts}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
