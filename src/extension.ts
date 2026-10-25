import * as vscode from 'vscode';
import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as readline from 'readline';
import * as process from 'process';

interface AuditCheck {
    check: string;
    passed: boolean;
}

interface ResponseAudit {
    compliance_score: number;
    output_tokens: number;
    checks: AuditCheck[];
}

interface OptimizationTrace {
    raw_token_count: number;
    optimized_token_count: number;
    query_token_savings_pct: number;
    raw_bpe_tokens: number;
    optimized_bpe_tokens: number;
    query_bpe_savings_pct: number;
    unoptimized_payload_token_count: number;
    compiled_payload_token_count: number;
    payload_token_savings_pct: number;
    tool_schema_tokens_saved: number;
    compressed_query_text: string;
    total_context_chunks: number;
    raw_context_tokens: number;
    pruned_context_tokens: number;
    context_token_savings_pct: number;
    raw_history_tokens: number;
    pruned_history_tokens: number;
    raw_tools_tokens: number;
    compressed_tools_tokens: number;
    raw_examples_tokens: number;
    selected_examples_tokens: number;
    detected_intent: string;
    secondary_intents: string[];
    intent_confidence: number;
    compression_intensity_used: string;
    pruning_strategy_used: string;
    stripped_tokens: string[];
    normalized_phrases: string[];
    injected_example_ids: string[];
    injected_tool_ids: string[];
    kept_context_indices: number[];
    kept_history_turns: number;
    extracted_entities: Record<string, unknown>;
    negations: string[];
    stage_logs: string[];
    cache_hit: boolean;
    cache_type: string;
    cache_key: string;
    simhash_signature: string;
    latency_ms: number;
}

interface BridgeResponse {
    id: string;
    ok: boolean;
    status?: string;
    clarification_request?: string;
    payload?: string;
    trace?: OptimizationTrace;
    backend?: string;
    raw_input_tokens?: number;
    optimized_input_tokens?: number;
    raw_response?: string;
    raw_llm_latency_ms?: number;
    raw_audit?: ResponseAudit;
    optimized_payload?: string;
    optimized_response?: string;
    optimized_llm_latency_ms?: number;
    optimized_audit?: ResponseAudit;
    error?: string;
}

/**
 * Resolves the best Python interpreter in the workspace (`NLP-start/bin/python`, `.venv/bin/python`, or `python3`).
 */
function resolvePythonPath(engineDir: string): string {
    const config = vscode.workspace.getConfiguration('nlpSmarty');
    const configured = config.get<string>('pythonPath', 'python3');
    if (configured && configured !== 'python3') {
        return configured;
    }
    const candidates = [
        path.join(engineDir, 'NLP-start', 'bin', 'python'),
        path.join(engineDir, 'NLP-start', 'Scripts', 'python.exe'),
        path.join(engineDir, '.venv', 'bin', 'python'),
        path.join(engineDir, 'venv', 'bin', 'python'),
    ];
    for (const cand of candidates) {
        if (fs.existsSync(cand)) {
            return cand;
        }
    }
    return configured || 'python3';
}

/**
 * Virtual Document Provider so we can open native VS Code side-by-side red/green Diff views (`vscode.diff`).
 */
class SmartyVirtualDocProvider implements vscode.TextDocumentContentProvider {
    private _onDidChange = new vscode.EventEmitter<vscode.Uri>();
    readonly onDidChange = this._onDidChange.event;
    private docs = new Map<string, string>();

    public setDoc(key: string, content: string): vscode.Uri {
        this.docs.set(key, content);
        const uri = vscode.Uri.parse(`nlp-smarty-diff:${key}`);
        this._onDidChange.fire(uri);
        return uri;
    }

    provideTextDocumentContent(uri: vscode.Uri): string {
        return this.docs.get(uri.path) || '';
    }
}

/**
 * Persistent Python Sidecar Client (`bridge_server.py`).
 * Streams every optimization request, response, and stage-by-stage stderr log to BOTH
 * the OutputChannel (`NLP-Smarty Optimizer`) and `console.log` (Main Window Debug Console).
 */
class SmartyPythonBridge {
    private proc: ChildProcessWithoutNullStreams | null = null;
    private pending = new Map<string, { resolve: (res: BridgeResponse) => void; reject: (err: Error) => void }>();
    private reqCounter = 0;

    constructor(private context: vscode.ExtensionContext, private outputChannel: vscode.OutputChannel) {}

    public logBoth(message: string): void {
        this.outputChannel.appendLine(message);
        console.log(message);
    }

    public getEngineDir(): string {
        const config = vscode.workspace.getConfiguration('nlpSmarty');
        return config.get<string>('engineDirectory', '') || this.context.extensionPath;
    }

    private ensureProcess(): void {
        if (this.proc && !this.proc.killed) {
            return;
        }
        const config = vscode.workspace.getConfiguration('nlpSmarty');
        const engineDir = this.getEngineDir();
        const pythonPath = resolvePythonPath(engineDir);
        const scriptPath = path.join(engineDir, 'bridge_server.py');

        this.logBoth(`[NLP-Smarty] Spawning sidecar: ${pythonPath} ${scriptPath}`);
        this.proc = spawn(pythonPath, ['-u', scriptPath], {
            cwd: engineDir,
            env: {
                ...process.env,
                HF_TOKEN: config.get<string>('huggingFaceToken', '') || process.env.HF_TOKEN || '',
                HF_MODEL_ID: config.get<string>('huggingFaceModel', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
            },
        });

        const rl = readline.createInterface({ input: this.proc.stdout });
        rl.on('line', (line: string) => {
            try {
                const msg: BridgeResponse = JSON.parse(line);
                if (msg.id === 'init') {
                    this.logBoth('[NLP-Smarty] Python sidecar READY.');
                    return;
                }
                const handler = this.pending.get(msg.id);
                if (handler) {
                    this.pending.delete(msg.id);
                    handler.resolve(msg);
                }
            } catch (err) {
                this.logBoth(`[NLP-Smarty stdout] ${line}`);
            }
        });

        this.proc.stderr.on('data', (chunk: Buffer) => {
            const text = chunk.toString().trimEnd();
            if (text) {
                this.logBoth(text);
            }
        });

        this.proc.on('exit', (code) => {
            this.logBoth(`[NLP-Smarty] Sidecar exited with code ${code}`);
            this.proc = null;
        });
    }

    public async request(command: string, params: Record<string, unknown>): Promise<BridgeResponse> {
        this.ensureProcess();
        const id = `req_${++this.reqCounter}_${Date.now()}`;
        const payload = JSON.stringify({ id, command, ...params }) + '\n';

        const config = vscode.workspace.getConfiguration('nlpSmarty');
        if (config.get<boolean>('autoShowLogs', true)) {
            this.outputChannel.show(true); // Preserve editor focus while popping open Output panel
        }

        return new Promise<BridgeResponse>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.proc!.stdin.write(payload, (err) => {
                if (err) {
                    this.pending.delete(id);
                    reject(err);
                }
            });
        });
    }

    public dispose(): void {
        if (this.proc) {
            this.proc.kill();
            this.proc = null;
        }
    }
}

/**
 * Persistent Singleton Split-Screen Companion Panel (`ViewColumn.Beside`)
 * Updates live whenever the user optimizes a prompt or file so they always see
 * Old vs. New Prompt, Per-Section Token Savings, Word-Level Diff, and Stage Logs alongside their file.
 */
class SmartyInspectorPanel {
    private static currentPanel: vscode.WebviewPanel | undefined;

    public static showOrUpdate(
        rawPrompt: string,
        compiledPayload: string,
        t: OptimizationTrace,
        sourceLabel: string,
        onAction: (action: string) => void
    ): void {
        const title = `Smarty Stats: -${t.payload_token_savings_pct}% Payload | -${t.query_token_savings_pct}% Query (${t.latency_ms}ms)`;
        if (SmartyInspectorPanel.currentPanel) {
            SmartyInspectorPanel.currentPanel.title = title;
            SmartyInspectorPanel.currentPanel.reveal(vscode.ViewColumn.Beside, true);
        } else {
            SmartyInspectorPanel.currentPanel = vscode.window.createWebviewPanel(
                'nlpSmartyTelemetry',
                title,
                { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
                { enableScripts: true, retainContextWhenHidden: true }
            );
            SmartyInspectorPanel.currentPanel.onDidDispose(() => {
                SmartyInspectorPanel.currentPanel = undefined;
            });
            SmartyInspectorPanel.currentPanel.webview.onDidReceiveMessage((msg) => {
                if (msg && msg.action) {
                    onAction(msg.action);
                }
            });
        }

        SmartyInspectorPanel.currentPanel.webview.html = buildLiveComparisonHtml(
            rawPrompt,
            compiledPayload,
            t,
            sourceLabel
        );
    }
}

function escapeHtml(s: string): string {
    return (s || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Highlights stripped filler phrases in red strikethrough inside the Raw Prompt
 * so the user can visually see every word that was removed.
 */
function renderHighlightedRawPrompt(rawPrompt: string, strippedTokens: string[], normalizedPhrases: string[]): string {
    let html = escapeHtml(rawPrompt);
    const sortedStripped = [...(strippedTokens || [])].sort((a, b) => b.length - a.length);
    for (const phrase of sortedStripped) {
        const escapedPhrase = escapeHtml(phrase.trim());
        if (!escapedPhrase) {
            continue;
        }
        const safeRegex = new RegExp(escapedPhrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
        html = html.replace(
            safeRegex,
            (match) => `<span class="diff-del" title="Stripped conversational filler">${match}</span>`
        );
    }
    for (const norm of normalizedPhrases || []) {
        const parts = norm.split('->');
        if (parts.length === 2) {
            const fromStr = escapeHtml(parts[0].trim());
            const toStr = escapeHtml(parts[1].trim());
            const safeRegex = new RegExp(`\\b${fromStr.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
            html = html.replace(
                safeRegex,
                (match) =>
                    `<span class="diff-del" title="Normalized idiom">${match}</span> <span class="diff-add">[${toStr}]</span>`
            );
        }
    }
    return html;
}

function buildLiveComparisonHtml(
    rawPrompt: string,
    compiledPayload: string,
    t: OptimizationTrace,
    sourceLabel: string
): string {
    const coreQuery = t.compressed_query_text || rawPrompt;
    const highlightedRaw = renderHighlightedRawPrompt(rawPrompt, t.stripped_tokens, t.normalized_phrases);

    const calcPct = (before: number, after: number) =>
        before > 0 ? Math.max(0, Math.round(((before - after) / before) * 1000) / 10) : 0;

    const toolsPct = calcPct(t.raw_tools_tokens, t.compressed_tools_tokens);
    const exPct = calcPct(t.raw_examples_tokens, t.selected_examples_tokens);
    const histPct = calcPct(t.raw_history_tokens, t.pruned_history_tokens);

    const stageLogRows = (t.stage_logs || [])
        .map((logLine) => `<div class="log-line">${escapeHtml(logLine)}</div>`)
        .join('');

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  :root {
    --del-bg: rgba(248, 81, 73, 0.22);
    --del-fg: #ff7b72;
    --add-bg: rgba(46, 160, 67, 0.22);
    --add-fg: #3fb950;
  }
  body {
    font-family: var(--vscode-font-family);
    color: var(--vscode-editor-foreground);
    background: var(--vscode-editor-background);
    padding: 16px;
    margin: 0;
  }
  .header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--vscode-widget-border);
    padding-bottom: 10px;
  }
  .header-title { font-size: 16px; font-weight: 700; }
  .header-sub { font-size: 11px; opacity: 0.75; margin-top: 2px; }
  .btn-group button {
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    border: none;
    padding: 6px 11px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
    margin-left: 6px;
    font-weight: 600;
  }
  .btn-group button.secondary {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
  }
  .btn-group button:hover { opacity: 0.9; }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 16px;
  }
  .card {
    background: var(--vscode-editorWidget-background);
    border: 1px solid var(--vscode-widget-border);
    border-radius: 6px;
    padding: 10px 12px;
  }
  .label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.8; }
  .metric { font-size: 18px; font-weight: 700; margin-top: 4px; color: var(--vscode-textLink-foreground); }
  .sub-metric { font-size: 11px; opacity: 0.8; margin-top: 2px; }
  .split-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
  }
  .prompt-box {
    background: var(--vscode-textCodeBlock-background);
    border: 1px solid var(--vscode-widget-border);
    border-radius: 6px;
    padding: 12px;
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 12.5px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .diff-del {
    background: var(--del-bg);
    color: var(--del-fg);
    text-decoration: line-through;
    padding: 1px 4px;
    border-radius: 3px;
  }
  .diff-add {
    background: var(--add-bg);
    color: var(--add-fg);
    font-weight: 600;
    padding: 1px 4px;
    border-radius: 3px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-bottom: 16px;
  }
  th, td {
    text-align: left;
    padding: 7px 10px;
    border-bottom: 1px solid var(--vscode-widget-border);
  }
  th {
    background: var(--vscode-editorWidget-background);
    font-size: 11px;
    text-transform: uppercase;
    opacity: 0.85;
  }
  .bar-wrap {
    width: 100px;
    height: 7px;
    background: rgba(128,128,128,0.25);
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
  }
  .bar-fill {
    height: 100%;
    background: var(--vscode-textLink-foreground);
  }
  .badges span {
    display: inline-block;
    background: var(--vscode-badge-background);
    color: var(--vscode-badge-foreground);
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 11px;
    margin-right: 6px;
    margin-bottom: 6px;
  }
  .log-container {
    background: var(--vscode-textCodeBlock-background);
    border: 1px solid var(--vscode-widget-border);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 11.5px;
    margin-bottom: 16px;
  }
  .log-line {
    padding: 3px 0;
    border-bottom: 1px dashed rgba(128,128,128,0.2);
  }
  .log-line:last-child { border-bottom: none; }
  pre {
    background: var(--vscode-textCodeBlock-background);
    border: 1px solid var(--vscode-widget-border);
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 12px;
    line-height: 1.45;
  }
  h3 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.4px; margin: 14px 0 8px 0; }
</style>
</head>
<body>
  <div class="header-bar">
    <div>
      <div class="header-title">NLP-Smarty Live Optimization &amp; Telemetry Inspector</div>
      <div class="header-sub">Source: ${escapeHtml(sourceLabel)} &bull; Strategy: ${escapeHtml(t.pruning_strategy_used)} &bull; SimHash: ${escapeHtml(t.simhash_signature)}</div>
    </div>
    <div class="btn-group">
      <button onclick="sendAction('openDiff')">Open Red/Green Diff Tab</button>
      <button class="secondary" onclick="sendAction('runAB')">Run LLM A/B Compare</button>
      <button class="secondary" onclick="sendAction('showLogs')">Open Output Logs</button>
      <button class="secondary" onclick="sendAction('copyXml')">Copy Compiled XML</button>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="card">
      <div class="label">Core Prompt Compression</div>
      <div class="metric">${t.raw_token_count} &rarr; ${t.optimized_token_count} words (-${t.query_token_savings_pct}%)</div>
      <div class="sub-metric">BPE Subwords: ${t.raw_bpe_tokens} &rarr; ${t.optimized_bpe_tokens} (-${t.query_bpe_savings_pct}%)</div>
    </div>
    <div class="card">
      <div class="label">Full Agent Payload Budget</div>
      <div class="metric">${t.unoptimized_payload_token_count} &rarr; ${t.compiled_payload_token_count} (-${t.payload_token_savings_pct}%)</div>
      <div class="sub-metric">Saved ${t.unoptimized_payload_token_count - t.compiled_payload_token_count} BPE tokens total</div>
    </div>
    <div class="card">
      <div class="label">Intent &amp; Adaptive Mode</div>
      <div class="metric">${escapeHtml(t.detected_intent)}</div>
      <div class="sub-metric">Confidence: ${t.intent_confidence} &bull; Mode: ${escapeHtml(t.compression_intensity_used)}</div>
    </div>
    <div class="card">
      <div class="label">Engine Latency &amp; Cache</div>
      <div class="metric">${t.latency_ms} ms</div>
      <div class="sub-metric">${t.cache_hit ? 'CACHE HIT (' + escapeHtml(t.cache_type) + ')' : 'LIVE EXECUTION'} (${escapeHtml(t.cache_key)})</div>
    </div>
  </div>

  <h3>1. Side-by-Side Prompt Comparison (Old vs. New Core Task)</h3>
  <div class="split-grid">
    <div>
      <div class="label" style="margin-bottom:6px;">Old Prompt (${t.raw_token_count} words / ${t.raw_bpe_tokens} BPE tokens — Red = Stripped Filler)</div>
      <div class="prompt-box">${highlightedRaw}</div>
    </div>
    <div>
      <div class="label" style="margin-bottom:6px;">New Optimized Core Task (${t.optimized_token_count} words / ${t.optimized_bpe_tokens} BPE tokens)</div>
      <div class="prompt-box"><span class="diff-add">${escapeHtml(coreQuery)}</span></div>
    </div>
  </div>

  <h3>2. Per-Section Token Budget &amp; Compression Breakdown</h3>
  <table>
    <thead>
      <tr>
        <th>Pipeline Component</th>
        <th>Before (Unoptimized)</th>
        <th>After (Optimized)</th>
        <th>Token Savings</th>
        <th>What Happened</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>User Query (Stage 1)</strong></td>
        <td>${t.raw_bpe_tokens} BPE (${t.raw_token_count}w)</td>
        <td>${t.optimized_bpe_tokens} BPE (${t.optimized_token_count}w)</td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(100, t.query_bpe_savings_pct)}%"></div></div><strong>-${t.query_bpe_savings_pct}%</strong></td>
        <td>Stripped ${t.stripped_tokens.length} filler phrase(s), normalized ${t.normalized_phrases.length} idiom(s)</td>
      </tr>
      <tr>
        <td><strong>Workspace File Context (Stage 3 MMR)</strong></td>
        <td>${t.raw_context_tokens} BPE (${t.total_context_chunks} chunks)</td>
        <td>${t.pruned_context_tokens} BPE (${t.kept_context_indices.length} chunks)</td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(100, t.context_token_savings_pct)}%"></div></div><strong>-${t.context_token_savings_pct}%</strong></td>
        <td>Kept chunk indices: <code>${escapeHtml(JSON.stringify(t.kept_context_indices))}</code> via Hybrid BM25+TFIDF MMR</td>
      </tr>
      <tr>
        <td><strong>Tool Schemas (Step 2 Compiler)</strong></td>
        <td>${t.raw_tools_tokens} BPE (all ${4} tools)</td>
        <td>${t.compressed_tools_tokens} BPE (${t.injected_tool_ids.length} matched)</td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(100, toolsPct)}%"></div></div><strong>-${toolsPct}%</strong></td>
        <td>Injected: <code>${escapeHtml(JSON.stringify(t.injected_tool_ids))}</code> (saved ${t.tool_schema_tokens_saved} schema toks)</td>
      </tr>
      <tr>
        <td><strong>Few-Shot Examples (Stage 3 MMR)</strong></td>
        <td>${t.raw_examples_tokens} BPE (all 6 examples)</td>
        <td>${t.selected_examples_tokens} BPE (${t.injected_example_ids.length} selected)</td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(100, exPct)}%"></div></div><strong>-${exPct}%</strong></td>
        <td>Selected IDs: <code>${escapeHtml(JSON.stringify(t.injected_example_ids))}</code></td>
      </tr>
      <tr>
        <td><strong>Conversation History (Step 7)</strong></td>
        <td>${t.raw_history_tokens} BPE</td>
        <td>${t.pruned_history_tokens} BPE (${t.kept_history_turns} turns)</td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(100, histPct)}%"></div></div><strong>-${histPct}%</strong></td>
        <td>Recency-weighted MMR (0.85^age decay)</td>
      </tr>
      <tr style="background: var(--vscode-editorWidget-background); font-weight: 700;">
        <td><strong>Total Compiled Agent Payload</strong></td>
        <td>${t.unoptimized_payload_token_count} BPE tokens</td>
        <td>${t.compiled_payload_token_count} BPE tokens</td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(100, t.payload_token_savings_pct)}%"></div></div><strong>-${t.payload_token_savings_pct}%</strong></td>
        <td>Compiled with XML anti-hallucination guardrails</td>
      </tr>
    </tbody>
  </table>

  <div class="badges">
    <span>Protected Negations: ${escapeHtml(JSON.stringify(t.negations))}</span>
    <span>Extracted Entities: ${escapeHtml(JSON.stringify(t.extracted_entities))}</span>
    <span>Stripped Phrases: ${escapeHtml(JSON.stringify(t.stripped_tokens))}</span>
    <span>Normalized Idioms: ${escapeHtml(JSON.stringify(t.normalized_phrases))}</span>
  </div>

  <h3>3. Accurate Stage-by-Stage Execution Logs</h3>
  <div class="log-container">
    ${stageLogRows || '<div class="log-line">No stage logs recorded.</div>'}
  </div>

  <h3>4. Full Compiled XML Payload Sent to Downstream Agent</h3>
  <pre>${escapeHtml(compiledPayload)}</pre>

  <script>
    const vscode = acquireVsCodeApi();
    function sendAction(action) {
      vscode.postMessage({ action });
    }
  </script>
</body>
</html>`;
}

/**
 * Queries VS Code's built-in Language Model API (`vscode.lm`) if available,
 * or falls back to the HuggingFace `app.py` bridge (`compare_hf`).
 */
async function runSideBySideComparison(
    bridge: SmartyPythonBridge,
    rawPrompt: string,
    workspaceFiles: Array<{ path: string; content: string }>,
    conversationHistory: Array<{ role: string; content: string }>,
    token?: vscode.CancellationToken
): Promise<BridgeResponse> {
    const config = vscode.workspace.getConfiguration('nlpSmarty');
    const preferredBackend = config.get<string>('downstreamBackend', 'auto');
    const baseParams = {
        text: rawPrompt,
        workspace_files: workspaceFiles,
        conversation_history: conversationHistory,
        compression_intensity: config.get<string>('compressionIntensity', 'AUTO'),
        max_total_tokens: config.get<number>('maxTokenBudget', 512),
        hf_model_id: config.get<string>('huggingFaceModel', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
    };

    if (preferredBackend !== 'huggingface' && 'lm' in vscode) {
        try {
            const models = await vscode.lm.selectChatModels({});
            if (models && models.length > 0) {
                const model = models[0];
                const optRes = await bridge.request('optimize', baseParams);
                if (!optRes.ok || !optRes.payload || !optRes.trace) {
                    return optRes;
                }

                const rawContextStr = workspaceFiles.map((f) => `[${f.path}]\n${f.content}`).join('\n\n');
                const unoptimizedInput = rawContextStr ? `${rawContextStr}\n\n${rawPrompt}` : rawPrompt;

                const callVscodeLm = async (inputText: string): Promise<{ text: string; ms: number }> => {
                    const t0 = Date.now();
                    const messages = [vscode.LanguageModelChatMessage.User(inputText)];
                    const chatResp = await model.sendRequest(messages, {}, token);
                    let out = '';
                    for await (const fragment of chatResp.text) {
                        out += fragment;
                    }
                    return { text: out, ms: Date.now() - t0 };
                };

                const [rawOut, optOut] = await Promise.all([
                    callVscodeLm(unoptimizedInput),
                    callVscodeLm(optRes.payload),
                ]);

                const auditRes = await bridge.request('audit_pair', {
                    ...baseParams,
                    raw_response: rawOut.text,
                    optimized_response: optOut.text,
                });

                return {
                    id: optRes.id,
                    ok: true,
                    status: 'OK',
                    backend: `vscode.lm (${model.name || model.id})`,
                    raw_input_tokens: optRes.trace.unoptimized_payload_token_count,
                    optimized_input_tokens: optRes.trace.compiled_payload_token_count,
                    raw_response: rawOut.text,
                    raw_llm_latency_ms: rawOut.ms,
                    raw_audit: auditRes.raw_audit,
                    optimized_payload: optRes.payload,
                    optimized_response: optOut.text,
                    optimized_llm_latency_ms: optOut.ms,
                    optimized_audit: auditRes.optimized_audit,
                    trace: optRes.trace,
                };
            }
        } catch (err) {
            // Fall through to HuggingFace `app.py` backend
        }
    }

    return bridge.request('compare_hf', baseParams);
}

/**
 * Extracts a prompt from an active editor file when no text is highlighted:
 * - If the file is a prompt file (`.txt`, `.md`, `.prompt`) or short (< 120 lines), uses the whole file text as the prompt!
 * - Otherwise asks the user for a query to run against the active code file.
 */
async function resolvePromptAndWorkspaceContext(
    editor: vscode.TextEditor | undefined,
    preferEntireFileAsPrompt: boolean = false
): Promise<{ rawPrompt: string; workspaceFiles: Array<{ path: string; content: string }>; sourceLabel: string } | undefined> {
    if (!editor) {
        const input = await vscode.window.showInputBox({
            prompt: 'Enter a raw prompt to optimize',
            placeHolder: 'e.g., Could you please help me write a Python function to reverse a string without using slicing?',
        });
        if (!input) {
            return undefined;
        }
        return { rawPrompt: input, workspaceFiles: [], sourceLabel: 'Input Box' };
    }

    const fileName = path.basename(editor.document.fileName);
    const fullText = editor.document.getText();
    const ext = path.extname(fileName).toLowerCase();

    if (!editor.selection.isEmpty) {
        const selectedText = editor.document.getText(editor.selection);
        return {
            rawPrompt: selectedText,
            workspaceFiles: [{ path: fileName, content: fullText }],
            sourceLabel: `Selection in ${fileName}`,
        };
    }

    const isPromptLikeFile =
        preferEntireFileAsPrompt ||
        ['.prompt', '.txt', '.md'].includes(ext) ||
        editor.document.lineCount <= 40;

    if (isPromptLikeFile && fullText.trim().length > 0) {
        return {
            rawPrompt: fullText.trim(),
            workspaceFiles: [],
            sourceLabel: `Full File (${fileName})`,
        };
    }

    const input = await vscode.window.showInputBox({
        prompt: `Enter a prompt to optimize alongside ${fileName} (or select text in the file first)`,
        placeHolder: 'e.g., Could you please help me refactor this file without breaking existing functions?',
    });
    if (!input) {
        return undefined;
    }
    return {
        rawPrompt: input,
        workspaceFiles: [{ path: fileName, content: fullText }],
        sourceLabel: `Prompt + Context (${fileName})`,
    };
}

export function activate(context: vscode.ExtensionContext) {
    const outputChannel = vscode.window.createOutputChannel('NLP-Smarty Optimizer');
    const bridge = new SmartyPythonBridge(context, outputChannel);
    const diffProvider = new SmartyVirtualDocProvider();

    context.subscriptions.push(
        vscode.workspace.registerTextDocumentContentProvider('nlp-smarty-diff', diffProvider)
    );

    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = '$(zap) NLP-Smarty Ready';
    statusBar.tooltip = 'Click to optimize active file/selection and view Old vs. New Prompt Stats';
    statusBar.command = 'nlpSmarty.optimizeSelection';
    statusBar.show();

    let lastRawPrompt = '';
    let lastCompiledPayload = '';
    let lastTrace: OptimizationTrace | undefined;

    const openNativeDiffView = async (rawPrompt: string, compiledPayload: string, t: OptimizationTrace) => {
        const coreOptimized = t.compressed_query_text || compiledPayload;
        const leftContent =
            `// ══════════════ OLD RAW PROMPT (${t.raw_token_count} words | ${t.raw_bpe_tokens} BPE tokens) ══════════════\n` +
            `${rawPrompt}\n\n` +
            `// ══════════════ UNOPTIMIZED BASELINE STATS ══════════════\n` +
            `// Full Unoptimized Payload : ${t.unoptimized_payload_token_count} BPE tokens\n` +
            `// Unpruned Context Chunks  : ${t.total_context_chunks} chunks (${t.raw_context_tokens} BPE tokens)\n` +
            `// Raw Tool Schemas         : ${t.raw_tools_tokens} BPE tokens\n` +
            `// All Few-Shot Examples    : ${t.raw_examples_tokens} BPE tokens\n`;

        const rightContent =
            `// ══════════════ NEW OPTIMIZED CORE PROMPT (${t.optimized_token_count} words | ${t.optimized_bpe_tokens} BPE tokens | -${t.query_token_savings_pct}%) ══════════════\n` +
            `${coreOptimized}\n\n` +
            `// ══════════════ OPTIMIZATION TELEMETRY & SAVINGS (-${t.payload_token_savings_pct}% Total Payload) ══════════════\n` +
            `// Compiled Payload Budget  : ${t.compiled_payload_token_count} BPE tokens (saved ${t.unoptimized_payload_token_count - t.compiled_payload_token_count} tokens)\n` +
            `// Detected Intent & Mode   : ${t.detected_intent} (conf=${t.intent_confidence}, mode=${t.compression_intensity_used})\n` +
            `// Stripped Filler Phrases  : ${JSON.stringify(t.stripped_tokens)}\n` +
            `// Normalized Idioms        : ${JSON.stringify(t.normalized_phrases)}\n` +
            `// Protected Negations      : ${JSON.stringify(t.negations)}\n` +
            `// Extracted Entities       : ${JSON.stringify(t.extracted_entities)}\n` +
            `// Kept Context Chunks (MMR): ${JSON.stringify(t.kept_context_indices)} / ${t.total_context_chunks} (${t.pruned_context_tokens} BPE tokens, -${t.context_token_savings_pct}%)\n` +
            `// Injected Tools           : ${JSON.stringify(t.injected_tool_ids)} (saved ${t.tool_schema_tokens_saved} schema tokens)\n` +
            `// Latency & Cache          : ${t.latency_ms} ms (${t.cache_hit ? t.cache_type : 'LIVE'})\n\n` +
            `// ══════════════ FULL COMPILED XML PAYLOAD ══════════════\n` +
            `${compiledPayload}\n`;

        const leftUri = diffProvider.setDoc('Old_Raw_Prompt.txt', leftContent);
        const rightUri = diffProvider.setDoc('New_Optimized_Prompt.txt', rightContent);

        await vscode.commands.executeCommand(
            'vscode.diff',
            leftUri,
            rightUri,
            `NLP-Smarty Diff: Old (${t.raw_token_count}w) ↔ New (${t.optimized_token_count}w, -${t.query_token_savings_pct}%)`,
            { viewColumn: vscode.ViewColumn.Beside, preserveFocus: false }
        );
    };

    const handleInspectorAction = async (action: string) => {
        if (!lastTrace) {
            return;
        }
        if (action === 'openDiff') {
            await openNativeDiffView(lastRawPrompt, lastCompiledPayload, lastTrace);
        } else if (action === 'showLogs') {
            outputChannel.show(false);
            const logFile = path.join(bridge.getEngineDir(), '.nlp_smarty_optimization.log');
            if (fs.existsSync(logFile)) {
                const doc = await vscode.workspace.openTextDocument(logFile);
                await vscode.window.showTextDocument(doc, { viewColumn: vscode.ViewColumn.Beside, preserveFocus: false });
            }
        } else if (action === 'copyXml') {
            await vscode.env.clipboard.writeText(lastCompiledPayload);
            vscode.window.showInformationMessage('Copied compiled XML payload to clipboard!');
        } else if (action === 'runAB') {
            await vscode.commands.executeCommand('nlpSmarty.compareSideBySide');
        }
    };

    // ── Command 1: Optimize Prompt / Active File & Show Side-by-Side Comparison (`Cmd+Alt+O`) ──
    const optimizeCmd = vscode.commands.registerCommand('nlpSmarty.optimizeSelection', async () => {
        const resolved = await resolvePromptAndWorkspaceContext(vscode.window.activeTextEditor, true);
        if (!resolved) {
            return;
        }

        const config = vscode.workspace.getConfiguration('nlpSmarty');
        const res = await bridge.request('optimize', {
            text: resolved.rawPrompt,
            workspace_files: resolved.workspaceFiles,
            compression_intensity: config.get<string>('compressionIntensity', 'AUTO'),
            max_total_tokens: config.get<number>('maxTokenBudget', 512),
        });

        if (!res.ok || !res.trace) {
            vscode.window.showErrorMessage(`NLP-Smarty Error: ${res.error || res.clarification_request || 'Unknown error'}`);
            return;
        }

        const t = res.trace;
        lastRawPrompt = resolved.rawPrompt;
        lastCompiledPayload = res.payload || '';
        lastTrace = t;

        statusBar.text = `$(zap) Smarty: ${t.raw_token_count}→${t.optimized_token_count}w (-${t.query_token_savings_pct}%) | Payload -${t.payload_token_savings_pct}% (${t.latency_ms}ms)`;
        await vscode.env.clipboard.writeText(res.payload || '');

        // Always open the side-by-side Live Comparison & Stats Webview alongside the file
        SmartyInspectorPanel.showOrUpdate(
            resolved.rawPrompt,
            res.payload || '',
            t,
            resolved.sourceLabel,
            handleInspectorAction
        );
    });

    // ── Command 2: Open Native VS Code Red/Green Diff Editor (`nlpSmarty.openPromptDiff`) ──
    const diffCmd = vscode.commands.registerCommand('nlpSmarty.openPromptDiff', async () => {
        const resolved = await resolvePromptAndWorkspaceContext(vscode.window.activeTextEditor, true);
        if (!resolved) {
            return;
        }
        const config = vscode.workspace.getConfiguration('nlpSmarty');
        const res = await bridge.request('optimize', {
            text: resolved.rawPrompt,
            workspace_files: resolved.workspaceFiles,
            compression_intensity: config.get<string>('compressionIntensity', 'AUTO'),
            max_total_tokens: config.get<number>('maxTokenBudget', 512),
        });
        if (!res.ok || !res.trace) {
            vscode.window.showErrorMessage(`NLP-Smarty Error: ${res.error || res.clarification_request}`);
            return;
        }
        lastRawPrompt = resolved.rawPrompt;
        lastCompiledPayload = res.payload || '';
        lastTrace = res.trace;
        await openNativeDiffView(resolved.rawPrompt, res.payload || '', res.trace);
    });

    // ── Command 3: Side-by-Side Raw vs. Optimized Downstream Agent Comparison (`Cmd+Alt+C`) ──
    const compareCmd = vscode.commands.registerCommand('nlpSmarty.compareSideBySide', async () => {
        const resolved = await resolvePromptAndWorkspaceContext(vscode.window.activeTextEditor, true);
        if (!resolved) {
            return;
        }

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'NLP-Smarty: Running Raw vs. Optimized Prompt on Downstream Agent...',
                cancellable: true,
            },
            async (_progress, token) => {
                const cmp = await runSideBySideComparison(
                    bridge,
                    resolved.rawPrompt,
                    resolved.workspaceFiles,
                    [],
                    token
                );
                if (!cmp.ok || !cmp.trace) {
                    vscode.window.showErrorMessage(`NLP-Smarty Error: ${cmp.error || cmp.clarification_request}`);
                    return;
                }
                lastRawPrompt = resolved.rawPrompt;
                lastCompiledPayload = cmp.optimized_payload || '';
                lastTrace = cmp.trace;
                statusBar.text = `$(zap) Smarty A/B: ${cmp.raw_audit?.compliance_score ?? 0}% vs ${cmp.optimized_audit?.compliance_score ?? 100}% compliance`;
                renderSideBySideWebview(resolved.rawPrompt, cmp);
            }
        );
    });

    // ── Command 4: Inline Compress Selection (`Cmd+Alt+I`) — Now ALSO opens the Comparison View Alongside! ──
    const inlineReplaceCmd = vscode.commands.registerCommand('nlpSmarty.inlineCompress', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        const hasSelection = !editor.selection.isEmpty;
        const rawText = hasSelection ? editor.document.getText(editor.selection) : editor.document.getText();
        if (!rawText.trim()) {
            vscode.window.showInformationMessage('Open a prompt file or select text first to compress.');
            return;
        }

        const res = await bridge.request('optimize', { text: rawText });
        if (!res.ok || !res.payload || !res.trace) {
            return;
        }
        const t = res.trace;
        const compressedCore = t.compressed_query_text || res.payload;

        lastRawPrompt = rawText;
        lastCompiledPayload = res.payload;
        lastTrace = t;

        await editor.edit((editBuilder) => {
            if (hasSelection) {
                editBuilder.replace(editor.selection, compressedCore);
            } else {
                const fullRange = new vscode.Range(
                    editor.document.positionAt(0),
                    editor.document.positionAt(rawText.length)
                );
                editBuilder.replace(fullRange, compressedCore);
            }
        });

        statusBar.text = `$(zap) Smarty: ${t.raw_token_count} → ${t.optimized_token_count} words (-${t.query_token_savings_pct}%, ${t.latency_ms}ms)`;

        // Open the side-by-side comparison inspector alongside so the user sees Old vs. New stats!
        SmartyInspectorPanel.showOrUpdate(
            rawText,
            res.payload,
            t,
            `Inline Compress (${path.basename(editor.document.fileName)})`,
            handleInspectorAction
        );
    });

    // ── Command 5: Show Accurate Optimization Logs (`nlpSmarty.showLogs`) ──
    const showLogsCmd = vscode.commands.registerCommand('nlpSmarty.showLogs', async () => {
        outputChannel.show(false);
        const logFile = path.join(bridge.getEngineDir(), '.nlp_smarty_optimization.log');
        if (fs.existsSync(logFile)) {
            const doc = await vscode.workspace.openTextDocument(logFile);
            await vscode.window.showTextDocument(doc, { viewColumn: vscode.ViewColumn.Beside, preserveFocus: false });
        }
    });

    // ── Auto-Optimize on Save for `.prompt` files or when `nlpSmarty.optimizeOnSave` is enabled ──
    const saveListener = vscode.workspace.onDidSaveTextDocument(async (doc) => {
        const config = vscode.workspace.getConfiguration('nlpSmarty');
        const autoOnSave = config.get<boolean>('optimizeOnSave', false);
        const isPromptExt = doc.fileName.endsWith('.prompt') || doc.fileName.endsWith('.smarty');
        if (!autoOnSave && !isPromptExt) {
            return;
        }
        const rawText = doc.getText().trim();
        if (!rawText) {
            return;
        }
        const res = await bridge.request('optimize', {
            text: rawText,
            compression_intensity: config.get<string>('compressionIntensity', 'AUTO'),
            max_total_tokens: config.get<number>('maxTokenBudget', 512),
        });
        if (res.ok && res.trace) {
            lastRawPrompt = rawText;
            lastCompiledPayload = res.payload || '';
            lastTrace = res.trace;
            SmartyInspectorPanel.showOrUpdate(
                rawText,
                res.payload || '',
                res.trace,
                `Saved File (${path.basename(doc.fileName)})`,
                handleInspectorAction
            );
        }
    });

    // ── Surface 6: Native VS Code Chat Participant (`@smarty` with `/optimize` and `/compare`) ──
    if ('chat' in vscode) {
        const participant = vscode.chat.createChatParticipant(
            'nlp-smarty.participant',
            async (request, chatContext, stream, token) => {
                const history = chatContext.history.map((turn) => {
                    if (turn instanceof vscode.ChatRequestTurn) {
                        return { role: 'user', content: turn.prompt };
                    } else if (turn instanceof vscode.ChatResponseTurn) {
                        const textParts = turn.response
                            .map((part) => (part instanceof vscode.ChatResponseMarkdownPart ? part.value.value : ''))
                            .join(' ');
                        return { role: 'assistant', content: textParts };
                    }
                    return { role: 'user', content: '' };
                });

                const editor = vscode.window.activeTextEditor;
                const workspaceFiles: Array<{ path: string; content: string }> = [];
                if (editor) {
                    workspaceFiles.push({
                        path: path.basename(editor.document.fileName),
                        content: editor.document.getText(),
                    });
                }

                if (request.command === 'optimize') {
                    const res = await bridge.request('optimize', {
                        text: request.prompt,
                        workspace_files: workspaceFiles,
                        conversation_history: history,
                    });
                    if (!res.ok || !res.trace) {
                        stream.markdown(`**Error:** ${res.error}`);
                        return;
                    }
                    const t = res.trace;
                    lastRawPrompt = request.prompt;
                    lastCompiledPayload = res.payload || '';
                    lastTrace = t;

                    SmartyInspectorPanel.showOrUpdate(
                        request.prompt,
                        res.payload || '',
                        t,
                        '@smarty /optimize',
                        handleInspectorAction
                    );

                    stream.markdown(
                        `### Old vs. New Prompt Optimization Stats\n` +
                        `| Metric | Old (Raw) | New (Optimized) | Savings |\n` +
                        `| :--- | :--- | :--- | :--- |\n` +
                        `| **Prompt Text** | \`${request.prompt}\` | \`${t.compressed_query_text}\` | **-${t.query_token_savings_pct}% words** |\n` +
                        `| **Query Subwords (BPE)** | \`${t.raw_bpe_tokens}\` | \`${t.optimized_bpe_tokens}\` | **-${t.query_bpe_savings_pct}%** |\n` +
                        `| **Workspace Context** | \`${t.raw_context_tokens} toks (${t.total_context_chunks} chunks)\` | \`${t.pruned_context_tokens} toks (chunks ${JSON.stringify(t.kept_context_indices)})\` | **-${t.context_token_savings_pct}%** |\n` +
                        `| **Tool Schemas** | \`${t.raw_tools_tokens} toks\` | \`${t.compressed_tools_tokens} toks (${JSON.stringify(t.injected_tool_ids)})\` | **Saved ${t.tool_schema_tokens_saved} toks** |\n` +
                        `| **Full Agent Payload** | \`${t.unoptimized_payload_token_count} toks\` | \`${t.compiled_payload_token_count} toks\` | **-${t.payload_token_savings_pct}%** |\n\n` +
                        `- **Intent:** \`${t.detected_intent}\` (conf: \`${t.intent_confidence}\`, intensity: \`${t.compression_intensity_used}\`)\n` +
                        `- **Stripped Filler:** \`${JSON.stringify(t.stripped_tokens)}\` | **Protected Negations:** \`${JSON.stringify(t.negations)}\`\n` +
                        `- **Cache:** \`${t.cache_hit ? t.cache_type : 'MISS'}\` | **Latency:** \`${t.latency_ms} ms\`\n\n` +
                        `\`\`\`xml\n${res.payload}\n\`\`\``
                    );
                    return;
                }

                stream.progress('Optimizing prompt deterministically and running Raw vs. Optimized A/B comparison...');
                const cmp = await runSideBySideComparison(bridge, request.prompt, workspaceFiles, history, token);
                if (!cmp.ok || !cmp.trace) {
                    stream.markdown(`**Clarification / Error:** ${cmp.clarification_request || cmp.error}`);
                    return;
                }

                const t = cmp.trace;
                lastRawPrompt = request.prompt;
                lastCompiledPayload = cmp.optimized_payload || '';
                lastTrace = t;

                SmartyInspectorPanel.showOrUpdate(
                    request.prompt,
                    cmp.optimized_payload || '',
                    t,
                    '@smarty /compare',
                    handleInspectorAction
                );

                const rawScore = cmp.raw_audit?.compliance_score ?? 0;
                const optScore = cmp.optimized_audit?.compliance_score ?? 100;

                stream.markdown(
                    `### NLP-Smarty A/B Downstream Agent Comparison (\`${cmp.backend}\`)\n\n` +
                    `**Old Prompt:** \`${request.prompt}\`\n\n` +
                    `**New Core Prompt:** \`${t.compressed_query_text}\`\n\n` +
                    `| Metric | Raw Unoptimized Prompt | NLP-Smarty Compiled Prompt | Delta |\n` +
                    `| :--- | :--- | :--- | :--- |\n` +
                    `| **Query Words / BPE** | \`${t.raw_token_count}w (${t.raw_bpe_tokens} BPE)\` | \`${t.optimized_token_count}w (${t.optimized_bpe_tokens} BPE)\` | **-${t.query_token_savings_pct}% words** |\n` +
                    `| **Full Payload Tokens** | \`${cmp.raw_input_tokens}\` | \`${cmp.optimized_input_tokens}\` | **-${t.payload_token_savings_pct}%** |\n` +
                    `| **Downstream LLM Latency** | \`${cmp.raw_llm_latency_ms} ms\` | \`${cmp.optimized_llm_latency_ms} ms\` | Optimizer took \`${t.latency_ms} ms\` |\n` +
                    `| **Deterministic Audit Score** | \`${rawScore}/100\` | **\`${optScore}/100\`** | **+${optScore - rawScore} pts** |\n\n` +
                    `#### 1. Optimized Agent Response (Score: \`${optScore}/100\`)\n${cmp.optimized_response}\n\n` +
                    `#### 2. Raw Unoptimized Agent Response (Score: \`${rawScore}/100\`)\n${cmp.raw_response}\n\n` +
                    `<details><summary>View Stage-by-Stage Execution Logs &amp; Compiled XML</summary>\n\n` +
                    `${(t.stage_logs || []).map((l) => `- \`${l}\``).join('\n')}\n\n` +
                    `\`\`\`xml\n${cmp.optimized_payload}\n\`\`\`\n</details>`
                );
            }
        );
        context.subscriptions.push(participant);
    }

    context.subscriptions.push(
        optimizeCmd,
        diffCmd,
        compareCmd,
        inlineReplaceCmd,
        showLogsCmd,
        saveListener,
        statusBar,
        outputChannel,
        { dispose: () => bridge.dispose() }
    );
}

function renderSideBySideWebview(rawPrompt: string, cmp: BridgeResponse) {
    const t = cmp.trace!;
    const panel = vscode.window.createWebviewPanel(
        'nlpSmartyABCompare',
        `NLP-Smarty A/B: ${cmp.raw_audit?.compliance_score ?? 0}% vs ${cmp.optimized_audit?.compliance_score ?? 100}%`,
        vscode.ViewColumn.Beside,
        { enableScripts: false }
    );

    const renderChecks = (audit?: ResponseAudit) =>
        (audit?.checks || [])
            .map((c) => `<li>${c.passed ? 'PASS' : 'FAIL'} — ${escapeHtml(c.check)}</li>`)
            .join('');

    const highlightedRaw = renderHighlightedRawPrompt(rawPrompt, t.stripped_tokens, t.normalized_phrases);

    panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-editor-foreground); background: var(--vscode-editor-background); padding: 16px; }
  .top-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  .ab-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; margin-bottom: 16px; }
  .card { background: var(--vscode-editorWidget-background); border: 1px solid var(--vscode-widget-border); border-radius: 6px; padding: 12px; }
  .metric { font-size: 18px; font-weight: 700; margin-top: 4px; color: var(--vscode-textLink-foreground); }
  .label { font-size: 11px; text-transform: uppercase; opacity: 0.8; }
  .diff-del { background: rgba(248, 81, 73, 0.22); color: #ff7b72; text-decoration: line-through; padding: 1px 4px; border-radius: 3px; }
  .diff-add { background: rgba(46, 160, 67, 0.22); color: #3fb950; font-weight: 600; padding: 1px 4px; border-radius: 3px; }
  pre { background: var(--vscode-textCodeBlock-background); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }
</style>
</head>
<body>
  <h2>Raw vs. Optimized Downstream Agent Comparison (${escapeHtml(cmp.backend || '')})</h2>
  <div class="top-grid">
    <div class="card"><div class="label">Query Words / BPE</div><div class="metric">${t.raw_token_count}w &rarr; ${t.optimized_token_count}w (-${t.query_token_savings_pct}%)</div></div>
    <div class="card"><div class="label">Payload Token Reduction</div><div class="metric">${cmp.raw_input_tokens} &rarr; ${cmp.optimized_input_tokens} (-${t.payload_token_savings_pct}%)</div></div>
    <div class="card"><div class="label">Compliance Score (Raw vs Opt)</div><div class="metric">${cmp.raw_audit?.compliance_score ?? 0}/100 &rarr; ${cmp.optimized_audit?.compliance_score ?? 100}/100</div></div>
    <div class="card"><div class="label">Optimizer Overhead</div><div class="metric">${t.latency_ms} ms (${t.cache_hit ? t.cache_type : 'LIVE'})</div></div>
  </div>
  <div class="ab-grid">
    <div class="card">
      <h3>Old Prompt (${t.raw_token_count} words)</h3>
      <pre>${highlightedRaw}</pre>
    </div>
    <div class="card">
      <h3>New Optimized Core Prompt (${t.optimized_token_count} words)</h3>
      <pre><span class="diff-add">${escapeHtml(t.compressed_query_text || '')}</span></pre>
    </div>
  </div>
  <div class="ab-grid">
    <div class="card">
      <h3>Raw Unoptimized Agent Output (${cmp.raw_audit?.compliance_score ?? 0}/100)</h3>
      <ul>${renderChecks(cmp.raw_audit)}</ul>
      <pre>${escapeHtml(cmp.raw_response || '')}</pre>
    </div>
    <div class="card">
      <h3>NLP-Smarty Optimized Agent Output (${cmp.optimized_audit?.compliance_score ?? 100}/100)</h3>
      <ul>${renderChecks(cmp.optimized_audit)}</ul>
      <pre>${escapeHtml(cmp.optimized_response || '')}</pre>
    </div>
  </div>
  <h3>Stage-by-Stage Execution Logs</h3>
  <pre>${escapeHtml((t.stage_logs || []).join('\n'))}</pre>
  <h3>Compiled XML Prompt Sent to Agent</h3>
  <pre>${escapeHtml(cmp.optimized_payload || '')}</pre>
</body>
</html>`;
}

export function deactivate() {}