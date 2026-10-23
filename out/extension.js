"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
const child_process_1 = require("child_process");
const path = require("path");
const readline = require("readline");
/**
 * Persistent Python Sidecar Client (`bridge_server.py`).
 * Keeps the Python optimizer alive across editor commands for ~1-3ms latency.
 */
class SmartyPythonBridge {
    context;
    outputChannel;
    proc = null;
    pending = new Map();
    reqCounter = 0;
    constructor(context, outputChannel) {
        this.context = context;
        this.outputChannel = outputChannel;
    }
    ensureProcess() {
        if (this.proc && !this.proc.killed) {
            return;
        }
        const config = vscode.workspace.getConfiguration('nlpSmarty');
        const pythonPath = config.get('pythonPath', 'python3');
        const engineDir = config.get('engineDirectory', '') || this.context.extensionPath;
        const scriptPath = path.join(engineDir, 'bridge_server.py');
        this.outputChannel.appendLine(`[NLP-Smarty] Spawning sidecar: ${pythonPath} ${scriptPath}`);
        this.proc = (0, child_process_1.spawn)(pythonPath, ['-u', scriptPath], { cwd: engineDir });
        const rl = readline.createInterface({ input: this.proc.stdout });
        rl.on('line', (line) => {
            try {
                const msg = JSON.parse(line);
                if (msg.id === 'init') {
                    this.outputChannel.appendLine('[NLP-Smarty] Python sidecar READY.');
                    return;
                }
                const handler = this.pending.get(msg.id);
                if (handler) {
                    this.pending.delete(msg.id);
                    handler.resolve(msg);
                }
            }
            catch (err) {
                this.outputChannel.appendLine(`[NLP-Smarty] Non-JSON stdout: ${line}`);
            }
        });
        this.proc.stderr.on('data', (chunk) => {
            this.outputChannel.appendLine(`[NLP-Smarty stderr] ${chunk.toString()}`);
        });
        this.proc.on('exit', (code) => {
            this.outputChannel.appendLine(`[NLP-Smarty] Sidecar exited with code ${code}`);
            this.proc = null;
        });
    }
    async optimize(params) {
        this.ensureProcess();
        const id = `req_${++this.reqCounter}_${Date.now()}`;
        const payload = JSON.stringify({ id, command: 'optimize', ...params }) + '\n';
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.proc.stdin.write(payload, (err) => {
                if (err) {
                    this.pending.delete(id);
                    reject(err);
                }
            });
        });
    }
    dispose() {
        if (this.proc) {
            this.proc.kill();
            this.proc = null;
        }
    }
}
function activate(context) {
    const outputChannel = vscode.window.createOutputChannel('NLP-Smarty Optimizer');
    const bridge = new SmartyPythonBridge(context, outputChannel);
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = '$(zap) NLP-Smarty Ready';
    statusBar.command = 'nlpSmarty.optimizeSelection';
    statusBar.show();
    // ── Command 1: Optimize Prompt + Active Workspace Context (Opens Telemetry Webview) ──
    const optimizeCmd = vscode.commands.registerCommand('nlpSmarty.optimizeSelection', async () => {
        const editor = vscode.window.activeTextEditor;
        let rawPrompt = '';
        if (editor && !editor.selection.isEmpty) {
            rawPrompt = editor.document.getText(editor.selection);
        }
        else {
            const input = await vscode.window.showInputBox({
                prompt: 'Enter a raw prompt to optimize against your open workspace files',
                placeHolder: 'e.g., Could you please help me refactor this module without breaking the API?',
            });
            if (!input) {
                return;
            }
            rawPrompt = input;
        }
        // Gather open editor files as real workspace context for Stage 3 MMR pruning
        const workspaceFiles = [];
        if (editor) {
            workspaceFiles.push({
                path: path.basename(editor.document.fileName),
                content: editor.document.getText(),
            });
        }
        const config = vscode.workspace.getConfiguration('nlpSmarty');
        const res = await bridge.optimize({
            text: rawPrompt,
            workspace_files: workspaceFiles,
            compression_intensity: config.get('compressionIntensity', 'AUTO'),
            max_total_tokens: config.get('maxTokenBudget', 512),
        });
        if (!res.ok || !res.trace) {
            vscode.window.showErrorMessage(`NLP-Smarty Error: ${res.error || 'Unknown error'}`);
            return;
        }
        const t = res.trace;
        statusBar.text = `$(zap) Smarty: -${t.payload_token_savings_pct}% payload (${t.latency_ms}ms)`;
        statusBar.tooltip = `Query: ${t.raw_token_count} -> ${t.optimized_token_count} tokens | Intent: ${t.detected_intent} | Cache: ${t.cache_type}`;
        if (res.status === 'NEEDS_CLARIFICATION') {
            vscode.window.showWarningMessage(res.clarification_request || 'Clarification needed.');
            return;
        }
        await vscode.env.clipboard.writeText(res.payload || '');
        vscode.window.showInformationMessage(`Compiled prompt copied! Query: -${t.query_token_savings_pct}% | Payload: -${t.payload_token_savings_pct}% (${t.latency_ms}ms)`);
        renderTelemetryWebview(rawPrompt, res.payload || '', t);
    });
    // ── Command 2: Inline Compress Selection in Active Editor ──
    const inlineReplaceCmd = vscode.commands.registerCommand('nlpSmarty.inlineCompress', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.selection.isEmpty) {
            vscode.window.showInformationMessage('Select a prompt in the editor first to compress inline.');
            return;
        }
        const rawText = editor.document.getText(editor.selection);
        const res = await bridge.optimize({ text: rawText });
        if (!res.ok || !res.payload || !res.trace) {
            return;
        }
        const coreMatch = res.payload.match(/<core_task>\n([\s\S]*?)\n<\/core_task>/);
        const compressedCore = coreMatch ? coreMatch[1].trim() : res.payload;
        await editor.edit((editBuilder) => {
            editBuilder.replace(editor.selection, compressedCore);
        });
        statusBar.text = `$(zap) Smarty: ${res.trace.raw_token_count} -> ${res.trace.optimized_token_count} words (${res.trace.latency_ms}ms)`;
    });
    // ── Surface 3: Native VS Code Chat Participant (`@smarty`) for Real Multi-Turn Testing ──
    if ('chat' in vscode) {
        const participant = vscode.chat.createChatParticipant('nlp-smarty.participant', async (request, chatContext, stream, token) => {
            const history = chatContext.history.map((turn) => {
                if (turn instanceof vscode.ChatRequestTurn) {
                    return { role: 'user', content: turn.prompt };
                }
                else if (turn instanceof vscode.ChatResponseTurn) {
                    const textParts = turn.response
                        .map((part) => (part instanceof vscode.ChatResponseMarkdownPart ? part.value.value : ''))
                        .join(' ');
                    return { role: 'assistant', content: textParts };
                }
                return { role: 'user', content: '' };
            });
            const editor = vscode.window.activeTextEditor;
            const workspaceFiles = editor
                ? [{ path: path.basename(editor.document.fileName), content: editor.document.getText() }]
                : [];
            const res = await bridge.optimize({
                text: request.prompt,
                workspace_files: workspaceFiles,
                conversation_history: history,
            });
            if (!res.ok || !res.trace) {
                stream.markdown(`**Error:** ${res.error}`);
                return;
            }
            const t = res.trace;
            stream.markdown(`### Deterministic Optimization Telemetry\n` +
                `- **Intent:** \`${t.detected_intent}\` (confidence: \`${t.intent_confidence}\`, intensity: \`${t.compression_intensity_used}\`)\n` +
                `- **Query Tokens:** \`${t.raw_token_count} -> ${t.optimized_token_count}\` (**-${t.query_token_savings_pct}%**)\n` +
                `- **Payload Tokens:** \`${t.unoptimized_payload_token_count} -> ${t.compiled_payload_token_count}\` (**-${t.payload_token_savings_pct}%**)\n` +
                `- **Kept File Chunks (MMR):** \`${JSON.stringify(t.kept_context_indices)}\` | **History Turns:** \`${t.kept_history_turns}\`\n` +
                `- **Cache:** \`${t.cache_hit ? t.cache_type : 'MISS'}\` | **Latency:** \`${t.latency_ms} ms\`\n\n` +
                `\`\`\`xml\n${res.payload}\n\`\`\``);
        });
        context.subscriptions.push(participant);
    }
    context.subscriptions.push(optimizeCmd, inlineReplaceCmd, statusBar, outputChannel, {
        dispose: () => bridge.dispose(),
    });
}
function renderTelemetryWebview(rawPrompt, compiledPayload, t) {
    const panel = vscode.window.createWebviewPanel('nlpSmartyTelemetry', `NLP-Smarty: -${t.payload_token_savings_pct}% Tokens (${t.latency_ms}ms)`, vscode.ViewColumn.Beside, { enableScripts: false });
    const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-editor-foreground); background: var(--vscode-editor-background); padding: 16px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  .card { background: var(--vscode-editorWidget-background); border: 1px solid var(--vscode-widget-border); border-radius: 6px; padding: 12px; }
  .metric { font-size: 20px; font-weight: 700; margin-top: 4px; color: var(--vscode-textLink-foreground); }
  .label { font-size: 11px; text-transform: uppercase; opacity: 0.8; }
  pre { background: var(--vscode-textCodeBlock-background); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; line-height: 1.45; }
  .badges span { display: inline-block; background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); padding: 3px 8px; border-radius: 12px; font-size: 11px; margin-right: 6px; margin-bottom: 6px; }
</style>
</head>
<body>
  <h2>NLP-Smarty Real-Data Optimization Inspector</h2>
  <div class="grid">
    <div class="card"><div class="label">Query Compression</div><div class="metric">${t.raw_token_count} &rarr; ${t.optimized_token_count} (-${t.query_token_savings_pct}%)</div></div>
    <div class="card"><div class="label">Total Payload Budget</div><div class="metric">${t.unoptimized_payload_token_count} &rarr; ${t.compiled_payload_token_count} (-${t.payload_token_savings_pct}%)</div></div>
    <div class="card"><div class="label">Intent &amp; Intensity</div><div class="metric">${t.detected_intent} (${t.compression_intensity_used})</div></div>
    <div class="card"><div class="label">Engine Latency</div><div class="metric">${t.latency_ms} ms (${t.cache_hit ? t.cache_type : 'LIVE'})</div></div>
  </div>
  <div class="badges">
    <span>Schema Saved: ${t.tool_schema_tokens_saved} toks</span>
    <span>Negations: ${escapeHtml(JSON.stringify(t.negations))}</span>
    <span>Entities: ${escapeHtml(JSON.stringify(t.extracted_entities))}</span>
    <span>Kept Context Chunks: ${escapeHtml(JSON.stringify(t.kept_context_indices))}</span>
    <span>SimHash: ${t.simhash_signature}</span>
  </div>
  <h3>Raw Input Prompt</h3>
  <pre>${escapeHtml(rawPrompt)}</pre>
  <h3>Compiled Downstream Agent Payload (Copied to Clipboard)</h3>
  <pre>${escapeHtml(compiledPayload)}</pre>
</body>
</html>`;
}
function deactivate() { }
//# sourceMappingURL=extension.js.map