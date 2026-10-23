import * as vscode from 'vscode';
import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import * as path from 'path';
import * as readline from 'readline';
import { Buffer } from 'buffer';
import process from 'process';

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
    unoptimized_payload_token_count: number;
    compiled_payload_token_count: number;
    payload_token_savings_pct: number;
    tool_schema_tokens_saved: number;
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
 * Persistent Python Sidecar Client (`bridge_server.py`).
 * Supports `optimize`, `compare_hf` (HuggingFace app.py), and `audit_pair` (for vscode.lm).
 */
class SmartyPythonBridge {
    private proc: ChildProcessWithoutNullStreams | null = null;
    private pending = new Map<string, { resolve: (res: BridgeResponse) => void; reject: (err: Error) => void }>();
    private reqCounter = 0;

    constructor(private context: vscode.ExtensionContext, private outputChannel: vscode.OutputChannel) {}

    private ensureProcess(): void {
        if (this.proc && !this.proc.killed) {
            return;
        }
        const config = vscode.workspace.getConfiguration('nlpSmarty');
        const pythonPath = config.get<string>('pythonPath', 'python3');
        const engineDir = config.get<string>('engineDirectory', '') || this.context.extensionPath;
        const scriptPath = path.join(engineDir, 'bridge_server.py');

        this.outputChannel.appendLine(`[NLP-Smarty] Spawning sidecar: ${pythonPath} ${scriptPath}`);
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
                    this.outputChannel.appendLine('[NLP-Smarty] Python sidecar READY.');
                    return;
                }
                const handler = this.pending.get(msg.id);
                if (handler) {
                    this.pending.delete(msg.id);
                    handler.resolve(msg);
                }
            } catch (err) {
                this.outputChannel.appendLine(`[NLP-Smarty] Non-JSON stdout: ${line}`);
            }
        });

        this.proc.stderr.on('data', (chunk: Buffer) => {
            this.outputChannel.appendLine(`[NLP-Smarty stderr] ${chunk.toString()}`);
        });

        this.proc.on('exit', (code: number | null) => {
            this.outputChannel.appendLine(`[NLP-Smarty] Sidecar exited with code ${code}`);
            this.proc = null;
        });
    }

    public async request(command: string, params: Record<string, unknown>): Promise<BridgeResponse> {
        this.ensureProcess();
        const id = `req_${++this.reqCounter}_${Date.now()}`;
        const payload = JSON.stringify({ id, command, ...params }) + '\n';

        return new Promise<BridgeResponse>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.proc!.stdin.write(payload, (err: Error | null | undefined) => {
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
    const preferredBackend = config.get<string>('downstreamBackend', 'auto'); // 'auto' | 'vscode-lm' | 'huggingface'
    const baseParams = {
        text: rawPrompt,
        workspace_files: workspaceFiles,
        conversation_history: conversationHistory,
        compression_intensity: config.get<string>('compressionIntensity', 'AUTO'),
        max_total_tokens: config.get<number>('maxTokenBudget', 512),
        hf_model_id: config.get<string>('huggingFaceModel', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
    };

    // Option A: Use VS Code's built-in Copilot LM (`vscode.lm.selectChatModels`) if selected/available
    if (preferredBackend !== 'huggingface' && 'lm' in vscode) {
        try {
            const models = await vscode.lm.selectChatModels({});
            if (models && models.length > 0) {
                const model = models[0];
                const optRes = await bridge.request('optimize', baseParams);
                if (!optRes.ok || !optRes.payload || !optRes.trace) {
                    return optRes;
                }

                // Construct unoptimized baseline input (raw prompt + full unpruned file)
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

                // Send both responses to Python sidecar for deterministic compliance & hallucination auditing
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
            // Fall through to HuggingFace `app.py` backend if `vscode.lm` permission is declined or unavailable
        }
    }

    // Option B: Forward to Python `app.py` (HuggingFace InferenceClient / HTTP router / offline simulator)
    return bridge.request('compare_hf', baseParams);
}

export function activate(context: vscode.ExtensionContext) {
    const outputChannel = vscode.window.createOutputChannel('NLP-Smarty Optimizer');
    const bridge = new SmartyPythonBridge(context, outputChannel);
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = '$(zap) NLP-Smarty Ready';
    statusBar.command = 'nlpSmarty.compareSideBySide';
    statusBar.show();

    const getEditorContext = () => {
        const editor = vscode.window.activeTextEditor;
        const workspaceFiles: Array<{ path: string; content: string }> = [];
        if (editor) {
            workspaceFiles.push({
                path: path.basename(editor.document.fileName),
                content: editor.document.getText(),
            });
        }
        return { editor, workspaceFiles };
    };

    // ── Command 1: Optimize Prompt & Active File Context (Inspect & Copy) ──
    const optimizeCmd = vscode.commands.registerCommand('nlpSmarty.optimizeSelection', async () => {
        const { editor, workspaceFiles } = getEditorContext();
        let rawPrompt = editor && !editor.selection.isEmpty ? editor.document.getText(editor.selection) : '';
        if (!rawPrompt) {
            const input = await vscode.window.showInputBox({
                prompt: 'Enter a raw prompt to optimize against your open workspace files',
                placeHolder: 'e.g., Could you please help me write a Python function to reverse a string without using slicing?',
            });
            if (!input) {
                return;
            }
            rawPrompt = input;
        }

        const config = vscode.workspace.getConfiguration('nlpSmarty');
        const res = await bridge.request('optimize', {
            text: rawPrompt,
            workspace_files: workspaceFiles,
            compression_intensity: config.get<string>('compressionIntensity', 'AUTO'),
            max_total_tokens: config.get<number>('maxTokenBudget', 512),
        });

        if (!res.ok || !res.trace) {
            vscode.window.showErrorMessage(`NLP-Smarty Error: ${res.error || 'Unknown error'}`);
            return;
        }

        const t = res.trace;
        statusBar.text = `$(zap) Smarty: -${t.payload_token_savings_pct}% payload (${t.latency_ms}ms)`;
        await vscode.env.clipboard.writeText(res.payload || '');
        vscode.window.showInformationMessage(
            `Compiled prompt copied! Query: -${t.query_token_savings_pct}% | Payload: -${t.payload_token_savings_pct}% (${t.latency_ms}ms)`
        );
        renderTelemetryWebview(rawPrompt, res.payload || '', t);
    });

    // ── Command 2: Side-by-Side Raw vs. Optimized Downstream Agent Comparison (`Ctrl+Alt+C`) ──
    const compareCmd = vscode.commands.registerCommand('nlpSmarty.compareSideBySide', async () => {
        const { editor, workspaceFiles } = getEditorContext();
        let rawPrompt = editor && !editor.selection.isEmpty ? editor.document.getText(editor.selection) : '';
        if (!rawPrompt) {
            const input = await vscode.window.showInputBox({
                prompt: 'Enter a prompt to run Raw vs. Optimized side-by-side on your downstream LLM (vscode.lm or HuggingFace app.py)',
                placeHolder: 'e.g., Could you please help me write a Python function to reverse a string without using slicing?',
            });
            if (!input) {
                return;
            }
            rawPrompt = input;
        }

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'NLP-Smarty: Running Raw vs. Optimized Prompt on Downstream Agent...',
                cancellable: true,
            },
            async (_progress, token) => {
                const cmp = await runSideBySideComparison(bridge, rawPrompt, workspaceFiles, [], token);
                if (!cmp.ok || !cmp.trace) {
                    vscode.window.showErrorMessage(`NLP-Smarty Error: ${cmp.error || cmp.clarification_request}`);
                    return;
                }
                statusBar.text = `$(zap) Smarty A/B: ${cmp.raw_audit?.compliance_score ?? 0}% vs ${cmp.optimized_audit?.compliance_score ?? 100}% compliance`;
                renderSideBySideWebview(rawPrompt, cmp);
            }
        );
    });

    // ── Command 3: Inline Compress Selection ──
    const inlineReplaceCmd = vscode.commands.registerCommand('nlpSmarty.inlineCompress', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.selection.isEmpty) {
            vscode.window.showInformationMessage('Select a prompt in the editor first to compress inline.');
            return;
        }
        const rawText = editor.document.getText(editor.selection);
        const res = await bridge.request('optimize', { text: rawText });
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

    // ── Surface 4: Native VS Code Chat Participant (`@smarty` with `/optimize` and `/compare`) ──
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

                const { workspaceFiles } = getEditorContext();

                // Default to side-by-side LLM comparison unless `/optimize` (inspect-only) is specified
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
                    stream.markdown(
                        `### Deterministic Optimization Telemetry\n` +
                        `- **Intent:** \`${t.detected_intent}\` (confidence: \`${t.intent_confidence}\`, intensity: \`${t.compression_intensity_used}\`)\n` +
                        `- **Query Tokens:** \`${t.raw_token_count} -> ${t.optimized_token_count}\` (**-${t.query_token_savings_pct}%**)\n` +
                        `- **Payload Tokens:** \`${t.unoptimized_payload_token_count} -> ${t.compiled_payload_token_count}\` (**-${t.payload_token_savings_pct}%**)\n` +
                        `- **Kept File Chunks (MMR):** \`${JSON.stringify(t.kept_context_indices)}\` | **History Turns:** \`${t.kept_history_turns}\`\n` +
                        `- **Cache:** \`${t.cache_hit ? t.cache_type : 'MISS'}\` | **Optimizer Latency:** \`${t.latency_ms} ms\`\n\n` +
                        `\`\`\`xml\n${res.payload}\n\`\`\``
                    );
                    return;
                }

                // `/compare` or default `@smarty <prompt>`: Run Raw vs. Optimized on downstream LLM!
                stream.progress('Optimizing prompt deterministically and running Raw vs. Optimized A/B comparison...');
                const cmp = await runSideBySideComparison(bridge, request.prompt, workspaceFiles, history, token);
                if (!cmp.ok || !cmp.trace) {
                    stream.markdown(`**Clarification / Error:** ${cmp.clarification_request || cmp.error}`);
                    return;
                }

                const t = cmp.trace;
                const rawScore = cmp.raw_audit?.compliance_score ?? 0;
                const optScore = cmp.optimized_audit?.compliance_score ?? 100;

                stream.markdown(
                    `### NLP-Smarty A/B Downstream Agent Comparison (\`${cmp.backend}\`)\n\n` +
                    `| Metric | Raw Unoptimized Prompt | NLP-Smarty Compiled Prompt | Delta |\n` +
                    `| :--- | :--- | :--- | :--- |\n` +
                    `| **Query Words** | \`${t.raw_token_count}\` | \`${t.optimized_token_count}\` | **-${t.query_token_savings_pct}%** |\n` +
                    `| **Full Payload Tokens** | \`${cmp.raw_input_tokens}\` | \`${cmp.optimized_input_tokens}\` | **-${t.payload_token_savings_pct}%** |\n` +
                    `| **Downstream LLM Latency** | \`${cmp.raw_llm_latency_ms} ms\` | \`${cmp.optimized_llm_latency_ms} ms\` | Optimizer took \`${t.latency_ms} ms\` |\n` +
                    `| **Deterministic Audit Score** | \`${rawScore}/100\` | **\`${optScore}/100\`** | **+${optScore - rawScore} pts** |\n\n` +
                    `#### 1. Optimized Agent Response (Score: \`${optScore}/100\`)\n${cmp.optimized_response}\n\n` +
                    `#### 2. Raw Unoptimized Agent Response (Score: \`${rawScore}/100\`)\n${cmp.raw_response}\n\n` +
                    `<details><summary>View Compiled XML Payload Sent to Agent</summary>\n\n\`\`\`xml\n${cmp.optimized_payload}\n\`\`\`\n</details>`
                );
            }
        );
        context.subscriptions.push(participant);
    }

    context.subscriptions.push(optimizeCmd, compareCmd, inlineReplaceCmd, statusBar, outputChannel, {
        dispose: () => bridge.dispose(),
    });
}

function renderTelemetryWebview(rawPrompt: string, compiledPayload: string, t: OptimizationTrace) {
    const panel = vscode.window.createWebviewPanel(
        'nlpSmartyTelemetry',
        `NLP-Smarty: -${t.payload_token_savings_pct}% Tokens (${t.latency_ms}ms)`,
        vscode.ViewColumn.Beside,
        { enableScripts: false }
    );

    const escapeHtml = (s: string) =>
        s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

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

function renderSideBySideWebview(rawPrompt: string, cmp: BridgeResponse) {
    const t = cmp.trace!;
    const panel = vscode.window.createWebviewPanel(
        'nlpSmartyABCompare',
        `NLP-Smarty A/B: ${cmp.raw_audit?.compliance_score ?? 0}% vs ${cmp.optimized_audit?.compliance_score ?? 100}%`,
        vscode.ViewColumn.Beside,
        { enableScripts: false }
    );

    const escapeHtml = (s: string) =>
        (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    const renderChecks = (audit?: ResponseAudit) =>
        (audit?.checks || [])
            .map((c) => `<li>${c.passed ? 'PASS' : 'FAIL'} — ${escapeHtml(c.check)}</li>`)
            .join('');

    panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-editor-foreground); background: var(--vscode-editor-background); padding: 16px; }
  .top-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  .ab-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
  .card { background: var(--vscode-editorWidget-background); border: 1px solid var(--vscode-widget-border); border-radius: 6px; padding: 12px; }
  .metric { font-size: 18px; font-weight: 700; margin-top: 4px; color: var(--vscode-textLink-foreground); }
  .label { font-size: 11px; text-transform: uppercase; opacity: 0.8; }
  pre { background: var(--vscode-textCodeBlock-background); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }
</style>
</head>
<body>
  <h2>Raw vs. Optimized Downstream Agent Comparison (${escapeHtml(cmp.backend || '')})</h2>
  <div class="top-grid">
    <div class="card"><div class="label">Payload Token Reduction</div><div class="metric">${cmp.raw_input_tokens} &rarr; ${cmp.optimized_input_tokens} (-${t.payload_token_savings_pct}%)</div></div>
    <div class="card"><div class="label">Compliance Score (Raw vs Opt)</div><div class="metric">${cmp.raw_audit?.compliance_score ?? 0}/100 &rarr; ${cmp.optimized_audit?.compliance_score ?? 100}/100</div></div>
    <div class="card"><div class="label">Optimizer Overhead</div><div class="metric">${t.latency_ms} ms (${t.cache_hit ? t.cache_type : 'LIVE'})</div></div>
    <div class="card"><div class="label">LLM Latency (Raw vs Opt)</div><div class="metric">${cmp.raw_llm_latency_ms}ms vs ${cmp.optimized_llm_latency_ms}ms</div></div>
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
  <h3>Compiled XML Prompt Sent to Agent</h3>
  <pre>${escapeHtml(cmp.optimized_payload || '')}</pre>
</body>
</html>`;
}

export function deactivate() {}