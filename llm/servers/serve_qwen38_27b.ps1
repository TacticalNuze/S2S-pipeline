<#
.SYNOPSIS
Serves Swift-1.5 Qwen3.8 27B (IQ3_S, 12.1 GB, with MTP) with llama.cpp on the GPU.

.DESCRIPTION
Best quality of the local models, for comparing the LLM against a cloud model on its own.
It takes 12.7 GB of VRAM, so it does NOT fit next to the live voice bot's ASR + TTS
(about 6 GB more): use serve_atlas_chat_9b.ps1 or serve_qwen35_9b.ps1 for the bot.

Measured on an RTX 5060 Ti 16 GB (llama.cpp b11669, thinking off): first token 0.3-0.7 s,
~28-38 tok/s on short Darija replies, ~47 tok/s on long answers.

.EXAMPLE
.\llm\servers\serve_qwen38_27b.ps1
.\llm\servers\serve_qwen38_27b.ps1 -Context 32768 -Port 8081
#>
param(
    [string]$Model,                       # default: the file in the Hugging Face cache folder below
    [int]$Context = 16384,
    [int]$Port = 8080,
    [string]$BindHost = "127.0.0.1",
    [string]$LlamaDir,                    # default: $env:LLAMA_CPP_DIR or ~\tools\llama.cpp
    [Parameter(ValueFromRemainingArguments)] [string[]]$ExtraArgs   # passed through to llama-server
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_common.ps1")

$server = Resolve-LlamaServer $LlamaDir
$modelPath = Resolve-Model $Model `
    (Join-Path $HOME ".cache\huggingface\hub\models-Switf1.5-Qwen27b\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf") `
    "Get it from https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b (GSQ-RCO GGUF, IQ3_S-mtp file)."

Start-LlamaServer -Server $server -Model $modelPath -Alias "qwen3.8-27b" -BindHost $BindHost -Port $Port -ExtraArgs $ExtraArgs -ModelArgs @(
    "-c", $Context,
    # MTP: the model drafts 2 tokens ahead with its built-in prediction layer. Measured best:
    # 2 beat 1 and 3 (47 vs 41 and 43 tok/s on long answers).
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--gpu-layers-draft", "all",
    # Reasoning model: thinking off, or the voice bot waits for hidden reasoning before speaking.
    "--reasoning", "off",
    # Qwen's recommended non-thinking sampling.
    "--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0", "--presence-penalty", "1.5"
)
