<#
.SYNOPSIS
Serves Qwen3.5 9B (Q4_K_M, 5.9 GB, with MTP) with llama.cpp on the GPU.

.DESCRIPTION
Small enough to run next to the live voice bot: with moulsot ASR and Habibi TTS loaded too,
the GPU peaked at 11.9 GB of 16.3 GB. Fast, but its Darija is noticeably weaker than
Atlas-Chat's and the 27B's (see benchmark/results/llm_compare_2026-10-21.md).

Measured on an RTX 5060 Ti 16 GB (llama.cpp b11669, thinking off): first token ~0.15 s,
first sentence ~0.6 s, ~58 tok/s.

.EXAMPLE
.\llm\servers\serve_qwen35_9b.ps1
#>
param(
    [string]$Model,                       # default: the file in the Hugging Face cache
    [int]$Context = 8192,
    [int]$Port = 8080,
    [string]$BindHost = "127.0.0.1",
    [string]$LlamaDir,                    # default: $env:LLAMA_CPP_DIR or ~\tools\llama.cpp
    [Parameter(ValueFromRemainingArguments)] [string[]]$ExtraArgs   # passed through to llama-server
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_common.ps1")

$server = Resolve-LlamaServer $LlamaDir
$modelPath = Resolve-Model $Model `
    (Join-Path $HOME ".cache\huggingface\hub\models--unsloth--Qwen3.5-9B-MTP-GGUF\snapshots\*\Qwen3.5-9B-Q4_K_M.gguf") `
    "Download it with: uv run hf download unsloth/Qwen3.5-9B-MTP-GGUF Qwen3.5-9B-Q4_K_M.gguf"

Start-LlamaServer -Server $server -Model $modelPath -Alias "qwen3.5-9b" -BindHost $BindHost -Port $Port -ExtraArgs $ExtraArgs -ModelArgs @(
    "-c", $Context,
    # MTP: drafts 2 tokens ahead with the model's built-in prediction layer.
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--gpu-layers-draft", "all",
    # Reasoning model: thinking off, or the voice bot waits for hidden reasoning before speaking.
    "--reasoning", "off",
    # Qwen's recommended non-thinking sampling.
    "--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0", "--presence-penalty", "1.5"
)
