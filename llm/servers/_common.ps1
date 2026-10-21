<#
Shared by the serve_*.ps1 scripts: finds llama-server and the model file, prints the .env
settings for the pipeline, then runs the server in this window (Ctrl+C stops it).
#>

function Resolve-LlamaServer([string]$LlamaDir) {
    if (-not $LlamaDir) { $LlamaDir = if ($env:LLAMA_CPP_DIR) { $env:LLAMA_CPP_DIR } else { Join-Path $HOME "tools\llama.cpp" } }
    $exe = Join-Path $LlamaDir "llama-server.exe"
    if (-not (Test-Path $exe)) {
        throw "llama-server.exe not found in '$LlamaDir'. Pass -LlamaDir or set LLAMA_CPP_DIR (see README, 'Local LLM server')."
    }
    return $exe
}

# $Pattern may contain wildcards, e.g. the snapshot hash folder of the Hugging Face cache.
function Resolve-Model([string]$Path, [string]$Pattern, [string]$DownloadHint) {
    $candidate = if ($Path) { $Path } else { $Pattern }
    $found = Get-Item -Path $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $found) {
        throw "Model not found: $candidate`n$DownloadHint`nOr pass the file with -Model <path.gguf>."
    }
    return $found.FullName
}

function Start-LlamaServer {
    param(
        [string]$Server,
        [string]$Model,
        [string]$Alias,
        [string]$BindHost,
        [int]$Port,
        [string[]]$ModelArgs,
        [string[]]$ExtraArgs
    )
    $busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        throw "Port $Port is already in use (process $($busy[0].OwningProcess)). Stop that server first or pass -Port."
    }

    $common = @(
        "-m", $Model,
        "--alias", $Alias,
        "--host", $BindHost,
        "--port", $Port,
        "--n-gpu-layers", "all",
        "--flash-attn", "on",
        "-ctk", "q8_0", "-ctv", "q8_0",   # 8-bit KV cache: half the memory of f16, no measurable quality loss
        "--parallel", "1",                # one conversation at a time: all VRAM goes to that slot
        "--jinja"                         # use the chat template embedded in the GGUF
    )

    Write-Host ""
    Write-Host "Model : $Model"
    Write-Host "Server: http://${BindHost}:$Port  (web UI and OpenAI API)"
    Write-Host ""
    Write-Host "Pipeline settings for .env:"
    Write-Host "  OPENAI_BASE_URL=http://${BindHost}:$Port/v1"
    Write-Host "  OPENAI_API_KEY=local"
    Write-Host "  OPENAI_MODEL=$Alias"
    Write-Host ""
    Write-Host "Starting llama-server (Ctrl+C to stop)..."
    & $Server @common @ModelArgs @ExtraArgs
}
