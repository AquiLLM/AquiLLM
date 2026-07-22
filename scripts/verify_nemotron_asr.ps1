[CmdletBinding()]
param(
    [switch]$AssertGpuIdle,
    [int[]]$AllowGpuPid = @(),
    [switch]$PrepareEnvironments,
    [switch]$VerifyNemotron
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ModelId = "nvidia/nemotron-3.5-asr-streaming-0.6b"
$Revision = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
$ServedName = "nemotron-3.5-asr-streaming-0.6b"
$Image = "aquillm-vllm-transcribe:test"
$Project = "aquillm-asr-verification"
$CacheVolume = "aquillm_nemotron_asr_hf_cache"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StateRoot = Join-Path ([IO.Path]::GetTempPath()) "aquillm-nemotron-asr-verification"
$ArtifactRoot = Join-Path $StateRoot "artifacts"
$ComposePath = Join-Path $StateRoot "compose.yml"
$NemotronEnvPath = Join-Path $StateRoot "nemotron.env"
$WhisperEnvPath = Join-Path $StateRoot "whisper.env"
$ActivationPath = Join-Path $StateRoot "activate.ps1"
$MemoryCsv = Join-Path $ArtifactRoot "gpu-memory.csv"

function Set-Utf8NoBom {
    param([Parameter(Mandatory)] [string]$Path, [Parameter(ValueFromPipeline)] [string]$Content)
    process {
        [IO.File]::WriteAllText(
            $Path,
            $Content,
            [Text.UTF8Encoding]::new($false)
        )
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [Parameter(ValueFromRemainingArguments)] [string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-RunningGpuContainers {
    $result = @()
    $containerIds = @(& docker ps -q)
    if ($LASTEXITCODE -ne 0) {
        throw "docker ps failed"
    }
    foreach ($containerId in $containerIds) {
        if (-not $containerId) { continue }
        $inspection = (& docker inspect $containerId | ConvertFrom-Json)[0]
        $requests = @($inspection.HostConfig.DeviceRequests | Where-Object { $null -ne $_ })
        $gpuRequests = @(
            $requests | Where-Object {
                $capabilities = @($_.Capabilities | ForEach-Object { @($_) })
                ($capabilities -join ",") -match "gpu" -or $_.Driver -eq "nvidia"
            }
        )
        if ($gpuRequests.Count -gt 0) {
            $result += [pscustomobject]@{
                Id = $inspection.Id.Substring(0, 12)
                Name = $inspection.Name.TrimStart("/")
                Image = $inspection.Config.Image
                DeviceRequests = ($gpuRequests | ConvertTo-Json -Compress -Depth 8)
            }
        }
    }
    return $result
}

function Assert-GpuIsIdle {
    param([int[]]$AllowedPids = @())

    $rawRows = @(& nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi compute inventory failed" }
    $numericProcesses = @()
    foreach ($row in $rawRows) {
        if ($row -match '^\s*(\d+)\s*,\s*(.*?)\s*,\s*(\d+)\s*$') {
            $pidValue = [int]$Matches[1]
            if ($pidValue -notin $AllowedPids) {
                $numericProcesses += $row
            }
        }
    }
    $gpuContainers = @(Get-RunningGpuContainers)

    Write-Host "nvidia-smi compute applications (WDDM may report N/A):"
    if ($rawRows.Count) { $rawRows | ForEach-Object { Write-Host "  $_" } } else { Write-Host "  <none>" }
    Write-Host "running Docker containers with GPU DeviceRequests:"
    if ($gpuContainers.Count) { $gpuContainers | Format-Table | Out-String | Write-Host } else { Write-Host "  <none>" }
    Write-Host "explicitly allowed GPU PIDs: $($AllowedPids -join ', ')"

    if ($numericProcesses.Count -or $gpuContainers.Count) {
        throw "GPU is not isolated: found unallowed numeric compute processes or GPU containers"
    }
}

function Write-IsolatedEnvironmentFiles {
    New-Item -ItemType Directory -Force $StateRoot, $ArtifactRoot | Out-Null
    @"
ASR_SERVICE_ENV=$($NemotronEnvPath.Replace('\', '/'))
TRANSCRIBE_VLLM_MODEL=$ModelId
TRANSCRIBE_VLLM_REVISION=$Revision
TRANSCRIBE_VLLM_SERVED_MODEL_NAME=$ServedName
TRANSCRIBE_VLLM_TOKENIZER=$ModelId
TRANSCRIBE_VLLM_TENSOR_PARALLEL_SIZE=1
TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION=0.20
TRANSCRIBE_VLLM_MAX_MODEL_LEN=50000
TRANSCRIBE_VLLM_DTYPE=float32
TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
TRANSCRIBE_VLLM_TRUST_REMOTE_CODE=1
TRANSCRIBE_VLLM_EXTRA_ARGS=--enforce-eager --max-num-seqs 1 --max-num-batched-tokens 50000 --generation-config /opt/aquillm/nemotron-generation-config
"@ | Set-Utf8NoBom -Path $NemotronEnvPath
    @"
ASR_SERVICE_ENV=$($WhisperEnvPath.Replace('\', '/'))
TRANSCRIBE_VLLM_MODEL=openai/whisper-large-v3-turbo
TRANSCRIBE_VLLM_REVISION=
TRANSCRIBE_VLLM_SERVED_MODEL_NAME=whisper-large-v3-turbo
TRANSCRIBE_VLLM_TOKENIZER=openai/whisper-large-v3-turbo
TRANSCRIBE_VLLM_TENSOR_PARALLEL_SIZE=1
TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION=0.08
TRANSCRIBE_VLLM_MAX_MODEL_LEN=448
TRANSCRIBE_VLLM_DTYPE=float16
TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN=0
TRANSCRIBE_VLLM_TRUST_REMOTE_CODE=1
TRANSCRIBE_VLLM_EXTRA_ARGS=--quantization bitsandbytes --load-format bitsandbytes --model-loader-extra-config '{"load_in_8bit":true}' --max-num-seqs 1 --max-num-batched-tokens 448 --limit-mm-per-prompt '{"audio":{"count":1,"length":30}}'
"@ | Set-Utf8NoBom -Path $WhisperEnvPath
    @"
name: $Project
services:
  vllm_transcribe:
    image: $Image
    env_file:
      - `${ASR_SERVICE_ENV:?ASR_SERVICE_ENV must name the generated service env}
    environment:
      VLLM_USE_V2_MODEL_RUNNER: "0"
      VLLM_SERVICE_KIND: transcribe
      LLM_CHOICE: QWEN3_30B
      VLLM_MODEL: `${TRANSCRIBE_VLLM_MODEL:?}
      VLLM_REVISION: `${TRANSCRIBE_VLLM_REVISION:-}
      VLLM_SERVED_MODEL_NAME: `${TRANSCRIBE_VLLM_SERVED_MODEL_NAME:?}
      VLLM_TOKENIZER: `${TRANSCRIBE_VLLM_TOKENIZER:?}
      VLLM_TENSOR_PARALLEL_SIZE: `${TRANSCRIBE_VLLM_TENSOR_PARALLEL_SIZE:?}
      VLLM_GPU_MEMORY_UTILIZATION: `${TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION:?}
      VLLM_MAX_MODEL_LEN: `${TRANSCRIBE_VLLM_MAX_MODEL_LEN:?}
      VLLM_DTYPE: `${TRANSCRIBE_VLLM_DTYPE:?}
      VLLM_ALLOW_LONG_MAX_MODEL_LEN: `${TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN:?}
      VLLM_TRUST_REMOTE_CODE: `${TRANSCRIBE_VLLM_TRUST_REMOTE_CODE:?}
      VLLM_EXTRA_ARGS: `${TRANSCRIBE_VLLM_EXTRA_ARGS:?}
      HF_HOME: /root/.cache/huggingface
    ports:
      - "8005:8000"
    volumes:
      - nemotron_hf_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2).read()\""]
      interval: 10s
      timeout: 10s
      retries: 90
      start_period: 60s
volumes:
  nemotron_hf_cache:
    external: true
    name: $CacheVolume
"@ | Set-Utf8NoBom -Path $ComposePath
    @"
`$env:NEMOTRON_ASR_ENV = '$NemotronEnvPath'
`$env:WHISPER_ASR_ENV = '$WhisperEnvPath'
`$env:NEMOTRON_ASR_OVERRIDE = '$ComposePath'
`$env:NEMOTRON_ASR_RUNTIME_COMPOSE = '$ComposePath'
"@ | Set-Utf8NoBom -Path $ActivationPath

    $env:NEMOTRON_ASR_ENV = $NemotronEnvPath
    $env:WHISPER_ASR_ENV = $WhisperEnvPath
    $env:NEMOTRON_ASR_OVERRIDE = $ComposePath
    $env:NEMOTRON_ASR_RUNTIME_COMPOSE = $ComposePath
    Write-Host "Prepared isolated runtime files under $StateRoot"
    Write-Host "Dot-source $ActivationPath to expose paths in another PowerShell process."
}

function Assert-RenderedNemotronConfig {
    $renderedPath = Join-Path $ArtifactRoot "nemotron-compose-config.json"
    $rendered = & docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath config --format json
    if ($LASTEXITCODE -ne 0) { throw "docker compose config failed" }
    ($rendered -join [Environment]::NewLine) | Set-Utf8NoBom -Path $renderedPath
    $service = (($rendered -join [Environment]::NewLine) | ConvertFrom-Json).services.vllm_transcribe
    $expected = @{
        VLLM_MODEL = $ModelId
        VLLM_REVISION = $Revision
        VLLM_SERVED_MODEL_NAME = $ServedName
        VLLM_TOKENIZER = $ModelId
        VLLM_DTYPE = "float32"
        VLLM_MAX_MODEL_LEN = "50000"
        VLLM_GPU_MEMORY_UTILIZATION = "0.20"
        VLLM_TENSOR_PARALLEL_SIZE = "1"
        VLLM_USE_V2_MODEL_RUNNER = "0"
        VLLM_SERVICE_KIND = "transcribe"
    }
    foreach ($entry in $expected.GetEnumerator()) {
        if ([string]$service.environment.($entry.Key) -ne $entry.Value) {
            throw "Rendered $($entry.Key) did not match: $($service.environment.($entry.Key))"
        }
    }
    if ($service.image -ne $Image) { throw "Rendered image was $($service.image)" }
    if ($service.environment.VLLM_EXTRA_ARGS -match "bitsandbytes|whisper") {
        throw "Rendered Nemotron arguments contain Whisper rollback flags"
    }
    Write-Host "Rendered standalone Compose config passed exact Nemotron assertions."
}

function Assert-RepositoryComposeProfiles {
    $mirrorRoot = Join-Path $StateRoot "repository-compose-render"
    $mirrorCompose = Join-Path $mirrorRoot "deploy\compose"
    New-Item -ItemType Directory -Force $mirrorCompose | Out-Null
    Copy-Item (Join-Path $RepoRoot "deploy\compose\*.yml") $mirrorCompose -Force
    Copy-Item (Join-Path $RepoRoot ".env.example") (Join-Path $mirrorRoot ".env") -Force

    foreach ($composeName in @("base.yml", "development.yml", "production.yml")) {
        $composeFile = Join-Path $mirrorCompose $composeName
        $rendered = @(& docker compose --profile vllm `
            --env-file (Join-Path $RepoRoot ".env.example") `
            --env-file $NemotronEnvPath -f $composeFile config --format json)
        if ($LASTEXITCODE -ne 0) { throw "Failed to render repository $composeName" }
        $renderedText = $rendered -join [Environment]::NewLine
        $service = ($renderedText | ConvertFrom-Json).services.vllm_transcribe
        if ([string]$service.environment.VLLM_DTYPE -ne "float32") {
            throw "$composeName did not render explicit float32"
        }
        if ($service.environment.VLLM_MODEL -ne $ModelId) {
            throw "$composeName rendered the wrong Nemotron model"
        }
        if ($service.environment.VLLM_REVISION -ne $Revision) {
            throw "$composeName rendered the wrong Nemotron revision"
        }
        $renderedText | Set-Utf8NoBom -Path (
            Join-Path $ArtifactRoot "repository-$($composeName.Replace('.yml', ''))-config.json"
        )
    }
    Write-Host "Rendered base/development/production profiles passed float32 assertions."
}

function Prefetch-PinnedCheckpoint {
    Invoke-Checked -FilePath docker -ArgumentList @("volume", "create", $CacheVolume) | Out-Null
    $prefetchPath = Join-Path $StateRoot "prefetch.py"
    @'
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open

model = "nvidia/nemotron-3.5-asr-streaming-0.6b"
revision = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
filenames = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
cache_dir = Path(os.environ["HF_HOME"]) / "hub"
paths = {
    filename: Path(hf_hub_download(model, filename, revision=revision, cache_dir=cache_dir))
    for filename in filenames
}
weight = paths["model.safetensors"]
digest = hashlib.sha256()
with weight.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
assert weight.stat().st_size == 2_552_062_944, weight.stat().st_size
assert digest.hexdigest() == "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174"
with safe_open(weight, framework="pt", device="cpu") as checkpoint:
    names = list(checkpoint.keys())
    params = sum(checkpoint.get_tensor(name).numel() for name in names)
counts = {
    prefix: sum(name.startswith(prefix + ".") for name in names)
    for prefix in ["encoder", "decoder", "prompt_projector", "encoder_projector", "joint"]
}
assert len(names) == 655
assert params == 637_997_088
assert counts == {"encoder": 636, "decoder": 11, "prompt_projector": 4, "encoder_projector": 2, "joint": 2}
print(json.dumps({
    "model": model,
    "revision": revision,
    "files": {name: str(path) for name, path in paths.items()},
    "model_safetensors_bytes": weight.stat().st_size,
    "model_safetensors_sha256": digest.hexdigest(),
    "tensor_count": len(names),
    "parameter_count": params,
    "prefix_counts": counts,
}, indent=2, sort_keys=True))
'@ | Set-Utf8NoBom -Path $prefetchPath
    $output = & docker run --rm --entrypoint python3 `
        -e HF_HOME=/root/.cache/huggingface `
        -v "${CacheVolume}:/root/.cache/huggingface" `
        -v "${StateRoot}:/verification" `
        $Image /verification/prefetch.py
    if ($LASTEXITCODE -ne 0) { throw "Pinned checkpoint prefetch/inventory failed" }
    ($output -join [Environment]::NewLine) | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "checkpoint-inventory.json")
    $output | Write-Host
}

function Invoke-ContainerPytest {
    param([string]$Command, [hashtable]$Environment = @{}, [string[]]$Volumes = @())
    $arguments = @("run", "--rm", "--gpus", "all", "--entrypoint", "bash")
    foreach ($entry in $Environment.GetEnumerator()) {
        $arguments += @("-e", "$($entry.Key)=$($entry.Value)")
    }
    foreach ($volume in $Volumes) { $arguments += @("-v", $volume) }
    $arguments += @("-v", "${RepoRoot}:/workspace", "-w", "/workspace", $Image, "-lc", $Command)
    Invoke-Checked -FilePath docker -ArgumentList $arguments
}

function Start-GpuMemorySampler {
    param([string]$OutputPath)
    Remove-Item -Force -ErrorAction SilentlyContinue $OutputPath
    return Start-Job -ScriptBlock {
        param($Path)
        "timestamp,memory_used_mib" | Set-Content -Encoding ascii $Path
        while ($true) {
            $value = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
            "$(Get-Date -Format o),$value" | Add-Content -Encoding ascii $Path
            Start-Sleep -Seconds 1
        }
    } -ArgumentList $OutputPath
}

function Get-MemorySummary {
    param([string]$Path, [int]$Baseline)
    $samples = @(Import-Csv $Path | ForEach-Object { [int]$_.memory_used_mib })
    if (-not $samples.Count) { throw "No GPU memory samples were recorded" }
    $steadyWindow = @($samples | Select-Object -Last ([Math]::Min(30, $samples.Count)))
    return [pscustomobject]@{
        baseline_mib = $Baseline
        peak_mib = ($samples | Measure-Object -Maximum).Maximum
        steady_mib = [Math]::Round(($steadyWindow | Measure-Object -Average).Average, 1)
        sample_count = $samples.Count
    }
}

function Wait-AsrHealth {
    $deadline = (Get-Date).AddSeconds(900)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:8005/health" -TimeoutSec 3 | Out-Null
            return
        } catch {
            Start-Sleep -Seconds 5
        }
    }
    & docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath ps
    & docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath logs --no-color vllm_transcribe
    throw "Nemotron ASR health timeout after 900 seconds"
}

function Invoke-FullNemotronVerification {
    Write-IsolatedEnvironmentFiles
    Assert-RenderedNemotronConfig
    Assert-RepositoryComposeProfiles
    & docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath down --remove-orphans
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid

    Invoke-Checked -FilePath docker -ArgumentList @(
        "build", "-f", "deploy/docker/vllm/Dockerfile.transcribe", "-t", $Image, "."
    )
    $imageInspect = & docker image inspect $Image | ConvertFrom-Json
    $imageInspect | ConvertTo-Json -Depth 12 | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "image-inspect.json")
    Invoke-Checked -FilePath docker -ArgumentList @(
        "run", "--rm", "--gpus", "all", "--entrypoint", "python3", $Image,
        "/probe_nemotron_plugin.py", "--generation-config", "/opt/aquillm/nemotron-generation-config"
    )
    Invoke-Checked -FilePath docker -ArgumentList @(
        "run", "--rm", "--entrypoint", "python3", $Image, "-c",
        "import importlib.metadata as m; import aquillm_vllm_nemotron_asr as p; print(m.version('aquillm-vllm-nemotron-asr')); print(p.__file__); print(m.version('vllm')); print(m.version('transformers')); print(m.version('torch')); print(m.version('librosa'))"
    )
    Invoke-Checked -FilePath docker -ArgumentList @(
        "run", "--rm", "--entrypoint", "python3", $Image, "-m", "pip", "check"
    )
    Prefetch-PinnedCheckpoint

    Invoke-ContainerPytest `
        -Environment @{ VLLM_GPU_MEMORY_UTILIZATION = "0.05"; VLLM_USE_V2_MODEL_RUNNER = "0" } `
        -Command "python3 -m pip install -q pytest==8.4.1 && python3 -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_engine_lifecycle.py deploy/vllm_plugins/nemotron_asr/tests/test_model.py -q"
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid

    Invoke-ContainerPytest `
        -Environment @{
            ASR_FULL_PARITY_PHASE = "direct"
            ASR_PARITY_ARTIFACT_DIR = "/artifacts"
            HF_HOME = "/root/.cache/huggingface"
            HF_HUB_OFFLINE = "1"
            TRANSFORMERS_OFFLINE = "1"
            CUBLAS_WORKSPACE_CONFIG = ":4096:8"
        } `
        -Volumes @("${CacheVolume}:/root/.cache/huggingface", "${ArtifactRoot}:/artifacts") `
        -Command "python3 -m pip install -q pytest==8.4.1 && python3 -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_transformers_parity.py -q -m gpu"
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid

    Invoke-ContainerPytest `
        -Environment @{
            ASR_FULL_PARITY_PHASE = "plugin"
            ASR_PARITY_ARTIFACT_DIR = "/artifacts"
            HF_HOME = "/root/.cache/huggingface"
            HF_HUB_OFFLINE = "1"
            TRANSFORMERS_OFFLINE = "1"
            VLLM_USE_V2_MODEL_RUNNER = "0"
            VLLM_ALLOW_LONG_MAX_MODEL_LEN = "1"
            CUBLAS_WORKSPACE_CONFIG = ":4096:8"
        } `
        -Volumes @("${CacheVolume}:/root/.cache/huggingface", "${ArtifactRoot}:/artifacts") `
        -Command "python3 -m pip install -q pytest==8.4.1 && python3 -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_transformers_parity.py deploy/vllm_plugins/nemotron_asr/tests/test_weight_loading.py -q -m gpu"
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid

    $baseline = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim())
    $sampler = Start-GpuMemorySampler -OutputPath $MemoryCsv
    try {
        Invoke-Checked -FilePath docker -ArgumentList @(
            "compose", "--project-name", $Project, "--env-file", $NemotronEnvPath,
            "-f", $ComposePath, "up", "-d", "--no-deps", "vllm_transcribe"
        )
        Wait-AsrHealth
        $startupLogs = @(& docker compose --project-name $Project `
            --env-file $NemotronEnvPath -f $ComposePath logs --no-color vllm_transcribe) `
            -join [Environment]::NewLine
        foreach ($requiredLog in @(
            "'dtype': 'float32'",
            "Initializing a V1 LLM engine",
            "dtype=torch.float32",
            "max_seq_len=50000",
            "enforce_eager=True",
            "'max_num_batched_tokens': 50000",
            "'max_num_seqs': 1"
        )) {
            if (-not $startupLogs.Contains($requiredLog)) {
                throw "Startup log did not prove: $requiredLog"
            }
        }
        $models = Invoke-RestMethod -Uri "http://127.0.0.1:8005/v1/models" -Headers @{ Authorization = "Bearer EMPTY" }
        $ids = @($models.data | ForEach-Object { $_.id })
        if ($ids.Count -ne 1 -or $ids[0] -ne $ServedName) {
            throw "Unexpected served model IDs: $($ids -join ', ')"
        }
        $hostPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path $hostPython)) { throw "Host test Python not found: $hostPython" }
        $env:RUN_ASR_RUNTIME = "1"
        $env:ASR_BASE_URL = "http://127.0.0.1:8005/v1"
        $env:SECRET_KEY = "nemotron-runtime-verification-only"
        $env:GOOGLE_OAUTH2_CLIENT_ID = "nemotron-runtime-verification"
        $env:GOOGLE_OAUTH2_CLIENT_SECRET = "nemotron-runtime-verification"
        $env:OPENAI_API_KEY = "sk-nemotron-runtime-verification"
        $env:GEMINI_API_KEY = "nemotron-runtime-verification"
        Push-Location $RepoRoot
        $runtimeStarted = Get-Date
        try {
            $runtimeOutput = @(& $hostPython -m pytest tests/asr -q 2>&1)
            $runtimeExitCode = $LASTEXITCODE
            $runtimeText = $runtimeOutput -join [Environment]::NewLine
            Write-Host $runtimeText
            $runtimeText | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "http-pytest.log")
            if ($runtimeExitCode -ne 0) {
                throw "Live ASR HTTP pytest failed with exit code $runtimeExitCode"
            }
            if (-not $runtimeText.Contains("29 passed")) {
                throw "Live ASR HTTP pytest did not report all 29 passing tests"
            }
            [pscustomobject]@{
                test_count = 29
                elapsed_seconds = [Math]::Round(((Get-Date) - $runtimeStarted).TotalSeconds, 3)
                base_url = $env:ASR_BASE_URL
                model = $ServedName
            } | ConvertTo-Json | Set-Utf8NoBom -Path (
                Join-Path $ArtifactRoot "http-pytest-summary.json"
            )
        }
        finally { Pop-Location }
        Start-Sleep -Seconds 30
    } finally {
        Stop-Job $sampler -ErrorAction SilentlyContinue
        Receive-Job $sampler -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $sampler -Force -ErrorAction SilentlyContinue
        (& docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath logs --no-color vllm_transcribe) -join [Environment]::NewLine | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "nemotron-service.log")
        & docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath down --remove-orphans
    }
    $memory = Get-MemorySummary -Path $MemoryCsv -Baseline $baseline
    $memory | ConvertTo-Json | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "gpu-memory-summary.json")
    $memory | Format-List | Out-String | Write-Host
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid
    Write-Host "Nemotron verification artifacts: $ArtifactRoot"
}

if (-not ($AssertGpuIdle -or $PrepareEnvironments -or $VerifyNemotron)) {
    throw "Select at least one switch: -AssertGpuIdle, -PrepareEnvironments, or -VerifyNemotron"
}
if ($PrepareEnvironments) {
    Write-IsolatedEnvironmentFiles
    Assert-RenderedNemotronConfig
    Assert-RepositoryComposeProfiles
}
if ($AssertGpuIdle) {
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid
}
if ($VerifyNemotron) {
    Invoke-FullNemotronVerification
}
