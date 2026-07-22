[CmdletBinding()]
param(
    [switch]$AssertGpuIdle,
    [int[]]$AllowGpuPid = @(),
    [switch]$PrepareEnvironments,
    [switch]$SelfTest,
    [switch]$VerifyNemotron,
    [switch]$VerifyWhisperRollback,
    [switch]$VerifyProfile,
    [switch]$AllowIncompleteProfile
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ModelId = "nvidia/nemotron-3.5-asr-streaming-0.6b"
$Revision = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
$ServedName = "nemotron-3.5-asr-streaming-0.6b"
$Image = "aquillm-vllm-transcribe:test"
$Project = "aquillm-asr-verification"
$ProfileProject = "aquillm-vllm-profile-verification"
$ProfileImage = "aquillm-vllm-profile:test"
$CacheVolume = "aquillm_nemotron_asr_hf_cache"
$ProfileCacheVolume = "aquillm_vllm_profile_hf_cache"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StateRoot = Join-Path ([IO.Path]::GetTempPath()) "aquillm-nemotron-asr-verification"
$ArtifactRoot = Join-Path $StateRoot "artifacts"
$ComposePath = Join-Path $StateRoot "compose.yml"
$NemotronEnvPath = Join-Path $StateRoot "nemotron.env"
$WhisperEnvPath = Join-Path $StateRoot "whisper.env"
$ProfileEnvPath = Join-Path $StateRoot "profile.env"
$ProfileComposePath = Join-Path $StateRoot "profile-compose.yml"
$ActivationPath = Join-Path $StateRoot "activate.ps1"
$MemoryCsv = Join-Path $ArtifactRoot "gpu-memory.csv"
$DirectParityArtifactNames = @(
    "direct-transformers.json",
    "direct-joint-logits.npy"
)
$DirectParitySummaryName = "direct-phase-summary.json"

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

function Get-ProcessEnvironmentSnapshot {
    param([Parameter(Mandatory)] [string[]]$Names)

    $processEnvironment = [Environment]::GetEnvironmentVariables("Process")
    $snapshot = @{}
    foreach ($name in $Names) {
        $exists = $processEnvironment.Contains($name)
        $snapshot[$name] = [pscustomobject]@{
            Exists = $exists
            Value = if ($exists) { [string]$processEnvironment[$name] } else { $null }
        }
    }
    return $snapshot
}

function Restore-ProcessEnvironment {
    param([Parameter(Mandatory)] [hashtable]$Snapshot)

    foreach ($name in $Snapshot.Keys) {
        $entry = $Snapshot[$name]
        $value = if ($entry.Exists) { $entry.Value } else { $null }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

function Invoke-WithTemporaryEnvironment {
    param(
        [Parameter(Mandatory)] [hashtable]$Environment,
        [Parameter(Mandatory)] [scriptblock]$ScriptBlock
    )

    $snapshot = Get-ProcessEnvironmentSnapshot -Names @($Environment.Keys)
    try {
        foreach ($entry in $Environment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable(
                $entry.Key,
                [string]$entry.Value,
                "Process"
            )
        }
        & $ScriptBlock
    }
    finally {
        Restore-ProcessEnvironment -Snapshot $snapshot
    }
}

function Remove-DirectParityArtifacts {
    param([Parameter(Mandatory)] [string]$Root)

    foreach ($name in $DirectParityArtifactNames) {
        Remove-Item -LiteralPath (Join-Path $Root $name) `
            -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath (Join-Path $Root $DirectParitySummaryName) `
        -Force -ErrorAction SilentlyContinue
}

function Assert-FreshDirectParityArtifacts {
    param(
        [Parameter(Mandatory)] [string]$Root,
        [Parameter(Mandatory)] [DateTime]$PhaseStartedUtc
    )

    foreach ($name in $DirectParityArtifactNames) {
        $path = Join-Path $Root $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Direct parity artifact was not produced: $path"
        }
        $artifact = Get-Item -LiteralPath $path
        if ($artifact.Length -le 0) {
            throw "Direct parity artifact was empty: $path"
        }
        if ($artifact.LastWriteTimeUtc -lt $PhaseStartedUtc) {
            throw "Direct parity artifact was stale: $path ($($artifact.LastWriteTimeUtc.ToString('o')) < $($PhaseStartedUtc.ToString('o')))"
        }
    }
}

function Invoke-VerificationSelfTests {
    $setName = "AQUILLM_VERIFY_ENV_SET_$PID"
    $unsetName = "AQUILLM_VERIFY_ENV_UNSET_$PID"
    $outerSnapshot = Get-ProcessEnvironmentSnapshot -Names @($setName, $unsetName)
    try {
        [Environment]::SetEnvironmentVariable($setName, "before", "Process")
        [Environment]::SetEnvironmentVariable($unsetName, $null, "Process")
        $deliberateFailureObserved = $false
        try {
            Invoke-WithTemporaryEnvironment `
                -Environment @{ $setName = "during"; $unsetName = "during" } `
                -ScriptBlock {
                    if ([Environment]::GetEnvironmentVariable($setName, "Process") -ne "during") {
                        throw "Temporary set environment value was not applied"
                    }
                    if ([Environment]::GetEnvironmentVariable($unsetName, "Process") -ne "during") {
                        throw "Temporary unset environment value was not applied"
                    }
                    throw "deliberate environment restoration self-test failure"
                }
        }
        catch {
            if ($_.Exception.Message -ne "deliberate environment restoration self-test failure") {
                throw
            }
            $deliberateFailureObserved = $true
        }
        if (-not $deliberateFailureObserved) {
            throw "Environment restoration self-test did not observe its deliberate failure"
        }
        if ([Environment]::GetEnvironmentVariable($setName, "Process") -ne "before") {
            throw "Previously set environment value was not restored"
        }
        if ($null -ne [Environment]::GetEnvironmentVariable($unsetName, "Process")) {
            throw "Previously unset environment value was not removed"
        }
    }
    finally {
        Restore-ProcessEnvironment -Snapshot $outerSnapshot
    }

    $selfTestRoot = Join-Path $StateRoot "self-test-$PID"
    New-Item -ItemType Directory -Force $selfTestRoot | Out-Null
    try {
        $phaseStartedUtc = [DateTime]::UtcNow
        foreach ($name in $DirectParityArtifactNames) {
            $path = Join-Path $selfTestRoot $name
            "stale" | Set-Content -LiteralPath $path -Encoding ascii
            (Get-Item -LiteralPath $path).LastWriteTimeUtc = $phaseStartedUtc.AddMinutes(-5)
        }
        $staleFailureObserved = $false
        try {
            Assert-FreshDirectParityArtifacts `
                -Root $selfTestRoot `
                -PhaseStartedUtc $phaseStartedUtc
        }
        catch {
            if ($_.Exception.Message -notmatch "Direct parity artifact was stale") {
                throw
            }
            $staleFailureObserved = $true
        }
        if (-not $staleFailureObserved) {
            throw "Artifact freshness self-test accepted stale files"
        }
        foreach ($name in $DirectParityArtifactNames) {
            "fresh" | Set-Content `
                -LiteralPath (Join-Path $selfTestRoot $name) `
                -Encoding ascii
        }
        Assert-FreshDirectParityArtifacts `
            -Root $selfTestRoot `
            -PhaseStartedUtc $phaseStartedUtc
    }
    finally {
        Remove-DirectParityArtifacts -Root $selfTestRoot
        Remove-Item -LiteralPath $selfTestRoot -Force -ErrorAction SilentlyContinue
    }
    Assert-RuntimeArguments -Runtime Whisper -Inspection ([pscustomobject]@{
        runtime_proc_1_args = @(
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "openai/whisper-large-v3-turbo",
            "--max-num-batched-tokens", "1500",
            "--limit-mm-per-prompt", '{"audio":{"count":1,"length":30}}'
        )
    })
    Assert-RuntimeArguments -Runtime Nemotron -Inspection ([pscustomobject]@{
        runtime_proc_1_args = @(
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", $ModelId, "--generation-config",
            "/opt/aquillm/nemotron-generation-config"
        )
    })
    if ((Get-ProfileFailureClassification -Reason "CUDA out of memory") -ne "capacity") {
        throw "Profile capacity classification self-test failed"
    }
    $oomInspection = [pscustomobject]@{
        State = [pscustomobject]@{ OOMKilled = $true }
    }
    if ((Get-ProfileFailureClassification `
        -Reason "error response from daemon" `
        -Inspection $oomInspection) -ne "capacity") {
        throw "Profile OOM container-state precedence self-test failed"
    }
    if ((Get-ProfileFailureClassification -Reason "health timeout after 900 seconds") -ne "timeout") {
        throw "Profile timeout classification self-test failed"
    }
    if ((Get-ProfileFailureClassification -Reason "context deadline exceeded") -ne "timeout") {
        throw "Profile deadline classification self-test failed"
    }
    if ((Get-ProfileFailureClassification -Reason "unrecognized arguments: --bad") -ne "configuration/infrastructure") {
        throw "Profile configuration/infrastructure classification self-test failed"
    }
    $fakeImageId = "sha256:" + ("a" * 64)
    $provenance = New-ProfileImageProvenance `
        -GenericImageId $fakeImageId `
        -TranscribeImageId $fakeImageId
    if ($provenance.generic_image_id -ne $fakeImageId -or
        $provenance.transcribe_image_id -ne $fakeImageId) {
        throw "Profile image provenance self-test failed"
    }
    Write-Host "Environment restoration, artifact freshness, runtime argument, profile failure classification, and profile image provenance self-tests passed."
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
INGEST_TRANSCRIBE_MODEL=$ServedName
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
TRANSCRIBE_VLLM_EXTRA_ARGS=--max-num-seqs 1 --max-num-batched-tokens 1500 --limit-mm-per-prompt '{"audio":{"count":1,"length":30}}'
INGEST_TRANSCRIBE_MODEL=whisper-large-v3-turbo
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
      INGEST_TRANSCRIBE_MODEL: `${INGEST_TRANSCRIBE_MODEL:?}
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
PROFILE_MAIN_MODEL=hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16
PROFILE_MAIN_SERVED_NAME=qwen3.6:27b-mtp-awq
PROFILE_MAIN_GPU_MEMORY_UTILIZATION=0.45
PROFILE_MAIN_MAX_MODEL_LEN=40960
PROFILE_MAIN_EXTRA_ARGS=--kv-cache-dtype turboquant_4bit_nc --dtype float16 --download-dir /root/.cache/huggingface/hub --attention-backend TURBOQUANT --chat-template /templates/qwen_fixed_chat_template.jinja --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-seqs 1 --no-enable-prefix-caching --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
PROFILE_EMBED_MODEL=Qwen/Qwen3-VL-Embedding-2B
PROFILE_EMBED_GPU_MEMORY_UTILIZATION=0.12
PROFILE_EMBED_MAX_MODEL_LEN=2048
PROFILE_EMBED_EXTRA_ARGS=--quantization bitsandbytes --load-format bitsandbytes --dtype float16 --model-loader-extra-config '{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16","bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}'
PROFILE_RERANK_MODEL=Qwen/Qwen3-VL-Reranker-2B
PROFILE_RERANK_GPU_MEMORY_UTILIZATION=0.08
PROFILE_RERANK_MAX_MODEL_LEN=1024
PROFILE_RERANK_EXTRA_ARGS=--runner pooling --dtype float16 --chat-template /templates/qwen3_vl_reranker.jinja --hf-overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
"@ | Set-Utf8NoBom -Path $ProfileEnvPath
    @"
name: $ProfileProject
services:
  vllm:
    image: $ProfileImage
    environment:
      VLLM_HOST: 0.0.0.0
      VLLM_PORT: "8000"
      VLLM_MODEL: `${PROFILE_MAIN_MODEL:?}
      VLLM_SERVED_MODEL_NAME: `${PROFILE_MAIN_SERVED_NAME:?}
      VLLM_TENSOR_PARALLEL_SIZE: "1"
      VLLM_GPU_MEMORY_UTILIZATION: `${PROFILE_MAIN_GPU_MEMORY_UTILIZATION:?}
      VLLM_MAX_MODEL_LEN: `${PROFILE_MAIN_MAX_MODEL_LEN:?}
      VLLM_TRUST_REMOTE_CODE: "1"
      VLLM_EXTRA_ARGS: `${PROFILE_MAIN_EXTRA_ARGS:?}
      HF_HOME: /root/.cache/huggingface
    volumes: ["profile_hf_cache:/root/.cache/huggingface"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
    healthcheck: &healthcheck
      test: ["CMD-SHELL", "python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2).read()\""]
      interval: 10s
      timeout: 10s
      retries: 90
      start_period: 60s
  vllm_transcribe:
    image: $Image
    environment:
      VLLM_USE_V2_MODEL_RUNNER: "0"
      VLLM_SERVICE_KIND: transcribe
      VLLM_MODEL: $ModelId
      VLLM_REVISION: $Revision
      VLLM_SERVED_MODEL_NAME: $ServedName
      VLLM_TOKENIZER: $ModelId
      VLLM_TENSOR_PARALLEL_SIZE: "1"
      VLLM_GPU_MEMORY_UTILIZATION: "0.20"
      VLLM_MAX_MODEL_LEN: "50000"
      VLLM_DTYPE: float32
      VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
      VLLM_TRUST_REMOTE_CODE: "1"
      VLLM_EXTRA_ARGS: --enforce-eager --max-num-seqs 1 --max-num-batched-tokens 50000 --generation-config /opt/aquillm/nemotron-generation-config
      HF_HOME: /root/.cache/huggingface
    volumes: ["nemotron_hf_cache:/root/.cache/huggingface"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
    healthcheck: *healthcheck
  vllm_embed:
    image: $ProfileImage
    environment:
      VLLM_MODEL: `${PROFILE_EMBED_MODEL:?}
      VLLM_SERVED_MODEL_NAME: `${PROFILE_EMBED_MODEL:?}
      VLLM_RUNNER: pooling
      VLLM_TENSOR_PARALLEL_SIZE: "1"
      VLLM_GPU_MEMORY_UTILIZATION: `${PROFILE_EMBED_GPU_MEMORY_UTILIZATION:?}
      VLLM_MAX_MODEL_LEN: `${PROFILE_EMBED_MAX_MODEL_LEN:?}
      VLLM_TRUST_REMOTE_CODE: "1"
      VLLM_EXTRA_ARGS: `${PROFILE_EMBED_EXTRA_ARGS:?}
      HF_HOME: /root/.cache/huggingface
    volumes: ["profile_hf_cache:/root/.cache/huggingface"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
    healthcheck: *healthcheck
  vllm_rerank:
    image: $ProfileImage
    environment:
      VLLM_MODEL: `${PROFILE_RERANK_MODEL:?}
      VLLM_SERVED_MODEL_NAME: `${PROFILE_RERANK_MODEL:?}
      VLLM_TASK: score
      VLLM_TENSOR_PARALLEL_SIZE: "1"
      VLLM_GPU_MEMORY_UTILIZATION: `${PROFILE_RERANK_GPU_MEMORY_UTILIZATION:?}
      VLLM_MAX_MODEL_LEN: `${PROFILE_RERANK_MAX_MODEL_LEN:?}
      VLLM_TRUST_REMOTE_CODE: "1"
      VLLM_EXTRA_ARGS: `${PROFILE_RERANK_EXTRA_ARGS:?}
      HF_HOME: /root/.cache/huggingface
    volumes: ["profile_hf_cache:/root/.cache/huggingface"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
    healthcheck: *healthcheck
volumes:
  nemotron_hf_cache:
    external: true
    name: $CacheVolume
  profile_hf_cache:
    external: true
    name: $ProfileCacheVolume
"@ | Set-Utf8NoBom -Path $ProfileComposePath
    @"
`$env:NEMOTRON_ASR_ENV = '$NemotronEnvPath'
`$env:WHISPER_ASR_ENV = '$WhisperEnvPath'
`$env:NEMOTRON_ASR_OVERRIDE = '$ComposePath'
`$env:NEMOTRON_ASR_RUNTIME_COMPOSE = '$ComposePath'
`$env:NEMOTRON_ASR_PROFILE_ENV = '$ProfileEnvPath'
`$env:NEMOTRON_ASR_PROFILE_COMPOSE = '$ProfileComposePath'
"@ | Set-Utf8NoBom -Path $ActivationPath

    $env:NEMOTRON_ASR_ENV = $NemotronEnvPath
    $env:WHISPER_ASR_ENV = $WhisperEnvPath
    $env:NEMOTRON_ASR_OVERRIDE = $ComposePath
    $env:NEMOTRON_ASR_RUNTIME_COMPOSE = $ComposePath
    $env:NEMOTRON_ASR_PROFILE_ENV = $ProfileEnvPath
    $env:NEMOTRON_ASR_PROFILE_COMPOSE = $ProfileComposePath
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

function Assert-RenderedWhisperConfig {
    $renderedPath = Join-Path $ArtifactRoot "whisper-compose-config.json"
    $rendered = @(& docker compose --project-name $Project --env-file $WhisperEnvPath -f $ComposePath config --format json)
    if ($LASTEXITCODE -ne 0) { throw "docker compose Whisper config failed" }
    $renderedText = $rendered -join [Environment]::NewLine
    $renderedText | Set-Utf8NoBom -Path $renderedPath
    $service = ($renderedText | ConvertFrom-Json).services.vllm_transcribe
    $expected = @{
        VLLM_MODEL = "openai/whisper-large-v3-turbo"
        VLLM_REVISION = ""
        VLLM_SERVED_MODEL_NAME = "whisper-large-v3-turbo"
        VLLM_TOKENIZER = "openai/whisper-large-v3-turbo"
        VLLM_TENSOR_PARALLEL_SIZE = "1"
        VLLM_GPU_MEMORY_UTILIZATION = "0.08"
        VLLM_MAX_MODEL_LEN = "448"
        VLLM_DTYPE = "float16"
        VLLM_ALLOW_LONG_MAX_MODEL_LEN = "0"
        VLLM_TRUST_REMOTE_CODE = "1"
        VLLM_USE_V2_MODEL_RUNNER = "0"
        VLLM_SERVICE_KIND = "transcribe"
        INGEST_TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
    }
    foreach ($entry in $expected.GetEnumerator()) {
        if ([string]$service.environment.($entry.Key) -ne $entry.Value) {
            throw "Rendered Whisper $($entry.Key) did not match: $($service.environment.($entry.Key))"
        }
    }
    $extra = [string]$service.environment.VLLM_EXTRA_ARGS
    foreach ($required in @(
        "--max-num-seqs 1",
        "--max-num-batched-tokens 1500",
        '"audio":{"count":1,"length":30}'
    )) {
        if (-not $extra.Contains($required)) {
            throw "Rendered Whisper arguments did not contain: $required"
        }
    }
    if ($extra -match "generation-config|nemotron") {
        throw "Rendered Whisper arguments contain Nemotron-only flags"
    }
    if ($service.image -ne $Image) { throw "Rendered Whisper image was $($service.image)" }
    Write-Host "Rendered standalone Compose config passed exact Whisper rollback assertions."
}

function Get-LocalImageId {
    param([Parameter(Mandatory)] [string]$Name)
    $id = (& docker image inspect $Name --format "{{.Id}}") -join ""
    if ($LASTEXITCODE -ne 0 -or -not $id) { throw "Could not inspect image: $Name" }
    return $id.Trim()
}

function New-ProfileImageProvenance {
    param(
        [Parameter(Mandatory)] [string]$GenericImageId,
        [Parameter(Mandatory)] [string]$TranscribeImageId
    )
    foreach ($id in @($GenericImageId, $TranscribeImageId)) {
        if ($id -notmatch '^sha256:[0-9a-f]{64}$') {
            throw "Profile image provenance contains an invalid local image ID: $id"
        }
    }
    return [pscustomobject]@{
        captured_at = ([DateTimeOffset](Get-Date)).ToString("o")
        generic_image = $ProfileImage
        generic_image_id = $GenericImageId
        transcribe_image = $Image
        transcribe_image_id = $TranscribeImageId
    }
}

function Assert-ExpectedModelAlias {
    param(
        [Parameter(Mandatory)] [string]$ModelsJson,
        [Parameter(Mandatory)] [string]$ExpectedAlias,
        [Parameter(Mandatory)] [string]$Service
    )
    try {
        $document = $ModelsJson | ConvertFrom-Json
        $ids = @($document.data | ForEach-Object { [string]$_.id })
    }
    catch {
        throw "$Service returned invalid /v1/models JSON: $($_.Exception.Message)"
    }
    if ($ids.Count -ne 1 -or $ids[0] -ne $ExpectedAlias) {
        throw "$Service /v1/models mismatch: expected only '$ExpectedAlias', received '$($ids -join ', ')'"
    }
    return $ids
}

function Get-ProfileFailureClassification {
    param(
        [Parameter(Mandatory)] [string]$Reason,
        [string]$Logs = "",
        $Inspection = $null
    )
    $evidence = ($Reason + [Environment]::NewLine + $Logs).ToLowerInvariant()
    $oomKilled = $false
    if ($null -ne $Inspection -and $null -ne $Inspection.State) {
        $oomKilled = [bool]$Inspection.State.OOMKilled
    }
    if ($oomKilled) { return "capacity" }
    $configurationPatterns = @(
        "unrecognized arguments", "validationerror", "invalid argument",
        "no such file", "permission denied", "manifest unknown",
        "pull access denied", "failed to solve", "error response from daemon",
        "could not select device driver", "nvidia-container-cli",
        "failed to create task"
    )
    foreach ($pattern in $configurationPatterns) {
        if ($evidence.Contains($pattern)) { return "configuration/infrastructure" }
    }
    $capacityPatterns = @(
        "cuda out of memory", "out of memory", "insufficient memory",
        "cannot allocate memory", "not enough memory", "no space left on device"
    )
    foreach ($pattern in $capacityPatterns) {
        if ($evidence.Contains($pattern)) { return "capacity" }
    }
    if ($evidence -match 'timed?\s*out|timeout|deadline exceeded') { return "timeout" }
    return "configuration/infrastructure"
}

function Get-ComposeServiceInspection {
    param(
        [Parameter(Mandatory)] [string]$ProjectName,
        [Parameter(Mandatory)] [string]$EnvironmentPath,
        [Parameter(Mandatory)] [string]$ComposeFile,
        [Parameter(Mandatory)] [string]$Service,
        [Parameter(Mandatory)] [string]$ArtifactName
    )
    $containerId = (@(& docker compose --project-name $ProjectName --env-file $EnvironmentPath -f $ComposeFile ps -q $Service) -join "").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $containerId) { throw "No container ID for $Service" }
    $inspection = (& docker inspect $containerId | ConvertFrom-Json)[0]
    $runtimeArguments = @(& docker exec $containerId python3 -c "from pathlib import Path; print(Path('/proc/1/cmdline').read_bytes().replace(bytes([0]), bytes([10])).decode(), end='')")
    if ($LASTEXITCODE -ne 0 -or -not $runtimeArguments.Count) {
        throw "Could not read live /proc/1/cmdline for $Service"
    }
    $evidence = [pscustomobject]@{
        container_id = $containerId
        image = $inspection.Image
        configured_path = $inspection.Path
        configured_args = @($inspection.Args)
        runtime_proc_1_args = @($runtimeArguments)
        inspected_at = ([DateTimeOffset](Get-Date)).ToString("o")
    }
    $evidence | ConvertTo-Json -Depth 12 | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot $ArtifactName)
    return $evidence
}

function Assert-RuntimeArguments {
    param(
        [Parameter(Mandatory)] $Inspection,
        [Parameter(Mandatory)] [ValidateSet("Whisper", "Nemotron")] [string]$Runtime
    )
    $arguments = @($Inspection.runtime_proc_1_args)
    $argumentText = $arguments -join " "
    if ($Runtime -eq "Whisper") {
        foreach ($forbidden in @("--revision", "--generation-config", "/opt/aquillm/nemotron-generation-config")) {
            if ($arguments -contains $forbidden -or $argumentText.Contains($forbidden)) {
                throw "Whisper runtime arguments contain forbidden value: $forbidden"
            }
        }
        foreach ($required in @("--max-num-batched-tokens", "1500", "--limit-mm-per-prompt")) {
            if (-not ($arguments -contains $required)) { throw "Whisper runtime arguments omit: $required" }
        }
        foreach ($forbidden in @("--quantization", "bitsandbytes", "--load-format", "--model-loader-extra-config")) {
            if ($arguments -contains $forbidden -or $argumentText.Contains($forbidden)) {
                throw "Whisper runtime arguments contain unsupported quantization value: $forbidden"
            }
        }
    } else {
        if (-not ($arguments -contains "--generation-config") -or
            -not ($arguments -contains "/opt/aquillm/nemotron-generation-config")) {
            throw "Nemotron runtime arguments omit the pinned generation config"
        }
        foreach ($forbidden in @("bitsandbytes", "--quantization", "--load-format")) {
            if ($arguments -contains $forbidden -or $argumentText.Contains($forbidden)) {
                throw "Nemotron runtime arguments contain Whisper rollback value: $forbidden"
            }
        }
    }
}

function Invoke-AsrSdkSmoke {
    param(
        [Parameter(Mandatory)] [string]$Model,
        [Parameter(Mandatory)] [string]$ArtifactName,
        [string]$Prompt = ""
    )
    $hostPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $hostPython)) { throw "Host test Python not found: $hostPython" }
    $scriptPath = Join-Path $StateRoot "asr-sdk-smoke.py"
    @'
import json
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8005/v1", api_key="EMPTY", timeout=180.0)
models = [item.id for item in client.models.list().data]
expected = os.environ["ASR_SMOKE_MODEL"]
assert models == [expected], models
kwargs = {"model": expected}
prompt = os.environ.get("ASR_SMOKE_PROMPT", "")
if prompt:
    kwargs["prompt"] = prompt
fixture = Path(os.environ["ASR_SMOKE_FIXTURE"])
with fixture.open("rb") as audio:
    result = client.audio.transcriptions.create(file=audio, **kwargs)
assert isinstance(result.text, str) and result.text.strip(), result
Path(os.environ["ASR_SMOKE_ARTIFACT"]).write_text(json.dumps({
    "model": expected,
    "models": models,
    "prompt": prompt,
    "text": result.text,
}, indent=2, sort_keys=True), encoding="utf-8")
print(result.text)
'@ | Set-Utf8NoBom -Path $scriptPath
    Invoke-WithTemporaryEnvironment -Environment @{
        ASR_SMOKE_MODEL = $Model
        ASR_SMOKE_PROMPT = $Prompt
        ASR_SMOKE_FIXTURE = (Join-Path $RepoRoot "tests\fixtures\audio\librispeech_1272-128104-0000.flac")
        ASR_SMOKE_ARTIFACT = (Join-Path $ArtifactRoot $ArtifactName)
    } -ScriptBlock {
        Invoke-Checked -FilePath $hostPython -ArgumentList @($scriptPath)
    }
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

function Assert-RenderedProfileConfig {
    $rendered = @(& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath config --format json)
    if ($LASTEXITCODE -ne 0) { throw "Failed to render standalone profile Compose" }
    $renderedText = $rendered -join [Environment]::NewLine
    $renderedText | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "profile-compose-config.json")
    $services = ($renderedText | ConvertFrom-Json).services
    $expected = @{
        vllm = @{ image = $ProfileImage; fraction = "0.45"; model = "hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16" }
        vllm_transcribe = @{ image = $Image; fraction = "0.20"; model = $ModelId }
        vllm_embed = @{ image = $ProfileImage; fraction = "0.12"; model = "Qwen/Qwen3-VL-Embedding-2B" }
        vllm_rerank = @{ image = $ProfileImage; fraction = "0.08"; model = "Qwen/Qwen3-VL-Reranker-2B" }
    }
    foreach ($entry in $expected.GetEnumerator()) {
        $service = $services.($entry.Key)
        if ($null -eq $service) { throw "Profile omitted required service: $($entry.Key)" }
        if ($service.image -ne $entry.Value.image) { throw "$($entry.Key) rendered unexpected image: $($service.image)" }
        if ([string]$service.environment.VLLM_GPU_MEMORY_UTILIZATION -ne $entry.Value.fraction) {
            throw "$($entry.Key) rendered unexpected GPU fraction"
        }
        if ($service.environment.VLLM_MODEL -ne $entry.Value.model) {
            throw "$($entry.Key) rendered unexpected model: $($service.environment.VLLM_MODEL)"
        }
    }
    if ($services.PSObject.Properties.Name -contains "vllm_ocr") {
        throw "Standalone required profile unexpectedly contains optional OCR"
    }
    Write-Host "Rendered required profile passed exact service/order budget assertions; OCR is excluded."
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
    param(
        [string]$Command,
        [hashtable]$Environment = @{},
        [string[]]$Volumes = @(),
        [string]$ExpectedOutput = ""
    )
    $arguments = @("run", "--rm", "--gpus", "all", "--entrypoint", "bash")
    foreach ($entry in $Environment.GetEnumerator()) {
        $arguments += @("-e", "$($entry.Key)=$($entry.Value)")
    }
    foreach ($volume in $Volumes) { $arguments += @("-v", $volume) }
    $arguments += @("-v", "${RepoRoot}:/workspace", "-w", "/workspace", $Image, "-lc", $Command)
    if (-not $ExpectedOutput) {
        Invoke-Checked -FilePath docker -ArgumentList $arguments
        return
    }

    # Pytest's pass summary is stdout. Leave stderr attached to the host so
    # native warnings do not become terminating PowerShell ErrorRecords under
    # the script-wide ErrorActionPreference = Stop.
    $output = @(& docker @arguments)
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) {
        throw "Container pytest failed with exit code $exitCode"
    }
    $outputText = $output -join [Environment]::NewLine
    if (-not $outputText.Contains($ExpectedOutput)) {
        throw "Container pytest did not report expected output: $ExpectedOutput"
    }
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
    param(
        [string]$Path,
        [int]$Baseline,
        [DateTimeOffset]$BaselineAt,
        [DateTimeOffset]$RequestCompletedAt,
        [DateTimeOffset]$PostWindowEndedAt
    )
    $samples = @(
        Import-Csv $Path | ForEach-Object {
            [pscustomobject]@{
                timestamp = [DateTimeOffset]::Parse($_.timestamp)
                memory_used_mib = [int]$_.memory_used_mib
            }
        }
    )
    if (-not $samples.Count) { throw "No GPU memory samples were recorded" }
    $postWindowDuration = ($PostWindowEndedAt - $RequestCompletedAt).TotalSeconds
    if ($postWindowDuration -lt 30) {
        throw "Post-request GPU memory window was shorter than 30 seconds"
    }
    $steadyWindow = @(
        $samples | Where-Object {
            $_.timestamp -ge $RequestCompletedAt -and
            $_.timestamp -le $PostWindowEndedAt
        }
    )
    if ($steadyWindow.Count -lt 6) {
        throw "Post-request GPU memory window had only $($steadyWindow.Count) samples; require at least 6"
    }
    return [pscustomobject]@{
        baseline_mib = $Baseline
        baseline_at = $BaselineAt.ToString("o")
        peak_mib = ($samples.memory_used_mib | Measure-Object -Maximum).Maximum
        steady_mib = [Math]::Round(
            ($steadyWindow.memory_used_mib | Measure-Object -Average).Average,
            1
        )
        sample_count = $samples.Count
        request_completed_at = $RequestCompletedAt.ToString("o")
        post_window_ended_at = $PostWindowEndedAt.ToString("o")
        post_window_duration_seconds = [Math]::Round($postWindowDuration, 3)
        post_window_sample_count = $steadyWindow.Count
        post_window_first_sample_at = $steadyWindow[0].timestamp.ToString("o")
        post_window_last_sample_at = $steadyWindow[-1].timestamp.ToString("o")
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
    Invoke-VerificationSelfTests
    Write-IsolatedEnvironmentFiles
    Assert-RenderedNemotronConfig
    Assert-RepositoryComposeProfiles
    & docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath down --remove-orphans
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid

    Invoke-Checked -FilePath docker -ArgumentList @(
        "build", "-f", (Join-Path $RepoRoot "deploy\docker\vllm\Dockerfile.transcribe"),
        "-t", $Image, $RepoRoot
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

    Remove-DirectParityArtifacts -Root $ArtifactRoot
    $directPhaseStartedUtc = [DateTime]::UtcNow
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
        -ExpectedOutput "2 passed" `
        -Command "python3 -m pip install -q pytest==8.4.1 && python3 -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_transformers_parity.py -q -m gpu"
    Assert-FreshDirectParityArtifacts `
        -Root $ArtifactRoot `
        -PhaseStartedUtc $directPhaseStartedUtc
    [pscustomobject]@{
        phase_started_utc = $directPhaseStartedUtc.ToString("o")
        validated_utc = ([DateTime]::UtcNow).ToString("o")
        expected_pytest_output = "2 passed"
        artifacts = @(
            $DirectParityArtifactNames | ForEach-Object {
                $artifact = Get-Item -LiteralPath (Join-Path $ArtifactRoot $_)
                [pscustomobject]@{
                    name = $_
                    bytes = $artifact.Length
                    last_write_utc = $artifact.LastWriteTimeUtc.ToString("o")
                }
            }
        )
    } | ConvertTo-Json -Depth 4 | Set-Utf8NoBom -Path (
        Join-Path $ArtifactRoot $DirectParitySummaryName
    )
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

    $baselineAt = [DateTimeOffset](Get-Date)
    $baseline = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim())
    $sampler = Start-GpuMemorySampler -OutputPath $MemoryCsv
    $requestCompletedAt = $null
    $postWindowEndedAt = $null
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
        $runtimeEnvironment = @{
            RUN_ASR_RUNTIME = "1"
            ASR_BASE_URL = "http://127.0.0.1:8005/v1"
            SECRET_KEY = "nemotron-runtime-verification-only"
            GOOGLE_OAUTH2_CLIENT_ID = "nemotron-runtime-verification"
            GOOGLE_OAUTH2_CLIENT_SECRET = "nemotron-runtime-verification"
            OPENAI_API_KEY = "sk-nemotron-runtime-verification"
            GEMINI_API_KEY = "nemotron-runtime-verification"
        }
        $requestCompletedAt = Invoke-WithTemporaryEnvironment `
            -Environment $runtimeEnvironment `
            -ScriptBlock {
                Push-Location $RepoRoot
                $runtimeStarted = Get-Date
                try {
                    $runtimeOutput = @(& $hostPython -m pytest tests/asr -q 2>&1)
                    $runtimeExitCode = $LASTEXITCODE
                    $completedAt = [DateTimeOffset](Get-Date)
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
                    return $completedAt
                }
                finally {
                    Pop-Location
                }
            }
        Start-Sleep -Seconds 35
        $postWindowEndedAt = [DateTimeOffset](Get-Date)
    } finally {
        Stop-Job $sampler -ErrorAction SilentlyContinue
        Receive-Job $sampler -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $sampler -Force -ErrorAction SilentlyContinue
        (& docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath logs --no-color vllm_transcribe) -join [Environment]::NewLine | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "nemotron-service.log")
        & docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath down --remove-orphans
    }
    $memory = Get-MemorySummary `
        -Path $MemoryCsv `
        -Baseline $baseline `
        -BaselineAt $baselineAt `
        -RequestCompletedAt $requestCompletedAt `
        -PostWindowEndedAt $postWindowEndedAt
    $memory | ConvertTo-Json | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "gpu-memory-summary.json")
    $memory | Format-List | Out-String | Write-Host
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid
    Write-Host "Nemotron verification artifacts: $ArtifactRoot"
}

function Invoke-WhisperRollbackVerification {
    Invoke-VerificationSelfTests
    Write-IsolatedEnvironmentFiles
    Assert-RenderedWhisperConfig
    Assert-RenderedNemotronConfig
    New-Item -ItemType Directory -Force $ArtifactRoot | Out-Null
    Invoke-Checked -FilePath docker -ArgumentList @("volume", "create", $CacheVolume) | Out-Null
    & docker compose --project-name $Project --env-file $WhisperEnvPath -f $ComposePath down --remove-orphans
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid

    $imageIdBefore = Get-LocalImageId -Name $Image
    $whisperInspection = $null
    $nemotronInspection = $null
    $whisperStartedAt = $null
    $nemotronStartedAt = $null
    try {
        $whisperStartedAt = [DateTimeOffset](Get-Date)
        Invoke-Checked -FilePath docker -ArgumentList @(
            "compose", "--project-name", $Project, "--env-file", $WhisperEnvPath,
            "-f", $ComposePath, "up", "-d", "--no-deps", "--wait", "--wait-timeout", "900",
            "vllm_transcribe"
        )
        $whisperInspection = Get-ComposeServiceInspection `
            -ProjectName $Project -EnvironmentPath $WhisperEnvPath `
            -ComposeFile $ComposePath -Service "vllm_transcribe" `
            -ArtifactName "whisper-live-container.json"
        if ($whisperInspection.image -ne $imageIdBefore) {
            throw "Whisper container did not reuse exact image ID: $($whisperInspection.image)"
        }
        Assert-RuntimeArguments -Inspection $whisperInspection -Runtime Whisper
        $whisperModels = Invoke-RestMethod -Uri "http://127.0.0.1:8005/v1/models" -Headers @{ Authorization = "Bearer EMPTY" }
        $whisperIds = @($whisperModels.data | ForEach-Object { $_.id })
        if ($whisperIds.Count -ne 1 -or $whisperIds[0] -ne "whisper-large-v3-turbo") {
            throw "Unexpected Whisper served model IDs: $($whisperIds -join ', ')"
        }
        Invoke-AsrSdkSmoke `
            -Model "whisper-large-v3-turbo" `
            -Prompt "MISTER QUILTER" `
            -ArtifactName "whisper-sdk-smoke.json"
        $packageOutput = @(& docker run --rm --entrypoint python3 $Image -c "import importlib.metadata as m; import aquillm_vllm_nemotron_asr as p; print('image_plugin=' + p.__file__); print('vllm=' + m.version('vllm')); print('transformers=' + m.version('transformers')); print('torch=' + m.version('torch')); print('bitsandbytes=' + m.version('bitsandbytes')); print('accelerate=' + m.version('accelerate'))")
        if ($LASTEXITCODE -ne 0) { throw "Same-image package evidence probe failed" }
        $packageOutput -join [Environment]::NewLine | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "rollback-package-versions.txt")
        $pipCheckOutput = @(& docker run --rm --entrypoint python3 $Image -m pip check)
        if ($LASTEXITCODE -ne 0) { throw "Same-image pip check failed" }
        (($pipCheckOutput -join [Environment]::NewLine) + [Environment]::NewLine) | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "rollback-pip-check.txt")
    }
    finally {
        (& docker compose --project-name $Project --env-file $WhisperEnvPath -f $ComposePath logs --no-color vllm_transcribe) -join [Environment]::NewLine | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "whisper-service.log")
        & docker compose --project-name $Project --env-file $WhisperEnvPath -f $ComposePath down --remove-orphans
    }
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid

    try {
        $nemotronStartedAt = [DateTimeOffset](Get-Date)
        Invoke-Checked -FilePath docker -ArgumentList @(
            "compose", "--project-name", $Project, "--env-file", $NemotronEnvPath,
            "-f", $ComposePath, "up", "-d", "--no-deps", "--wait", "--wait-timeout", "900",
            "vllm_transcribe"
        )
        $nemotronInspection = Get-ComposeServiceInspection `
            -ProjectName $Project -EnvironmentPath $NemotronEnvPath `
            -ComposeFile $ComposePath -Service "vllm_transcribe" `
            -ArtifactName "nemotron-restored-live-container.json"
        if ($nemotronInspection.image -ne $imageIdBefore) {
            throw "Restored Nemotron container did not reuse exact image ID: $($nemotronInspection.image)"
        }
        Assert-RuntimeArguments -Inspection $nemotronInspection -Runtime Nemotron
        $nemotronModels = Invoke-RestMethod -Uri "http://127.0.0.1:8005/v1/models" -Headers @{ Authorization = "Bearer EMPTY" }
        $nemotronIds = @($nemotronModels.data | ForEach-Object { $_.id })
        if ($nemotronIds.Count -ne 1 -or $nemotronIds[0] -ne $ServedName) {
            throw "Unexpected restored Nemotron served model IDs: $($nemotronIds -join ', ')"
        }
        Invoke-AsrSdkSmoke -Model $ServedName -ArtifactName "nemotron-restored-sdk-smoke.json"
    }
    finally {
        (& docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath logs --no-color vllm_transcribe) -join [Environment]::NewLine | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "nemotron-restored-service.log")
        & docker compose --project-name $Project --env-file $NemotronEnvPath -f $ComposePath down --remove-orphans
    }
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid
    $imageIdAfter = Get-LocalImageId -Name $Image
    if ($imageIdAfter -ne $imageIdBefore) { throw "Transcription image changed during environment-only rollback" }
    [pscustomobject]@{
        verified_at = ([DateTimeOffset](Get-Date)).ToString("o")
        image = $Image
        image_id_before = $imageIdBefore
        image_id_after = $imageIdAfter
        image_rebuilt = $false
        whisper_model = "whisper-large-v3-turbo"
        whisper_started_at = $whisperStartedAt.ToString("o")
        whisper_prompt = "MISTER QUILTER"
        nemotron_model = $ServedName
        nemotron_started_at = $nemotronStartedAt.ToString("o")
        shared_cache_volume = $CacheVolume
    } | ConvertTo-Json -Depth 4 | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "whisper-rollback-summary.json")
    Write-Host "Same-image Whisper rollback and explicit Nemotron restoration passed."
}

function Invoke-ProfileVerification {
    Write-IsolatedEnvironmentFiles
    Assert-RenderedProfileConfig
    Invoke-Checked -FilePath docker -ArgumentList @("volume", "create", $CacheVolume) | Out-Null
    Invoke-Checked -FilePath docker -ArgumentList @("volume", "create", $ProfileCacheVolume) | Out-Null
    Invoke-Checked -FilePath docker -ArgumentList @(
        "build", "-f", (Join-Path $RepoRoot "deploy\docker\vllm\Dockerfile"),
        "-t", $ProfileImage, $RepoRoot
    )
    Invoke-Checked -FilePath docker -ArgumentList @(
        "build", "-f", (Join-Path $RepoRoot "deploy\docker\vllm\Dockerfile.transcribe"),
        "-t", $Image, $RepoRoot
    )
    $profileImageId = Get-LocalImageId -Name $ProfileImage
    $transcribeImageId = Get-LocalImageId -Name $Image
    $genericProbe = @(& docker run --rm --entrypoint python3 $ProfileImage -c "import importlib.metadata as m; assert m.version('vllm') == '0.21.0', m.version('vllm'); print('vllm=' + m.version('vllm')); print('transformers=' + m.version('transformers')); print('torch=' + m.version('torch'))")
    if ($LASTEXITCODE -ne 0) {
        throw "Generic profile image dependency probe failed"
    }
    ($genericProbe -join [Environment]::NewLine) | Set-Utf8NoBom -Path (
        Join-Path $ArtifactRoot "profile-generic-image-probe.txt"
    )
    Invoke-Checked -FilePath docker -ArgumentList @(
        "run", "--rm", "--gpus", "all", "--entrypoint", "python3", $Image,
        "/probe_nemotron_plugin.py", "--generation-config", "/opt/aquillm/nemotron-generation-config"
    )
    $imageProvenance = New-ProfileImageProvenance `
        -GenericImageId $profileImageId `
        -TranscribeImageId $transcribeImageId
    $imageProvenance | ConvertTo-Json -Depth 5 | Set-Utf8NoBom -Path (
        Join-Path $ArtifactRoot "profile-image-provenance.json"
    )
    & docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath down --remove-orphans
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid

    $serviceOrder = @("vllm", "vllm_transcribe", "vllm_embed", "vllm_rerank")
    $fractions = @{ vllm = "0.45"; vllm_transcribe = "0.20"; vllm_embed = "0.12"; vllm_rerank = "0.08" }
    $expectedAliases = @{
        vllm = "qwen3.6:27b-mtp-awq"
        vllm_transcribe = $ServedName
        vllm_embed = "Qwen/Qwen3-VL-Embedding-2B"
        vllm_rerank = "Qwen/Qwen3-VL-Reranker-2B"
    }
    $gpuName = ((& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1) -join "").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $gpuName) { throw "Could not identify profile GPU" }
    $records = [Collections.Generic.List[object]]::new()
    $passing = [Collections.Generic.List[string]]::new()
    $failure = $null
    try {
        foreach ($service in $serviceOrder) {
            $startedAt = [DateTimeOffset](Get-Date)
            $exitCode = 0
            try {
                $output = @(& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath `
                    -f $ProfileComposePath up -d --no-deps --wait --wait-timeout 900 $service)
                $exitCode = $LASTEXITCODE
                $outputText = $output -join [Environment]::NewLine
                $outputText | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "profile-$service-up.log")
                if ($exitCode -ne 0) { throw "docker compose up --wait failed with exit code $exitCode" }
                $healthOutput = @(& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath `
                    -f $ProfileComposePath exec -T $service python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health', timeout=5).status)")
                if ($LASTEXITCODE -ne 0 -or ($healthOutput -join "") -notmatch "200") {
                    throw "Explicit in-container health probe failed for $service"
                }
                $memoryRaw = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
                if ($LASTEXITCODE -ne 0 -or $memoryRaw -notmatch '^\d+$') {
                    throw "GPU memory probe failed for $service"
                }
                $memoryUsed = [int]$memoryRaw
                $modelsOutput = @(& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath `
                    -f $ProfileComposePath exec -T $service python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/v1/models', timeout=5).read().decode())")
                if ($LASTEXITCODE -ne 0) { throw "Model inventory probe failed for $service" }
                $modelsText = $modelsOutput -join [Environment]::NewLine
                $modelIds = Assert-ExpectedModelAlias `
                    -ModelsJson $modelsText `
                    -ExpectedAlias $expectedAliases[$service] `
                    -Service $service
                $record = [pscustomobject]@{
                    service = $service
                    fraction = $fractions[$service]
                    started_at = $startedAt.ToString("o")
                    healthy_at = ([DateTimeOffset](Get-Date)).ToString("o")
                    startup_seconds = [Math]::Round(((Get-Date) - $startedAt.DateTime).TotalSeconds, 3)
                    overall_gpu_memory_mib = $memoryUsed
                    expected_model_alias = $expectedAliases[$service]
                    model_ids = @($modelIds)
                    models_json = $modelsText
                }
                $records.Add($record)
                $passing.Add($service)
                $record | ConvertTo-Json -Depth 5 | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "profile-$service-measurement.json")
            }
            catch {
                $failureReason = $_.Exception.Message
                $failureLogs = @(& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath logs --no-color $service) -join [Environment]::NewLine
                $failureLogs | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "profile-$service-failure.log")
                $failedContainerId = (@(& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath ps -aq $service) -join "").Trim()
                $failedInspection = $null
                if ($failedContainerId) {
                    try {
                        $failedInspection = (& docker inspect $failedContainerId | ConvertFrom-Json)[0]
                        $failedInspection | ConvertTo-Json -Depth 12 | Set-Utf8NoBom -Path (
                            Join-Path $ArtifactRoot "profile-$service-failure-inspect.json"
                        )
                    }
                    catch {
                        $failureLogs += [Environment]::NewLine + "docker inspect failed: $($_.Exception.Message)"
                    }
                }
                $classification = Get-ProfileFailureClassification `
                    -Reason $failureReason `
                    -Logs $failureLogs `
                    -Inspection $failedInspection
                $failure = [pscustomobject]@{
                    service = $service
                    exit_code = $exitCode
                    classification = $classification
                    reason = $failureReason
                    started_at = $startedAt.ToString("o")
                    elapsed_seconds = [Math]::Round(((Get-Date) - $startedAt.DateTime).TotalSeconds, 3)
                }
                (& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath ps -a) -join [Environment]::NewLine | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "profile-failure-ps.txt")
                & docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath stop $service
                break
            }
        }
    }
    finally {
        (& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath ps -a) -join [Environment]::NewLine | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "profile-final-ps.txt")
        (& docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath logs --no-color) -join [Environment]::NewLine | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "profile-all-services.log")
        & docker compose --project-name $ProfileProject --env-file $ProfileEnvPath -f $ProfileComposePath down --remove-orphans
    }
    [pscustomobject]@{
        measured_at = ([DateTimeOffset](Get-Date)).ToString("o")
        gpu = $gpuName
        service_order = $serviceOrder
        fractions = $fractions
        optional_ocr_excluded = $true
        generic_image = $ProfileImage
        generic_image_id = $profileImageId
        transcribe_image = $Image
        transcribe_image_id = $transcribeImageId
        allow_incomplete_profile = [bool]$AllowIncompleteProfile
        passing_services = @($passing)
        largest_passing_prefix = @($passing)
        full_profile_passed = ($null -eq $failure -and $passing.Count -eq $serviceOrder.Count)
        failure = $failure
        measurements = @($records)
    } | ConvertTo-Json -Depth 8 | Set-Utf8NoBom -Path (Join-Path $ArtifactRoot "profile-summary.json")
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid
    if ($null -ne $failure) {
        $message = "Required profile verification failed at $($failure.service) ($($failure.classification)); largest passing prefix: $($passing -join ', ')"
        if ($AllowIncompleteProfile -and $failure.classification -in @("capacity", "timeout")) {
            Write-Warning "$message. Incomplete measurement was explicitly allowed."
        } else {
            throw "Required profile verification failed: $message"
        }
    } else {
        Write-Host "All required profile services started in order; OCR remained excluded."
    }
}

if (-not ($AssertGpuIdle -or $PrepareEnvironments -or $SelfTest -or $VerifyNemotron -or $VerifyWhisperRollback -or $VerifyProfile)) {
    throw "Select at least one switch: -AssertGpuIdle, -PrepareEnvironments, -SelfTest, -VerifyNemotron, -VerifyWhisperRollback, or -VerifyProfile"
}
if ($AllowIncompleteProfile -and -not $VerifyProfile) {
    throw "-AllowIncompleteProfile is valid only with -VerifyProfile"
}
if ($SelfTest) {
    Invoke-VerificationSelfTests
}
if ($PrepareEnvironments) {
    Write-IsolatedEnvironmentFiles
    Assert-RenderedNemotronConfig
    Assert-RenderedWhisperConfig
    Assert-RenderedProfileConfig
    Assert-RepositoryComposeProfiles
}
if ($AssertGpuIdle) {
    Assert-GpuIsIdle -AllowedPids $AllowGpuPid
}
if ($VerifyNemotron) {
    Invoke-FullNemotronVerification
}
if ($VerifyWhisperRollback) {
    Invoke-WhisperRollbackVerification
}
if ($VerifyProfile) {
    Invoke-ProfileVerification
}
