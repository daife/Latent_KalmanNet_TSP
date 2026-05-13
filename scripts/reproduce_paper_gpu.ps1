param(
    [string]$EnvPath = ".\.conda",
    [int]$Epochs = 300,
    [double[]]$LorenzNoiseValues = @(0.5, 0.1, 0.01, 0.0),
    [double[]]$PendulumNoiseValues = @(0.9, 0.4, 0.1, 0.01),
    [switch]$SkipDataGeneration,
    [switch]$SkipTraining,
    [switch]$SkipPendulum,
    [switch]$SkipLorenz,
    [switch]$SkipReferenceFigureExport,
    [switch]$AllowCpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ConfigPath = Join-Path $ProjectRoot "configurations\config_file.yaml"
$ModelLorenzPath = Join-Path $ProjectRoot "model_Lorenz.py"
$LogDir = Join-Path $ProjectRoot "logs"
$RunDevice = if ($AllowCpu) { "device_optional" } else { "gpu" }
$FigureDir = Join-Path $ProjectRoot "results\figures\$RunDevice"
$ModelRoot = "./results/models_$RunDevice"
$ResolvedEnvPath = Resolve-Path (Join-Path $ProjectRoot $EnvPath)
$PythonExe = Join-Path $ResolvedEnvPath "python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Cannot find project environment Python: $PythonExe. Create the project conda environment first."
}

New-Item -ItemType Directory -Force -Path $LogDir, $FigureDir | Out-Null
New-Item -ItemType Directory -Force -Path `
    (Join-Path $ProjectRoot "Simulations\Lorenz"), `
    (Join-Path $ProjectRoot "Simulations\Pendulum"), `
    (Join-Path $ProjectRoot "results\models_$RunDevice") | Out-Null

Write-Host "KNet checkpoints will be trained from scratch and saved under $ModelRoot."
Write-Host "The script will not load or fall back to pretrained checkpoints under ./KNetLatent_models."
Write-Host "Dataset generation steps overwrite matching files under ./Simulations."
Write-Host "Using Python executable: $PythonExe"

function Set-Utf8NoBomFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Value
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Value, $encoding)
}

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogName
    )

    $stdout = Join-Path $LogDir "$LogName.out.log"
    $stderr = Join-Path $LogDir "$LogName.err.log"
    $runner = Join-Path $LogDir "$LogName.run.cmd"
    Remove-Item -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue

    Push-Location $ProjectRoot
    try {
        function Quote-CmdArgument {
            param([Parameter(Mandatory = $true)][string]$Value)
            '"' + ($Value -replace '"', '\"') + '"'
        }

        $quotedPython = Quote-CmdArgument $PythonExe
        $quotedArgs = ($Arguments | ForEach-Object { Quote-CmdArgument $_ }) -join " "
        $quotedStdout = Quote-CmdArgument $stdout
        $quotedStderr = Quote-CmdArgument $stderr

        $runnerContent = @"
@echo off
set MPLBACKEND=Agg
$quotedPython $quotedArgs 1>$quotedStdout 2>$quotedStderr
exit /b %ERRORLEVEL%
"@
        Set-Content -LiteralPath $runner -Value $runnerContent -Encoding ASCII

        & cmd.exe /d /c "`"$runner`""
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0 -and (Test-Path $stderr)) {
            $rawWarnings = Get-Content $stderr | Where-Object {
                $_ -and
                $_ -notmatch "The PostScript backend does not support transparency" -and
                $_ -notmatch "partially transparent artists will be rendered opaque"
            }
            if ($rawWarnings) {
                Write-Host "Warnings from ${LogName}:"
                $rawWarnings | Select-Object -Last 20
            }
        }

        if ($exitCode -ne 0) {
            Write-Host "FAILED: $LogName"
            if (Test-Path $stderr) { Get-Content $stderr -Tail 80 }
            if (Test-Path $stdout) { Get-Content $stdout -Tail 80 }
            throw "Command failed: python $($Arguments -join ' ')"
        }
    }
    finally {
        Remove-Item -LiteralPath $runner -Force -ErrorAction SilentlyContinue
        Pop-Location
    }
}

function Assert-TorchDevice {
    if ($AllowCpu) {
        Write-Host "Checking project conda environment. CPU fallback is allowed by -AllowCpu."
        Invoke-ProjectPython -LogName "device_check" -Arguments @(
            "-c",
            "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('selected_mode', 'cuda' if torch.cuda.is_available() else 'cpu')"
        )
    }
    else {
        Write-Host "Checking project conda environment and CUDA availability..."
        Invoke-ProjectPython -LogName "device_check" -Arguments @(
            "-c",
            "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); assert torch.cuda.is_available(), 'PyTorch CUDA is not available in this environment. Re-run with -AllowCpu to permit CPU execution.'"
        )
    }
}

function Set-ExperimentConfig {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("Lorenz", "Pendulum")][string]$Dataset,
        [Parameter(Mandatory = $true)][string]$Scenario,
        [Parameter(Mandatory = $true)][bool]$GenerateData,
        [Parameter(Mandatory = $true)][double]$RealR2,
        [Parameter(Mandatory = $true)][string]$ModelFolder
    )

    $generate = if ($GenerateData) { "True" } else { "False" }
    $yaml = @"
########### Data #########################################
dataset_name : "$Dataset"          # "Lorenz" , "Pendulum"
sinerio : "$Scenario"             # "Baseline", "Decimation", "Test_Long_Trajectories"
data_gen_flag : $generate            # True - Generating Dataset. False - Loading Dataset
real_r2 : $RealR2                       # observation noise std
#real_q2 : 0.1                     # dynamic noise std

########### Directories ##################################
folder_KNetLatent_model : "$ModelFolder" # both models - encoder and KGain
folder_simulations : "./Simulations"
folder_encoder_model : "./Encoder"

########### EKF ##########################################
Evaluate_EKF_flag : False         # True - Evaluate EKF results. False - Load EKF results

########### Architecture #################################
load_KNetLatent_trained : False  # True - loading trained KGain model. False - KGain model start from scratch
flag_Train : True                # True - Training full pipeline. False - Only inference
fix_encoder_flag : False          # True - Encoder is fixed and KGain trainable. False - Encoder is trainable and KGain fixed
prior_flag : True               # True - Encoder with prior. False - Encoder without prior
warm_start_flag : False

############ Hyper Parameters #############################
lr_kalman : 0.001
wd_kalman : 0.01
batch_size : 16
epoches : $Epochs
"@
    Set-Utf8NoBomFile -Path $ConfigPath -Value $yaml
}

function Set-LorenzTaylorOrder {
    param([Parameter(Mandatory = $true)][int]$J)

    $content = Get-Content -LiteralPath $ModelLorenzPath -Raw
    $content = [regex]::Replace($content, "(?m)^J\s*=\s*\d+\s*$", "J=$J")
    Set-Utf8NoBomFile -Path $ModelLorenzPath -Value $content
}

function Ensure-LorenzEncoderAliases {
    $source = Join-Path $ProjectRoot "Encoder\Lorenz\Baseline_with_prior"
    $target = Join-Path $ProjectRoot "Encoder\Lorenz\Test_Long_Trajectories_with_prior"

    if (-not (Test-Path $source)) {
        throw "Missing Lorenz baseline encoder directory: $source"
    }

    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Get-ChildItem -LiteralPath $source -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $target $_.Name) -Force
    }
}

function Copy-MainVisualFigures {
    param([Parameter(Mandatory = $true)][string]$Tag)

    $mapping = @{
        "Baseline.eps" = "${Tag}_Baseline.eps"
        "Long_Trajectories.eps" = "${Tag}_Long_Trajectories.eps"
        "Approximated_state_evolution_function.eps" = "${Tag}_Approximated_state_evolution_function.eps"
        "Decimation.eps" = "${Tag}_Decimation.eps"
        "Pendulum design steps.eps" = "${Tag}_Pendulum_design_steps.eps"
        "trajectory_design_steps.eps" = "${Tag}_trajectory_design_steps.eps"
        "trajectory_design_steps_zoom.eps" = "${Tag}_trajectory_design_steps_zoom.eps"
    }

    foreach ($sourceName in $mapping.Keys) {
        $source = Join-Path $ProjectRoot $sourceName
        if (Test-Path $source) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $FigureDir $mapping[$sourceName]) -Force
            Remove-Item -LiteralPath $source -Force
        }
    }
}

function Ensure-PendulumBaselineData {
    $script = @'
from pathlib import Path
import numpy as np
from PendulumGeneration_new import Pendulum

img_size = 28
params = Pendulum.pendulum_default_params()
params[Pendulum.SIM_DT_KEY] = 5e-3
params[Pendulum.LENGTH_KEY] = 1
params[Pendulum.DT_KEY] = 5e-2
params[Pendulum.SIMULATION_LENGTH_KEY] = 2

data = Pendulum(img_size=img_size, pendulum_params=params, seed=0)
training_set_size = 1000
validation_set_size = 100
test_set_size = 100
q2 = 0.001
data.transition_noise_std = np.sqrt(q2)

continuous = data.sample_continuous_data_set(training_set_size + validation_set_size + test_set_size)
np.random.shuffle(continuous)
img = data.generate_images(continuous) / 255.0

out_dir = Path("Simulations/Pendulum")
out_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    out_dir / "states_q2_0.001_Baseline.npz",
    training_set=continuous[:training_set_size, :, :],
    validation_set=continuous[training_set_size:(training_set_size + validation_set_size), :, :],
    test_set=continuous[training_set_size + validation_set_size:, :, :],
)
np.savez(
    out_dir / "observations_q2_0.001_Baseline.npz",
    training_set=img[:training_set_size, ...],
    validation_set=img[training_set_size:(training_set_size + validation_set_size), ...],
    test_set=img[(training_set_size + validation_set_size):, ...],
)
print("Saved Pendulum Baseline data under Simulations/Pendulum")
'@
    $tempScript = Join-Path $ProjectRoot "scripts\_generate_pendulum_baseline.py"
    Set-Utf8NoBomFile -Path $tempScript -Value $script
    try {
        Invoke-ProjectPython -LogName "generate_pendulum_baseline" -Arguments @($tempScript)
    }
    finally {
        Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    }
}

function Generate-LorenzData {
    param([Parameter(Mandatory = $true)][string]$Scenario)

    $tag = "generate_lorenz_$($Scenario.ToLower())"
    Set-ExperimentConfig -Dataset "Lorenz" -Scenario $Scenario -GenerateData $true -RealR2 $LorenzNoiseValues[0] -ModelFolder $ModelRoot
    Invoke-ProjectPython -LogName $tag -Arguments @("-u", "-c", "import config")
}

function Train-MainVisual {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("Lorenz", "Pendulum")][string]$Dataset,
        [Parameter(Mandatory = $true)][string]$Scenario,
        [Parameter(Mandatory = $true)][double]$RealR2,
        [Parameter(Mandatory = $true)][string]$ExperimentName
    )

    $safeR = ($RealR2.ToString()).Replace(".", "p")
    $tag = "${Dataset}_${ExperimentName}_${Scenario}_r${safeR}_epoch${Epochs}".ToLower()
    $modelFolder = "$ModelRoot/$ExperimentName"
    Set-ExperimentConfig -Dataset $Dataset -Scenario $Scenario -GenerateData $false -RealR2 $RealR2 -ModelFolder $modelFolder
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot ($modelFolder.TrimStart("./") -replace "/", "\")) | Out-Null
    Invoke-ProjectPython -LogName $tag -Arguments @("-u", "main_visual.py")
    Copy-MainVisualFigures -Tag $tag
}

Assert-TorchDevice
Ensure-LorenzEncoderAliases

if (-not $SkipReferenceFigureExport) {
    Invoke-ProjectPython -LogName "export_reference_reproduction_figures" -Arguments @("scripts\export_reproduction_figures.py")
}

try {
if (-not $SkipLorenz) {
    Set-LorenzTaylorOrder -J 5

    if (-not $SkipDataGeneration) {
        Generate-LorenzData -Scenario "Baseline"
        Generate-LorenzData -Scenario "Test_Long_Trajectories"
        Generate-LorenzData -Scenario "Decimation"
    }

    if (-not $SkipTraining) {
        foreach ($r in $LorenzNoiseValues) {
            Train-MainVisual -Dataset "Lorenz" -Scenario "Baseline" -RealR2 $r -ExperimentName "lorenz_baseline_j5"
        }

        foreach ($r in $LorenzNoiseValues) {
            Train-MainVisual -Dataset "Lorenz" -Scenario "Test_Long_Trajectories" -RealR2 $r -ExperimentName "lorenz_long_j5"
        }

        foreach ($r in $LorenzNoiseValues) {
            Train-MainVisual -Dataset "Lorenz" -Scenario "Decimation" -RealR2 $r -ExperimentName "lorenz_decimation_j5"
        }

        Set-LorenzTaylorOrder -J 1
        foreach ($r in $LorenzNoiseValues) {
            Train-MainVisual -Dataset "Lorenz" -Scenario "Baseline" -RealR2 $r -ExperimentName "lorenz_wrong_f_j1"
        }
        Set-LorenzTaylorOrder -J 5
    }
}

if (-not $SkipPendulum) {
    if (-not $SkipDataGeneration) {
        Ensure-PendulumBaselineData
    }

    if (-not $SkipTraining) {
        foreach ($r in $PendulumNoiseValues) {
            Train-MainVisual -Dataset "Pendulum" -Scenario "Baseline" -RealR2 $r -ExperimentName "pendulum_baseline"
        }
    }
}
}
finally {
    Set-LorenzTaylorOrder -J 5
}

Write-Host "Reproduction script completed."
Write-Host "Logs: $LogDir"
Write-Host "Figures: $FigureDir"
Write-Host "Models: $(Join-Path $ProjectRoot "results\models_$RunDevice")"
