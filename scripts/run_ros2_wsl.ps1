param(
    [ValidateSet("mock", "deep")]
    [string]$Mode = "mock"
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$linuxProject = (wsl -d Ubuntu-24.04 -- wslpath -a $project).Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not resolve WSL project path" }

$runner = "$linuxProject/scripts/run_ros2_wsl.sh"
wsl -d Ubuntu-24.04 -- bash $runner $Mode $linuxProject
if ($LASTEXITCODE -ne 0) { throw "ROS2 pipeline exited with code $LASTEXITCODE" }
