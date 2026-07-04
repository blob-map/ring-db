param(
    [string]$Version
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Usage {
    Write-Host "Usage: .\release_tag.ps1 <version>"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        $display = (@($FilePath) + $ArgumentList) -join " "
        throw "Command failed with exit code ${LASTEXITCODE}: $display"
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Update-PackageVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Version
    )

    $content = [System.IO.File]::ReadAllText($Path)
    $regex = [regex]::new('(?ms)(^\[package\]\s*.*?^version\s*=\s*")[^"]+(")')

    if (-not $regex.IsMatch($content)) {
        throw "Could not find [package] version in $Path"
    }

    $updated = $regex.Replace($content, {
        param($match)
        $match.Groups[1].Value + $Version + $match.Groups[2].Value
    }, 1)

    Write-Utf8NoBom $Path $updated
}

function Update-DeriveDependencyVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Version
    )

    $content = [System.IO.File]::ReadAllText($Path)
    $regex = [regex]::new('(ring-db-derive\s*=\s*\{\s*path\s*=\s*"derive"\s*,\s*version\s*=\s*")[^"]+(")')

    if (-not $regex.IsMatch($content)) {
        throw "Could not find ring-db-derive path dependency in $Path"
    }

    $updated = $regex.Replace($content, {
        param($match)
        $match.Groups[1].Value + $Version + $match.Groups[2].Value
    }, 1)

    Write-Utf8NoBom $Path $updated
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    Write-Usage
    exit 1
}

if ($Version.StartsWith("v")) {
    throw "Pass the Cargo version without a leading 'v' (for example: 0.7.0)."
}

foreach ($tool in @("cargo", "git")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required command not found on PATH: $tool"
    }
}

Push-Location -LiteralPath $PSScriptRoot
try {
    Write-Host "Updating dependencies..."
    Invoke-Checked "cargo" @("update")

    Write-Host "Pre-release checks..."
    Invoke-Checked "cargo" @("check")
    Invoke-Checked "cargo" @("test")
    Invoke-Checked "cargo" @("fmt")
    Invoke-Checked "cargo" @("clippy", "--", "-D", "warnings")

    Write-Host "Updating Cargo.toml to $Version"
    Update-PackageVersion "Cargo.toml" $Version

    Write-Host "Updating derive/Cargo.toml to $Version"
    Update-PackageVersion "derive/Cargo.toml" $Version

    Write-Host "Updating ring-db-derive version reference in Cargo.toml"
    Update-DeriveDependencyVersion "Cargo.toml" $Version

    Write-Host "Post-version build check..."
    Invoke-Checked "cargo" @("check")
    Invoke-Checked "cargo" @("test")

    Invoke-Checked "git" @("add", ".")
    Invoke-Checked "git" @("commit", "-m", "Release v$Version")
    Invoke-Checked "git" @("tag", "-a", "v$Version", "-m", "Release v$Version")

    Invoke-Checked "git" @("push")
    Invoke-Checked "git" @("push", "origin", "v$Version")

    Write-Host "Release v$Version complete"
}
finally {
    Pop-Location
}
