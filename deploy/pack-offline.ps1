<#
.SYNOPSIS
Pack a tested Linux ARM64 application image and deployment files into an offline bundle.

.DESCRIPTION
Verifies that the local Docker image is linux/arm64, exports images.tar, copies
compose.yaml, lab.env.example, and README.md, and generates Linux-compatible
images.txt and SHA256SUMS (UTF-8 without BOM, LF line endings).

.PARAMETER OutputDir
New directory path for the offline bundle. Must not already exist.

.PARAMETER Image
Tested Linux ARM64 application image tag.

.EXAMPLE
pwsh deploy/pack-offline.ps1 -OutputDir atc-bundle -Image ghcr.io/org/atc-app:tag

.EXAMPLE
pwsh deploy/pack-offline.ps1 atc-bundle ghcr.io/org/atc-app:tag
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [Alias('o')]
    [string]$OutputDir,

    [Parameter(Mandatory = $true, Position = 1)]
    [Alias('i')]
    [string]$Image
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Fail {
    param([Parameter(Mandatory = $true)][string]$Message)
    [Console]::Error.WriteLine($Message)
    exit 1
}

function Write-Utf8LfFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

if (Test-Path -LiteralPath $OutputDir) {
    Fail "Output directory already exists: $OutputDir"
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail 'docker CLI is required.'
}

$platform = (& docker image inspect --format '{{.Os}}/{{.Architecture}}' $Image 2>&1)
if ($LASTEXITCODE -ne 0) {
    Fail ($platform -join [Environment]::NewLine)
}
$platformText = ($platform | Out-String).Trim()
if ($platformText -ne 'linux/arm64') {
    Fail "Expected a native ARM64 image, got: $platformText"
}

$imageMeta = (& docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' $Image 2>&1)
if ($LASTEXITCODE -ne 0) {
    Fail ($imageMeta -join [Environment]::NewLine)
}
$imageMetaText = ($imageMeta | Out-String).Trim()

$resolvedOutput = (New-Item -ItemType Directory -Path $OutputDir).FullName

$bundleFiles = @('compose.yaml', 'lab.env.example', 'README.md')
foreach ($name in $bundleFiles) {
    $src = Join-Path $PSScriptRoot $name
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
        Fail "Required deployment file not found: $src"
    }
    Copy-Item -LiteralPath $src -Destination (Join-Path $resolvedOutput $name)
}

$imagesTxtPath = Join-Path $resolvedOutput 'images.txt'
Write-Utf8LfFile -Path $imagesTxtPath -Content ("{0}`t{1}`n" -f $Image, $imageMetaText)

$imagesTarPath = Join-Path $resolvedOutput 'images.tar'
& docker image save --output $imagesTarPath $Image
if ($LASTEXITCODE -ne 0) {
    Fail "docker image save failed with exit code $LASTEXITCODE"
}

$checksumFiles = @('compose.yaml', 'lab.env.example', 'README.md', 'images.txt', 'images.tar')
$checksumLines = foreach ($name in $checksumFiles) {
    $filePath = Join-Path $resolvedOutput $name
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $filePath).Hash.ToLowerInvariant()
    "{0}  {1}" -f $hash, $name
}

$checksumPath = Join-Path $resolvedOutput 'SHA256SUMS'
Write-Utf8LfFile -Path $checksumPath -Content (($checksumLines -join "`n") + "`n")
