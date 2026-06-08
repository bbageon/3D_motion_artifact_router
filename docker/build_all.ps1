# ArtifactRouter generator 서비스 이미지 일괄 빌드 (로컬 전용).
# 사용: powershell -File docker/build_all.ps1   (repo 루트에서 실행)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # repo root (docker/ 의 부모)
Push-Location $root
try {
    foreach ($g in @("momask", "motiongpt", "mdm")) {
        Write-Host "==== building artifactrouter/$g ====" -ForegroundColor Cyan
        docker build -f "docker/$g/Dockerfile" -t "artifactrouter/$g`:latest" .
        if ($LASTEXITCODE -ne 0) { throw "build failed: $g" }
    }
    Write-Host "==== done. images: ====" -ForegroundColor Green
    docker images artifactrouter/* --format "{{.Repository}}:{{.Tag}}  {{.Size}}"
} finally {
    Pop-Location
}
