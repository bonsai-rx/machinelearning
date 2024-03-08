.\bonsai\modules\Export-Image.ps1 "..\src\Bonsai.ML.Visualizers\bin\Release\net472"
$exampleEnvironments = Get-ChildItem -Directory -Recurse -Filter ".bonsai" -Path "examples"
foreach ($environment in $exampleEnvironments)
{
    # bootstrap environment
    $galleryPath = Join-Path $environment.FullName "Gallery"
    New-Item -Path $galleryPath -Force -ItemType Directory | Out-Null
    Get-ChildItem -Path "..\src\bin\Release" -Filter *.nupkg | Copy-Item -Destination $galleryPath
    . (Join-Path $environment.FullName "Setup.ps1")

    # export example workflows
    $workflowPath = $environment.Parent.FullName
    $bootstrapperPath = Join-Path $environment.FullName "Bonsai.exe"
    .\bonsai\modules\Export-Image.ps1 -workflowPath $workflowPath -bootstrapperPath $bootstrapperPath
}
dotnet docfx @args