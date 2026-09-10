$ErrorActionPreference = "Stop"

Set-Location "C:\Users\willi\Documents\Code\toric-verified-numerics"

$scripts = @(
    "bound_metric/bound_inverse.jl"
    "bound_metric/bound_riem.jl"
    "bound_metric/bound_ricci.jl"
    "bound_residual/bound_residual.jl"
    "eigenvalues/orchestrate.jl"
    "eigenvalues/verify_eigenvalue.m"
    "eigenvalues/update_bound.jl"
    "apply_fixed_point/apply_fixed_point.jl"
    "bound_hsc/find_negative_curvature_region.jl"
    "bound_hsc/verify_proposition_5_6.jl"
    "eigenvalue_comparison/upper_bound.jl"
    "eigenvalue_comparison/eigenvalue_for_true_KE_metric.jl"
)

# Check both runtimes before starting the long numerical calculations.
Get-Command julia, matlab -ErrorAction Stop | Out-Null

$timingFile = Join-Path $PWD (
    "pipeline_timings_{0}.csv" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)

$results = [System.Collections.Generic.List[object]]::new()
$total = [System.Diagnostics.Stopwatch]::StartNew()
$overallStatus = "Failed"

try {
    foreach ($script in $scripts) {
        Write-Host "`n>>> $script" -ForegroundColor Cyan

        $started = Get-Date
        $status = "Failed"
        $timer = [System.Diagnostics.Stopwatch]::StartNew()

        try {
            if ([System.IO.Path]::GetExtension($script) -eq ".m") {
                & matlab -batch "run('$script')"
            }
            else {
                & julia --project=. $script
            }
            if ($LASTEXITCODE -ne 0) {
                throw "$script failed with exit code $LASTEXITCODE"
            }
            $status = "Completed"
        }
        finally {
            $timer.Stop()

            $results.Add([pscustomobject]@{
                Script    = $script
                Status    = $status
                Started   = $started.ToString("o")
                Duration  = $timer.Elapsed.ToString()
                Seconds   = $timer.Elapsed.TotalSeconds
            })

            $results | Export-Csv -LiteralPath $timingFile -NoTypeInformation

            Write-Host (
                "{0}: {1}" -f $status, $timer.Elapsed
            ) -ForegroundColor Yellow
        }
    }

    $overallStatus = "Completed"
}
finally {
    $total.Stop()

    $results.Add([pscustomobject]@{
        Script    = "TOTAL"
        Status    = $overallStatus
        Started   = ""
        Duration  = $total.Elapsed.ToString()
        Seconds   = $total.Elapsed.TotalSeconds
    })

    $results | Export-Csv -LiteralPath $timingFile -NoTypeInformation

    Write-Host "`nTotal time: $($total.Elapsed)"
    Write-Host "Timings saved to: $timingFile"
}
