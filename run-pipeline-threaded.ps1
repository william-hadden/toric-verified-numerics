$ErrorActionPreference = "Stop"

$repo = "C:\Users\willi\Documents\Code\toric-verified-numerics"
Set-Location $repo

# $maxThreads = [Environment]::ProcessorCount
$maxThreads = 32

$scripts = @(
    # "bound_metric/bound_inverse.jl",
    # "bound_metric/bound_riem.jl",
    # "bound_metric/bound_ricci.jl",
    # "bound_residual/bound_residual.jl",
    "eigenvalues/orchestrate.jl",
    "eigenvalues/verify_eigenvalue.m",
    "eigenvalues/update_bound.jl",
    "apply_fixed_point/apply_fixed_point.jl",
    "bound_hsc/find_negative_curvature_region.jl",
    "bound_hsc/verify_proposition_5_6.jl",
    "eigenvalue_comparison/upper_bound.jl",
    "eigenvalue_comparison/eigenvalue_for_true_KE_metric.jl"
)

Get-Command julia, matlab -ErrorAction Stop | Out-Null

$timingFile = Join-Path $PWD (
    "pipeline_timings_{0}.csv" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)

$results = [System.Collections.Generic.List[object]]::new()
$total = [System.Diagnostics.Stopwatch]::StartNew()
$overallStatus = "Failed"

function Set-NumericThreadEnvironment {
    param([int] $Threads)

    $env:OMP_NUM_THREADS = "$Threads"
    $env:OPENBLAS_NUM_THREADS = "$Threads"
    $env:MKL_NUM_THREADS = "$Threads"
    $env:VECLIB_MAXIMUM_THREADS = "$Threads"
    $env:NUMEXPR_NUM_THREADS = "$Threads"
}

try {
    foreach ($script in $scripts) {
        Write-Host "`n>>> $script" -ForegroundColor Cyan

        $started = Get-Date
        $status = "Failed"
        $timer = [System.Diagnostics.Stopwatch]::StartNew()

        try {
            if ([System.IO.Path]::GetExtension($script) -eq ".m") {
                Set-NumericThreadEnvironment -Threads $maxThreads

                $matlabScript = $script.Replace("\", "/").Replace("'", "''")

                $batchCommand = "try, maxNumCompThreads($maxThreads); catch ME, warning(ME.message); end; run('$matlabScript');"

                Write-Host "MATLAB threads: $maxThreads"
                & matlab -batch $batchCommand
            }
            else {
                if ($script -eq "eigenvalues/orchestrate.jl") {
                    $juliaThreads = $maxThreads
                    Set-NumericThreadEnvironment -Threads $maxThreads
                }
                else {
                    $juliaThreads = 1
                    Set-NumericThreadEnvironment -Threads 1
                }

                Write-Host "Julia threads: $juliaThreads"
                & julia "--threads=$juliaThreads" --project=. $script
            }

            if ($LASTEXITCODE -ne 0) {
                throw "$script failed with exit code $LASTEXITCODE"
            }

            $status = "Completed"
        }
        finally {
            $timer.Stop()

            $results.Add([pscustomobject]@{
                Script   = $script
                Status   = $status
                Started  = $started.ToString("o")
                Duration = $timer.Elapsed.ToString()
                Seconds  = $timer.Elapsed.TotalSeconds
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
        Script   = "TOTAL"
        Status   = $overallStatus
        Started  = ""
        Duration = $total.Elapsed.ToString()
        Seconds  = $total.Elapsed.TotalSeconds
    })

    $results | Export-Csv -LiteralPath $timingFile -NoTypeInformation

    Write-Host "`nTotal time: $($total.Elapsed)"
    Write-Host "Timings saved to: $timingFile"
}