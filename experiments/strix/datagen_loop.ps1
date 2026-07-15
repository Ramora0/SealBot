# On-distribution data generation: cand_mixnet vs strix, dev openings ONLY
# (held 50-74 must never enter training data), games recorded for the
# bench_targets true+soft stream. SEAL_THREADS=8: runs alongside GPU
# training; data games do not need clean clocks (never gate on these).
$py = "C:/Users/Lee/coding/python/ai/hexo-strix/.venv/Scripts/python.exe"
$repo = "C:\Users\Lee\coding\python\AI\SealBot"
$env:SEAL_EVAL = "mixnet"
$env:SEAL_MIXNET_BLOB = "$repo\cand_mixnet\mixnet.bin"
$env:SEAL_TRUNK_BLEND = "0"
$env:SEAL_POLICY_MODE = "74"
$env:SEAL_VCF = "11"
$env:SEAL_VCF_K = "9"
$env:SEAL_VCF_BUDGET = "25000"
$env:SEAL_SMP_MODE = "2"
$env:SEAL_THREADS = "8"
$env:SEALBOT_ROOT = $repo
$env:STRIX_ROOT = "C:\Users\Lee\coding\python\ai\hexo-strix"
$env:STRIX_CKPT = "C:\Users\Lee\OneDrive\Desktop\checkpoint_00237000.pt"
Set-Location "$repo\experiments\strix"
for ($i = 1; $i -le 14; $i++) {
    $out = "data_runs\datagen_r$i.json"
    if (Test-Path $out) { continue }
    Write-Output "=== datagen round $i ==="
    & $py bench_vs_strix.py --bot-dir "$repo\cand_mixnet" --games 100 `
        --tl 0.44 --sims 64 --m-actions 16 --pipeline 2 `
        --openings "$repo\autoresearch\results\_dev_openings.pkl" `
        --record --out $out
}
Write-Output "=== datagen loop complete ==="
