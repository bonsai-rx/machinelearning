# run_bonsai_test.ps1

param(
    [String]$BASE_PATH,
    [String]$N_SAMPLES
)

dotnet new bonsaienv -o $BASE_PATH --allow-scripts Yes
Copy-Item "$BASE_PATH\\Bonsai.config" "$BASE_PATH\\.bonsai\\"
Copy-Item "$BASE_PATH\\NuGet.config" "$BASE_PATH\\.bonsai\\"

& "$BASE_PATH\\.bonsai\\Bonsai.exe" "$BASE_PATH\\receptive_field.bonsai" --no-editor --start --property PythonHome="$BASE_PATH\\.venv" --property ImagesCsv="$BASE_PATH\\ReceptiveFieldSimpleCell\\images.csv" --property ResponsesCsv="$BASE_PATH\\ReceptiveFieldSimpleCell\\responses.csv" --property NSamples=$N_SAMPLES
