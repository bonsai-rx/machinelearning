#!/bin/bash
# run_bonsai_test.sh

BASE_PATH="$1"
N_SAMPLES="$2"

dotnet new bonsaienvl -o $BASE_PATH/.bonsai --allow-scripts Yes
cp $BASE_PATH/Bonsai.config $BASE_PATH/.bonsai/
cp $BASE_PATH/NuGet.config $BASE_PATH/.bonsai/

source $BASE_PATH/.bonsai/activate
source $BASE_PATH/.bonsai/run $BASE_PATH/receptive_field.bonsai --no-editor --start --property PythonHome=\"$BASE_PATH/.venv\" --property ImagesCsv=\"$BASE_PATH/ReceptiveFieldSimpleCell/images.csv\" --property ResponsesCsv=\"$BASE_PATH/ReceptiveFieldSimpleCell/responses.csv\" --property NSamples=$N_SAMPLES