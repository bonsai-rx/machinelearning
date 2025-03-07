#! /bin/bash

SETUP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

DEFAULT_VERSION="latest"
VERSION="$DEFAULT_VERSION"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version) VERSION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Setting up Bonsai v=$VERSION environment..."

if [ ! -f "$SETUP_SCRIPT_DIR/Bonsai.exe" ]; then
    CONFIG="$SETUP_SCRIPT_DIR/Bonsai.config"
    if [ -f "$CONFIG" ]; then
        DETECTED=$(xmllint --xpath '//PackageConfiguration/Packages/Package[@id="Bonsai"]/@version' "$CONFIG" | sed -e 's/^[^"]*"//' -e 's/"$//')
        echo "Version detected v=$DETECTED."
        RELEASE="https://github.com/bonsai-rx/bonsai/releases/download/$DETECTED/Bonsai.zip"
    else
        if [ $VERSION = "latest" ]; then
            RELEASE="https://github.com/bonsai-rx/bonsai/releases/latest/download/Bonsai.zip"
        else
            RELEASE="https://github.com/bonsai-rx/bonsai/releases/download/$VERSION/Bonsai.zip"
        fi
    fi
    echo "Download URL: $RELEASE"
    wget $RELEASE -O "$SETUP_SCRIPT_DIR/temp.zip"
    mv -f "$SETUP_SCRIPT_DIR/NuGet.config" "$SETUP_SCRIPT_DIR/temp.config"
    unzip -d "$SETUP_SCRIPT_DIR" -o "$SETUP_SCRIPT_DIR/temp.zip"
    mv -f "$SETUP_SCRIPT_DIR/temp.config" "$SETUP_SCRIPT_DIR/NuGet.config"
    rm -rf "$SETUP_SCRIPT_DIR/temp.zip"
    rm -rf "$SETUP_SCRIPT_DIR/Bonsai32.exe"
fi

source "$SETUP_SCRIPT_DIR/activate"
source "$SETUP_SCRIPT_DIR/run" --no-editor
