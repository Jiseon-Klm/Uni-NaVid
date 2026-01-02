#!/bin/bash
# Setup script for Korean (Hangul) input on Ubuntu

echo "Setting up Korean (Hangul) input method..."

# Check if ibus-hangul is installed
if ! dpkg -l | grep -q ibus-hangul; then
    echo "Installing ibus-hangul..."
    sudo apt-get update
    sudo apt-get install -y ibus-hangul
fi

# Start IBus daemon if not running
if ! pgrep -x ibus-daemon > /dev/null; then
    echo "Starting IBus daemon..."
    ibus-daemon -drx &
    sleep 2
fi

# Add Korean input to GNOME settings (if using GNOME/Unity)
if command -v gsettings &> /dev/null; then
    echo "Adding Korean input to system settings..."
    current_sources=$(gsettings get org.gnome.desktop.input-sources sources)
    if [[ ! "$current_sources" == *"hangul"* ]]; then
        echo "Run this command to add Korean input:"
        echo "gsettings set org.gnome.desktop.input-sources sources \"[('xkb', 'us'), ('ibus', 'hangul')]\""
        echo ""
        echo "Or use the GUI: Settings > Region & Language > Input Sources > Add Korean (Hangul)"
    else
        echo "Korean input is already configured!"
    fi
fi

echo ""
echo "Setup complete!"
echo ""
echo "To use Korean input:"
echo "  - Press Super+Space or Ctrl+Space to toggle"
echo "  - Or click the IBus icon in the system tray"
echo ""
echo "To configure, run: ibus-setup"

