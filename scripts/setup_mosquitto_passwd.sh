#!/bin/bash

# Setup Mosquitto password file for PV Monitoring System
# This script creates a password file for MQTT broker authentication

PASSWD_FILE="config/mosquitto/passwd"

echo "Mosquitto Password Setup"
echo "========================="
echo ""

# Check if mosquitto_passwd is available
if ! command -v mosquitto_passwd &> /dev/null; then
    echo "Warning: mosquitto_passwd command not found."
    echo "Please install mosquitto or create the password file manually."
    echo ""
    echo "Manual creation:"
    echo "  touch $PASSWD_FILE"
    echo "  echo 'username:password' > $PASSWD_FILE"
    echo ""
    exit 1
fi

# Create directory if it doesn't exist
mkdir -p config/mosquitto

# Prompt for username
read -p "Enter MQTT username [default: pv-monitor]: " username
username=${username:-pv-monitor}

# Create password file
echo "Creating password file for user: $username"
mosquitto_passwd -c "$PASSWD_FILE" "$username"

echo ""
echo "Password file created successfully at: $PASSWD_FILE"
echo "You can add more users with: mosquitto_passwd -b $PASSWD_FILE <username> <password>"
echo ""
