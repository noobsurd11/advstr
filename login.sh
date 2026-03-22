#!/bin/bash
if ping -c 1 -q google.com &>/dev/null; then
    echo "Already Logged In"
    exit 0
fi

read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD
echo
PORTAL_URL="https://login.iitmandi.ac.in:1003/portal?"

echo "[*] Fetching login page..."
curl -s -k -L "$PORTAL_URL" -o login.html

MAGIC=$(grep -oP 'name="magic" value="\K[^"]+' login.html)
REDIR=$(grep -oP 'name="4Tredir" value="\K[^"]+' login.html)

echo "[*] Submitting credentials..."
RESPONSE=$(curl -s -k -X POST "$PORTAL_URL" \
  -d "username=$USERNAME" \
  -d "password=$PASSWORD" \
  -d "4Tredir=$REDIR" \
  -d "magic=$MAGIC")

# --- Step 4: Extract redirect URL from JS ---
REDIRECT_URL=$(echo "$RESPONSE" | grep -oP 'window.location="\K[^"]+')

if [ -z "$REDIRECT_URL" ]; then
    echo "INVALID CREDENTIALS."
    exit 1
fi

echo "[*] Following redirect to complete login..."
curl -s -k "$REDIRECT_URL" -o /tmp/login_final.html
echo "Login successful"
