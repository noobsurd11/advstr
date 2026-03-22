#!/bin/bash

LOGOUT_URL="https://login.iitmandi.ac.in:1003/logout?"

curl -s -k "$LOGOUT_URL" -o /tmp/logOUT.html
echo "Logout successful "
