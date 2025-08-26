#!/usr/bin/env python
"""
Quick password hash generator for testing
"""
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Generate a hash for a test password
# You should change this password to your own secure password
test_password = "TradingPro2025!"
hash_value = pwd_context.hash(test_password)

print("[+] Generated password hash for 'TradingPro2025!':")
print(f"APP_PASSWORD_HASH={hash_value}")
print()
print("[!] IMPORTANT: Update your .env file with this hash")
print("[!] Or run generate_config.py manually to set your own password")