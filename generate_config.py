#!/usr/bin/env python
"""
Configuration Generator for Maximum Profit Trading
Generates secure passwords and helps set up your .env file
"""

import os
import secrets
import base64
from passlib.context import CryptContext

def generate_password_hash():
    """Generate bcrypt hash for your password"""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    print("[!] PASSWORD SETUP")
    print("=" * 40)
    
    password = input("Enter your login password: ").strip()
    if len(password) < 6:
        print("[-] Password must be at least 6 characters")
        return
    
    hash_value = pwd_context.hash(password)
    print(f"\n[+] Generated password hash:")
    print(f"APP_PASSWORD_HASH={hash_value}")
    
    return hash_value

def generate_totp_secret():
    """Generate TOTP secret for 2FA (optional)"""
    secret = base64.b32encode(secrets.token_bytes(20)).decode('utf-8')
    print(f"\n[!] 2FA SETUP (Optional)")
    print(f"TOTP_SECRET={secret}")
    print(f"Use this in Google Authenticator or similar app")
    
    return secret

def get_tradier_info():
    """Get Tradier account information"""
    print("\n[$] TRADIER BROKER SETUP")
    print("=" * 40)
    print("1. Sign up at: https://tradier.com")
    print("2. Apply for options trading (Level 2+)")
    print("3. Get API credentials from Account > API Access")
    print()
    
    account_id = input("Enter your Tradier Account ID: ").strip()
    access_token = input("Enter your Tradier Access Token: ").strip()
    
    if not account_id or not access_token:
        print("[-] Both Account ID and Access Token are required")
        return None, None
    
    print("\n[+] Tradier credentials configured")
    return account_id, access_token

def get_trading_params():
    """Get trading parameters"""
    print("\n[#] TRADING PARAMETERS")
    print("=" * 40)
    
    try:
        capital = float(input("Enter your starting capital ($): ").strip())
        if capital < 10:
            print("[-] Minimum capital is $10")
            return None
    except ValueError:
        print("[-] Invalid capital amount")
        return None
    
    mode = input("Trading mode (paper/live) [paper]: ").strip().lower() or "paper"
    if mode not in ['paper', 'live']:
        print("[-] Mode must be 'paper' or 'live'")
        return None
    
    return capital, mode

def update_env_file(config_data):
    """Update the .env file with new configuration"""
    env_file = ".env"
    
    # Read current .env
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Update or add configuration lines
    updated_lines = []
    keys_updated = set()
    
    for line in lines:
        if '=' in line and not line.startswith('#'):
            key = line.split('=')[0].strip()
            if key in config_data:
                updated_lines.append(f"{key}={config_data[key]}\n")
                keys_updated.add(key)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
    
    # Add any missing keys
    for key, value in config_data.items():
        if key not in keys_updated:
            updated_lines.append(f"{key}={value}\n")
    
    # Write updated .env file
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"\n[+] Configuration saved to {env_file}")

def main():
    """Main configuration setup"""
    
    print("[!] MAXIMUM PROFIT TRADING - CONFIGURATION SETUP")
    print("=" * 60)
    print()
    
    config_data = {}
    
    # Generate password hash
    password_hash = generate_password_hash()
    if password_hash:
        config_data['APP_PASSWORD_HASH'] = password_hash
    
    # Optional 2FA
    setup_2fa = input("\nSet up 2FA? (y/n) [n]: ").strip().lower()
    if setup_2fa == 'y':
        totp_secret = generate_totp_secret()
        config_data['TOTP_SECRET'] = totp_secret
    
    # Tradier setup
    setup_broker = input("\nSet up Tradier broker? (y/n) [y]: ").strip().lower()
    if setup_broker != 'n':
        account_id, access_token = get_tradier_info()
        if account_id and access_token:
            config_data['TRADIER_ACCOUNT_ID'] = account_id
            config_data['TRADIER_ACCESS_TOKEN'] = access_token
    
    # Trading parameters
    trading_params = get_trading_params()
    if trading_params:
        capital, mode = trading_params
        config_data['INITIAL_CAPITAL'] = str(capital)
        config_data['EXECUTION_MODE'] = mode
        config_data['TRADIER_ENV'] = mode  # Match the execution mode
    
    # Update .env file
    if config_data:
        update_env_file(config_data)
        
        print("\n[+] CONFIGURATION COMPLETE!")
        print("=" * 40)
        print("Your .env file has been updated with:")
        for key, value in config_data.items():
            if 'PASSWORD' in key or 'TOKEN' in key:
                print(f"  {key}=***hidden***")
            else:
                print(f"  {key}={value}")
        
        print(f"\n[+] Ready to start trading!")
        print(f"Run: python app.py")
        print(f"Then go to: http://localhost:8080")
    else:
        print("\n[-] Configuration incomplete. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Configuration cancelled")
    except Exception as e:
        print(f"\n[-] Error: {e}")
        print("Please check your inputs and try again")