#!/usr/bin/env python
from passlib.hash import bcrypt
import pyotp
import sys

def main():
    print("Options Trading App - Setup Helper")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--password':
        password = input("Enter password to hash: ")
        hashed = bcrypt.hash(password)
        print(f"\nPassword hash (add to .env):")
        print(f"APP_PASSWORD_HASH={hashed}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--totp':
        secret = pyotp.random_base32()
        print(f"\nTOTP Secret (add to .env):")
        print(f"TOTP_SECRET={secret}")
        
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name='admin@example.com',
            issuer_name='Options Trading App'
        )
        print(f"\nProvisioning URI for QR code:")
        print(provisioning_uri)
        print(f"\nCurrent TOTP code: {totp.now()}")
    
    else:
        print("\nUsage:")
        print("  python generate_password.py --password   # Generate password hash")
        print("  python generate_password.py --totp        # Generate TOTP secret")

if __name__ == "__main__":
    main()