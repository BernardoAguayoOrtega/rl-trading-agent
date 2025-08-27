"""
Secure Secrets Management for AI Trading Agent
=============================================

This module provides secure methods to load and manage sensitive configuration
like API keys, database credentials, and other secrets.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any


class SecretsManager:
    """Secure secrets management with multiple loading strategies"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize secrets manager
        
        Args:
            env_file: Path to .env file (optional)
        """
        self.env_file = env_file or ".env"
        self.secrets_cache: Dict[str, str] = {}
        self._load_secrets()
    
    def _load_secrets(self) -> None:
        """Load secrets from various sources in priority order"""
        # 1. Load from .env file if it exists
        self._load_from_env_file()
        
        # 2. Load from environment variables (these override .env file)
        self._load_from_environment()
        
        # 3. Validate required secrets
        self._validate_secrets()
    
    def _load_from_env_file(self) -> None:
        """Load secrets from .env file"""
        env_path = Path(self.env_file)
        
        if not env_path.exists():
            print(f"ℹ️  No .env file found at {env_path}")
            print(f"💡 Create one from .env.example for convenience")
            return
        
        try:
            with open(env_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        
                        # Only cache if not already in environment
                        if key not in os.environ:
                            self.secrets_cache[key] = value
                            os.environ[key] = value
                    else:
                        warnings.warn(f"Invalid line format in {env_path}:{line_num}: {line}")
            
            print(f"✅ Loaded secrets from {env_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to load {env_path}: {e}")
    
    def _load_from_environment(self) -> None:
        """Load secrets from environment variables"""
        env_secrets = {
            key: value for key, value in os.environ.items()
            if any(secret_key in key.upper() for secret_key in [
                'API_KEY', 'SECRET', 'PASSWORD', 'TOKEN', 'CREDENTIAL'
            ])
        }
        
        if env_secrets:
            print(f"✅ Found {len(env_secrets)} secrets in environment variables")
            self.secrets_cache.update(env_secrets)
    
    def _validate_secrets(self) -> None:
        """Validate that required secrets are available"""
        required_secrets = ['OPENAI_API_KEY']
        missing_secrets = []
        
        for secret in required_secrets:
            if not self.get_secret(secret):
                missing_secrets.append(secret)
        
        if missing_secrets:
            print(f"⚠️  Missing required secrets: {', '.join(missing_secrets)}")
            print("💡 Add them to your .env file or set as environment variables")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value securely
        
        Args:
            key: Secret key name
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        # Priority: Environment variable > .env file > default
        value = os.environ.get(key) or self.secrets_cache.get(key) or default
        
        if value and key.upper() in ['OPENAI_API_KEY', 'API_KEY']:
            # Validate API key format
            if not self._validate_api_key(value):
                warnings.warn(f"Invalid API key format for {key}")
                return None
        
        return value
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        if not api_key or len(api_key) < 10:
            return False
        
        # OpenAI API keys typically start with 'sk-'
        if api_key.startswith('sk-') and len(api_key) > 40:
            return True
        
        # For development/testing, allow placeholder values
        if api_key in ['your-api-key-here', 'test-key', 'development']:
            return False
        
        return True
    
    def set_secret(self, key: str, value: str) -> None:
        """Set a secret value (for runtime configuration)
        
        Args:
            key: Secret key name
            value: Secret value
        """
        os.environ[key] = value
        self.secrets_cache[key] = value
    
    def list_available_secrets(self) -> Dict[str, str]:
        """List available secrets (masked for security)
        
        Returns:
            Dictionary of secret keys with masked values
        """
        masked_secrets = {}
        
        for key, value in self.secrets_cache.items():
            if value:
                # Mask the value for security
                if len(value) > 8:
                    masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:]
                else:
                    masked_value = '*' * len(value)
                masked_secrets[key] = masked_value
            else:
                masked_secrets[key] = "[NOT SET]"
        
        return masked_secrets
    
    def validate_openai_key(self) -> bool:
        """Validate OpenAI API key specifically
        
        Returns:
            True if valid OpenAI API key is available
        """
        api_key = self.get_secret('OPENAI_API_KEY')
        return api_key is not None and self._validate_api_key(api_key)


# Global secrets manager instance
secrets_manager = SecretsManager()


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret
    
    Args:
        key: Secret key name
        default: Default value if secret not found
        
    Returns:
        Secret value or default
    """
    return secrets_manager.get_secret(key, default)


def setup_secrets_interactive() -> None:
    """Interactive setup for secrets"""
    print("🔐 AI Trading Agent - Secrets Setup")
    print("=" * 40)
    
    # Check current status
    if secrets_manager.validate_openai_key():
        print("✅ OpenAI API key is already configured")
        return
    
    print("⚠️  OpenAI API key not found or invalid")
    print("\nOptions:")
    print("1. Set environment variable: export OPENAI_API_KEY='your-key'")
    print("2. Create .env file with your API key")
    print("3. Set it interactively now")
    
    choice = input("\nChoose option (1/2/3) or press Enter to skip: ").strip()
    
    if choice == "3":
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key and secrets_manager._validate_api_key(api_key):
            secrets_manager.set_secret('OPENAI_API_KEY', api_key)
            print("✅ API key set successfully!")
        else:
            print("❌ Invalid API key format")
    elif choice == "2":
        print("\n💡 Create a .env file with:")
        print("OPENAI_API_KEY=your-actual-api-key-here")
    elif choice == "1":
        print("\n💡 Set environment variable:")
        print("export OPENAI_API_KEY='your-actual-api-key-here'")
    
    print("\n🔄 Reload the application after setting your API key")


if __name__ == "__main__":
    setup_secrets_interactive()
