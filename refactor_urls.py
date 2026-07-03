import re

with open('backend/app/services/upstox_service.py', 'r', encoding='utf8') as f:
    code = f.read()

# Replace hardcoded v2 URLs
code = re.sub(r'"https://api\.upstox\.com/v2([^"]*)"', r'f"{self.base_url_v2}\1"', code)
# Replace hardcoded v3 URLs
code = re.sub(r'"https://api\.upstox\.com/v3([^"]*)"', r'f"{self.base_url_v3}\1"', code)
# Fix cases where f-string was already used: f'f"{self.base_url...
code = code.replace('f"f"{', 'f"{')

# Now inject the __init__ logic
init_search = 'self.base_url = "https://api.upstox.com/v2"'
init_replace = '''        use_sub_server = os.getenv("USE_SUB_SERVER", "false").lower() == "true"
        sub_server_url = os.getenv("SUB_SERVER_URL", "").rstrip("/")
        
        if use_sub_server and sub_server_url:
            self.base_url_v2 = f"{sub_server_url}/v2"
            self.base_url_v3 = f"{sub_server_url}/v3"
            self.host_base = sub_server_url
        else:
            self.base_url_v2 = "https://api.upstox.com/v2"
            self.base_url_v3 = "https://api.upstox.com/v3"
            self.host_base = "https://api.upstox.com"'''
code = code.replace(init_search, init_replace)

# Now inject the configuration.host
config_search = '''        # Configuration for Upstox Client
        self.configuration = config.Configuration()'''
config_replace = '''        # Configuration for Upstox Client
        self.configuration = config.Configuration()
        self.configuration.host = self.host_base'''
code = code.replace(config_search, config_replace)

with open('backend/app/services/upstox_service.py', 'w', encoding='utf8') as f:
    f.write(code)
print('Refactoring done!')
