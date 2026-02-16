
try:
    import Ant_A3_tradehub_sdk_testing
    print("Import 'Ant_A3_tradehub_sdk_testing' SUCCESS")
    print(dir(Ant_A3_tradehub_sdk_testing))
except ImportError as e:
    print(f"Import 'Ant_A3_tradehub_sdk_testing' FAILED: {e}")

try:
    import test_alice_sdk
    print("Import 'test_alice_sdk' SUCCESS")
    print(dir(test_alice_sdk))
except ImportError as e:
    print(f"Import 'test_alice_sdk' FAILED: {e}")
    
try:
    from Ant_A3_tradehub_sdk_testing import Connect
    print("Import 'Connect' from lib SUCCESS")
except ImportError:
    print("Import 'Connect' from lib FAILED")

import pkgutil
print("Installed Modules matching 'Ant':")
for loader, module_name, is_pkg in pkgutil.iter_modules():
    if 'ant' in module_name.lower() or 'alice' in module_name.lower():
        print(module_name)
