
try:
    with open("inspection_result.txt", "w") as f:
        import TradeMaster
        f.write("Import 'TradeMaster' SUCCESS\n")
        f.write(f"Version: {getattr(TradeMaster, '__version__', 'Unknown')}\n")
        f.write(f"Dir: {dir(TradeMaster)}\n")
        
        # Check for common client classes
        for attr in dir(TradeMaster):
            if not attr.startswith("_"):
                obj = getattr(TradeMaster, attr)
                if isinstance(obj, type):
                    f.write(f"\nClass: {attr}\n")
                    f.write(f"Dir: {dir(obj)}\n")
                    # Check __init__ signature
                    import inspect
                    try:
                        f.write(f"  __init__: {inspect.signature(obj.__init__)}\n")
                    except:
                        pass
        
        # Also check 'Connect' specifically if not found above
        if 'Connect' in dir(TradeMaster):
            f.write("\nConnect class found!\n")

except ImportError as e:
    with open("inspection_result.txt", "w") as f:
        f.write(f"Import 'TradeMaster' FAILED: {e}\n")
