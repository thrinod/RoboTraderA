
try:
    with open("inspection_result_tradesync.txt", "w") as f:
        from TradeMaster import TradeSync
        f.write("Import 'TradeSync' SUCCESS\n")
        f.write(f"Dir: {dir(TradeSync)}\n")
        
        # Check attributes
        for attr in dir(TradeSync):
            if not attr.startswith("_"):
                obj = getattr(TradeSync, attr)
                if isinstance(obj, type):
                    f.write(f"\nClass: {attr}\n")
                    f.write(f"Dir: {dir(obj)}\n")
                    import inspect
                    try:
                        f.write(f"  __init__: {inspect.signature(obj.__init__)}\n")
                    except:
                        pass
except ImportError as e:
    with open("inspection_result_tradesync.txt", "w") as f:
        f.write(f"Import 'TradeSync' FAILED: {e}\n")
