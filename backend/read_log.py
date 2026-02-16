import os

def main():
    if os.path.exists("backend_debug.log"):
        try:
            with open("backend_debug.log", "r", encoding="utf-16") as f:
                content = f.read()
                print(content[-2000:]) # Last 2000 chars
        except Exception as e:
            # Try utf-8 if 16 fails
            try:
                with open("backend_debug.log", "r", encoding="utf-8") as f:
                     content = f.read()
                     print(content[-2000:])
            except Exception as e2:
                print(f"Error reading log: {e}, {e2}")
    else:
        print("Log file not found")

if __name__ == "__main__":
    main()
