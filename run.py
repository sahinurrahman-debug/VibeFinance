"""Start the VibeFinance dashboard:  python run.py [--port 8000] [--reset]"""
import argparse
import os
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--reset", action="store_true", help="delete and re-seed the SQLite database")
    args = ap.parse_args()
    if args.reset:
        from app import config
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
            print("database reset")
    import uvicorn
    print(f"VibeFinance -> http://{args.host}:{args.port}")
    uvicorn.run("app.server:app", host=args.host, port=args.port, log_level="warning")
