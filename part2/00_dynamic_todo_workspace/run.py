#!/usr/bin/env python3
"""Entry point:  python3 run.py [--port 8000]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.app import main

if __name__ == "__main__":
    main()
