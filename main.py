#!/usr/bin/env python3
"""
Punto de entrada principal para el agente ETS2.
Ejecutar: python main.py
"""

import os
import sys

# Asegurar que src esté en el path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.main import main

if __name__ == "__main__":
    main()
