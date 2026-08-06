# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec для MuraveiVision (portable, onedir, windowed)."""

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'ultralytics',
        'customtkinter',
        'onnxruntime',
        'cv2',
        'PIL',
        'dotenv',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PySide6',
        'PySide6.QtWebEngineWidgets',
        'matplotlib',
        'pandas',
        'torch',
        'torchvision',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

import os
import ultralytics
ultralytics_dir = os.path.dirname(ultralytics.__file__)
a.datas += Tree(ultralytics_dir, prefix='ultralytics', excludes=['*.pyc'])

import customtkinter
ctk_dir = os.path.dirname(customtkinter.__file__)
a.datas += Tree(ctk_dir, prefix='customtkinter', excludes=['*.pyc'])

import onnxruntime
ort_dir = os.path.dirname(onnxruntime.__file__)
a.binaries += [(os.path.join(ort_dir, f), 'onnxruntime')
               for f in os.listdir(ort_dir)
               if f.endswith('.dll') or f.endswith('.so') or f.endswith('.pyd')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MuraveiVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MuraveiVision',
)
