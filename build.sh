#!/usr/bin/env bash
pip install --upgrade pip
pip install --upgrade setuptools wheel

git add build.sh
git commit -m "Add build script for Render"
git push origin main

