@echo off
REM For Split
docker build -t traffic-split -f Split/Dockerfile .

REM For Tracking
docker build -t traffic-tracking -f Tracking/Dockerfile .
