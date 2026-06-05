#!/bin/bash

REMOTE_USER="e19275"
REMOTE_HOST="tesla"
REMOTE_BASE="/new-home/e19/e19275/NILM/NILMFormer/result"
LOCAL_BASE="result"

rsync -av \
    --include="*/" \
    --include="CNN1D_*.pt" \
    --exclude="*" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/" \
    "${LOCAL_BASE}/"
