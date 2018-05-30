#!/usr/bin/env bash

(../bin HFO --offense-agents 1 --defense-npcs 2 --headless --fullstat --seed 123 &) && python ./agent.py