#!/usr/bin/env bash
killall -9 rcssserver
(./bin/HFO --offense-agents 1 --defense-npcs 2 --headless --fullstat --seed 123 &) && python ./defense/agent.py