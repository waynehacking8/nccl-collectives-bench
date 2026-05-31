GPUS ?= 4
.PHONY: setup sweep analyze
setup:   ; bash scripts/setup_nccl_tests.sh
sweep:   ; bash scripts/run_sweep.sh $(GPUS)
analyze: ; python analysis/parse.py results/*.txt && python analysis/plot.py && python analysis/theoretical.py
