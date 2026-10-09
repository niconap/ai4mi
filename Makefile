red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@

PYTHON ?= python
MODEL ?= enet
EPOCHS ?= 50
SEED ?= 1000
WORKERS ?= 2
THREADS ?= 4
GPU ?= 0
RESULTS_ROOT ?= results/local
EXPERIMENTS ?= B0 B1 B2
VARIANTS ?= float zscore crop native_patches spacing sampling context dual

LOCAL_DATA_float = CT_FULL_P0
LOCAL_DATA_zscore = CT_FULL_P0
LOCAL_DATA_crop = CT_FULL_crop
LOCAL_DATA_native_patches = CT_FULL_native_patches
LOCAL_DATA_spacing = CT_FULL_spacing
LOCAL_DATA_sampling = CT_FULL_P0
LOCAL_DATA_context = CT_FULL_P0
LOCAL_DATA_dual = CT_FULL_dual
LOCAL_ARGS_zscore = --normalization zscore
LOCAL_ARGS_sampling = --slice-sampling organ --organ-sampling-prob 0.5
LOCAL_ARGS_context = --context 3

.PHONY: train-local
train-local: $(foreach experiment,$(EXPERIMENTS),$(foreach variant,$(VARIANTS),train-$(experiment)-$(variant)))

define LOCAL_TRAIN_RULE
.PHONY: train-$(1)-$(2)
train-$(1)-$(2):
	@test -f data/$$(LOCAL_DATA_$(2))/preprocessing.json || { echo "Missing prepared dataset: data/$$(LOCAL_DATA_$(2))/preprocessing.json"; exit 1; }
	OMP_NUM_THREADS=$$(THREADS) MKL_NUM_THREADS=$$(THREADS) OPENBLAS_NUM_THREADS=$$(THREADS) $$(PYTHON) main.py \
		--dataset SEGTHOR_FULL --data-root data/$$(LOCAL_DATA_$(2)) \
		--model $$(MODEL) --experiment $(1) --mode full --epochs $$(EPOCHS) \
		--seed $$(SEED) --workers $$(WORKERS) $$(LOCAL_ARGS_$(2)) \
		--dest $$(RESULTS_ROOT)/$$(MODEL)/$(1)/$(2) $$(if $$(filter 1,$$(GPU)),--gpu)
endef

$(foreach experiment,$(EXPERIMENTS),$(foreach variant,$(VARIANTS),$(eval $(call LOCAL_TRAIN_RULE,$(experiment),$(variant)))))


# Extraction and slicing for Segthor
## Original one
data/segthor_part1: data/segthor_part1.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_part1.sha256
	unzip -q $<
	rm -f $@/.DS_STORE

data/SEGTHOR:
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_part1 --dest_dir $@_tmp \
		--shape 256 256 --retain 5
	mv $@_tmp $@
