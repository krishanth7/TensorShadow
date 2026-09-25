# TensorShadow developer tasks. Run `make help` for a list.
GO        ?= go
BIN       := bin
ADDR      ?= http://localhost:8080
PKGS      := ./...

.PHONY: help build run test race cover bench lint demo load docker smoke clean models serve-onnx serve-tf live-test

help: ## Show this help
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

build: ## Build the server and CLI tools into ./bin
	$(GO) build -trimpath -o $(BIN)/ ./go_backend ./go_backend/cmd/...

run: ## Run the backend on :8080 with configs/config.yaml
	$(GO) run ./go_backend

test: ## Run all Go tests
	$(GO) test $(PKGS)

race: ## Run all Go tests with the race detector
	$(GO) test -race -count=1 $(PKGS)

cover: ## Per-package test coverage
	$(GO) test -cover $(PKGS)

bench: ## Run Go micro-benchmarks
	$(GO) test -run '^$$' -bench . -benchmem ./go_backend/internal/...

lint: ## gofmt + go vet
	@test -z "$$(gofmt -l .)" || (gofmt -l . && echo "run gofmt -w ." && exit 1)
	$(GO) vet $(PKGS)

demo: build ## Feature walkthrough against a running backend (writes docs/images)
	$(BIN)/tsdemo -addr $(ADDR) -out docs/images

load: build ## 15 s mixed load test against a running backend
	$(BIN)/tsload -addr $(ADDR) -c 64 -d 15s -scenario mixed

docker: ## Build the container image
	docker build -t tensorshadow:latest .

smoke: ## Repository structure check
	python3 smoke_test.py

clean: ## Remove build output
	rm -rf $(BIN)

models: ## Train and export the SavedModel + ONNX model (needs tensorflow, onnx, onnxruntime)
	python3 tensorflow_model/export_models.py

serve-onnx: ## ONNX Runtime model server (Open Inference Protocol v2) on :8001
	python3 tensorflow_model/onnx_server.py --port 8001

serve-tf: ## TensorFlow Serving (Docker) on :8501
	docker run --rm -p 8501:8501 -v $(CURDIR)/tensorflow_model/serving/tensorshadow:/models/tensorshadow \
		-e MODEL_NAME=tensorshadow tensorflow/serving

live-test: ## Parity test against running TF Serving (:8501) and ONNX Runtime (:8001)
	TS_TFSERVING_URL=http://localhost:8501 TS_ONNX_URL=http://localhost:8001 \
		$(GO) test -count=1 -run TestLiveModelServers -v ./go_backend/internal/inference/
