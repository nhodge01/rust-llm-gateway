# Rust LLM Gateway

A high-performance, asynchronous gateway built in Rust for proxying requests to various Large Language Model backends like vLLM. This gateway is designed to be a lightweight, stable, and efficient entry point for your LLM services, providing an OpenAI-compatible API endpoint for seamless integration.

---

### ✨ Features

* **OpenAI API Compatible:** Exposes a `/v1/chat/completions` endpoint.
* **Real-time Streaming:** Uses Server-Sent Events (SSE) to stream responses word-by-word.
* **Dynamic Backend Routing:** Routes requests to different model backends based on the `model` field in the request body.
* **Asynchronous & Performant:** Built with Axum and Tokio for high concurrency and low overhead.
* **Load Tested:** Proven to be stable and efficient under concurrent loads.

### 🛠️ Technology Stack

* **Gateway:** Rust, Axum, Tokio, Serde
* **LLM Backend:** vLLM (via Docker)
* **Containerization:** Docker

---

## 🚀 Getting Started

Follow these instructions to set up and run the gateway on your local machine.

### Prerequisites

* [Rust & Cargo](https://www.rust-lang.org/tools/install)
* [Docker](https://docs.docker.com/engine/install/)
* An NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

### ⚙️ Configuration

The gateway is configured using a `.env` file in the root of the project directory.

1.  Create a file named `.env`.
2.  Copy the contents of the example below and adjust the values for your setup.

#### `.env` File Definition

```env
# The IP address and port the gateway will listen on.
GATEWAY_LISTEN_ADDR="127.0.0.1:3000"

# A JSON object defining your backend models and their URLs.
# The key is the "model name" that clients will request.
# The value is the base URL of the backend serving that model.
# IMPORTANT: This JSON must be on a single line.
VLLM_BACKENDS='{"TheBloke/Mistral-7B-Instruct-v0.2-AWQ": "http://localhost:8000"}'

# Sets the default log level for the gateway.
# Options: "info", "debug", "warn", "error".
RUST_LOG="info"