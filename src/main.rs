use axum::{
    extract::{Json, State},
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Router,
    http::StatusCode,
};
use futures_core::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashMap, convert::Infallible, net::SocketAddr, pin::Pin, sync::Arc};
use tokio::net::TcpListener;
// use tokio_stream::StreamExt as TokioStreamExt; // <--- FIX: Removed this line to resolve ambiguity
use tracing::{info, error, Level};
use tracing_subscriber::EnvFilter;
use anyhow::{Context, Result};
use dotenv::dotenv;
use bytes::Bytes;
use futures::{stream, StreamExt}; // We will use this trait for both .map() and .flatten()


// --- Data Structures for OpenAI API Compatibility ---
#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

// --- Application State ---
struct AppState {
    http_client: Client,
    vllm_backends: HashMap<String, String>, // model_name -> vLLM_base_url
}

// --- Custom Error Type ---
enum AppError {
    ModelNotFound(String),
    BackendRequestFailed(reqwest::Error),
    BackendRespondedError { status: StatusCode, text: String, url: String },
}

// Implement IntoResponse to convert AppError into an HTTP response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::ModelNotFound(model) => (
                StatusCode::BAD_REQUEST,
                format!("Model '{}' not found in gateway configuration.", model),
            ),
            AppError::BackendRequestFailed(e) => {
                error!("Request to backend failed: {}", e);
                (StatusCode::BAD_GATEWAY, format!("Upstream request failed: {}", e))
            }
            AppError::BackendRespondedError { status, text, url } => {
                error!("Backend at {} returned error {}: {}", url, status, text);
                (status, format!("Upstream service error: {}", text))
            }
        };

        let body = Json(json!({ "error": error_message }));
        (status, body).into_response()
    }
}

// --- Main Function ---
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for better logging control via RUST_LOG env var
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .init();

    dotenv().ok(); // Load .env file if it exists

    // Load and parse backend configuration from environment variables
    let vllm_backends_json = std::env::var("VLLM_BACKENDS")
        .context("VLLM_BACKENDS environment variable not set")?;
    let vllm_backends: HashMap<String, String> = serde_json::from_str(&vllm_backends_json)
        .context("Failed to parse VLLM_BACKENDS. Make sure it's valid JSON on a single line.")?;

    info!("Configured vLLM Backends:");
    for (model_name, url) in &vllm_backends {
        info!("  - Model: '{}' -> URL: '{}'", model_name, url);
    }

    let app_state = Arc::new(AppState {
        http_client: Client::new(),
        vllm_backends,
    });

    // Define application routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(proxy_chat)) // OpenAI compatible route
        .with_state(app_state);

    // Get listen address from environment or use default
    let addr_str = std::env::var("GATEWAY_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    let addr: SocketAddr = addr_str.parse()
        .context(format!("Invalid GATEWAY_LISTEN_ADDR format: {}", addr_str))?;

    let listener = TcpListener::bind(&addr).await
        .context(format!("Failed to bind to address: {}", addr_str))?;
    info!("🚀 Gateway listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app.into_make_service())
        .await
        .context("Server failed to start")?;

    Ok(())
}

// --- Handlers ---
async fn health_check() -> &'static str {
    "OK"
}

async fn proxy_chat(
    State(state): State<Arc<AppState>>,
    Json(mut body): Json<ChatRequest>,
) -> Result<Sse<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>>, AppError> {
    body.stream = Some(true);

    info!("Received chat request for model: {}", body.model);

    let vllm_base_url = state.vllm_backends.get(&body.model)
        .ok_or_else(|| AppError::ModelNotFound(body.model.clone()))?;

    let target_url = format!("{}/v1/chat/completions", vllm_base_url);
    info!("Routing request for model '{}' to: {}", body.model, &target_url);

    let res = state.http_client
        .post(&target_url)
        .json(&body)
        .send()
        .await
        .map_err(AppError::BackendRequestFailed)?;

    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_else(|_| "No response body".to_string());
        return Err(AppError::BackendRespondedError { status, text, url: target_url });
    }

    Ok(Sse::new(stream_response(res)))
}

// --- Stream Response Function ---
fn stream_response(
    res: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> {
    let stream = res.bytes_stream()
        .map(|chunk_result| { // Now this unambiguously uses `futures::StreamExt::map`
            let chunk: Bytes = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    let err_msg = format!("[Gateway Error: Could not read chunk from backend: {}]", e);
                    error!("{}", err_msg);
                    let event = Event::default().data(err_msg);
                    return stream::iter(vec![Ok(event)]);
                }
            };

            let text = match String::from_utf8(chunk.to_vec()) {
                 Ok(s) => s,
                 Err(e) => {
                    let err_msg = format!("[Gateway Error: Non-UTF8 data received: {}]", e);
                    error!("{}", err_msg);
                    let event = Event::default().data(err_msg);
                    return stream::iter(vec![Ok(event)]);
                 }
            };

            let events = text.lines()
                .filter_map(|line| {
                    if let Some(data) = line.strip_prefix("data: ") {
                        Some(Ok(Event::default().data(data.trim().to_string())))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            stream::iter(events)
        })
        .flatten(); // This also uses `futures::StreamExt`

    Box::pin(stream)
}