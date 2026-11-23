//! `POST /v1/audio/transcriptions`: an OpenAI-shaped speech-to-text passthrough.
//!
//! Modelled on the embeddings passthrough (KAN-529): no chat pipeline, no routing,
//! the model named by the caller (after `model_aliases`) is the model that runs, and
//! the provider is called directly with its own credential. The body is
//! `multipart/form-data`; it is parsed only far enough to read (and, when the
//! provider's bare model name differs, rewrite) the `model` part. Every other byte,
//! including the audio, is forwarded untouched under the original boundary.
//!
//! Audio bytes and transcript text are never logged: only model, status, sizes and
//! latency.

use bytes::{Bytes, BytesMut};
use common::configuration::ModelAlias;
use common::llm_providers::LlmProviders;
use http_body_util::combinators::BoxBody;
use http_body_util::BodyExt;
use hyper::header::{HeaderValue, CONTENT_LENGTH, CONTENT_TYPE};
use hyper::{Request, Response, StatusCode};
use opentelemetry::trace::get_active_span;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, info_span, warn, Instrument};

use crate::app_state::AppState;
use crate::handlers::embeddings::{
    json_response, upstream_headers, upstream_url_for, EmbeddingsError as TranscriptionsError,
};
use crate::handlers::llm::resolve_model_alias;
use crate::handlers::{extract_request_id, take_attribution_from_headers, Attribution};
use crate::metrics as bs_metrics;
use crate::metrics::labels as metric_labels;
use crate::tracing::{llm as tracing_llm, operation_component, set_service_name};

/// OpenAI's own upload limit for transcriptions.
pub const MAX_BODY_BYTES: usize = 25 * 1024 * 1024;
const OPERATION: &str = "audio/transcriptions";

/// Everything needed to send the request upstream, and the labels to meter it under.
#[derive(Debug)]
pub struct UpstreamRequest {
    pub url: String,
    pub headers: hyper::HeaderMap,
    pub body: Bytes,
    pub provider_label: String,
    pub model_label: String,
}

fn bad_request(message: &str) -> TranscriptionsError {
    TranscriptionsError::new(
        StatusCode::BAD_REQUEST,
        message.to_string(),
        Some("model"),
        None,
    )
}

fn too_large() -> TranscriptionsError {
    TranscriptionsError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        format!("The request body exceeds the {MAX_BODY_BYTES} byte limit"),
        Some("file"),
        Some("request_too_large"),
    )
}

/// The `boundary` parameter of a `multipart/form-data` Content-Type.
pub fn boundary_from_content_type(content_type: &str) -> Option<String> {
    let mut parts = content_type.split(';');
    if !parts
        .next()?
        .trim()
        .eq_ignore_ascii_case("multipart/form-data")
    {
        return None;
    }
    parts.find_map(|p| {
        let (k, v) = p.trim().split_once('=')?;
        if !k.trim().eq_ignore_ascii_case("boundary") {
            return None;
        }
        let v = v.trim().trim_matches('"');
        (!v.is_empty()).then(|| v.to_string())
    })
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from > hay.len() || hay.len() - from < needle.len() {
        return None;
    }
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| i + from)
}

/// Byte range of the `model` part's value, found without decoding the other parts.
pub fn model_value_range(body: &[u8], boundary: &str) -> Option<std::ops::Range<usize>> {
    let delim = format!("--{boundary}").into_bytes();
    let mut pos = find(body, &delim, 0)?;
    loop {
        let after = pos + delim.len();
        // Closing delimiter `--boundary--`.
        if body.get(after..after + 2) == Some(b"--") {
            return None;
        }
        let headers_start = find(body, b"\r\n", after)? + 2;
        let headers_end = find(body, b"\r\n\r\n", headers_start.saturating_sub(2))? + 4;
        let headers = String::from_utf8_lossy(&body[headers_start - 2..headers_end - 4])
            .to_ascii_lowercase()
            .replace(' ', "");
        let value_end = find(body, &[b"\r\n".as_slice(), &delim].concat(), headers_end)?;
        if headers.contains("name=\"model\"") {
            return Some(headers_end..value_end);
        }
        pos = value_end + 2;
    }
}

/// Resolve the multipart `model` against aliases and providers and build the upstream
/// request. Pure: no network, no globals.
pub fn build_upstream_request(
    body: &[u8],
    content_type: &str,
    providers: &LlmProviders,
    aliases: &Option<HashMap<String, ModelAlias>>,
) -> Result<UpstreamRequest, TranscriptionsError> {
    let boundary = boundary_from_content_type(content_type)
        .ok_or_else(|| bad_request("Content-Type must be multipart/form-data with a boundary"))?;
    let range = model_value_range(body, &boundary)
        .ok_or_else(|| bad_request("Missing required parameter: 'model' form field"))?;
    let requested = std::str::from_utf8(&body[range.clone()])
        .ok()
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| bad_request("Missing required parameter: 'model' must be a string"))?
        .to_string();

    let resolved = resolve_model_alias(&requested, aliases);
    let provider = providers.get(&resolved).ok_or_else(|| {
        warn!(model = %resolved, "transcriptions model not found in configured providers");
        TranscriptionsError::new(
            StatusCode::NOT_FOUND,
            format!("The model '{resolved}' does not exist or is not configured"),
            Some("model"),
            Some("model_not_found"),
        )
    })?;

    let bare_model = provider.model.clone().unwrap_or_else(|| {
        resolved
            .split_once('/')
            .map(|(_, m)| m.to_string())
            .unwrap_or_else(|| resolved.clone())
    });

    let url = upstream_url_for(&provider, &bare_model, OPERATION)
        .map_err(|m| TranscriptionsError::new(StatusCode::INTERNAL_SERVER_ERROR, m, None, None))?;
    let mut headers = upstream_headers(&provider)?;
    // The client's Content-Type, boundary included: the body is forwarded as it came.
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_str(content_type).map_err(|_| bad_request("Invalid Content-Type"))?,
    );

    // Embeddings rewrites `model` to the provider's bare name; so does this, in the
    // model part only. Unchanged bodies are forwarded as the same bytes.
    let body = if requested == bare_model && &body[range.clone()] == requested.as_bytes() {
        Bytes::copy_from_slice(body)
    } else {
        let mut out = BytesMut::with_capacity(body.len() + bare_model.len());
        out.extend_from_slice(&body[..range.start]);
        out.extend_from_slice(bare_model.as_bytes());
        out.extend_from_slice(&body[range.end..]);
        out.freeze()
    };

    let provider_label = bs_metrics::provider_label(providers, &resolved);
    Ok(UpstreamRequest {
        url,
        headers,
        body,
        provider_label,
        model_label: bare_model,
    })
}

/// `(input_tokens, output_tokens)` from a transcription response with a `usage` object.
pub fn tokens_from_response(body: &[u8]) -> Option<(u64, u64)> {
    let v = serde_json::from_slice::<Value>(body).ok()?;
    let usage = v.get("usage")?;
    Some((
        usage.get("input_tokens")?.as_u64()?,
        usage.get("output_tokens")?.as_u64()?,
    ))
}

/// Meter a successful response: input as prompt, output as completion, or the
/// usage-missing counter (whisper, `response_format=text`). One usage-log line when
/// tokens are reported.
pub fn record_transcription_usage(
    provider: &str,
    model: &str,
    attribution: &Attribution,
    usage_log: Option<&crate::usage_log::UsageLog>,
    response_body: &[u8],
) {
    match tokens_from_response(response_body) {
        Some((input, output)) => {
            for (kind, n) in [
                (metric_labels::TOKEN_KIND_PROMPT, input),
                (metric_labels::TOKEN_KIND_COMPLETION, output),
            ] {
                bs_metrics::record_llm_tokens(
                    provider,
                    model,
                    &attribution.org,
                    &attribution.service,
                    kind,
                    n,
                );
            }
            if let Some(log) = usage_log {
                log.record(
                    crate::usage_log::ENDPOINT_TRANSCRIPTIONS,
                    attribution,
                    provider,
                    model,
                    &crate::usage_log::TokenCounts {
                        prompt: Some(input),
                        completion: Some(output),
                        ..Default::default()
                    },
                );
            }
        }
        None => bs_metrics::record_llm_tokens_usage_missing(
            provider,
            model,
            &attribution.org,
            &attribution.service,
        ),
    }
}

/// Read the whole body, giving up with 413 once it passes the cap.
async fn read_capped(
    mut body: hyper::body::Incoming,
    limit: usize,
) -> Result<Result<Bytes, TranscriptionsError>, hyper::Error> {
    let mut buf = BytesMut::new();
    while let Some(frame) = body.frame().await {
        if let Ok(data) = frame?.into_data() {
            if buf.len() + data.len() > limit {
                return Ok(Err(too_large()));
            }
            buf.extend_from_slice(&data);
        }
    }
    Ok(Ok(buf.freeze()))
}

fn upstream_failure(message: String) -> Response<BoxBody<Bytes, hyper::Error>> {
    TranscriptionsError {
        status: StatusCode::BAD_GATEWAY,
        body: json!({
            "error": {
                "message": message,
                "type": "api_error",
                "param": null,
                "code": "upstream_unreachable",
            }
        }),
    }
    .into_response()
}

pub async fn transcriptions(
    request: Request<hyper::body::Incoming>,
    state: Arc<AppState>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, hyper::Error> {
    let request_id = extract_request_id(&request);
    let request_span = info_span!(
        "llm",
        component = "llm",
        request_id = %request_id,
        http.method = %request.method(),
        http.path = %request.uri().path(),
        llm.model = tracing::field::Empty,
    );
    transcriptions_inner(request, state)
        .instrument(request_span)
        .await
}

async fn transcriptions_inner(
    request: Request<hyper::body::Incoming>,
    state: Arc<AppState>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, hyper::Error> {
    set_service_name(operation_component::LLM);

    let mut client_headers = request.headers().clone();
    let attribution =
        take_attribution_from_headers(&mut client_headers, state.attribution_header.as_deref());
    let content_type = client_headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let declared = request
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > MAX_BODY_BYTES) {
        return Ok(too_large().into_response());
    }
    let body = match read_capped(request.into_body(), MAX_BODY_BYTES).await? {
        Ok(b) => b,
        Err(e) => return Ok(e.into_response()),
    };

    let upstream = {
        let providers = state.llm_providers.read().await;
        build_upstream_request(&body, &content_type, &providers, &state.model_aliases)
    };
    let upstream = match upstream {
        Ok(u) => u,
        Err(e) => return Ok(e.into_response()),
    };

    tracing::Span::current().record(tracing_llm::MODEL_NAME, upstream.model_label.as_str());
    get_active_span(|span| {
        span.set_attribute(opentelemetry::KeyValue::new(
            tracing_llm::PROVIDER,
            upstream.provider_label.clone(),
        ));
    });

    let request_bytes = upstream.body.len();
    let started = std::time::Instant::now();
    let response = match state
        .http_client
        .post(&upstream.url)
        .headers(upstream.headers)
        .body(upstream.body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(err) => {
            warn!(error = %err, "transcriptions upstream request failed");
            bs_metrics::record_llm_upstream(
                &upstream.provider_label,
                &upstream.model_label,
                0,
                bs_metrics::llm_error_class_from_reqwest(&err),
                started.elapsed(),
            );
            return Ok(upstream_failure(format!(
                "Failed to reach the transcription provider: {err}"
            )));
        }
    };

    let status = response.status();
    let response_type = response.headers().get(CONTENT_TYPE).cloned();
    let response_body = match response.bytes().await {
        Ok(b) => b,
        Err(err) => {
            warn!(error = %err, "transcriptions upstream body read failed");
            bs_metrics::record_llm_upstream(
                &upstream.provider_label,
                &upstream.model_label,
                status.as_u16(),
                bs_metrics::llm_error_class_from_reqwest(&err),
                started.elapsed(),
            );
            return Ok(upstream_failure(format!(
                "Failed to read the transcription provider response: {err}"
            )));
        }
    };

    bs_metrics::record_llm_upstream(
        &upstream.provider_label,
        &upstream.model_label,
        status.as_u16(),
        metric_labels::LLM_ERR_NONE,
        started.elapsed(),
    );
    info!(
        model = %upstream.model_label,
        status = status.as_u16(),
        request_bytes,
        response_bytes = response_body.len(),
        latency_ms = started.elapsed().as_millis() as u64,
        "transcriptions upstream response"
    );
    if status.is_success() {
        record_transcription_usage(
            &upstream.provider_label,
            &upstream.model_label,
            &attribution,
            state.usage_log.as_ref(),
            &response_body,
        );
        if let Some((input, output)) = tokens_from_response(&response_body) {
            get_active_span(|span| {
                span.set_attribute(opentelemetry::KeyValue::new(
                    tracing_llm::PROMPT_TOKENS,
                    input as i64,
                ));
                span.set_attribute(opentelemetry::KeyValue::new(
                    tracing_llm::COMPLETION_TOKENS,
                    output as i64,
                ));
            });
        }
    }

    // Plain-text formats (srt, vtt, text) keep the provider's content type.
    let mut out = json_response(status, response_body);
    if let Some(t) = response_type {
        out.headers_mut().insert(CONTENT_TYPE, t);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::capture::Capture;
    use crate::usage_log::UsageLog;
    use common::configuration::{LlmProvider, LlmProviderType};

    const BOUNDARY: &str = "----fikarBoundary42";

    fn azure() -> LlmProvider {
        LlmProvider {
            name: "azure_openai/gpt-4o-mini-transcribe".to_string(),
            provider_interface: LlmProviderType::AzureOpenAI,
            access_key: Some("azure-secret".to_string()),
            model: Some("gpt-4o-mini-transcribe".to_string()),
            endpoint: Some("fikar-openai.openai.azure.com".to_string()),
            port: Some(443),
            default: None,
            ..LlmProvider::default()
        }
    }

    fn openai() -> LlmProvider {
        LlmProvider {
            name: "openai/whisper-1".to_string(),
            provider_interface: LlmProviderType::OpenAI,
            access_key: Some("oai-secret".to_string()),
            model: Some("whisper-1".to_string()),
            endpoint: Some("api.openai.com".to_string()),
            port: Some(443),
            default: None,
            ..LlmProvider::default()
        }
    }

    fn providers(list: Vec<LlmProvider>) -> LlmProviders {
        LlmProviders::try_from(list).unwrap()
    }

    fn content_type() -> String {
        format!("multipart/form-data; boundary={BOUNDARY}")
    }

    /// A multipart body with binary "audio" (including CRLF and boundary-like bytes).
    fn multipart(model: Option<&str>) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.webm\"\r\nContent-Type: audio/webm\r\n\r\n"
            )
            .as_bytes(),
        );
        b.extend_from_slice(&[0, 255, 13, 10, 45, 45, 1, 2, 3, 13, 10]);
        b.extend_from_slice(b"\r\n");
        if let Some(m) = model {
            b.extend_from_slice(
                format!(
                    "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{m}\r\n"
                )
                .as_bytes(),
            );
        }
        b.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n--{BOUNDARY}--\r\n"
            )
            .as_bytes(),
        );
        b
    }

    fn build(body: &[u8], p: &LlmProviders) -> Result<UpstreamRequest, TranscriptionsError> {
        build_upstream_request(body, &content_type(), p, &None)
    }

    #[test]
    fn azure_url_and_unchanged_body() {
        let p = providers(vec![azure()]);
        let body = multipart(Some("gpt-4o-mini-transcribe"));
        let up = build(&body, &p).unwrap();
        assert_eq!(
            up.url,
            "https://fikar-openai.openai.azure.com/openai/deployments/gpt-4o-mini-transcribe/audio/transcriptions?api-version=2025-01-01-preview"
        );
        assert_eq!(up.body.as_ref(), body.as_slice());
        assert_eq!(
            up.headers.get("content-type").unwrap(),
            content_type().as_str()
        );
        assert_eq!(up.headers.get("api-key").unwrap(), "azure-secret");
        assert_eq!(up.provider_label, "azure_openai");
        assert_eq!(up.model_label, "gpt-4o-mini-transcribe");
    }

    #[test]
    fn openai_url() {
        let p = providers(vec![openai()]);
        let up = build(&multipart(Some("whisper-1")), &p).unwrap();
        assert_eq!(up.url, "https://api.openai.com/v1/audio/transcriptions");
        assert_eq!(
            up.headers.get("authorization").unwrap(),
            "Bearer oai-secret"
        );
        let mut prov = openai();
        prov.base_url_path_prefix = Some("/openai/v1".to_string());
        prov.endpoint = Some("x.openai.azure.com".to_string());
        assert_eq!(
            upstream_url_for(&prov, "whisper-1", OPERATION).unwrap(),
            "https://x.openai.azure.com/openai/v1/audio/transcriptions"
        );
    }

    #[test]
    fn model_part_is_rewritten_alone_for_prefixed_and_aliased_names() {
        let p = providers(vec![azure()]);
        let original = multipart(Some("azure_openai/gpt-4o-mini-transcribe"));
        let up = build(&original, &p).unwrap();
        let expected = multipart(Some("gpt-4o-mini-transcribe"));
        assert_eq!(up.body.as_ref(), expected.as_slice());

        let aliases = Some(HashMap::from([(
            "stt".to_string(),
            ModelAlias {
                target: "azure_openai/gpt-4o-mini-transcribe".to_string(),
            },
        )]));
        let up =
            build_upstream_request(&multipart(Some("stt")), &content_type(), &p, &aliases).unwrap();
        assert_eq!(up.body.as_ref(), expected.as_slice());
    }

    #[test]
    fn missing_model_or_boundary_is_400() {
        let p = providers(vec![azure()]);
        for body in [multipart(None), multipart(Some("")), b"junk".to_vec()] {
            let err = build(&body, &p).unwrap_err();
            assert_eq!(err.status, StatusCode::BAD_REQUEST);
            assert_eq!(err.body["error"]["type"], "invalid_request_error");
        }
        let err = build_upstream_request(&multipart(Some("x")), "application/json", &p, &None)
            .unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn unknown_model_is_404() {
        let p = providers(vec![azure()]);
        let err = build(&multipart(Some("nope")), &p).unwrap_err();
        assert_eq!(err.status, StatusCode::NOT_FOUND);
        assert_eq!(err.body["error"]["type"], "invalid_request_error");
        assert_eq!(err.body["error"]["code"], "model_not_found");
    }

    #[test]
    fn oversized_is_413() {
        let e = too_large();
        assert_eq!(e.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(MAX_BODY_BYTES, 25 * 1024 * 1024);
        assert_eq!(e.body["error"]["code"], "request_too_large");
    }

    #[test]
    fn boundary_parsing() {
        assert_eq!(
            boundary_from_content_type("multipart/form-data; boundary=\"abc\"; charset=x"),
            Some("abc".to_string())
        );
        assert_eq!(boundary_from_content_type("multipart/form-data"), None);
        assert_eq!(boundary_from_content_type("text/plain; boundary=a"), None);
    }

    /// Wire check: Fikar headers never reach the provider, the multipart body and
    /// boundary arrive intact.
    #[tokio::test]
    async fn fikar_headers_stripped_and_body_forwarded_over_the_wire() {
        let mut server = mockito::Server::new_async().await;
        let addr = server.host_with_port();
        let (host, port) = addr.split_once(':').unwrap();
        let mut prov = openai();
        prov.endpoint = Some(host.to_string());
        prov.port = Some(port.parse().unwrap());
        let p = providers(vec![prov]);
        let body = multipart(Some("whisper-1"));
        let up = build(&body, &p).unwrap();

        let mock = server
            .mock("POST", "/v1/audio/transcriptions")
            .match_header("authorization", "Bearer oai-secret")
            .match_header("content-type", content_type().as_str())
            .match_header("x-fikar-org", mockito::Matcher::Missing)
            .match_header("x-fikar-service", mockito::Matcher::Missing)
            .match_header("x-fikar-user", mockito::Matcher::Missing)
            .match_header("x-fikar-execution", mockito::Matcher::Missing)
            .match_body(mockito::Matcher::from(body.clone()))
            .with_status(200)
            .with_body(RESPONSE)
            .create_async()
            .await;

        let mut h = hyper::HeaderMap::new();
        for k in [
            "x-fikar-org",
            "x-fikar-service",
            "x-fikar-user",
            "x-fikar-execution",
        ] {
            h.insert(k, HeaderValue::from_static("v"));
        }
        take_attribution_from_headers(&mut h, Some("x-fikar-org"));
        assert!(h.is_empty());
        // What the handler sends is `up.headers` only, never the client's.
        let resp = reqwest::Client::new()
            .post(&up.url)
            .headers(up.headers)
            .body(up.body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        mock.assert_async().await;
    }

    const RESPONSE: &str = r#"{"text":"hello there","usage":{"type":"tokens","input_tokens":29,"input_token_details":{"text_tokens":0,"audio_tokens":29},"output_tokens":7,"total_tokens":36}}"#;

    fn attribution() -> Attribution {
        Attribution {
            org: "acme".into(),
            service: "chat".into(),
            user: "42".into(),
            execution: "turn-1".into(),
        }
    }

    #[test]
    fn usage_is_metered_or_flagged_missing() {
        assert_eq!(tokens_from_response(RESPONSE.as_bytes()), Some((29, 7)));
        let cap = Capture::default();
        metrics::with_local_recorder(&cap, || {
            record_transcription_usage(
                "azure_openai",
                "gpt-4o-mini-transcribe",
                &attribution(),
                None,
                RESPONSE.as_bytes(),
            );
            // whisper json, plain text, and garbage: forwarded, flagged, no panic.
            for body in [br#"{"text":"hi"}"#.as_slice(), b"hello world", b""] {
                record_transcription_usage(
                    "azure_openai",
                    "gpt-4o-mini-transcribe",
                    &attribution(),
                    None,
                    body,
                );
            }
        });
        let got = cap.0.lock().unwrap().clone();
        let key = |kind: &str| {
            format!("brightstaff_llm_tokens_total{{kind={kind},model=gpt-4o-mini-transcribe,org=acme,provider=azure_openai,service=chat}}")
        };
        assert_eq!(got.get(&key("prompt")), Some(&29));
        assert_eq!(got.get(&key("completion")), Some(&7));
        assert_eq!(
            got.get("brightstaff_llm_tokens_usage_missing_total{model=gpt-4o-mini-transcribe,org=acme,provider=azure_openai,service=chat}"),
            Some(&3)
        );
    }

    #[tokio::test]
    async fn metered_transcription_writes_exactly_one_line() {
        let dir = tempfile::tempdir().unwrap();
        let log = UsageLog::start(dir.path().to_path_buf(), "pinned");
        let a = attribution();
        record_transcription_usage(
            "azure_openai",
            "gpt-4o-mini-transcribe",
            &a,
            Some(&log),
            RESPONSE.as_bytes(),
        );
        record_transcription_usage(
            "azure_openai",
            "gpt-4o-mini-transcribe",
            &a,
            Some(&log),
            b"plain text",
        );
        log.flush().await;
        let files: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(files.len(), 1);
        let text = std::fs::read_to_string(files[0].as_ref().unwrap().path()).unwrap();
        assert_eq!(text.lines().count(), 1);
        let v: Value = serde_json::from_str(text.trim_end()).unwrap();
        assert_eq!(v["endpoint"], "transcriptions");
        assert_eq!(v["org"], "acme");
        assert_eq!(v["service"], "chat");
        assert_eq!(v["user"], "42");
        assert_eq!(v["model"], "gpt-4o-mini-transcribe");
        assert_eq!(v["prompt"], 29);
        assert_eq!(v["completion"], 7);
    }
}
