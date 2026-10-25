use crate::apis::anthropic::{MessagesContentBlock, MessagesImageSource};
use crate::apis::openai::{ContentPart, FunctionCall, ImageUrl, Message, MessageContent, ToolCall};
use crate::clients::TransformError;
use log::warn;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

pub trait ExtractText {
    fn extract_text(&self) -> String;
}

/// Trait for utility functions on content collections
pub trait ContentUtils<T> {
    fn extract_tool_calls(&self) -> Result<Option<Vec<ToolCall>>, TransformError>;
    fn split_for_openai(&self) -> Result<SplitForOpenAIResult, TransformError>;
}

pub type SplitForOpenAIResult = (Vec<ContentPart>, Vec<ToolCall>, Vec<(String, String, bool)>);

/// Helper to create a current unix timestamp
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// Content Utilities
impl ContentUtils<ToolCall> for Vec<MessagesContentBlock> {
    fn extract_tool_calls(&self) -> Result<Option<Vec<ToolCall>>, TransformError> {
        let mut tool_calls = Vec::new();

        for block in self {
            match block {
                MessagesContentBlock::ToolUse {
                    id, name, input, ..
                }
                | MessagesContentBlock::ServerToolUse { id, name, input }
                | MessagesContentBlock::McpToolUse { id, name, input } => {
                    let arguments = serde_json::to_string(&input)?;
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments,
                        },
                    });
                }
                _ => continue,
            }
        }

        Ok(if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        })
    }

    fn split_for_openai(
        &self,
    ) -> Result<(Vec<ContentPart>, Vec<ToolCall>, Vec<(String, String, bool)>), TransformError>
    {
        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();

        for block in self {
            match block {
                MessagesContentBlock::Text { text, .. } => {
                    content_parts.push(ContentPart::Text {
                        text: text.clone(),
                        cache_control: None,
                    });
                }
                MessagesContentBlock::Image { source } => {
                    let url = convert_image_source_to_url(source);
                    content_parts.push(ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url,
                            detail: Some("auto".to_string()),
                        },
                    });
                }
                MessagesContentBlock::ToolUse {
                    id, name, input, ..
                }
                | MessagesContentBlock::ServerToolUse { id, name, input }
                | MessagesContentBlock::McpToolUse { id, name, input } => {
                    let arguments = serde_json::to_string(&input)?;
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments,
                        },
                    });
                }
                MessagesContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    ..
                } => {
                    let result_text = content.extract_text();
                    tool_results.push((
                        tool_use_id.clone(),
                        result_text,
                        is_error.unwrap_or(false),
                    ));
                }
                MessagesContentBlock::WebSearchToolResult {
                    tool_use_id,
                    content,
                    is_error,
                }
                | MessagesContentBlock::CodeExecutionToolResult {
                    tool_use_id,
                    content,
                    is_error,
                }
                | MessagesContentBlock::McpToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let result_text = content.extract_text();
                    tool_results.push((
                        tool_use_id.clone(),
                        result_text,
                        is_error.unwrap_or(false),
                    ));
                }
                _ => {
                    // Skip unsupported content types
                    continue;
                }
            }
        }

        Ok((content_parts, tool_calls, tool_results))
    }
}

/// Convert image source to URL
pub fn convert_image_source_to_url(source: &MessagesImageSource) -> String {
    match source {
        MessagesImageSource::Base64 { media_type, data } => {
            format!("data:{};base64,{}", media_type, data)
        }
        MessagesImageSource::Url { url } => url.clone(),
    }
}

/// Convert image URL to Anthropic image source
fn convert_image_url_to_source(image_url: &ImageUrl) -> MessagesImageSource {
    if image_url.url.starts_with("data:") {
        // Parse data URL
        let parts: Vec<&str> = image_url.url.splitn(2, ',').collect();
        if parts.len() == 2 {
            let header = parts[0];
            let data = parts[1];
            let media_type = header
                .strip_prefix("data:")
                .and_then(|s| s.split(';').next())
                .unwrap_or("image/jpeg")
                .to_string();

            MessagesImageSource::Base64 {
                media_type,
                data: data.to_string(),
            }
        } else {
            MessagesImageSource::Url {
                url: image_url.url.clone(),
            }
        }
    } else {
        MessagesImageSource::Url {
            url: image_url.url.clone(),
        }
    }
}

/// Convert OpenAI message to Anthropic content blocks
pub fn convert_openai_message_to_anthropic_content(
    message: &Message,
) -> Result<Vec<MessagesContentBlock>, TransformError> {
    let mut blocks = Vec::new();

    // Handle regular content
    match &message.content {
        Some(MessageContent::Text(text)) if !text.is_empty() => {
            blocks.push(MessagesContentBlock::Text {
                text: text.clone(),
                cache_control: None,
            });
        }
        Some(MessageContent::Text(_)) => {}
        Some(MessageContent::Parts(parts)) => {
            for part in parts {
                match part {
                    ContentPart::Text { text, .. } => {
                        blocks.push(MessagesContentBlock::Text {
                            text: text.clone(),
                            cache_control: None,
                        });
                    }
                    ContentPart::ImageUrl { image_url } => {
                        let source = convert_image_url_to_source(image_url);
                        blocks.push(MessagesContentBlock::Image { source });
                    }
                }
            }
        }
        None => {}
    }

    // Handle tool calls
    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            let input =
                parse_tool_arguments(&tool_call.function.name, &tool_call.function.arguments);
            blocks.push(MessagesContentBlock::ToolUse {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
                cache_control: None,
            });
        }
    }

    Ok(blocks)
}

/// Parse tool-call arguments, tolerating JSON cut off by max_tokens (KAN-672).
///
/// Complete arguments parse as before. Truncated ones (`{"city":"`) are repaired to the
/// longest valid prefix with the open string and brackets closed (`{"city":""}`), and
/// anything that still fails becomes `{}`, so one cut tool call cannot fail the response.
pub fn parse_tool_arguments(tool_name: &str, raw: &str) -> Value {
    if let Ok(value) = serde_json::from_str(raw) {
        return value;
    }
    match repair_truncated_json(raw) {
        Some(value) => {
            warn!(
                "tool call arguments for {} were truncated ({} bytes); repaired to a partial object",
                tool_name,
                raw.len()
            );
            value
        }
        None => {
            warn!(
                "tool call arguments for {} are not valid JSON ({} bytes); using {{}}",
                tool_name,
                raw.len()
            );
            Value::Object(Default::default())
        }
    }
}

/// Best-effort repair of JSON that was cut off at the end.
///
/// Scans once, remembering the last point where the text so far is a complete prefix
/// (after an opening bracket or a complete value, never after a dangling key, colon or
/// comma). At the end it tries, in order: closing an open value string, closing the
/// brackets after a trailing scalar, and closing the brackets at that last good point.
fn repair_truncated_json(raw: &str) -> Option<Value> {
    fn closers(stack: &[u8]) -> String {
        stack
            .iter()
            .rev()
            .map(|&open| if open == b'{' { '}' } else { ']' })
            .collect()
    }
    fn parse(text: String) -> Option<Value> {
        serde_json::from_str(&text).ok()
    }

    let bytes = raw.as_bytes();
    let mut stack: Vec<u8> = Vec::new();
    // In an object, true until the key's colon: a string here is a key, not a value.
    let mut expect_key = false;
    let mut in_string = false;
    let mut string_is_key = false;
    // Start of an escape sequence that has not finished yet, and the \u hex digits it still needs.
    let mut escape_start: Option<usize> = None;
    let mut hex_left = 0;
    let mut in_scalar = false;
    let mut last_good: Option<(usize, Vec<u8>)> = None;

    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if let Some(start) = escape_start {
                if hex_left > 0 {
                    hex_left -= 1;
                    if hex_left == 0 {
                        escape_start = None;
                    }
                } else if b == b'u' && i == start + 1 {
                    hex_left = 4;
                } else {
                    escape_start = None;
                }
            } else if b == b'\\' {
                escape_start = Some(i);
            } else if b == b'"' {
                in_string = false;
                if !string_is_key {
                    last_good = Some((i + 1, stack.clone()));
                }
            }
            continue;
        }
        if in_scalar && matches!(b, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
            in_scalar = false;
            last_good = Some((i, stack.clone()));
        }
        match b {
            b'"' => {
                in_string = true;
                string_is_key = stack.last() == Some(&b'{') && expect_key;
            }
            b'{' | b'[' => {
                stack.push(b);
                expect_key = b == b'{';
                last_good = Some((i + 1, stack.clone()));
            }
            b'}' | b']' => {
                stack.pop();
                expect_key = false;
                last_good = Some((i + 1, stack.clone()));
            }
            b',' => expect_key = stack.last() == Some(&b'{'),
            b':' => expect_key = false,
            b' ' | b'\t' | b'\n' | b'\r' => {}
            _ => in_scalar = true,
        }
    }

    if in_string && !string_is_key {
        let cut = escape_start.unwrap_or(raw.len());
        if let Some(value) = parse(format!("{}\"{}", &raw[..cut], closers(&stack))) {
            return Some(value);
        }
    } else if in_scalar {
        if let Some(value) = parse(format!("{}{}", raw, closers(&stack))) {
            return Some(value);
        }
    }
    let (cut, stack) = last_good?;
    parse(format!("{}{}", &raw[..cut], closers(&stack)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_tool_arguments_complete_unchanged() {
        let raw = r#"{"city":"Paris","days":[1,2],"opts":{"metric":true,"note":null}}"#;
        assert_eq!(
            parse_tool_arguments("t", raw),
            serde_json::from_str::<Value>(raw).unwrap()
        );
        assert_eq!(parse_tool_arguments("t", "{}"), json!({}));
    }

    #[test]
    fn test_parse_tool_arguments_cut_string() {
        assert_eq!(
            parse_tool_arguments("t", r#"{"city":""#),
            json!({"city": ""})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"city":"Par"#),
            json!({"city": "Par"})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"city":"a\"#),
            json!({"city": "a"})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"city":"a\u00"#),
            json!({"city": "a"})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"city":"a\nb"#),
            json!({"city": "a\nb"})
        );
    }

    #[test]
    fn test_parse_tool_arguments_cut_key() {
        assert_eq!(parse_tool_arguments("t", r#"{""#), json!({}));
        assert_eq!(parse_tool_arguments("t", r#"{"ci"#), json!({}));
        assert_eq!(parse_tool_arguments("t", r#"{"city""#), json!({}));
        assert_eq!(parse_tool_arguments("t", r#"{"city":"#), json!({}));
        assert_eq!(
            parse_tool_arguments("t", r#"{"city":"Paris","da"#),
            json!({"city": "Paris"})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"city":"Paris", "days": "#),
            json!({"city": "Paris"})
        );
    }

    #[test]
    fn test_parse_tool_arguments_dangling_comma() {
        assert_eq!(
            parse_tool_arguments("t", r#"{"city":"Paris","#),
            json!({"city": "Paris"})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"days":[1,2,"#),
            json!({"days": [1, 2]})
        );
    }

    #[test]
    fn test_parse_tool_arguments_cut_scalar() {
        assert_eq!(
            parse_tool_arguments("t", r#"{"days":12"#),
            json!({"days": 12})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"ok":true"#),
            json!({"ok": true})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"a":1,"ok":tr"#),
            json!({"a": 1})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"a":1,"n":1."#),
            json!({"a": 1})
        );
    }

    #[test]
    fn test_parse_tool_arguments_nested() {
        assert_eq!(
            parse_tool_arguments("t", r#"{"loc":{"city":"Pa"#),
            json!({"loc": {"city": "Pa"}})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"loc":{"city":"Paris"},"tags":["a","b"#),
            json!({"loc": {"city": "Paris"}, "tags": ["a", "b"]})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"rows":[{"k":[1,{"x":"#),
            json!({"rows": [{"k": [1, {}]}]})
        );
        assert_eq!(
            parse_tool_arguments("t", r#"{"s":"{[\"x"#),
            json!({"s": "{[\"x"})
        );
    }

    #[test]
    fn test_parse_tool_arguments_garbage_falls_back_to_empty_object() {
        assert_eq!(parse_tool_arguments("t", ""), json!({}));
        assert_eq!(parse_tool_arguments("t", "not json"), json!({}));
        assert_eq!(parse_tool_arguments("t", "}"), json!({}));
        assert_eq!(parse_tool_arguments("t", r#"{"a":1}}"#), json!({}));
    }
}
