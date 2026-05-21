use crate::models::anthropic::{
    ContentBlockStart, Delta, DeltaUsage, ErrorData, MessageDeltaData, MessageStartData,
    StreamEvent, Usage,
};
use crate::models::openai;
use crate::translate::core;
use serde_json::{json, Value};

#[derive(Debug)]
enum BlockState {
    Idle,
    Thinking { index: usize },
    Text { index: usize },
}

impl BlockState {
    fn current_index(&self) -> Option<usize> {
        match self {
            Self::Idle => None,
            Self::Thinking { index } | Self::Text { index } => {
                Some(*index)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub openai_index: usize,
}

#[derive(Debug)]
pub struct StreamState {
    message_id: Option<String>,
    model: Option<String>,
    fallback_model: String,
    block: BlockState,
    next_index: usize,
    message_started: bool,
    pub buffered_tool_calls: Vec<BufferedToolCall>,
}

pub fn initial_state(fallback_model: String) -> StreamState {
    StreamState {
        message_id: None,
        model: None,
        fallback_model,
        block: BlockState::Idle,
        next_index: 0,
        message_started: false,
        buffered_tool_calls: Vec::new(),
    }
}

pub fn translate_chunk(state: &mut StreamState, chunk: &openai::StreamChunk) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    if let Some(id) = &chunk.id {
        if state.message_id.is_none() {
            state.message_id = Some(id.clone());
        }
    }
    if let Some(model) = &chunk.model {
        if state.model.is_none() {
            state.model = Some(model.clone());
        }
    }

    let Some(choice) = chunk.choices.first() else {
        return events;
    };

    if !state.message_started {
        events.push(StreamEvent::MessageStart {
            message: MessageStartData {
                id: state
                    .message_id
                    .clone()
                    .unwrap_or_else(|| "msg_proxy".to_string()),
                message_type: "message".to_string(),
                role: "assistant".to_string(),
                model: state
                    .model
                    .clone()
                    .unwrap_or_else(|| state.fallback_model.clone()),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
            },
        });
        state.message_started = true;
    }

    for reasoning in [&choice.delta.reasoning, &choice.delta.reasoning_content]
        .into_iter()
        .flatten()
    {
        flush_buffered_tool_calls(&mut events, state);
        emit_reasoning(&mut events, state, reasoning);
    }

    if let Some(content) = &choice.delta.content {
        if !content.is_empty() {
            flush_buffered_tool_calls(&mut events, state);
            emit_text(&mut events, state, content);
        }
    }

    if let Some(tool_calls) = &choice.delta.tool_calls {
        if !tool_calls.is_empty() {
            close_current_block(&mut events, state);
        }
        emit_tool_calls(&mut events, state, tool_calls);
    }

    if let Some(finish_reason) = &choice.finish_reason {
        flush_buffered_tool_calls(&mut events, state);
        emit_finish(&mut events, state, finish_reason, chunk.usage.as_ref());
    }

    events
}

pub fn translate_done(state: &mut StreamState) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    flush_buffered_tool_calls(&mut events, state);
    events.push(StreamEvent::MessageStop);
    events
}

pub fn translate_error(message: String) -> Vec<StreamEvent> {
    vec![StreamEvent::Error {
        error: ErrorData {
            error_type: "stream_error".to_string(),
            message,
        },
    }]
}

fn close_current_block(events: &mut Vec<StreamEvent>, state: &mut StreamState) {
    if let Some(index) = state.block.current_index() {
        events.push(StreamEvent::ContentBlockStop { index });
        state.next_index = index + 1;
        state.block = BlockState::Idle;
    }
}

fn flush_buffered_tool_calls(events: &mut Vec<StreamEvent>, state: &mut StreamState) {
    if state.buffered_tool_calls.is_empty() {
        return;
    }

    let mut seen = Vec::new();
    let buffered = std::mem::take(&mut state.buffered_tool_calls);

    for btc in buffered {
        if btc.id.is_empty() || btc.name.is_empty() {
            continue;
        }

        let input: Value = serde_json::from_str(&btc.arguments).unwrap_or_else(|_| json!({}));
        let name = btc.name.clone();

        if seen.iter().any(|(s_name, s_input): &(String, Value)| *s_name == name && *s_input == input) {
            tracing::debug!("Filtering out duplicate streaming tool call: {} with input {:?}", name, input);
            continue;
        }
        seen.push((name, input));

        let index = state.next_index;

        events.push(StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlockStart::ToolUse {
                id: btc.id.clone(),
                name: btc.name.clone(),
            },
        });

        if !btc.arguments.is_empty() {
            events.push(StreamEvent::ContentBlockDelta {
                index,
                delta: Delta::InputJsonDelta {
                    partial_json: btc.arguments.clone(),
                },
            });
        }

        events.push(StreamEvent::ContentBlockStop { index });
        state.next_index = index + 1;
    }

    state.block = BlockState::Idle;
}

fn emit_reasoning(events: &mut Vec<StreamEvent>, state: &mut StreamState, reasoning: &str) {
    if !matches!(state.block, BlockState::Thinking { .. }) {
        close_current_block(events, state);
        let index = state.next_index;
        events.push(StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlockStart::Thinking {
                thinking: String::new(),
            },
        });
        state.block = BlockState::Thinking { index };
    }

    if let BlockState::Thinking { index } = state.block {
        events.push(StreamEvent::ContentBlockDelta {
            index,
            delta: Delta::ThinkingDelta {
                thinking: reasoning.to_string(),
            },
        });
    }
}

fn emit_text(events: &mut Vec<StreamEvent>, state: &mut StreamState, content: &str) {
    if !matches!(state.block, BlockState::Text { .. }) {
        close_current_block(events, state);
        let index = state.next_index;
        events.push(StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlockStart::Text {
                text: String::new(),
            },
        });
        state.block = BlockState::Text { index };
    }

    if let BlockState::Text { index } = state.block {
        events.push(StreamEvent::ContentBlockDelta {
            index,
            delta: Delta::TextDelta {
                text: content.to_string(),
            },
        });
    }
}

fn emit_tool_calls(
    _events: &mut Vec<StreamEvent>,
    state: &mut StreamState,
    tool_calls: &[openai::DeltaToolCall],
) {
    for tool_call in tool_calls {
        let openai_index = tool_call.index;

        if let Some(existing) = state
            .buffered_tool_calls
            .iter_mut()
            .find(|btc| btc.openai_index == openai_index)
        {
            if let Some(id) = &tool_call.id {
                existing.id = id.clone();
            }
            if let Some(function) = &tool_call.function {
                if let Some(name) = &function.name {
                    existing.name = name.clone();
                }
                if let Some(args) = &function.arguments {
                    existing.arguments.push_str(args);
                }
            }
        } else {
            let id = tool_call.id.clone().unwrap_or_default();
            let mut name = String::new();
            let mut arguments = String::new();

            if let Some(function) = &tool_call.function {
                if let Some(n) = &function.name {
                    name = n.clone();
                }
                if let Some(args) = &function.arguments {
                    arguments = args.clone();
                }
            }

            state.buffered_tool_calls.push(BufferedToolCall {
                id,
                name,
                arguments,
                openai_index,
            });
        }
    }
}

fn emit_finish(
    events: &mut Vec<StreamEvent>,
    state: &mut StreamState,
    finish_reason: &str,
    usage: Option<&openai::Usage>,
) {
    close_current_block(events, state);

    let stop_reason = core::map_stop_reason(Some(finish_reason));

    events.push(StreamEvent::MessageDelta {
        delta: MessageDeltaData {
            stop_reason,
            stop_sequence: None,
        },
        usage: DeltaUsage {
            input_tokens: usage.map(|u| u.prompt_tokens),
            output_tokens: usage.map(|u| u.completion_tokens).unwrap_or(0),
        },
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text_chunk(id: &str, model: &str, content: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "content": content } }]
        }))
        .unwrap()
    }

    fn reasoning_chunk(id: &str, model: &str, reasoning: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning": reasoning } }]
        }))
        .unwrap()
    }

    fn reasoning_content_chunk(id: &str, model: &str, reasoning: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning_content": reasoning } }]
        }))
        .unwrap()
    }

    fn finish_chunk(id: &str, model: &str, reason: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }]
        }))
        .unwrap()
    }

    fn finish_chunk_with_usage(
        id: &str,
        model: &str,
        reason: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id,
            "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }))
        .unwrap()
    }

    fn tool_start_chunk(id: &str, model: &str, tool_id: &str, name: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 0, "id": tool_id, "type": "function",
                    "function": { "name": name } }]
            }}]
        }))
        .unwrap()
    }

    fn tool_args_chunk(id: &str, model: &str, args: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 0, "function": { "arguments": args } }]
            }}]
        }))
        .unwrap()
    }

    fn event_types(events: &[StreamEvent]) -> Vec<&str> {
        events.iter().map(|e| e.event_type()).collect()
    }

    #[test]
    fn text_stream_produces_correct_event_sequence() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Hello"));
        assert_eq!(
            event_types(&e1),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );

        let e2 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", " world"));
        assert_eq!(event_types(&e2), ["content_block_delta"]);

        let e3 = translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "stop"));
        assert_eq!(event_types(&e3), ["content_block_stop", "message_delta"]);

        let e4 = translate_done(&mut state);
        assert_eq!(event_types(&e4), ["message_stop"]);
    }

    #[test]
    fn thinking_then_text_produces_two_blocks() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(&mut state, &reasoning_chunk("1", "gpt-4o", "Let me think"));
        assert_eq!(
            event_types(&e1),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );

        let e2 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Answer: 42"));
        assert_eq!(
            event_types(&e2),
            [
                "content_block_stop",
                "content_block_start",
                "content_block_delta"
            ]
        );

        if let StreamEvent::ContentBlockStart { index, .. } = &e2[1] {
            assert_eq!(*index, 1);
        }
    }

    #[test]
    fn reasoning_content_produces_thinking_block() {
        let mut state = initial_state("fallback".into());

        let events = translate_chunk(&mut state, &reasoning_content_chunk("1", "gpt-4o", "Think"));

        assert_eq!(
            event_types(&events),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
        if let StreamEvent::ContentBlockDelta { delta, .. } = &events[2] {
            assert!(matches!(delta, Delta::ThinkingDelta { thinking } if thinking == "Think"));
        }
    }

    #[test]
    fn tool_call_stream() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(
            &mut state,
            &tool_start_chunk("1", "gpt-4o", "call_abc", "read_file"),
        );
        assert_eq!(event_types(&e1), ["message_start"]);

        let e2 = translate_chunk(
            &mut state,
            &tool_args_chunk("1", "gpt-4o", "{\"path\":\"/tmp\"}"),
        );
        assert!(event_types(&e2).is_empty());

        let e3 = translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "tool_calls"));
        assert_eq!(
            event_types(&e3),
            [
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta"
            ]
        );

        if let StreamEvent::ContentBlockStart { content_block, index } = &e3[0] {
            assert_eq!(*index, 0);
            match content_block {
                ContentBlockStart::ToolUse { id, name } => {
                    assert_eq!(id, "call_abc");
                    assert_eq!(name, "read_file");
                }
                _ => panic!("expected tool_use block"),
            }
        }

        if let StreamEvent::ContentBlockDelta { delta, index } = &e3[1] {
            assert_eq!(*index, 0);
            match delta {
                Delta::InputJsonDelta { partial_json } => {
                    assert_eq!(partial_json, "{\"path\":\"/tmp\"}");
                }
                _ => panic!("expected InputJsonDelta"),
            }
        }

        if let StreamEvent::MessageDelta { delta, .. } = &e3[3] {
            assert_eq!(delta.stop_reason.as_deref(), Some("tool_use"));
        }
    }

    #[test]
    fn finish_chunk_with_usage_maps_input_and_output_tokens() {
        let mut state = initial_state("fallback".into());

        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Hello"));
        let events = translate_chunk(
            &mut state,
            &finish_chunk_with_usage("1", "gpt-4o", "stop", 7, 3),
        );

        if let StreamEvent::MessageDelta { usage, .. } = &events[1] {
            assert_eq!(usage.input_tokens, Some(7));
            assert_eq!(usage.output_tokens, 3);
        } else {
            panic!("expected message_delta");
        }
    }

    #[test]
    fn text_then_tool_call() {
        let mut state = initial_state("fallback".into());

        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "I'll read that."));

        let e2 = translate_chunk(
            &mut state,
            &tool_start_chunk("1", "gpt-4o", "call_xyz", "read_file"),
        );

        // The text block should be stopped because tool_calls commenced.
        assert_eq!(event_types(&e2), ["content_block_stop"]);
    }

    #[test]
    fn message_start_uses_chunk_metadata() {
        let mut state = initial_state("my-fallback".into());

        let events = translate_chunk(&mut state, &text_chunk("chatcmpl-42", "gpt-4o", "hi"));

        if let StreamEvent::MessageStart { message } = &events[0] {
            assert_eq!(message.id, "chatcmpl-42");
            assert_eq!(message.model, "gpt-4o");
            assert_eq!(message.role, "assistant");
        }
    }

    #[test]
    fn fallback_model_used_when_chunk_omits_model() {
        let mut state = initial_state("my-fallback".into());

        let chunk: openai::StreamChunk = serde_json::from_value(json!({
            "choices": [{ "index": 0, "delta": { "content": "hey" } }]
        }))
        .unwrap();

        let events = translate_chunk(&mut state, &chunk);

        if let StreamEvent::MessageStart { message } = &events[0] {
            assert_eq!(message.model, "my-fallback");
        }
    }

    #[test]
    fn error_event_produced() {
        let events = translate_error("connection reset".into());
        assert_eq!(event_types(&events), ["error"]);

        if let StreamEvent::Error { error } = &events[0] {
            assert!(error.message.contains("connection reset"));
        }
    }

    #[test]
    fn empty_content_not_emitted() {
        let mut state = initial_state("fallback".into());

        let chunk: openai::StreamChunk = serde_json::from_value(json!({
            "id": "1", "model": "gpt-4o",
            "choices": [{ "index": 0, "delta": { "content": "" } }]
        }))
        .unwrap();

        let events = translate_chunk(&mut state, &chunk);

        let deltas: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ContentBlockDelta { .. }))
            .collect();
        assert!(deltas.is_empty());
    }

    #[test]
    fn streaming_deduplication() {
        let mut state = initial_state("fallback".into());

        // Stream first tool call
        let _e1 = translate_chunk(&mut state, &tool_start_chunk("1", "gpt-4o", "call_1", "read_file"));
        let _e2 = translate_chunk(&mut state, &tool_args_chunk("1", "gpt-4o", "{\"path\":\"/tmp\"}"));

        // Stream a second identical tool call (as a separate tool call index / id)
        let chunk3: openai::StreamChunk = serde_json::from_value(json!({
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 1, "id": "call_2", "type": "function", "function": { "name": "read_file" } }]
            }}]
        })).unwrap();
        let _e3 = translate_chunk(&mut state, &chunk3);

        let chunk4: openai::StreamChunk = serde_json::from_value(json!({
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 1, "function": { "arguments": "{\"path\":\"/tmp\"}" } }]
            }}]
        })).unwrap();
        let _e4 = translate_chunk(&mut state, &chunk4);

        // Stream a third distinct tool call
        let chunk5: openai::StreamChunk = serde_json::from_value(json!({
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 2, "id": "call_3", "type": "function", "function": { "name": "read_file" } }]
            }}]
        })).unwrap();
        let _e5 = translate_chunk(&mut state, &chunk5);

        let chunk6: openai::StreamChunk = serde_json::from_value(json!({
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 2, "function": { "arguments": "{\"path\":\"/etc/hosts\"}" } }]
            }}]
        })).unwrap();
        let _e6 = translate_chunk(&mut state, &chunk6);

        // Finish stream
        let e7 = translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "tool_calls"));

        let events = event_types(&e7);
        // We expect only 2 unique tool call blocks:
        // call_1 (start, delta, stop) and call_3 (start, delta, stop) + message_delta.
        assert_eq!(
            events,
            [
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta"
            ]
        );

        if let StreamEvent::ContentBlockStart { content_block, .. } = &e7[0] {
            if let ContentBlockStart::ToolUse { id, name } = content_block {
                assert_eq!(id, "call_1");
                assert_eq!(name, "read_file");
            } else { panic!("expected ToolUse"); }
        }

        if let StreamEvent::ContentBlockStart { content_block, .. } = &e7[3] {
            if let ContentBlockStart::ToolUse { id, name } = content_block {
                assert_eq!(id, "call_3");
                assert_eq!(name, "read_file");
            } else { panic!("expected ToolUse"); }
        }
    }
}
