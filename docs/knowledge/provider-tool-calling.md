# Provider tool calling — physical wire contract

Source of truth for the three providers we speak (Anthropic, OpenAI-compatible,
Ollama). Captured during the v0.8 architect spike; this file documents the
**physical wire shapes only**, not how Mneme should expose them.

Mneme's `client.chat()` normalizes all three into a single `AssistantMessage`
shape with `tool_calls: list[ToolCall]` where `ToolCall.arguments` is **always
a `dict`** (never a string). The differences below explain why that
normalization layer has to exist.

## 1. The four physical hard walls

| Aspect | Anthropic | OpenAI-compatible | Ollama |
|---|---|---|---|
| Request: declare tools | `tools=[{name, description, input_schema}]` at top level | `tools=[{type:"function", function:{name, description, parameters}}]` | same as OpenAI |
| Response: tool signal | `content` array contains `{type:"tool_use", id:"toolu_...", name, input:{}}`; `stop_reason="tool_use"` | `choices[0].message.tool_calls=[{id, type:"function", function:{name, arguments:"<JSON STRING>"}}]`; `finish_reason="tool_calls"`; `content` may be `null` | `message.tool_calls=[{function:{name, arguments:<OBJECT>}}]`; no `id` |
| Sending tool results back | role=`user` with `content=[{type:"tool_result", tool_use_id, content, is_error}]` | append assistant msg with the original `tool_calls`, then append `{role:"tool", tool_call_id, content}` | append `{role:"tool", content, tool_name}` — no `tool_call_id` |
| Streaming tool args | `input_json_delta` events; accumulate into a JSON string then parse | `delta.tool_calls[].function.arguments` is sent in fragments; key by `index` to reassemble | tool_calls usually arrive whole in a single chunk |

## 2. The four traps that bit us during the spike

1. **Ollama `arguments` is an object, not a JSON string.** OpenAI gives you
   `"arguments":"{\"path\":\"foo\"}"` (a string you must `json.loads`).
   Ollama gives you `"arguments":{"path":"foo"}` (already parsed). Our
   normalization rule: after parsing, `ToolCall.arguments` is always a
   `dict`. OpenAI branch calls `json.loads`; Ollama branch passes the dict
   through.

2. **Anthropic tool results don't use `role:"tool"`.** They go back as a
   `user` message whose `content` is a list of `tool_result` blocks. Sending
   `role:"tool"` to `/v1/messages` is a 400.

3. **Ollama has no `tool_call_id`.** The wire-level pairing is by `tool_name`
   in the followup `{role:"tool", content, tool_name}` message. This is fine
   for M1 because we issue one tool call per loop iteration. For M2 / parallel
   tool calls we'll need to forbid parallel-on-Ollama (provider capability
   matrix lives elsewhere; not in this file).

4. **Anthropic streaming tool args: `input_json_delta` is a string fragment
   stream.** You concatenate, then `json.loads` once `content_block_stop`
   fires. Don't try to `json.loads` mid-stream.

## 3. M1 scope notes (out of scope for this file but listed so future-me knows)

- M1 forces `stream=False` whenever `tools` is passed (the loop's intermediate
  rounds need a complete `AssistantMessage` to decide whether to keep going).
  The final round, once the LLM stops emitting `tool_use`, may stream — but
  in M1 it doesn't, because the agent loop drives non-stream end-to-end and
  the CLI just prints the final text on completion.
- `id` synthesis for Ollama tool calls: we generate a ULID locally
  (`toolu_<ulid>`) so the rest of the codebase can always assume a non-empty
  `ToolCall.id`. The ID is local-only; we don't echo it back to Ollama on
  the tool result message.

## 4. Test transports

`tests/test_llm_client.py` declares one MockTransport per provider that
returns a canned tool-use response. The assertions check both directions:
the **request body** declares tools in the right shape, and the parsed
`AssistantMessage` carries `tool_calls` with `arguments` as a dict.
