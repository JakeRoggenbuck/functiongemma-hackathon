
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, random, time
import re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import errors
from google.genai import types

TOOL_CALL_SYSTEM_PROMPT = (
    "You are a precise tool-calling assistant. "
    "Return only the function call JSON. "
    "Use exact tool names and infer concise, explicit argument values from the user request. "
    "If the user asks for multiple actions, emit all required function calls."
)


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": TOOL_CALL_SYSTEM_PROMPT}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    def _call_generate_content():
        return client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
            config=types.GenerateContentConfig(tools=gemini_tools),
        )

    gemini_response = _retry_with_backoff(_call_generate_content)

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def _retry_with_backoff(fn, max_attempts=6, base_sleep_s=1.0, max_sleep_s=16.0):
    """Retry transient Gemini API failures with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except errors.ServerError as exc:
            if attempt == max_attempts:
                raise
            # Retry on temporary service-side errors (e.g. 503 high demand).
            if exc.code and int(exc.code) not in (500, 502, 503, 504):
                raise
            sleep_s = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
            jitter = random.uniform(0.0, 0.3 * sleep_s)
            time.sleep(sleep_s + jitter)


def _count_action_signals(text):
    """Estimate how many distinct actions the user is asking for."""
    lowered = text.lower()
    signal_groups = [
        {"weather", "temperature", "forecast"},
        {"alarm", "wake me", "wake-up"},
        {"message", "text", "send"},
        {"remind", "reminder"},
        {"contact", "contacts", "find", "look up", "search"},
        {"play", "music", "song", "playlist"},
        {"timer", "countdown"},
    ]
    matches = 0
    for keywords in signal_groups:
        if any(k in lowered for k in keywords):
            matches += 1
    return matches


def _is_hard_or_many_tools(messages, tools):
    """
    Route hard requests to cloud directly:
    - many candidate tools
    - clear multi-action phrasing
    - multiple distinct action intents
    """
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    lowered = user_text.lower()
    tool_count = len(tools)
    action_count = _count_action_signals(user_text)
    connector_count = sum(lowered.count(tok) for tok in (" and ", ", and ", ", then ", " then ", " also "))

    if tool_count >= 5:
        return True
    if action_count >= 2:
        return True
    if connector_count >= 2:
        return True
    if connector_count >= 1 and action_count >= 1 and tool_count >= 3:
        return True
    return False


def _strip_trailing_punct(text):
    return re.sub(r"[.!?]+$", "", text.strip())


def _extract_time_parts(text):
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    return hour, minute


def _extract_weather_location(clause):
    patterns = [
        r"weather(?:\s+like)?\s+in\s+([a-zA-Z][a-zA-Z\s'-]*)",
        r"weather\s+for\s+([a-zA-Z][a-zA-Z\s'-]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clause, flags=re.IGNORECASE)
        if match:
            return _strip_trailing_punct(match.group(1))
    return None


def _extract_search_query(clause):
    patterns = [
        r"(?:find|look up|search for|search)\s+([a-zA-Z][a-zA-Z\s'-]*?)(?:\s+in my contacts)?$",
        r"(?:find|look up|search for|search)\s+([a-zA-Z][a-zA-Z\s'-]*)\s+in my contacts",
    ]
    cleaned = _strip_trailing_punct(clause)
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return _strip_trailing_punct(match.group(1))
    return None


def _extract_timer_minutes(clause):
    match = re.search(r"\b(\d+)\s*(?:minutes?|mins?)\b", clause, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_song(clause):
    cleaned = _strip_trailing_punct(clause)
    match = re.search(r"\bplay\s+(.+)$", cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    song = match.group(1).strip()
    song = re.sub(r"^some\s+", "", song, flags=re.IGNORECASE)
    return song.strip()


def _extract_reminder(clause):
    cleaned = _strip_trailing_punct(clause)
    match = re.search(
        r"remind me(?:\s+(?:about|to))?\s+(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    title = _strip_trailing_punct(match.group(1))
    time_value = match.group(2).upper().replace(".", "")
    return {"title": title, "time": time_value}


def _extract_message(clause, last_contact):
    cleaned = _strip_trailing_punct(clause)
    lower = cleaned.lower()
    if "saying" not in lower:
        return None

    before, after = re.split(r"\bsaying\b", cleaned, maxsplit=1, flags=re.IGNORECASE)
    message_text = _strip_trailing_punct(after)
    if not message_text:
        return None

    recipient = None
    to_match = re.search(r"\bto\s+([a-zA-Z][a-zA-Z'-]*)\b", before, flags=re.IGNORECASE)
    if to_match:
        recipient = to_match.group(1)
    if recipient is None:
        text_match = re.search(r"\btext\s+([a-zA-Z][a-zA-Z'-]*)\b", before, flags=re.IGNORECASE)
        if text_match:
            recipient = text_match.group(1)
    if recipient is None and re.search(r"\b(him|her|them)\b", before, flags=re.IGNORECASE):
        recipient = last_contact

    if not recipient:
        return None
    return {"recipient": recipient, "message": message_text}


def _heuristic_local_calls(messages, tools):
    """Fast local parser for structured assistant requests."""
    tool_names = {t["name"] for t in tools}
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").strip()
    if not user_text:
        return None

    clauses = [
        c.strip()
        for c in re.split(r",\s*and\s+|\s+and\s+|,\s+|\s+then\s+|\s+also\s+", user_text, flags=re.IGNORECASE)
        if c.strip()
    ]

    calls = []
    last_contact = None
    for clause in clauses:
        lower = clause.lower()

        if "search_contacts" in tool_names and any(k in lower for k in ("find ", "look up ", "search")):
            query = _extract_search_query(clause)
            if query:
                calls.append({"name": "search_contacts", "arguments": {"query": query}})
                last_contact = query
                continue

        if "send_message" in tool_names and any(k in lower for k in ("message", "text ", "send ")):
            msg = _extract_message(clause, last_contact)
            if msg:
                calls.append({"name": "send_message", "arguments": msg})
                continue

        if "get_weather" in tool_names and "weather" in lower:
            location = _extract_weather_location(clause)
            if location:
                calls.append({"name": "get_weather", "arguments": {"location": location}})
                continue

        if "set_alarm" in tool_names and any(k in lower for k in ("alarm", "wake me up", "wake me")):
            time_parts = _extract_time_parts(clause)
            if time_parts:
                hour, minute = time_parts
                calls.append({"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}})
                continue

        if "set_timer" in tool_names and "timer" in lower:
            minutes = _extract_timer_minutes(clause)
            if minutes is not None:
                calls.append({"name": "set_timer", "arguments": {"minutes": minutes}})
                continue

        if "create_reminder" in tool_names and "remind me" in lower:
            reminder = _extract_reminder(clause)
            if reminder:
                calls.append({"name": "create_reminder", "arguments": reminder})
                continue

        if "play_music" in tool_names and "play" in lower:
            song = _extract_song(clause)
            if song:
                calls.append({"name": "play_music", "arguments": {"song": song}})
                continue

    if not calls:
        return None

    signal_count = _count_action_signals(user_text)
    confidence = 0.99 if len(calls) >= max(1, signal_count) else 0.75
    return {
        "function_calls": calls,
        "total_time_ms": 0,
        "confidence": confidence,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid strategy: deterministic local parse first, then adaptive local/cloud routing."""
    heuristic_local = _heuristic_local_calls(messages, tools)
    if heuristic_local and heuristic_local["confidence"] >= 0.99:
        heuristic_local["source"] = "on-device"
        return heuristic_local

    if _is_hard_or_many_tools(messages, tools):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (hard-route)"
        return cloud

    local = generate_cactus(messages, tools)

    dynamic_threshold = min(confidence_threshold, 0.92)
    if local["confidence"] >= dynamic_threshold:
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
