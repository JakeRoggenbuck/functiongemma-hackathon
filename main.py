import atexit
import json
import os
import re
import sys
import time

sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/lfm2.5-1.2b-instruct"

from cactus import cactus_complete, cactus_destroy, cactus_init
from google import genai
from google.genai import types


_CACTUS_MODEL = None
_GEMINI_CLIENT = None


def _get_cactus_model():
    global _CACTUS_MODEL
    if _CACTUS_MODEL is None:
        _CACTUS_MODEL = cactus_init(functiongemma_path)
    return _CACTUS_MODEL


@atexit.register
def _cleanup_models():
    global _CACTUS_MODEL
    if _CACTUS_MODEL is not None:
        try:
            cactus_destroy(_CACTUS_MODEL)
        finally:
            _CACTUS_MODEL = None


def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _GEMINI_CLIENT


def _normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def _extract_time_components(text):
    text = text.strip().lower().replace(".", "")
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    if not m:
        return None

    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    ampm = m.group(3)

    if ampm == "am":
        if hour == 12:
            hour = 0
    elif ampm == "pm":
        if hour != 12:
            hour += 12

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour, minute
    return None


def _extract_time_string(text):
    m = re.search(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\b", text)
    if m:
        h = m.group(1)
        return h[:-2].strip() + " " + h[-2:].upper()

    m = re.search(r"\b(\d{1,2}\s*(?:AM|PM|am|pm))\b", text)
    if m:
        h = m.group(1).strip()
        num = re.match(r"\d{1,2}", h).group(0)
        suffix = h[len(num):].strip().upper()
        return f"{num}:00 {suffix}"

    return None


def _extract_location(text):
    patterns = [
        r"weather(?:\s+like)?\s+in\s+([A-Za-z .'-]+?)(?:[,.!?]|$)",
        r"in\s+([A-Za-z .'-]+?)\s+weather(?:[,.!?]|$)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return _normalize_text(m.group(1)).strip(" .,!?")
    return None


def _extract_message_fields(text):
    # send a message to Alice saying good morning
    patterns = [
        r"(?:send|text)\s+(?:a\s+)?message\s+to\s+([A-Za-z][A-Za-z .'-]*)\s+(?:saying|that says|saying:)\s+(.+?)(?:[.!?]|$)",
        r"(?:send|text)\s+([A-Za-z][A-Za-z .'-]*)\s+(?:saying|that says|saying:)\s+(.+?)(?:[.!?]|$)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            recipient = _normalize_text(m.group(1)).strip(" .,!?")
            message = _normalize_text(m.group(2)).strip(" .,!?")
            return recipient, message
    return None


def _extract_contact_query(text):
    patterns = [
        r"(?:find|look up|search for)\s+([A-Za-z][A-Za-z .'-]*)\s+(?:in\s+my\s+contacts|from\s+my\s+contacts)",
        r"(?:find|look up|search for)\s+([A-Za-z][A-Za-z .'-]*)(?:[,.!?]|$)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return _normalize_text(m.group(1)).strip(" .,!?")
    return None


def _extract_song(text):
    # play bohemian rhapsody / play some jazz music
    m = re.search(r"\bplay\s+(.+?)(?:[.!?]|$)", text, flags=re.IGNORECASE)
    if not m:
        return None
    song = _normalize_text(m.group(1)).strip(" .,!?")
    song = re.sub(r"^(?:some\s+)", "", song, flags=re.IGNORECASE)
    song = re.sub(r"\s+music$", "", song, flags=re.IGNORECASE)
    return song if song else None


def _extract_timer_minutes(text):
    m = re.search(r"\b(?:timer\s+for|set\s+a\s+)?(\d{1,3})\s+minute(?:s)?\b", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_reminder_fields(text):
    # remind me to call dentist at 2:00 PM
    patterns = [
        r"remind\s+me\s+(?:to\s+)?(.+?)\s+at\s+((?:\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)))",
        r"remind\s+me\s+about\s+(.+?)\s+at\s+((?:\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)))",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            title = _normalize_text(m.group(1)).strip(" .,!?")
            t = _extract_time_string(m.group(2))
            if t:
                return title, t
    return None


def _available_tools_map(tools):
    return {t["name"]: t for t in tools}


def _segment_intents(user_text):
    # Split conjunctive requests into chunks while preserving semantic parts.
    parts = re.split(r"\s+(?:and|then|also)\s+|,\s*", user_text)
    return [p.strip() for p in parts if p.strip()]


def _heuristic_calls(messages, tools):
    user_text = " ".join(m["content"] for m in messages if m.get("role") == "user")
    user_text = _normalize_text(user_text)
    if not user_text:
        return []

    tool_map = _available_tools_map(tools)
    intents = _segment_intents(user_text)
    calls = []

    for intent in intents:
        l = intent.lower()

        if "weather" in l and "get_weather" in tool_map:
            loc = _extract_location(intent) or _extract_location(user_text)
            if loc:
                calls.append({"name": "get_weather", "arguments": {"location": loc}})
                continue

        if ("alarm" in l or "wake me up" in l) and "set_alarm" in tool_map:
            hm = _extract_time_components(intent) or _extract_time_components(user_text)
            if hm:
                calls.append({"name": "set_alarm", "arguments": {"hour": hm[0], "minute": hm[1]}})
                continue

        if ("send" in l or "text" in l) and "send_message" in tool_map:
            fields = _extract_message_fields(intent) or _extract_message_fields(user_text)
            if fields:
                calls.append({"name": "send_message", "arguments": {"recipient": fields[0], "message": fields[1]}})
                continue

        if "remind" in l and "create_reminder" in tool_map:
            fields = _extract_reminder_fields(intent) or _extract_reminder_fields(user_text)
            if fields:
                calls.append({"name": "create_reminder", "arguments": {"title": fields[0], "time": fields[1]}})
                continue

        if ("find" in l or "look up" in l or "search" in l) and "search_contacts" in tool_map:
            query = _extract_contact_query(intent) or _extract_contact_query(user_text)
            if query:
                calls.append({"name": "search_contacts", "arguments": {"query": query}})
                continue

        if "play" in l and "play_music" in tool_map:
            song = _extract_song(intent) or _extract_song(user_text)
            if song:
                calls.append({"name": "play_music", "arguments": {"song": song}})
                continue

        if "timer" in l and "set_timer" in tool_map:
            mins = _extract_timer_minutes(intent) or _extract_timer_minutes(user_text)
            if mins is not None:
                calls.append({"name": "set_timer", "arguments": {"minutes": mins}})
                continue

    # De-duplicate exact repeated calls while preserving order.
    deduped = []
    seen = set()
    for c in calls:
        key = (c["name"], json.dumps(c["arguments"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _type_ok(value, expected_type):
    t = (expected_type or "").lower()
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    return True


def _valid_call(call, tools_map):
    name = call.get("name")
    args = call.get("arguments", {})
    if name not in tools_map:
        return False
    if not isinstance(args, dict):
        return False

    schema = tools_map[name].get("parameters", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for req in required:
        if req not in args:
            return False

    for k, v in args.items():
        if k in properties and not _type_ok(v, properties[k].get("type")):
            return False

    return True


def _normalize_call_arguments(call):
    name = call.get("name")
    args = dict(call.get("arguments", {}))

    if name == "set_alarm":
        hour = args.get("hour")
        minute = args.get("minute", 0)
        if isinstance(hour, str) and hour.isdigit():
            hour = int(hour)
        if isinstance(minute, str) and minute.isdigit():
            minute = int(minute)
        if isinstance(hour, int) and isinstance(minute, int):
            args["hour"], args["minute"] = hour, minute

    if name == "set_timer":
        minutes = args.get("minutes")
        if isinstance(minutes, str):
            m = re.search(r"\d+", minutes)
            if m:
                args["minutes"] = int(m.group(0))

    if name == "play_music" and isinstance(args.get("song"), str):
        song = _normalize_text(args["song"]).strip(" .,!?")
        song = re.sub(r"^(?:some\s+)", "", song, flags=re.IGNORECASE)
        song = re.sub(r"\s+music$", "", song, flags=re.IGNORECASE)
        args["song"] = song

    if name == "create_reminder" and isinstance(args.get("time"), str):
        ts = _extract_time_string(args["time"])
        if ts:
            args["time"] = ts

    call["arguments"] = args
    return call


def _validate_and_clean_calls(calls, tools):
    tools_map = _available_tools_map(tools)
    cleaned = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        c = {
            "name": call.get("name"),
            "arguments": call.get("arguments", {}) if isinstance(call.get("arguments", {}), dict) else {},
        }
        c = _normalize_call_arguments(c)
        if _valid_call(c, tools_map):
            cleaned.append(c)
    return cleaned


def generate_cactus(messages, tools, system_prompt=None, max_tokens=192, tool_rag_top_k=2):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_cactus_model()

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    system_prompt = system_prompt or "You are a helpful assistant that can use tools."

    start_time = time.perf_counter()
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=max_tokens,
        tool_rag_top_k=tool_rag_top_k,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    wall_time_ms = (time.perf_counter() - start_time) * 1000.0

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": wall_time_ms,
            "confidence": 0,
        }

    total_time_ms = raw.get("total_time_ms", 0)
    if not total_time_ms:
        total_time_ms = wall_time_ms

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": total_time_ms,
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = _get_gemini_client()

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

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

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


def _dynamic_threshold(messages, tools, base_threshold=0.93):
    user_text = " ".join(m["content"] for m in messages if m.get("role") == "user").lower()
    tool_count = len(tools)

    conjunctions = len(re.findall(r"\b(and|then|also)\b", user_text))
    likely_multi = conjunctions >= 1 or user_text.count(",") >= 1

    threshold = base_threshold
    if likely_multi:
        threshold -= 0.08
    if tool_count >= 4:
        threshold -= 0.04
    if tool_count == 1:
        threshold += 0.02

    return max(0.75, min(0.98, threshold))


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid strategy: heuristic on-device parse, validated local inference, selective cloud fallback."""
    start_ms = time.perf_counter()

    heuristic = _heuristic_calls(messages, tools)
    heuristic_clean = _validate_and_clean_calls(heuristic, tools)
    if heuristic_clean:
        return {
            "function_calls": heuristic_clean,
            "total_time_ms": (time.perf_counter() - start_ms) * 1000.0,
            "confidence": 1.0,
            "source": "on-device",
        }

    local = generate_cactus(messages, tools)
    local["function_calls"] = _validate_and_clean_calls(local.get("function_calls", []), tools)

    dyn_threshold = _dynamic_threshold(messages, tools, base_threshold=min(confidence_threshold, 0.95))

    if local["function_calls"] and local["confidence"] >= dyn_threshold:
        local["source"] = "on-device"
        return local

    # Local repair pass before cloud fallback.
    repair_prompt = (
        "You must output only valid function calls. "
        "Use only the provided tools, include all required arguments, and avoid extra calls."
    )
    repaired = generate_cactus(
        messages,
        tools,
        system_prompt=repair_prompt,
        max_tokens=224,
        tool_rag_top_k=0,
    )
    repaired["function_calls"] = _validate_and_clean_calls(repaired.get("function_calls", []), tools)

    if repaired["function_calls"] and repaired["confidence"] >= (dyn_threshold - 0.04):
        repaired["source"] = "on-device"
        repaired["total_time_ms"] += local.get("total_time_ms", 0)
        return repaired

    try:
        cloud = generate_cloud(messages, tools)
        cloud["function_calls"] = _validate_and_clean_calls(cloud.get("function_calls", []), tools)
    except Exception:
        # If cloud fails, return the best local attempt rather than erroring the benchmark.
        best_local = repaired if repaired["function_calls"] else local
        best_local["source"] = "on-device"
        best_local["total_time_ms"] = local.get("total_time_ms", 0) + repaired.get("total_time_ms", 0)
        best_local["local_confidence"] = max(local.get("confidence", 0), repaired.get("confidence", 0))
        return best_local

    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = max(local.get("confidence", 0), repaired.get("confidence", 0))
    cloud["total_time_ms"] += local.get("total_time_ms", 0) + repaired.get("total_time_ms", 0)
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
