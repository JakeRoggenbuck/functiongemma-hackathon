
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, random, time
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


def _expected_action_count(user_text):
    """Estimate requested action count from intent signals and sentence connectors."""
    lowered = user_text.lower()
    action_signals = _count_action_signals(user_text)
    connector_hits = sum(lowered.count(tok) for tok in (" and ", ", and ", ", then ", " then ", " also ", ";"))
    if action_signals <= 1 and connector_hits == 0:
        return 1
    # A connector usually indicates at least one extra action segment.
    return max(action_signals, 1 + connector_hits)


def _is_hard_or_many_tools(messages, tools):
    """
    Route to cloud when complexity is likely high.
    Prioritize hard-task F1:
    - multi-action request -> cloud
    - very large toolset -> cloud
    - single tool with many required args -> cloud
    """
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    expected_actions = _expected_action_count(user_text)
    tool_count = len(tools)

    if expected_actions >= 2:
        return True

    if tool_count >= 4:
        return True

    if tool_count == 1:
        required = len(tools[0].get("parameters", {}).get("required", []))
        if required >= 2:
            return True

    return False


def _generate_cloud_with_min_calls(messages, tools, min_calls=1):
    """Call cloud once, and retry once if output likely missed actions."""
    first = generate_cloud(messages, tools)
    if len(first.get("function_calls", [])) >= min_calls:
        return first

    second = generate_cloud(messages, tools)
    if len(second.get("function_calls", [])) > len(first.get("function_calls", [])):
        return second
    return first


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid strategy with simple cloud routing for multi-tool or multi-action requests."""
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    expected_actions = _expected_action_count(user_text)

    if _is_hard_or_many_tools(messages, tools):
        cloud = _generate_cloud_with_min_calls(messages, tools, min_calls=expected_actions)
        cloud["source"] = "cloud (rule-route)"
        return cloud

    local = generate_cactus(messages, tools)

    if local["confidence"] >= confidence_threshold:
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
