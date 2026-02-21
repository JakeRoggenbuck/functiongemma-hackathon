
import atexit
import json
import os
import sys
import time

sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

from cactus import cactus_complete, cactus_destroy, cactus_init
from google import genai
from google.genai import types


_LOCAL_MODEL = None
_CLOUD_CLIENT = None


def _get_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        _LOCAL_MODEL = cactus_init(functiongemma_path)
    return _LOCAL_MODEL


def _get_cloud_client():
    global _CLOUD_CLIENT
    if _CLOUD_CLIENT is None:
        _CLOUD_CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _CLOUD_CLIENT


@atexit.register
def _cleanup_models():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None:
        cactus_destroy(_LOCAL_MODEL)
        _LOCAL_MODEL = None


def _normalize_string(value):
    if not isinstance(value, str):
        return value
    return " ".join(value.strip().split()).lower()


def _type_matches(value, expected_type):
    type_name = (expected_type or "").lower()
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return True


def _sanitize_calls(calls, tools):
    tools_by_name = {tool["name"]: tool for tool in tools}
    cleaned = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in tools_by_name or not isinstance(args, dict):
            continue

        schema = tools_by_name[name].get("parameters", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        valid = True

        for req_key in required:
            if req_key not in args:
                valid = False
                break
        if not valid:
            continue

        normalized_args = {}
        for key, val in args.items():
            normalized_val = val.strip() if isinstance(val, str) else val
            expected = properties.get(key, {}).get("type")
            if key in properties and not _type_matches(normalized_val, expected):
                valid = False
                break
            normalized_args[key] = normalized_val

        if not valid:
            continue

        cleaned.append({"name": name, "arguments": normalized_args})

    return cleaned


def _dedupe_calls(calls):
    seen = set()
    deduped = []
    for call in calls:
        key = (call["name"], json.dumps(call["arguments"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def _estimate_complexity(messages, tools):
    user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    text = " ".join(user_texts).lower()
    words = [w.strip(".,!?;:") for w in text.split() if w.strip(".,!?;:")]
    conjunction_count = sum(1 for w in words if w in {"and", "then", "also", "after"})
    punctuation_count = text.count(",") + text.count(";")

    complexity = 0.0
    complexity += min(len(tools), 6) * 0.08
    complexity += min(len(words), 40) * 0.007
    complexity += conjunction_count * 0.12
    complexity += punctuation_count * 0.06

    if len(words) > 20:
        complexity += 0.08
    return min(1.0, complexity)


def _quality_score(result, tools):
    calls = result.get("function_calls", [])
    confidence = float(result.get("confidence", 0) or 0)
    if not calls:
        return 0.20 * confidence

    valid_calls = _sanitize_calls(calls, tools)
    if not calls:
        validity_ratio = 0.0
    else:
        validity_ratio = len(valid_calls) / len(calls)

    duplicate_penalty = 0.0
    if len(valid_calls) != len(_dedupe_calls(valid_calls)):
        duplicate_penalty = 0.1

    score = (0.65 * confidence) + (0.35 * validity_ratio) - duplicate_penalty
    return max(0.0, min(1.0, score))


def _local_thresholds(confidence_threshold, complexity):
    base = min(confidence_threshold, 0.95)
    accept_local = base - (0.25 * complexity)
    trigger_repair = max(0.35, accept_local - 0.18)
    accept_local = max(0.55, min(0.95, accept_local))
    return accept_local, trigger_repair


def _run_cactus(messages, tools, system_prompt, max_tokens, tool_rag_top_k):
    model = _get_local_model()
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    start_time = time.time()

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=max_tokens,
        tool_rag_top_k=tool_rag_top_k,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    elapsed_ms = (time.time() - start_time) * 1000

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": elapsed_ms, "confidence": 0}

    total_time_ms = raw.get("total_time_ms", 0) or elapsed_ms
    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": total_time_ms,
        "confidence": raw.get("confidence", 0),
    }


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    prompt = "You are a helpful assistant that can use tools."
    result = _run_cactus(messages, tools, system_prompt=prompt, max_tokens=256, tool_rag_top_k=2)
    result["function_calls"] = _sanitize_calls(result.get("function_calls", []), tools)
    return result


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = _get_cloud_client()

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
                function_calls.append({"name": part.function_call.name, "arguments": dict(part.function_call.args)})

    return {"function_calls": _sanitize_calls(function_calls, tools), "total_time_ms": total_time_ms}


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid routing strategy:
    1) Run fast local pass.
    2) Accept local if quality is high enough for estimated task complexity.
    3) Run a constrained local repair pass for borderline cases.
    4) Fall back to cloud only when local outputs remain low quality.
    """
    complexity = _estimate_complexity(messages, tools)
    accept_local, trigger_repair = _local_thresholds(confidence_threshold, complexity)

    local = generate_cactus(messages, tools)
    local["function_calls"] = _dedupe_calls(_sanitize_calls(local.get("function_calls", []), tools))
    local_score = _quality_score(local, tools)

    if local_score >= accept_local:
        local["source"] = "on-device"
        return local

    should_repair = local_score >= trigger_repair or (local.get("confidence", 0) >= (accept_local - 0.08))
    if should_repair:
        repair_prompt = (
            "You are a strict tool-calling planner. "
            "Return only valid function calls using available tools. "
            "Include all required arguments and avoid unnecessary calls."
        )
        repaired = _run_cactus(
            messages,
            tools,
            system_prompt=repair_prompt,
            max_tokens=224,
            tool_rag_top_k=0,
        )
        repaired["function_calls"] = _dedupe_calls(_sanitize_calls(repaired.get("function_calls", []), tools))
        repaired_score = _quality_score(repaired, tools)
        repaired["total_time_ms"] += local.get("total_time_ms", 0)

        if repaired_score >= (accept_local - 0.04):
            repaired["source"] = "on-device"
            return repaired
    else:
        repaired = {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    cloud = generate_cloud(messages, tools)
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
