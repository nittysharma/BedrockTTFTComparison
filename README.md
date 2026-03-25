# Amazon Bedrock OSS GPT Testing

This repository contains a test script and documentation for benchmarking and invoking the `openai.gpt-oss-120b-1:0` model on Amazon Bedrock.

## Using Priority Tier and Reasoning Effort

The newly released open-weight GPT models on AWS Bedrock natively support dynamic reasoning. You can configure how hard the model "thinks" before answering by passing the `reasoning_effort` parameter. Furthermore, for production use cases under high load, you can ensure low latency by utilizing the `priority` service tier.

Both of these configurations can be seamlessly passed to the `converse` or `converse_stream` APIs via the `additionalModelRequestFields` dictionary.

### Code Snippet: Python Boto3

Here is an exact, standalone code snippet demonstrating how to format a Bedrock `converse_stream` call with **Priority Tier** and **High Reasoning Effort**:

```python
import boto3

# Initialize the Bedrock Runtime client
client = boto3.client("bedrock-runtime", region_name="ap-south-1")

model_id = "openai.gpt-oss-120b-1:0"
prompt = "Explain competitive programming strategies in detail."

# 1. Format the standard messages block
messages = [
    {
        "role": "user",
        "content": [{"text": prompt}]
    }
]

# 2. Configure Priority Tier and Reasoning Effort
# These specific configurations are mapped via additionalModelRequestFields
additional_fields = {
    "service_tier": "priority",   # Guaranteed throughput, bypasses queueing
    "reasoning_effort": "high"    # Options: "low", "medium", "high"
}

try:
    print(f"Invoking {model_id} with High Reasoning Effort on Priority Tier...")
    
    # 3. Call the Converse Stream API
    response = client.converse_stream(
        modelId=model_id,
        messages=messages,
        additionalModelRequestFields=additional_fields
    )

    # 4. Process the streaming response
    for event in response.get("stream"):
        if "contentBlockDelta" in event:
            print(event["contentBlockDelta"]["delta"].get("text", ""), end="")
            
    print("\n\nFinished streaming.")

except Exception as e:
    print(f"Error invoking model: {e}")
```

### Explaining the Parameters

* **`service_tier: "priority"`**: 
  AWS Bedrock Standard inference shares capacity. If the region is heavily utilized, requests may queue or return throttling exceptions (`429`). The Priority tier guarantees your request bypasses noisy-neighbor queuing, ensuring stable Time To First Token (TTFT) when the network is under load.
  
* **`reasoning_effort: "high"`**: 
  Because `gpt-oss-120b` utilizes a mixture-of-experts (MoE) architecture, it can dynamically assign more or less physical compute to a prompt. 
  - `"low"`: Extremely fast TTFT, shorter chain-of-thought. Suitable for summarization and direct factual lookups.
  - `"medium"`: A balanced approach.
  - `"high"`: Forces maximum background reasoning tokens to evaluate logical fallacies, math, or complex constraints. Note that total time will increase as the model thinks before outputting the visible text. This scales up accuracy on complex prompts.

### Running the automated testing benchmark

You can test these interactions locally by running the included python scripts. Note that testing without concurrent load may mask the advantages of the Priority tier.

#### Using Converse API (Native Bedrock):

```bash
# Run the full test suite iterating through all levels of reasoning
# on both Standard and Priority tiers:
python test_bedrock_priority.py

# Alternatively, test a specific logic puzzle:
python test_bedrock_priority.py --prompt "Five people are sitting in a row..."

# Test with reasoning completely disabled (LLM will not perform reasoning):
python test_bedrock_priority.py --exclude-reasoning

# Test a specific reasoning effort level:
python test_bedrock_priority.py --reasoning-effort high --tier priority
```

#### Using InvokeModel API:

For customers who want to test with InvokeModel API (direct model invocation) instead of Converse API:

```bash
# Run the full test suite using InvokeModel API:
python test_bedrock_priority_openai.py

# Test specific configurations:
python test_bedrock_priority_openai.py --reasoning-effort high --tier priority

# Custom prompt:
python test_bedrock_priority_openai.py --prompt "Your question here"
```

**Note**: The InvokeModel version ([`test_bedrock_priority_openai.py`](test_bedrock_priority_openai.py)) uses `invoke_model_with_response_stream` and is useful when:
- Need direct model access without Converse API abstraction
- Testing model-specific features not in Converse
- Working with models requiring custom payload formats
- Comparing Converse vs InvokeModel performance

---

## Enhanced Features: Reasoning Token Separation

### 🎯 Key Customer Requirements Addressed

This implementation addresses three critical customer needs:

#### 1. **Separate Reasoning Token Tracking**
   - **Requirement**: A new key is required in the response for reasoning tokens
   - **Solution**: Response now includes separate fields:
     - `reasoning_tokens`: List of individual reasoning token strings
     - `content_tokens`: List of content token strings
     - `full_reasoning`: Complete reasoning text
     - `full_content`: Complete content text
   - **Previous Behavior**: Had to check for `<reasoning>` substring in `token.choices[0].delta`
   - **New Behavior**: Automatic separation and tracking of reasoning vs content

#### 2. **True Token-by-Token Streaming**
   - **Requirement**: Receiving direct response rather than receiving tokens, though stream is True
   - **Solution**: Implemented proper streaming with `flush=True` for real-time token delivery
   - **Previous Behavior**: Response appeared all at once despite streaming being enabled
   - **New Behavior**: Each token is printed immediately as it arrives

#### 3. **Reasoning Display Control**
   - **Requirement**: Flexibility to include/exclude reasoning through params
   - **Solution**: Added `include_reasoning` parameter (default: True)
   - **Usage**:
     - `include_reasoning=True`: Show reasoning in output to user
     - `include_reasoning=False`: Hide reasoning from output
   - **Important Note**: The `openai.gpt-oss-120b-1:0` model ALWAYS generates reasoning internally (this is by design). The `include_reasoning` parameter controls visibility, not generation.
   - **Benefit**: Clean output control - hide technical reasoning when not needed for end users

### Example Usage

See [`example_usage.py`](example_usage.py) for a complete demonstration:

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="ap-south-1")

# Example with reasoning enabled at LLM level
result = stream_with_reasoning_separation(
    client=client,
    model_id="openai.gpt-oss-120b-1:0",
    prompt="Solve this logic puzzle...",
    include_reasoning=True,  # LLM performs reasoning
    reasoning_effort="high"
)

# Example with reasoning completely disabled (faster, cheaper)
result_no_reasoning = stream_with_reasoning_separation(
    client=client,
    model_id="openai.gpt-oss-120b-1:0",
    prompt="Solve this logic puzzle...",
    include_reasoning=False,  # LLM does NOT perform reasoning
    reasoning_effort="high"   # This parameter is ignored when include_reasoning=False
)

# Access separated tokens
print(f"Reasoning tokens: {len(result['reasoning_tokens'])}")
print(f"Content tokens: {len(result['content_tokens'])}")
print(f"Full reasoning: {result['full_reasoning']}")
print(f"Full content: {result['full_content']}")
```

### Response Structure

The enhanced response now returns a dictionary with:

```python
{
    "ttft": 0.1234,                    # Time to first token (seconds)
    "total_time": 2.5678,              # Total completion time (seconds)
    "reasoning_tokens": [...],          # List of reasoning token strings
    "content_tokens": [...],            # List of content token strings
    "full_reasoning": "...",            # Complete reasoning text
    "full_content": "...",              # Complete content text
    "token_count": 150                  # Total token count
}
```

### Command-Line Options

```bash
# Full test suite
python test_bedrock_priority.py

# Disable reasoning completely (LLM will not perform reasoning)
python test_bedrock_priority.py --exclude-reasoning

# Test specific reasoning effort
python test_bedrock_priority.py --reasoning-effort high

# Test specific tier
python test_bedrock_priority.py --tier priority

# Custom prompt with reasoning disabled (faster, cheaper)
python test_bedrock_priority.py \
  --prompt "Your complex prompt here" \
  --exclude-reasoning
```
