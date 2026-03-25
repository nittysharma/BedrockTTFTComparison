"""
Amazon Bedrock Priority Testing - InvokeModel API

This version uses Bedrock's InvokeModelWithResponseStream API with model-specific payloads.
Unlike Converse API which is a unified interface, InvokeModel allows direct model invocation
with custom request formats.

Use this when:
- Need direct model access without Converse API abstraction
- Testing model-specific features not supported by Converse
- Working with models that require custom payload formats
- Comparing Converse vs InvokeModel performance

Note: This still uses Bedrock's native APIs, not OpenAI REST endpoints.
For true OpenAI compatibility, use OpenAI Python SDK configured for Bedrock.
"""

import boto3
import time
import argparse
import json

def test_model_invokemodel(region, model_id, tier, prompt, reasoning_effort=None, include_reasoning=True):
    """
    Test model using InvokeModelWithResponseStream API.

    Args:
        region: AWS region
        model_id: Model identifier
        tier: Service tier (priority/default)
        prompt: Input prompt
        reasoning_effort: Reasoning effort level (low/medium/high)
        include_reasoning: Whether to display reasoning to user

    Returns:
        Dictionary containing timing and token information
    """
    print(f"\n--- Testing Model: {model_id} | Tier: {tier} ---")

    # Set reasoning effort level
    if reasoning_effort:
        print(f"Reasoning Effort: {reasoning_effort}")
        effective_reasoning_effort = reasoning_effort
    else:
        print(f"Reasoning Effort: DEFAULT (model decides)")
        effective_reasoning_effort = None

    # Note about display control
    if not include_reasoning:
        print(f"Display Mode: HIDING reasoning from output")
        print(f"Note: Model still generates reasoning internally")
    else:
        print(f"Display Mode: SHOWING reasoning in output")

    # Build model-specific request body
    # For openai.gpt-oss models, use their native format
    request_body = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    # Add reasoning effort if specified
    if effective_reasoning_effort:
        request_body["reasoning_effort"] = effective_reasoning_effort

    # Debug output
    import os
    debug_mode = os.environ.get("DEBUG_BEDROCK")

    if debug_mode:
        print(f"\n[DEBUG] InvokeModel API Call Parameters:")
        print(f"  - Model: {model_id}")
        print(f"  - Service Tier: {tier}")
        print(f"  - Request Body: {json.dumps(request_body, indent=2)}")
        print(f"  - Prompt: {prompt[:50]}...")
        print()

    start_time = time.time()
    first_token_time = None

    try:
        # Create Bedrock runtime client
        client = boto3.client("bedrock-runtime", region_name=region)

        # Prepare invoke parameters
        invoke_params = {
            "modelId": model_id,
            "body": json.dumps(request_body),
            "contentType": "application/json",
            "accept": "application/json"
        }

        # Add service tier for priority
        if tier == "priority":
            invoke_params["serviceTier"] = "priority"

        if debug_mode:
            print(f"[DEBUG] Invoke Parameters: {json.dumps({k: v for k, v in invoke_params.items() if k != 'body'}, indent=2)}")

        # Invoke model with streaming
        response = client.invoke_model_with_response_stream(**invoke_params)

        # Separate tracking for reasoning and content tokens
        reasoning_tokens = []
        content_tokens = []
        full_reasoning = ""
        full_content = ""

        # Track state for XML parsing
        in_reasoning_tag = False
        buffer = ""  # Buffer to accumulate text between tag boundaries

        print("\n[Streaming Tokens]")
        token_count = 0

        # Process response stream
        stream = response.get('body')

        for event in stream:
            if debug_mode and token_count < 5:
                print(f"\n[DEBUG Event {token_count}]: {list(event.keys())}")

            # Parse the chunk from the stream
            if 'chunk' in event:
                chunk_data = event['chunk']
                chunk_json = json.loads(chunk_data['bytes'].decode('utf-8'))

                if debug_mode and token_count < 5:
                    print(f"[DEBUG] Chunk JSON: {json.dumps(chunk_json, indent=2)}")

                # Track first token timing
                if first_token_time is None:
                    first_token_time = time.time()

                # Parse model response format
                # The format varies by model, for openai.gpt models:
                if 'choices' in chunk_json:
                    for choice in chunk_json['choices']:
                        text_chunk = None

                        # Handle delta format
                        if 'delta' in choice:
                            delta = choice['delta']
                            # Check for reasoning content (structured format)
                            if 'reasoning_content' in delta or 'reasoning' in delta:
                                reasoning_text = delta.get('reasoning_content') or delta.get('reasoning', '')
                                if reasoning_text:
                                    token_count += 1
                                    reasoning_tokens.append(reasoning_text)
                                    full_reasoning += reasoning_text
                                    if include_reasoning:
                                        print(f"[REASONING: {reasoning_text}]", end="", flush=True)

                            # Check for regular content
                            if 'content' in delta and delta['content']:
                                text_chunk = delta['content']

                        # Handle text format (some models use this)
                        elif 'text' in choice:
                            text_chunk = choice['text']

                        # Process text chunk with XML tag parsing
                        if text_chunk:
                            token_count += 1
                            buffer += text_chunk

                            # Parse XML tags in buffer
                            while True:
                                if not in_reasoning_tag:
                                    # Look for opening <reasoning> tag
                                    reasoning_start = buffer.find('<reasoning>')
                                    if reasoning_start >= 0:
                                        # Content before tag
                                        if reasoning_start > 0:
                                            content_text = buffer[:reasoning_start]
                                            content_tokens.append(content_text)
                                            full_content += content_text
                                            print(content_text, end="", flush=True)

                                        buffer = buffer[reasoning_start + 11:]  # Remove '<reasoning>'
                                        in_reasoning_tag = True
                                    else:
                                        # No opening tag found, but keep last 11 chars in buffer
                                        # in case tag is split across chunks
                                        if len(buffer) > 11:
                                            content_text = buffer[:-11]
                                            content_tokens.append(content_text)
                                            full_content += content_text
                                            print(content_text, end="", flush=True)
                                            buffer = buffer[-11:]
                                        break
                                else:
                                    # Look for closing </reasoning> tag
                                    reasoning_end = buffer.find('</reasoning>')
                                    if reasoning_end >= 0:
                                        # Reasoning content before closing tag
                                        if reasoning_end > 0:
                                            reasoning_text = buffer[:reasoning_end]
                                            reasoning_tokens.append(reasoning_text)
                                            full_reasoning += reasoning_text
                                            if include_reasoning:
                                                print(f"[REASONING: {reasoning_text}]", end="", flush=True)

                                        buffer = buffer[reasoning_end + 12:]  # Remove '</reasoning>'
                                        in_reasoning_tag = False
                                    else:
                                        # No closing tag found, keep last 12 chars in buffer
                                        if len(buffer) > 12:
                                            reasoning_text = buffer[:-12]
                                            reasoning_tokens.append(reasoning_text)
                                            full_reasoning += reasoning_text
                                            if include_reasoning:
                                                print(f"[REASONING: {reasoning_text}]", end="", flush=True)
                                            buffer = buffer[-12:]
                                        break

                        # Handle finish reason
                        if 'finish_reason' in choice and choice['finish_reason']:
                            # Flush remaining buffer
                            if buffer:
                                if in_reasoning_tag:
                                    reasoning_tokens.append(buffer)
                                    full_reasoning += buffer
                                    if include_reasoning:
                                        print(f"[REASONING: {buffer}]", end="", flush=True)
                                else:
                                    content_tokens.append(buffer)
                                    full_content += buffer
                                    print(buffer, end="", flush=True)
                                buffer = ""
                            print(f"\n[Stream stopped: {choice['finish_reason']}]")

                # Handle usage if present
                if 'usage' in chunk_json:
                    usage = chunk_json['usage']
                    print(f"\n[Token Usage - Prompt: {usage.get('prompt_tokens', 0)}, "
                          f"Completion: {usage.get('completion_tokens', 0)}, "
                          f"Total: {usage.get('total_tokens', 0)}]")

        # Flush any remaining buffer at end
        if buffer:
            if in_reasoning_tag:
                reasoning_tokens.append(buffer)
                full_reasoning += buffer
                if include_reasoning:
                    print(f"[REASONING: {buffer}]", end="", flush=True)
            else:
                content_tokens.append(buffer)
                full_content += buffer
                print(buffer, end="", flush=True)

        end_time = time.time()

        ttft = first_token_time - start_time if first_token_time else None
        total_time = end_time - start_time

        print(f"\n\n[Statistics]")
        print(f"Total Tokens Received: {token_count}")
        print(f"Reasoning Tokens: {len(reasoning_tokens)} ({len(full_reasoning)} chars)")
        print(f"Content Tokens: {len(content_tokens)} ({len(full_content)} chars)")
        if ttft is not None:
            print(f"Time to First Token (TTFT): {ttft:.4f} seconds")
        else:
            print("Time to First Token (TTFT): N/A")
        print(f"Total Time: {total_time:.4f} seconds")

        return {
            "ttft": ttft,
            "total_time": total_time,
            "reasoning_tokens": reasoning_tokens,
            "content_tokens": content_tokens,
            "full_reasoning": full_reasoning,
            "full_content": full_content,
            "token_count": token_count
        }

    except Exception as e:
        print(f"Error invoking model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Test Bedrock models using InvokeModel API"
    )
    parser.add_argument("--model", default="openai.gpt-oss-120b-1:0", help="Model ID")
    parser.add_argument("--region", default="ap-south-1", help="AWS Region")
    parser.add_argument("--prompt", default="Hi How are you?",
                       help="Prompt")
    parser.add_argument("--include-reasoning", action="store_true", default=True,
                       help="Show reasoning in output (default: True)")
    parser.add_argument("--exclude-reasoning", action="store_true",
                       help="Hide reasoning from output")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                       help="Test single reasoning effort level")
    parser.add_argument("--tier", choices=["default", "priority"],
                       help="Test single service tier")
    args = parser.parse_args()

    # Handle reasoning inclusion flag
    include_reasoning = not args.exclude_reasoning

    print(f"Bedrock Latency Test (InvokeModel API)")
    print(f"Model: {args.model}")
    print(f"Region: {args.region}")
    print(f"Include Reasoning: {include_reasoning}\n")

    # Determine test configurations
    if args.reasoning_effort:
        efforts = [args.reasoning_effort]
    elif not include_reasoning:
        efforts = [None]
    else:
        efforts = [None, "low", "medium", "high"]

    if args.tier:
        tiers = [args.tier]
    else:
        tiers = ["default", "priority"]

    results = []

    for effort in efforts:
        for tier in tiers:
            result = test_model_invokemodel(
                args.region, args.model, tier, args.prompt, effort, include_reasoning
            )

            if result:
                results.append({
                    "tier": tier.capitalize(),
                    "effort": effort,
                    "ttft": result["ttft"],
                    "total": result["total_time"],
                    "reasoning_tokens": len(result["reasoning_tokens"]),
                    "content_tokens": len(result["content_tokens"]),
                    "token_count": result["token_count"]
                })
            else:
                results.append({
                    "tier": tier.capitalize(),
                    "effort": effort,
                    "ttft": None,
                    "total": None,
                    "reasoning_tokens": 0,
                    "content_tokens": 0,
                    "token_count": 0
                })

            time.sleep(2)
            

    print("\n\n" + "=" * 120)
    print("--- SUMMARY (InvokeModel API) ---")
    print("=" * 120)
    print(f"Model ID: {args.model}")
    print(f"{'Tier':<12} | {'Reasoning':<10} | {'TTFT (s)':<10} | {'Total (s)':<10} | "
          f"{'Tokens':<8} | {'R-Tokens':<10} | {'C-Tokens':<10} | {'Improvement'}")
    print("-" * 120)

    # Group by effort level
    for effort in efforts:
        eff_str = effort if effort else "none"

        tier_results = [r for r in results if r["effort"] == effort]

        for res in tier_results:
            ttft_str = f"{res['ttft']:.4f}" if res["ttft"] else "N/A"
            tot_str = f"{res['total']:.4f}" if res["total"] else "N/A"
            tokens_str = str(res["token_count"])
            r_tokens_str = str(res["reasoning_tokens"])
            c_tokens_str = str(res["content_tokens"])

            # Calculate improvement if we have both tiers
            improv_str = "N/A"
            if res["tier"] == "Priority" and len(tiers) > 1:
                std_res = next((r for r in tier_results if r["tier"] == "Default"), None)
                if std_res and std_res["ttft"] and res["ttft"]:
                    diff = std_res["ttft"] - res["ttft"]
                    pct = (diff / std_res["ttft"]) * 100
                    improv_str = f"{diff:.4f}s ({pct:.1f}%)"

            print(f"{res['tier']:<12} | {eff_str:<10} | {ttft_str:<10} | {tot_str:<10} | "
                  f"{tokens_str:<8} | {r_tokens_str:<10} | {c_tokens_str:<10} | {improv_str}")

        print("-" * 120)

if __name__ == "__main__":
    main()
