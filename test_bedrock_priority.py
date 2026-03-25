import boto3
import time
import argparse

def test_model(client, model_id, tier, prompt, reasoning_effort=None, include_reasoning=True):
    """
    Test model with streaming response, separating reasoning and content tokens.

    NOTE: The openai.gpt-oss-120b-1:0 model ALWAYS generates reasoning tokens.
    This is part of the model's architecture and cannot be disabled.

    Args:
        client: Bedrock runtime client
        model_id: Model identifier
        tier: Service tier (priority/default)
        prompt: Input prompt
        reasoning_effort: Reasoning effort level (low/medium/high)
                         Controls the depth/length of reasoning
        include_reasoning: Whether to DISPLAY reasoning to user (default: True)
                          - True: Show reasoning in output
                          - False: Hide reasoning from output
                          NOTE: Model still generates reasoning internally regardless

    Returns:
        Dictionary containing:
            - ttft: Time to first token
            - total_time: Total completion time
            - reasoning_tokens: List of reasoning tokens (always populated)
            - content_tokens: List of content tokens
            - full_reasoning: Complete reasoning text (always present)
            - full_content: Complete content text
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

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    start_time = time.time()
    first_token_time = None

    additional_fields = {}
    # Only add reasoning_effort if reasoning is enabled
    if effective_reasoning_effort:
        additional_fields["reasoning_effort"] = effective_reasoning_effort

    # Debug output
    import os
    if os.environ.get("DEBUG_BEDROCK"):
        print(f"\n[DEBUG] API Call Parameters:")
        print(f"  - Model: {model_id}")
        print(f"  - Service Tier: {tier}")
        print(f"  - Additional Fields: {additional_fields}")
        print(f"  - Prompt: {prompt[:50]}...")
        print()

    try:
        if tier == "priority":
            additional_fields["service_tier"] = "priority"
            response = client.converse_stream(
                modelId=model_id,
                messages=messages,
                additionalModelRequestFields=additional_fields
            )
        else:
            # Let's try serviceTier arg first
            try:
                kwargs = {
                    "modelId": model_id,
                    "messages": messages,
                    "serviceTier": {"type": tier}
                }
                if additional_fields:
                    kwargs["additionalModelRequestFields"] = additional_fields
                response = client.converse_stream(**kwargs)
            except Exception as e:
                print(f"Failed with serviceTier argument: {e}")
                # Fallback to no argument which defaults to standard/default
                print("Falling back to omitting serviceTier argument...")
                kwargs = {
                    "modelId": model_id,
                    "messages": messages
                }
                if additional_fields:
                    kwargs["additionalModelRequestFields"] = additional_fields
                response = client.converse_stream(**kwargs)

        # Separate tracking for reasoning and content tokens
        reasoning_tokens = []
        content_tokens = []
        full_reasoning = ""
        full_content = ""

        print("\n[Streaming Tokens]")
        token_count = 0

        # Debug mode
        import os
        debug_mode = os.environ.get("DEBUG_BEDROCK")

        for event in response.get("stream"):
            if debug_mode and token_count < 5:  # Show first 5 events in debug mode
                print(f"\n[DEBUG Event {token_count}]: {list(event.keys())}")
                if "contentBlockDelta" in event:
                    print(f"[DEBUG] Delta text: '{event['contentBlockDelta']['delta'].get('text', '')}'")

            # Track first token timing
            if first_token_time is None and "contentBlockDelta" in event:
                first_token_time = time.time()

            # Process content deltas (token-by-token)
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]

                # Check for reasoning content (comes in delta.reasoningContent.text)
                if "reasoningContent" in delta:
                    reasoning_text = delta["reasoningContent"].get("text", "")
                    if reasoning_text:
                        token_count += 1
                        reasoning_tokens.append(reasoning_text)
                        full_reasoning += reasoning_text
                        # Print reasoning tokens only if include_reasoning is True
                        if include_reasoning:
                            print(f"[REASONING: {reasoning_text}]", end="", flush=True)

                # Check for regular content (comes in delta.text)
                elif "text" in delta:
                    content_text = delta.get("text", "")
                    if content_text:
                        token_count += 1
                        content_tokens.append(content_text)
                        full_content += content_text
                        # Print content tokens
                        print(content_text, end="", flush=True)

            # Handle other event types (e.g., metadata, stopReason)
            if "messageStop" in event:
                print(f"\n[Stream stopped: {event['messageStop'].get('stopReason', 'unknown')}]")

            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    usage = metadata["usage"]
                    print(f"\n[Token Usage - Input: {usage.get('inputTokens', 0)}, Output: {usage.get('outputTokens', 0)}]")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai.gpt-oss-120b-1:0", help="Model ID")
    parser.add_argument("--region", default="ap-south-1", help="AWS Region")
    parser.add_argument("--prompt", default="What are all design pattern while writing production ready", help="Prompt")
    parser.add_argument("--include-reasoning", action="store_true", default=True, help="Show reasoning in output (default: True). Note: Model always generates reasoning internally.")
    parser.add_argument("--exclude-reasoning", action="store_true", help="Hide reasoning from output (model still generates it internally)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="Test single reasoning effort level")
    parser.add_argument("--tier", choices=["default", "priority"], help="Test single service tier")
    args = parser.parse_args()

    # Handle reasoning inclusion flag
    include_reasoning = not args.exclude_reasoning

    client = boto3.client("bedrock-runtime", region_name=args.region)

    print(f"Bedrock Latency Test\nModel: {args.model}\nRegion: {args.region}")
    print(f"Include Reasoning: {include_reasoning}\n")

    # Determine test configurations
    if args.reasoning_effort:
        efforts = [args.reasoning_effort]
    elif not include_reasoning:
        # If reasoning is disabled, only test with None
        efforts = [None]
    else:
        # Test all levels
        efforts = [None, "low", "medium", "high"]

    if args.tier:
        tiers = [args.tier]
    else:
        tiers = ["default", "priority"]

    results = []

    for effort in efforts:
        for tier in tiers:
            result = test_model(client, args.model, tier, args.prompt, effort, include_reasoning)

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
    print("--- SUMMARY ---")
    print("=" * 120)
    print(f"Model ID: {args.model}")
    print(f"{'Tier':<12} | {'Reasoning':<10} | {'TTFT (s)':<10} | {'Total (s)':<10} | {'Tokens':<8} | {'R-Tokens':<10} | {'C-Tokens':<10} | {'Improvement'}")
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

            print(f"{res['tier']:<12} | {eff_str:<10} | {ttft_str:<10} | {tot_str:<10} | {tokens_str:<8} | {r_tokens_str:<10} | {c_tokens_str:<10} | {improv_str}")

        print("-" * 120)

if __name__ == "__main__":
    main()
