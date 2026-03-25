"""
Example usage of the enhanced Bedrock streaming API with reasoning token separation.

This demonstrates the three key features:
1. Separate tracking of reasoning tokens vs content tokens
2. Token-by-token streaming (not receiving direct response)
3. Flexibility to enable/disable reasoning at LLM level via include_reasoning parameter
   - include_reasoning=True: LLM performs reasoning
   - include_reasoning=False: LLM does NOT perform reasoning (faster, cheaper)
"""

import boto3
import time

def stream_with_reasoning_separation(client, model_id, prompt, include_reasoning=True, reasoning_effort="high"):
    """
    Example function showing how to use the enhanced streaming API.

    Args:
        client: Bedrock runtime client
        model_id: Model identifier
        prompt: Input prompt
        include_reasoning: Whether to enable reasoning at LLM level (default: True)
                          When False, the LLM will NOT perform reasoning at all
        reasoning_effort: Level of reasoning (low/medium/high) - only used if include_reasoning=True

    Returns:
        Dictionary with:
            - reasoning_tokens: List of reasoning token strings (empty if reasoning disabled)
            - content_tokens: List of content token strings
            - full_reasoning: Complete reasoning text (empty if reasoning disabled)
            - full_content: Complete content text
    """
    print(f"\n{'='*60}")
    print(f"Reasoning Enabled: {include_reasoning}")
    if include_reasoning:
        print(f"Reasoning Effort: {reasoning_effort}")
    else:
        print(f"Reasoning: DISABLED at LLM level (no reasoning_effort sent)")
    print(f"{'='*60}\n")

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    additional_fields = {
        "service_tier": "priority"
    }

    # Only add reasoning_effort if reasoning is enabled
    if include_reasoning and reasoning_effort:
        additional_fields["reasoning_effort"] = reasoning_effort

    response = client.converse_stream(
        modelId=model_id,
        messages=messages,
        additionalModelRequestFields=additional_fields
    )

    # Separate tracking for reasoning and content
    reasoning_tokens = []
    content_tokens = []
    full_reasoning = ""
    full_content = ""

    print("[STREAMING OUTPUT]")
    print("-" * 60)

    # Process streaming response token by token
    for event in response.get("stream"):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]

            # Check for reasoning content (comes in delta.reasoningContent.text)
            if "reasoningContent" in delta:
                reasoning_text = delta["reasoningContent"].get("text", "")
                if reasoning_text:
                    reasoning_tokens.append(reasoning_text)
                    full_reasoning += reasoning_text
                    # Print reasoning tokens with label
                    print(f"[REASONING: {reasoning_text}]", end="", flush=True)

            # Check for regular content (comes in delta.text)
            elif "text" in delta:
                content_text = delta.get("text", "")
                if content_text:
                    content_tokens.append(content_text)
                    full_content += content_text
                    # Print content tokens
                    print(content_text, end="", flush=True)

    print("\n" + "-" * 60)
    print(f"\n[STATISTICS]")
    print(f"  Reasoning Tokens: {len(reasoning_tokens)}")
    print(f"  Content Tokens: {len(content_tokens)}")
    print(f"  Reasoning Chars: {len(full_reasoning)}")
    print(f"  Content Chars: {len(full_content)}")

    return {
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "full_reasoning": full_reasoning,
        "full_content": full_content
    }


def main():
    # Initialize Bedrock client
    client = boto3.client("bedrock-runtime", region_name="ap-south-1")
    model_id = "openai.gpt-oss-120b-1:0"

    # Test prompt that triggers reasoning
    prompt = "Solve this logic puzzle: If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"

    print("\n" + "="*60)
    print("DEMONSTRATION: Reasoning Token Separation")
    print("="*60)

    # Example 1: With reasoning enabled (default behavior)
    print("\n\n### Example 1: include_reasoning=True (LLM performs reasoning) ###")
    result1 = stream_with_reasoning_separation(
        client=client,
        model_id=model_id,
        prompt=prompt,
        include_reasoning=True,
        reasoning_effort="high"
    )

    time.sleep(2)

    # Example 2: With reasoning disabled (no reasoning from LLM)
    print("\n\n### Example 2: include_reasoning=False (LLM does NOT perform reasoning) ###")
    result2 = stream_with_reasoning_separation(
        client=client,
        model_id=model_id,
        prompt=prompt,
        include_reasoning=False,
        reasoning_effort="high"  # This will be ignored when include_reasoning=False
    )

    # Compare results
    print("\n\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"\nWith reasoning ENABLED (include_reasoning=True):")
    print(f"  - LLM performed reasoning: {len(result1['reasoning_tokens'])} reasoning tokens generated")
    print(f"  - Total output included reasoning blocks")
    print(f"  - Slower but more thoughtful responses")

    print(f"\nWith reasoning DISABLED (include_reasoning=False):")
    print(f"  - LLM did NOT perform reasoning: {len(result2['reasoning_tokens'])} reasoning tokens (should be 0)")
    print(f"  - Only content was generated")
    print(f"  - Faster response time, lower cost")

    print("\n" + "="*60)
    print("KEY FEATURES DEMONSTRATED:")
    print("="*60)
    print("✓ 1. Separate tracking: reasoning_tokens vs content_tokens in response")
    print("✓ 2. True streaming: Token-by-token delivery with flush=True")
    print("✓ 3. Flexibility: include_reasoning parameter controls LLM reasoning behavior")
    print("     - True: LLM performs reasoning (reasoning_effort parameter used)")
    print("     - False: LLM skips reasoning entirely (faster, cheaper)")


if __name__ == "__main__":
    main()
