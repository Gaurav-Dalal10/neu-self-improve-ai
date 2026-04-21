from final_project_part2.agentflow_wrapper.engine_clients import OpenAICompatibleClient


def test_engine_client_payloads():
    c = OpenAICompatibleClient(base_url="https://example.com", api_key="k", model="m")
    payload = c._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        top_p=0.9,
        max_tokens=64,
        stop=["<END>"],
    )
    assert payload["model"] == "m"
    assert payload["messages"][0]["content"] == "hi"
    assert payload["stop"] == ["<END>"]
