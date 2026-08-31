from agents.stream_filter import FinalAnswerStreamFilter


def test_marker_in_single_chunk_emits_tail():
    f = FinalAnswerStreamFilter()
    out = f.feed("Thought: I should answer now.\nFinal Answer: Revenue grew 12%.")
    assert out == "Revenue grew 12%."


def test_text_before_marker_is_not_emitted():
    f = FinalAnswerStreamFilter()
    assert f.feed("Thought: I need to search the knowledge base.\n") == ""
    assert f.feed("Action: search_documents\n") == ""


def test_chunks_after_marker_pass_through_verbatim():
    f = FinalAnswerStreamFilter()
    f.feed("Final Answer: The company")
    assert f.feed(" reported **$5B**") == " reported **$5B**"
    assert f.feed(" in revenue.") == " in revenue."


def test_marker_split_across_two_chunks():
    f = FinalAnswerStreamFilter()
    assert f.feed("Thought: done.\nFinal An") == ""
    out = f.feed("swer: The answer is 42.")
    assert out == "The answer is 42."


def test_marker_split_across_three_chunks():
    f = FinalAnswerStreamFilter()
    assert f.feed("Fin") == ""
    assert f.feed("al Answ") == ""
    assert f.feed("er: Done.") == "Done."


def test_bold_marker_variant():
    f = FinalAnswerStreamFilter()
    out = f.feed("Thought: ok\n**Final Answer:** The total is **$9M**.")
    assert out == "The total is **$9M**."


def test_lowercase_marker_variant():
    f = FinalAnswerStreamFilter()
    out = f.feed("final answer: yes.")
    assert out == "yes."


def test_flush_no_marker_strips_reasoning_lines():
    f = FinalAnswerStreamFilter()
    f.feed("Thought: I should search.\n")
    f.feed("Action: search_documents\n")
    f.feed('Action Input: {"query": "revenue"}\n')
    f.feed("Observation: [doc.pdf] Revenue was $5B.\n")
    f.feed("The revenue was **$5B** in 2024.\n")
    assert f.flush() == "The revenue was **$5B** in 2024."


def test_flush_no_marker_no_reasoning_returns_full_text():
    f = FinalAnswerStreamFilter()
    f.feed("The answer is 42.")
    assert f.flush() == "The answer is 42."


def test_flush_empty_stream_returns_fallback_message():
    f = FinalAnswerStreamFilter()
    assert f.flush() == FinalAnswerStreamFilter.FALLBACK_MESSAGE


def test_flush_only_reasoning_returns_fallback_message():
    f = FinalAnswerStreamFilter()
    f.feed("Thought: hmm\nAction: search_documents\n")
    assert f.flush() == FinalAnswerStreamFilter.FALLBACK_MESSAGE


def test_flush_after_streaming_returns_empty():
    f = FinalAnswerStreamFilter()
    f.feed("Final Answer: Done.")
    assert f.flush() == ""


def test_reset_returns_to_buffering_with_fresh_buffer():
    f = FinalAnswerStreamFilter()
    f.feed("Final Answer: first task answer.")
    f.reset()
    assert f.feed("Thought: starting second task\n") == ""
    assert f.feed("Final Answer: second answer.") == "second answer."


def test_clean_final_answer_removes_reasoning_and_json_residue():
    f = FinalAnswerStreamFilter()
    cleaned = f.clean_final_answer(
        "Final Answer: Revenue grew **12%**.\n"
        "Thought: I should verify one more thing.\n"
        "Action: search_documents\n"
        '{ "query": "revenue" }\n'
        "Tool Output: hidden detail\n"
        "The cited filing supports this answer."
    )
    assert cleaned == (
        "Revenue grew **12%**.\n"
        "The cited filing supports this answer."
    )


def test_clean_final_answer_preserves_source_citation_lines():
    f = FinalAnswerStreamFilter()
    cleaned = f.clean_final_answer(
        "Final Answer:\n"
        "[apple 10-k 2024.pdf]\n"
        "Revenue was **$391B**."
    )
    assert cleaned == "[apple 10-k 2024.pdf]\nRevenue was **$391B**."


def test_clean_final_answer_strips_think_block():
    f = FinalAnswerStreamFilter()
    cleaned = f.clean_final_answer("<think>I need to calculate</think>Revenue grew **12%**.")
    assert cleaned == "Revenue grew **12%**."


def test_clean_final_answer_strips_multiline_think_block():
    f = FinalAnswerStreamFilter()
    cleaned = f.clean_final_answer("<think>Let me break this down.</think>\nThe answer is **42**.")
    assert cleaned == "The answer is **42**."


def test_feed_strips_think_tags_before_marker():
    f = FinalAnswerStreamFilter()
    assert f.feed("<think>Hmm, let me think...</think>\n") == ""
    result = f.feed("Final Answer: Revenue was **$5B**.")
    assert result == "Revenue was **$5B**."
    assert f.flush() == ""


def test_feed_strips_think_tags_after_marker():
    f = FinalAnswerStreamFilter()
    f.feed("Final Answer: <think> Reasoning here </think>The answer is 42.")
    out = f._strip_think_tags("The answer is 42.")
    assert out == "The answer is 42."


def test_clean_final_answer_strips_tool_trace():
    f = FinalAnswerStreamFilter()
    raw = (
        '{"action": "search_documents", "action_input": {"query": "revenue"}}'
        '}The revenue was **$5B** in 2024.'
    )
    cleaned = f.clean_final_answer(raw)
    assert cleaned == "The revenue was **$5B** in 2024."


def test_clean_final_answer_strips_multiline_tool_trace():
    f = FinalAnswerStreamFilter()
    raw = (
        '{"action": "search_documents",\n'
        '"action_input": {"query": "revenue"}}\n'
        '{"observation": "Revenue was $5B"}'
        "}Amazon revenue in 2024 was **$637B**."
    )
    cleaned = f.clean_final_answer(raw)
    assert cleaned == "Amazon revenue in 2024 was **$637B**."


def test_clean_final_answer_preserves_answer_with_numbers():
    f = FinalAnswerStreamFilter()
    cleaned = f.clean_final_answer("Amazon revenue was **$637,959** million in 2024.")
    assert cleaned == "Amazon revenue was **$637,959** million in 2024."
