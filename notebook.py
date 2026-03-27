import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="Scaling AI systems")

with app.setup:
    import json
    import os
    import subprocess
    from dataclasses import asdict, dataclass
    import asyncio

    import marimo as mo
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain.agents.middleware import dynamic_prompt, ModelRequest
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langfuse.langchain import CallbackHandler
    import art
    from art.langgraph import init_chat_model
    from art.local import LocalBackend

    from agent.tools import search_emails, read_email
    from data.local_db import generate_database, sqlite_engine
    from data.types import SyntheticQuery
    from data.query_iterators import load_synthetic_queries

    load_dotenv()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Scaling AI Systems: email research agent

    This project walks through the full lifecycle of building and improving an **LLM-powered agent**
    that answers natural-language questions by autonomously searching an email inbox.

    Rather than retrieving a single document, the agent must reason about which emails are
    relevant, decide what to read, and synthesise a final answer — all within a constrained
    tool-use loop. We will build it from scratch, instrument it for observability, and then
    train it to get better using reinforcement learning.

    The notebook is structured around three progressive steps:

    1. **Agent** — define the environment (a real email corpus), give the agent tools to interact
       with it, and wire everything together into a working LangChain agent.
    2. **Observability** — instrument every agent run with [Langfuse](https://langfuse.com) to
       capture full execution traces. Visibility into agent behaviour is a prerequisite for
       systematic improvement.
    3. **Training** — score each run with a shaped reward rubric and feed those scores back to
       the model via [OpenPipe ART](https://art.openpipe.ai) reinforcement learning, closing the
       loop from a working prototype to an improving policy.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1 · Agent

    An agent always operates inside an **environment** — the set of tools it can call and the
    data those tools expose. Before we can build anything, we need to set that environment up.

    We'll use the [Enron email corpus](https://en.wikipedia.org/wiki/Enron_Corpus): when Enron
    collapsed in 2001 following a massive accounting scandal, ~500 K of its internal emails were
    made public during litigation. It's one of the few large, realistic email datasets available
    for research, which makes it perfect for our purposes.

    Let's start by downloading a cleaned subset from Hugging Face and loading it into a local
    ::devicon:sqlite:: SQLite database.
    """)
    return


@app.cell
def _():
    with mo.status.spinner(title="Preparing Enron email database…"):
        generate_database()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Data

    Let's preview what we're working with. Below are the 100 most recent emails in the corpus,
    with subject, sender, recipients and body. Notice the range of topics — internal memos,
    trading discussions, logistics — which makes this a genuinely challenging retrieval problem.
    """)
    return


@app.cell
def _(emails, recipients):
    _df = mo.sql(
        f"""
        SELECT
            e.message_id,
            e.subject,
            e.from_address,
            e.date,
            e.body,
            GROUP_CONCAT(CASE WHEN r.recipient_type = 'to'  THEN r.recipient_address END, ', ') AS to_addresses,
            GROUP_CONCAT(CASE WHEN r.recipient_type = 'cc'  THEN r.recipient_address END, ', ') AS cc_addresses,
            GROUP_CONCAT(CASE WHEN r.recipient_type = 'bcc' THEN r.recipient_address END, ', ') AS bcc_addresses
        FROM emails e
        LEFT JOIN recipients r ON r.email_id = e.id
        GROUP BY e.id
        ORDER BY e.date DESC
        LIMIT 100
        """,
        engine=sqlite_engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tools

    The agent has access to exactly two tools — everything it can do goes through these.
    Together they define the agent's full **action space**: if a piece of information isn't
    reachable via these two tools, the agent simply cannot find it.

    Let's look at each tool and try them interactively.

    #### `search_emails(inbox, keywords, ...)`
    Runs a full-text search over subject and body. Optional filters let the agent narrow results
    by sender, recipient, or date range. This is typically the agent's first move: cast a wide
    net, then decide what to read in full.
    """)
    return


@app.cell(hide_code=True)
def _():
    search_emails_form = (
        mo.md("""
        **`Test search_emails`**

        {search_inbox} {search_keywords}

        {search_from_addr} {search_to_addr}

        {search_sent_after} {search_sent_before}
        """)
        .batch(
            search_inbox=mo.ui.text(placeholder="user@enron.com", label="Inbox (required)"),
            search_keywords=mo.ui.text(placeholder="budget forecast", label="Keywords (space-separated, required)"),
            search_from_addr=mo.ui.text(placeholder="sender@enron.com", label="From (optional)"),
            search_to_addr=mo.ui.text(placeholder="recipient@enron.com", label="To (optional)"),
            search_sent_after=mo.ui.text(placeholder="2000-01-01", label="Sent after (optional)"),
            search_sent_before=mo.ui.text(placeholder="2002-12-31", label="Sent before (optional)"),
        )
        .form(show_clear_button=True, bordered=True)
    )
    search_emails_form
    return (search_emails_form,)


@app.cell(hide_code=True)
def _(search_emails_form):
    mo.stop(
        search_emails_form.value is None,
        mo.md("_Submit the form to see results._"),
    )

    _v = search_emails_form.value

    _results = search_emails.invoke({
        "inbox": _v["search_inbox"],
        "keywords": _v["search_keywords"].split() if _v["search_keywords"] else [],
        "from_addr": _v["search_from_addr"] or None,
        "to_addr": _v["search_to_addr"] or None,
        "sent_after": _v["search_sent_after"] or None,
        "sent_before": _v["search_sent_before"] or None,
    })

    mo.ui.table(
        [{"message_id": r.message_id, "snippet": mo.Html(r.snippet)} for r in _results],
        label=f"{len(_results)} result(s)",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### `read_email(message_id)`
    Fetches the complete body of a single email by its `message_id`. The agent calls this after
    `search_emails` to read the full content of a promising result — because search only returns
    a short snippet, not the full text.

    Let's test it with a `message_id` returned by the search above.
    """)
    return


@app.cell(hide_code=True)
def _():
    read_email_form = (
        mo.md("""
        **`Test read_email`**

        {message_id}
        """)
        .batch(
            message_id=mo.ui.text(placeholder="<message_id>", label="Message ID")
        )
        .form(show_clear_button=True, bordered=True)
    )
    read_email_form
    return (read_email_form,)


@app.cell(hide_code=True)
def _(read_email_form):
    mo.stop(
        read_email_form.value is None,
        mo.md("_Submit the form to see results._"),
    )

    _email = read_email.invoke({"message_id": read_email_form.value["message_id"]})

    mo.vstack([
        mo.md(f"**Subject:** {_email.subject or '(no subject)'}"),
        mo.md(f"**From:** {_email.from_address or '(unknown)'}"),
        mo.md(f"**Date:** {_email.date}"),
        mo.md(f"**To:** {', '.join(_email.to_addresses) or '(none)'}"),
        mo.md(f"**CC:** {', '.join(_email.cc_addresses) or '(none)'}"),
        mo.md(f"**BCC:** {', '.join(_email.bcc_addresses) or '(none)'}"),
        mo.md("---"),
        mo.md(_email.body or '(no body)'),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Agent definition

    With the environment and tools in place, let's wire everything together into an agent.

    We use LangChain's `create_agent` with a `dynamic_prompt` middleware that injects
    runtime context — the user's inbox address and the current date — into the system prompt
    at each invocation. This is important: without the inbox address the agent wouldn't know
    *whose* emails to search, and without the date it couldn't reason about relative time
    references like "last week".
    """)
    return


@app.cell
def _():
    @dataclass
    class Context:
        inbox: str
        date: str

    model = ChatOpenAI(
        model="gpt-5-mini",
        base_url=os.environ.get("OPENAI_BASE_URL", ""),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return Context, model


@app.cell
def _(Context, model):
    @dynamic_prompt
    def build_prompt(request: ModelRequest[Context]) -> str:
        return f"""
            You are an email search agent. You are given a user query and a list of tools
            you can use to search the user's email.
            Use the tools to search the user's emails and find the answer to the user's query.
            You may take multiple turns to find the answer — if your first search doesn't find
            the answer, try with different keywords.

            User's email address is {request.runtime.context.inbox}.
            Today's date is {request.runtime.context.date}.
        """

    agent = create_agent(
        model,
        tools=[search_emails, read_email],
        middleware=[build_prompt],
        context_schema=Context,
    )
    return agent, build_prompt


@app.cell
def _(Context, agent):
    def ask_agent(question: str, inbox: str, date: str) -> str:
        result = agent.invoke(
            {"messages": [("user", question)]},
            context=Context(inbox=inbox, date=date),
        )
        return result["messages"][-1].content

    return (ask_agent,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's try the agent on a free-form question. The agent will autonomously decide which tools
    to call, in what order, and with which arguments — we just give it a question and wait for
    an answer.
    """)
    return


@app.cell(hide_code=True)
def _():
    agent_form = (
        mo.md("""
        **Ask the agent**

        {question}

        {inbox} {date}
        """)
        .batch(
            question=mo.ui.text(
                placeholder="When is Shari's move to Portland targeted for?",
                label="Question",
                full_width=True,
            ),
            inbox=mo.ui.text(
                value="tim.belden@enron.com",
                label="Inbox (your email address)",
            ),
            date=mo.ui.date(
                value="2000-12-30",
                label="Date",
            ),
        )
        .form(show_clear_button=True, bordered=True)
    )
    agent_form
    return (agent_form,)


@app.cell(hide_code=True)
def _(agent_form, ask_agent):
    mo.stop(
        agent_form.value is None,
        mo.md("_Submit the form to ask the agent a question._"),
    )

    _v = agent_form.value

    with mo.status.spinner(title="Agent is thinking…"):
        _answer = ask_agent(
            question=_v["question"],
            inbox=_v["inbox"],
            date=str(_v["date"]),
        )

    mo.vstack([
        mo.md(f"**Question:** {_v['question']}"),
        mo.md("---"),
        mo.md(_answer),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2 · Observability

    The agent works — but right now it's a black box. When it runs, it decides autonomously which
    tools to call, in what order, and with which arguments. It's hard to understand *why* it produced
    a particular answer, catch errors, or track latency and token cost. All of these become critical
    as you move from a prototype to a production system.

    What we need is an **observability platform**: a tool that instruments every step of the agent's
    execution and records it as a structured **trace** — every LLM call, every tool invocation,
    every message exchanged, with timing and token counts attached.

    Let's set up [Langfuse](https://langfuse.com), an open-source LLM observability platform with
    native LangChain integration. We'll run it locally via ::logos:docker-icon:: Docker Compose.
    """)
    return


@app.cell(hide_code=True)
def _():
    with mo.status.spinner(title="Starting Langfuse…"):
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            check=True,
        )
    _user = os.environ.get("LANGFUSE_INIT_USER_EMAIL", "")
    _password = os.environ.get("LANGFUSE_INIT_USER_PASSWORD", "")
    mo.callout(
        mo.md(
            f"Langfuse is starting → open [http://localhost:3000](http://localhost:3000) to verify."
            f"\n\n**Username:** `{_user}`"
            f"\n**Password:** `{_password}`"
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tracing

    Integrating Langfuse into a LangChain agent requires a single change: pass a `CallbackHandler`
    instance in the `config` dictionary when invoking the agent. The callback intercepts every
    internal event — LLM calls, tool calls, chain starts and ends — and streams them to Langfuse
    in real time, without any modification to the agent itself.

    Let's run the same question as before and inspect the trace in the Langfuse UI.
    """)
    return


@app.cell
def _(Context, agent):
    langfuse_handler = CallbackHandler()

    def ask_agent_traced(question: str, inbox: str, date: str) -> str:
        result = agent.invoke(
            {"messages": [("user", question)]},
            config={"callbacks": [langfuse_handler]},
            context=Context(inbox=inbox, date=date),
        )

        return result["messages"][-1].content

    return ask_agent_traced, langfuse_handler


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Same question as before — this time with tracing enabled. After submitting, open the
    Langfuse UI to see the full trace: each tool call, each LLM turn, and how long each step took.
    """)
    return


@app.cell(hide_code=True)
def _():
    traced_agent_form = (
        mo.md("""
        **Ask the agent (traced)**

        {question}

        {inbox} {date}
        """)
        .batch(
            question=mo.ui.text(
                placeholder="When is Shari's move to Portland targeted for?",
                label="Question",
                full_width=True,
            ),
            inbox=mo.ui.text(
                value="tim.belden@enron.com",
                label="Inbox (your email address)",
            ),
            date=mo.ui.date(
                value="2000-12-30",
                label="Date",
            ),
        )
        .form(show_clear_button=True, bordered=True)
    )
    traced_agent_form
    return (traced_agent_form,)


@app.cell(hide_code=True)
def _(ask_agent_traced, traced_agent_form):
    mo.stop(
        traced_agent_form.value is None,
        mo.md("_Submit the form to ask the agent a question._"),
    )

    _v = traced_agent_form.value
    _host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    _project_id = os.environ.get("LANGFUSE_INIT_PROJECT_ID", "")

    with mo.status.spinner(title="Agent is thinking…"):
        _answer = ask_agent_traced(
            question=_v["question"],
            inbox=_v["inbox"],
            date=str(_v["date"]),
        )

    mo.vstack([
        mo.md(f"**Question:** {_v['question']}"),
        mo.md("---"),
        mo.md(_answer),
        mo.callout(
            mo.md(f"Trace recorded → open the [Langfuse UI]({_host}/project/{_project_id}/traces) to inspect it."),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3 · Training

    We have a working agent and a platform to observe it. But how do we make it **better**?

    **Reinforcement Learning (RL)** treats the problem as a loop: run the policy, score the
    outcome with a reward signal, and update the weights to make high-reward trajectories more
    likely. Repeat. Over many iterations, the model learns which search strategies lead to
    correct answers.

    The key design decision is the **reward function**. A naive binary reward — `+1` if the
    final answer is correct, `0` otherwise — gives the model very little signal: it doesn't
    know *why* it failed, or how close it got. Instead, let's use **reward shaping**: a rubric
    that tracks intermediate signals during the trajectory and adds partial bonuses for
    good intermediate behaviour, even when the final answer is wrong.

    ### Reward shaping

    | Outcome | Base | Partial bonuses | Range |
    |---|---|---|---|
    | **Correct answer** | +1.0 | found · read · efficiency | [1.0, 1.3] |
    | **Wrong answer** | −1.0 | found · read | [−1.0, −0.8] |
    | **No answer / error** | 0.0 | found · read | [0.0, 0.2] |

    The *found* and *read* bonuses reward the agent for locating the right email even if it
    ultimately gives a wrong answer. The *efficiency* bonus nudges it toward concise, targeted
    searches rather than brute-force browsing.
    """)
    return


@app.cell
def _():
    @dataclass
    class Rubric:
        answer_correct: bool = False
        ever_found_right_email: bool = False
        ever_read_right_email: bool = False
        num_turns: int = 0
        num_tool_calls: int = 0
        error: bool = False

        def to_dict(self) -> dict:
            return asdict(self)

    def calculate_reward(rubric: Rubric, max_turns: int) -> float:
        partial = 0.0
        partial += 0.1 if rubric.ever_found_right_email else 0  # agent found the needle
        partial += 0.1 if rubric.ever_read_right_email else 0   # agent read the needle

        if rubric.error:
            return 0.0 + partial

        if rubric.answer_correct:
            efficiency = 0.1 * (1 - rubric.num_turns / max_turns) if max_turns > 0 else 0
            return 1.0 + partial + efficiency

        return -1.0 + partial

    def judge_answer_correctness(question: str, answer: str, ground_truth: str, model) -> bool:
        response = model.invoke([
            {
                "role": "system",
                "content": (
                    "You will be given a question and two different answers: the correct answer "
                    "and the answer given by an AI. Determine if the AI answer is correct. "
                    "Return only the word True or False, no other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Correct answer: {ground_truth}\n"
                    f"AI answer: {answer}"
                ),
            },
        ])
        return response.content.strip().lower().startswith("t")

    return Rubric, calculate_reward, judge_answer_correctness


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Evaluation dataset

    To compute rewards we need questions with **known answers** — a ground-truth dataset to
    measure the agent against.

    We'll use [`corbt/enron_emails_sample_questions`](https://huggingface.co/datasets/corbt/enron_emails_sample_questions)
    from Hugging Face ::logos:hugging-face-icon::, a set of synthetic queries over the Enron corpus.
    Each entry contains the question, the expected answer, the inbox address to search in, and
    the `message_id`(s) of the relevant emails. This last field is especially useful: it lets us
    check not just whether the final answer is correct, but whether the agent ever *found* the
    right email during its search.
    """)
    return


@app.cell
def _():
    with mo.status.spinner(title="Loading evaluation queries…"):
        eval_queries = load_synthetic_queries(split="test", limit=20)
    return (eval_queries,)


@app.cell(hide_code=True)
def _(eval_queries):
    mo.ui.table(
        [
            {
                "id": q.id,
                "inbox": q.inbox_address,
                "question": q.question,
                "answer": q.answer,
                "message_ids": ", ".join(q.message_ids),
            }
            for q in eval_queries
        ],
        label=f"{len(eval_queries)} evaluation queries",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Rollout

    A **rollout** is a single scored execution of the agent on one query. Let's walk through
    what happens:

    1. Invoke the agent with the question and context.
    2. Walk the resulting message history to populate the rubric — checking which tool calls
       were made, whether the right email was found or read, and how many turns it took.
    3. Use an LLM-as-judge to evaluate whether the final answer matches the ground truth.
    4. Compute the shaped reward from the rubric.

    Let's run a rollout on a query from the evaluation set and inspect the full result.
    """)
    return


@app.cell
def _(
    Context,
    Rubric,
    agent,
    calculate_reward,
    judge_answer_correctness,
    langfuse_handler,
    model,
):
    def rollout(query, max_turns=10):
        rubric = Rubric()

        try:
            result = agent.invoke(
                {"messages": [("user", query.question)]},
                config={"callbacks": [langfuse_handler]},
                context=Context(inbox=query.inbox_address, date=query.query_date),
            )
            messages = result["messages"]
        except Exception:
            rubric.error = True
            return rubric, calculate_reward(rubric, max_turns), []

        correct_ids = set(query.message_ids)

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    rubric.num_tool_calls += 1
                    if tc["name"] == "read_email":
                        if tc["args"].get("message_id", "") in correct_ids:
                            rubric.ever_read_right_email = True
            elif isinstance(msg, ToolMessage) and msg.name == "search_emails":
                try:
                    results = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(results, list):
                        for r in results:
                            if isinstance(r, dict) and r.get("message_id") in correct_ids:
                                rubric.ever_found_right_email = True
                except (json.JSONDecodeError, TypeError):
                    pass

        rubric.num_turns = sum(1 for m in messages if isinstance(m, AIMessage))

        final_ai = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
        if final_ai:
            rubric.answer_correct = judge_answer_correctness(
                query.question, final_ai[-1].content, query.answer, model
            )
        else:
            # No textual answer means the agent ended with a tool call — treat as error.
            rubric.error = True

        return rubric, calculate_reward(rubric, max_turns), messages

    return (rollout,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's pick a query from the dataset and observe the full trajectory — the agent's
    reasoning steps, the tools it called, and the final rubric breakdown with its reward.
    """)
    return


@app.cell(hide_code=True)
def _(eval_queries):
    _query_options = {f"[{q.id}] {q.question[:80]}": i for i, q in enumerate(eval_queries)}

    rollout_form = (
        mo.md("""
        **Run a rollout**

        {query_selector}
        """)
        .batch(
            query_selector=mo.ui.dropdown(
                options=_query_options,
                label="Query",
                full_width=True,
            ),
        )
        .form(show_clear_button=True, bordered=True)
    )
    rollout_form
    return (rollout_form,)


@app.cell(hide_code=True)
def _(eval_queries, rollout, rollout_form):
    mo.stop(
        rollout_form.value is None or rollout_form.value["query_selector"] is None,
        mo.md("_Select a query and submit to run a rollout._"),
    )

    _idx = rollout_form.value["query_selector"]
    _query = eval_queries[_idx]

    with mo.status.spinner(title="Running rollout…"):
        _rubric, _reward, _messages = rollout(_query)

    mo.vstack([
        mo.md(f"**Question:** {_query.question}"),
        mo.md(f"**Ground truth:** {_query.answer}"),
        mo.md(f"**Agent answer:** {_messages[-1].content if _messages else '(no answer)'}"),
        mo.md("---"),
        mo.md("#### Rubric"),
        mo.ui.table(
            [{"signal": k, "value": v} for k, v in _rubric.to_dict().items()],
            label="Rubric breakdown",
        ),
        mo.callout(
            mo.md(f"**Reward: `{_reward:.2f}`**"),
            kind="success" if _reward >= 1.0 else ("warn" if _reward >= 0 else "danger"),
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
