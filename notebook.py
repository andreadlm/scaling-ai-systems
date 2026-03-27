import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="Scaling AI systems")

with app.setup:
    import json
    import os
    import subprocess
    from dataclasses import asdict, dataclass

    import marimo as mo
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain.agents.middleware import dynamic_prompt, ModelRequest
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langfuse.langchain import CallbackHandler

    from agent.tools import search_emails, read_email
    from data.local_db import generate_database, sqlite_engine
    from data.query_iterators import load_synthetic_queries

    load_dotenv()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Scaling AI Systems: email research agent

    The goal of this project is to build an **agent** that can answer natural-language questions
    by autonomously searching through an email inbox.

    Rather than retrieving a single document, the agent must reason about which emails are
    relevant, decide what to read, and synthesise a final answer, all within a constrained
    tool-use loop.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1 · Environment

    An agent always operates inside an **environment** — the set of tools it can call and the
    data those tools expose. Here the environment is an email inbox.

    We use the [Enron email corpus](https://en.wikipedia.org/wiki/Enron_Corpus): when Enron
    collapsed in 2001 following a massive accounting scandal, ~500 K of its internal emails were
    made public during litigation. This makes it one of the few large, realistic email datasets
    available for research.

    A cleaned subset is available on Hugging Face ::logos:hugging-face-icon:: as
    [`corbt/enron-emails`](https://huggingface.co/datasets/corbt/enron-emails). We download it
    and load it into a local ::devicon:sqlite:: SQLite database.
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
    Let's preview the data — here are the 100 most recent emails with their recipients:
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
    The agent can call three tools. Together they define everything the agent is allowed to do:
    its full action space.

    Let's test each tool interactively before wiring them into the agent.

    ### `search_emails(inbox, keywords, ...)`
    Runs a **full-text search** over subject and body. Optional filters let the agent narrow
    results by sender, recipient, or date range.
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
    ### `read_email(message_id)`
    Fetches the **complete body** of a single email by its `message_id`. The agent calls this
    after `search_emails` to read the full content of a promising result.
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
    ## 2 · Agent Definition

    With the environment in place, we can define the **agent**. It is built with LangChain's `create_agent` and is given the two
    email tools as its action space and a simple system prompt.
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
    return (agent,)


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
    ## 3 · Observability

    When an agent runs, it operates autonomously: it decides which tools to call, in what
    order, and with which arguments. This opacity makes it hard to understand *why* the agent
    produced a particular answer, catch errors, or track latency and token cost — all of which
    become critical concerns as you scale from a prototype to a production system.

    An **observability platform** instruments every step of the agent's execution and records
    it as a structured **trace**: every LLM call, every tool invocation, every message
    exchanged, with timing and token counts attached. You can then drill into these traces to
    debug surprising answers, measure reasoning depth, and spot regressions across runs.

    [Langfuse](https://langfuse.com) is an open-source LLM observability platform with native
    LangChain integration. It captures full execution traces with latency breakdowns, token
    usage, and structured inputs/outputs at every level of the agent's reasoning loop.

    The quickest way to get started is via ::logos:docker-icon:: Docker Compose — see the
    [self-hosting guide](https://langfuse.com/self-hosting/deployment/docker-compose).
    """)
    return


@app.cell(hide_code=True)
def _():
    with mo.status.spinner(title="Starting Langfuse…"):
        subprocess.run(
            ["docker", "compose", "up", "-d", "langfuse-web"],
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
    ### Tracing an agent invocation

    Integrating Langfuse into a LangChain agent requires a single change: pass a
    `CallbackHandler` instance in the `config` dictionary when invoking the agent. The
    callback intercepts every internal event — LLM calls, tool calls, chain starts and ends
    — and streams them to Langfuse in real time, without any changes to the agent itself.
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

    return (ask_agent_traced,)


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
    ## 4 · Reinforcement Learning

    So far we have an agent that *works*, and an observability layer that lets us *see* what
    it does. But how do we make the agent **better**?

    **Reinforcement Learning (RL)** treats the problem as a loop:

    | RL concept | Our agent |
    |---|---|
    | **Policy** | The LLM that decides which tool to call and what answer to give |
    | **Environment** | The email database + the two tools (`search_emails`, `read_email`) |
    | **Trajectory** | One complete agent run: the full sequence of messages from user question to final answer |
    | **Reward** | A numeric score telling the policy how well it did on that trajectory |

    The RL training loop is simple in principle: run the policy many times, score each
    trajectory, and update the policy weights to make high-reward trajectories more likely.

    The critical piece is the **reward function** — it must capture not just whether the
    final answer is correct, but also whether the agent *behaved well* along the way.

    ### Reward shaping with a rubric

    A naive reward — `+1` if correct, `0` otherwise — gives the model very little signal to
    learn from. Instead we use a **rubric**: a structured scorecard that tracks intermediate
    signals during the trajectory and feeds them into a shaped reward.

    **Rubric fields:**

    | Field | Type | What it tracks |
    |---|---|---|
    | `answer_correct` | bool | Whether the final answer matches the ground truth (evaluated by an LLM-as-judge) |
    | `ever_found_right_email` | bool | Whether `search_emails` ever returned the target email among its results |
    | `ever_read_right_email` | bool | Whether the agent called `read_email` with the correct `message_id` |
    | `num_turns` | int | Total number of LLM calls in the trajectory |
    | `num_tool_calls` | int | Total number of tool invocations |
    | `error` | bool | Whether the agent raised an unhandled exception |

    **Reward tiers** — the reward has three base levels, with additive partial bonuses
    (+0.1 each) for process signals that indicate good intermediate behaviour:

    | Outcome | Base | Partial bonuses | Range |
    |---|---|---|---|
    | **Correct answer** | +1.0 | +0.1 found · +0.1 read · +0.1 efficiency | [1.0, 1.3] |
    | **Wrong answer** | −1.0 | +0.1 found · +0.1 read | [−1.0, −0.8] |
    | **No answer / error** | 0.0 | +0.1 found · +0.1 read | [0.0, 0.2] |

    The efficiency bonus rewards trajectories that solve the task in fewer turns, nudging
    the policy toward concise, targeted searches.
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

    To compute rewards we need questions with known answers. The
    [`corbt/enron_emails_sample_questions`](https://huggingface.co/datasets/corbt/enron_emails_sample_questions)
    dataset on Hugging Face ::logos:hugging-face-icon:: provides synthetic queries over the Enron corpus, each paired
    with a ground-truth answer and the `message_id`(s) of the relevant email(s).

    Each `SyntheticQuery` carries its own `inbox_address` and `query_date`, so the agent's
    context is set automatically for each evaluation run.
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
    ### Running a rollout

    A **rollout** is a single scored execution of the agent on a query: invoke the agent,
    walk the message history to populate the rubric, and compute the shaped reward.
    """)
    return


@app.cell
def _(
    Context,
    Rubric,
    agent,
    calculate_reward,
    judge_answer_correctness,
    model,
):
    def rollout(query, max_turns=10):
        rubric = Rubric()

        try:
            result = agent.invoke(
                {"messages": [("user", query.question)]},
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
