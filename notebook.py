import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", app_title="Scaling AI systems")

with app.setup:
    import marimo as mo
    import datetime
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import dynamic_prompt, ModelRequest
    from langchain_openai import ChatOpenAI

    from data.local_db import generate_database, sqlite_engine
    from agent.tools import search_emails, read_email


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
        base_url="http://localhost:4141/v1",
        api_key="",
    )

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
    ).with_config(
        {"recursion_limit": 10}
    )
    return Context, agent


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
def _(Context, agent, agent_form):
    mo.stop(
        agent_form.value is None,
        mo.md("_Submit the form to ask the agent a question._"),
    )

    _v = agent_form.value

    with mo.status.spinner(title="Agent is thinking…"):
        _result = agent.invoke(
            {"messages": [("user", _v["question"])]},
            context=Context(
                inbox=_v["inbox"], 
                date=str(_v["date"])
            ),
        )

    _answer = _result["messages"][-1].content

    mo.vstack([
        mo.md(f"**Question:** {_v['question']}"),
        mo.md("---"),
        mo.md(_answer),
    ])
    return


if __name__ == "__main__":
    app.run()
