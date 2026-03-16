import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", app_title="Scaling AI systems")

with app.setup:
    import marimo as mo


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
    from data.local_db import generate_database, DEFAULT_DB_PATH as db_path

    return db_path, generate_database


@app.cell
def _(generate_database):
    with mo.status.spinner(title="Preparing Enron email database…"):
        generate_database()
    return


@app.cell
def _(db_path):
    import sqlalchemy
    sqlite_engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    return (sqlite_engine,)


@app.cell
def _(emails, recipients, sqlite_engine):
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

    * `search_emails(keywords, sent_after, sent_before)`: runs a **full-text search** over subject
    and body. Returns up to 10 results, each with a `message_id` and a highlighted snippet showing
    where the keywords matched. Date filters narrow the search window and help the agent avoid reading
    stale emails. This is the agent's primary discovery tool: it tells the agent *which* emails to look at,
    not what they contain.
    """)
    return


@app.cell(hide_code=True)
def _():
    search_emails_form = (
        mo.md("""
        **`Test search_emails`**

        {search_keywords} {search_sent_after} {search_sent_before}
        """)
        .batch(
            search_keywords=mo.ui.text(placeholder="budget forecast", label="Keywords"),
            search_sent_after=mo.ui.text(placeholder="2000-01-01", label="Sent after"),
            search_sent_before=mo.ui.text(placeholder="2002-12-31", label="Sent before")
        )
        .form(show_clear_button=True, bordered=True)
    )
    search_emails_form
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * `read_email(message_id)`: fetches the **complete body** of a single email by its
    `message_id`. The agent calls this after `search_emails` to read the full content of
    a promising result.
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
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * `return_final_answer(answer, sources)`: terminates the loop. The agent calls this when
    it has enough information to answer the user's question, passing both the answer text and
    the list of `message_id`s that support it. Once called, the agent stops.
    """)
    return


if __name__ == "__main__":
    app.run()
