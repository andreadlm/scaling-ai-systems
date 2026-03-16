from langchain.tools import tool


@tool
def search_emails(keywords: str, sent_after: str, sent_before: str) -> list[dict]:
    """Finds up to 10 emails matching the given keywords with date filters applied.
    Returns message IDs and matching snippets.

    Args:
        keywords: Search keywords to match against email content.
        sent_after: Filter emails sent after this date (ISO 8601, e.g. '2024-01-01').
        sent_before: Filter emails sent before this date (ISO 8601, e.g. '2024-12-31').
    """
    pass


@tool
def read_email(message_id: str) -> str:
    """Returns the full email body for the given message ID.

    Args:
        message_id: The unique identifier of the email message to retrieve.
    """
    pass


@tool
def return_final_answer(answer: str, sources: list[str]) -> dict:
    """Returns the final answer to the user's question, and the list of message IDs
    that supported the answer.

    Args:
        answer: The final answer to return to the user.
        sources: The list of message IDs that supported the answer.
    """
    pass
