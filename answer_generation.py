import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from retrieval_pipeline import retrieve

load_dotenv()


def generate_answer(query, k=5):
    """Retrieve relevant chunks and ask Claude to answer using only those chunks."""
    relevant_docs = retrieve(query, k=k)

    # Combine the query and the relevant documents' contents
    combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer this question based on provided documents.
"""

    # Create a Claude model via the company qGenie gateway.
    # base_url + token come from .env. qGenie uses Bearer auth, so the token is
    # passed as an Authorization header - nothing leaves Qualcomm's approved channel.
    model = ChatAnthropic(
        model="claude-sonnet-4-6",
        base_url=os.environ["ANTHROPIC_BASE_URL"],
        api_key="placeholder",  # satisfies the SDK's auth check; real auth is the header below
        default_headers={"Authorization": f"Bearer {os.environ['ANTHROPIC_AUTH_TOKEN']}"},
    )

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)
    return result.content


if __name__ == "__main__":
    query = "Which island does SpaceX lease for its launches in the pacific?"
    answer = generate_answer(query)

    print(f"User query: {query}")
    print("\n--- Generated Response ---")
    print(answer)
