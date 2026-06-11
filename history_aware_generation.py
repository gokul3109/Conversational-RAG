import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic

from retrieval_pipeline import retrieve


# Load env
load_dotenv()

# Set up AI model
model = ChatAnthropic(
        model="claude-sonnet-4-6",
        base_url=os.environ["ANTHROPIC_BASE_URL"],
        api_key="placeholder",  # satisfies the SDK's auth check; real auth is the header below
        default_headers={"Authorization": f"Bearer {os.environ['ANTHROPIC_AUTH_TOKEN']}"},
)

# Store our conversation as messages
chat_history = []


def log_chat_history():
    """Print chat_history readably so you can see what Step 1 rewrites against."""
    print(f"--- chat_history ({len(chat_history)} messages) ---")
    if not chat_history:
        print("  (empty)")
    for i, msg in enumerate(chat_history):
        role = msg.__class__.__name__.replace("Message", "")  # Human / AI
        text = msg.content.replace("\n", " ")
        preview = text if len(text) <= 100 else text[:100] + "..."
        print(f"  [{i}] {role}: {preview}")
    print("-" * 40)


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Show the history this turn will rewrite against
    log_chat_history()

    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages=[
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question")
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    docs = retrieve(search_question, k=5)

    # print(f"Found {len(docs)} relevant documents")
    # for i, doc in enumerate(docs, 1):
    #     # Show first 2 lines of each document
    #     lines = doc.page_content.split('\n')[:2]
    #     preview = '\n'.join(lines)
    #     print(f" Doc {i}: {preview}...")

    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer this question based on provided documents.
    """

    # Step 4: Get the answer
    messages=[
            SystemMessage(content="You are helpful assistant that answers question based on provided documents and conversation history.  If you can't find the answer in the documents, say I don't have enough information to answer this question based on provided documents.")
        ] + chat_history + [
            HumanMessage(content=combined_input)
        ]

    result = model.invoke(messages)
    answer = result.content

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        ask_question(question)

if __name__ ==  "__main__":
    start_chat()