# chatbot.py - VicRoads RAG Chatbot using Ollama
from search import get_context_passages
from ollama import Client

# -------------------------
# Ollama client
# -------------------------
ollama_client = Client()  # No model here; specify in generate

# -------------------------
# Generate answer using retrieved passages
# -------------------------
def generate_answer(question, context, model="llama3"):
    """
    Generate a concise answer from Ollama using provided context passages.
    Only use the retrieved passages. Do not add extra commentary.
    """
    # Flatten context into one block of text
    combined_context = " ".join(context)

    prompt = (
        f"You are interactive VicRoads Chatbot. "
        f"Answer the question using ONLY the information below. "
        f"Provide a concise answer in a natural way complete sentence without prefacing with phrases like 'According to...'."
        f"If the answer is not present, reply with 'Sorry, I do not have information about this currently.'\n\n"
        f"Question: {question}\n\n"
        f"Context: {combined_context}\n\n"
        f"Answer:"
    )

    # Generate response using Ollama
    result = ollama_client.generate(
        model=model,  # Replace with your Ollama model
        prompt=prompt
    )

    # Extract the generated text
    try:
        return result.response.strip()
    except AttributeError:
        return ""


# -------------------------
# Main Chatbot Loop
# -------------------------
if __name__ == "__main__":
    print("VicRoads RAG Chatbot (text-based). Type 'exit' to quit.")

    while True:
        try:
            question = input("\nUser: ")
        except EOFError:
            print("\nExiting chatbot.")
            break

        if question.lower() == "exit":
            break

        # Retrieve context passages
        context_passages = get_context_passages(question, mode="dense")

        # If no context, provide fallback message
        if not context_passages:
            print("Bot: Sorry, I don't have information about this currently.")
            continue

        # Generate answer using retrieved context
        answer = generate_answer(question, context_passages)

        # If the model returns empty or irrelevant response, fallback
        if not answer:
            print("Bot: Sorry, I don't have information about this currently.")
        else:
            print(f"Bot: {answer}")
