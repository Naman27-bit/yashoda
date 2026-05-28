from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="tinyllama")
template = """
you are expert in answering the information related to the medicines.
here are some relevent information about the medicines:
{context}
answer the question based on the above information.
question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n---------------")
    question = input(
        "Hii! I'm yashoda, your medical assistant. "
        "How can I help you today? (q to quit): "
    )
    if question.lower().strip() == "q":
        print("Goodbye!Take care!")
        break

    context = retriever.invoke(question)
    answer = chain.invoke({"context": context, "question": question})
    print(f"yashoda: {answer}")
