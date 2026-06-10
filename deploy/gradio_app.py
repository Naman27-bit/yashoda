import gradio as gr

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


MODEL_NAME = "tinyllama"


template = """
you are expert in answering the information related to the medicines.
here are some relevent information about the medicines:
{context}
answer the question based on the above information.
question: {question}
"""


prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model=MODEL_NAME)
chain = prompt | model


def respond(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "Please enter a question."

    context = retriever.invoke(question)
    answer = chain.invoke({"context": context, "question": question})
    return str(answer)


with gr.Blocks(title="Yashoda - Medical Assistant") as demo:
    gr.Markdown(
        "# Yashoda\n"
        "Medical assistant (powered by Ollama and LangChain)\n"
        "Developed by Mr. Naman Kumar"
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g., What are the side effects of Augmentin?",
            lines=3,
        )

    out = gr.Textbox(label="Answer", lines=10)

    with gr.Row():
        btn = gr.Button("Get Answer")
        clear = gr.Button("Clear")

    btn.click(fn=respond, inputs=inp, outputs=out)
    clear.click(fn=lambda: ("", ""), inputs=None, outputs=[inp, out])


if __name__ == "__main__":
    demo.launch(share=True)
