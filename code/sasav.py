import os
import time
import base64
import json
import pymupdf4llm
from openai import OpenAI
import dearpygui.dearpygui as dpg
from tools import get_profile, get_dvr, get_iso, pdf2md, build_vector_db, get_dvr_render, get_iso_abdo
from utilities import log, show_image_in_viewer
import numpy as np

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

from agents import Agent, WebSearchTool, trace, Runner, gen_trace_id, function_tool
from agents.model_settings import ModelSettings
from IPython.display import display, Markdown
import asyncio
from openai import AsyncOpenAI
from agents import set_default_openai_client

RESULT_IMAGE_PATH = "../assets/autonomy.png"  # <-- the image to display when button is pressed
# RESULT_IMAGE_PATH = "workspace_abdo/dvr_render/for_video/28.png"  # <-- the image to display when button is pressed
# _current_texture_tag = None
# _current_texture_registry_tag = "texture_registry"



# def run_pasav(dpg: dpg, file_path: str, knowledge_base_db_path: str, model_name: str, api_key: str):
def run_pasav(dpg, file_path, knowledge_base_db_path, model_name, api_key, image_queue):
    image_queue.put(RESULT_IMAGE_PATH)
    log(dpg, "get file path: " + file_path)
    log(dpg, "get model: " + model_name)
    log(dpg, "get apikey: " + api_key)
    # show_image_in_viewer(dpg, RESULT_IMAGE_PATH)
    # show_image_in_viewer(RESULT_IMAGE_PATH)
    # show_image_in_viewer(dpg, "workspace_chameleon/iso/rendering_0.1.png")

    client = OpenAI(api_key=api_key)
    # messages = [{"role": "user", "content": "write a haiku about ai and translate it into chinese"}]
    # response = client.chat.completions.create(model="gpt-4.1-nano",
    #                                           messages=messages)
    # log(dpg, response.choices[0].message.content)

    # call profile: retrieve metadata of the dataset
    # v_min, v_max = get_profile(dpg, file_path)
    # print("Range:", v_min, v_max)

    # call knowledge retrieval: from web search and local RAG
    ask_knowledge_base(dpg, knowledge_base_db_path, model_name, api_key)
    # asyncio.run(ask_internet(dpg, api_key))
    # print("after!!!!!!!!!!!!!!!!!!!!!!!111")

    # call range of interest perception
    # for opacity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     img_path = get_iso(file_path, opacity, "diagonal")
    #     image_queue.put(img_path)

    # do TF selection, run agents_tf.py by it self


    # Flame
    # render 14 views using the agent suggested tfs from previous step
    # viewpoints = [[0, 0, 500, 0, 0, 0, 0, 1, 0], # front
    #               [0, 0, -500, 0, 0, 0, 0, 1, 0], # back
    #               [-500, 0, 0, 0, 0, 0, 0, 1, 0], # left
    #               [500, 0, 0, 0, 0, 0, 0, 1, 0], # right
    #               [0, 500, 0, 0, 0, 0, 0, 0, -1], # top
    #               [0, -500, 0, 0, 0, 0, 0, 0, 1], # bottom
    #               [-400, 400, 400, 0, 0, 0, 1, 1, -1], # diagonal 1
    #               [400, 400, 400, 0, 0, 0, -1, 1, -1], # diagonal 2
    #               [400, 400, -400, 0, 0, 0, -1, 1, 1], # diagonal 3
    #               [-400, 400, -400, 0, 0, 0, 1, 1, 1], # diagonal 4
    #               [-400, -400, 400, 0, 0, 0, -1, 1, 1], # diagonal 5
    #               [400, -400, 400, 0, 0, 0, 1, 1, 1], # diagonal 6
    #               [400, -400, -400, 0, 0, 0, 1, 1, -1], # diagonal 7
    #               [-400, -400, -400, 0, 0, 0, -1, 1, -1]] # diagonal 8
    # tfs_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/code/workspace/tfs.json"
    # for i in range(len(viewpoints)):
    #     img_path = get_dvr_render(file_path, tfs_path, viewpoints[i], i)
    #     image_queue.put(img_path)
    # arr = np.load("fibonacci.npy")   # shape (N, 9)
    # size = arr.shape[0]
    # print(arr.shape)
    # tfs_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/code/workspace/tfs.json"
    # for i in range(size):
    #     cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz = arr[i]
    #     viewpoint = [cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz]
    #     img_path = get_dvr_render(file_path, tfs_path, viewpoint, i)
    #     image_queue.put(img_path)

    # Miranda
    # render 14 views using the agent suggested tfs from previous step
    viewpoints = [[700, 0, 0, 0, 0, 0, 0, 0, 1], # front
                  [-700, 0, 0, 0, 0, 0, 0, 0, 1], # back
                  [0, -700, 0, 0, 0, 0, 0, 0, 1], # left
                  [0, 700, 0, 0, 0, 0, 0, 0, 1], # right
                  [0, 0, 700, 0, 0, 0, -1, 0, 0], # top
                  [0, 0, -700, 0, 0, 0, 1, 0, 0], # bottom
                  [500, -500, 500, 0, 0, 0, -1, 1, 1], # diagonal 1
                  [500, 500, 500, 0, 0, 0, -1, -1, 1], # diagonal 2
                  [-500, 500, 500, 0, 0, 0, 1, -1, 1], # diagonal 3
                  [-500, -500, 500, 0, 0, 0, 1, 1, 1], # diagonal 4
                  [500, -500, -500, 0, 0, 0, 1, -1, 1], # diagonal 5
                  [500, 500, -500, 0, 0, 0, 1, 1, 1], # diagonal 6
                  [-500, 500, -500, 0, 0, 0, -1, 1, 1], # diagonal 7
                  [-500, -500, -500, 0, 0, 0, -1, -1, 1]] # diagonal 8

    # tfs_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/code/workspace/tfs.json"
    # for i in range(len(viewpoints)):
    #     img_path = get_dvr_render(file_path, tfs_path, viewpoints[i], i)
    #     image_queue.put(img_path)
    # arr = np.load("fibonacci_miranda_800.npy")   # shape (N, 9)
    # size = arr.shape[0]
    # print(arr.shape)
    # tfs_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/code/workspace/tfs.json"
    # for i in range(size):
    #     cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz = arr[i]
    #     viewpoint = [cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz]
    #     img_path = get_dvr_render(file_path, tfs_path, viewpoint, i)
    #     image_queue.put(img_path)

    # # Richtmyer
    # arr = np.load("fibonacci_richtmyer_700.npy")   # shape (N, 9)
    # size = arr.shape[0]
    # print(arr.shape)
    # tfs_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/code/workspace/tfs.json"
    # for i in range(size):
    #     cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz = arr[i]
    #     viewpoint = [cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz]
    #     img_path = get_dvr_render(file_path, tfs_path, viewpoint, i)
    #     image_queue.put(img_path)

    # # Abdo
    # arr = np.load("fibonacci_abdo_1500.npy")   # shape (N, 9)
    # size = arr.shape[0]
    # print(arr.shape)
    # for i in range(size):
    #     cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz = arr[i]
    #     viewpoint = [float(cam_x), float(cam_y), float(cam_z), float(fx), float(fy), float(fz), float(ux), float(uy), float(uz)]
    #     print(viewpoint)
    #     img_path = get_iso_abdo(file_path, viewpoint, i)
    #     image_queue.put(img_path)
    
    # # Chameloen
    # arr = np.load("fibonacci_chameleon_600.npy")   # shape (N, 9)
    # size = arr.shape[0]
    # print(arr.shape)
    # for i in range(size):
    #     cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz = arr[i]
    #     viewpoint = [float(cam_x), float(cam_y), float(cam_z), float(fx), float(fy), float(fz), float(ux), float(uy), float(uz)]
    #     print(viewpoint)
    #     img_path = get_iso_abdo(file_path, viewpoint, i)
    #     image_queue.put(img_path)

    # call renderer: initial rendering of the dataset
    # get_dvr(dpg, file_path, 0.1)
    # get_dvr(dpg, file_path, 0.2)
    # get_dvr(dpg, file_path, 0.3)
    # get_dvr(dpg, file_path, 0.4)
    # get_dvr(dpg, file_path, 0.5)
    # get_dvr(dpg, file_path, 0.6)
    # get_dvr(dpg, file_path, 0.7)
    # get_dvr(dpg, file_path, 0.8)
    # get_iso(dpg, file_path, 0.1)
    # get_iso(dpg, file_path, 0.2)
    # get_iso(dpg, file_path, 0.3)
    # get_iso(dpg, file_path, 0.4)
    # get_iso(dpg, file_path, 0.5)
    # get_iso(dpg, file_path, 0.6)
    # get_iso(dpg, file_path, 0.7)
    # get_iso(dpg, file_path, 0.8)

    # opacity_start_value = 0.1
    # with open("workspace/dvr/" + "rendering_" + str(opacity_start_value) + ".png", "rb") as img_file:
    #     image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "You are an assistant specializing in recognize the object from visualizatin image of scientific dataset."
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "What is the object shown in the image? What are the important component of the object that are worth visualing?"},
    #                 # {"type": "text", "text": "This is a critical points visualizatio. How many critical points are there?"},
    #                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    #             ],
    #         }
    #     ],
    # )

    # log(dpg, response.choices[0].message.content)

    # # call LLM in batch mode
    # log(dpg, "creating batch jsonl file...")
    # prompts = [
    #     "Who is Jianxin Sun",
    #     "Who is Hongfeng Yu",
    #     "Who is Tom Peterka"
    # ]
    # with open("batch_input.jsonl", "w", encoding="utf-8") as f:
    #     for i, p in enumerate(prompts):
    #         line = {
    #             "custom_id": f"req-{i}",
    #             "method": "POST",
    #             "url": "/v1/responses",
    #             "body": {
    #                 "model": "gpt-4.1-mini",
    #                 "input": p,
    #             },
    #         }
    #         f.write(json.dumps(line) + "\n")
    # infile = client.files.create(
    #     file=open("batch_input.jsonl", "rb"),
    #     purpose="batch",
    # )
    # batch = client.batches.create(
    #     input_file_id=infile.id,
    #     endpoint="/v1/responses",
    #     completion_window="24h",
    # )
    # print("Batch ID:", batch.id)
    # # 3) Poll until complete
    # while True:
    #     b = client.batches.retrieve(batch.id)
    #     print("Status:", b.status)
    #     if b.status in ("completed", "failed", "cancelled", "expired"):
    #         break
    #     time.sleep(10)
    # # 4) Download results (output_file_id contains successful outputs)
    # if b.status == "completed" and b.output_file_id:
    #     content = client.files.content(b.output_file_id).read()
    #     with open("batch_output.jsonl", "wb") as f:
    #         f.write(content)

    #     # Parse outputs:
    #     with open("batch_output.jsonl", "r", encoding="utf-8") as f:
    #         for line in f:
    #             obj = json.loads(line)
    #             # obj will include your custom_id + the response payload
    #             print(obj["custom_id"], obj["response"]["status_code"])

def run_build_knowledge_base(dpg: dpg, pdf_folder: str, md_folder: str, api_key):
    # convert pdf papers to markdown for downstream vectorization
    log(dpg, "Converting pdf papers to markdown...")
    # pdf2md(dpg, pdf_folder, md_folder)
    # build vector database
    log(dpg, "Building the vecter database...")
    build_vector_db(dpg, md_folder, api_key)

# async def ask_internet(dpg: dpg, api_key: str):
#     client = AsyncOpenAI(api_key=api_key)
#     set_default_openai_client(client)

#     INSTRUCTIONS = "You are a research assistant. Given a search term which is a description of the main object within scientific data, you search the web for to \
#     suggest a list of top important subregions from the object that worth scientific investigation. Only output the result as a list of key words with total number of keywords in the list > 1 and <= 10."
#     search_agent = Agent(
#         name="Search agent",
#         instructions=INSTRUCTIONS,
#         tools=[WebSearchTool(search_context_size="low")],
#         model="gpt-4o-mini",
#         model_settings=ModelSettings(tool_choice="required"),
#     )
#     message = "Flame combustion simulation volume"
#     with trace("Search"):
#         result = await Runner.run(search_agent, message)
#     print(display(Markdown(result.final_output)))
#     print(Markdown(result.final_output))
#     print(result.final_output)

def ask_knowledge_base(dpg, db_path: str, model_name: str, api_key: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        temperature=0,
        model=model_name,
        api_key=api_key
    )

    log(dpg, "ask insight")

    SYSTEM_PROMPT_TEMPLATE = """
    You are a knowledgeable scientific data recognization agent.
    You duty is to suggest the most valuable regions of a scientific dataset which is a flame simulation.
    If relevant, use the given context to answer any question.
    If you don't know the answer, say so.
    Context:
    {context}
    """

    USER_PROMPT = """List top important regions to investigate for the scientific dataset involve flame simulation with ranking from most important to least important. 
    Only output the result as a list of key words with total number of keywords in the list > 1 and <= 10.
    If you are not sure, don't add more keywords than necessary.
    """

    # -------- RAG --------
    docs = retriever.invoke(USER_PROMPT)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=USER_PROMPT)
    ])

    print("with rag:", response.content)

    # ✅ Token usage
    usage = response.response_metadata.get("token_usage", {})
    print("RAG token usage:")
    print("  Input tokens:", usage.get("prompt_tokens"))
    print("  Output tokens:", usage.get("completion_tokens"))
    print("  Total tokens:", usage.get("total_tokens"))

    # -------- NO RAG --------
    response_no_rag = llm.invoke([
        SystemMessage(content=""),
        HumanMessage(content=USER_PROMPT)
    ])

    print("without rag:", response_no_rag.content)

    # ✅ Token usage
    usage_no_rag = response_no_rag.response_metadata.get("token_usage", {})
    print("No-RAG token usage:")
    print("  Input tokens:", usage_no_rag.get("prompt_tokens"))
    print("  Output tokens:", usage_no_rag.get("completion_tokens"))
    print("  Total tokens:", usage_no_rag.get("total_tokens"))

async def ask_internet(dpg: dpg, api_key: str):
    from openai import AsyncOpenAI
    from agents import Agent, Runner, WebSearchTool, ModelSettings
    from agents.tracing import trace
    from IPython.display import display, Markdown

    client = AsyncOpenAI(api_key=api_key)
    set_default_openai_client(client)

    INSTRUCTIONS = """You are a research assistant. Given a search term which is a description of the main object within scientific data, you search the web to suggest a list of top important subregions from the object that worth scientific investigation. Only output the result as a list of key words with total number of keywords in the list > 1 and <= 10."""

    search_agent = Agent(
        name="Search agent",
        instructions=INSTRUCTIONS,
        tools=[WebSearchTool(search_context_size="low")],
        model="gpt-4o-mini",
        model_settings=ModelSettings(tool_choice="required"),
    )

    message = "Flame combustion simulation volume"

    with trace("Search"):
        result = await Runner.run(search_agent, message)

    # ===== Print result =====
    print(display(Markdown(result.final_output)))
    print(result.final_output)

    # ===== Extract token usage =====
    try:
        usage = result.raw_responses[-1].usage  # last LLM call
        print("\n=== Token Usage ===")
        print(f"Input tokens: {usage.input_tokens}")
        print(f"Output tokens: {usage.output_tokens}")
        print(f"Total tokens: {usage.total_tokens}")
    except Exception as e:
        print("Token usage not available:", e)