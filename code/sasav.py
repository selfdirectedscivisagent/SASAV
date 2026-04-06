import keyword
import os
import time
import base64
import json
import mimetypes
import math
import pymupdf4llm
from openai import OpenAI
from pathlib import Path
import dearpygui.dearpygui as dpg
from tools import get_iso_final, get_iso_final_video, get_iso_fine_tune, get_iso_view_sphere, get_profile, get_profile_vtk, get_dvr, get_iso, pdf2md, build_vector_db, get_dvr_render, get_iso_abdo
from utilities import log, show_image_in_viewer
import numpy as np
from typing import List, Dict, Any

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
    # log(dpg, "get apikey: " + api_key)
    # show_image_in_viewer(dpg, RESULT_IMAGE_PATH)
    # show_image_in_viewer(RESULT_IMAGE_PATH)
    # show_image_in_viewer(dpg, "workspace_chameleon/iso/rendering_0.1.png")

    client = OpenAI(api_key=api_key)
    # messages = [{"role": "user", "content": "write a haiku about ai and translate it into chinese"}]
    # response = client.chat.completions.create(model="gpt-4.1-nano",
    #                                           messages=messages)
    # log(dpg, response.choices[0].message.content)

    log(dpg, "+++++ Data Profiling +++++")
    log(dpg, "----- Input Profiling -----")
    data_org_vtk_path, data_medium_vtk_path, data_small_vtk_path = get_profile(dpg, file_path)
    log(dpg, "Original data path: " + data_org_vtk_path)
    log(dpg, "Medium data path: " + data_medium_vtk_path)
    log(dpg, "Small data path: " + data_small_vtk_path)
    # data_org_vtk_path = "/home/js/ws/proactiveSciVisAgent/github/SASAV/data/data_org.vtk" # delete
    # data_medium_vtk_path = "/home/js/ws/proactiveSciVisAgent/github/SASAV/data/data_medium.vtk" # delete
    # data_small_vtk_path = "/home/js/ws/proactiveSciVisAgent/github/SASAV/data/data_small.vtk" # delete
    log(dpg, "----- Initial Rendering -----")
    viewpoints = [[0, 0, -600, 0, 0, 0, 0, -1, 0], # front
                  [0, 0, 600, 0, 0, 0, 0, -1, 0], # back
                  [-600, 0, 0, 0, 0, 0, 0, -1, 0], # left
                  [600, 0, 0, 0, 0, 0, 0, -1, 0], # right
                  [0, -600, 0, 0, 0, 0, 0, 0, 1], # top
                  [0, 600, 0, 0, 0, 0, 0, 0, -1]] # bottom
    iso_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i in range(len(iso_list)):
        for j in range(len(viewpoints)):
            if j == 0:
                direction = "front"
            if j == 1:
                direction = "back"
            if j == 2:
                direction = "left"
            if j == 3:
                direction = "right"
            if j == 4:
                direction = "top"
            if j == 5:
                direction = "bottom"
            img_path = get_dvr(data_small_vtk_path, viewpoints[j], iso_list[i], direction)
            image_queue.put(img_path)
    log(dpg, "----- Evaluator -----")
    directions = ["front", "back", "left", "right", "top", "bottom"]
    overall_scores = []
    respective_directions = []
    for i in range(len(iso_list)):
        image_paths = []
        for j in range(len(directions)):
            image_path = "workspace/dvr/" + str(iso_list[i]) + "_" + directions[j] + ".png"
            image_paths.append(image_path)
        log(dpg, "evaluating isovalue of index: " + str(i))
        overall_score, respective_direction = evaluator(client, image_paths)
        overall_scores.append(overall_score)
        respective_directions.append(respective_direction)
    print(overall_scores)
    print(respective_direction)
    best_isovalue = max(overall_scores)
    print("best isovalue:", best_isovalue)
    best_index = overall_scores.index(max(overall_scores))
    print("best index:", best_index)
    print("best isovalue:", overall_scores[best_index])
    print("best direction:", respective_directions[best_index])
    log(dpg, "----- Recognizer -----")
    image_paths = []
    # best_index = 0 # delete
    for j in range(len(directions)):
        image_path = "workspace/dvr/" + str(iso_list[best_index]) + "_" + directions[j] + ".png"
        image_paths.append(image_path)
    print(image_paths)
    result = recognizer(client, image_paths)
    print("Recognized Object:", result["object"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
    log(dpg, "Recognized Object:" + result["object"])
    object = result["object"]

    log(dpg, "+++++ Knowledge Retrieval +++++")
    # object = "Chameleon" # delete
    # call knowledge retrieval: from web search and local RAG
    # ask_knowledge_base(dpg, knowledge_base_db_path, model_name, api_key)
    keywords = asyncio.run(ask_internet(dpg, api_key, object))
    print("keywords:", keywords)

    log(dpg, "+++++ TFs Suggestion +++++")
    log(dpg, "----- Isosuface Rendering -----")
    viewpoints = [[0, 0, -600, 0, 0, 0, 0, -1, 0], # front
                  [-600, 0, 0, 0, 0, 0, 0, -1, 0], # left
                  [0, -600, 0, 0, 0, 0, 0, 0, 1], # top
                  [-500, -500, -500, 0, 0, 0, 1, -1, 1]] # diagonal
    iso_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(iso_list)):
        for j in range(len(viewpoints)):
            if j == 0:
                direction = "front"
            if j == 1:
                direction = "left"
            if j == 2:
                direction = "top"
            if j == 3:
                direction = "diagonal"
            img_path = get_iso(data_small_vtk_path, viewpoints[j], iso_list[i], direction)
            image_queue.put(img_path)
    log(dpg, "----- Sematic Analyzer -----")
    directions = ["front", "left", "top", "diagonal"]
    # keywords = ['skin', 'bone', 'tail', 'eyes', 'feet', 'tongue', 'casque', 'crest', 'spines'] # delete
    confidences = []
    selected_keywords = []
    for i in range(len(iso_list)):
        image_paths = []
        for j in range(len(directions)):
            image_path = "workspace/iso/" + str(iso_list[i]) + "_" + directions[j] + ".png"
            image_paths.append(image_path)
        log(dpg, "evaluating isovalue of index: " + str(i))
        selected_keyword, confidence = semetricAnalyzer(client, image_paths, keywords)
        selected_keywords.append(selected_keyword)
        confidences.append(confidence)
    print(selected_keywords)
    print(confidences)
    threshold = 0.97
    candidates = get_sorted_indices_above_threshold(confidences, threshold)
    # candidates = [0, 3]
    # print(candidates)

    log(dpg, "----- Transfer Function Designer -----")
    directions = ["front", "left", "top", "diagonal"]
    # keywords = ['skin', 'bone', 'tail', 'eyes', 'feet', 'tongue', 'casque', 'crest', 'spines'] # delete
    confidences = []
    selected_keywords = []
    rgbs = []
    opacities = []
    for i in range(len(iso_list)):
        image_paths = []
        for j in range(len(directions)):
            image_path = "workspace/iso/" + str(iso_list[i]) + "_" + directions[j] + ".png"
            image_paths.append(image_path)
        log(dpg, "evaluating isovalue of index: " + str(i))
        selected_keyword, confidence, rgb, opacity = TFDesigner(client, image_paths, keywords)
        selected_keywords.append(selected_keyword)
        confidences.append(confidence)
        rgbs.append(rgb)
        opacities.append(opacity)
    print(selected_keywords)
    print(confidences)
    print(rgbs)
    print(opacities)
    # candidate = [0, 3]
    # rgbs[0] = [0.65, 0.85, 0.70]
    # rgbs[3] = [1.0, 1.0, 1.0]
    # opacities[0] = 0.45
    # opacities[3] = 0.99
    log(dpg, "----- Isovalue Fine-Tuning -----")
    best_viewpoint = [600, 0, 0, 0, 0, 0, 0, -1, 0]
    search_range = 9
    search_step = 0.01
    candidates = [0, 3]
    opacities = [0.12, 0.45]
    for i in range(len(candidates)):
        isovalue = iso_list[candidates[i]]
        for j in range(search_range):
            offset = (j + 1)*search_step
            isovalue_try = isovalue - offset
            print(isovalue_try)
            img_path = get_iso_fine_tune(data_small_vtk_path, best_viewpoint, isovalue_try)
            image_queue.put(img_path)
        for j in range(search_range):
            offset = (j + 1)*search_step
            isovalue_try = isovalue + offset
            print(isovalue_try)
            img_path = get_iso_fine_tune(data_small_vtk_path, best_viewpoint, isovalue_try)
            image_queue.put(img_path)
    log(dpg, "----- Generating Final Visualization Image -----")
    img_path = get_iso_final(data_org_vtk_path, best_viewpoint, opacities)
    image_queue.put(img_path)
    log(dpg, "+++++ Isovalue Fine-Tuning +++++")
    log(dpg, "----- View Sphere Rendering -----")
    log(dpg, "Load Fibonacci Lattice as Candidate Viewpoints")
    arr = np.load("fibonacci.npy")   # shape (N, 9)
    size = arr.shape[0]
    print(arr.shape)
    for i in range(size):
        cam_x, cam_y, cam_z, fx, fy, fz, ux, uy, uz = arr[i]
        viewpoint = [float(cam_x), float(cam_y), float(cam_z), float(fx), float(fy), float(fz), float(ux), float(uy), float(uz)]
        print(viewpoint)
        log(dpg, "Rendering View " + str(i))
        img_path = get_iso_view_sphere(data_small_vtk_path, viewpoint, i)
        image_queue.put(img_path)
    log(dpg, "----- View Evaluation -----")
    views = example_views_from_npy()
    print(f"Loaded {len(views)} views")
    log(dpg, "Load Rendering from View Sphere")
    log(dpg, "Suggest Anchor Viewpoints...")
    result = suggest_view_trajectory(views, client)
    print("\n=== Validated Model JSON Result ===")
    print(json.dumps(result, indent=2))
    keyframes = build_anchor_keyframes(views, result)
    print("\n=== Anchor Keyframes ===")
    print(json.dumps(keyframes, indent=2))
    save_json(keyframes, "workspace/anchor_keyframes.json")
    log(dpg, "Suggest Avoid Viewpoints...")
    avoid_views = build_avoid_viewframes(views, result)
    print("\n=== Avoid Views ===")
    print(json.dumps(avoid_views, indent=2))
    save_json(avoid_views, "workspace/avoid_views.json")

    log(dpg, "----- Building Viewpoint Trajectory -----")
    # Input produced from your previous step
    anchor_json_path = "workspace/anchor_keyframes.json"
    # Output
    dense_json_path = "workspace/dense_camera_trajectory.json"
    # Parameters
    frames_per_segment = 30
    use_spline = True
    anchor_keyframes = load_anchor_keyframes_from_json(anchor_json_path)
    print(len(anchor_keyframes))
    dense_trajectory = generate_smooth_camera_trajectory(
        anchor_keyframes=anchor_keyframes,
        frames_per_segment=frames_per_segment,
        use_spline=use_spline,
    )
    save_trajectory_to_json(dense_trajectory, dense_json_path)
    print(f"Generated {len(dense_trajectory)} frames total.")

    log(dpg, "----- Generating Final Visualization Animation -----")
    viewpoints = load_viewpoints_from_trajectory("workspace/dense_camera_trajectory.json")
    print(len(viewpoints))
    # render all frames for video
    for i in range(len(viewpoints)):
        img_path = get_iso_final_video(data_small_vtk_path, viewpoints[i], i)
        image_queue.put(img_path)
        log(dpg, "Rendering Animation View " + str(i))
    log(dpg, "Finish!")


def get_sorted_indices_above_threshold(arr, threshold):
    # Step 1: collect (index, value) pairs above threshold
    filtered = [(i, v) for i, v in enumerate(arr) if v > threshold]

    # Step 2: sort by value in descending order
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Step 3: extract indices only
    return [i for i, v in filtered]
   
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

async def ask_internet(dpg: dpg, api_key: str, object: str):
    from openai import AsyncOpenAI
    from agents import Agent, Runner, WebSearchTool, ModelSettings
    from agents.tracing import trace
    import re

    client = AsyncOpenAI(api_key=api_key)
    set_default_openai_client(client)

    INSTRUCTIONS = """You are a research assistant.

    Given a search term describing the main object within scientific data, search the web and identify important subregions of that object that are worth scientific investigation.

    Output requirements:
    - Output ONLY a list of keywords.
    - Each keyword MUST be a SINGLE WORD (no spaces, no phrases).
    - Each keyword should represent one meaningful subregion.
    - Total number of keywords must be between 2 and 10.
    - Do NOT include the original object name.
    - Do NOT include explanations, numbering, or formatting.
    - Output as plain text, comma-separated.
    - If the object is animal, at least include keywords of "skin" and "bone" in the output

    Example output:
    core, boundary, shell, hotspot
    """

    search_agent = Agent(
        name="Search agent",
        instructions=INSTRUCTIONS,
        tools=[WebSearchTool(search_context_size="low")],
        model="gpt-4o-mini",
        model_settings=ModelSettings(tool_choice="required"),
    )

    with trace("Search"):
        result = await Runner.run(search_agent, object)

    raw_output = result.final_output.strip()

    # ===== Parse into list =====
    # Remove markdown bullets/numbers if any
    cleaned = re.sub(r"[\n\r•\-*\d\.]+", ",", raw_output)

    # Split by comma
    keywords = [k.strip() for k in cleaned.split(",") if k.strip()]

    # Deduplicate while preserving order
    seen = set()
    keywords = [k for k in keywords if not (k in seen or seen.add(k))]

    # ===== Token usage (optional debug) =====
    try:
        usage = result.raw_responses[-1].usage
        print("\n=== Token Usage ===")
        print(f"Input tokens: {usage.input_tokens}")
        print(f"Output tokens: {usage.output_tokens}")
        print(f"Total tokens: {usage.total_tokens}")
    except Exception:
        pass

    return keywords

def to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def evaluator(client, image_paths):
    input_items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": """
You are given 6 images of the same volumetric dataset.
Each image is a volume rendering viewed from one direct face direction of the volume cube.

Your task:
1. Judge how clear / recognizable the main object is from these renderings.
2. Return a single overall score in the range [0, 1].
- 0.0 = completely unclear / not recognizable
- 1.0 = very clear / highly recognizable
3. Also provide:
- per_view_scores: score for each of the 6 views in [0, 1]
- short_reason: one brief explanation
4. Output JSON only.

Required JSON format:
{
"overall_score": 0.0,
"per_view_scores": {
    "front": 0.0,
    "back": 0.0,
    "left": 0.0,
    "right": 0.0,
    "top": 0.0,
    "bottom": 0.0
},
"short_reason": "..."
}

Important scoring rule:
- Focus only on visual recognizability/clarity of the object from the renderings.
- Consider silhouette clarity, visibility of meaningful structure, contrast, and whether the object shape is easy to identify.
- Be conservative and consistent.
"""
                },

                *[
                    {
                        "type": "input_image",
                        "image_url": to_data_url(p),
                    } for p in image_paths
                ],
            ],
        }
    ]

    response = client.responses.create(
        model="gpt-5.4",
        input=input_items,
    )

    result = json.loads(response.output_text)
    # print("Overall score:", result["overall_score"])
    # print("Per-view scores:", result["per_view_scores"])
    # print("Reason:", result["short_reason"])

    overall_score = result["overall_score"]
    per_view_scores = result["per_view_scores"]

    # 🔥 Find best direction
    best_direction = max(per_view_scores, key=per_view_scores.get)

    return overall_score, best_direction

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def recognizer(client, image_paths):
    assert len(image_paths) == 6, "Expect exactly 6 images"

    view_names = ["front", "back", "left", "right", "top", "bottom"]

    # Build image inputs
    content = [
        {
            "type": "input_text",
            "text": (
                "You are given 6 volume renderings of the same 3D object, "
                "captured from 6 orthogonal directions: front, back, left, right, top, bottom.\n\n"
                "Task:\n"
                "1. Identify what the object is.\n"
                "2. Provide a confidence score between 0 and 1.\n\n"
                "Rules:\n"
                "- Use all views jointly.\n"
                "- If uncertain, lower the confidence.\n"
                "- Be concise and objective.\n"
                "- Output strictly in JSON format:\n\n"
                "{\n"
                '  "object": "<recognized object name>",\n'
                '  "confidence": <float between 0 and 1>,\n'
                '  "reason": "<short explanation>"\n'
                "}"
            )
        }
    ]

    # Attach images
    for i, path in enumerate(image_paths):
        img_base64 = encode_image(path)
        content.append({
            "type": "input_text",
            "text": f"View: {view_names[i]}"
        })
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{img_base64}"
        })

    response = client.responses.create(
        model="gpt-5.4",
        input=[{
            "role": "user",
            "content": content
        }],
        temperature=0.1,  # stable output
    )

    # Extract text output
    output_text = response.output_text

    try:
        result = json.loads(output_text)
    except:
        print("Raw output:\n", output_text)
        raise ValueError("Failed to parse JSON output")

    return result





def encode_image_as_data_url(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"   # fallback if extension is unknown

    return f"data:{mime_type};base64,{b64}"


def semetricAnalyzer(client, image_paths, keywords):
    image_data_urls = [encode_image_as_data_url(p) for p in image_paths]

    prompt = f"""
You are an expert in scientific visualization and visual recognition. 

You are given:
1. Four volume-rendered images (front, left, top, diagonal)
2. Candidate keywords:
{keywords}

Task:
- Select ONE best matching keyword
- Output confidence in [0,1]


Return JSON only:
{{
    "selected_keyword": "...",
    "confidence": 0.0,
    "reason": "..."
}}

"""

    response = client.responses.create(
        model="gpt-5.4",
        temperature=0.1,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_urls[0],
                        "detail": "low",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_urls[1],
                        "detail": "low",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_urls[2],
                        "detail": "low",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_urls[3],
                        "detail": "low",
                    },
                ],
            }
        ],
    )

    result = json.loads(response.output_text)
    print("Keyword:", result["selected_keyword"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

    keyword = result["selected_keyword"]
    confidence = result["confidence"]

    return keyword, confidence

def TFDesigner(client, image_paths, keywords):
    image_data_urls = [encode_image_as_data_url(p) for p in image_paths]

    prompt = f"""
You are an expert in scientific visualization and visual recognition. 

You are given:
1. Four volume-rendered images (front, left, top, diagonal)
2. Candidate keywords:
{keywords}

Task:
- Select ONE best matching keyword
- Output confidence in [0,1]
- Assign final colors and opacities for a combined 3D isosurface rendering.

Return JSON only:
{{
    "selected_keyword": "...",
    "confidence": 0.0,
    "reason": "..."
    "rgb": [R, G, B]
    "opacity": 0.10
}}

Rules:
- rgb must be 3 float values in [0.0, 1.0]
- opacity must be a float in [0.0, 1.0]
"""

    response = client.responses.create(
        model="gpt-5.4",
        temperature=0.1,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_urls[0],
                        "detail": "low",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_urls[1],
                        "detail": "low",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_urls[2],
                        "detail": "low",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_urls[3],
                        "detail": "low",
                    },
                ],
            }
        ],
    )

    result = json.loads(response.output_text)
    print("Keyword:", result["selected_keyword"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
    print("rgb:", result["rgb"])
    print("opacity:", result["opacity"])

    keyword = result["selected_keyword"]
    confidence = result["confidence"]
    rgb = result["rgb"]
    opacity = result["opacity"]

    return keyword, confidence, rgb, opacity

def example_views_from_npy() -> List[Dict[str, Any]]:
    arr = np.load("fibonacci.npy")

    views = []
    for i, row in enumerate(arr):
        views.append({
            "view_id": f"view_{i:03d}",
            "image_path": f"workspace/view_sphere/{i}.png",
            "camera_position": row[0:3].tolist(),
            "focal_position": row[3:6].tolist(),
            "view_up": row[6:9].tolist(),
        })
    return views

def upload_image_file(image_path: str, client) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    return uploaded.id

def build_prompt_text(views: List[Dict[str, Any]]) -> str:
    compact_views = [
        {
            "view_id": v["view_id"],
            "camera_position": v["camera_position"],
            "focal_position": v["focal_position"],
            "view_up": v["view_up"],
        }
        for v in views
    ]

    return f"""
You are an expert scientific visualization assistant.

I will provide 32 volume-rendered images of the same scientific volumetric dataset,
plus the camera parameters used to create each rendering.

Your task:
1. Compare all 32 views jointly.
2. Identify which views are the most informative for understanding the 3D structure.
3. Identify which views are redundant.
4. Suggest a small ordered set of anchor viewpoints for an animation trajectory.
5. Prefer a trajectory that:
   - reveals the global shape first
   - reduces occlusion ambiguity
   - shows complementary structure from non-redundant viewpoints
   - transitions smoothly in a logical order

Important:
- Do NOT invent new anchor view IDs. Only select from the provided 32 view IDs.
- You may suggest 8 to 10 anchor views.
- Treat the images as the primary evidence.
- Use the camera parameters only as supporting geometric context.

Here are the 32 viewpoints as JSON:
{json.dumps(compact_views, indent=2)}

Return JSON only with exactly this schema:
{{
  "dataset_summary": "short summary of visible 3D structure",
  "ranked_views": [
    {{
      "view_id": "string",
      "informativeness": 0.0,
      "novelty": 0.0,
      "occlusion_reduction": 0.0,
      "reason": "string"
    }}
  ],
  "selected_anchor_views": [
    {{
      "view_id": "string",
      "role": "overview | transition | feature_emphasis | finale | other",
      "reason": "string"
    }}
  ],
  "anchor_order": ["view_id_1", "view_id_2", "view_id_3"],
  "trajectory_strategy": {{
    "style": "string",
    "reason": "string"
  }},
  "avoid_views": [
    {{
      "view_id": "string",
      "reason": "string"
    }}
  ]
}}
""".strip()


def validate_model_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Model output is not a JSON object.")

    # Fallback: build anchor_order from selected_anchor_views
    if "anchor_order" not in result:
        selected = result.get("selected_anchor_views", [])
        if isinstance(selected, list) and len(selected) > 0:
            result["anchor_order"] = [
                item["view_id"] for item in selected
                if isinstance(item, dict) and "view_id" in item
            ]

    # Ensure lists exist even if omitted
    result.setdefault("ranked_views", [])
    result.setdefault("selected_anchor_views", [])
    result.setdefault("avoid_views", [])

    if "anchor_order" not in result or not result["anchor_order"]:
        raise ValueError(
            "Model output missing 'anchor_order', and no usable fallback could be built "
            "from 'selected_anchor_views'."
        )

    return result


def suggest_view_trajectory(views: List[Dict[str, Any]], client) -> Dict[str, Any]:
    if len(views) != 32:
        raise ValueError(f"Expected 32 views, got {len(views)}")

    file_ids = [upload_image_file(v["image_path"], client) for v in views]

    content = [{"type": "input_text", "text": build_prompt_text(views)}]

    for file_id, v in zip(file_ids, views):
        content.append({
            "type": "input_text",
            "text": f"Image for viewpoint: {v['view_id']}",
        })
        content.append({
            "type": "input_image",
            "file_id": file_id,
            "detail": "high",
        })

    response = client.responses.create(
        model="gpt-5.4",
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_object"}},
    )

    if hasattr(response, "usage") and response.usage:
        print("\n=== Token Usage ===")
        print(f"Input tokens:  {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print(f"Total tokens:  {response.usage.total_tokens}")

    raw_text = response.output_text
    print("\n=== Raw Model Output ===")
    print(raw_text)

    result = json.loads(raw_text)
    result = validate_model_result(result)

    return result


def build_anchor_keyframes(
    views: List[Dict[str, Any]],
    model_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    view_map = {v["view_id"]: v for v in views}
    anchor_order = model_result.get("anchor_order", [])

    if not anchor_order:
        raise ValueError("anchor_order is missing or empty.")

    keyframes = []
    for i, view_id in enumerate(anchor_order):
        if view_id not in view_map:
            raise ValueError(f"Model returned unknown view_id in anchor_order: {view_id}")

        v = view_map[view_id]
        keyframes.append({
            "keyframe_index": i,
            "view_id": view_id,
            "camera_position": v["camera_position"],
            "focal_position": v["focal_position"],
            "view_up": v["view_up"],
        })

    return keyframes


def build_avoid_viewframes(
    views: List[Dict[str, Any]],
    model_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    view_map = {v["view_id"]: v for v in views}
    avoid_views = model_result.get("avoid_views", [])

    avoid_viewframes = []
    for i, item in enumerate(avoid_views):
        view_id = item.get("view_id")
        reason = item.get("reason", "")

        if view_id not in view_map:
            continue

        v = view_map[view_id]
        avoid_viewframes.append({
            "avoid_index": i,
            "view_id": view_id,
            "reason": reason,
            "camera_position": v["camera_position"],
            "focal_position": v["focal_position"],
            "view_up": v["view_up"],
        })

    return avoid_viewframes


def save_json(data: Any, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to: {output_path}")

def load_anchor_keyframes_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_trajectory_to_json(trajectory: List[Dict[str, Any]], output_path: str):
    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"Saved dense trajectory to: {output_path}")

# ============================================================
# Basic vector helpers
# ============================================================

def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vec_mul(a, s: float):
    return [a[i] * s for i in range(3)]


def vec_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def vec_cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


def vec_norm(a):
    return math.sqrt(vec_dot(a, a))


def vec_normalize(a, eps: float = 1e-8):
    n = vec_norm(a)
    if n < eps:
        return [0.0, 0.0, 0.0]
    return [a[i] / n for i in range(3)]


def lerp(a, b, t: float):
    return a + (b - a) * t


def vec_lerp(a, b, t: float):
    return [lerp(a[i], b[i], t) for i in range(3)]

# ============================================================
# Catmull-Rom spline interpolation for 3D vectors
# ============================================================

def catmull_rom_scalar(p0, p1, p2, p3, t: float):
    """
    Standard Catmull-Rom spline.
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t2 +
        (-p0 + 3.0*p1 - 3.0*p2 + p3) * t3
    )


def catmull_rom_vec3(p0, p1, p2, p3, t: float):
    return [
        catmull_rom_scalar(p0[i], p1[i], p2[i], p3[i], t)
        for i in range(3)
    ]


# ============================================================
# Camera basis / up-vector stabilization
# ============================================================

def orthonormalize_view_up(camera_position, focal_position, view_up):
    """
    Recompute a stable view_up that is perpendicular to viewing direction.
    """
    forward = vec_normalize(vec_sub(focal_position, camera_position))
    up_guess = vec_normalize(view_up)

    # If up is degenerate or nearly parallel to forward, choose a fallback.
    if vec_norm(up_guess) < 1e-8 or abs(vec_dot(forward, up_guess)) > 0.98:
        candidates = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
        best = None
        best_score = -1.0
        for c in candidates:
            score = 1.0 - abs(vec_dot(forward, c))
            if score > best_score:
                best_score = score
                best = c
        up_guess = best

    right = vec_normalize(vec_cross(forward, up_guess))
    if vec_norm(right) < 1e-8:
        # final fallback
        up_guess = [0.0, 1.0, 0.0]
        right = vec_normalize(vec_cross(forward, up_guess))
        if vec_norm(right) < 1e-8:
            up_guess = [0.0, 0.0, 1.0]
            right = vec_normalize(vec_cross(forward, up_guess))

    true_up = vec_normalize(vec_cross(right, forward))
    return true_up

def smoothstep(t: float):
    return t * t * (3.0 - 2.0 * t)

def generate_smooth_camera_trajectory(
    anchor_keyframes: List[Dict[str, Any]],
    frames_per_segment: int = 60,
    use_spline: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate a smooth dense trajectory from anchor keyframes.

    Input anchor_keyframes format:
    [
      {
        "keyframe_index": 0,
        "view_id": "face_pz",
        "camera_position": [...],
        "focal_position": [...],
        "view_up": [...]
      },
      ...
    ]

    Output dense trajectory format:
    [
      {
        "frame_index": 0,
        "segment_index": 0,
        "view_id_start": "face_pz",
        "view_id_end": "diag_ppp",
        "camera_position": [...],
        "focal_position": [...],
        "view_up": [...]
      },
      ...
    ]
    """
    if len(anchor_keyframes) < 2:
        raise ValueError("Need at least 2 anchor keyframes.")

    anchors = sorted(anchor_keyframes, key=lambda k: k["keyframe_index"])

    # Extract arrays
    cams = [k["camera_position"] for k in anchors]
    focals = [k["focal_position"] for k in anchors]
    ups = [k["view_up"] for k in anchors]
    view_ids = [k["view_id"] for k in anchors]

    n = len(anchors)
    dense = []
    frame_counter = 0

    for seg in range(n - 1):
        # local control points for spline
        i0 = max(seg - 1, 0)
        i1 = seg
        i2 = seg + 1
        i3 = min(seg + 2, n - 1)

        cam0, cam1, cam2, cam3 = cams[i0], cams[i1], cams[i2], cams[i3]
        foc0, foc1, foc2, foc3 = focals[i0], focals[i1], focals[i2], focals[i3]
        up0, up1, up2, up3 = ups[i0], ups[i1], ups[i2], ups[i3]

        for f in range(frames_per_segment):
            # For all but the last segment, avoid duplicating the endpoint.
            # The next segment will start from it.
            if seg < n - 2:
                t = f / frames_per_segment
            else:
                t = f / (frames_per_segment - 1) if frames_per_segment > 1 else 1.0

            t_eased = smoothstep(t)

            if use_spline:
                cam = catmull_rom_vec3(cam0, cam1, cam2, cam3, t_eased)
                focal = catmull_rom_vec3(foc0, foc1, foc2, foc3, t_eased)
                up = catmull_rom_vec3(up0, up1, up2, up3, t_eased)
            else:
                cam = vec_lerp(cam1, cam2, t_eased)
                focal = vec_lerp(foc1, foc2, t_eased)
                up = vec_lerp(up1, up2, t_eased)

            up = orthonormalize_view_up(cam, focal, up)

            dense.append(
                {
                    "frame_index": frame_counter,
                    "segment_index": seg,
                    "view_id_start": view_ids[seg],
                    "view_id_end": view_ids[seg + 1],
                    "camera_position": [float(x) for x in cam],
                    "focal_position": [float(x) for x in focal],
                    "view_up": [float(x) for x in up],
                }
            )
            frame_counter += 1

    return dense

def load_viewpoints_from_trajectory(json_path):
    """
    Load trajectory JSON and convert to list of 9D viewpoint vectors.
    
    Output:
        [
          [cx, cy, cz, fx, fy, fz, ux, uy, uz],
          ...
        ]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    viewpoints = []

    for frame in data:
        cam = frame["camera_position"]
        focal = frame["focal_position"]
        up = frame["view_up"]

        viewpoint = cam + focal + up   # concatenate lists
        viewpoints.append(viewpoint)

    return viewpoints