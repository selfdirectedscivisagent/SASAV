import json
import subprocess
import os
import pymupdf4llm
import dearpygui.dearpygui as dpg
from pathlib import Path
from utilities import log, show_image_in_viewer

import glob
import tiktoken
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_profile(dpg: dpg, file_path: str):
    log(dpg, "-----Analyzing Input Data-----")
    # call profile
    executable = "../vtk/Profile/build/./Profile"
    vtk_file = file_path
    commend = [executable,
            vtk_file]
    result = subprocess.run(commend, capture_output=True, text=True)
    os.system("mv metadata.json " + "workspace")
    # read the generated profile
    p = Path("workspace/metadata.json")
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    # light validation
    required = ["scalar_range", "dimensions", "extent", "spacing", "origin", "bounds"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"Missing keys in metadata json: {missing}")
    log(dpg, "Input:" + meta.get("input_file", "(unknown)"))
    log(dpg, "Scalar range:" + str(meta["scalar_range"]))
    log(dpg, "Dimensions:" + str(meta["dimensions"]))
    log(dpg, "Extent:" + str(meta["extent"]))
    log(dpg, "Spacing:" + str(meta["spacing"]))
    log(dpg, "Origin:" + str(meta["origin"]))
    log(dpg, "Bounds:" + str(meta["bounds"]))
    # Example: compute physical size in world units (extent-based)
    # extent = [iMin,iMax,jMin,jMax,kMin,kMax]
    ext = meta["extent"]
    spacing = meta["spacing"]
    size_x = (ext[1] - ext[0]) * spacing[0]
    size_y = (ext[3] - ext[2]) * spacing[1]
    size_z = (ext[5] - ext[4]) * spacing[2]
    log(dpg, "Approx physical size (world units):" + str([size_x, size_y, size_z]))
    v_min, v_max = meta["scalar_range"]
    return v_min, v_max

def get_dvr(dpg: dpg, file_path: str, opacity_start_value):
    executable = "../vtk/SimpleRayCast/build/./SimpleRayCast"
    vtk_file = file_path
    position_x = 800
    position_y = 0
    position_z = 0
    focal_x = 0
    focal_y = 0
    focal_z = 0
    up_x = 0
    up_y = -1
    up_z = 0
    commend = [executable,
            vtk_file,
            str(position_x),
            str(position_y),
            str(position_z),
            str(focal_x),
            str(focal_y),
            str(focal_z),
            str(up_x),
            str(up_y),
            str(up_z),
            str(opacity_start_value)]
    # print(commend)
    result = subprocess.run(commend, capture_output=True, text=True)
    os.system("mv rendering.png rendering_" + str(opacity_start_value) + ".png")
    os.system("mv rendering_" + str(opacity_start_value) + ".png" + " workspace/dvr")
    show_image_in_viewer(dpg, "workspace/dvr/" + "rendering_" + str(opacity_start_value) + ".png")

# def get_iso(dpg: dpg, file_path: str, opacity_start_value):
#     executable = "../vtk/SimpleRayCast_iso/build/./SimpleRayCast"
#     vtk_file = file_path
#     position_x = 800
#     position_y = 0
#     position_z = 0
#     focal_x = 0
#     focal_y = 0
#     focal_z = 0
#     up_x = 0
#     up_y = -1
#     up_z = 0
#     commend = [executable,
#             vtk_file,
#             str(position_x),
#             str(position_y),
#             str(position_z),
#             str(focal_x),
#             str(focal_y),
#             str(focal_z),
#             str(up_x),
#             str(up_y),
#             str(up_z),
#             str(opacity_start_value)]
#     # print(commend)
#     result = subprocess.run(commend, capture_output=True, text=True)
#     os.system("mv rendering.png rendering_" + str(opacity_start_value) + ".png" )
#     os.system("mv rendering_" + str(opacity_start_value) + ".png" + " workspace/iso")
#     # show_image_in_viewer(dpg, "workspace_chameleon/iso/rendering_0.1.png")
#     # show_image_in_viewer(dpg, "workspace_chameleon/iso/" + "rendering_" + str(opacity_start_value) + ".png")
#     # show_image_in_viewer("workspace_chameleon/iso/" + "rendering_" + str(opacity_start_value) + ".png")

def get_iso(file_path, opacity_start_value, view):
    executable = "../vtk/SimpleRayCast_iso/build/./SimpleRayCast"
    vtk_file = file_path
    # Chameleon
    # position_x = 800
    # position_y = 0
    # position_z = 0
    # focal_x = 0
    # focal_y = 0
    # focal_z = 0
    # up_x = 0
    # up_y = -1
    # up_z = 0

    position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 0, 0, 500, 0, 0, 0, 0, 1, 0
    # Flame
    # if (view == "front"):
    #     position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 0, 0, 500, 0, 0, 0, 0, 1, 0
    # if (view == "side"):
    #     position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = -500, 0, 0, 0, 0, 0, 0, 1, 0
    # if (view == "top"):
    #     position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 0, 500, 0, 0, 0, 0, 0, 0, -1
    # if (view == "diagonal"):
    #     position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = -400, 400, 400, 0, 0, 0, 1, 1, -1
    
    # Miranda
    # if (view == "front"):
    #     position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 700, 0, 0, 0, 0, 0, 0, 0, 1
    # if (view == "side"):
    #     position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 0, -700, 0, 0, 0, 0, 0, 0, 1
    # if (view == "top"):
    #     position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 0, 0, 700, 0, 0, 0, -1, 0, 0
    # if (view == "diagonal"):
    #     position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 500, -500, 500, 0, 0, 0, -1, 1, 1

    # Richtmyer
    if (view == "front"):
        position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 700, 0, 0, 0, 0, 0, 0, 0, 1
    if (view == "side"):
        position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 0, -700, 0, 0, 0, 0, 0, 0, 1
    if (view == "top"):
        position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 0, 0, 700, 0, 0, 0, -1, 0, 0
    if (view == "diagonal"):
        position_x, position_y, position_z, focal_x, focal_y, focal_z, up_x, up_y, up_z = 500, -500, 500, 0, 0, 0, -1, 1, 1

    command = [
        executable, vtk_file,
        str(position_x), str(position_y), str(position_z),
        str(focal_x), str(focal_y), str(focal_z),
        str(up_x), str(up_y), str(up_z),
        str(opacity_start_value)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    out_dir = "workspace/iso"
    os.makedirs(out_dir, exist_ok=True)

    direction = ""
    if (view == "front"):
        direction = "_front"
    if (view == "side"):
        direction = "_side"
    if (view == "top"):
        direction = "_top"
    if (view == "diagonal"):
        direction = "_diagonal"

    out_path = os.path.join(out_dir, str(opacity_start_value) + direction + ".png")
    os.replace("rendering.png", out_path)
    return out_path

def get_iso_abdo(file_path, view, index):
    executable = "../vtk/SimpleRayCast_iso/build/./SimpleRayCast"
    vtk_file = file_path

    position_x = view[0]
    position_y = view[1]
    position_z = view[2]
    focal_x = view[3]
    focal_y = view[4]
    focal_z = view[5]
    up_x = view[6]
    up_y = view[7]
    up_z = view[8]

    command = [
        executable, vtk_file,
        str(position_x), str(position_y), str(position_z),
        str(focal_x), str(focal_y), str(focal_z),
        str(up_x), str(up_y), str(up_z)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    out_dir = "workspace/dvr_render"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, str(index) + ".png")
    os.replace("rendering.png", out_path)
    return out_path

def get_dvr_render(file_path, tfs_path, view, index):
    executable = "../vtk/SimpleRayCast_dvr_renderer/build/./SimpleRayCast"
    vtk_file = file_path

    position_x = view[0]
    position_y = view[1]
    position_z = view[2]
    focal_x = view[3]
    focal_y = view[4]
    focal_z = view[5]
    up_x = view[6]
    up_y = view[7]
    up_z = view[8]

    command = [
        executable, vtk_file, tfs_path,
        str(position_x), str(position_y), str(position_z),
        str(focal_x), str(focal_y), str(focal_z),
        str(up_x), str(up_y), str(up_z)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    out_dir = "workspace/dvr_render"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, str(index) + ".png")
    os.replace("rendering.png", out_path)
    return out_path

def pdf2md(dpg: dpg, pdf_folder: str, md_folder: str):
    log(dpg, "Selected PDF Folder: " + pdf_folder)
    log(dpg, "Selected MD Folder: " + md_folder)
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            # convert pdf to markdown
            md_text = pymupdf4llm.to_markdown(pdf_path)

            # save markdown
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_path = os.path.join(md_folder, md_filename)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            log(dpg, f"Converted: {filename} -> {md_filename}")

    log(dpg, "All PDFs converted to Markdown.")


def build_vector_db(dpg: dpg, md_folder: str, api_key: str):
    MODEL = "gpt-4.1-nano"
    db_name = md_folder + "/vector_db"

    files = glob.glob(md_folder + "/*.md", recursive=True)
    log(dpg, f"Found {len(files)} files in the knowledge base")
    entire_knowledge_base = ""

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            entire_knowledge_base += f.read()
            entire_knowledge_base += "\n\n"

    log(dpg, f"Total characters in knowledge base: {len(entire_knowledge_base):,}")

    encoding = tiktoken.encoding_for_model(MODEL)
    tokens = encoding.encode(entire_knowledge_base)
    token_count = len(tokens)
    log(dpg, f"Total tokens for {MODEL}: {token_count:,}")

    documents = []
    loader = DirectoryLoader(md_folder, glob="*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()
    for doc in docs:
        documents.append(doc)
    # log(dpg , str(len(documents)))

    # Divide into chunks using the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    log(dpg, f"Divided into {len(chunks)} chunks")
    # log(dpg, f"First chunk:\n\n{chunks[0]}")

    # Pick an embedding model
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key) # need google API key
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    log(dpg, f"Vectorstore created with {vectorstore._collection.count()} documents")

    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    log(dpg, f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")