import json
import subprocess
import os
import pymupdf4llm
import vtk
import numpy as np
from pathlib import Path
import dearpygui.dearpygui as dpg
from pathlib import Path
from utilities import log, show_image_in_viewer
from vtk.util import numpy_support

import glob
import tiktoken
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def npArray2VtkBinary(data, fileName, spacing_x, spacing_y, spacing_z):
    # Convert the numpy array to a VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

    # Create a VTK image data object
    image_data = vtk.vtkImageData()

    # Set the dimensions of the image data (same as the shape of the numpy array)
    image_data.SetDimensions(data.shape)
    image_data.SetSpacing(spacing_x, spacing_y, spacing_z)
    
    #Set origin
    x_origin = (data.shape[0] - 1)*spacing_x/2
    y_origin = (data.shape[1] - 1)*spacing_y/2
    z_origin = (data.shape[2] - 1)*spacing_z/2
    image_data.SetOrigin(-x_origin, -y_origin, -z_origin)

    # Allocate scalars for the image data
    image_data.AllocateScalars(vtk.VTK_FLOAT, 1)

    # Get the VTK array from the image data object and set its values to the converted numpy array
    vtk_array = image_data.GetPointData().GetScalars()
    vtk_array.DeepCopy(vtk_data_array)

    # Create a VTK writer to save the image data to a .vtk file
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(fileName)
    writer.SetInputData(image_data)

    # Write in binary format (instead of ASCII)
    writer.SetFileTypeToBinary()
    # alternatively: writer.SetFileType(vtk.VTK_BINARY)

    # Write the .vtk file
    writer.Write()

    print("3D volume data has been saved to", fileName)

def get_profile(dpg: dpg, file_path: str):
    folder = str(Path(file_path).parent)
    print("folder:", folder)   # ../data/raw
    dims = file_path.split('_')[1] # x_sizexy_sizexz_size
    x_size, y_size, z_size = map(int, dims.split('x'))
    log(dpg, "Input data size: " + str(x_size) + "x" + str(y_size) + "x" + str(z_size))
    log(dpg, "Loading data...")
    f = open(file_path, "r")
    data = np.fromfile(f, dtype=np.uint16)
    print(data.shape)
    data = np.reshape(data, (x_size, y_size, z_size), order='F')
    print(data.shape)
    print(data.dtype)
    # Change to float32 data type
    data = data.astype(np.float32)
    # Normalize to [0, 1]
    data_norm = (data - np.min(data))/(np.max(data) - np.min(data))
    # Down sampling
    data_norm_ds = data_norm[::2, ::2, ::2]
    # Down sampling again
    data_norm_ds_ds = data_norm_ds[::2, ::2, ::2]

    # Save to small version
    log(dpg, "Saving small version of the data in .vtk...")
    x_size_small, y_size_small, z_size_small = data_norm_ds_ds.shape # 256, 256, 270
    print("small size: ", x_size_small, y_size_small, z_size_small)
    data_small_vtk_path = folder + "/data_small.vtk"
    spacing_x = 1
    spacing_y = 1
    spacing_z = 1
    npArray2VtkBinary(data_norm_ds_ds, data_small_vtk_path, spacing_x, spacing_y, spacing_z)

    # Save to medium version
    log(dpg, "Saving medium version of the data .vtk...")
    x_size_medium, y_size_medium, z_size_medium = data_norm_ds.shape # 512, 512, 540
    print("medium size: ", x_size_medium, y_size_medium, z_size_medium)
    data_medium_vtk_path = folder + "/data_medium.vtk"
    spacing_x = (x_size_small - 1)/(x_size_medium - 1)
    spacing_y = (y_size_small - 1)/(y_size_medium - 1)
    spacing_z = (z_size_small - 1)/(z_size_medium - 1)
    npArray2VtkBinary(data_norm_ds, data_medium_vtk_path, spacing_x, spacing_y, spacing_z)

    # Save to large/original version
    log(dpg, "Saving original version of the data .vtk...")
    x_size_org, y_size_org, z_size_org = data_norm.shape # 1024, 1024, 1080
    print("org size: ", x_size_org, y_size_org, z_size_org)
    data_org_vtk_path = folder + "/data_org.vtk"
    spacing_x = (x_size_small - 1)/(x_size_org - 1)
    spacing_y = (y_size_small - 1)/(y_size_org - 1)
    spacing_z = (z_size_small - 1)/(z_size_org - 1)
    npArray2VtkBinary(data_norm, data_org_vtk_path, spacing_x, spacing_y, spacing_z)

    return data_org_vtk_path, data_medium_vtk_path, data_small_vtk_path

    # log(dpg, "-----Analyzing Input Data-----")
    # # call profile
    # executable = "../vtk/Profile/build/./Profile"
    # vtk_file = file_path
    # commend = [executable,
    #         vtk_file]
    # result = subprocess.run(commend, capture_output=True, text=True)
    # os.system("mv metadata.json " + "workspace")
    # # read the generated profile
    # p = Path("workspace/metadata.json")
    # if not p.exists():
    #     raise FileNotFoundError(f"Metadata file not found: {p}")
    # with p.open("r", encoding="utf-8") as f:
    #     meta = json.load(f)
    # # light validation
    # required = ["scalar_range", "dimensions", "extent", "spacing", "origin", "bounds"]
    # missing = [k for k in required if k not in meta]
    # if missing:
    #     raise ValueError(f"Missing keys in metadata json: {missing}")
    # log(dpg, "Input:" + meta.get("input_file", "(unknown)"))
    # log(dpg, "Scalar range:" + str(meta["scalar_range"]))
    # log(dpg, "Dimensions:" + str(meta["dimensions"]))
    # log(dpg, "Extent:" + str(meta["extent"]))
    # log(dpg, "Spacing:" + str(meta["spacing"]))
    # log(dpg, "Origin:" + str(meta["origin"]))
    # log(dpg, "Bounds:" + str(meta["bounds"]))
    # # Example: compute physical size in world units (extent-based)
    # # extent = [iMin,iMax,jMin,jMax,kMin,kMax]
    # ext = meta["extent"]
    # spacing = meta["spacing"]
    # size_x = (ext[1] - ext[0]) * spacing[0]
    # size_y = (ext[3] - ext[2]) * spacing[1]
    # size_z = (ext[5] - ext[4]) * spacing[2]
    # log(dpg, "Approx physical size (world units):" + str([size_x, size_y, size_z]))
    # v_min, v_max = meta["scalar_range"]
    # return v_min, v_max

def get_profile_vtk(dpg: dpg, file_path: str):
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

def get_dvr(file_path, view, opacity_start_value, direction):
    executable = "../vtk/SimpleRayCast/build/./SimpleRayCast"
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
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    out_dir = "workspace/dvr"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, str(opacity_start_value) + "_" + direction + ".png")
    os.replace("rendering.png", out_path)
    return out_path

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

def get_iso(file_path, view, opacity_start_value, direction):
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
        str(up_x), str(up_y), str(up_z),
        str(opacity_start_value)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    out_dir = "workspace/iso"
    os.makedirs(out_dir, exist_ok=True)

    if (direction == "front"):
        direction = "_front"
    if (direction == "left"):
        direction = "_left"
    if (direction == "top"):
        direction = "_top"
    if (direction == "diagonal"):
        direction = "_diagonal"

    out_path = os.path.join(out_dir, str(opacity_start_value) + direction + ".png")
    os.replace("rendering.png", out_path)
    return out_path

def get_iso_fine_tune(file_path, view, opacity_start_value):
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
        str(up_x), str(up_y), str(up_z),
        str(opacity_start_value)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    out_dir = "workspace/iso_ft"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, str(opacity_start_value) + ".png")
    os.replace("rendering.png", out_path)
    return out_path

def get_iso_final(file_path, view, opacity_start_value):
    executable = "../vtk/SimpleRayCast_iso_final/build/./SimpleRayCast"
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

    out_dir = "workspace/image"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "final.png")
    os.replace("rendering.png", out_path)
    return out_path

def get_iso_final_video(file_path, view, index):
    executable = "../vtk/SimpleRayCast_iso_final/build/./SimpleRayCast"
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

    out_dir = "workspace/video"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, str(index) + ".png")
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

def get_iso_view_sphere(file_path, view, index):
    executable = "../vtk/SimpleRayCast_iso_view_sphere/build/./SimpleRayCast"
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

    out_dir = "workspace/view_sphere"
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