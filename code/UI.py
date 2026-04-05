import dearpygui.dearpygui as dpg
import threading
from sasav import run_pasav, run_build_knowledge_base
from utilities import show_image_in_viewer
import queue

image_queue = queue.Queue()

# --- global handles for the currently shown image texture ---
_current_texture_registry_tag = "texture_registry"

def poll_image_queue():
    try:
        while True:
            img_path = image_queue.get_nowait()
            print(img_path)
            show_image_in_viewer(img_path)
    except queue.Empty:
        pass
    dpg.set_frame_callback(dpg.get_frame_count() + 1, poll_image_queue)

def log(msg: str):
    old = dpg.get_value("log_console") or ""
    dpg.set_value("log_console", old + msg + "\n")
    dpg.set_y_scroll("log_child", 1e9)

def run_agent():
    api_key = dpg.get_value("api_key_input")
    model_name = dpg.get_value("model_name_input")
    model_name = "gpt-4.1-nano"
    # model_name = "gpt-5.2"
    file_path = dpg.get_value("file_path_input")
    # file_path = "/home/user/abdomen.dat"
    knowledge_base_db_path = dpg.get_value("knowledge_base_db_path_input")
    knowledge_base_db_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/knowledge/md_folder/vector_db"

    # TODO: only for testing, need to be deleted
    # Chameleon
    # file_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/data/chameleon_256x256x256_normalized_space-1.vtk"
    # Flame
    # file_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/data/flame_256x256x256.vtk"
    # file_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/data/miranda_256x256x256.vtk"
    # file_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/data/richtmyer_256x256x240.vtk"
    # file_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/agent/abdomenatlas_label.vtk"
    file_path = "/home/js/ws/proactiveSciVisAgent/proactiveSciVisAgent/data/chameleon_256x256x270.vtk"

    iterative = dpg.get_item_user_data("iterative_toggle_btn")
    log(f"HPC: {iterative}")
    log("===== Starting SASAV Agent =====")
    log(f"API Key: {api_key[:5]}****" if api_key else "API Key: (empty)")
    log(f"Path: {file_path}" if file_path else "Path: (empty)")

    threading.Thread(
        target=run_pasav,
        args=(dpg, file_path, knowledge_base_db_path, model_name, api_key, image_queue),
        daemon=True
    ).start()
    log("Agent started")

def run_builder():
    knowledge_base_path = dpg.get_value("knowledge_base_path_input")
    api_key = dpg.get_value("api_key_input")
    pdf_knowledge_base_path = knowledge_base_path + "/pdf_folder"
    md_knowledge_base_path = knowledge_base_path + "/md_folder"
    log(f"Knowledge Base Folder: {knowledge_base_path}")

    threading.Thread(
        target=run_build_knowledge_base,
        args=(dpg, pdf_knowledge_base_path, md_knowledge_base_path, api_key),
        daemon=True
    ).start()
    log("Knowledge base builder started")

def on_pick_file(sender, app_data, user_data):
    selections = app_data.get("selections", {})
    if selections:
        picked = next(iter(selections.values()))
    else:
        picked = app_data.get("file_path_name", "")

    if picked:
        dpg.set_value("file_path_input", picked)
        log(f"Picked file: {picked}")

def on_pick_folder(sender, app_data, user_data):
    current_path = app_data.get("current_path", "")
    selections = app_data.get("selections", {})

    picked = ""

    if selections:
        # For directory_selector=True, current_path is usually the selected folder
        # and selections may duplicate the last folder name.
        picked = current_path
    else:
        picked = current_path

    if picked:
        dpg.set_value("knowledge_base_path_input", picked)
        log(f"Picked folder: {picked}")

def toggle_iterative():
    state = dpg.get_item_user_data("iterative_toggle_btn") or False
    state = not state
    dpg.set_item_user_data("iterative_toggle_btn", state)
    dpg.configure_item("iterative_toggle_btn", label="ON" if state else "OFF")
    log(f"Iterative mode: {state}")

dpg.create_context()

# A dedicated texture registry
with dpg.texture_registry(show=False, tag=_current_texture_registry_tag):
    pass

# File dialog for scientific data file
with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=on_pick_file,
    tag="file_dialog",
    width=700,
    height=400,
):
    dpg.add_file_extension(".*", color=(150, 150, 255, 255))
    dpg.add_file_extension(".vtk", color=(0, 255, 0, 255))
    dpg.add_file_extension(".vti", color=(0, 255, 0, 255))
    dpg.add_file_extension(".npy", color=(255, 255, 0, 255))

# Folder dialog for knowledge base path
with dpg.file_dialog(
    directory_selector=True,
    show=False,
    callback=on_pick_folder,
    tag="folder_dialog",
    width=700,
    height=400,
):
    pass

CONFIG_W, CONFIG_H = 520, 800
IMG_W, IMG_H = 1024, 1024
GAP = 10

with dpg.window(label="Configuration", width=CONFIG_W, height=CONFIG_H, pos=(0, 0)):
    dpg.add_text("Foundation Model Name:")
    dpg.add_input_text(tag="model_name_input", hint="e.g. gpt-4o-mini", width=480)

    dpg.add_spacer(height=8)
    dpg.add_text("Foundation Model API Key:")
    dpg.add_input_text(tag="api_key_input", hint="e.g. OPENAI_API_KEY...", width=480, password=True)

    dpg.add_spacer(height=8)
    dpg.add_text("Scientific Data Path:")
    with dpg.group(horizontal=True):
        dpg.add_input_text(tag="file_path_input", hint="e.g. /home/user/data/data.vtk", width=400)
        dpg.add_button(label="Browse", callback=lambda: dpg.show_item("file_dialog"))
    
    dpg.add_spacer(height=8)
    dpg.add_text("Knowledge Base Path:")
    with dpg.group(horizontal=True):
        dpg.add_input_text(tag="knowledge_base_db_path_input", hint="e.g. /home/user/knowledge/vector_db", width=400)
        dpg.add_button(label="Browse", callback=lambda: dpg.show_item("folder_dialog"))

    dpg.add_spacer(height=8)
    with dpg.group(horizontal=True):
        dpg.add_text("Utilize HPC: ")
        dpg.add_button(
            tag="iterative_toggle_btn",
            label="ON",
            user_data=True,
            callback=toggle_iterative
        )

    dpg.add_spacer(height=10)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Start SASAV Agent", callback=run_agent)

    dpg.add_spacer(height=8)
    dpg.add_separator()
    dpg.add_text("SASAV Working Log:")

    with dpg.child_window(tag="log_child", width=490, height=400, border=True):
        dpg.add_text("", tag="log_console", wrap=450)

with dpg.window(label="Knowledge Base", width=CONFIG_W, height=200, pos=(0, CONFIG_H + GAP)):
    dpg.add_spacer(height=8)
    dpg.add_text("Knowledge Base Folder:")

    with dpg.group(horizontal=True):
        dpg.add_input_text(
            tag="knowledge_base_path_input",
            hint="e.g. /home/user/knowledge",
            width=400
        )
        dpg.add_button(label="Browse", callback=lambda: dpg.show_item("folder_dialog"))

    dpg.add_spacer(height=10)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Start Building Knowledge Base", callback=run_builder)

with dpg.window(
    label="SASAV Workspace",
    width=IMG_W,
    height=IMG_H,
    pos=(CONFIG_W + GAP, 0),
):
    dpg.add_separator()
    with dpg.child_window(tag="image_container", width=-1, height=-1, border=False):
        dpg.add_text("SASAV will work here to interatively analyze results")

viewport_w = CONFIG_W + GAP + IMG_W + 40
viewport_h = max(CONFIG_H, IMG_H) + 60

dpg.create_viewport(title="SASAV", width=viewport_w, height=viewport_h)
dpg.setup_dearpygui()
dpg.show_viewport()

poll_image_queue()

dpg.start_dearpygui()
dpg.destroy_context()